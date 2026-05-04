"""
STATOUR Researcher Agent — Fixed
===================================
Web search (Tavily) + RAG (ChromaDB).
ALWAYS delivers results. Handles empty responses.
Short prompts to avoid token overflow.

Fixes from original:
- Uses BaseAgent (eliminates duplicated code)
- Thread-safe via inherited lock
- Follow-up detection uses word-boundary matching (no false positives)
- Query optimization with consistent casing
- Follow-up fallback uses assistant context when no prior long user message
- Prompt injection sanitization
- Integrated caching for search results
- 3-level fallback chain preserved and improved
- Structured logging throughout

Usage:
    from agents.researcher_agent import ResearcherAgent

    agent = ResearcherAgent()
    response = agent.chat("Combien de touristes ont visité le Maroc en 2024?")
    response = agent.chat("oui")  # Follow-up resolved automatically
"""

import os
import sys
import re
from typing import Optional, List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    SYSTEM_PROMPTS,
    RESEARCHER_TEMPERATURE,
    RESEARCHER_MAX_COMPLETION_TOKENS,
    RESEARCHER_REASONING_EFFORT,
    WEB_RESULTS_COUNT,
    RAG_RESULTS_COUNT,
    SNIPPET_LENGTH,
    MAX_CONTEXT_CHARS,
)
from utils.base_agent import BaseAgent, detect_language, sanitize_input
from utils.logger import get_logger
from utils.cache import shared_cache

logger = get_logger("statour.researcher")


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Minimal fallback system prompts (used when main prompt causes empty response)
_MINIMAL_SYSTEM = (
    "Tu es un expert en tourisme marocain. / You are a Moroccan tourism expert. "
    "Réponds directement en 5-8 lignes. Cite tes sources. "
    "Respond in the user's language. Never say 'I cannot search'."
)

_KNOWLEDGE_SYSTEM = (
    "Tu es un expert en tourisme marocain. Réponds directement. "
    "You are a Moroccan tourism expert. Respond directly."
)

# Follow-up trigger words — matched as WHOLE words only (no substring issues)
_FOLLOWUP_WORDS = frozenset({
    "oui", "yes", "ok", "okay", "d'accord", "go", "vas-y", "vasy",
    "fais-le", "do", "exactly", "exactement", "donne", "donne-moi",
    "give", "please", "svp", "stp", "ahead", "sure", "sûr",
    "bien", "absolument", "absolutely", "yep", "yup", "yeah",
    "نعم", "أجل", "طبعا", "continue", "encore", "next",
    "suivant", "suite", "more", "plus",
})

# Maximum word count for a message to be considered a follow-up
_FOLLOWUP_MAX_WORDS = 6

# Phrases to strip from search queries (conversational fluff)
_STRIP_PHRASES = [
    "can you search for", "can u search for",
    "could you search for", "could u search for",
    "search for", "look for", "find me", "find out",
    "look up", "tell me about", "tell me", "give me",
    "show me", "i want to know", "i need to know",
    "can you find", "can u find",
    "please", "svp", "stp",
    "use tavily", "use google",
    "on the internet", "on internet", "on the web",
    "sur internet", "sur le web",
    "cherche", "recherche", "cherche-moi", "trouve-moi",
    "fais une recherche", "lance une recherche",
    "peux tu chercher", "peux-tu chercher",
]

# Cache source identifiers
_CACHE_SOURCE_CONTEXT = "researcher_context"


# ══════════════════════════════════════════════════════════════════════════════
# Language-specific instruction helpers
# ══════════════════════════════════════════════════════════════════════════════

def _synth_instruction(lang: str) -> str:
    """Return a synthesis instruction in the user's language."""
    if lang == "fr":
        return (
            "Synthétise en 5-8 lignes. Cite tes sources. "
            "Si la réponse exacte manque, complète avec tes connaissances. RÉPONDS."
        )
    if lang == "ar":
        return (
            "لخّص في 5-8 أسطر. اذكر المصادر. "
            "إذا كانت الإجابة الدقيقة ناقصة، أكمل من معرفتك. أجب."
        )
    return (
        "Synthesize in 5-8 lines. Cite sources. "
        "If the exact answer is missing, fill in with your expertise. RESPOND."
    )


def _no_results_instruction(lang: str) -> str:
    """Return an instruction for when no search results are found."""
    if lang == "fr":
        return (
            "Aucun résultat web. Utilise tes connaissances d'expert tourisme marocain. "
            "Donne ta meilleure réponse en 5-8 lignes. Cite des sources de référence."
        )
    if lang == "ar":
        return (
            "لا توجد نتائج ويب. استخدم معرفتك كخبير سياحة مغربي. "
            "قدّم أفضل إجابة في 5-8 أسطر مع ذكر المصادر."
        )
    return (
        "No web results. Use your Moroccan tourism expertise. "
        "Give your best answer in 5-8 lines. Cite reference sources."
    )


def _error_message(lang: str) -> str:
    """Return an error message in the user's language."""
    if lang == "fr":
        return "⚠️ Impossible d'obtenir une réponse. Reformulez votre question."
    if lang == "ar":
        return "⚠️ تعذر الحصول على إجابة. يرجى إعادة صياغة سؤالك."
    return "⚠️ Could not generate a response. Please rephrase your question."


# ══════════════════════════════════════════════════════════════════════════════
# Researcher Agent
# ══════════════════════════════════════════════════════════════════════════════

class ResearcherAgent(BaseAgent):
    """
    Tourism research agent with web search (Tavily) and RAG (ChromaDB).

    Features:
        - Automatic web + knowledge base search before every response
        - 3-level fallback chain (full history → minimal prompt → knowledge only)
        - Follow-up detection with word-boundary matching
        - Query optimization (strip conversational fluff, enrich tourism queries)
        - Result caching
        - Prompt injection sanitization
        - Multilingual support (French, English, Arabic)

    Inherits from BaseAgent:
        - Shared AzureOpenAI client
        - History trimming
        - Thread-safe LLM calls
        - Reset/length methods
    """

    def __init__(self):
        super().__init__(
            system_prompt=SYSTEM_PROMPTS["researcher"],
            agent_name="Chercheur Tourisme STATOUR",
            temperature=RESEARCHER_TEMPERATURE,
            max_tokens=RESEARCHER_MAX_COMPLETION_TOKENS,
            reasoning_effort=RESEARCHER_REASONING_EFFORT,
        )

        # ── Initialize RAG ──
        logger.info("Initializing RAG knowledge base...")
        try:
            from tools.rag_tools import RAGManager
            self.rag = RAGManager()
            self._ensure_vectorstore()
            self._rag_available = True
        except Exception as e:
            logger.error("RAG initialization failed: %s", e)
            self.rag = None
            self._rag_available = False

        # ── Initialize Web Search ──
        logger.info("Initializing web search...")
        try:
            from tools.search_tools import TourismSearchTool
            self.searcher = TourismSearchTool()
            self._search_available = True
        except Exception as e:
            logger.error("Web search initialization failed: %s", e)
            self.searcher = None
            self._search_available = False

        logger.info(
            "%s ready (RAG=%s, Web=%s)",
            self.agent_name,
            "✅" if self._rag_available else "❌",
            "✅" if self._search_available else "❌",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Vectorstore Bootstrap
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_vectorstore(self) -> None:
        """Build vectorstore if empty or if documents have been updated since last build."""
        import os, glob as _glob
        if not self.rag:
            return

        try:
            stats = self.rag.get_stats()
            needs_rebuild = stats["total_chunks"] == 0

            # Also rebuild if any .md document is newer than the vectorstore sqlite file
            if not needs_rebuild:
                vs_db = os.path.join(stats.get("vectorstore_path", ""), "chroma.sqlite3")
                if os.path.exists(vs_db):
                    vs_mtime = os.path.getmtime(vs_db)
                    doc_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "knowledge_base", "documents"
                    )
                    md_files = _glob.glob(os.path.join(doc_dir, "*.md"))
                    if any(os.path.getmtime(f) > vs_mtime for f in md_files):
                        logger.info("Documents updated since last build — rebuilding vectorstore")
                        needs_rebuild = True

            if needs_rebuild:
                logger.info("Building vector store...")
                result = self.rag.build_vectorstore(force_rebuild=True)
                logger.info("Vectorstore: %s", result.get("message", "done"))
            else:
                logger.info(
                    "Vectorstore loaded: %d chunks", stats["total_chunks"]
                )
        except Exception as e:
            logger.error("Vectorstore initialization failed: %s", e)

    # ──────────────────────────────────────────────────────────────────────
    # Follow-up Detection (word-boundary safe)
    # ──────────────────────────────────────────────────────────────────────

    def _is_followup(self, msg: str) -> bool:
        """
        Detect conversational follow-ups using whole-word matching.
        Prevents false positives like 'booking' matching 'ok'.

        Args:
            msg: Lowercase stripped message.

        Returns:
            True if the message looks like a follow-up.
        """
        words = set(re.findall(r"\b[\w']+\b", msg.lower()))
        return len(words) <= _FOLLOWUP_MAX_WORDS and bool(words & _FOLLOWUP_WORDS)

    def _resolve_followup(self, user_message: str) -> str:
        """
        Resolve follow-up messages into a meaningful search query.

        Strategy:
            1. If not a follow-up → return as-is
            2. Look for a prior substantive user message
            3. If none, extract topic from last assistant message
            4. If all fails, return original message

        Args:
            user_message: The user's input.

        Returns:
            Resolved search query.
        """
        msg = user_message.strip()

        if not self._is_followup(msg):
            return user_message

        # Need conversation history to resolve
        if len(self.conversation_history) < 3:
            return user_message

        # Strategy 1: Find a prior substantive user message (>5 words)
        for entry in reversed(self.conversation_history):
            if entry["role"] == "user" and len(entry["content"].split()) > 5:
                logger.debug(
                    "Follow-up '%s' resolved to prior user message: '%s'",
                    msg[:30], entry["content"][:60],
                )
                return entry["content"]

        # Strategy 2: Extract context from last assistant message
        for entry in reversed(self.conversation_history):
            if entry["role"] == "assistant" and len(entry["content"].split()) > 5:
                first_line = entry["content"].split("\n")[0][:200]
                resolved = f"{user_message} — context: {first_line}"
                logger.debug(
                    "Follow-up '%s' resolved with assistant context",
                    msg[:30],
                )
                return resolved

        logger.debug(
            "Follow-up '%s' could not be resolved — using as-is",
            msg[:30],
        )
        return user_message

    # ──────────────────────────────────────────────────────────────────────
    # Search Query Optimization
    # ──────────────────────────────────────────────────────────────────────

    def _optimize_query(self, query: str) -> str:
        """
        Strip conversational fluff and enrich tourism queries.

        Examples:
            'can u search for how many people visited morocco in 2023'
            → 'how many people visited morocco in 2023 statistics official data'

        Args:
            query: Raw or resolved search query.

        Returns:
            Optimized query string.
        """
        cleaned = query.lower().strip()

        # Remove conversational phrases
        for phrase in _STRIP_PHRASES:
            cleaned = cleaned.replace(phrase, "")

        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Enrich tourism + Morocco queries with stat keywords
        tourism_words = {
            "visit", "tourist", "tourism", "arrival", "travel",
            "touriste", "visiteur", "arrivée", "voyage", "séjour",
            "nuitée", "hébergement",
        }
        stat_words = {
            "statistics", "statistiques", "data", "number",
            "nombre", "chiffre", "figure", "stats", "données",
        }

        has_tourism = any(w in cleaned for w in tourism_words)
        has_morocco = "morocco" in cleaned or "maroc" in cleaned

        if has_tourism and has_morocco and not any(w in cleaned for w in stat_words):
            cleaned += " statistics official data"

        # If cleaning destroyed the query, fall back to lowered original
        if len(cleaned.split()) < 2:
            cleaned = query.lower().strip()

        return cleaned

    # ──────────────────────────────────────────────────────────────────────
    # Specialized multi-query search (tourism factor analysis)
    # ──────────────────────────────────────────────────────────────────────

    # Trigger words that indicate the user wants a factor/context analysis
    # rather than a direct fact lookup.
    _FACTOR_TRIGGERS = (
        "pourquoi", "why", "cause", "raison", "expliqu", "facteur",
        "contexte", "impact", "influence", "baisse", "hausse", "évolution",
        "tendance", "flux", "conjoncture", "géopolitique", "geopolitique",
    )

    _FACTOR_MONTHS = (
        "janvier", "février", "fevrier", "mars", "avril", "mai", "juin",
        "juillet", "août", "aout", "septembre", "octobre", "novembre",
        "décembre", "decembre",
    )

    def _is_factor_question(self, message: str) -> bool:
        """True when the user asks about tourism factors / context / causes."""
        msg = message.lower()
        return any(t in msg for t in self._FACTOR_TRIGGERS)

    def _build_specialized_queries(self, message: str) -> List[tuple]:
        """Build multiple targeted Tavily queries for tourism factor analysis.

        Returns a list of (query_string, category_label) tuples capped at 4
        so we never explode the Tavily quota for a single user turn.
        """
        msg_lower = message.lower()

        year_match = re.search(r'\b(20[12]\d)\b', message)
        year = year_match.group(1) if year_match else ""

        month = next((m for m in self._FACTOR_MONTHS if m in msg_lower), None)
        period = f"{month} {year}".strip() if month else year
        period = period or "récent"

        queries: List[tuple] = [
            (
                f"tourisme maroc arrivées {period} statistiques bilan",
                "Statistiques tourisme Maroc",
            ),
            (
                f"trafic aérien maroc {period} compagnies vols routes",
                "Connectivité aérienne",
            ),
            (
                f"france espagne tourisme international {period} voyages",
                "Marchés émetteurs (France/Espagne)",
            ),
        ]

        if any(w in msg_lower for w in ("mre", "marocains résidant", "diaspora")):
            queries.append((
                f"mre maroc {period} séjours transferts",
                "Marocains Résidant à l'Étranger",
            ))
        else:
            queries.append((
                f"événements maroc {period} conférences salons tourisme",
                "Événements et contexte",
            ))

        return queries[:4]

    def _gather_specialized_context(self, message: str) -> str:
        """Run the specialized queries and format a grouped context block."""
        if not (self._search_available and self.searcher):
            return ""

        blocks: List[str] = []
        for query, label in self._build_specialized_queries(message):
            try:
                results = self.searcher.search(query, max_results=2)
            except Exception as e:
                logger.debug("Specialized search failed for '%s': %s", label, e)
                continue
            if not results:
                continue
            lines = [
                f"- [{r.get('title','')}]({r.get('url','')}): "
                f"{(r.get('content') or '')[:SNIPPET_LENGTH]}"
                for r in results
            ]
            blocks.append(f"{label}:\n" + "\n".join(lines))

        if not blocks:
            return ""
        return ("WEB (analyse multi-facteurs):\n" + "\n\n".join(blocks))[:MAX_CONTEXT_CHARS]

    # ──────────────────────────────────────────────────────────────────────
    # Context Gathering (Web + RAG)
    # ──────────────────────────────────────────────────────────────────────

    def _gather_context(self, search_query: str) -> str:
        """
        Gather context from web search and RAG knowledge base.
        Uses caching to avoid redundant API calls.

        Args:
            search_query: Optimized search query.

        Returns:
            Combined context string, or empty string if nothing found.
        """
        # ── Check cache ──
        cached = shared_cache.get(search_query, source=_CACHE_SOURCE_CONTEXT)
        if cached is not None:
            logger.debug("Context cache hit for: %s", search_query[:50])
            return cached

        context_parts: List[str] = []

        # ── Web search ──
        if self._search_available and self.searcher:
            try:
                web_results = self.searcher.search(
                    search_query, max_results=WEB_RESULTS_COUNT
                )
                if web_results:
                    web_text = "\n".join(
                        f"- [{r['title']}]({r['url']}): "
                        f"{r['content'][:SNIPPET_LENGTH]}"
                        for r in web_results
                    )
                    context_parts.append(f"WEB:\n{web_text}")
                    logger.debug(
                        "Web search: %d results for '%s'",
                        len(web_results), search_query[:50],
                    )
            except Exception as e:
                logger.warning("Web search error: %s", e)

        # ── RAG search ──
        if self._rag_available and self.rag:
            try:
                rag_results = self.rag.search(
                    search_query, n_results=RAG_RESULTS_COUNT
                )
                if rag_results:
                    rag_text = "\n".join(
                        f"- [APF — base interne STATOUR]: "
                        f"{r.get('content', '')[:SNIPPET_LENGTH]}"
                        for r in rag_results
                    )
                    context_parts.append(
                        f"DOCS (base interne — table APF, statistiques d'arrivées Maroc):\n{rag_text}"
                    )
                    logger.debug(
                        "RAG search: %d results for '%s'",
                        len(rag_results), search_query[:50],
                    )
            except Exception as e:
                logger.warning("RAG search error: %s", e)

        # ── Combine and cache ──
        if context_parts:
            combined = "\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]
            shared_cache.set(search_query, combined, source=_CACHE_SOURCE_CONTEXT)
            return combined

        logger.info("No search results found for: %s", search_query[:50])
        return ""

    # ──────────────────────────────────────────────────────────────────────
    # User Prompt Builder
    # ──────────────────────────────────────────────────────────────────────

    def _build_user_prompt(
        self,
        user_message: str,
        context: str,
        lang: str,
    ) -> str:
        """
        Build the augmented user prompt with search context.

        Args:
            user_message: Original (sanitized) user message.
            context: Combined search results.
            lang: Detected language.

        Returns:
            Augmented prompt string.
        """
        if context:
            return (
                f"QUESTION: {user_message}\n\n"
                f"RESULTATS:\n{context}\n\n"
                f"{_synth_instruction(lang)}"
            )
        else:
            return (
                f"QUESTION: {user_message}\n\n"
                f"{_no_results_instruction(lang)}"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Main Chat — Thread-safe, 3-level fallback
    # ──────────────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Process a user message with automatic web + RAG search.

        Workflow:
            1. Sanitize input
            2. Detect language
            3. Resolve follow-ups → real search query
            4. Optimize query for search
            5. Gather context (web + RAG)
            6. 3-level fallback chain:
               a. Full conversation history
               b. Minimal system prompt (bypass history bloat)
               c. Pure knowledge (no search context)
            7. Last resort: return raw search results

        Args:
            user_message: The user's input text.

        Returns:
            Agent response. Never raises — always returns something.
        """
        with self._lock:
            return self._chat_internal(user_message)

    def _chat_internal(self, user_message: str) -> str:
        """Internal chat logic (must be called with self._lock held)."""

        self._trim_history()

        # ── Sanitize ──
        sanitized = sanitize_input(user_message)
        if not sanitized:
            return "⚠️ Message invalide. Veuillez reformuler."

        lang = detect_language(sanitized)

        # ── Resolve follow-ups ──
        search_query = self._resolve_followup(sanitized)

        # ── Optimize for search ──
        optimized_query = self._optimize_query(search_query)
        logger.debug(
            "Query pipeline: '%s' → '%s' → '%s'",
            user_message[:40], search_query[:40], optimized_query[:40],
        )

        # ── Gather context ──
        # Factor/context questions (pourquoi, baisse, tendance...) get the
        # multi-query pipeline (flights, markets, events). Otherwise the
        # default single-query web+RAG path.
        if self._is_factor_question(sanitized):
            context = self._gather_specialized_context(sanitized)
            if not context:
                context = self._gather_context(optimized_query)
        else:
            context = self._gather_context(optimized_query)

        # ── Build augmented prompt ──
        user_prompt = self._build_user_prompt(sanitized, context, lang)

        # ═══ ATTEMPT 1: Full conversation history ═══
        attempt1_messages = self.conversation_history + [
            {"role": "user", "content": user_prompt}
        ]
        result = self._call_llm(messages=attempt1_messages)

        if result:
            self._append_exchange(sanitized, result)
            logger.debug("Attempt 1 succeeded (%d chars)", len(result))
            return result

        logger.warning("Attempt 1 empty → trying minimal prompt...")

        # ═══ ATTEMPT 2: Minimal system prompt (bypass history bloat) ═══
        result = self._call_llm(
            messages=[
                {"role": "system", "content": _MINIMAL_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        )

        if result:
            self._append_exchange(sanitized, result)
            logger.debug("Attempt 2 succeeded (%d chars)", len(result))
            return result

        logger.warning("Attempt 2 empty → trying knowledge-only...")

        # ═══ ATTEMPT 3: Pure knowledge, no search context ═══
        knowledge_prompt = f"{sanitized}\n{_synth_instruction(lang)}"
        result = self._call_llm(
            messages=[
                {"role": "system", "content": _KNOWLEDGE_SYSTEM},
                {"role": "user", "content": knowledge_prompt},
            ]
        )

        if result:
            self._append_exchange(sanitized, result)
            logger.debug("Attempt 3 succeeded (%d chars)", len(result))
            return result

        logger.warning("Attempt 3 empty → returning raw results...")

        # ═══ LAST RESORT: Raw search results ═══
        if context:
            fallback = (
                f"📋 Résultats de recherche pour: {sanitized}\n\n"
                f"{context[:1500]}"
            )
            self._append_exchange(sanitized, fallback)
            logger.info("Returning raw search results as fallback")
            return fallback

        # ═══ COMPLETE FAILURE ═══
        logger.error("All attempts failed for: %s", sanitized[:80])
        return _error_message(lang)

    # ──────────────────────────────────────────────────────────────────────
    # Direct Search Methods (for CLI and orchestrator)
    # ──────────────────────────────────────────────────────────────────────

    def search_knowledge(self, query: str) -> str:
        """
        Search the RAG knowledge base directly.

        Args:
            query: Search query.

        Returns:
            Formatted search results string.
        """
        if not self._rag_available or not self.rag:
            return "❌ Base de connaissances non disponible."
        return self.rag.search_formatted(query, n_results=5)

    def search_web(self, query: str) -> str:
        """
        Search the web directly.

        Args:
            query: Search query.

        Returns:
            Formatted search results string.
        """
        if not self._search_available or not self.searcher:
            return "❌ Recherche web non disponible."
        return self.searcher.search_formatted(query, max_results=5)

    # ──────────────────────────────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return agent status including tool availability."""
        rag_stats = {}
        if self._rag_available and self.rag:
            try:
                rag_stats = self.rag.get_stats()
            except Exception:
                pass

        return {
            "agent": self.agent_name,
            "rag_available": self._rag_available,
            "search_available": self._search_available,
            "rag_chunks": rag_stats.get("total_chunks", 0),
            "conversation_length": self.get_conversation_length(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Interactive CLI for testing the Researcher Agent."""
    try:
        from config.settings import validate_config
        validate_config(require_tavily=True)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return

    print("=" * 60)
    print("🔍 STATOUR — Chercheur Tourisme")
    print("   Ministère du Tourisme du Maroc")
    print("=" * 60)
    print()
    print("Commands: /reset  /rag <q>  /web <q>  /stats  /status  /quit")
    print("=" * 60)
    print()

    agent = ResearcherAgent()

    while True:
        try:
            user_input = input("👤 Vous: ").strip()
            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd == "/quit":
                print("\n👋 Au revoir!")
                break

            if cmd == "/reset":
                print(f"\n{agent.reset_conversation()}\n")
                continue

            if cmd == "/stats":
                if agent._rag_available and agent.rag:
                    print(f"\n📚 RAG: {agent.rag.get_stats()}")
                print(f"📊 History: {agent.get_history_stats()}")
                cache_stats = shared_cache.stats()
                print(f"🗄️  Cache: {cache_stats}\n")
                continue

            if cmd == "/status":
                status = agent.get_status()
                print(f"\n📋 Status:")
                for key, value in status.items():
                    print(f"   {key}: {value}")
                print()
                continue

            if cmd.startswith("/rag "):
                query = user_input[5:].strip()
                if query:
                    print(f"\n{agent.search_knowledge(query)}\n")
                else:
                    print("\n⚠️ Usage: /rag <query>\n")
                continue

            if cmd.startswith("/web "):
                query = user_input[5:].strip()
                if query:
                    print(f"\n{agent.search_web(query)}\n")
                else:
                    print("\n⚠️ Usage: /web <query>\n")
                continue

            print("\n⏳ Recherche...")
            response = agent.chat(user_input)
            print(f"\n🔍 {agent.agent_name}:\n{response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur inattendue: {e}\n")
            logger.error("CLI error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()