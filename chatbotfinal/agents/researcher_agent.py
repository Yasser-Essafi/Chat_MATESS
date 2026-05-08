"""
STATOUR Researcher Agent — Native Tool Calling
================================================
Agentic research powered by gpt-5-mini's native function calling.
The LLM decides when and what to search; tools are executed and results
fed back until the model produces a final synthesis.

Tools available (defined in tools/search_tools.py:SEARCH_TOOLS_SCHEMA):
- web_search       → Tavily (factual / news)
- semantic_search  → Exa.ai  (causal / contextual)

RAG (ChromaDB) is pre-injected into the user message as a fast path —
the LLM gets internal MTAESS context for free without spending a tool call.

Usage:
    from agents.researcher_agent import ResearcherAgent

    agent = ResearcherAgent()
    response = agent.chat("Pourquoi les arrivées ont augmenté en juillet 2025 ?")
"""

import os
import sys
import json
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    SYSTEM_PROMPTS,
    RESEARCHER_TEMPERATURE,
    RESEARCHER_MAX_COMPLETION_TOKENS,
    RESEARCHER_REASONING_EFFORT,
    RAG_RESULTS_COUNT,
)
from utils.base_agent import (
    BaseAgent,
    sanitize_input,
    _prepare_messages,
    _should_skip_reasoning_effort,
)
from utils.logger import get_logger

logger = get_logger("statour.researcher")


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Max tool-calling rounds per question. The LLM can fan out 1-4 searches before
# being forced to synthesize. Beyond this we stop and return whatever text it
# has produced so far to avoid runaway loops.
MAX_SEARCH_TURNS = 4


# ══════════════════════════════════════════════════════════════════════════════
# Researcher Agent
# ══════════════════════════════════════════════════════════════════════════════

class ResearcherAgent(BaseAgent):
    """
    Tourism research agent with native tool calling.

    Workflow:
        1. RAG pre-injection (fast path) — internal APF/MTAESS context
        2. Agentic loop:
           a. LLM decides whether to call web_search / semantic_search
           b. Python executes the tool calls
           c. Results are fed back; LLM either calls more tools or synthesizes
        3. Up to MAX_SEARCH_TURNS rounds, then forced synthesis

    Inherits from BaseAgent: shared client, history mgmt, thread safety.
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

        # ── Initialize Web Search (Tavily/Brave) ──
        logger.info("Initializing web search...")
        try:
            from tools.search_tools import TourismSearchTool
            self.searcher = TourismSearchTool()
            self._search_available = True
        except Exception as e:
            logger.error("Web search initialization failed: %s", e)
            self.searcher = None
            self._search_available = False

        # ── Detect Exa.ai availability (semantic search) ──
        self._exa_available = (
            self._search_available
            and getattr(self.searcher, "_exa_available", False)
        )

        logger.info(
            "%s ready (RAG=%s, Web=%s, Exa=%s)",
            self.agent_name,
            "✅" if self._rag_available else "❌",
            "✅" if self._search_available else "❌",
            "✅" if self._exa_available else "❌",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Vectorstore Bootstrap
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_vectorstore(self) -> None:
        """Build vectorstore if empty or if documents have been updated since last build."""
        import os as _os
        import glob as _glob
        if not self.rag:
            return

        try:
            stats = self.rag.get_stats()
            needs_rebuild = stats["total_chunks"] == 0

            if not needs_rebuild:
                vs_db = _os.path.join(stats.get("vectorstore_path", ""), "chroma.sqlite3")
                if _os.path.exists(vs_db):
                    vs_mtime = _os.path.getmtime(vs_db)
                    doc_dir = _os.path.join(
                        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                        "knowledge_base", "documents"
                    )
                    md_files = _glob.glob(_os.path.join(doc_dir, "*.md"))
                    if any(_os.path.getmtime(f) > vs_mtime for f in md_files):
                        logger.info("Documents updated since last build — rebuilding vectorstore")
                        needs_rebuild = True

            if needs_rebuild:
                logger.info("Building vector store...")
                result = self.rag.build_vectorstore(force_rebuild=True)
                logger.info("Vectorstore: %s", result.get("message", "done"))
            else:
                logger.info("Vectorstore loaded: %d chunks", stats["total_chunks"])
        except Exception as e:
            logger.error("Vectorstore initialization failed: %s", e)

    # ──────────────────────────────────────────────────────────────────────
    # Agentic Research Loop (native tool calling)
    # ──────────────────────────────────────────────────────────────────────

    def _agentic_research(self, user_message: str, augmented_message: str) -> str:
        """
        Core agentic loop with native tool calling.
        The LLM autonomously decides when and what to search.

        Args:
            user_message: Original sanitized user message (for logging only).
            augmented_message: Message + RAG context, sent to the LLM.

        Returns:
            Final synthesized response, or fallback message if the loop fails.
        """
        from tools.search_tools import SEARCH_TOOLS_SCHEMA

        # Build initial message list: system + prior history (sans system) + new user
        base_msgs = [
            {"role": "system", "content": self.system_prompt},
            *[m for m in self.conversation_history if m["role"] != "system"],
            {"role": "user", "content": augmented_message},
        ]
        # Strip _is_summary keys + convert system→developer if needed
        messages = _prepare_messages(base_msgs)

        tools = SEARCH_TOOLS_SCHEMA if (self._search_available or self._exa_available) else []
        last_text = None

        for turn in range(MAX_SEARCH_TURNS):
            api_kwargs = {
                "model": self.deployment,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
            }
            if tools:
                api_kwargs["tools"] = tools
                api_kwargs["tool_choice"] = "auto"
            if self.reasoning_effort and not _should_skip_reasoning_effort():
                api_kwargs["reasoning_effort"] = self.reasoning_effort

            try:
                api_response = self.client.chat.completions.create(**api_kwargs)
            except Exception as e:
                logger.error("Researcher API call failed (turn %d): %s", turn, e)
                break

            if not api_response.choices:
                break

            message = api_response.choices[0].message

            if message.content:
                last_text = message.content.strip()

            # No more tool calls → synthesis is done
            if not message.tool_calls:
                break

            # Append the assistant turn (with tool_calls) to messages
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            # Execute each tool call and append its result
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except Exception:
                    fn_args = {}

                query = fn_args.get("query", "")
                max_r = fn_args.get("max_results", 3)

                if fn_name == "web_search" and self._search_available:
                    try:
                        results = self.searcher.search(query, max_results=max_r)
                    except Exception as e:
                        logger.warning("web_search('%s') failed: %s", query[:60], e)
                        results = []
                    result_text = "\n".join(
                        f"- {r.get('title','')}: {r.get('content','')[:250]} [{r.get('url','')}]"
                        for r in results
                    ) or "Aucun résultat trouvé."
                    logger.info("Tool web_search('%s'): %d results", query[:50], len(results))

                elif fn_name == "semantic_search" and self._exa_available:
                    try:
                        results = self.searcher._exa.search(query, max_results=max_r)
                    except Exception as e:
                        logger.warning("semantic_search('%s') failed: %s", query[:60], e)
                        results = []
                    result_text = "\n".join(
                        f"- {r.get('title','')}: {r.get('content','')[:300]}"
                        for r in results
                    ) or "Aucun résultat sémantique trouvé."
                    logger.info("Tool semantic_search('%s'): %d results", query[:50], len(results))

                else:
                    result_text = f"Tool '{fn_name}' not available."
                    logger.warning("Tool not available: %s", fn_name)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

        return last_text or "⚠️ Recherche terminée sans réponse."

    # ──────────────────────────────────────────────────────────────────────
    # Main Chat — Thread-safe
    # ──────────────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Process a user message with RAG pre-injection + agentic tool calling.

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

        sanitized = sanitize_input(user_message)
        if not sanitized:
            return "⚠️ Message invalide. Veuillez reformuler."

        # ── RAG pre-injection (fast path) ──
        # Internal MTAESS context goes to the LLM up-front so it doesn't have
        # to spend a tool call asking for what we already have indexed.
        rag_context = ""
        if self._rag_available and self.rag:
            try:
                rag_results = self.rag.search(sanitized, n_results=RAG_RESULTS_COUNT)
                if rag_results:
                    rag_context = "\n".join(
                        f"[Base interne MTAESS]: {r.get('content', '')[:300]}"
                        for r in rag_results
                    )
            except Exception as e:
                logger.debug("RAG search failed (non-critical): %s", e)

        augmented = sanitized
        if rag_context:
            augmented = f"{sanitized}\n\n[Contexte base interne disponible]:\n{rag_context}"

        response = self._agentic_research(sanitized, augmented)

        if response and not response.startswith("⚠️"):
            self._append_exchange(sanitized, response)
            return response

        return response or "⚠️ Je n'ai pas pu générer une réponse. Veuillez reformuler."

    # ──────────────────────────────────────────────────────────────────────
    # Direct Search Methods (CLI helpers)
    # ──────────────────────────────────────────────────────────────────────

    def search_knowledge(self, query: str) -> str:
        """Search the RAG knowledge base directly."""
        if not self._rag_available or not self.rag:
            return "❌ Base de connaissances non disponible."
        return self.rag.search_formatted(query, n_results=5)

    def search_web(self, query: str) -> str:
        """Search the web directly via Tavily/Brave."""
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
            "exa_available": self._exa_available,
            "rag_chunks": rag_stats.get("total_chunks", 0),
            "conversation_length": self.get_conversation_length(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Interactive CLI for testing the Researcher Agent."""
    from utils.cache import shared_cache

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
