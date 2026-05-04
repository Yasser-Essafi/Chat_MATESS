"""
STATOUR Multi-Agent Orchestrator — Fixed
==========================================
3-layer classification: Instant Rules → LLM Understanding → Fallback
Maintains full conversation context for follow-ups.
Auto-reroutes Analytics → Researcher when data unavailable.
Handles typos, mixed languages, frustration, short follow-ups.

Fixes from original:
- Uses shared AzureOpenAI client (singleton)
- Classifier temperature = 0.0 (was 1.0 — caused random routing)
- Thread-safe via lock on all mutable state
- Classification caching (identical messages don't re-call LLM)
- Better conversation context (300 chars, was 120)
- Configuration validation at startup
- Structured logging throughout
- General exception handler in CLI
- Python 3.8+ compatible type hints (Optional instead of str | None)

Usage:
    from agents.orchestrator_agent import Orchestrator

    orch = Orchestrator()
    result = orch.route("Combien de touristes en 2024?")
    print(result["response"])
"""

import os
import sys
import time
import re
import threading
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    validate_config,
    AZURE_OPENAI_DEPLOYMENT,
    CLASSIFIER_TEMPERATURE,
    CLASSIFIER_MAX_TOKENS,
    CLASSIFIER_REASONING_EFFORT,
    MODEL_IS_REASONING,
    AGENT_NAMES,
    ACTIVE_AGENTS,
)
from utils.base_agent import (
    get_shared_client,
    _should_skip_reasoning_effort,
    _mark_reasoning_effort_rejected,
)
from utils.logger import get_logger
from utils.cache import SearchCache

logger = get_logger("statour.orchestrator")


# ══════════════════════════════════════════════════════════════════════════════
# LLM Classifier Prompt Template
# ══════════════════════════════════════════════════════════════════════════════
# {min_year}, {max_year}, {last_agent}, {conversation_context} are replaced
# at runtime with actual values.

CLASSIFIER_SYSTEM_PROMPT = """Tu es le cerveau de routage de STATOUR, la plateforme du Ministère du Tourisme du Maroc. Tu dois comprendre L'INTENTION de l'utilisateur — même si le message contient des fautes, de l'argot, du mélange de langues (français/anglais/arabe/darija), ou est très court.

Tu as 4 agents:

ANALYTICS — Analyse les données du dataset interne. Deux domaines de données :
(1) APF (arrivées aux postes frontières) : arrivées touristes, MRE, TES, postes frontières, voies d'entrée, pays de résidence, continents.
(2) Hébergement STDN/EHTC : nuitées, arrivées hôtelières, hébergement, hôtels, établissements, délégations, régions hôtelières, types d'hébergement, capacité.
Le dataset couvre {min_year}–{max_year} UNIQUEMENT. Questions sur des chiffres, statistiques, graphiques, tendances, classements, top N, comparaisons dans cette période → ANALYTICS. Commandes / → ANALYTICS.

RESEARCHER — Recherche sur internet (Tavily) + base documentaire (RAG). Actualités, stratégie, Vision 2030, contexte externe, données historiques HORS {min_year}–{max_year}, explications politiques/économiques, benchmarks internationaux, news, recommandations → RESEARCHER. Aussi quand l'utilisateur demande EXPLICITEMENT de chercher/search/googler → RESEARCHER.

PREDICTION — Prévisions et estimations de flux touristiques FUTURS. Questions sur des années FUTURES (2027, 2028…), estimations, projections, scénarios (optimiste/pessimiste/de base), forecasts → PREDICTION.

NORMAL — Salutations, aide, questions sur la plateforme STATOUR elle-même, conversation générale, météo, heure, qui es-tu → NORMAL.

CONTEXTE DE CONVERSATION:
{conversation_context}

REGLES STRICTES:
1. Commandes / → TOUJOURS ANALYTICS
2. Demande EXPLICITE de chercher/search/google/internet/tavily → TOUJOURS RESEARCHER
3. Prévision/estimation/projection/forecast pour années FUTURES ou mots "estimer", "prévoir" → PREDICTION
4. Années HORS {min_year}–{max_year} (sauf futur) → RESEARCHER
5. Données/chiffres/stats pour {min_year}–{max_year} (arrivées, MRE, TES, pays de résidence, régions, frontières, voies) → ANALYTICS
6. Actualités/stratégie/contexte/pourquoi/news/benchmarks → RESEARCHER
7. Salutations/aide/conversation → NORMAL
8. GRAPHIQUE/CHART/VISUALISATION:
   - Données INTERNES (arrivées Maroc, pays de résidence, régions, voies, {min_year}–{max_year}) → ANALYTICS
   - Données EXTERNES (autres pays, monde, tendances globales, hors dataset) → RESEARCHER (pipeline automatique)
   - FOLLOW-UP chart sur réponse ANALYTICS précédente → ANALYTICS
   - FOLLOW-UP chart sur réponse RESEARCHER précédente → RESEARCHER (pipeline automatique)
9. FOLLOW-UP court (continuation du même sujet) → MÊME agent: {last_agent}
10. Doute → {last_agent} si existe, sinon ANALYTICS

UN SEUL MOT: ANALYTICS ou RESEARCHER ou PREDICTION ou NORMAL"""


# ══════════════════════════════════════════════════════════════════════════════
# No-Data Indicators (triggers auto-reroute Analytics → Researcher)
# ══════════════════════════════════════════════════════════════════════════════

NO_DATA_INDICATORS = [
    # French — data genuinely absent from the dataset.
    # Kept narrow + verb-anchored to avoid matching ordinary explanatory text
    # like "ces données ne sont pas non disponibles à l'export" (false positive
    # observed in production tests).
    "aucun enregistrement pour", "aucune ligne pour", "aucune donnée pour",
    "aucun résultat pour", "aucun résultat trouvé",
    "pas de données pour", "pas de données disponibles pour",
    "n'apparaît pas dans", "n'apparait pas dans", "n'existe pas dans",
    "données non disponibles pour", "données indisponibles pour",
    "ne contient pas de données", "pas dans le dataset",
    "hors de la plage", "en dehors de la plage",
    "0 ligne", "0 lignes", "empty dataframe",
    # English — data genuinely absent
    "no data for", "no records for", "no rows for", "no results for",
    "not available for", "not in the dataset", "does not contain",
    "0 rows", "outside the range",
]

# Code execution failures — do NOT reroute (data exists, the generated code had a bug)
_CODE_FAILURE_INDICATORS = [
    "failed after", "code execution failed",
    "keyerror", "key error", "column not found", "colonne introuvable",
    "nameerror", "name error", "syntaxerror",
]


# ══════════════════════════════════════════════════════════════════════════════
# Instant Classification Constants
# ══════════════════════════════════════════════════════════════════════════════

# Pure greetings (exact match only)
_PURE_GREETINGS = frozenset({
    "bonjour", "bonsoir", "salut", "salam", "hello", "hi", "hey",
    "coucou", "bjr", "bsr", "merci", "thanks", "thank you",
    "shukran", "au revoir", "bye", "bslama", "a bientot",
    "à bientôt", "good morning", "good evening",
    "مرحبا", "السلام عليكم", "سلام", "شكرا",
})

# Explicit search request phrases
_SEARCH_PHRASES = [
    "search", "cherche", "recherche", "tavily", "google",
    "googler", "internet", "look up", "look for", "find online",
    "web search", "en ligne", "sur le web", "sur internet",
    "go search", "can you search", "can u search",
    "could you search", "could u search",
    "peux tu chercher", "peux-tu chercher",
    "fais une recherche", "va chercher", "lance une recherche",
    "cherche sur", "recherche sur", "look it up",
    "search for", "find me", "trouve moi", "trouve-moi",
    "find out", "look online", "check online",
    "check the internet", "go online", "use the internet",
    "use tavily", "use google", "utilise tavily",
    "utilise google", "cherche moi", "cherche-moi",
    "search about", "search on", "look into",
]

# Exact-match follow-up words
_FOLLOWUP_EXACT = frozenset({
    "oui", "non", "yes", "no", "ok", "okay", "d'accord",
    "go", "go ahead", "vas-y", "vas y", "allez",
    "exactement", "exactly", "bien sur", "bien sûr",
    "of course", "sure", "please", "svp", "stp",
    "do it", "just do it", "fais-le", "fais le",
    "continue", "encore", "next", "suivant", "suite",
    "نعم", "لا", "أجل", "طبعا",
})

# Follow-up phrases (substring match)
_FOLLOWUP_PHRASES = [
    "donne moi", "donne-moi", "give me", "show me",
    "montre moi", "montre-moi", "just give", "juste donne",
    "le nombre", "le total", "la liste", "les chiffres",
    "plus de details", "plus de détails", "more details",
    "et pour", "et si", "what about", "how about",
    "même chose", "meme chose", "same for", "pareil pour",
    "aussi pour", "aussi", "also", "and for",
    "i don't care", "i dont care", "je m'en fiche",
    "just tell me", "dis moi juste", "dis-moi juste",
    "arrête de demander", "arrete de demander",
    "stop asking", "just answer", "réponds juste",
    "reponds juste", "what i want", "ce que je veux",
    # Confirmation + do-all follow-ups
    "fait moi tout", "fais moi tout", "fais tout", "fait tout",
    "fait le tout", "fais le tout", "fait moi les", "fais moi les",
    "fait moi ça", "fais moi ça", "fait ça", "fais ça",
    "ok fait", "ok fais", "okay fait", "okay fais",
    "oui fait", "oui fais", "vas-y fait", "allez fait",
]

# Analytics keywords (fallback layer)
_ANALYTICS_KEYWORDS = [
    "donnée", "données", "data", "dataset", "statistique",
    "chiffre", "chiffres", "nombre", "combien", "total", "somme",
    "graphe", "graphique", "chart", "courbe", "diagramme", "plot",
    "tableau", "table", "excel", "csv", "colonne",
    "arrivée", "arrivées", "touriste", "touristes", "nuitée",
    "mre", "tes", "frontière", "aéroport", "pays de résidence",
    "région", "top ", "classement", "ranking",
    "tendance", "évolution", "croissance", "baisse", "hausse",
    "compare", "comparaison", "versus", "vs",
    "moyenne", "médiane", "pourcentage",
    "mensuel", "annuel", "trimestriel", "par mois", "par an",
    "analyse", "analyser", "calculer",
    "aérien", "maritime", "terrestre", "voie",
    "continent", "visited", "visitors", "how many",
    "tourists", "arrivals",
]

# Researcher keywords (fallback layer)
_RESEARCHER_KEYWORDS = [
    "actualité", "actualités", "news", "nouvelle",
    "stratégie", "vision 2030", "plan", "politique",
    "pourquoi", "expliquer", "explication", "cause",
    "contexte", "recommandation", "conseil",
    "onmt", "onda", "benchmark", "mondial",
    "événement", "cop", "salon", "forum",
    "marché", "concurrence", "article", "investissement", "impact",
]

# Prediction keywords (instant Layer-1 and fallback Layer-3)
# These capture forecasting intent clearly distinct from analytics / researcher.
# Bare years (2027-2031) used to be here but mis-classified conceptual queries
# like "Vision 2030" or "Coupe du Monde 2030" as prediction. Replaced with
# combined patterns that require both a future-year and a forecasting verb.
_PREDICTION_KEYWORDS = [
    "prévision", "prévisio", "prévois", "prévoir",
    "estimation", "estime", "estimer", "estimé",
    "projection", "projeter", "projette",
    "forecast", "predict", "prediction",
    "flux futur", "flux 202", "touristes 202",
    "combien en 2027", "combien en 2028",
    "combien en 2029", "combien en 2030", "combien en 2031",
    "arrivées en 2027", "arrivées en 2028",
    "arrivées en 2029", "arrivées en 2030", "arrivées en 2031",
    "tendance future", "tendance prévue",
    "scénario", "scenario", "optimiste", "pessimiste",
]


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    """
    Multi-agent orchestrator for STATOUR.

    Features:
        - 3-layer classification: Instant Rules → LLM → Keyword Fallback
        - Auto-reroute Analytics → Researcher when data not found
        - Full conversation context for follow-up resolution
        - Classification caching
        - Thread-safe
        - Structured logging

    Agents:
        - NormalAgent: Greetings, help, general conversation
        - ResearcherAgent: Web search, RAG, external context
        - DataAnalyticsAgent: Data analysis, charts, statistics
    """

    def __init__(self):
        """Initialize the orchestrator and all sub-agents."""

        # ── Validate configuration ──
        try:
            validate_config(require_tavily=True)
        except Exception as e:
            logger.error("Configuration validation failed: %s", e)
            raise

        print("=" * 60)
        print(f"  {AGENT_NAMES['orchestrator']}")
        print("  Ministère du Tourisme du Maroc")
        print("=" * 60)
        print()

        # ── Shared client ──
        self.client = get_shared_client()
        self.deployment = AZURE_OPENAI_DEPLOYMENT

        # ── Thread safety ──
        self._lock = threading.Lock()

        # ── Classification cache ──
        self._classify_cache = SearchCache(max_size=100, ttl_seconds=300)

        # ── Initialize agents ──
        print("🔧 Initialisation des agents...\n")

        print("  [1/3] 🏛️  Assistant Général...")
        from agents.normal_agent import NormalAgent
        self.normal_agent = NormalAgent()
        print("  ✅ Assistant Général prêt\n")

        print("  [2/3] 🔍 Chercheur Tourisme...")
        from agents.researcher_agent import ResearcherAgent
        self.researcher_agent = ResearcherAgent()
        print("  ✅ Chercheur Tourisme prêt\n")

        print("  [3/3] 📊 Analyste de Données...")
        from agents.data_analytics_agent import DataAnalyticsAgent
        self.analytics_agent = DataAnalyticsAgent()
        print("  ✅ Analyste de Données prêt\n")

        print("  [4/4] 🔮 Prévisionniste...")
        try:
            from agents.prediction_agent import PredictionAgent
            # Prediction needs an APF DataFrame for its rule-based forecasts.
            # SQL-on-demand mode keeps the APF df cached on _apf_df.
            _df = getattr(self.analytics_agent, "_apf_df", None)
            if _df is not None:
                self.prediction_agent = PredictionAgent(_df)
            else:
                self.prediction_agent = None
                logger.warning("PredictionAgent not initialized: no APF DataFrame")
        except Exception as _e:
            self.prediction_agent = None
            logger.warning("PredictionAgent init failed (non-critical): %s", _e)
        print("  ✅ Prévisionniste prêt\n")

        # ── State ──
        self.last_agent: Optional[str] = None
        self.message_count: int = 0
        self.routing_history: List[Tuple[str, str, bool]] = []
        self.conversation_log: List[Tuple[str, str]] = []

        # ── Session summary (ConversationSummaryBuffer for the classifier) ──
        # Compresses old exchanges so the routing LLM always gets compact context.
        self._session_summary: str = ""
        self._last_summarized_at: int = 0   # message_count when last summarized

        # ── Dataset year range ──
        self.min_year, self.max_year = self._get_dataset_year_range()

        # ── Ready ──
        print("-" * 60)
        print("✅ Tous les agents sont opérationnels.")
        print(f"   📊 Datasets: {len(self.analytics_agent.datasets)}")
        if self.min_year and self.max_year:
            print(f"   📅 Plage données: {self.min_year} → {self.max_year}")
        print("   📚 RAG + 🌐 Web: prêts")
        print(f"   🔮 Prévisionniste: {'prêt' if self.prediction_agent else 'indisponible'}")
        print("-" * 60)
        print()

        logger.info(
            "Orchestrator ready — %d datasets, year range: %s–%s, prediction=%s",
            len(self.analytics_agent.datasets),
            self.min_year or "?",
            self.max_year or "?",
            "ready" if self.prediction_agent else "unavailable",
        )

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _get_dataset_year_range(self) -> Tuple[Optional[int], Optional[int]]:
        """Extract min/max year — first from KPI cache (cheap), then from
        the in-memory APF DataFrame the analytics agent keeps around."""
        try:
            kc = getattr(self.analytics_agent, "kpi_cache", None)
            if kc:
                years = kc.years_available()
                if years:
                    return int(min(years)), int(max(years))
        except Exception as e:
            logger.debug("Year range from KPI cache failed: %s", e)
        try:
            df = getattr(self.analytics_agent, "_apf_df", None)
            if df is not None:
                for col in df.columns:
                    if "year" in col.lower():
                        return int(df[col].min()), int(df[col].max())
        except Exception as e:
            logger.debug("Could not determine year range from _apf_df: %s", e)
        return None, None

    def _years_in_message(self, message: str) -> List[int]:
        """Extract 4-digit years (1900-2099) from the message."""
        return [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", message)]

    def _all_years_outside_range(self, years: List[int]) -> bool:
        """Check if ALL mentioned years are outside the dataset range."""
        if not years or not self.min_year or not self.max_year:
            return False
        return all(y < self.min_year or y > self.max_year for y in years)

    def _build_conversation_context(self) -> str:
        """Build compact context for the LLM classifier — last 3 exchanges only."""
        if not self.conversation_log:
            return "Premier message — pas d'historique."

        recent = self.conversation_log[-6:]  # last 3 exchanges
        lines = []
        for role, text in recent:
            short = text[:150].replace("\n", " ")
            prefix = "USER" if role == "user" else role.upper()
            lines.append(f"  {prefix}: {short}")
        return "Échanges récents:\n" + "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1 — INSTANT RULES (zero latency, 100% reliable)
    # ══════════════════════════════════════════════════════════════════════

    def _classify_instant(self, user_message: str) -> Optional[str]:
        """
        Catch obvious cases instantly without calling the LLM.

        Returns:
            Agent key ('normal', 'researcher', 'analytics') or None.
        """
        msg = user_message.lower().strip()

        # ── Slash commands → ALWAYS analytics ──
        if msg.startswith("/"):
            return "analytics"

        # ── Pure greetings (exact match) → normal ──
        if msg in _PURE_GREETINGS:
            return "normal"

        # ── Explicit search/internet requests → ALWAYS researcher ──
        for phrase in _SEARCH_PHRASES:
            if phrase in msg:
                return "researcher"

        # NOTE: Chart/visualization keywords are NOT forced to analytics here.
        # Whether a chart uses internal or external data is determined by the LLM (Layer 2).
        # The two-phase researcher→analytics pipeline in _route_internal handles external charts.

        # ── Prediction / forecasting keywords → prediction agent ──
        for kw in _PREDICTION_KEYWORDS:
            if kw in msg:
                return "prediction"

        # ── Years outside dataset range → researcher ──
        years = self._years_in_message(user_message)
        if years and self._all_years_outside_range(years):
            return "researcher"

        # ── Follow-up / short conversational replies → same agent ──
        if self.last_agent:
            # Exact match follow-ups (single words like "oui", "ok", "go")
            if msg in _FOLLOWUP_EXACT:
                return self.last_agent

            # Message STARTS WITH a follow-up word + is short (≤ 6 words)
            # Handles "ok fait moi tous", "oui vas-y tout", "okay donne tout", etc.
            first_word = msg.split()[0] if msg.split() else ""
            if first_word in _FOLLOWUP_EXACT and len(msg.split()) <= 6:
                return self.last_agent

            # Phrase-based follow-ups — but skip if message has analytics keywords.
            # "donne moi le top 5 des nationalites" must go to analytics, not stay on
            # normal just because "donne moi" is in _FOLLOWUP_PHRASES.
            has_analytics_kw = any(kw in msg for kw in _ANALYTICS_KEYWORDS)
            if not has_analytics_kw:
                for phrase in _FOLLOWUP_PHRASES:
                    if phrase in msg:
                        return self.last_agent

            # Very short message (1-4 words) — only follow-up if no analytics intent
            if len(msg.split()) <= 4 and self.message_count > 0 and not has_analytics_kw:
                return self.last_agent

        # ── "Données les plus récentes / dernières" → always analytics ──
        _RECENCY_DATA = [
            "données les plus récentes", "données récentes", "données actuelles",
            "dernières données", "dernière période", "période récente",
            "données disponibles", "données de la dernière", "plus récentes",
        ]
        if any(p in msg for p in _RECENCY_DATA) and any(kw in msg for kw in _ANALYTICS_KEYWORDS):
            return "analytics"

        # ── Not obvious — needs LLM ──
        return None

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 2 — LLM CLASSIFICATION (understands intent, nuance, typos)
    # ══════════════════════════════════════════════════════════════════════

    def _classify_llm(self, user_message: str) -> Optional[str]:
        """
        LLM-based intent classification.
        Handles reasoning models that don't support custom temperature.
        """
        from config.settings import MODEL_IS_REASONING

        # ── Check cache ──
        cache_key = f"{self.last_agent or 'none'}:{user_message}"
        cached = self._classify_cache.get(cache_key, source="classify")
        if cached is not None:
            logger.debug(
                "Classification cache hit: '%s' → %s",
                user_message[:40], cached,
            )
            return cached

        # ── Build classifier prompt ──
        min_y = self.min_year or "?"
        max_y = self.max_year or "?"
        last = self.last_agent or "AUCUN (premier message)"
        context = self._build_conversation_context()

        prompt = (
            CLASSIFIER_SYSTEM_PROMPT
            .replace("{min_year}", str(min_y))
            .replace("{max_year}", str(max_y))
            .replace("{last_agent}", str(last).upper())
            .replace("{conversation_context}", context)
        )

        # ── Build kwargs ──
        kwargs = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
            "max_completion_tokens": CLASSIFIER_MAX_TOKENS,
        }

        # Only pass temperature for non-reasoning models
        if not MODEL_IS_REASONING and CLASSIFIER_TEMPERATURE != 1:
            kwargs["temperature"] = CLASSIFIER_TEMPERATURE

        # reasoning_effort — "minimal" → near-zero latency for one-word routing.
        # Auto-recover if the API version doesn't support it.
        if (
            MODEL_IS_REASONING
            and CLASSIFIER_REASONING_EFFORT
            and not _should_skip_reasoning_effort()
        ):
            kwargs["reasoning_effort"] = CLASSIFIER_REASONING_EFFORT

        # ── Call LLM ──
        try:
            response = self.client.chat.completions.create(**kwargs)

            if not response.choices:
                logger.warning("LLM classifier returned no choices")
                return None

            raw = response.choices[0].message.content
            if not raw:
                logger.warning("LLM classifier returned empty content")
                return None

            raw = raw.strip().upper()

            result = None
            if "PREDICTION" in raw:
                result = "prediction"
            elif "ANALYTICS" in raw:
                result = "analytics"
            elif "RESEARCHER" in raw:
                result = "researcher"
            elif "NORMAL" in raw:
                result = "normal"

            if result:
                self._classify_cache.set(cache_key, result, source="classify")
                logger.debug(
                    "LLM classified '%s' → %s (raw: '%s')",
                    user_message[:40], result, raw[:20],
                )

            return result

        except Exception as e:
            error_str = str(e).lower()

            # Auto-recover from reasoning_effort rejection
            if "reasoning_effort" in error_str or "reasoning effort" in error_str:
                _mark_reasoning_effort_rejected()
                kwargs.pop("reasoning_effort", None)
                try:
                    response = self.client.chat.completions.create(**kwargs)
                    if response.choices and response.choices[0].message.content:
                        raw = response.choices[0].message.content.strip().upper()
                        result = None
                        if "PREDICTION" in raw:
                            result = "prediction"
                        elif "ANALYTICS" in raw:
                            result = "analytics"
                        elif "RESEARCHER" in raw:
                            result = "researcher"
                        elif "NORMAL" in raw:
                            result = "normal"
                        if result:
                            self._classify_cache.set(cache_key, result, source="classify")
                        return result
                except Exception as retry_e:
                    logger.error("Classifier retry after reasoning_effort removal failed: %s", retry_e)
                return None

            # Auto-recover from temperature/system-role errors
            if "temperature" in error_str or "system" in error_str:
                logger.warning(
                    "Classifier: model compatibility issue — retrying clean"
                )
                try:
                    # Retry with no temperature, convert system to developer
                    retry_msgs = [
                        {"role": "developer", "content": prompt},
                        {"role": "user", "content": user_message},
                    ]
                    response = self.client.chat.completions.create(
                        model=self.deployment,
                        messages=retry_msgs,
                        max_completion_tokens=CLASSIFIER_MAX_TOKENS,
                    )

                    if not response.choices:
                        return None

                    raw = response.choices[0].message.content
                    if not raw:
                        return None

                    raw = raw.strip().upper()
                    result = None
                    if "PREDICTION" in raw:
                        result = "prediction"
                    elif "ANALYTICS" in raw:
                        result = "analytics"
                    elif "RESEARCHER" in raw:
                        result = "researcher"
                    elif "NORMAL" in raw:
                        result = "normal"

                    if result:
                        self._classify_cache.set(
                            cache_key, result, source="classify"
                        )
                    return result

                except Exception as retry_err:
                    logger.error(
                        "Classifier retry also failed: %s", retry_err
                    )
                    return None

            logger.error("LLM classification failed: %s", e)
            return None
    # ══════════════════════════════════════════════════════════════════════
    # LAYER 3 — KEYWORD FALLBACK (only if LLM fails completely)
    # ══════════════════════════════════════════════════════════════════════

    def _classify_fallback(self, user_message: str) -> str:
        """
        Last resort keyword matching. Only runs if LLM returns None.

        Returns:
            Agent key (always returns a value — never None).
        """
        msg = user_message.lower().strip()

        # Prediction keywords (check before analytics to avoid misrouting forecasts)
        for kw in _PREDICTION_KEYWORDS:
            if kw in msg:
                return "prediction"

        # Analytics keywords
        for kw in _ANALYTICS_KEYWORDS:
            if kw in msg:
                return "analytics"

        # Researcher keywords
        for kw in _RESEARCHER_KEYWORDS:
            if kw in msg:
                return "researcher"

        # Last resort: reuse last agent or default to analytics
        fallback = self.last_agent or "analytics"
        logger.debug(
            "Keyword fallback for '%s' → %s (last_agent=%s)",
            user_message[:40], fallback, self.last_agent,
        )
        return fallback

    # ══════════════════════════════════════════════════════════════════════
    # MAIN CLASSIFY — 3 LAYERS
    # ══════════════════════════════════════════════════════════════════════

    def classify(self, user_message: str) -> str:
        """
        Classify user intent through 3 layers:
            Layer 1: Instant rules (0ms, 100% reliable for obvious cases)
            Layer 2: LLM understanding (1-2s, handles nuance/typos/ambiguity)
            Layer 3: Keyword fallback (0ms, last resort if LLM fails)

        Args:
            user_message: The user's input text.

        Returns:
            Agent key: 'normal', 'researcher', or 'analytics'.
        """
        # ── LAYER 1: Instant ──
        instant = self._classify_instant(user_message)
        if instant:
            logger.debug(
                "Layer 1 (instant): '%s' → %s",
                user_message[:40], instant,
            )
            return instant

        # ── LAYER 2: LLM ──
        llm_result = self._classify_llm(user_message)
        if llm_result:
            # Safety: double-check years even if LLM says analytics
            years = self._years_in_message(user_message)
            if (
                llm_result == "analytics"
                and years
                and self._all_years_outside_range(years)
            ):
                logger.info(
                    "LLM said analytics but years %s are outside range — "
                    "overriding to researcher",
                    years,
                )
                return "researcher"

            logger.debug(
                "Layer 2 (LLM): '%s' → %s",
                user_message[:40], llm_result,
            )
            return llm_result

        # ── LAYER 3: Fallback ──
        fallback = self._classify_fallback(user_message)
        logger.debug(
            "Layer 3 (fallback): '%s' → %s",
            user_message[:40], fallback,
        )
        return fallback

    # ══════════════════════════════════════════════════════════════════════
    # AUTO-ENRICHMENT (Analytics response + external tourism factors)
    # ══════════════════════════════════════════════════════════════════════

    _ENRICHMENT_TRIGGER_WORDS = (
        "pourquoi", "why", "cause", "raison", "expliqu", "facteur",
        "contexte", "impact", "influence", "baisse", "hausse", "évolution",
        "comparaison", "vs", "versus", "flux", "tendance",
    )

    _ENRICHMENT_MONTHS = (
        "janvier", "février", "fevrier", "mars", "avril", "mai", "juin",
        "juillet", "août", "aout", "septembre", "octobre", "novembre",
        "décembre", "decembre",
    )

    def _should_enrich_with_context(self, message: str, response: str) -> bool:
        """Decide whether an analytics answer should be enriched with external factors.

        True if: the user asked "why/context/..." OR the response is a
        quantitative answer scoped to a specific period.
        """
        msg_lower = message.lower()

        if any(w in msg_lower for w in self._ENRICHMENT_TRIGGER_WORDS):
            return True

        has_numbers = bool(re.search(r'\*\*[\d,]+\*\*', response))
        has_year = bool(re.search(r'\b20[12]\d\b', msg_lower))
        has_month = any(m in msg_lower for m in self._ENRICHMENT_MONTHS)
        return has_numbers and (has_year or has_month)

    def _build_external_search_query(self, message: str) -> str:
        """Build a Tavily query targeting external tourism factors for the period."""
        msg_lower = message.lower()

        year_match = re.search(r'\b(20[12]\d)\b', message)
        year = year_match.group(1) if year_match else str(self.max_year or "")

        month = next((m for m in self._ENRICHMENT_MONTHS if m in msg_lower), None)
        period = f"{month} {year}".strip() if month else year

        return (
            f"tourisme maroc {period} arrivées facteurs économiques "
            f"vols conjoncture"
        ).strip()

    def _get_external_factors(self, message: str) -> str:
        """Search Tavily for context factors and return a formatted appendix.

        Returns empty string on any failure — enrichment must never break
        the main analytics response.
        """
        try:
            searcher = getattr(self.researcher_agent, "searcher", None)
            if not searcher:
                return ""

            query = self._build_external_search_query(message)
            logger.info("External factors search: %s", query)

            results = searcher.search(query, max_results=3)
            if not results:
                return ""

            lines = ["\n\n---\n### 🌐 Facteurs externes (contexte de la période)\n"]
            for r in results[:3]:
                title = r.get("title", "")
                content = (r.get("content") or "")[:200]
                url = r.get("url", "")
                if title and content:
                    lines.append(f"**{title}**\n{content}...\n*({url})*\n")

            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception as e:
            logger.debug("External factors search failed (non-critical): %s", e)
            return ""

    # ══════════════════════════════════════════════════════════════════════
    # AUTO-REROUTE (Analytics → Researcher when data unavailable)
    # ══════════════════════════════════════════════════════════════════════

    def _should_reroute_to_researcher(self, response: str) -> bool:
        """Check if analytics response indicates data genuinely absent from the dataset.
        Does NOT return True for code execution failures — those should not be rerouted
        because the data exists; only the generated code had a bug.
        """
        resp_lower = response.lower()
        if any(ind in resp_lower for ind in _CODE_FAILURE_INDICATORS):
            return False  # code crash ≠ missing data — never reroute
        return any(ind in resp_lower for ind in NO_DATA_INDICATORS)

    # ══════════════════════════════════════════════════════════════════════
    # ROUTE & EXECUTE — Thread-safe
    # ══════════════════════════════════════════════════════════════════════

    def route(self, user_message: str) -> Dict:
        """
        Classify, route, and execute a user message.

        Args:
            user_message: The user's input text.

        Returns:
            Dict with keys:
                - agent (str): Agent key
                - agent_icon (str): Emoji icon
                - agent_name (str): Display name
                - response (str): Agent response
                - rerouted (bool): Whether auto-reroute occurred
                - classification_time_ms (float): Time for classification
                - total_time_ms (float): Total processing time
        """
        return self._route_internal(user_message)

    def _route_internal(self, user_message: str) -> Dict:
        """Internal routing logic."""

        self.message_count += 1
        start = time.time()

        # ── Classify ──
        agent_key = self.classify(user_message)
        classification_time = (time.time() - start) * 1000

        icons = {"normal": "🏛️", "researcher": "🔍", "analytics": "📊", "prediction": "🔮"}
        agent_icon = icons.get(agent_key, "🤖")
        agent_name = AGENT_NAMES.get(agent_key, "Agent")

        response = ""
        rerouted = False

        # ── Detect chart/visualization request ──
        _CHART_KW = ["chart", "graphe", "graphique", "visualis", "courbe",
                     "diagramme", "histogram", "heatmap", "plot", "trace", "affiche"]
        is_chart_request = any(kw in user_message.lower() for kw in _CHART_KW)

        # ── Execute ──
        try:
            if agent_key == "analytics":
                # Always inject last researcher context when switching FROM researcher.
                # No reference-word check — any analytics request after researcher needs context.
                actual_message = user_message
                if self.last_agent == "researcher":
                    prev_researcher = [
                        t for r, t in self.conversation_log if r == "researcher"
                    ]
                    if prev_researcher:
                        ctx = prev_researcher[-1][:800]
                        actual_message = (
                            f"{user_message}\n\n"
                            f"[Contexte des recherches précédentes du Chercheur STATOUR:\n"
                            f"{ctx}]"
                        )

                response = self.analytics_agent.chat(actual_message)

                # Auto-reroute to researcher if analytics found no data in the dataset.
                # EXCEPTION: chart requests never reroute — researcher cannot render charts.
                # Analytics must produce the chart even with partial/limited data.
                if self._should_reroute_to_researcher(response) and not is_chart_request:
                    logger.info("Auto-rerouting to researcher (no data in dataset)")
                    response = self.researcher_agent.chat(user_message)
                    rerouted = True
                    agent_key = "researcher"
                    agent_icon = "🔍"
                    agent_name = "Chercheur Tourisme"
                else:
                    # Auto-enrich quantitative analytics answers with external
                    # tourism context (flights, conjuncture, events) — skip for
                    # charts since the visual already carries the story.
                    if (
                        not is_chart_request
                        and self._should_enrich_with_context(user_message, response)
                    ):
                        external = self._get_external_factors(user_message)
                        if external:
                            response = response + external

            elif agent_key == "researcher":
                response = self.researcher_agent.chat(user_message)

                # Two-phase pipeline: researcher gathers external data → analytics renders chart.
                # This handles "chart des tendances mondiales", "chart Maroc vs Espagne", etc.
                if is_chart_request and response and "⚠️" not in response:
                    logger.info(
                        "Chart request with external data — handing off to analytics for rendering"
                    )
                    chart_msg = (
                        f"{user_message}\n\n"
                        f"[Données collectées par le Chercheur STATOUR pour ce graphique:\n"
                        f"{response[:1200]}\n\n"
                        f"Génère le graphique en Python à partir de ces données. "
                        f"Si les données sont sous forme de texte, crée un DataFrame Python "
                        f"avec des listes (pd.DataFrame({{'col': [...], 'val': [...]}})) — "
                        f"n'essaie pas de charger des fichiers.]"
                    )
                    chart_response = self.analytics_agent.chat(chart_msg)
                    if chart_response and "⚠️ No response" not in chart_response:
                        # Keep researcher info in response header, replace body with chart
                        response = chart_response
                        agent_key = "analytics"
                        agent_icon = "📊"
                        agent_name = "Analyste de Données"
                        rerouted = True
                        logger.info("Two-phase chart pipeline complete (researcher→analytics)")

            elif agent_key == "prediction":
                if self.prediction_agent:
                    pred_result = self.prediction_agent.chat(user_message)
                    response = pred_result["response"]
                else:
                    # Fallback: route to analytics if prediction agent unavailable
                    logger.warning("PredictionAgent unavailable — routing to analytics")
                    response = self.analytics_agent.chat(user_message)
                    agent_key = "analytics"
                    agent_icon = "📊"
                    agent_name = "Analyste de Données"

            else:
                response = self.normal_agent.chat(user_message)

        except Exception as e:
            response = f"❌ Erreur: {str(e)}"
            logger.error(
                "Agent %s failed for '%s': %s",
                agent_key, user_message[:50], e,
                exc_info=True,
            )

        total_time = (time.time() - start) * 1000

        # ── Update state ──
        self.last_agent = agent_key

        self.conversation_log.append(("user", user_message))
        # Store more context than before (500 chars, was 200)
        self.conversation_log.append((agent_key, response[:500]))

        # Keep conversation log bounded (30 entries, was 20)
        if len(self.conversation_log) > 30:
            self.conversation_log = self.conversation_log[-30:]

        self.routing_history.append((user_message, agent_key, rerouted))

        # Keep routing history bounded
        if len(self.routing_history) > 100:
            self.routing_history = self.routing_history[-100:]

        logger.info(
            "Routed '%s' → %s%s [classify=%.0fms, total=%.0fms]",
            user_message[:50],
            agent_key,
            " (rerouted)" if rerouted else "",
            classification_time,
            total_time,
        )

        return {
            "agent": agent_key,
            "agent_icon": agent_icon,
            "agent_name": agent_name,
            "response": response,
            "rerouted": rerouted,
            "classification_time_ms": round(classification_time, 1),
            "total_time_ms": round(total_time, 1),
        }

    def chat(self, user_message: str) -> str:
        """Simple chat interface — returns just the response text."""
        result = self.route(user_message)
        return result["response"]

    # ══════════════════════════════════════════════════════════════════════
    # ORCHESTRATOR COMMANDS
    # ══════════════════════════════════════════════════════════════════════

    def handle_orchestrator_commands(self, user_message: str) -> Optional[str]:
        """
        Handle orchestrator-level commands.

        Returns:
            Command output string, or None if not a recognized command.
        """
        msg = user_message.strip().lower()

        if msg == "/help":
            return self._help_text()
        elif msg == "/agents":
            return self._agents_status()
        elif msg == "/history":
            return self._routing_history_text()
        elif msg == "/resetall":
            return self._reset_all()
        elif msg == "/cache":
            return self._cache_stats()

        return None

    def _help_text(self) -> str:
        """Generate help text."""
        yr = ""
        if self.min_year and self.max_year:
            yr = (
                f"\n📅 Données: {self.min_year}–{self.max_year} "
                f"(hors plage → web auto)"
            )

        return f"""📋 **Commandes Orchestrateur:**
  `/help`      — Cette aide
  `/agents`    — Statut des agents
  `/history`   — Historique de routage
  `/cache`     — Statistiques cache
  `/resetall`  — Réinitialiser tout

📋 **Commandes Données:**
  `/datasets`  — Datasets chargés
  `/stats`     — Stats du dataset actif
  `/schema`    — Schéma du dataset
  `/columns`   — Colonnes
  `/sample`    — Échantillon
  `/switch X`  — Changer de dataset
  `/load X`    — Charger un fichier
  `/reset`     — Reset conversation
{yr}

💡 Posez votre question naturellement. Le routage est automatique."""

    def _agents_status(self) -> str:
        """Generate agent status report."""
        lines = ["📊 **Agents STATOUR:**\n"]

        agents_info = [
            ("🏛️", "Assistant Général", "normal", self.normal_agent),
            ("🔍", "Chercheur Tourisme", "researcher", self.researcher_agent),
            ("📊", "Analyste de Données", "analytics", self.analytics_agent),
        ]

        for icon, name, key, agent in agents_info:
            active = "✅" if ACTIVE_AGENTS.get(key) else "❌"
            msgs = agent.get_conversation_length()
            lines.append(f"  {icon} {name} — {active} — {msgs} msgs")

        lines.append(f"\n  📁 Datasets: {len(self.analytics_agent.datasets)}")
        if self.min_year and self.max_year:
            lines.append(f"  📅 Plage: {self.min_year}–{self.max_year}")
        lines.append(f"  💬 Messages: {self.message_count}")
        lines.append(f"  🎯 Dernier agent: {self.last_agent or '-'}")

        return "\n".join(lines)

    def _routing_history_text(self) -> str:
        """Generate routing history report."""
        if not self.routing_history:
            return "📭 Aucun historique de routage."

        lines = ["📜 **Historique de routage (10 derniers):**\n"]
        icons = {"normal": "🏛️", "researcher": "🔍", "analytics": "📊"}

        for msg, agent, rr in self.routing_history[-10:]:
            short = msg[:60] + "..." if len(msg) > 60 else msg
            suffix = " → 🔍 (rerouted)" if rr else ""
            icon = icons.get(agent, "🤖")
            lines.append(f"  {icon} `{short}`{suffix}")

        return "\n".join(lines)

    def _cache_stats(self) -> str:
        """Generate cache statistics report."""
        stats = self._classify_cache.stats()
        lines = [
            "🗄️  **Cache Classification:**",
            f"  Size: {stats['size']}/{stats['max_size']}",
            f"  Hits: {stats['hits']}",
            f"  Misses: {stats['misses']}",
            f"  Hit rate: {stats['hit_rate_pct']}%",
            f"  TTL: {stats['ttl_seconds']}s",
        ]

        from utils.cache import shared_cache
        shared_stats = shared_cache.stats()
        lines.extend([
            "",
            "🗄️  **Cache Partagé (search/RAG):**",
            f"  Size: {shared_stats['size']}/{shared_stats['max_size']}",
            f"  Hits: {shared_stats['hits']}",
            f"  Misses: {shared_stats['misses']}",
            f"  Hit rate: {shared_stats['hit_rate_pct']}%",
            f"  By source: {shared_stats['by_source']}",
        ])

        return "\n".join(lines)

    def _reset_all(self) -> str:
        """Reset all agents and orchestrator state."""
        self.normal_agent.reset_conversation()
        self.researcher_agent.reset_conversation()
        self.analytics_agent.reset_conversation()

        self.last_agent = None
        self.conversation_log.clear()
        self.routing_history.clear()
        self.message_count = 0
        self._classify_cache.clear()

        from utils.cache import shared_cache
        shared_cache.clear()

        logger.info("Full system reset")
        return "🔄 Tout réinitialisé (agents + caches)."


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Interactive CLI for the STATOUR Orchestrator."""

    try:
        orch = Orchestrator()
    except Exception as e:
        print(f"\n❌ Failed to start orchestrator: {e}")
        return

    print("\n💡 Posez votre question naturellement. Routage automatique.")
    print("   /help pour les commandes, /quit pour quitter.\n")

    while True:
        try:
            user_input = input("👤 Vous: ").strip()
            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("\n👋 Au revoir!")
                break

            # ── Handle orchestrator commands ──
            cmd_result = orch.handle_orchestrator_commands(user_input)
            if cmd_result:
                print(f"\n🎯 Orchestrateur:\n{cmd_result}\n")
                continue

            # ── Route to agent ──
            result = orch.route(user_input)

            header = f"{result['agent_icon']} {result['agent_name']}"
            timing = (
                f"[classify={result['classification_time_ms']}ms | "
                f"total={result['total_time_ms']}ms]"
            )

            if result["rerouted"]:
                header += " (auto-reroute)"

            print(f"\n{header}  {timing}")
            print(f"{result['response']}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur inattendue: {e}\n")
            logger.error("CLI error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()