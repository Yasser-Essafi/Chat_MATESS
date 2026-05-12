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
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

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
    ORCHESTRATOR_RECENT_EXCHANGES,
    ORCHESTRATOR_SUMMARY_MAX_TOKENS,
)
from utils.base_agent import (
    get_shared_client,
    _should_skip_reasoning_effort,
    _mark_reasoning_effort_rejected,
)
from utils.logger import get_logger
from utils.cache import SearchCache

logger = get_logger("statour.orchestrator")


def _norm_text(text: str) -> str:
    if any(marker in (text or "") for marker in ("Ã", "Â", "â")):
        try:
            text = text.encode("latin1").decode("utf-8")
        except Exception:
            pass
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _period_metadata(text: str) -> Dict:
    norm = _norm_text(text)
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")]
    months = {
        "janvier": 1, "jan": 1, "fevrier": 2, "fev": 2, "février": 2,
        "mars": 3, "avril": 4, "avr": 4, "mai": 5, "juin": 6,
        "juillet": 7, "juil": 7, "aout": 8, "août": 8, "septembre": 9,
        "sep": 9, "octobre": 10, "novembre": 11, "decembre": 12, "décembre": 12,
    }
    month = next((num for name, num in months.items() if re.search(r"\b" + re.escape(_norm_text(name)) + r"\b", norm)), None)
    out: Dict[str, object] = {}
    if years:
        out["years"] = years
        if len(years) == 1:
            out["year"] = years[0]
    if month:
        out["month"] = month
    return out


def _metric_context_metadata(text: str, agent_key: str) -> str:
    norm = _norm_text(text)
    if agent_key == "human_advisor":
        return "advisory"
    if agent_key == "prediction":
        return "prediction"
    if agent_key == "researcher":
        return "research"
    if "apf" in norm or "frontiere" in norm or re.search(r"\b(mre|tes)\b", norm):
        return "apf"
    if any(k in norm for k in ["hotel", "hoteliere", "hebergement", "nuitee", "dms", "stdn", "ehtc"]):
        return "hebergement"
    if "arrive" in norm:
        return "ambiguous"
    return agent_key if agent_key in {"normal", "executive_insight"} else "ambiguous"


def _chart_path_from_response(text: str) -> Optional[str]:
    if not text:
        return None
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    charts_dir = os.path.realpath(os.path.join(project_root, "charts"))
    for pattern in [r"Chart:\s*(.+?\.html)", r"((?:[^\s]*/charts/|charts/)[^\s]+\.html)"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        path = match.group(1).strip()
        if not os.path.isabs(path):
            path = os.path.join(project_root, path)
        real = os.path.realpath(path)
        if real.startswith(charts_dir) and os.path.exists(real):
            return real
    return None


def _chart_paths_from_response(text: str, limit: int = 4) -> List[str]:
    if not text:
        return []
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    charts_dir = os.path.realpath(os.path.join(project_root, "charts"))
    paths: List[str] = []
    for pattern in [r"Chart:\s*(.+?\.html)", r"((?:[^\s]*/charts/|charts/)[^\s]+\.html)"]:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            path = match.group(1).strip().strip("`),.;")
            if not os.path.isabs(path):
                path = os.path.join(project_root, path)
            real = os.path.realpath(path)
            if real.startswith(charts_dir) and os.path.exists(real) and real not in paths:
                paths.append(real)
                if len(paths) >= limit:
                    return paths
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# LLM Classifier Prompt Template
# ══════════════════════════════════════════════════════════════════════════════
# {min_year}, {max_year}, {last_agent}, {conversation_context} are replaced
# at runtime with actual values.

CLASSIFIER_SYSTEM_PROMPT = """Tu es le routeur de STATOUR (Ministère du Tourisme Maroc).
Analyse l'INTENTION de l'utilisateur et route vers le bon agent.

TES 4 AGENTS :

ANALYTICS — Analyse données Fabric Lakehouse Gold :
  • APF (postes frontières) : arrivées, MRE, TES, voies, nationalités, continents → {apf_min_year}–{apf_max_year}
  • HÉBERGEMENT (EHTC) : nuitées, arrivées hôtelières, taux occupation, délégations → {hbg_min_year}–{hbg_max_year}
  → Questions avec chiffres/stats/graphiques dans ces plages → ANALYTICS

RESEARCHER — Recherche web + RAG :
  → Actualités, contexte, Vision 2030, facteurs externes, "pourquoi", données hors plages
  → Si l'utilisateur veut EXPLIQUER des données déjà calculées (pourquoi hausse/baisse) → RESEARCHER
  → Stratégie, benchmarks internationaux, news → RESEARCHER

PREDICTION — Prévisions futures :
  → Années futures (2027+), estimations, scénarios, forecasts → PREDICTION

NORMAL — Conversation générale :
  → Salutations, aide, questions sur STATOUR, conversation → NORMAL

RÈGLES STRICTES :
1. Commandes / → ANALYTICS
2. "cherche/search/google/internet" explicite → RESEARCHER
3. Données dans plage APF ({apf_min_year}–{apf_max_year}) ou HÉBERGEMENT ({hbg_min_year}–{hbg_max_year}) → ANALYTICS
4. Données hors plages → RESEARCHER
5. Années 2027+ ou "estimer/prévoir" → PREDICTION
6. "pourquoi X a augmenté/baissé" sur données déjà calculées → RESEARCHER
7. Salutations/aide → NORMAL
8. Follow-up court sur même sujet → MÊME agent: {last_agent}
9. Doute → {last_agent} si existant, sinon ANALYTICS

CONTEXTE CONVERSATION :
{conversation_context}

UN SEUL MOT : ANALYTICS ou RESEARCHER ou PREDICTION ou NORMAL"""


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
# Domain Detection Constants
# ══════════════════════════════════════════════════════════════════════════════
# Used to track which data table the conversation is currently about so that
# follow-ups like "et en 2025" stay in the right domain (hébergement vs APF).

_APF_DOMAIN_KW = [
    "mre", "tes", "frontière", "frontieres", "poste frontière", "postes frontières",
    "apf", "voie aérienne", "voie maritime", "voie terrestre",
    "arrivées aux postes", "marocains résidant", "diaspora",
    "séjournistes", "sejournistes",
]
_HEBERGEMENT_DOMAIN_KW = [
    "nuitée", "nuitee", "nuitées", "nuitees",
    "hébergement", "hebergement",
    "hôtel", "hotel",
    "ehtc", "stdn",
    "taux d'occupation", "taux occupation",
    "chambre", "capacité chambre",
    "maison d'hôte", "riad", "camping",
    "délégation", "delegation",
    "arrivées hôtelières", "arrivees hotelières",
    "établissement classé", "etablissement classe",
]


def _detect_domain(text: str) -> Optional[str]:
    """
    Infer domain from combined user+assistant text (0ms, no LLM).
    Returns "hebergement", "apf", or None when ambiguous.
    Scoring: whichever domain has more keyword hits wins; ties → None.
    """
    t = text.lower()
    h_score = sum(1 for kw in _HEBERGEMENT_DOMAIN_KW if kw in t)
    a_score = sum(1 for kw in _APF_DOMAIN_KW if kw in t)
    if h_score > a_score:
        return "hebergement"
    if a_score > h_score:
        return "apf"
    return None


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

# Pre-compute accent-stripped versions for matching user input without accents
_PREDICTION_KEYWORDS_NORM = [_norm_text(kw) for kw in _PREDICTION_KEYWORDS]


def _unsupported_prediction_target(message: str) -> bool:
    norm = _norm_text(message)
    target = r"(?:mars|lune|jupiter|venus|saturne|mercure|neptune|uranus)"
    if re.search(rf"\b(?:sur|vers|a|au|aux)\s+{target}\b", norm):
        return True
    if re.search(rf"\b(?:visit\w*|visite\w*)\b.{{0,40}}\b{target}\b", norm):
        return True
    if re.search(rf"\b{target}\b", norm) and re.search(r"\bplanete|spatial|extra[-\s]?terrestre\b", norm):
        return True
    return False


def _has_prediction_intent(msg: str, norm: str) -> bool:
    """Check prediction keywords against both raw-lowered and accent-stripped text."""
    if _unsupported_prediction_target(msg):
        return False
    for kw in _PREDICTION_KEYWORDS:
        if kw in msg:
            return True
    for kw_norm in _PREDICTION_KEYWORDS_NORM:
        if kw_norm in norm:
            return True
    return False


def _requires_planned_flow(message: str) -> bool:
    """Detect multi-deliverable turns that should be planned, not single-routed.

    This is intentionally broad: it does not special-case a prompt, it only
    identifies when the user asks for a historical/analytical part and a
    forecasting part in the same turn. The planner can then decide the exact
    tool sequence.
    """
    raw = (message or "").lower()
    norm = _norm_text(message)
    if not _has_prediction_intent(raw, norm):
        return False

    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", message or "")]
    distinct_years = sorted(set(years))
    has_multi_year_scope = len(distinct_years) >= 2 and distinct_years[0] < distinct_years[-1]
    has_historical_scope = has_multi_year_scope and (
        any(k in norm for k in ["historique", "depuis", "jusqu", "entre"])
        or re.search(r"\bde\s+(?:19|20)\d{2}\b", norm)
    )
    has_analysis_intent = any(
        k in norm
        for k in [
            "analyse", "analyser", "evolution", "tendance", "croissance",
            "baisse", "hausse", "comparaison", "compare", "diagnostic",
        ]
    )
    has_tourism_metric = any(
        k in norm
        for k in [
            "tourisme", "touriste", "arrive", "nuitee", "recette",
            "hebergement", "apf", "mre", "tes",
        ]
    )
    has_connector = any(k in norm for k in [" puis ", " et ", " avec ", " ainsi que ", " ensuite "])
    has_non_prediction_chart = (
        any(k in norm for k in ["graph", "chart", "courbe", "visualis", "diagramme"])
        and has_analysis_intent
    )

    return bool(
        has_analysis_intent
        and (has_tourism_metric or has_multi_year_scope)
        and (has_historical_scope or (has_multi_year_scope and has_connector) or has_non_prediction_chart)
    )


@dataclass
class ConversationRuntimeState:
    """Per-conversation state kept outside persisted chat-history JSON.

    The frontend already persists display messages. This runtime state stores
    only the operational memory needed by the router and agents: active domain,
    last agent, compact routing context and per-agent folded histories.
    """

    conversation_id: str
    last_agent: Optional[str] = None
    message_count: int = 0
    routing_history: List[Tuple[str, str, bool]] = field(default_factory=list)
    conversation_log: List[Tuple[str, str]] = field(default_factory=list)
    active_domain: Optional[str] = None
    session_summary: str = ""
    agent_states: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class OrchestrationRun:
    run_id: str
    max_steps: int = 5
    steps: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    fallbacks: List[Dict[str, Any]] = field(default_factory=list)
    token_budget: Dict[str, int] = field(default_factory=lambda: {
        "evidence_chars": 6000,
        "response_chars": 9000,
        "max_charts": 4,
    })
    final_agent: Optional[str] = None

    def add_step(
        self,
        stage: str,
        label: str,
        agent: Optional[str] = None,
        status: str = "done",
        detail: str = "",
        **extra,
    ) -> None:
        if len(self.steps) >= self.max_steps + 3:
            return
        dur = extra.get("duration_ms")
        item = {
            "stage": stage,
            "label": label,
            "agent": agent,
            "status": status,
            "detail": detail[:500] if detail else "",
            "ts": round(time.time(), 3),
        }
        if dur is not None:
            item["duration_ms"] = dur
        item.update(extra)
        self.steps.append(item)
        dur_str = f" ({dur:.0f}ms)" if dur else ""
        logger.info(
            "[%s] STEP %s: %s%s",
            self.run_id[:8],
            stage,
            label,
            dur_str,
        )


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

        logger.info("=" * 60)
        logger.info("  %s", AGENT_NAMES['orchestrator'])
        logger.info("  Ministere du Tourisme du Maroc")
        logger.info("=" * 60)

        # ── Shared client ──
        self.client = get_shared_client()
        self.deployment = AZURE_OPENAI_DEPLOYMENT

        # ── Thread safety ──
        self._lock = threading.RLock()
        self._state_context = threading.local()
        self._conversation_locks: Dict[str, threading.RLock] = {}

        # ── Classification cache ──
        self._classify_cache = SearchCache(max_size=100, ttl_seconds=300)

        # ── Initialize agents ──
        logger.info("Initialisation des agents...")

        logger.info("  [1/3] Assistant General...")
        from agents.normal_agent import NormalAgent
        self.normal_agent = NormalAgent()
        logger.info("  [1/3] Assistant General pret")

        logger.info("  [2/3] Chercheur Tourisme...")
        from agents.researcher_agent import ResearcherAgent
        self.researcher_agent = ResearcherAgent()
        logger.info("  [2/3] Chercheur Tourisme pret")

        logger.info("  [3/3] Analyste de Donnees...")
        from agents.data_analytics_agent import DataAnalyticsAgent
        self.analytics_agent = DataAnalyticsAgent()
        logger.info("  [3/3] Analyste de Donnees pret")

        logger.info("  [4/5] Previsionniste...")
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
        logger.info("  [4/5] Previsionniste pret")

        logger.info("  [5/5] Analyste Executif...")
        from agents.executive_insight_agent import ExecutiveInsightAgent
        self.executive_agent = ExecutiveInsightAgent(
            analytics_agent=self.analytics_agent,
            researcher_agent=self.researcher_agent,
        )
        logger.info("  [5/5] Analyste Executif pret")
        from orchestration.nodes import build_default_registry
        self.node_registry = build_default_registry(self)

        # ── State ──
        self.last_agent: Optional[str] = None
        self.message_count: int = 0
        self.routing_history: List[Tuple[str, str, bool]] = []
        self.conversation_log: List[Tuple[str, str]] = []
        # Active domain tracked across turns for follow-up context propagation.
        # "hebergement" | "apf" | None (unknown/mixed)
        self._active_domain: Optional[str] = None
        self._runtime_states: Dict[str, ConversationRuntimeState] = {}

        # ── Session summary (ConversationSummaryBuffer for the classifier) ──
        # Compresses old exchanges so the routing LLM always gets compact context.
        self._session_summary: str = ""
        self._last_summarized_at: int = 0   # message_count when last summarized

        # ── Dataset year range ──
        self.min_year, self.max_year = self._get_dataset_year_range()

        # Year ranges for both tables
        self.year_ranges = self._get_all_year_ranges()
        self.apf_min_year = self.year_ranges["apf"][0]
        self.apf_max_year = self.year_ranges["apf"][1]
        self.hbg_min_year = self.year_ranges["hebergement"][0]
        self.hbg_max_year = self.year_ranges["hebergement"][1]

        # ── Ready ──
        logger.info("-" * 60)
        logger.info("Tous les agents sont operationnels.")
        logger.info("   Datasets: %s", len(self.analytics_agent.datasets))
        logger.info("   APF: %s-%s", self.apf_min_year, self.apf_max_year)
        logger.info("   Hebergement: %s-%s", self.hbg_min_year, self.hbg_max_year)
        rag_state = "prêt" if getattr(self.researcher_agent, "_rag_available", False) else "indisponible"
        web_state = "prêt" if getattr(self.researcher_agent, "_search_available", False) else "indisponible"
        logger.info("   RAG: %s | Web: %s", rag_state, web_state)
        logger.info("   Previsionniste: %s", "pret" if self.prediction_agent else "indisponible")
        logger.info("   Analyste Executif: pret")
        logger.info("-" * 60)

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

    def _thread_state(self) -> Optional[ConversationRuntimeState]:
        ctx = getattr(self, "_state_context", None)
        return getattr(ctx, "state", None) if ctx is not None else None

    def _push_thread_state(self, state: ConversationRuntimeState):
        ctx = getattr(self, "_state_context", None)
        if ctx is None:
            ctx = threading.local()
            self._state_context = ctx
        previous = getattr(ctx, "state", None)
        ctx.state = state
        return previous

    def _pop_thread_state(self, previous: Optional[ConversationRuntimeState]) -> None:
        ctx = getattr(self, "_state_context", None)
        if ctx is None:
            return
        if previous is None:
            try:
                delattr(ctx, "state")
            except AttributeError:
                pass
        else:
            ctx.state = previous

    def _get_conversation_lock(self, cid: str) -> threading.RLock:
        with self._lock:
            locks = getattr(self, "_conversation_locks", None)
            if locks is None:
                locks = {}
                self._conversation_locks = locks
            lock = locks.get(cid)
            if lock is None:
                lock = threading.RLock()
                locks[cid] = lock
            return lock

    @property
    def last_agent(self) -> Optional[str]:
        state = self._thread_state()
        if state is not None:
            return state.last_agent
        return getattr(self, "_global_last_agent", None)

    @last_agent.setter
    def last_agent(self, value: Optional[str]) -> None:
        state = self._thread_state()
        if state is not None:
            state.last_agent = value
        else:
            self.__dict__["_global_last_agent"] = value

    @property
    def message_count(self) -> int:
        state = self._thread_state()
        if state is not None:
            return state.message_count
        return getattr(self, "_global_message_count", 0)

    @message_count.setter
    def message_count(self, value: int) -> None:
        state = self._thread_state()
        if state is not None:
            state.message_count = value
        else:
            self.__dict__["_global_message_count"] = value

    @property
    def routing_history(self) -> List[Tuple[str, str, bool]]:
        state = self._thread_state()
        if state is not None:
            return state.routing_history
        if "_global_routing_history" not in self.__dict__:
            self.__dict__["_global_routing_history"] = []
        return self.__dict__["_global_routing_history"]

    @routing_history.setter
    def routing_history(self, value: List[Tuple[str, str, bool]]) -> None:
        state = self._thread_state()
        if state is not None:
            state.routing_history = value
        else:
            self.__dict__["_global_routing_history"] = value

    @property
    def conversation_log(self) -> List[Tuple[str, str]]:
        state = self._thread_state()
        if state is not None:
            return state.conversation_log
        if "_global_conversation_log" not in self.__dict__:
            self.__dict__["_global_conversation_log"] = []
        return self.__dict__["_global_conversation_log"]

    @conversation_log.setter
    def conversation_log(self, value: List[Tuple[str, str]]) -> None:
        state = self._thread_state()
        if state is not None:
            state.conversation_log = value
        else:
            self.__dict__["_global_conversation_log"] = value

    @property
    def _active_domain(self) -> Optional[str]:
        state = self._thread_state()
        if state is not None:
            return state.active_domain
        return getattr(self, "_global_active_domain", None)

    @_active_domain.setter
    def _active_domain(self, value: Optional[str]) -> None:
        state = self._thread_state()
        if state is not None:
            state.active_domain = value
        else:
            self.__dict__["_global_active_domain"] = value

    @property
    def _session_summary(self) -> str:
        state = self._thread_state()
        if state is not None:
            return state.session_summary
        return getattr(self, "_global_session_summary", "")

    @_session_summary.setter
    def _session_summary(self, value: str) -> None:
        state = self._thread_state()
        if state is not None:
            state.session_summary = value or ""
        else:
            self.__dict__["_global_session_summary"] = value or ""

    @property
    def _current_cid(self) -> str:
        state = self._thread_state()
        if state is not None:
            return state.conversation_id
        return getattr(self, "_global_current_cid", "_default")

    @_current_cid.setter
    def _current_cid(self, value: str) -> None:
        state = self._thread_state()
        if state is not None:
            state.conversation_id = value or "_default"
        else:
            self.__dict__["_global_current_cid"] = value or "_default"

    def _new_runtime_state(self, conversation_id: Optional[str]) -> ConversationRuntimeState:
        return ConversationRuntimeState(conversation_id=conversation_id or "_default")

    @staticmethod
    def _msg_get(message, key: str, default=None):
        if isinstance(message, dict):
            return message.get(key, default)
        return getattr(message, key, default)

    def _build_runtime_state_from_messages(
        self,
        conversation_id: str,
        messages: Optional[List] = None,
    ) -> ConversationRuntimeState:
        """Rehydrate routing and agent context from persisted UI messages."""
        state = self._new_runtime_state(conversation_id)
        messages = messages or []

        agent_prompts = {
            "normal": self.normal_agent.system_prompt,
            "researcher": self.researcher_agent.system_prompt,
            "analytics": self.analytics_agent.system_prompt,
        }
        agent_histories = {
            key: [{"role": "system", "content": prompt}]
            for key, prompt in agent_prompts.items()
        }

        last_user: Optional[str] = None
        domain_text_parts: List[str] = []

        for msg in messages:
            role = self._msg_get(msg, "role")
            content = self._msg_get(msg, "content", "") or ""
            if not content:
                continue

            if role == "user":
                last_user = content
                state.message_count += 1
                state.conversation_log.append(("user", content[:500]))
                domain_text_parts.append(content[:600])
                continue

            if role == "assistant":
                agent = self._msg_get(msg, "agent") or "normal"
                state.last_agent = agent if agent in {"normal", "researcher", "analytics", "prediction"} else "normal"
                state.conversation_log.append((state.last_agent, content[:500]))
                domain_text_parts.append(content[:600])

                if state.last_agent in agent_histories and last_user:
                    agent_histories[state.last_agent].append({"role": "user", "content": last_user})
                    agent_histories[state.last_agent].append({"role": "assistant", "content": content})

        if len(state.conversation_log) > 30:
            state.conversation_log = state.conversation_log[-30:]

        detected = _detect_domain(" ".join(domain_text_parts[-8:]))
        state.active_domain = detected
        state.agent_states = {
            key: {"conversation_history": history, "history_summary": ""}
            for key, history in agent_histories.items()
        }
        return state

    def load_conversation_state(
        self,
        conversation_id: Optional[str],
        messages: Optional[List] = None,
        force: bool = False,
    ) -> None:
        """Ensure runtime memory exists for a conversation.

        Called by the Flask conversation activation endpoint. It preserves an
        existing runtime state unless force=True, so switching conversations
        no longer erases follow-up context.
        """
        cid = conversation_id or "_default"
        with self._lock:
            if force or cid not in self._runtime_states:
                self._runtime_states[cid] = self._build_runtime_state_from_messages(cid, messages)

    def reset_conversation_state(self, conversation_id: Optional[str] = None) -> None:
        """Reset one conversation runtime state, or the default/global state."""
        cid = conversation_id or "_default"
        with self._lock:
            self._runtime_states[cid] = self._new_runtime_state(cid)
            if cid == "_default":
                self.last_agent = None
                self._active_domain = None
                self.conversation_log.clear()
                self.routing_history.clear()
                self.message_count = 0
                self._session_summary = ""
                self._classify_cache.clear()

    def clear_runtime_states(self) -> None:
        with self._lock:
            self._runtime_states.clear()
            self.reset_conversation_state("_default")

    def _load_runtime_attrs(self, state: ConversationRuntimeState) -> None:
        self.last_agent = state.last_agent
        self.message_count = state.message_count
        self.routing_history = list(state.routing_history)
        self.conversation_log = list(state.conversation_log)
        self._active_domain = state.active_domain
        self._session_summary = state.session_summary
        self._current_cid = state.conversation_id

    def _save_runtime_attrs(self, state: ConversationRuntimeState) -> None:
        state.last_agent = self.last_agent
        state.message_count = self.message_count
        state.routing_history = list(self.routing_history[-100:])
        state.conversation_log = list(self.conversation_log[-30:])
        state.active_domain = self._active_domain
        state.session_summary = self._session_summary

    def _compact_runtime_log(self) -> None:
        """Fold old router context into a compact summary."""
        keep = max(4, ORCHESTRATOR_RECENT_EXCHANGES * 2)
        if len(self.conversation_log) <= keep + 4:
            return

        to_compact = self.conversation_log[:-keep]
        recent = self.conversation_log[-keep:]
        lines = []
        for role, text in to_compact:
            clean = re.sub(r"```[\s\S]*?```", "[code]", text or "")
            lines.append(f"{role}: {clean[:350]}")

        prior = f"\nRésumé précédent:\n{self._session_summary}\n" if self._session_summary else ""
        prompt = (
            "Résume pour un routeur multi-agent STATOUR les échanges anciens ci-dessous.\n"
            "Garde uniquement les éléments utiles aux follow-ups: domaine actif, période, métrique, "
            "pays/région/voie/poste, dernières demandes, résultats clés, graphiques et questions ouvertes. "
            "Mentionne explicitement si le contexte concerne APF ou hébergement.\n"
            f"{prior}\nÉchanges anciens:\n" + "\n".join(lines) +
            "\n\nRésumé routeur en français, max 160 mots:"
        )

        try:
            kwargs = {
                "model": self.deployment,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": ORCHESTRATOR_SUMMARY_MAX_TOKENS,
            }
            response = self.client.chat.completions.create(**kwargs)
            if response.choices and response.choices[0].message.content:
                self._session_summary = response.choices[0].message.content.strip()
            else:
                raise RuntimeError("empty summary")
        except Exception as e:
            logger.debug("Router context compaction failed: %s", e)
            fallback = " | ".join(
                text[:120].replace("\n", " ")
                for role, text in to_compact
                if role == "user"
            )
            self._session_summary = (self._session_summary + " | " + fallback).strip(" |")[:1600]

        self.conversation_log = recent

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

    def _get_all_year_ranges(self) -> dict:
        """
        Extract year ranges for both fact tables from catalog metadata.
        Returns dict: {"apf": (min, max), "hebergement": (min, max)}
        """
        ranges = {"apf": (self.min_year, self.max_year), "hebergement": (None, None)}

        hbg_table = "fact_statistiqueshebergementnationaliteestimees"
        if hbg_table in self.analytics_agent.datasets:
            try:
                cols = self.analytics_agent.datasets[hbg_table].get("columns", [])
                date_col = next(
                    (c for c in cols if "date" in c["name"].lower()),
                    None
                )
                if date_col and "min" in date_col and "max" in date_col:
                    min_hbg = int(str(date_col["min"])[:4]) if date_col["min"] else None
                    max_hbg = int(str(date_col["max"])[:4]) if date_col["max"] else None
                    ranges["hebergement"] = (min_hbg, max_hbg)
                    logger.info(
                        "Hébergement year range: %s–%s", min_hbg, max_hbg
                    )
            except Exception as e:
                logger.debug("Could not extract hébergement year range: %s", e)

        return ranges

    def _years_in_message(self, message: str) -> List[int]:
        """Extract 4-digit years (1900-2099) from the message."""
        return [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", message)]

    def _all_years_outside_range(self, years: List[int]) -> bool:
        """Check if ALL mentioned years are outside the dataset range."""
        if not years or not self.min_year or not self.max_year:
            return False
        return all(y < self.min_year or y > self.max_year for y in years)

    def _build_conversation_context(self) -> str:
        """Build compact context for the LLM classifier."""
        if not self.conversation_log:
            return "Premier message — pas d'historique."

        recent = self.conversation_log[-max(4, ORCHESTRATOR_RECENT_EXCHANGES * 2):]
        lines = []
        if self._session_summary:
            lines.append(f"Résumé ancien: {self._session_summary[:1200]}")
        if self._active_domain:
            lines.append(f"Domaine actif: {self._active_domain}")
        for role, text in recent:
            short = text[:350].replace("\n", " ")
            prefix = "USER" if role == "user" else role.upper()
            lines.append(f"  {prefix}: {short}")
        return "Mémoire conversationnelle:\n" + "\n".join(lines)

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
        norm = _norm_text(user_message).strip()

        # ── Slash commands → ALWAYS analytics ──
        if msg.startswith("/"):
            return "analytics"

        # ── Pure greetings (exact match) → normal ──
        if msg in _PURE_GREETINGS:
            return "normal"

        # ── Compound greetings ("hello ca va", "bonjour comment ca va") → normal ──
        words = msg.split()
        if 2 <= len(words) <= 6 and words[0] in _PURE_GREETINGS:
            return "normal"

        # ── Explicit search/internet requests → ALWAYS researcher ──
        try:
            from orchestration.nodes import is_human_advisor_request

            if is_human_advisor_request(user_message):
                return "human_advisor"
        except Exception:
            logger.debug("Human advisor detector unavailable.", exc_info=True)

        if (
            re.search(r"\bdms\b", norm)
            or "duree moyenne de sejour" in norm
            or "arrivees hotelieres" in norm
            or "arrivee hoteliere" in norm
            or "nuitees" in norm
            or ("graphique" in norm and "arrive" in norm and "mois" in norm)
            or ("apf" in norm and "arrive" in norm)
        ):
            return "analytics"

        for phrase in _SEARCH_PHRASES:
            if phrase in msg:
                return "researcher"

        # NOTE: Chart/visualization keywords are NOT forced to analytics here.
        # Whether a chart uses internal or external data is determined by the LLM (Layer 2).
        # The two-phase researcher→analytics pipeline in _route_internal handles external charts.

        # Compound analysis + forecast requests need planning. Let the graph
        # decompose them instead of letting the first forecast keyword win.
        if _requires_planned_flow(user_message):
            return None

        # ── Prediction / forecasting keywords → prediction agent ──
        # Uses accent-normalized matching so "previsions" matches "prévisions"
        if _has_prediction_intent(msg, norm):
            return "prediction"

        # ── Follow-up / short conversational replies → same agent ──
        # Must run BEFORE years-outside-range check so that "Et pour 2028?"
        # after a prediction turn stays on prediction instead of being hijacked.
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

        # ── Years outside dataset range → researcher ──
        # Runs AFTER prediction keywords + follow-up pinning so prediction
        # questions with future years don't get hijacked to researcher.
        years = self._years_in_message(user_message)
        if years and self._all_years_outside_range(years):
            return "researcher"

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
        cid = getattr(self, "_current_cid", "_default")
        cache_key = f"{cid}:{self.last_agent or 'none'}:{user_message}"
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
        context = self._build_conversation_context()

        prompt = (
            CLASSIFIER_SYSTEM_PROMPT
            .replace("{min_year}", str(min_y))                                          # backward compat
            .replace("{max_year}", str(max_y))                                          # backward compat
            .replace("{apf_min_year}", str(self.apf_min_year or self.min_year or "?"))
            .replace("{apf_max_year}", str(self.apf_max_year or self.max_year or "?"))
            .replace("{hbg_min_year}", str(self.hbg_min_year or "?"))
            .replace("{hbg_max_year}", str(self.hbg_max_year or "?"))
            .replace("{last_agent}", str(self.last_agent or "AUCUN").upper())
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
        norm = _norm_text(user_message).strip()

        # Prediction keywords (check before analytics to avoid misrouting forecasts)
        if _has_prediction_intent(msg, norm):
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
                msg_lower = user_message.lower().strip()
                msg_norm = _norm_text(user_message).strip()
                if _has_prediction_intent(msg_lower, msg_norm):
                    logger.info(
                        "LLM said analytics but years %s are future + "
                        "prediction keywords detected — overriding to prediction",
                        years,
                    )
                    return "prediction"
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

    def _run_base_agent(
        self,
        runtime_state: Optional[ConversationRuntimeState],
        agent_key: str,
        call,
    ):
        """Run a BaseAgent subclass with per-conversation memory loaded."""
        agent = {
            "normal": self.normal_agent,
            "researcher": self.researcher_agent,
            "analytics": self.analytics_agent,
        }.get(agent_key)
        if agent is None or not hasattr(agent, "export_context_state"):
            return call()

        with agent._lock:
            if runtime_state is None:
                return call()
            previous_state = agent.export_context_state()
            agent.import_context_state(runtime_state.agent_states.get(agent_key))
            try:
                result = call()
                runtime_state.agent_states[agent_key] = agent.export_context_state()
                return result
            finally:
                agent.import_context_state(previous_state)

    @staticmethod
    def _requested_chart_count(user_message: str) -> int:
        norm = _norm_text(user_message)
        count = 0
        for raw in re.findall(r"\b([2-4])\s+(?:graph|chart|courbe|visual)", norm):
            count = max(count, int(raw))
        if any(k in norm for k in ["plusieurs graph", "multiple chart", "multiple graph", "des graphiques"]):
            count = max(count, 2)
        if any(k in norm for k in ["dashboard", "tableau de bord"]):
            count = max(count, 3)
        if any(k in norm for k in ["graph", "chart", "courbe", "visualis", "diagramme", "heatmap"]):
            count = max(count, 1)
        return min(count, 4)

    @staticmethod
    def _is_compound_request(user_message: str) -> bool:
        norm = _norm_text(user_message)
        connectors = [" et ", " plus ", " avec ", " ainsi que ", " puis ", "compare", "analyse"]
        deliverables = ["graph", "chart", "courbe", "tableau", "analyse", "recommand", "paragraphe"]
        return sum(1 for d in deliverables if d in norm) >= 2 or any(c in norm for c in connectors)

    def _missing_deliverables(
        self,
        user_message: str,
        response: str,
        chart_paths: List[str],
    ) -> List[str]:
        try:
            from orchestration.quality import inspect_deliverables

            return inspect_deliverables(user_message, response, chart_paths).missing[:3]
        except Exception:
            logger.debug("Typed quality gate unavailable; using local fallback.", exc_info=True)

        missing: List[str] = []
        requested_charts = self._requested_chart_count(user_message)
        if requested_charts and len(chart_paths) < requested_charts:
            missing.append(f"{requested_charts - len(chart_paths)} graphique(s) supplementaire(s)")
        if self._is_compound_request(user_message):
            heading_count = len(re.findall(r"(?m)^##\s+", response or ""))
            paragraph_count = len([p for p in re.split(r"\n\s*\n", response or "") if len(p.strip()) > 80])
            if heading_count < 2 and paragraph_count < 2:
                missing.append("analyse narrative structuree")
        return missing[:3]

    def _completion_pass(
        self,
        user_message: str,
        previous_response: str,
        missing: List[str],
        runtime_state: Optional[ConversationRuntimeState],
    ) -> Optional[str]:
        if not missing:
            return None
        prompt = (
            f"{user_message}\n\n"
            "[BOUCLE ORCHESTRATEUR - COMPLETION]\n"
            f"Elements manquants detectes: {', '.join(missing)}.\n"
            "Complete la reponse sans repeter inutilement ce qui est deja fait. "
            "Si des graphiques sont requis, utilise save_chart(fig, 'titre') et limite-toi a 4 graphiques au total. "
            "Si un graphique additionnel n'est pas pertinent, explique la limite et fournis un tableau.\n\n"
            f"Reponse precedente a completer:\n{previous_response[:2500]}"
        )
        return self._run_base_agent(
            runtime_state,
            "analytics",
            lambda: self.analytics_agent._chat_internal(
                prompt,
                domain_context=self._active_domain,
            ),
        )

    # ══════════════════════════════════════════════════════════════════════
    # ROUTE & EXECUTE — Thread-safe
    # ══════════════════════════════════════════════════════════════════════

    def run_node(
        self,
        node_key: str,
        user_message: str,
        conversation_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **metadata,
    ) -> Dict:
        """Run a typed orchestration node and return the legacy API payload."""
        from orchestration.contracts import NodeContext

        context = NodeContext(
            message=user_message,
            conversation_id=conversation_id,
            run_id=run_id,
            orchestrator=self,
            domain_context=self._active_domain,
            metadata=metadata,
        )
        return self.node_registry.run(context, preferred_key=node_key).to_legacy_dict()

    def route(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict:
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
        cid = conversation_id or "_default"
        route_lock = self._get_conversation_lock(cid)
        wait_start = time.time()
        with route_lock:
            queue_time_ms = (time.time() - wait_start) * 1000
            with self._lock:
                state = self._runtime_states.get(cid)
                if state is None:
                    state = self._new_runtime_state(cid)
                    self._runtime_states[cid] = state

            previous = self._push_thread_state(state)
            try:
                result = self._route_internal(user_message, runtime_state=state, run_id=run_id)
                self._compact_runtime_log()
                result["queue_time_ms"] = round(queue_time_ms, 1)
                return result
            finally:
                self._pop_thread_state(previous)

    def _route_internal(
        self,
        user_message: str,
        runtime_state: Optional[ConversationRuntimeState] = None,
        run_id: Optional[str] = None,
    ) -> Dict:
        """Internal routing logic."""

        self.message_count += 1
        start = time.time()

        # ══════════════════════════════════════════════════════════════════════
        # NEW FLOW: Plan-Execute-Review-Humanize pipeline
        # ══════════════════════════════════════════════════════════════════════
        try:
            from config.settings import USE_NEW_FLOW
            if USE_NEW_FLOW or _requires_planned_flow(user_message):
                return self._route_new_flow(user_message, runtime_state, run_id)
        except Exception as e:
            logger.warning("New flow failed, falling back to legacy: %s", str(e)[:200])

        # ══════════════════════════════════════════════════════════════════════
        # LEGACY FLOW (fallback)
        # ══════════════════════════════════════════════════════════════════════
        run = OrchestrationRun(run_id=run_id or uuid.uuid4().hex[:10])
        run.add_step(
            "routing",
            "Classification initiale",
            agent="orchestrator",
            detail="Selection du meilleur agent a partir du routeur et du contexte.",
        )

        # ── Classify ──
        agent_key = self.classify(user_message)
        # Executive questions combine internal KPIs + external context. Keep
        # explicit web-search requests on the researcher and forecasts on the
        # prediction agent.
        try:
            from agents.executive_insight_agent import is_executive_insight_request
            norm_msg = _norm_text(user_message)
            period = _period_metadata(user_message)
            explicit_search = any(p in norm_msg for p in _SEARCH_PHRASES)
            broad_research = any(
                k in norm_msg
                for k in [
                    "coupe du monde", "mondial", "monde", "international",
                    "actualite", "actualites", "tendance", "tendances",
                ]
            )
            market_specific_without_month = (
                not period.get("month")
                and any(k in norm_msg for k in ["britannique", "francais", "espagnol", "allemand", "italien"])
            )
            has_internal_metric = any(
                k in norm_msg
                for k in ["arrive", "touriste", "nuitee", "hotel", "hebergement", "apf", "mre", "tes"]
            )
            if (
                is_executive_insight_request(user_message)
                and agent_key not in ("normal", "prediction")
                and not explicit_search
                and period
                and has_internal_metric
                and not broad_research
                and not market_specific_without_month
            ):
                agent_key = "executive_insight"
        except Exception:
            pass
        classification_time = (time.time() - start) * 1000
        run.add_step(
            "agent_selected",
            "Agent selectionne",
            agent=agent_key,
            detail=f"Agent retenu: {agent_key}.",
            duration_ms=round(classification_time, 1),
        )

        agent_icon = ""
        agent_name = AGENT_NAMES.get(agent_key, "Agent")

        response = ""
        rerouted = False
        chart_path = None

        # ── Detect chart/visualization request ──
        _CHART_KW = ["chart", "graphe", "graphique", "visualis", "courbe",
                     "diagramme", "histogram", "heatmap", "plot", "trace", "affiche"]
        is_chart_request = any(kw in user_message.lower() for kw in _CHART_KW)

        # ── Execute ──
        try:
            if agent_key == "human_advisor":
                from orchestration.contracts import NodeContext

                node_result = self.node_registry.run(
                    NodeContext(
                        message=user_message,
                        conversation_id=runtime_state.conversation_id if runtime_state else None,
                        run_id=run.run_id,
                        orchestrator=self,
                        domain_context=self._active_domain,
                    ),
                    preferred_key="human_advisor",
                )
                response = node_result.response
                agent_icon = node_result.agent_icon
                agent_name = node_result.agent_name
                run.steps.extend(node_result.trace)

            elif agent_key == "analytics":
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

                response = self._run_base_agent(
                    runtime_state,
                    "analytics",
                    lambda: self.analytics_agent._chat_internal(
                        actual_message, domain_context=self._active_domain
                    ),
                )

                # Grab chart paths produced by the sandbox execution
                for _cp in getattr(self.analytics_agent, "last_chart_paths", []):
                    if _cp and _cp not in (chart_path,):
                        if not chart_path:
                            chart_path = _cp

                # Auto-reroute to researcher if analytics found no data in the dataset.
                # EXCEPTION: chart requests never reroute — researcher cannot render charts.
                # Analytics must produce the chart even with partial/limited data.
                if self._should_reroute_to_researcher(response) and not is_chart_request:
                    logger.info("Auto-rerouting to researcher (no data in dataset)")
                    response = self._run_base_agent(
                        runtime_state,
                        "researcher",
                        lambda: self.researcher_agent._chat_internal(user_message),
                    )
                    rerouted = True
                    agent_key = "researcher"
                    agent_icon = "🔍"
                    agent_name = "Chercheur Tourisme"

            elif agent_key == "executive_insight":
                try:
                    from utils.mvp_services import get_readiness
                    freshness = get_readiness(self).get("latest_data", {})
                except Exception:
                    freshness = {}
                executive_result = self.executive_agent._decline_precheck(user_message) or self.executive_agent.run(
                    user_message,
                    domain_context=self._active_domain,
                    data_freshness=freshness,
                )
                response = executive_result["response"]
                chart_path = executive_result.get("chart_path")
                agent_icon = executive_result.get("agent_icon", agent_icon)
                agent_name = executive_result.get("agent_name", "Analyste Exécutif")

            elif agent_key == "researcher":
                response = self._run_base_agent(
                    runtime_state,
                    "researcher",
                    lambda: self.researcher_agent._chat_internal(user_message),
                )

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
                    chart_response = self._run_base_agent(
                        runtime_state,
                        "analytics",
                        lambda: self.analytics_agent._chat_internal(
                            chart_msg, domain_context=self._active_domain
                        ),
                    )
                    chart_failed = bool(chart_response) and any(
                        marker in chart_response
                        for marker in [
                            "Code execution failed",
                            "SyntaxError",
                            "Traceback",
                            "invalid syntax",
                        ]
                    )
                    if chart_failed:
                        logger.warning(
                            "Two-phase chart rendering failed; keeping researcher response for deterministic fallback"
                        )
                    elif chart_response and "⚠️ No response" not in chart_response:
                        # Keep researcher info in response header, replace body with chart
                        response = chart_response
                        agent_key = "analytics"
                        agent_icon = "📊"
                        agent_name = "Analyste de Données"
                        rerouted = True
                        logger.info("Two-phase chart pipeline complete (researcher→analytics)")

            elif agent_key == "prediction":
                if self.prediction_agent:
                    # Inject recent conversation context so the prediction agent can
                    # resolve implicit references ("même voie", "idem", "pour 2028 aussi").
                    # Mirror the researcher→analytics injection pattern (lines 893–921).
                    actual_pred_message = user_message
                    if self.conversation_log:
                        recent = self.conversation_log[-max(4, ORCHESTRATOR_RECENT_EXCHANGES * 2):]
                        ctx_lines = []
                        for role, text in recent:
                            short = text[:350].replace("\n", " ")
                            prefix = "USER" if role == "user" else role.upper()
                            ctx_lines.append(f"  {prefix}: {short}")
                        if self._session_summary:
                            ctx_lines.insert(0, f"  RÉSUMÉ: {self._session_summary[:1000]}")
                        ctx_block = "\n".join(ctx_lines)
                        actual_pred_message = (
                            f"{user_message}\n\n"
                            f"[Historique récent pour résoudre les références implicites:\n"
                            f"{ctx_block}]"
                        )
                    pred_result = self.prediction_agent.chat(actual_pred_message)
                    response = pred_result["response"]
                    chart_path = pred_result.get("chart_path")
                else:
                    # Fallback: route to analytics if prediction agent unavailable
                    logger.warning("PredictionAgent unavailable — routing to analytics")
                    response = self._run_base_agent(
                        runtime_state,
                        "analytics",
                        lambda: self.analytics_agent._chat_internal(
                            user_message, domain_context=self._active_domain
                        ),
                    )
                    agent_key = "analytics"
                    agent_icon = "📊"
                    agent_name = "Analyste de Données"

            else:
                response = self._run_base_agent(
                    runtime_state,
                    "normal",
                    lambda: self.normal_agent._chat_internal(user_message),
                )

        except Exception as e:
            response = f"❌ Erreur: {str(e)}"
            logger.error(
                "Agent %s failed for '%s': %s",
                agent_key, user_message[:50], e,
                exc_info=True,
            )

        current_chart_paths = []
        if chart_path:
            current_chart_paths.append(chart_path)
        # Pick up chart paths stored by the analytics agent's sandbox
        for _cp in getattr(self.analytics_agent, "last_chart_paths", []):
            if _cp and os.path.exists(_cp) and _cp not in current_chart_paths:
                current_chart_paths.append(_cp)
        # Fallback: try regex extraction from response text
        current_chart_paths.extend(
            p for p in _chart_paths_from_response(response)
            if p not in current_chart_paths
        )
        run.add_step(
            "agent",
            "Execution agent",
            agent=agent_key,
            detail="Premier passage termine.",
            artifact_count=len(current_chart_paths),
        )

        missing = self._missing_deliverables(user_message, response, current_chart_paths)
        if missing and len(run.steps) < run.max_steps:
            run.add_step(
                "review",
                "Controle de completude",
                agent="orchestrator",
                detail="Elements manquants: " + ", ".join(missing),
            )
            try:
                completion = self._completion_pass(
                    user_message,
                    response,
                    missing,
                    runtime_state,
                )
                if completion:
                    response = response.rstrip() + "\n\n## Complements d'analyse\n" + completion.strip()
                    agent_key = "analytics"
                    agent_icon = "📊"
                    agent_name = "Analyste de Donnees"
                    rerouted = True
                    run.fallbacks.append({
                        "stage": "completion",
                        "agent": "analytics",
                        "reason": ", ".join(missing),
                    })
                    current_chart_paths.extend(
                        p for p in _chart_paths_from_response(completion)
                        if p not in current_chart_paths
                    )
                    if current_chart_paths and not chart_path:
                        chart_path = current_chart_paths[0]
                    run.add_step(
                        "completion",
                        "Passage complementaire",
                        agent="analytics",
                        detail="Analytics a complete les livrables manquants.",
                        artifact_count=len(current_chart_paths),
                    )
            except Exception as e:
                run.errors.append({
                    "stage": "completion",
                    "agent": "analytics",
                    "error": type(e).__name__,
                    "message": str(e)[:300],
                })
                run.add_step(
                    "completion",
                    "Passage complementaire",
                    agent="analytics",
                    status="error",
                    detail=str(e)[:300],
                )

        total_time = (time.time() - start) * 1000

        # ── Update domain context ──
        # Detect domain from the combined user message + response so follow-ups
        # ("et en 2025", "et pour 2024?") inherit the right table context.
        # A newly detected domain always wins; if ambiguous (None) we keep the
        # current domain so short follow-ups don't reset it to unknown.
        detected = _detect_domain(user_message + " " + response[:600])
        if detected is not None:
            if detected != self._active_domain:
                logger.debug(
                    "Domain context: %s → %s", self._active_domain, detected
                )
            self._active_domain = detected
        # else: keep current domain — ambiguous turn, likely a follow-up

        # ── Update state ──
        self.last_agent = agent_key

        self.conversation_log.append(("user", user_message))
        self.conversation_log.append((agent_key, response[:800]))

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

        if not chart_path:
            chart_path = _chart_path_from_response(response)
        chart_paths = list(current_chart_paths)
        if chart_path and chart_path not in chart_paths:
            chart_paths.insert(0, chart_path)
        chart_paths.extend(
            p for p in _chart_paths_from_response(response)
            if p not in chart_paths
        )
        run.final_agent = agent_key
        metric_context = _metric_context_metadata(user_message, agent_key)
        period_meta = _period_metadata(user_message)
        if metric_context == "apf":
            data_scope_note = "APF/DGSN, arrivees aux postes frontieres"
        elif metric_context == "hebergement":
            data_scope_note = "hebergement classe EHTC/STDN + estimatif"
        elif metric_context == "prediction":
            data_scope_note = "projection non officielle basee sur historiques APF et hypotheses explicites"
        elif metric_context == "research":
            data_scope_note = "veille externe/RAG, a distinguer des donnees Fabric officielles"
        else:
            data_scope_note = ""

        return {
            "agent": agent_key,
            "agent_icon": agent_icon,
            "agent_name": agent_name,
            "response": response,
            "chart_path": chart_path,
            "chart_paths": chart_paths[: run.token_budget["max_charts"]],
            "sources": executive_result.get("sources", []) if 'executive_result' in locals() else [],
            "confidence": executive_result.get("confidence") if 'executive_result' in locals() else None,
            "data_freshness": executive_result.get("data_freshness", {}) if 'executive_result' in locals() else {},
            "rerouted": rerouted,
            "classification_time_ms": round(classification_time, 1),
            "total_time_ms": round(total_time, 1),
            "metric_context": metric_context,
            "period": period_meta,
            "data_scope_note": data_scope_note,
            "run_id": run.run_id,
            "trace": run.steps,
            "fallbacks": run.fallbacks,
            "errors": run.errors,
            "token_budget": run.token_budget,
        }

    def chat(self, user_message: str) -> str:
        """Simple chat interface — returns just the response text."""
        result = self.route(user_message)
        return result["response"]

    # ══════════════════════════════════════════════════════════════════════
    # NEW FLOW — Plan-Execute-Review-Humanize
    # ══════════════════════════════════════════════════════════════════════

    def _route_new_flow(
        self,
        user_message: str,
        runtime_state: Optional[ConversationRuntimeState] = None,
        run_id: Optional[str] = None,
    ) -> Dict:
        """Route using the new Plan-Execute-Review-Humanize pipeline."""
        from orchestration.graph import run_graph

        # Gather conversation context for the graph
        conversation_context = self._build_conversation_context()

        # Get tool instances from existing agents
        db_layer = getattr(self.analytics_agent, "_db", None)
        search_tool = getattr(self.researcher_agent, "searcher", None)
        rag_manager = getattr(self.researcher_agent, "rag", None)

        result = run_graph(
            message=user_message,
            db_layer=db_layer,
            search_tool=search_tool,
            rag_manager=rag_manager,
            prediction_engine=self.prediction_agent,
            conversation_context=conversation_context,
            domain_context=self._active_domain,
            run_id=run_id,
        )

        # Update orchestrator state (same as legacy flow)
        agent_key = result.get("agent", "normal")
        response = result.get("response", "")

        self.last_agent = agent_key
        self.conversation_log.append(("user", user_message))
        self.conversation_log.append((agent_key, response[:800]))
        if len(self.conversation_log) > 30:
            self.conversation_log = self.conversation_log[-30:]
        self.routing_history.append((user_message, agent_key, False))
        if len(self.routing_history) > 100:
            self.routing_history = self.routing_history[-100:]

        # Update domain context from response
        detected = _detect_domain(user_message + " " + response[:600])
        if detected is not None:
            self._active_domain = detected

        return result

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
            ("📌", "Analyste Exécutif", "executive_insight", self.executive_agent),
        ]

        for icon, name, key, agent in agents_info:
            active = "✅" if ACTIVE_AGENTS.get(key) else "❌"
            msgs = agent.get_conversation_length() if hasattr(agent, "get_conversation_length") else 0
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
        self._active_domain = None
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
