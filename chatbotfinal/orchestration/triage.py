"""
Triage Node — Complexity Assessment & Intent Extraction
=========================================================
First node in the Plan-Execute-Review pipeline.
Decides whether a message is simple (direct answer) or complex (needs planning).

Simple: greetings, thanks, single-fact KPI lookups, platform Q&A
Complex: analytics, multi-step comparisons, predictions, research, compound requests
"""

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from utils.base_agent import get_shared_client, _prepare_messages, _should_skip_reasoning_effort
from utils.logger import get_logger
from config.settings import (
    AZURE_OPENAI_DEPLOYMENT,
    CLASSIFIER_MAX_TOKENS,
    CLASSIFIER_REASONING_EFFORT,
    MODEL_IS_REASONING,
)
from .followup import is_short_followup, context_has_data_turn, latest_data_agent, is_clear_social_turn

logger = get_logger("statour.orchestration.triage")

_DATA_MAX_YEAR = 2026


def _norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Triage Result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TriageResult:
    """Output of the triage node."""
    complexity: str  # "simple" | "complex"
    intent: str  # short description of what the user wants
    tools_needed: List[str] = field(default_factory=list)  # e.g. ["sql", "web_search", "chart"]
    direct_answer: Optional[str] = None  # for simple cases, the answer itself
    confidence: float = 0.9
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic Fast Paths (no LLM needed)
# ══════════════════════════════════════════════════════════════════════════════

_GREETING_PATTERNS = re.compile(
    r"^(bonjour|salut|hello|hi|hey|salam|bonsoir|coucou|"
    r"good\s*morning|good\s*evening|good\s*afternoon|"
    r"مرحبا|السلام عليكم|صباح الخير)[\s!?.]*$",
    re.IGNORECASE,
)

_THANKS_PATTERNS = re.compile(
    r"^(merci|thanks|thank\s*you|شكرا|shukran|"
    r"merci\s*beaucoup|thanks\s*a\s*lot|parfait|super|"
    r"c'?est\s*bon|ok\s*merci|génial|excellent)[\s!?.]*$",
    re.IGNORECASE,
)

_FAREWELL_PATTERNS = re.compile(
    r"^(au\s*revoir|bye|goodbye|à\s*bientôt|bonne\s*journée|"
    r"bonne\s*soirée|ciao|مع السلامة)[\s!?.]*$",
    re.IGNORECASE,
)

_PLATFORM_QA_PATTERNS = re.compile(
    r"(qu.?est.?ce\s*que\s*statour|c.?est\s*quoi\s*statour|"
    r"comment\s*(?:ça\s*)?(?:fonctionne|marche)|"
    r"qui\s*es.tu|what\s*(?:is|are)\s*you|"
    r"aide|help|ماهو)",
    re.IGNORECASE,
)

_COMPLEXITY_MARKERS = re.compile(
    r"(compar|évolution|tendance|trend|analyse|graphique|chart|"
    r"prévision|predict|forecast|projection|"
    r"pourquoi|why|cause|facteur|"
    r"corrélation|impact|effet|"
    r"rapport|report|détail|approfondi|"
    r"par\s*(?:mois|ville|région|pays|année|continent)|"
    r"top\s*\d+|classement|ranking|"
    r"et\s*(?:aussi|également)|puis\s*(?:aussi|montre)|"
    r"plusieurs|multiple|"
    r"en\s*plus|ajoute|"
    r"visualis|montre.moi|affiche)",
    re.IGNORECASE,
)

_ANALYTICS_MARKERS = re.compile(
    r"(nuitées|arrivées|touristes?|tourisme|touristique|mre|tes|apf|"
    r"hébergement|hôtel|hotel|occupation|"
    r"recettes|dms|durée\s*moyenne|"
    r"combien|quel\s*(?:est|nombre|total)|"
    r"données|statistiques|chiffres|stats|"
    r"ehtc)",
    re.IGNORECASE,
)

_SEARCH_MARKERS = re.compile(
    r"(actualit|news|récent|dernier|stratégi|vision\s*2030|"
    r"cherche|search|recherche|article|"
    r"mondial|international|unwto|benchmark|"
    r"événement|event|conférence)",
    re.IGNORECASE,
)

_PREDICTION_MARKERS = re.compile(
    r"(prévi|prédic|forecast|projection|futur|prochain|"
    r"scénario|scenario|optimist|pessimist|"
    r"va\s*(?:augmenter|baisser|évoluer)|"
    r"combien\s*(?:en|pour)\s*20(?:2[7-9]|3))",
    re.IGNORECASE,
)

_CHART_MARKERS = re.compile(
    r"(graphique|chart|graph|courbe|diagramme|"
    r"visualis|montre.moi|affiche|histogramme|"
    r"camembert|pie|bar\s*chart|line\s*chart)",
    re.IGNORECASE,
)

_COMPLEXITY_MARKERS_NORM = re.compile(
    r"(compar|evolution|tendance|trend|analyse|graphique|chart|"
    r"prevision|prevoir|prediction|predict|forecast|projection|"
    r"pourquoi|why|cause|facteur|correlation|impact|effet|"
    r"rapport|report|detail|approfondi|historique|depuis|jusqu|"
    r"par\s*(?:mois|ville|region|pays|annee|continent)|"
    r"top\s*\d+|classement|ranking|puis|plusieurs|multiple|visualis|affiche)",
    re.IGNORECASE,
)

_ANALYTICS_MARKERS_NORM = re.compile(
    r"(nuitees|arrivees|touristes?|tourisme|touristiqu|mre|tes|apf|"
    r"hebergement|hotel|occupation|recettes|dms|duree\s*moyenne|"
    r"combien|quel\s*(?:est|nombre|total)|donnees|statistiques|"
    r"chiffres|stats|ehtc)",
    re.IGNORECASE,
)

_SEARCH_MARKERS_NORM = re.compile(
    r"(actualit|news|recent|dernier|strategi|vision\s*2030|"
    r"cherche|search|recherche|article|mondial|international|"
    r"unwto|benchmark|evenement|event|conference)",
    re.IGNORECASE,
)

_PREDICTION_MARKERS_NORM = re.compile(
    r"(previ|prevoir|prevois|predic|prediction|forecast|projection|"
    r"futur|prochain|scenario|optimist|pessimist|"
    r"va\s*(?:augmenter|baisser|evoluer)|"
    r"combien\s*(?:en|pour)\s*20(?:2[6-9]|3))",
    re.IGNORECASE,
)

_CHART_MARKERS_NORM = re.compile(
    r"(graphique|chart|graph|courbe|diagramme|visualis|"
    r"montre.moi|affiche|histogramme|camembert|pie|bar\s*chart|line\s*chart)",
    re.IGNORECASE,
)

_FORECAST_CONTEXT_NORM = re.compile(
    r"(previ|prevoir|prevois|predic|prediction|forecast|projection|"
    r"scenario|optimist|pessimist|baseline|de base)",
    re.IGNORECASE,
)

_FORECAST_QUESTION_NORM = re.compile(
    r"(combien|quel|quelle|estime|estimation|attend|attendu|sera|seront|"
    r"pourrait|pourraient|prevoi|prevoir|projette|projection|scenario)",
    re.IGNORECASE,
)


def _years_in_text(text: str) -> List[int]:
    return [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")]


def _has_prediction_request(message: str, conversation_context: str = "") -> bool:
    """Return True only when the current turn genuinely asks for forecasting.

    LLM triage/planning can be over-eager and add the prediction tool to plain
    historical analyses. This guard keeps prediction available for explicit
    forecasts and short follow-ups to a forecast, without special-casing any
    single prompt.
    """
    if _unsupported_prediction_target(message):
        return False

    norm = _norm_text(message)

    if _PREDICTION_MARKERS.search(message) or _PREDICTION_MARKERS_NORM.search(norm):
        return True

    years = _years_in_text(message)
    if any(year > _DATA_MAX_YEAR for year in years) and _FORECAST_QUESTION_NORM.search(norm):
        return True

    ctx_norm = _norm_text(conversation_context)
    is_short_followup = len(norm.split()) <= 8
    references_forecast_context = bool(ctx_norm and _FORECAST_CONTEXT_NORM.search(ctx_norm))
    has_followup_target = bool(
        years
        or re.search(r"\b(et\s+pour|pour|meme|pareil|optimiste|pessimiste|scenario|base|baseline)\b", norm)
    )
    return bool(is_short_followup and references_forecast_context and has_followup_target)


def _has_chart_request(message: str) -> bool:
    """Return True when the user explicitly asks for a visualization."""
    norm = _norm_text(message)
    return bool(_CHART_MARKERS.search(message) or _CHART_MARKERS_NORM.search(norm))


_SPACE_OR_OUT_OF_SCOPE_TARGETS = {
    "mars", "lune", "jupiter", "venus", "saturne", "mercure", "neptune", "uranus",
}


def _unsupported_prediction_target(message: str) -> bool:
    """Detect forecast prompts aimed at entities outside STATOUR tourism data."""
    norm = _norm_text(message)
    target = r"(?:mars|lune|jupiter|venus|saturne|mercure|neptune|uranus)"
    if re.search(rf"\b(?:sur|vers|a|au|aux)\s+{target}\b", norm):
        return True
    if re.search(rf"\bvisit\w*|visite\w*", norm):
        if re.search(rf"\b(?:visit\w*|visite\w*)\b.{{0,40}}\b{target}\b", norm):
            return True
    if re.search(rf"\b{target}\b", norm) and re.search(r"\bplanete|spatial|extra[-\s]?terrestre\b", norm):
        return True
    return False


def _contextual_followup_triage(message: str, conversation_context: str) -> Optional[TriageResult]:
    if not conversation_context or not is_short_followup(message) or is_clear_social_turn(message):
        return None
    if not context_has_data_turn(conversation_context):
        return None

    agent = latest_data_agent(conversation_context)
    if agent == "prediction":
        return TriageResult(
            complexity="complex",
            intent="prediction_followup",
            tools_needed=["prediction"],
            confidence=0.95,
        )

    tools = ["sql"]
    if _has_chart_request(message):
        tools.append("chart")
    return TriageResult(
        complexity="complex",
        intent="analytics_followup",
        tools_needed=tools,
        confidence=0.95,
    )


def _obvious_tool_triage(message: str, conversation_context: str) -> Optional[TriageResult]:
    norm = _norm_text(message)
    tools: List[str] = []
    if _ANALYTICS_MARKERS.search(message) or _ANALYTICS_MARKERS_NORM.search(norm):
        tools.append("sql")
    if _SEARCH_MARKERS.search(message) or _SEARCH_MARKERS_NORM.search(norm):
        tools.append("web_search")
    if _has_prediction_request(message, conversation_context):
        tools.append("prediction")
    if _has_chart_request(message):
        tools.append("chart")

    tools = _sanitize_tools(message, conversation_context, tools)
    if not tools:
        return None
    return TriageResult(
        complexity="complex",
        intent="tool_required",
        tools_needed=tools,
        confidence=0.9,
    )


def _sanitize_tools(
    message: str,
    conversation_context: str,
    tools: List[str],
) -> List[str]:
    """Normalize LLM-suggested tools and suppress unsupported prediction steps."""
    cleaned: List[str] = []
    prediction_allowed = _has_prediction_request(message, conversation_context)
    chart_allowed = _has_chart_request(message)

    for tool in tools:
        if tool == "prediction" and not prediction_allowed:
            logger.info("Triage suppressed prediction tool: no forecast intent in current turn")
            continue
        if tool == "chart" and not chart_allowed:
            logger.info("Triage suppressed chart tool: no chart intent in current turn")
            continue
        if tool not in cleaned:
            cleaned.append(tool)

    return cleaned


def _deterministic_triage(message: str) -> Optional[TriageResult]:
    """Instant triage for obvious cases. Returns None if LLM needed."""
    text = message.strip()

    if _GREETING_PATTERNS.match(text):
        return TriageResult(
            complexity="simple",
            intent="greeting",
            direct_answer=None,
            confidence=1.0,
        )

    if _THANKS_PATTERNS.match(text):
        return TriageResult(
            complexity="simple",
            intent="thanks",
            direct_answer=None,
            confidence=1.0,
        )

    if _FAREWELL_PATTERNS.match(text):
        return TriageResult(
            complexity="simple",
            intent="farewell",
            direct_answer=None,
            confidence=1.0,
        )

    if _PLATFORM_QA_PATTERNS.search(text) and len(text) < 80:
        return TriageResult(
            complexity="simple",
            intent="platform_qa",
            direct_answer=None,
            confidence=0.95,
        )

    return None


# ══════════════════════════════════════════════════════════════════════════════
# LLM Triage (for ambiguous cases)
# ══════════════════════════════════════════════════════════════════════════════

_TRIAGE_PROMPT = """Tu es le module de triage de STATOUR. Tu évalues la complexité d'une question utilisateur.

RÈGLES DE CLASSIFICATION:

SIMPLE (réponse directe, pas de planification nécessaire):
- Salutations, remerciements, au revoir
- Questions sur la plateforme STATOUR (qu'est-ce que c'est, comment ça marche)
- Questions à réponse unique et directe (un seul chiffre, un seul fait)
- Conversation générale sans besoin de données

COMPLEX (nécessite planification et exécution multi-étapes):
- Toute demande nécessitant des données SQL (statistiques, chiffres, tendances)
- Comparaisons entre périodes, villes, pays, catégories
- Demandes de graphiques ou visualisations
- Prévisions ou projections
- Recherche web (actualités, contexte, stratégie)
- Questions composées (plusieurs sous-questions)
- Analyse causale (pourquoi, facteurs, impacts)
- Demandes de rapport ou analyse approfondie

OUTILS DISPONIBLES:
- sql: requêtes sur les tables Fabric (APF frontières, hébergement)
- web_search: recherche web (actualités, contexte international)
- rag: base de connaissances STATOUR (documents internes)
- prediction: moteur de prévision (projections futures)
- chart: génération de graphiques Plotly

RÉPONDS EN JSON STRICT (pas de texte avant/après):
{"complexity": "simple|complex", "intent": "description courte", "tools_needed": ["tool1", "tool2"]}"""


def _llm_triage(message: str, conversation_context: str = "") -> TriageResult:
    """Use LLM to assess complexity when deterministic rules are insufficient."""
    import json

    client = get_shared_client()
    start = time.time()

    user_content = message
    if conversation_context:
        user_content = f"[Contexte conversation: {conversation_context}]\n\nMessage: {message}"

    messages = [
        {"role": "system", "content": _TRIAGE_PROMPT},
        {"role": "user", "content": user_content},
    ]

    kwargs: Dict[str, Any] = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "messages": _prepare_messages(messages),
        "max_completion_tokens": CLASSIFIER_MAX_TOKENS,
    }
    if MODEL_IS_REASONING and not _should_skip_reasoning_effort():
        kwargs["reasoning_effort"] = CLASSIFIER_REASONING_EFFORT

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip() if response.choices else ""

        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            parsed = json.loads(json_match.group())
            complexity = parsed.get("complexity", "complex")
            intent = parsed.get("intent", "unknown")
            tools = parsed.get("tools_needed", [])

            if complexity not in ("simple", "complex"):
                complexity = "complex"

            valid_tools = {"sql", "web_search", "rag", "prediction", "chart"}
            tools = [t for t in tools if t in valid_tools]
            tools = _sanitize_tools(message, conversation_context, tools)

            return TriageResult(
                complexity=complexity,
                intent=intent,
                tools_needed=tools,
                confidence=0.85,
                duration_ms=(time.time() - start) * 1000,
            )

    except Exception as e:
        logger.warning("Triage LLM failed: %s — falling back to heuristic", str(e)[:100])

    return _heuristic_triage(message)


def _heuristic_triage(message: str) -> TriageResult:
    """Fallback heuristic when LLM triage fails."""
    tools = []
    norm = _norm_text(message)

    if _ANALYTICS_MARKERS.search(message) or _ANALYTICS_MARKERS_NORM.search(norm):
        tools.append("sql")
    if _SEARCH_MARKERS.search(message) or _SEARCH_MARKERS_NORM.search(norm):
        tools.append("web_search")
    if _PREDICTION_MARKERS.search(message) or _PREDICTION_MARKERS_NORM.search(norm):
        tools.append("prediction")
    if _CHART_MARKERS.search(message) or _CHART_MARKERS_NORM.search(norm):
        tools.append("chart")

    tools = _sanitize_tools(message, "", tools)

    years = _years_in_text(message)
    has_historical_analysis = (
        any(k in norm for k in ["analyse", "evolution", "tendance", "historique", "depuis", "jusqu"])
        and any(y <= _DATA_MAX_YEAR for y in years)
        and (
            len(set(years)) >= 2
            or any(k in norm for k in ["tourisme", "touriste", "arrive", "nuitee", "recette", "hebergement", "apf"])
        )
    )
    if has_historical_analysis and "sql" not in tools:
        tools.append("sql")

    has_complexity = bool(_COMPLEXITY_MARKERS.search(message) or _COMPLEXITY_MARKERS_NORM.search(norm))

    if not tools and not has_complexity and len(message.split()) < 8:
        return TriageResult(
            complexity="simple",
            intent="general_question",
            tools_needed=[],
            confidence=0.6,
        )

    if not tools:
        tools = ["sql"]

    return TriageResult(
        complexity="complex",
        intent="analytics_request" if "sql" in tools else "research_request",
        tools_needed=tools,
        confidence=0.7,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def triage(message: str, conversation_context: str = "") -> TriageResult:
    """
    Assess complexity of a user message and determine required tools.

    Returns a TriageResult with:
    - complexity: "simple" or "complex"
    - intent: short description
    - tools_needed: list of tools for the executor
    """
    start = time.time()

    if _unsupported_prediction_target(message):
        return TriageResult(
            complexity="simple",
            intent="out_of_domain_clarification",
            direct_answer=(
                "Je ne peux pas produire une prevision touristique fiable pour cette cible, "
                "car elle sort du perimetre des donnees STATOUR. Si vous parlez du mois de "
                "mars, precisez la metrique touristique souhaitee (APF, MRE/TES, nuitees, "
                "region ou pays de residence) et je lancerai l'analyse adaptee."
            ),
            confidence=1.0,
            duration_ms=(time.time() - start) * 1000,
            metadata={"blocked_prediction": True},
        )

    # Layer 1: deterministic (instant, 0ms)
    result = _deterministic_triage(message)
    if result:
        result.duration_ms = (time.time() - start) * 1000
        logger.debug("Triage[deterministic]: %s → %s", message[:50], result.complexity)
        return result

    result = _contextual_followup_triage(message, conversation_context)
    if result:
        result.duration_ms = (time.time() - start) * 1000
        logger.debug("Triage[context-followup]: %s -> %s", message[:50], result.complexity)
        return result

    result = _obvious_tool_triage(message, conversation_context)
    if result:
        result.duration_ms = (time.time() - start) * 1000
        logger.debug("Triage[tool]: %s -> %s", message[:50], result.tools_needed)
        return result

    # Layer 2: LLM-based (fast, ~200ms with minimal reasoning)
    result = _llm_triage(message, conversation_context)
    result.duration_ms = (time.time() - start) * 1000
    logger.debug(
        "Triage[llm]: %s → %s (tools=%s, %.0fms)",
        message[:50], result.complexity, result.tools_needed, result.duration_ms,
    )
    return result
