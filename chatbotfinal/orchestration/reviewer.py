"""
Reviewer Node — Reflection & Gap Detection
============================================
After execution, evaluates whether the collected evidence
adequately answers the user's question.

If gaps are found, provides structured feedback for re-planning.
If sufficient, passes evidence to the Synthesizer/Humanizer.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from utils.base_agent import get_shared_client, _prepare_messages, _should_skip_reasoning_effort
from utils.logger import get_logger
from config.settings import (
    AZURE_OPENAI_DEPLOYMENT,
    MODEL_IS_REASONING,
    CLASSIFIER_REASONING_EFFORT,
)
from .executor import ExecutionResult

logger = get_logger("statour.orchestration.reviewer")


# ══════════════════════════════════════════════════════════════════════════════
# Review Result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReviewResult:
    """Output of the reviewer node."""
    verdict: str  # "sufficient" | "gaps_found" | "error"
    confidence: float = 0.8
    gaps: List[str] = field(default_factory=list)
    feedback: str = ""  # guidance for re-planning
    suggested_tools: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Reviewer Prompt
# ══════════════════════════════════════════════════════════════════════════════

_REVIEWER_PROMPT = """Tu es le contrôleur qualité de STATOUR. Tu évalues si les résultats d'analyse répondent adéquatement à la question posée.

CRITÈRES D'ÉVALUATION:
1. COMPLÉTUDE: La question est-elle entièrement adressée? Manque-t-il des aspects?
2. COHÉRENCE: Les données sont-elles cohérentes entre elles? Pas de contradictions?
3. CLARTÉ: Les résultats sont-ils suffisamment clairs pour être présentés?
4. GRAPHIQUE: Si un graphique était demandé, est-il présent?

RÈGLES MÉTIER:
- Si "arrivées" mentionné sans contexte → vérifier que BOTH APF et hébergement sont couverts
- Si comparaison demandée → vérifier que les 2+ entités sont bien comparées
- Si évolution temporelle → vérifier que la période est bien couverte
- Si prévision → vérifier qu'un horizon est donné avec scénarios

RÉPONDS EN JSON STRICT:
{"verdict": "sufficient|gaps_found", "confidence": 0.0-1.0, "gaps": ["gap1", "gap2"], "feedback": "instructions pour combler les lacunes", "suggested_tools": ["tool1"]}

SI SUFFISANT: {"verdict": "sufficient", "confidence": 0.9, "gaps": [], "feedback": "", "suggested_tools": []}"""


# ══════════════════════════════════════════════════════════════════════════════
# Heuristic Pre-Check (no LLM, instant)
# ══════════════════════════════════════════════════════════════════════════════

def _heuristic_review(message: str, execution: ExecutionResult) -> Optional[ReviewResult]:
    """
    Fast heuristic checks before calling LLM reviewer.
    Returns a result if obvious issues are found, None if LLM review needed.
    """
    msg_lower = message.lower()

    # If all steps failed, definitely gaps
    if not any(ev.success for ev in execution.evidence):
        return ReviewResult(
            verdict="gaps_found",
            confidence=1.0,
            gaps=["Toutes les étapes d'exécution ont échoué"],
            feedback="Re-tenter avec des requêtes plus simples ou un outil alternatif.",
            suggested_tools=["sql"],
        )

    # Check for chart request vs chart presence
    chart_keywords = ("graphique", "chart", "courbe", "visualis", "diagramme", "graph")
    chart_requested = any(kw in msg_lower for kw in chart_keywords)
    has_charts = bool(execution.chart_paths)

    if chart_requested and not has_charts:
        successful_data = any(
            ev.success and ev.tool in ("sql", "prediction") and ev.text_summary
            for ev in execution.evidence
        )
        if successful_data:
            return ReviewResult(
                verdict="gaps_found",
                confidence=0.9,
                gaps=["Graphique demandé mais non généré"],
                feedback="Ajouter une étape chart pour visualiser les données obtenues.",
                suggested_tools=["chart"],
            )

    # Check for very short/empty results when data was expected
    data_steps = [ev for ev in execution.evidence if ev.tool in ("sql", "prediction") and ev.success]
    if data_steps and all(len(ev.text_summary.strip()) < 20 for ev in data_steps):
        return ReviewResult(
            verdict="gaps_found",
            confidence=0.8,
            gaps=["Les résultats sont trop courts ou vides"],
            feedback="Les requêtes SQL n'ont peut-être pas retourné de données. Vérifier la période et les filtres.",
            suggested_tools=["sql"],
        )

    return None


# ══════════════════════════════════════════════════════════════════════════════
# LLM Review
# ══════════════════════════════════════════════════════════════════════════════

def _llm_review(message: str, execution: ExecutionResult) -> ReviewResult:
    """Use LLM to assess completeness of execution results."""
    client = get_shared_client()

    evidence_summary = execution.text_context()[:2000]
    charts_note = f"Graphiques générés: {len(execution.chart_paths)}" if execution.chart_paths else "Aucun graphique"

    user_content = (
        f"QUESTION ORIGINALE: {message}\n\n"
        f"RÉSULTATS COLLECTÉS:\n{evidence_summary}\n\n"
        f"{charts_note}\n"
        f"Erreurs: {', '.join(execution.errors[:3]) if execution.errors else 'Aucune'}"
    )

    messages = [
        {"role": "system", "content": _REVIEWER_PROMPT},
        {"role": "user", "content": user_content},
    ]

    kwargs: Dict[str, Any] = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "messages": _prepare_messages(messages),
        "max_completion_tokens": 300,
    }
    if MODEL_IS_REASONING and not _should_skip_reasoning_effort():
        kwargs["reasoning_effort"] = CLASSIFIER_REASONING_EFFORT

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip() if response.choices else ""

        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            parsed = json.loads(json_match.group())
            verdict = parsed.get("verdict", "sufficient")
            if verdict not in ("sufficient", "gaps_found"):
                verdict = "sufficient"

            return ReviewResult(
                verdict=verdict,
                confidence=float(parsed.get("confidence", 0.8)),
                gaps=parsed.get("gaps", []),
                feedback=parsed.get("feedback", ""),
                suggested_tools=parsed.get("suggested_tools", []),
            )
    except Exception as e:
        logger.warning("Reviewer LLM failed: %s — defaulting to sufficient", str(e)[:100])

    # Default to sufficient if LLM fails (avoid infinite loops)
    return ReviewResult(verdict="sufficient", confidence=0.6)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def review(message: str, execution: ExecutionResult, use_llm: bool = True) -> ReviewResult:
    """
    Review execution results for completeness and quality.

    Args:
        message: Original user question
        execution: Results from the Executor
        use_llm: Whether to use LLM review (True) or heuristic only (False)

    Returns:
        ReviewResult with verdict and optional re-planning guidance
    """
    start = time.time()

    # Fast heuristic check first
    heuristic = _heuristic_review(message, execution)
    if heuristic:
        heuristic.duration_ms = (time.time() - start) * 1000
        logger.info("Review[heuristic]: %s (confidence=%.2f)", heuristic.verdict, heuristic.confidence)
        return heuristic

    # If evidence is clearly good enough, skip LLM
    successful_steps = [ev for ev in execution.evidence if ev.success]
    has_substantial_data = any(len(ev.text_summary) > 100 for ev in successful_steps)

    if has_substantial_data and not use_llm:
        result = ReviewResult(verdict="sufficient", confidence=0.85)
        result.duration_ms = (time.time() - start) * 1000
        return result

    # LLM review for nuanced assessment
    if use_llm and has_substantial_data:
        result = _llm_review(message, execution)
    else:
        # Not enough data for meaningful LLM review
        if not has_substantial_data and successful_steps:
            result = ReviewResult(
                verdict="gaps_found",
                confidence=0.7,
                gaps=["Résultats insuffisants pour une réponse complète"],
                feedback="Les données récupérées sont insuffisantes. Essayer avec des requêtes alternatives.",
                suggested_tools=["sql"],
            )
        else:
            result = ReviewResult(verdict="sufficient", confidence=0.7)

    result.duration_ms = (time.time() - start) * 1000
    logger.info(
        "Review[llm]: %s (confidence=%.2f, gaps=%d, %.0fms)",
        result.verdict, result.confidence, len(result.gaps), result.duration_ms,
    )
    return result
