"""
Planner Node — Structured Execution Plan Generation
=====================================================
For complex queries, generates a step-by-step execution plan
specifying which tools to call and in what order.

Input: user message + triage result + conversation context
Output: ExecutionPlan with ordered steps
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
    ANALYTICS_REASONING_EFFORT,
)
from .triage import _has_chart_request, _has_prediction_request, _sanitize_tools

logger = get_logger("statour.orchestration.planner")


# ══════════════════════════════════════════════════════════════════════════════
# Plan Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlanStep:
    """A single step in the execution plan."""
    step_id: int
    tool: str  # "sql" | "web_search" | "rag" | "prediction" | "chart"
    description: str  # what this step achieves
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)  # step_ids this depends on


@dataclass
class ExecutionPlan:
    """Complete execution plan for a complex query."""
    steps: List[PlanStep] = field(default_factory=list)
    synthesis_hint: str = ""  # guidance for how to combine results
    chart_requested: bool = False
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def tool_sequence(self) -> List[str]:
        """Return ordered list of tools to call."""
        return [s.tool for s in self.steps]


# ══════════════════════════════════════════════════════════════════════════════
# Planner Prompt
# ══════════════════════════════════════════════════════════════════════════════

_PLANNER_PROMPT = """Tu es le planificateur de STATOUR, la plateforme analytique du Ministère du Tourisme du Maroc.

Tu reçois une question complexe et tu dois créer un plan d'exécution structuré.

OUTILS DISPONIBLES:
1. **sql** — Exécuter des requêtes T-SQL sur Microsoft Fabric Lakehouse Gold
   - Tables: fact_statistiques_apf (arrivées frontières: mre, tes, nationalite, poste_frontiere, region, continent, voie, date_stat)
   - Tables: fact_statistiqueshebergementnationaliteestimees (nuitées, arrivées hôtelières: eht_id, nationalite_name, categorie_name, region_name, date_stat, nuitees, arrivees)
   - Jointures: gld_dim_categories_classements, gld_dim_etablissements_hebergements, gld_dim_delegations
   - Données mensuelles (date_stat = 1er du mois). Plage: 2019, 2023-2026 (Jan-Fev 2026)

2. **web_search** — Recherche web (Tavily/Exa/Brave)
   - Actualités tourisme Maroc, contexte international, stratégie, Vision 2030
   - Facteurs externes (événements, compagnies aériennes, politique)

3. **rag** — Base de connaissances interne STATOUR (ChromaDB)
   - Documents métier: définitions, procédures, terminologie MTAESS

4. **prediction** — Moteur de prévision
   - Projections futures basées sur CAGR/saisonnalité + facteurs web
   - Nécessite: année cible, scénario (optimiste/baseline/pessimiste)

5. **chart** — Génération de graphique Plotly
   - Dépend toujours d'une étape sql ou prediction qui fournit les données
   - Types: line, bar, pie, scatter, heatmap

RÈGLES DE PLANIFICATION:
- Un step "chart" dépend TOUJOURS d'un step "sql" ou "prediction" (il faut des données d'abord)
- Si la question mentionne "arrivées" de manière ambiguë, planifier 2 steps sql (APF + hébergement)
- Si la question demande du contexte/cause → ajouter web_search APRÈS sql
- Si des termes métier inconnus → ajouter rag en premier
- Minimum 1 step, maximum 5 steps
- Ordre logique: rag → sql → web_search → prediction → chart

RÉPONSE EN JSON STRICT:
{"steps": [{"step_id": 1, "tool": "...", "description": "...", "parameters": {...}, "depends_on": []}], "synthesis_hint": "comment combiner les résultats", "chart_requested": true/false}"""


# ══════════════════════════════════════════════════════════════════════════════
# Plan Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_plan(
    message: str,
    tools_needed: List[str],
    conversation_context: str = "",
    domain_context: Optional[str] = None,
) -> ExecutionPlan:
    """
    Generate a structured execution plan for a complex query.

    Args:
        message: The user's question
        tools_needed: Tools identified by triage
        conversation_context: Recent conversation summary
        domain_context: Active domain (apf/hebergement/None)

    Returns:
        ExecutionPlan with ordered steps
    """
    start = time.time()

    # Try LLM planning first
    plan = _llm_plan(message, tools_needed, conversation_context, domain_context)
    if plan and plan.steps:
        plan = _sanitize_plan(plan, message, tools_needed, conversation_context)
        plan.duration_ms = (time.time() - start) * 1000
        logger.info(
            "Plan generated: %d steps [%s] (%.0fms)",
            len(plan.steps), " → ".join(plan.tool_sequence()), plan.duration_ms,
        )
        return plan

    # Fallback: deterministic plan from tools_needed
    plan = _deterministic_plan(message, _sanitize_tools(message, conversation_context, tools_needed))
    plan.duration_ms = (time.time() - start) * 1000
    logger.info(
        "Plan[fallback]: %d steps [%s] (%.0fms)",
        len(plan.steps), " → ".join(plan.tool_sequence()), plan.duration_ms,
    )
    return plan


def _llm_plan(
    message: str,
    tools_needed: List[str],
    conversation_context: str,
    domain_context: Optional[str],
) -> Optional[ExecutionPlan]:
    """Generate plan using LLM."""
    client = get_shared_client()

    user_content = f"Question: {message}\n"
    if tools_needed:
        user_content += f"Outils suggérés par le triage: {', '.join(tools_needed)}\n"
    if domain_context:
        user_content += f"Domaine actif: {domain_context}\n"
    if conversation_context:
        user_content += f"Contexte conversation: {conversation_context[:500]}\n"

    messages = [
        {"role": "system", "content": _PLANNER_PROMPT},
        {"role": "user", "content": user_content},
    ]

    kwargs: Dict[str, Any] = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "messages": _prepare_messages(messages),
        "max_completion_tokens": 600,
    }
    if MODEL_IS_REASONING and not _should_skip_reasoning_effort():
        kwargs["reasoning_effort"] = ANALYTICS_REASONING_EFFORT

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip() if response.choices else ""

        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return None

        parsed = json.loads(json_match.group())
        return _parse_plan(parsed)

    except Exception as e:
        logger.warning("Planner LLM failed: %s — using deterministic fallback", str(e)[:100])
        return None


def _parse_plan(data: Dict[str, Any]) -> Optional[ExecutionPlan]:
    """Parse LLM JSON into an ExecutionPlan."""
    raw_steps = data.get("steps", [])
    if not raw_steps:
        return None

    valid_tools = {"sql", "web_search", "rag", "prediction", "chart"}
    steps = []

    for i, s in enumerate(raw_steps[:5]):  # cap at 5 steps
        tool = s.get("tool", "")
        if tool not in valid_tools:
            continue
        steps.append(PlanStep(
            step_id=s.get("step_id", i + 1),
            tool=tool,
            description=s.get("description", ""),
            parameters=s.get("parameters", {}),
            depends_on=s.get("depends_on", []),
        ))

    if not steps:
        return None

    return ExecutionPlan(
        steps=steps,
        synthesis_hint=data.get("synthesis_hint", ""),
        chart_requested=bool(data.get("chart_requested", False)),
    )


def _chart_has_data_dependency(plan: ExecutionPlan) -> bool:
    tools = plan.tool_sequence()
    if "chart" not in tools:
        return True
    return any(tool in {"sql", "prediction", "rag"} for tool in tools)


def _sanitize_plan(
    plan: ExecutionPlan,
    message: str,
    tools_needed: List[str],
    conversation_context: str = "",
) -> ExecutionPlan:
    """Keep LLM plans aligned with the user's requested deliverables."""
    prediction_allowed = _has_prediction_request(message, conversation_context)
    chart_allowed = _has_chart_request(message)
    sanitized_tools = set(_sanitize_tools(message, conversation_context, tools_needed))

    if chart_allowed and "chart" in plan.tool_sequence() and not _chart_has_data_dependency(plan):
        if any(tool in sanitized_tools for tool in {"sql", "prediction", "rag"}):
            logger.info("Planner repaired chart-only plan by restoring data dependency")
            return _deterministic_plan(message, list(sanitized_tools))

    if prediction_allowed and ("prediction" in sanitized_tools or "prediction" in plan.tool_sequence()):
        if chart_allowed or "chart" not in plan.tool_sequence():
            return plan

    suppressed_tools = set()
    if not prediction_allowed:
        suppressed_tools.add("prediction")
    if not chart_allowed:
        suppressed_tools.add("chart")

    if not (suppressed_tools & set(plan.tool_sequence())):
        return plan

    kept_original = [step for step in plan.steps if step.tool not in suppressed_tools]
    if not kept_original:
        fallback_tools = list(sanitized_tools) or ["sql"]
        return _deterministic_plan(message, fallback_tools)

    old_to_new: Dict[int, int] = {}
    new_steps: List[PlanStep] = []
    for old_step in kept_original:
        new_id = len(new_steps) + 1
        old_to_new[old_step.step_id] = new_id
        new_steps.append(PlanStep(
            step_id=new_id,
            tool=old_step.tool,
            description=old_step.description,
            parameters=old_step.parameters,
            depends_on=[],
        ))

    for old_step, new_step in zip(kept_original, new_steps):
        new_step.depends_on = [
            old_to_new[dep_id]
            for dep_id in old_step.depends_on
            if dep_id in old_to_new
        ]

        if new_step.tool == "chart" and not new_step.depends_on:
            prior_data_steps = [
                candidate.step_id
                for candidate in new_steps
                if candidate.step_id < new_step.step_id and candidate.tool in {"sql", "rag"}
            ]
            if prior_data_steps:
                new_step.depends_on = [prior_data_steps[-1]]

    new_steps = [
        step for step in new_steps
        if step.tool != "chart" or step.depends_on or any(s.tool in {"sql", "rag"} for s in new_steps)
    ]

    if not new_steps:
        fallback_tools = list(sanitized_tools) or ["sql"]
        return _deterministic_plan(message, fallback_tools)

    if "prediction" in suppressed_tools:
        logger.info("Planner suppressed prediction step: no forecast intent in current turn")
    if "chart" in suppressed_tools:
        logger.info("Planner suppressed chart step: no chart intent in current turn")
    return ExecutionPlan(
        steps=new_steps,
        synthesis_hint=plan.synthesis_hint,
        chart_requested=plan.chart_requested and chart_allowed,
        metadata={
            **plan.metadata,
            "prediction_suppressed": "prediction" in suppressed_tools,
            "chart_suppressed": "chart" in suppressed_tools,
        },
    )


def _deterministic_plan(message: str, tools_needed: List[str]) -> ExecutionPlan:
    """Build a simple sequential plan from the tools list."""
    tool_order = ["rag", "sql", "web_search", "prediction", "chart"]
    ordered = sorted(tools_needed, key=lambda t: tool_order.index(t) if t in tool_order else 99)

    if not ordered:
        ordered = ["sql"]

    # Cap at 5 steps maximum
    ordered = ordered[:5]

    steps = []
    for i, tool in enumerate(ordered):
        deps = [i] if (tool == "chart" and i > 0) else []
        steps.append(PlanStep(
            step_id=i + 1,
            tool=tool,
            description=f"Étape {i+1}: {tool}",
            parameters={},
            depends_on=deps,
        ))

    has_chart = "chart" in ordered or any(
        kw in message.lower()
        for kw in ("graphique", "chart", "courbe", "visualis", "diagramme")
    )

    return ExecutionPlan(
        steps=steps,
        synthesis_hint="Combiner les résultats des étapes précédentes en une réponse cohérente.",
        chart_requested=has_chart,
    )
