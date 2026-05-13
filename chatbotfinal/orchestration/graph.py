"""
Orchestration Graph — Plan-Execute-Review-Humanize Pipeline
=============================================================
Ties all nodes together into the complete agentic flow.

Flow:
  User Message → Triage → [simple: Humanizer] or [complex: Planner → Executor → Reviewer → Humanizer]

This module is called by the Orchestrator when USE_NEW_FLOW is True.
"""

import time
import uuid
from typing import Optional, Dict, Any, List

from utils.base_agent import detect_language
from utils.logger import get_logger
from config.settings import MAX_REPLAN_ATTEMPTS

from .triage import triage, TriageResult
from .planner import generate_plan, ExecutionPlan
from .executor import Executor, ExecutionResult
from .reviewer import review, ReviewResult
from .humanizer import humanize_simple, humanize_complex
from .followup import resolve_followup
from .external_impact import should_handle_external_impact, run_external_impact_analysis

logger = get_logger("statour.orchestration.graph")


def run_graph(
    message: str,
    db_layer=None,
    search_tool=None,
    rag_manager=None,
    prediction_engine=None,
    conversation_context: str = "",
    domain_context: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the full Plan-Execute-Review-Humanize pipeline.

    Args:
        message: User's input message
        db_layer: DBLayer instance for Fabric SQL
        search_tool: TourismSearchTool instance
        rag_manager: RAGManager instance
        prediction_engine: PredictionAgent instance
        conversation_context: Recent conversation summary
        domain_context: Active domain (apf/hebergement/None)
        run_id: Unique run identifier

    Returns:
        Dict compatible with the legacy orchestrator response format
    """
    start = time.time()
    rid = run_id or uuid.uuid4().hex[:10]
    trace: List[Dict[str, Any]] = []
    language = detect_language(message)
    working_message = resolve_followup(message, conversation_context)
    followup_resolved = working_message != message

    # Specialized path for geopolitical/macroeconomic/event impact analysis.
    # Run it before triage: the matcher is deterministic and avoids spending
    # several seconds classifying questions that already have a controlled path.
    if should_handle_external_impact(working_message, conversation_context):
        if followup_resolved:
            trace.append({
                "stage": "context",
                "label": "Resolution du suivi",
                "status": "done",
                "agent": "orchestrator",
                "detail": "Le tour court reprend la forme analytique du tour precedent.",
            })
        external_result = run_external_impact_analysis(
            message=working_message,
            db_layer=db_layer,
            search_tool=search_tool,
            conversation_context=conversation_context,
            run_id=rid,
        )
        external_result["classification_time_ms"] = 0.0
        external_result["total_time_ms"] = round((time.time() - start) * 1000, 1)
        external_result["trace"] = trace + external_result.get("trace", [])
        return external_result

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: TRIAGE
    # ═══════════════════════════════════════════════════════════════════════
    triage_result = triage(message, conversation_context)
    trace.append({
        "stage": "triage",
        "label": "Évaluation de la complexité",
        "status": "done",
        "agent": "orchestrator",
        "detail": f"Complexité: {triage_result.complexity}, Intent: {triage_result.intent}",
        "duration_ms": round(triage_result.duration_ms, 1),
    })

    # ═══════════════════════════════════════════════════════════════════════
    # SIMPLE PATH: Direct humanized response
    # ═══════════════════════════════════════════════════════════════════════
    if followup_resolved:
        trace.append({
            "stage": "context",
            "label": "Resolution du suivi",
            "status": "done",
            "agent": "orchestrator",
            "detail": "Le tour court reprend la forme analytique du tour precedent.",
        })

    if triage_result.complexity == "simple":
        response = triage_result.direct_answer or humanize_simple(
            message=working_message,
            intent=triage_result.intent,
            conversation_context=conversation_context,
            language=language,
        )
        total_ms = (time.time() - start) * 1000
        trace.append({
            "stage": "humanize",
            "label": "Réponse directe",
            "status": "done",
            "agent": "humanizer",
            "detail": "Réponse simple et chaleureuse générée.",
            "duration_ms": round(total_ms - triage_result.duration_ms, 1),
        })

        return {
            "agent": "normal",
            "agent_icon": "",
            "agent_name": "Assistant STATOUR",
            "response": response,
            "chart_path": None,
            "chart_paths": [],
            "sources": [],
            "confidence": None,
            "data_freshness": {},
            "rerouted": False,
            "classification_time_ms": round(triage_result.duration_ms, 1),
            "total_time_ms": round(total_ms, 1),
            "metric_context": None,
            "period": {},
            "data_scope_note": "",
            "run_id": rid,
            "trace": trace,
            "fallbacks": [],
            "errors": [],
        }

    # Specialized path for geopolitical/macroeconomic/event impact analysis.
    # It resolves the event before SQL, so the query window is not guessed by
    # the generic planner.
    if should_handle_external_impact(working_message, conversation_context):
        external_result = run_external_impact_analysis(
            message=working_message,
            db_layer=db_layer,
            search_tool=search_tool,
            conversation_context=conversation_context,
            run_id=rid,
        )
        external_result["classification_time_ms"] = round(triage_result.duration_ms, 1)
        external_result["total_time_ms"] = round((time.time() - start) * 1000, 1)
        external_result["trace"] = trace + external_result.get("trace", [])
        return external_result

    # ═══════════════════════════════════════════════════════════════════════
    # COMPLEX PATH: Plan → Execute → Review → Humanize
    # ═══════════════════════════════════════════════════════════════════════

    executor = Executor(
        db_layer=db_layer,
        search_tool=search_tool,
        rag_manager=rag_manager,
        prediction_engine=prediction_engine,
    )

    all_chart_paths: List[str] = []
    all_errors: List[Dict[str, Any]] = []
    fallbacks: List[Dict[str, Any]] = []
    execution_result: Optional[ExecutionResult] = None

    for attempt in range(1 + MAX_REPLAN_ATTEMPTS):
        # ─── Plan ───
        plan_start = time.time()
        plan = generate_plan(
            message=working_message,
            tools_needed=triage_result.tools_needed,
            conversation_context=conversation_context,
            domain_context=domain_context,
        )
        trace.append({
            "stage": "plan",
            "label": f"Planification{' (re-plan)' if attempt > 0 else ''}",
            "status": "done",
            "agent": "planner",
            "detail": f"{len(plan.steps)} étapes: {' → '.join(plan.tool_sequence())}",
            "duration_ms": round(plan.duration_ms, 1),
        })

        # ─── Execute ───
        exec_start = time.time()
        execution_result = executor.execute_plan(plan, working_message, conversation_context)
        exec_duration = (time.time() - exec_start) * 1000
        trace.append({
            "stage": "execute",
            "label": f"Exécution{' (2ème passage)' if attempt > 0 else ''}",
            "status": "done" if execution_result.all_successful else "partial",
            "agent": "executor",
            "detail": f"{len(execution_result.evidence)} résultats, "
                     f"{len(execution_result.chart_paths)} graphiques",
            "duration_ms": round(exec_duration, 1),
            "artifact_count": len(execution_result.chart_paths),
        })
        all_chart_paths.extend(execution_result.chart_paths)

        if execution_result.errors:
            all_errors.extend([
                {"stage": "execute", "error": e} for e in execution_result.errors
            ])

        # ─── Review ───
        review_result = review(working_message, execution_result, use_llm=True)
        trace.append({
            "stage": "review",
            "label": "Contrôle qualité",
            "status": "done",
            "agent": "reviewer",
            "detail": f"Verdict: {review_result.verdict} "
                     f"(confiance: {review_result.confidence:.0%})"
                     + (f", Lacunes: {', '.join(review_result.gaps[:2])}" if review_result.gaps else ""),
            "duration_ms": round(review_result.duration_ms, 1),
        })

        if review_result.verdict == "sufficient":
            break

        # Gaps found — re-plan if we haven't exhausted attempts
        if attempt < MAX_REPLAN_ATTEMPTS:
            # Update tools_needed with reviewer suggestions
            if review_result.suggested_tools:
                for tool in review_result.suggested_tools:
                    if tool not in triage_result.tools_needed:
                        triage_result.tools_needed.append(tool)
            fallbacks.append({
                "stage": "review",
                "reason": "; ".join(review_result.gaps[:3]),
                "action": "re-plan",
            })
            logger.info("Review found gaps, re-planning (attempt %d)", attempt + 1)
        else:
            fallbacks.append({
                "stage": "review",
                "reason": "; ".join(review_result.gaps[:3]),
                "action": "proceed_with_gaps",
            })

    # ─── Humanize ───
    humanize_start = time.time()
    evidence_text = execution_result.text_context() if execution_result else ""
    response = humanize_complex(
        message=message,
        evidence_text=evidence_text,
        chart_paths=all_chart_paths,
        conversation_context=conversation_context,
        language=language,
    )
    humanize_duration = (time.time() - humanize_start) * 1000
    trace.append({
        "stage": "humanize",
        "label": "Rédaction de la réponse",
        "status": "done",
        "agent": "humanizer",
        "detail": f"Réponse de {len(response)} caractères rédigée.",
        "duration_ms": round(humanize_duration, 1),
    })

    total_ms = (time.time() - start) * 1000

    # Determine agent label based on dominant tool
    agent_key = _dominant_agent(triage_result, execution_result)
    agent_names = {
        "analytics": "Analyste de Données",
        "researcher": "Chercheur Tourisme",
        "prediction": "Prévisionniste STATOUR",
        "normal": "Assistant STATOUR",
    }

    chart_path = all_chart_paths[0] if all_chart_paths else None

    return {
        "agent": agent_key,
        "agent_icon": "",
        "agent_name": agent_names.get(agent_key, "Analyste STATOUR"),
        "response": response,
        "chart_path": chart_path,
        "chart_paths": all_chart_paths[:5],
        "sources": _extract_sources(execution_result),
        "confidence": f"{review_result.confidence:.0%}" if review_result else None,
        "data_freshness": {},
        "rerouted": False,
        "classification_time_ms": round(triage_result.duration_ms, 1),
        "total_time_ms": round(total_ms, 1),
        "metric_context": _infer_metric_context(triage_result, execution_result),
        "period": {},
        "data_scope_note": "",
        "run_id": rid,
        "trace": trace,
        "fallbacks": fallbacks,
        "errors": all_errors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dominant_agent(triage_result: TriageResult, execution: Optional[ExecutionResult]) -> str:
    """Determine which agent label best describes this response."""
    if not execution:
        return "normal"

    tool_counts: Dict[str, int] = {}
    for ev in execution.evidence:
        if ev.success:
            tool_counts[ev.tool] = tool_counts.get(ev.tool, 0) + 1

    if tool_counts.get("prediction") or "prediction" in triage_result.tools_needed:
        return "prediction"

    tool_to_agent = {
        "sql": "analytics",
        "chart": "analytics",
        "web_search": "researcher",
        "rag": "researcher",
        "prediction": "prediction",
    }

    if not tool_counts:
        # Use triage hint
        if "prediction" in triage_result.tools_needed:
            return "prediction"
        if "sql" in triage_result.tools_needed:
            return "analytics"
        if "web_search" in triage_result.tools_needed:
            return "researcher"
        return "analytics"

    # Pick the tool with most successful executions
    dominant_tool = max(tool_counts, key=tool_counts.get)
    return tool_to_agent.get(dominant_tool, "analytics")


def _extract_sources(execution: Optional[ExecutionResult]) -> List[Dict[str, Any]]:
    """Extract web sources from execution evidence."""
    if not execution:
        return []

    sources = []
    for ev in execution.evidence:
        if ev.tool == "web_search" and ev.success and ev.metadata.get("sources"):
            sources.extend(ev.metadata["sources"])
    return sources[:10]


def _infer_metric_context(triage_result: TriageResult, execution: Optional[ExecutionResult]) -> Optional[str]:
    """Infer metric context from tools used."""
    if not execution:
        return None

    tools_used = {ev.tool for ev in execution.evidence if ev.success}
    if "prediction" in tools_used:
        return "prediction"
    if "web_search" in tools_used and "sql" not in tools_used:
        return "research"
    if "sql" in tools_used:
        return "analytics"
    return None
