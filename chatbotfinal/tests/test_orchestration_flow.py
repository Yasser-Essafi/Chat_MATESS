"""
Tests for the new Plan-Execute-Review-Humanize orchestration flow.
==================================================================
Unit tests for each node + integration tests for the full graph.

Run: python -m pytest chatbotfinal/tests/test_orchestration_flow.py -v
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Triage Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTriageDeterministic:
    """Test the deterministic fast-path triage (no LLM calls)."""

    def test_greeting_french(self):
        from orchestration.triage import triage, _deterministic_triage
        result = _deterministic_triage("Bonjour")
        assert result is not None
        assert result.complexity == "simple"
        assert result.intent == "greeting"
        assert result.confidence == 1.0

    def test_greeting_english(self):
        from orchestration.triage import _deterministic_triage
        result = _deterministic_triage("Hello!")
        assert result is not None
        assert result.complexity == "simple"
        assert result.intent == "greeting"

    def test_greeting_arabic(self):
        from orchestration.triage import _deterministic_triage
        result = _deterministic_triage("مرحبا")
        assert result is not None
        assert result.complexity == "simple"
        assert result.intent == "greeting"

    def test_thanks(self):
        from orchestration.triage import _deterministic_triage
        result = _deterministic_triage("Merci beaucoup!")
        assert result is not None
        assert result.complexity == "simple"
        assert result.intent == "thanks"

    def test_farewell(self):
        from orchestration.triage import _deterministic_triage
        result = _deterministic_triage("Au revoir")
        assert result is not None
        assert result.complexity == "simple"
        assert result.intent == "farewell"

    def test_platform_qa(self):
        from orchestration.triage import _deterministic_triage
        result = _deterministic_triage("C'est quoi STATOUR?")
        assert result is not None
        assert result.complexity == "simple"
        assert result.intent == "platform_qa"

    def test_complex_query_not_caught(self):
        from orchestration.triage import _deterministic_triage
        result = _deterministic_triage("Combien de touristes en 2025 par mois et par ville?")
        assert result is None  # Should fall through to LLM

    def test_analytics_query_not_caught_by_deterministic(self):
        from orchestration.triage import _deterministic_triage
        result = _deterministic_triage("Donne-moi les nuitées à Marrakech pour 2024")
        assert result is None  # Needs LLM triage


class TestTriageHeuristic:
    """Test the heuristic fallback triage (no LLM)."""

    def test_analytics_heuristic(self):
        from orchestration.triage import _heuristic_triage
        result = _heuristic_triage("Combien de nuitées à Casablanca en 2025?")
        assert result.complexity == "complex"
        assert "sql" in result.tools_needed

    def test_search_heuristic(self):
        from orchestration.triage import _heuristic_triage
        result = _heuristic_triage("Quelles sont les actualités du tourisme marocain?")
        assert result.complexity == "complex"
        assert "web_search" in result.tools_needed

    def test_prediction_heuristic(self):
        from orchestration.triage import _heuristic_triage
        result = _heuristic_triage("Prévision des arrivées en 2028 scénario optimiste")
        assert result.complexity == "complex"
        assert "prediction" in result.tools_needed

    def test_unaccented_prediction_heuristic(self):
        from orchestration.triage import _heuristic_triage
        result = _heuristic_triage("prediction pour 2026 avec graphique")
        assert result.complexity == "complex"
        assert "prediction" in result.tools_needed
        assert "chart" in result.tools_needed

    def test_compound_analysis_prediction_heuristic(self):
        from orchestration.triage import _heuristic_triage
        result = _heuristic_triage(
            "Analyse le tourisme de 2018 jusqua 2025 puis cree une prediction "
            "pour l'annee 2026 avec graphique"
        )
        assert result.complexity == "complex"
        assert "sql" in result.tools_needed
        assert "prediction" in result.tools_needed
        assert "chart" in result.tools_needed

    def test_plain_historical_analysis_does_not_request_prediction(self):
        from orchestration.triage import _heuristic_triage
        result = _heuristic_triage("Analyse les arrivees touristiques de 2019 a 2025")
        assert result.complexity == "complex"
        assert "sql" in result.tools_needed
        assert "prediction" not in result.tools_needed

    def test_llm_tool_sanitizer_suppresses_unrequested_prediction(self):
        from orchestration.triage import _sanitize_tools
        tools = _sanitize_tools(
            "Analyse les arrivees touristiques de 2019 a 2025",
            "",
            ["sql", "prediction", "chart"],
        )
        assert tools == ["sql"]

    def test_short_prediction_followup_keeps_prediction_tool(self):
        from orchestration.triage import _sanitize_tools
        tools = _sanitize_tools(
            "Et pour 2028 ?",
            "USER: prevision 2027\nASSISTANT: projection 2027",
            ["prediction", "chart"],
        )
        assert "prediction" in tools

    def test_chart_heuristic(self):
        from orchestration.triage import _heuristic_triage
        result = _heuristic_triage("Graphique des arrivées par mois en 2024")
        assert result.complexity == "complex"
        assert "chart" in result.tools_needed
        assert "sql" in result.tools_needed

    def test_simple_short_message(self):
        from orchestration.triage import _heuristic_triage
        result = _heuristic_triage("ok super")
        assert result.complexity == "simple"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Planner Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPlannerDeterministic:
    """Test the deterministic fallback planner."""

    def test_single_tool_plan(self):
        from orchestration.planner import _deterministic_plan
        plan = _deterministic_plan("Combien de touristes?", ["sql"])
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "sql"

    def test_multi_tool_plan_ordering(self):
        from orchestration.planner import _deterministic_plan
        plan = _deterministic_plan("Analyse avec graphique", ["chart", "sql", "web_search"])
        tools = plan.tool_sequence()
        assert tools.index("sql") < tools.index("web_search")
        assert tools.index("web_search") < tools.index("chart")

    def test_empty_tools_defaults_to_sql(self):
        from orchestration.planner import _deterministic_plan
        plan = _deterministic_plan("Question quelconque", [])
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "sql"

    def test_chart_detection_from_message(self):
        from orchestration.planner import _deterministic_plan
        plan = _deterministic_plan("Montre-moi un graphique des arrivées", ["sql"])
        assert plan.chart_requested is True

    def test_plan_capped_at_5_tools(self):
        from orchestration.planner import _deterministic_plan
        plan = _deterministic_plan("test", ["sql", "web_search", "rag", "prediction", "chart", "sql"])
        assert len(plan.steps) <= 5


class TestPlanParser:
    """Test the JSON plan parser."""

    def test_parse_valid_plan(self):
        from orchestration.planner import _parse_plan
        data = {
            "steps": [
                {"step_id": 1, "tool": "sql", "description": "Get data", "parameters": {}, "depends_on": []},
                {"step_id": 2, "tool": "chart", "description": "Make chart", "parameters": {}, "depends_on": [1]},
            ],
            "synthesis_hint": "Combine",
            "chart_requested": True,
        }
        plan = _parse_plan(data)
        assert plan is not None
        assert len(plan.steps) == 2
        assert plan.chart_requested is True

    def test_parse_invalid_tools_filtered(self):
        from orchestration.planner import _parse_plan
        data = {
            "steps": [
                {"step_id": 1, "tool": "invalid_tool", "description": "bad"},
                {"step_id": 2, "tool": "sql", "description": "good"},
            ],
        }
        plan = _parse_plan(data)
        assert plan is not None
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "sql"

    def test_parse_empty_steps(self):
        from orchestration.planner import _parse_plan
        plan = _parse_plan({"steps": []})
        assert plan is None


# ══════════════════════════════════════════════════════════════════════════════
def test_sanitize_plan_removes_unrequested_prediction():
    from orchestration.planner import ExecutionPlan, PlanStep, _sanitize_plan
    plan = ExecutionPlan(
        steps=[
            PlanStep(step_id=1, tool="sql", description="Historique"),
            PlanStep(step_id=2, tool="prediction", description="Projection"),
            PlanStep(step_id=3, tool="chart", description="Graphique", depends_on=[2]),
        ],
        chart_requested=True,
    )

    sanitized = _sanitize_plan(
        plan,
        "Analyse les arrivees touristiques de 2019 a 2025 avec graphique",
        ["sql", "prediction", "chart"],
    )

    assert sanitized.tool_sequence() == ["sql", "chart"]
    assert sanitized.steps[1].depends_on == [1]


# Phase 3: Executor Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutor:
    """Test executor with mocked tools."""

    def test_sql_no_db(self):
        from orchestration.executor import Executor, Evidence
        from orchestration.planner import PlanStep, ExecutionPlan
        executor = Executor(db_layer=None)
        step = PlanStep(step_id=1, tool="sql", description="test")
        ev = executor._execute_step(step, "test query", {}, "")
        assert not ev.success
        assert "unavailable" in ev.error.lower()

    def test_web_search_no_tool(self):
        from orchestration.executor import Executor
        from orchestration.planner import PlanStep
        executor = Executor(search_tool=None)
        step = PlanStep(step_id=1, tool="web_search", description="test")
        ev = executor._execute_step(step, "test", {}, "")
        assert not ev.success
        assert "not available" in ev.error.lower()

    def test_rag_no_manager(self):
        from orchestration.executor import Executor
        from orchestration.planner import PlanStep
        executor = Executor(rag_manager=None)
        step = PlanStep(step_id=1, tool="rag", description="test")
        ev = executor._execute_step(step, "test", {}, "")
        assert not ev.success

    def test_prediction_no_engine(self):
        from orchestration.executor import Executor
        from orchestration.planner import PlanStep
        executor = Executor(prediction_engine=None)
        step = PlanStep(step_id=1, tool="prediction", description="test")
        ev = executor._execute_step(step, "test", {}, "")
        assert not ev.success

    def test_unknown_tool(self):
        from orchestration.executor import Executor
        from orchestration.planner import PlanStep
        executor = Executor()
        step = PlanStep(step_id=1, tool="nonexistent", description="test")
        ev = executor._execute_step(step, "test", {}, "")
        assert not ev.success
        assert "Unknown" in ev.error

    def test_sandbox_simple_print(self):
        from orchestration.executor import Executor
        executor = Executor()
        # Mock db_layer with a simple return
        mock_db = MagicMock()
        mock_db.source = "fabric"
        import pandas as pd
        mock_db.safe_query.return_value = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        executor._db = mock_db

        result = executor._execute_sandbox("df = sql('SELECT 1')\nprint(to_md(df))")
        assert result["error"] is None
        assert "a" in result["output"]

    def test_sandbox_error_handling(self):
        from orchestration.executor import Executor
        executor = Executor()
        result = executor._execute_sandbox("raise ValueError('test error')")
        assert result["error"] is not None
        assert "test error" in result["error"]

    def test_execute_plan_collects_evidence(self):
        from orchestration.executor import Executor
        from orchestration.planner import PlanStep, ExecutionPlan
        executor = Executor()
        plan = ExecutionPlan(
            steps=[
                PlanStep(step_id=1, tool="rag", description="search docs"),
                PlanStep(step_id=2, tool="web_search", description="search web"),
            ]
        )
        result = executor.execute_plan(plan, "test message")
        assert len(result.evidence) == 2
        assert result.total_duration_ms >= 0


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Reviewer Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestReviewer:
    """Test reviewer heuristics."""

    def test_all_failed_steps(self):
        from orchestration.reviewer import _heuristic_review
        from orchestration.executor import ExecutionResult, Evidence

        execution = ExecutionResult(
            evidence=[
                Evidence(step_id=1, tool="sql", success=False, error="Connection failed"),
            ]
        )
        result = _heuristic_review("test question", execution)
        assert result is not None
        assert result.verdict == "gaps_found"

    def test_chart_missing_when_requested(self):
        from orchestration.reviewer import _heuristic_review
        from orchestration.executor import ExecutionResult, Evidence

        execution = ExecutionResult(
            evidence=[
                Evidence(step_id=1, tool="sql", success=True,
                        text_summary="Total arrivées 2024: 15.5M visiteurs"),
            ],
            chart_paths=[],
        )
        result = _heuristic_review("Montre-moi un graphique des arrivées", execution)
        assert result is not None
        assert result.verdict == "gaps_found"
        assert any("graphique" in g.lower() for g in result.gaps)

    def test_sufficient_data(self):
        from orchestration.reviewer import _heuristic_review
        from orchestration.executor import ExecutionResult, Evidence

        execution = ExecutionResult(
            evidence=[
                Evidence(step_id=1, tool="sql", success=True,
                        text_summary="Arrivées 2024: 15.5M. Par ville: Marrakech 4.2M, Casablanca 3.1M..." * 3),
            ],
            chart_paths=["charts/test.html"],
        )
        result = _heuristic_review("Graphique des arrivées par ville 2024", execution)
        # Heuristic should return None (pass to LLM or skip)
        assert result is None

    def test_empty_results_detected(self):
        from orchestration.reviewer import _heuristic_review
        from orchestration.executor import ExecutionResult, Evidence

        execution = ExecutionResult(
            evidence=[
                Evidence(step_id=1, tool="sql", success=True, text_summary=""),
            ],
        )
        result = _heuristic_review("Données de 2025", execution)
        assert result is not None
        assert result.verdict == "gaps_found"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: Humanizer Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHumanizerFallbacks:
    """Test humanizer fallback responses (no LLM)."""

    def test_french_greeting_fallback(self):
        from orchestration.humanizer import _fallback_simple
        response = _fallback_simple("greeting", "fr")
        assert "Bonjour" in response
        assert len(response) > 10

    def test_english_greeting_fallback(self):
        from orchestration.humanizer import _fallback_simple
        response = _fallback_simple("greeting", "en")
        assert "Hello" in response

    def test_arabic_greeting_fallback(self):
        from orchestration.humanizer import _fallback_simple
        response = _fallback_simple("greeting", "ar")
        assert "مرحبا" in response

    def test_thanks_fallback(self):
        from orchestration.humanizer import _fallback_simple
        response = _fallback_simple("thanks", "fr")
        assert "plaisir" in response.lower() or "hésitez" in response.lower()

    def test_platform_qa_fallback(self):
        from orchestration.humanizer import _fallback_simple
        response = _fallback_simple("platform_qa", "fr")
        assert "STATOUR" in response
        assert "Ministère" in response

    def test_complex_fallback(self):
        from orchestration.humanizer import _fallback_complex
        response = _fallback_complex("test question", "Total: 15M visiteurs en 2024")
        assert "15M" in response
        assert len(response) > 20

    def test_complex_fallback_empty(self):
        from orchestration.humanizer import _fallback_complex
        response = _fallback_complex("test", "")
        assert "récupéré" in response.lower() or "reformul" in response.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6: Integration / Graph Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphSimplePath:
    """Test the full graph with simple queries (mocked LLM)."""

    @patch("orchestration.triage._llm_triage")
    @patch("orchestration.graph.humanize_simple")
    def test_greeting_goes_simple_path(self, mock_humanize, mock_llm_triage):
        from orchestration.graph import run_graph
        mock_humanize.return_value = "Bonjour ! Je suis ravi de vous aider."

        result = run_graph("Bonjour!")
        assert result["agent"] == "normal"
        assert "Bonjour" in result["response"]
        assert not result["chart_paths"]
        # LLM triage should NOT be called (deterministic catches greetings)
        mock_llm_triage.assert_not_called()

    @patch("orchestration.graph.humanize_simple")
    def test_thanks_simple_path(self, mock_humanize):
        from orchestration.graph import run_graph
        mock_humanize.return_value = "Avec plaisir !"

        result = run_graph("Merci!")
        assert result["agent"] == "normal"
        assert result["response"] == "Avec plaisir !"

    @patch("orchestration.graph.humanize_simple")
    def test_farewell_simple_path(self, mock_humanize):
        from orchestration.graph import run_graph
        mock_humanize.return_value = "Au revoir ! Bonne journée."

        result = run_graph("Au revoir")
        assert result["agent"] == "normal"


class TestGraphComplexPath:
    """Test the full graph with complex queries (mocked LLM and tools)."""

    @patch("orchestration.graph.humanize_complex")
    @patch("orchestration.graph.review")
    @patch("orchestration.graph.generate_plan")
    @patch("orchestration.graph.triage")
    def test_analytics_complex_path(self, mock_triage, mock_plan, mock_review, mock_humanize):
        from orchestration.graph import run_graph
        from orchestration.triage import TriageResult
        from orchestration.planner import ExecutionPlan, PlanStep
        from orchestration.reviewer import ReviewResult

        mock_triage.return_value = TriageResult(
            complexity="complex",
            intent="analytics",
            tools_needed=["sql"],
        )
        mock_plan.return_value = ExecutionPlan(
            steps=[PlanStep(step_id=1, tool="sql", description="Get data")],
        )
        mock_review.return_value = ReviewResult(verdict="sufficient", confidence=0.9)
        mock_humanize.return_value = "Les arrivées en 2024 ont atteint **15.5 millions**."

        result = run_graph("Combien d'arrivées en 2024?")

        assert result["agent"] == "analytics"
        assert "15.5 millions" in result["response"]
        assert len(result["trace"]) >= 4  # triage, plan, execute, review, humanize
        mock_triage.assert_called_once()
        mock_plan.assert_called_once()

    @patch("orchestration.graph.humanize_complex")
    @patch("orchestration.graph.review")
    @patch("orchestration.graph.generate_plan")
    @patch("orchestration.graph.triage")
    def test_trace_populated(self, mock_triage, mock_plan, mock_review, mock_humanize):
        from orchestration.graph import run_graph
        from orchestration.triage import TriageResult
        from orchestration.planner import ExecutionPlan, PlanStep
        from orchestration.reviewer import ReviewResult

        mock_triage.return_value = TriageResult(
            complexity="complex", intent="analytics", tools_needed=["sql"],
        )
        mock_plan.return_value = ExecutionPlan(
            steps=[PlanStep(step_id=1, tool="sql", description="test")],
        )
        mock_review.return_value = ReviewResult(verdict="sufficient", confidence=0.9)
        mock_humanize.return_value = "Réponse de test"

        result = run_graph("test analytics")
        trace_stages = [t["stage"] for t in result["trace"]]
        assert "triage" in trace_stages
        assert "plan" in trace_stages
        assert "execute" in trace_stages
        assert "review" in trace_stages
        assert "humanize" in trace_stages


# ══════════════════════════════════════════════════════════════════════════════
# Golden Tone Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGoldenTone:
    """Test that humanizer fallback responses have the right tone."""

    def test_no_robotic_greeting(self):
        from orchestration.humanizer import _fallback_simple
        response = _fallback_simple("greeting", "fr")
        robotic_phrases = [
            "je suis un assistant ia",
            "en tant qu'intelligence artificielle",
            "voici les informations demandées",
        ]
        for phrase in robotic_phrases:
            assert phrase not in response.lower()

    def test_complex_fallback_not_empty_template(self):
        from orchestration.humanizer import _fallback_complex
        evidence = "Marrakech: 4.2M nuitées, +12% vs 2023. Casablanca: 3.1M, +8%."
        response = _fallback_complex("Nuitées par ville?", evidence)
        assert "Marrakech" in response
        assert "4.2M" in response

    def test_platform_qa_is_informative(self):
        from orchestration.humanizer import _fallback_simple
        response = _fallback_simple("platform_qa", "fr")
        assert "tourisme" in response.lower() or "Tourisme" in response
        assert "données" in response.lower() or "statistique" in response.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Execution Result Helper Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionResult:
    """Test ExecutionResult helper methods."""

    def test_text_context_combines(self):
        from orchestration.executor import ExecutionResult, Evidence
        result = ExecutionResult(evidence=[
            Evidence(step_id=1, tool="sql", success=True, text_summary="Data from SQL"),
            Evidence(step_id=2, tool="web_search", success=True, text_summary="Data from web"),
        ])
        context = result.text_context()
        assert "[SQL]" in context
        assert "[WEB_SEARCH]" in context
        assert "Data from SQL" in context
        assert "Data from web" in context

    def test_text_context_skips_empty(self):
        from orchestration.executor import ExecutionResult, Evidence
        result = ExecutionResult(evidence=[
            Evidence(step_id=1, tool="sql", success=True, text_summary=""),
            Evidence(step_id=2, tool="web_search", success=True, text_summary="Has data"),
        ])
        context = result.text_context()
        assert "[SQL]" not in context
        assert "[WEB_SEARCH]" in context


# ══════════════════════════════════════════════════════════════════════════════
# Feature Flag Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureFlag:
    """Test that USE_NEW_FLOW flag works correctly."""

    def test_settings_has_flag(self):
        from config.settings import USE_NEW_FLOW, MAX_REPLAN_ATTEMPTS
        assert isinstance(USE_NEW_FLOW, bool)
        assert isinstance(MAX_REPLAN_ATTEMPTS, int)
        assert MAX_REPLAN_ATTEMPTS >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
