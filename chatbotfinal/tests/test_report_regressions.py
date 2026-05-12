import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_api_chat_without_conversation_id_creates_isolated_conversations(monkeypatch, tmp_path):
    import server
    import ui.state.session as session_mod
    from ui.state.session import SessionManager

    monkeypatch.setattr(session_mod, "HISTORY_DIR", str(tmp_path))
    os.makedirs(tmp_path, exist_ok=True)
    mgr = SessionManager()

    class StubOrchestrator:
        def route(self, message, conversation_id=None, run_id=None):
            return {
                "agent": "normal",
                "agent_icon": "",
                "agent_name": "Assistant STATOUR",
                "response": f"{conversation_id}:{message}",
                "rerouted": False,
                "classification_time_ms": 0.0,
                "total_time_ms": 1.0,
                "chart_paths": [],
                "trace": [],
                "fallbacks": [],
                "errors": [],
            }

    monkeypatch.setattr(server, "_get_mgr", lambda: mgr)
    monkeypatch.setattr(server, "_get_orch", lambda: StubOrchestrator())

    with server.app.test_request_context("/api/chat", method="POST"):
        first = server._process_chat_turn("Bonjour", None)
    with server.app.test_request_context("/api/chat", method="POST"):
        second = server._process_chat_turn("Bonjour", None)

    assert first["conversation_id"] != second["conversation_id"]
    assert len(mgr.get_conversation(first["conversation_id"]).messages) == 2
    assert len(mgr.get_conversation(second["conversation_id"]).messages) == 2


def test_orchestrator_routes_different_conversations_concurrently():
    from agents.orchestrator import Orchestrator

    orch = object.__new__(Orchestrator)
    orch._lock = threading.RLock()
    orch._state_context = threading.local()
    orch._conversation_locks = {}
    orch._runtime_states = {}

    def slow_route(message, runtime_state=None, run_id=None):
        time.sleep(0.2)
        orch.last_agent = "normal"
        orch.conversation_log.append(("user", message))
        return {
            "agent": "normal",
            "response": message,
            "total_time_ms": 200.0,
            "trace": [],
        }

    orch._route_internal = slow_route

    started = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: orch.route("Bonjour", conversation_id=f"conv_{i}"), range(4)))
    elapsed = time.time() - started

    assert elapsed < 0.55
    assert set(orch._runtime_states) == {"conv_0", "conv_1", "conv_2", "conv_3"}


def test_client_ip_ignores_spoofed_xff_without_trusted_proxy(monkeypatch):
    import server

    monkeypatch.setattr(server, "TRUSTED_PROXY_IPS", {"127.0.0.1"})
    with server.app.test_request_context(
        "/api/chat",
        environ_base={"REMOTE_ADDR": "203.0.113.10"},
        headers={"X-Forwarded-For": "198.51.100.7"},
    ):
        assert server._client_ip() == "203.0.113.10"

    with server.app.test_request_context(
        "/api/chat",
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
        headers={"X-Forwarded-For": "198.51.100.7"},
    ):
        assert server._client_ip() == "198.51.100.7"


def test_resp_exposes_processing_wall_and_queue_time():
    from server import _resp

    payload = _resp(
        "normal", "", "Assistant", "ok", False, 1.0, 50.0, None, "cid",
        processing_time_ms=10.0,
        wall_time_ms=50.0,
        queue_time_ms=40.0,
    )

    assert payload["total_time_ms"] == 50.0
    assert payload["processing_time_ms"] == 10.0
    assert payload["wall_time_ms"] == 50.0
    assert payload["queue_time_ms"] == 40.0


def test_analytics_followup_forces_tool_path():
    from orchestration.triage import triage

    context = (
        "Memoire conversationnelle:\n"
        "  USER: Compare les nuitees hotelieres par region en janvier-fevrier 2026.\n"
        "  ANALYTICS: Resultats SQL par region et graphique."
    )
    result = triage("Et pour Marrakech seulement ?", context)

    assert result.complexity == "complex"
    assert "sql" in result.tools_needed


def test_tourism_chart_prompt_routes_to_sql_before_chart():
    from orchestration.triage import triage

    result = triage(
        "Analyse la situation touristique de 2018 a 2025 a casablanca "
        "et cree moi un graphique qui montre cette evolution",
        "",
    )

    assert result.complexity == "complex"
    assert "sql" in result.tools_needed
    assert "chart" in result.tools_needed


def test_followup_resolution_preserves_previous_shape():
    from orchestration.followup import resolve_followup

    context = (
        "Memoire conversationnelle:\n"
        "  USER: Donne-moi les top 10 nationalites TES en fevrier 2026.\n"
        "  ANALYTICS: Top 10 TES pour fevrier 2026."
    )
    resolved = resolve_followup("Meme chose pour les MRE.", context)

    assert "Meme chose pour les MRE" in resolved
    assert "top 10 nationalites TES en fevrier 2026" in resolved
    assert "Conserver le meme type de livrable" in resolved


def test_prediction_tool_wins_agent_label_on_mixed_plan():
    from orchestration.graph import _dominant_agent
    from orchestration.triage import TriageResult
    from orchestration.executor import ExecutionResult, Evidence

    execution = ExecutionResult(evidence=[
        Evidence(step_id=1, tool="sql", success=True, text_summary="history"),
        Evidence(step_id=2, tool="prediction", success=True, text_summary="forecast"),
    ])
    triage_result = TriageResult(complexity="complex", intent="prediction", tools_needed=["sql", "prediction"])

    assert _dominant_agent(triage_result, execution) == "prediction"


def test_mars_planet_prompt_is_not_forecasted():
    from orchestration.triage import triage
    from agents.prediction_agent import PredictionAgent

    msg = "Combien de touristes marocains ont visite Mars en 2035 ? Donne une prevision precise."
    triage_result = triage(msg, "")
    assert triage_result.intent == "out_of_domain_clarification"
    assert triage_result.direct_answer

    df = pd.DataFrame({
        "date_stat": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "mre": [100, 200],
        "tes": [50.0, 60.0],
        "voie": ["V Aerienne", "V Maritime"],
        "poste_frontiere": ["A CMN", "P Tanger"],
        "region": ["CK", "TTA"],
        "continent": ["Europe", "Europe"],
        "nationalite": ["France", "France"],
    })
    result = PredictionAgent(df).chat(msg)
    assert result["chart_path"] is None
    assert result["prediction_context"]["blocked_prediction"] is True


def test_language_defaults_to_french_unless_english_is_clear():
    from utils.base_agent import detect_language

    assert detect_language("Et en 2025 ?") == "fr"
    assert detect_language("Bonjour") == "fr"
    assert detect_language("Hello how are you?") == "en"


def test_planner_repairs_chart_only_plan_when_sql_is_needed():
    from orchestration.planner import ExecutionPlan, PlanStep, _sanitize_plan

    plan = ExecutionPlan(steps=[PlanStep(step_id=1, tool="chart", description="Graphique")])
    repaired = _sanitize_plan(
        plan,
        "Analyse les arrivees touristiques de 2018 a 2025 avec un graphique",
        ["sql", "chart"],
    )

    assert repaired.tool_sequence() == ["sql", "chart"]


def test_executor_surfaces_dataframe_when_generated_code_does_not_print():
    from orchestration.executor import Executor

    result = Executor()._execute_sandbox(
        "df = pd.DataFrame({'annee': [2024, 2025], 'valeur': [10, 12]})"
    )

    assert result["error"] is None
    assert "DataFrame df" in result["output"]
    assert "2024" in result["output"]


def test_executor_retries_empty_city_query_using_province_name():
    from orchestration.executor import _SQL_GEN_PROMPT, _should_retry_location_query

    result = {"output": "(Aucune donnée)", "data": None, "error": None}
    code = (
        "df = sql(\"SELECT YEAR(date_stat) AS annee, SUM(arrivees) AS arrivees "
        "FROM [dbo_GOLD].[fact_statistiqueshebergementnationaliteestimees] "
        "WHERE region_name = 'Casablanca' GROUP BY YEAR(date_stat)\")"
    )

    assert "province_name" in _SQL_GEN_PROMPT
    assert _should_retry_location_query(
        "Analyse la situation touristique de 2018 a 2025 a Casablanca",
        code,
        result,
    )
