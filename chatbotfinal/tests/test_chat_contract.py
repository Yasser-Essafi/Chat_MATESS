import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ui.state.session import Conversation, Message
from agents.orchestrator import ConversationRuntimeState, Orchestrator
from orchestration.nodes import build_default_registry
from server import _CHARTS_DIR, _resp, extract_chart_paths


def test_message_metadata_and_chart_urls(tmp_path):
    chart = tmp_path / "chart_unit.html"
    chart.write_text("<html></html>", encoding="utf-8")
    msg = Message(
        role="assistant",
        content="ok",
        chart_paths=[str(chart)],
        run_id="run123",
        trace=[{"stage": "unit"}],
    )
    data = msg.to_dict()

    assert data["message_id"]
    assert data["run_id"] == "run123"
    assert data["chart_path"] == str(chart)
    assert data["chart_urls"] == ["/charts/chart_unit.html"]
    assert data["trace"] == [{"stage": "unit"}]


def test_conversation_fork_from_turn_helpers():
    conv = Conversation()
    u1 = Message(role="user", content="first")
    a1 = Message(role="assistant", content="answer")
    u2 = Message(role="user", content="second")
    conv.add_message(u1)
    conv.add_message(a1)
    conv.add_message(u2)

    assert conv.related_user_for_message(a1.message_id).message_id == u1.message_id
    conv.update_user_message(u1.message_id, "edited")
    conv.truncate_after_message(u1.message_id)

    assert [m.content for m in conv.messages] == ["edited"]
    assert conv.title == "edited"


def test_resp_keeps_single_and_multi_chart_contracts():
    payload = _resp(
        "analytics",
        "icon",
        "Agent",
        "body",
        False,
        1,
        2,
        "/charts/a.html",
        "cid",
        chart_urls=["/charts/a.html", "/charts/b.html"],
        run_id="run1",
        trace=[{"stage": "x"}],
        fallbacks=[{"stage": "fallback"}],
        errors=[],
    )

    assert payload["chart_url"] == "/charts/a.html"
    assert payload["chart_urls"] == ["/charts/a.html", "/charts/b.html"]
    assert payload["run_id"] == "run1"
    assert payload["trace"][0]["stage"] == "x"


def test_extract_chart_paths_is_bounded_to_charts_dir():
    os.makedirs(_CHARTS_DIR, exist_ok=True)
    path1 = os.path.join(_CHARTS_DIR, "unit_chart_1.html")
    path2 = os.path.join(_CHARTS_DIR, "unit_chart_2.html")
    try:
        for path in [path1, path2]:
            with open(path, "w", encoding="utf-8") as f:
                f.write("<html></html>")
        text = f"Chart: {path1}\nChart: {path2}"
        assert extract_chart_paths(text) == [os.path.realpath(path1), os.path.realpath(path2)]
    finally:
        for path in [path1, path2]:
            if os.path.exists(path):
                os.remove(path)


def test_orchestrator_deliverable_detection_without_init():
    orch = object.__new__(Orchestrator)

    assert orch._requested_chart_count("genere plusieurs graphiques") == 2
    missing = orch._missing_deliverables(
        "analyse les arrivees avec deux graphiques",
        "Une phrase courte.",
        [],
    )

    assert any("graphique" in item for item in missing)
    assert "analyse narrative structuree" in missing


def test_orchestrator_routes_conversational_advice_to_human_node():
    orch = object.__new__(Orchestrator)
    orch.last_agent = "analytics"

    assert orch._classify_instant("tu ma rien recommender ? tu pense quoi du tourisme au maroc ?") == "human_advisor"


def test_orchestrator_human_advisor_turn_has_no_chart_artifacts():
    orch = object.__new__(Orchestrator)
    orch.message_count = 0
    orch.last_agent = "analytics"
    orch._active_domain = None
    orch.conversation_log = []
    orch.routing_history = []
    orch.analytics_agent = type("Analytics", (), {"last_chart_paths": []})()
    orch.node_registry = build_default_registry(orch)

    result = orch._route_internal(
        "tu ma rien recommender ? tu pense quoi du tourisme au maroc ?",
        runtime_state=ConversationRuntimeState("_test"),
        run_id="run_test",
    )

    assert result["agent"] == "human_advisor"
    assert result["chart_paths"] == []
    assert result["chart_path"] is None
    assert result["metric_context"] == "advisory"
    assert "Chiffres cles" not in result["response"]
