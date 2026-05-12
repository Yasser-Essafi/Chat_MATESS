import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.contracts import NodeContext, NodeResult
from orchestration.nodes import CommandNode, HumanAdvisorNode, is_human_advisor_request
from orchestration.quality import inspect_deliverables, requested_chart_count
from orchestration.registry import NodeRegistry


class DummyNode:
    key = "dummy"

    def can_handle(self, context):
        return "dummy" in context.message

    def run(self, context):
        return NodeResult(agent="dummy", agent_name="Dummy", response="ok")


def test_node_registry_selects_and_runs_dummy_node():
    registry = NodeRegistry([DummyNode()])

    result = registry.run(NodeContext(message="run dummy"))

    assert registry.keys() == ["dummy"]
    assert result.agent == "dummy"
    assert result.response == "ok"


def test_node_registry_rejects_duplicate_keys():
    registry = NodeRegistry([DummyNode()])

    with pytest.raises(ValueError):
        registry.register(DummyNode())


def test_command_node_routes_analytics_commands():
    class Analytics:
        def list_datasets(self):
            return "dataset list"

    class Orch:
        analytics_agent = Analytics()

        def handle_orchestrator_commands(self, message):
            return None

    result = CommandNode(Orch()).run(NodeContext(message="/datasets"))

    assert result.agent == "analytics"
    assert result.response == "dataset list"
    assert result.trace[0]["stage"] == "command"


def test_quality_gate_detects_missing_requested_charts_and_narrative():
    report = inspect_deliverables(
        "analyse les arrivees avec deux graphiques et comparaison",
        "trop court",
        [],
    )

    assert requested_chart_count("deux graphiques") == 2
    assert any("graphique" in item for item in report.missing)
    assert "analyse narrative structuree" in report.missing


def test_human_advisor_handles_opinion_recommendation_without_chart():
    message = "tu ma rien recommender ? tu pense quoi du tourisme au maroc ?"

    result = HumanAdvisorNode(None).run(NodeContext(message=message))

    assert is_human_advisor_request(message)
    assert result.agent == "human_advisor"
    assert result.agent_name == "Conseiller Tourisme"
    assert result.chart_paths == []
    assert "Tu as raison" in result.response
    assert "Chiffres cles" not in result.response
    assert "Ce que je recommanderais" in result.response
