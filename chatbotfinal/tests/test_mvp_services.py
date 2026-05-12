import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mvp_services import clear_service_cache, dependency_status, get_dashboard_summary, get_readiness
from agents.executive_insight_agent import is_executive_insight_request


def test_dependency_status_shape():
    status = dependency_status()
    assert "dependencies" in status
    assert "missing" in status
    assert "flask" in status["dependencies"]
    assert "pyodbc" in status["dependencies"]


def test_readiness_without_orchestrator_shape():
    clear_service_cache()
    readiness = get_readiness(None)
    assert "ready" in readiness
    assert "fabric" in readiness
    assert "rag" in readiness
    assert "search" in readiness
    assert isinstance(readiness["blockers"], list)


def test_dashboard_fallback_shape():
    clear_service_cache()
    summary = get_dashboard_summary(None)
    assert summary["status"] in {"ok", "degraded"}
    assert isinstance(summary["kpis"], list)
    assert isinstance(summary["signals"], list)
    assert summary["signals"]


def test_executive_intent_detection():
    assert is_executive_insight_request("Pourquoi les arrivées ont augmenté ?")
    assert is_executive_insight_request("Impact d'un conflit régional")
    assert not is_executive_insight_request("bonjour")


def test_readiness_cache_reuses_value(monkeypatch):
    import utils.mvp_services as mvp

    clear_service_cache()
    calls = []

    def fake_compute(orch=None):
        calls.append(1)
        return {
            "ready": True,
            "checked_at": str(len(calls)),
            "fabric": {},
            "latest_data": {},
            "rag": {},
            "search": {},
            "dependency_status": {},
            "blockers": [],
        }

    monkeypatch.setattr(mvp, "_compute_readiness", fake_compute)

    first = mvp.get_readiness(None)
    second = mvp.get_readiness(None)
    refreshed = mvp.get_readiness(None, force_refresh=True)

    assert len(calls) == 2
    assert first == second
    assert refreshed["checked_at"] == "2"
