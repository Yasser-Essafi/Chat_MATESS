"""
Tests for follow-up flow bug fixes.

Covers:
  Bug 1 / 8 / 9: Prediction routing — accent normalization, ordering, Layer 2 override
  Bug 2: Prediction context passing
  Bug 3: Analytics fast-path history update
  Bug 4: Classifier cache includes conversation_id
  Bug 5: Conversation log truncation increased to 800 chars
  Bug 7: Dead code removed from server.py
"""

import os
import sys
import re
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.orchestrator import (
    _norm_text,
    _has_prediction_intent,
    _requires_planned_flow,
    _PREDICTION_KEYWORDS,
    _PREDICTION_KEYWORDS_NORM,
)


# ═══════════════════════════════════════════════════════════════════
# Bug 9: Accent normalization for prediction keywords
# ═══════════════════════════════════════════════════════════════════

class TestAccentNormalization:
    def test_norm_text_strips_accents(self):
        assert _norm_text("Prévisions") == "previsions"
        assert _norm_text("Estimé") == "estime"
        assert _norm_text("scénario") == "scenario"

    def test_prediction_keywords_norm_generated(self):
        assert len(_PREDICTION_KEYWORDS_NORM) == len(_PREDICTION_KEYWORDS)
        assert "prevision" in _PREDICTION_KEYWORDS_NORM
        assert "prevoir" in _PREDICTION_KEYWORDS_NORM
        assert "scenario" in _PREDICTION_KEYWORDS_NORM

    def test_has_prediction_intent_with_accents(self):
        assert _has_prediction_intent("prévision du tourisme", "prevision du tourisme")

    def test_has_prediction_intent_without_accents(self):
        """User types 'previsions' without accent — should still match."""
        msg = "previsions des arrivees en 2027"
        norm = _norm_text(msg)
        assert _has_prediction_intent(msg, norm)

    def test_has_prediction_intent_mixed_case(self):
        msg = "Prévision des arrivées"
        norm = _norm_text(msg)
        assert _has_prediction_intent(msg.lower(), norm)

    def test_no_false_positive(self):
        msg = "combien de touristes en 2024"
        norm = _norm_text(msg)
        assert not _has_prediction_intent(msg.lower(), norm)


# ═══════════════════════════════════════════════════════════════════
# Bug 1 + 8: Prediction routing order — prediction keywords checked
# before years-outside-range, follow-ups pinned before years check
# ═══════════════════════════════════════════════════════════════════

class TestInstantClassifierOrder:
    """Test that _classify_instant routes prediction correctly."""

    def _make_orchestrator_stub(self):
        """Create a minimal orchestrator stub for unit-testing classify_instant."""
        from agents.orchestrator import Orchestrator

        with patch.object(Orchestrator, "__init__", lambda self: None):
            orch = Orchestrator()
        orch.last_agent = None
        orch.message_count = 0
        orch.min_year = 2019
        orch.max_year = 2026
        orch.conversation_log = []
        orch._active_domain = None
        orch._session_summary = ""
        return orch

    def test_prediction_keyword_beats_future_year(self):
        """'prévisions 2027' should route to prediction, not researcher."""
        orch = self._make_orchestrator_stub()
        result = orch._classify_instant("Prévisions des arrivées en 2027")
        assert result == "prediction"

    def test_prediction_keyword_without_accent_beats_future_year(self):
        """'previsions 2028' without accents should still route to prediction."""
        orch = self._make_orchestrator_stub()
        result = orch._classify_instant("previsions des arrivees en 2028")
        assert result == "prediction"

    def test_bare_future_year_goes_to_researcher(self):
        """'tourisme en 2030' without prediction keywords → researcher."""
        orch = self._make_orchestrator_stub()
        result = orch._classify_instant("tourisme au Maroc en 2030")
        assert result == "researcher"

    def test_followup_after_prediction_stays_on_prediction(self):
        """'Et pour 2028?' after prediction turn should stay on prediction."""
        orch = self._make_orchestrator_stub()
        orch.last_agent = "prediction"
        orch.message_count = 1
        result = orch._classify_instant("Et pour 2028 ?")
        assert result == "prediction"

    def test_followup_oui_after_prediction(self):
        """'oui' after prediction stays on prediction."""
        orch = self._make_orchestrator_stub()
        orch.last_agent = "prediction"
        orch.message_count = 1
        result = orch._classify_instant("oui")
        assert result == "prediction"

    def test_compound_analysis_prediction_uses_planned_flow(self):
        """Historical analysis + forecast should not be collapsed to prediction."""
        message = (
            "Analyse le tourisme de 2018 jusqua 2025 puis cree une prediction "
            "pour l'annee 2026, avec un graphique"
        )
        orch = self._make_orchestrator_stub()

        assert _requires_planned_flow(message)
        assert orch._classify_instant(message) is None

    def test_pure_prediction_still_routes_prediction(self):
        orch = self._make_orchestrator_stub()

        assert not _requires_planned_flow("prevision des arrivees en 2028 avec graphique")
        assert orch._classify_instant("prevision des arrivees en 2028 avec graphique") == "prediction"


# ═══════════════════════════════════════════════════════════════════
# Bug 1 (Layer 2): LLM override with prediction intent
# ═══════════════════════════════════════════════════════════════════

class TestLayer2Override:
    def _make_orchestrator_stub(self):
        from agents.orchestrator import Orchestrator

        with patch.object(Orchestrator, "__init__", lambda self: None):
            orch = Orchestrator()
        orch.last_agent = None
        orch.message_count = 0
        orch.min_year = 2019
        orch.max_year = 2026
        orch.conversation_log = []
        orch._active_domain = None
        orch._session_summary = ""
        orch._current_cid = "_test"

        from utils.cache import SearchCache
        orch._classify_cache = SearchCache(max_size=10, ttl_seconds=60)
        return orch

    def test_llm_analytics_with_prediction_keywords_overrides_to_prediction(self):
        """If LLM says analytics but message has prediction keywords + future year → prediction."""
        orch = self._make_orchestrator_stub()

        with patch.object(orch, "_classify_instant", return_value=None), \
             patch.object(orch, "_classify_llm", return_value="analytics"), \
             patch.object(orch, "_years_in_message", return_value=[2028]), \
             patch.object(orch, "_all_years_outside_range", return_value=True):
            result = orch.classify("estimation des arrivées en 2028")
            assert result == "prediction"

    def test_llm_analytics_without_prediction_keywords_overrides_to_researcher(self):
        """If LLM says analytics but message has future year + no prediction keywords → researcher."""
        orch = self._make_orchestrator_stub()

        with patch.object(orch, "_classify_instant", return_value=None), \
             patch.object(orch, "_classify_llm", return_value="analytics"), \
             patch.object(orch, "_years_in_message", return_value=[2030]), \
             patch.object(orch, "_all_years_outside_range", return_value=True):
            result = orch.classify("tourisme au Maroc en 2030")
            assert result == "researcher"


# ═══════════════════════════════════════════════════════════════════
# Bug 2: Prediction agent receives conversation context
# ═══════════════════════════════════════════════════════════════════

class TestPredictionContextPassing:
    def test_chat_strips_history_but_resolves_refs(self):
        """chat() strips [Historique ...] but uses it for implicit reference resolution."""
        from agents.prediction_agent import PredictionAgent
        import pandas as pd

        df = pd.DataFrame({
            "date_stat": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "mre": [100, 200],
            "tes": [50.0, 60.0],
            "voie": ["V Aérienne", "V Maritime"],
            "poste_frontiere": ["A CMN", "P Tanger"],
            "region": ["CK", "TTA"],
            "continent": ["Europe", "Europe"],
            "nationalite": ["France", "France"],
        })

        agent = PredictionAgent(df)

        msg_with_ctx = (
            "Et pour 2028 ?\n\n"
            "[Historique récent:\n"
            "  USER: Prévisions 2027 voie aérienne\n"
            "  PREDICTION: 15M arrivées estimées]"
        )

        result = agent._resolve_implicit_refs(
            "Et pour 2028 ?",
            "récent:\n  USER: Prévisions 2027 voie aérienne\n  PREDICTION: 15M arrivées estimées]"
        )
        assert "aérien" in result.lower() or "voie" in result.lower()

    def test_resolve_no_history(self):
        """No history block → message returned unchanged."""
        from agents.prediction_agent import PredictionAgent
        import pandas as pd

        df = pd.DataFrame({
            "date_stat": pd.to_datetime(["2024-01-01"]),
            "mre": [100], "tes": [50.0],
            "voie": ["V Aérienne"], "poste_frontiere": ["A CMN"],
            "region": ["CK"], "continent": ["Europe"],
            "nationalite": ["France"],
        })
        agent = PredictionAgent(df)
        result = agent._resolve_implicit_refs("Prévisions 2027", "")
        assert result == "Prévisions 2027"


# ═══════════════════════════════════════════════════════════════════
# Bug 3: Analytics fast-path updates conversation_history
# ═══════════════════════════════════════════════════════════════════

class TestAnalyticsFastPathHistory:
    def test_official_kpi_fast_path_updates_history(self):
        """When try_official_kpi_answer returns, history should be updated."""
        from agents.data_analytics_agent import DataAnalyticsAgent

        with patch.object(DataAnalyticsAgent, "__init__", lambda self: None):
            agent = DataAnalyticsAgent()
        agent.datasets = {"test": {}}
        agent.last_chart_paths = []
        agent.conversation_history = [{"role": "system", "content": "sys"}]
        agent.kpi_cache = None

        with patch.object(agent, "try_official_kpi_answer", return_value="Fast answer: 10M"):
            result = agent._chat_internal("test question")

        assert result == "Fast answer: 10M"
        assert len(agent.conversation_history) == 3
        assert agent.conversation_history[1]["role"] == "user"
        assert agent.conversation_history[1]["content"] == "test question"
        assert agent.conversation_history[2]["role"] == "assistant"
        assert agent.conversation_history[2]["content"] == "Fast answer: 10M"

    def test_kpi_cache_fast_path_updates_history(self):
        """When kpi_cache.try_answer returns, history should be updated."""
        from agents.data_analytics_agent import DataAnalyticsAgent

        with patch.object(DataAnalyticsAgent, "__init__", lambda self: None):
            agent = DataAnalyticsAgent()
        agent.datasets = {"test": {}}
        agent.last_chart_paths = []
        agent.conversation_history = [{"role": "system", "content": "sys"}]

        mock_cache = MagicMock()
        mock_cache.try_answer.return_value = "Cached: 5M arrivals"
        agent.kpi_cache = mock_cache

        with patch.object(agent, "try_official_kpi_answer", return_value=None):
            result = agent._chat_internal("cached question")

        assert result == "Cached: 5M arrivals"
        assert len(agent.conversation_history) == 3
        assert agent.conversation_history[2]["content"] == "Cached: 5M arrivals"


# ═══════════════════════════════════════════════════════════════════
# Bug 4: Classifier cache key includes conversation_id
# ═══════════════════════════════════════════════════════════════════

class TestClassifierCacheKey:
    def test_cache_key_includes_cid(self):
        """Cache key should include conversation_id to prevent cross-conv pollution."""
        from agents.orchestrator import Orchestrator
        from utils.cache import SearchCache

        with patch.object(Orchestrator, "__init__", lambda self: None):
            orch = Orchestrator()
        orch.last_agent = "analytics"
        orch._current_cid = "conv_A"
        orch._classify_cache = SearchCache(max_size=10, ttl_seconds=60)
        orch._active_domain = None
        orch._session_summary = ""
        orch.conversation_log = []
        orch.min_year = 2019
        orch.max_year = 2026
        orch.message_count = 0

        with patch.object(orch, "_build_conversation_context", return_value=""):
            pass

        orch._classify_cache.set("conv_A:analytics:Et en 2024 ?", "analytics", source="classify")
        cached_a = orch._classify_cache.get("conv_A:analytics:Et en 2024 ?", source="classify")
        cached_b = orch._classify_cache.get("conv_B:analytics:Et en 2024 ?", source="classify")

        assert cached_a == "analytics"
        assert cached_b is None


# ═══════════════════════════════════════════════════════════════════
# Bug 5: Conversation log truncation is 800 chars
# ═══════════════════════════════════════════════════════════════════

class TestLogTruncation:
    def test_response_truncated_to_800(self):
        """Verify the truncation constant is applied at 800 chars."""
        from agents.orchestrator import Orchestrator

        src_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "orchestrator.py",
        )
        with open(src_file, "r", encoding="utf-8") as f:
            src = f.read()

        assert "response[:800]" in src, "conversation_log truncation should be 800 chars"
        assert "response[:500]" not in src, "old 500-char truncation should be removed"


# ═══════════════════════════════════════════════════════════════════
# Bug 7: Dead code removed from server.py
# ═══════════════════════════════════════════════════════════════════

class TestDeadCodeRemoved:
    def test_no_unreachable_code_in_chat_endpoint(self):
        """server.py chat endpoint should not have unreachable code blocks."""
        server_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "server.py",
        )
        with open(server_path, "r", encoding="utf-8") as f:
            src = f.read()

        chat_section = src.split("def chat():")[1].split("\ndef ")[0]
        try_blocks = [m.start() for m in re.finditer(r"^\s+try:", chat_section, re.MULTILINE)]
        assert len(try_blocks) == 1, (
            f"Expected exactly 1 try block in chat(), found {len(try_blocks)}"
        )

    def test_no_unreachable_code_in_insights_endpoint(self):
        """server.py insights endpoint should not have unreachable code blocks."""
        server_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "server.py",
        )
        with open(server_path, "r", encoding="utf-8") as f:
            src = f.read()

        insights_section = src.split("def executive_insights():")[1].split("\ndef ")[0]
        try_blocks = [m.start() for m in re.finditer(r"^\s+try:", insights_section, re.MULTILINE)]
        assert len(try_blocks) == 1, (
            f"Expected exactly 1 try block in executive_insights(), found {len(try_blocks)}"
        )


# ═══════════════════════════════════════════════════════════════════
# Bug 3 (fallback): Layer 3 fallback uses accent-aware matching
# ═══════════════════════════════════════════════════════════════════

class TestFallbackAccentMatching:
    def test_fallback_matches_unaccented_prediction(self):
        from agents.orchestrator import Orchestrator

        with patch.object(Orchestrator, "__init__", lambda self: None):
            orch = Orchestrator()
        orch.last_agent = None

        result = orch._classify_fallback("previsions tourisme 2028")
        assert result == "prediction"
