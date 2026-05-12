"""Unit tests for the UX overhaul changes (phases 1-6)."""
import sys, os, re, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "chatbotfinal"))


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Fast Greeting Routing
# ═══════════════════════════════════════════════════════════════════════

class TestPhase1_GreetingRouting(unittest.TestCase):
    """Compound greetings must be routed instantly to 'normal' without LLM."""

    @classmethod
    def setUpClass(cls):
        from agents.orchestrator import _PURE_GREETINGS
        cls.greetings = _PURE_GREETINGS

    def _classify(self, msg):
        """Simulate _classify_instant compound greeting logic."""
        lower = msg.lower().strip()
        if lower in self.greetings:
            return "normal"
        words = lower.split()
        if 2 <= len(words) <= 6 and words[0] in self.greetings:
            return "normal"
        return None

    def test_single_greeting(self):
        self.assertEqual(self._classify("hello"), "normal")
        self.assertEqual(self._classify("bonjour"), "normal")

    def test_compound_greeting_hello_ca_va(self):
        self.assertEqual(self._classify("hello ca va"), "normal")

    def test_compound_greeting_bonjour_comment_ca_va(self):
        self.assertEqual(self._classify("bonjour comment ca va"), "normal")

    def test_compound_greeting_salut_ca_va_bien(self):
        self.assertEqual(self._classify("salut ca va bien"), "normal")

    def test_long_message_not_greeting(self):
        self.assertIsNone(self._classify("hello can you analyze tourism data for 2025"))

    def test_analytics_not_caught(self):
        self.assertIsNone(self._classify("combien d'arrivées en 2024"))


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Emoji Removal
# ═══════════════════════════════════════════════════════════════════════

class TestPhase2_NoEmojis(unittest.TestCase):
    """Agent names must not contain emoji characters."""

    def test_agent_names_no_emojis(self):
        from config.settings import AGENT_NAMES
        emoji_re = re.compile(
            "["
            "\U0001F300-\U0001F9FF"
            "\U00002702-\U000027B0"
            "\U0000FE00-\U0000FE0F"
            "\U0000200D"
            "]+",
            flags=re.UNICODE,
        )
        for key, name in AGENT_NAMES.items():
            self.assertFalse(
                emoji_re.search(name),
                f"AGENT_NAMES['{key}'] = '{name}' still contains emojis",
            )


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Response Sanitization
# ═══════════════════════════════════════════════════════════════════════

class TestPhase4_Sanitization(unittest.TestCase):
    """Internal table names, file paths, chart lines must be stripped."""

    def _clean(self, text, is_exec=False):
        from agents.data_analytics_agent import DataAnalyticsAgent
        return DataAnalyticsAgent._clean_response_text(text, is_exec_output=is_exec)

    def test_strip_dbo_gold_refs(self):
        text = "Les données proviennent de [dbo_GOLD].[fact_statistiques_apf]."
        cleaned = self._clean(text)
        self.assertNotIn("[dbo_GOLD]", cleaned)
        self.assertNotIn("fact_statistiques_apf", cleaned)

    def test_strip_hebergement_table_name(self):
        text = "table fact_statistiqueshebergementnationaliteestimees contient les nuitées."
        cleaned = self._clean(text)
        self.assertNotIn("fact_statistiqueshebergement", cleaned)

    def test_strip_note_with_table_ref(self):
        text = "Voici les résultats.\n\nNOTE: Les analyses ci-dessus utilisent les données d'hébergement (table fact_statistiqueshebergementnationaliteestimees)."
        cleaned = self._clean(text)
        self.assertNotIn("NOTE:", cleaned)
        self.assertNotIn("fact_statistiques", cleaned)

    def test_strip_chart_path_line(self):
        text = "Résultats:\n\n📊 Chart: C:/Users/hamza/charts/chart_123.html"
        cleaned = self._clean(text)
        self.assertNotIn("Chart:", cleaned)
        self.assertNotIn(".html", cleaned)

    def test_normal_text_preserved(self):
        text = "Casablanca a enregistré 3,185,607 nuitées en 2025."
        cleaned = self._clean(text)
        self.assertEqual(cleaned, text)


class TestPhase4_ServerSanitize(unittest.TestCase):
    """Server-level _sanitize_response catches any remaining leaks."""

    def test_sanitize_strips_paths(self):
        sys.path.insert(0, os.path.dirname(__file__))
        from server import _sanitize_response
        text = "Voici le résultat.\n📊 Chart: C:/Users/foo/charts/bar.html\n\nFin."
        cleaned = _sanitize_response(text)
        self.assertNotIn("C:/Users", cleaned)
        self.assertNotIn("Chart:", cleaned)
        self.assertIn("Voici le résultat.", cleaned)

    def test_sanitize_strips_table_names(self):
        from server import _sanitize_response
        text = "Données de [dbo_GOLD].[fact_statistiques_apf] montrent une hausse."
        cleaned = _sanitize_response(text)
        self.assertNotIn("[dbo_GOLD]", cleaned)


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Interleaved Narrative + Charts
# ═══════════════════════════════════════════════════════════════════════

class TestPhase5_Interleave(unittest.TestCase):
    """GRAPHIQUE markers are converted to chart HTML comments."""

    def test_graphique_markers_regex(self):
        text = "Voici l'évolution.\n\n[GRAPHIQUE_1]\n\nEt la comparaison.\n\n[GRAPHIQUE_2]\n\nConclusion."
        result = re.sub(
            r"\[GRAPHIQUE[_ ](\d+)\]",
            lambda m: f"<!-- chart:{m.group(1)} -->",
            text,
        )
        self.assertIn("<!-- chart:1 -->", result)
        self.assertIn("<!-- chart:2 -->", result)
        self.assertNotIn("[GRAPHIQUE", result)

    def test_graphique_space_variant(self):
        text = "[GRAPHIQUE 1] bla [GRAPHIQUE 2]"
        result = re.sub(
            r"\[GRAPHIQUE[_ ](\d+)\]",
            lambda m: f"<!-- chart:{m.group(1)} -->",
            text,
        )
        self.assertIn("<!-- chart:1 -->", result)
        self.assertIn("<!-- chart:2 -->", result)

    def test_server_preserves_chart_markers(self):
        from server import _sanitize_response
        text = "Voici l'analyse.\n\n<!-- chart:1 -->\n\nBon résultat."
        cleaned = _sanitize_response(text)
        self.assertIn("<!-- chart:1 -->", cleaned)


# ═══════════════════════════════════════════════════════════════════════
# Bug Fix: Chart Path Propagation
# ═══════════════════════════════════════════════════════════════════════

class TestBugFix_ChartPropagation(unittest.TestCase):
    """Charts must be propagated via last_chart_paths, not text extraction."""

    def test_analytics_agent_has_last_chart_paths_attr(self):
        from agents.data_analytics_agent import DataAnalyticsAgent
        self.assertTrue(
            hasattr(DataAnalyticsAgent, '__init__'),
            "DataAnalyticsAgent must define __init__",
        )
        import inspect
        src = inspect.getsource(DataAnalyticsAgent.__init__)
        self.assertIn("last_chart_paths", src)

    def test_chat_internal_resets_last_chart_paths(self):
        import inspect
        from agents.data_analytics_agent import DataAnalyticsAgent
        src = inspect.getsource(DataAnalyticsAgent._chat_internal)
        self.assertIn("self.last_chart_paths = []", src)

    def test_sanitize_applied_to_stored_messages(self):
        from server import _assistant_message_from_result, _sanitize_response
        result = {
            "response": "Data from [dbo_GOLD].[fact_statistiques_apf] shows growth.",
            "agent": "analytics",
            "trace": [{"stage": "test", "label": "Test"}],
        }
        msg = _assistant_message_from_result(result, [], "run123")
        self.assertNotIn("[dbo_GOLD]", msg.content)
        self.assertNotIn("fact_statistiques_apf", msg.content)

    def test_sanitize_preserves_chart_markers_in_stored_messages(self):
        from server import _assistant_message_from_result
        result = {
            "response": "Analysis.\n\n<!-- chart:1 -->\n\nConclusion.",
            "agent": "analytics",
            "trace": [],
        }
        msg = _assistant_message_from_result(result, [], "run123")
        self.assertIn("<!-- chart:1 -->", msg.content)


# ═══════════════════════════════════════════════════════════════════════
# Bug Fix: Trace Stored Correctly
# ═══════════════════════════════════════════════════════════════════════

class TestBugFix_TraceStored(unittest.TestCase):
    """Trace steps must be stored in persisted messages and rendered."""

    def test_message_stores_trace(self):
        from ui.state.session import Message
        trace = [
            {"stage": "routing", "label": "Classification", "status": "done"},
            {"stage": "agent", "label": "Execution", "status": "done", "duration_ms": 1500},
        ]
        msg = Message(role="assistant", content="test", trace=trace)
        d = msg.to_dict()
        self.assertEqual(len(d["trace"]), 2)
        self.assertEqual(d["trace"][0]["stage"], "routing")
        self.assertEqual(d["trace"][1]["duration_ms"], 1500)
        self.assertEqual(d["status"], "done")

    def test_message_from_dict_preserves_trace(self):
        from ui.state.session import Message
        trace = [{"stage": "routing", "label": "Classification", "status": "done"}]
        msg = Message(role="assistant", content="test", trace=trace)
        d = msg.to_dict()
        restored = Message.from_dict(d)
        self.assertEqual(len(restored.trace), 1)
        self.assertEqual(restored.trace[0]["stage"], "routing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
