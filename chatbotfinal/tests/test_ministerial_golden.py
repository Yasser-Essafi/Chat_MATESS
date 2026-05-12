import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_analytics_agent import DataAnalyticsAgent, execute_code_safe
from agents.executive_insight_agent import ExecutiveInsightAgent
from agents.prediction_agent import PredictionAgent


class FakeKpiCache:
    def total_for_year(self, year):
        return {2025: 19_792_604}.get(year)

    def get(self, key, default=None):
        values = {
            "mre_by_year": {2025: 9_000_000},
            "tes_by_year": {2025: 10_792_604},
        }
        return values.get(key, default)


class FakeFabricDb:
    source = "fabric"

    def _qualify(self, table_name):
        return table_name

    def safe_query(self, sql):
        sql_lower = sql.lower()
        if "province_name" in sql_lower and "casablanca" in sql_lower and "marrakech" in sql_lower:
            return pd.DataFrame(
                [
                    {"annee": 2023, "ville": "Casablanca", "arrivees": 900_000, "nuitees": 1_800_000},
                    {"annee": 2024, "ville": "Casablanca", "arrivees": 1_000_000, "nuitees": 2_000_000},
                    {"annee": 2025, "ville": "Casablanca", "arrivees": 1_080_000, "nuitees": 2_160_000},
                    {"annee": 2025, "ville": "Marrakech", "arrivees": 2_700_000, "nuitees": 6_500_000},
                    {"annee": 2025, "ville": "Tanger", "arrivees": 1_250_000, "nuitees": 2_300_000},
                ]
            )
        if "arrivees_hotelieres" in sql_lower and "group by year" in sql_lower:
            return pd.DataFrame(
                {
                    "annee": [2024, 2025],
                    "arrivees_hotelieres": [1_493_086, 1_571_444],
                    "nuitees": [3_646_346, 3_831_440],
                }
            )
        if "sum(arrivees) as arrivees_hotelieres" in sql_lower and "sum(nuitees)" in sql_lower:
            return pd.DataFrame([{"arrivees_hotelieres": 20_260_175, "nuitees": 44_908_924}])
        if "count(distinct month" in sql_lower:
            return pd.DataFrame([{"annee": 2025, "mois_count": 12}])
        if "sum(nuitees) as nuitees" in sql_lower and "sum(arrivees) as arrivees" in sql_lower:
            return pd.DataFrame([{"nuitees": 44_908_924, "arrivees": 20_260_175}])
        if "group by region_name" in sql_lower:
            regions = ["Marrakech-Safi"] + [f"Region {i}" for i in range(2, 13)]
            return pd.DataFrame({"Region": regions, "Nuitees": [19_000_000] + list(range(11_000, 0, -1000))})
        if "group by month" in sql_lower:
            return pd.DataFrame({"mois": list(range(1, 13)), "arrivees": [1_000_000 + i for i in range(12)]})
        if "fact_statistiques_apf" in sql_lower and "group by year" in sql_lower:
            return pd.DataFrame({"annee": [2024, 2025], "arrivees": [2_555_082, 2_697_017]})
        raise AssertionError(f"Unexpected SQL: {sql}")


def make_analytics_agent():
    agent = DataAnalyticsAgent.__new__(DataAnalyticsAgent)
    agent._db = FakeFabricDb()
    agent.kpi_cache = FakeKpiCache()
    agent.chart_count = 0
    agent.last_chart_paths = []
    agent._cleanup_old_charts = lambda: None
    agent._write_monthly_chart = lambda df, title, x_col, y_col: os.path.join("charts", "golden.html")
    return agent


def test_apf_arrivals_2025_is_apf_only():
    response = make_analytics_agent().try_official_kpi_answer("Combien d'arrivees APF en 2025 ?")

    assert "19,792,604" in response
    assert "APF" in response
    assert "hebergement" not in response.lower()


def test_hotel_arrivals_2025_is_hebergement_only():
    response = make_analytics_agent().try_official_kpi_answer("Combien d'arrivees hotelieres en 2025 ?")

    assert "20,260,175" in response
    assert "hebergement" in response.lower()
    assert "MRE" not in response
    assert "TES" not in response
    assert "frontiere" not in response.lower()


def test_nuitees_by_region_2024_returns_twelve_regions_and_marrakech_first():
    response = make_analytics_agent().try_official_kpi_answer("Donne-moi les nuitees par region en 2024")

    assert response.count("| Marrakech-Safi |") == 1
    assert response.index("Marrakech-Safi") < response.index("Region 2")
    data_rows = [line for line in response.splitlines() if line.startswith("| ") and not line.startswith("| :")]
    assert len(data_rows) == 13  # header + 12 regions


def test_monthly_arrivals_chart_sets_contract_artifact_path():
    agent = make_analytics_agent()
    response = agent.try_official_kpi_answer("Fais un graphique des arrivees par mois en 2025")

    assert "Graphique" in response
    assert "Chart:" not in response
    assert agent.last_chart_paths == [os.path.join("charts", "golden.html")]


def test_casablanca_city_prompt_returns_two_contract_charts(monkeypatch):
    import agents.data_analytics_agent as analytics_module

    monkeypatch.setattr(analytics_module, "chart_line", lambda *args, **kwargs: object())
    monkeypatch.setattr(analytics_module, "chart_bar", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        analytics_module,
        "chart_save",
        lambda fig, prefix="chart", **kwargs: os.path.join("charts", f"{prefix}.html"),
    )
    agent = make_analytics_agent()

    response = agent.try_official_kpi_answer(
        "analyse l'etat du tourisme a casablanca les 3 dernieres annee, "
        "cree un graph devolution et un graph de comparaison avec dautre ville"
    )

    assert "Synthese executive" in response
    assert "Decisions" in response
    assert len(agent.last_chart_paths) == 2
    assert all(path.endswith(".html") for path in agent.last_chart_paths)


def test_dms_calculates_nuitees_over_hotel_arrivals():
    response = make_analytics_agent().try_official_kpi_answer("Quelle est la DMS ?")

    assert "2.22" in response
    assert "44,908,924 nuitees" in response
    assert "20,260,175 arrivees hotelieres" in response
    assert "APF" in response


def test_prediction_uses_current_question_not_injected_history(tmp_path):
    rows = []
    for year, total in [(2023, 12_000_000), (2024, 15_000_000), (2025, 19_792_604)]:
        for month in range(1, 13):
            rows.append({"date_stat_year": year, "date_stat_month": month, "total": total / 12})
    agent = PredictionAgent(pd.DataFrame(rows), charts_dir=str(tmp_path))
    agent._searcher = None
    agent._llm = None

    result = agent.chat(
        "Estime les touristes en 2030 scenario optimiste\n\n"
        "[Historique conversationnel]\n"
        "ancienne reponse: pandemie, voie maritime, fevrier et juillet"
    )

    response = result["response"].lower()
    assert result["prediction_context"]["target_year"] == 2030
    assert result["prediction_context"]["scenario"] == "optimiste"
    assert result["prediction_context"]["voie"] is None
    assert "pandemie" not in response
    assert "maritime" not in response
    assert "fev" not in response


def test_prediction_target_year_from_compound_prompt(tmp_path):
    rows = []
    for year, total in [(2023, 12_000_000), (2024, 15_000_000), (2025, 19_792_604)]:
        for month in range(1, 13):
            rows.append({"date_stat_year": year, "date_stat_month": month, "total": total / 12})
    agent = PredictionAgent(pd.DataFrame(rows), charts_dir=str(tmp_path))
    agent._searcher = None
    agent._llm = None

    prompt = (
        "Analyse le tourisme de 2018 jusqua 2025 puis cree une prediction "
        "pour l'annee 2026"
    )

    assert agent._extract_target_year(prompt) == 2026


def test_prediction_history_context_only_for_short_followups(tmp_path):
    rows = []
    for year, total in [(2023, 12_000_000), (2024, 15_000_000), (2025, 19_792_604)]:
        for month in range(1, 13):
            rows.append({"date_stat_year": year, "date_stat_month": month, "total": total / 12})
    agent = PredictionAgent(pd.DataFrame(rows), charts_dir=str(tmp_path))
    history = "USER: prevision 2027 voie aerienne\nPREDICTION: projection voie aerienne"

    long_prompt = "Analyse 2019-2025 puis prediction pour 2026"
    short_prompt = "Et pour 2028 ?"

    assert agent._resolve_implicit_refs(long_prompt, history) == long_prompt
    assert "voie aerienne" in agent._resolve_implicit_refs(short_prompt, history)


def test_decline_precheck_corrects_false_july_decline():
    agent = ExecutiveInsightAgent.__new__(ExecutiveInsightAgent)
    agent.analytics_agent = type("Analytics", (), {"_db": FakeFabricDb()})()
    agent.agent_key = "executive_insight"
    agent.agent_icon = "pin"
    agent.agent_name = "Executive"

    result = agent._decline_precheck("Pourquoi les arrivees ont baisse en juillet 2025 ?")

    assert result is not None
    assert "pas de baisse mesuree" in result["response"].lower()
    assert "Arrivees APF" in result["response"]
    assert "Arrivees hotelieres" in result["response"]


def test_sandbox_percent_lambda_sees_computed_total():
    code = """
df = pd.DataFrame({"Ville": ["A", "B"], "Nuitees": [80, 20]})
total_nuitees = df["Nuitees"].sum()
df["Part (%)"] = df["Nuitees"].apply(lambda x: x / total_nuitees * 100)
print(df["Part (%)"].round(1).tolist())
"""

    result = execute_code_safe(code, {"pd": pd}, timeout_seconds=5)

    assert result["error"] is None
    assert "[80.0, 20.0]" in result["output"]
