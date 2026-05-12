"""
STATOUR Prediction Agent
=========================
Rule-based forecasting for Moroccan tourism flows.

Design principles:
- No ML / ARIMA — transparent, explainable business rules only.
- Uses historical growth rates per voie (aerien/maritime/terrestre/total).
- Applies monthly seasonality coefficients derived from historical data.
- Adjusts for external factors (crises, events) provided by the user or detected from context.
- Returns three scenarios: baseline / optimiste / pessimiste.
- Outputs a Plotly projection chart alongside the numeric estimates.

Usage:
    agent = PredictionAgent(df)
    result = agent.chat("Estime le flux touristique pour 2027")
    # result["response"] → formatted text
    # result["chart_url"] → URL to Plotly chart (if generated)
"""

from __future__ import annotations

import math
import os
import re
import json
import threading
import traceback
import unicodedata
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger("statour.prediction_agent")


def _norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _unsupported_prediction_target(message: str) -> bool:
    norm = _norm_text(message)
    target = r"(?:mars|lune|jupiter|venus|saturne|mercure|neptune|uranus)"
    if re.search(rf"\b(?:sur|vers|a|au|aux)\s+{target}\b", norm):
        return True
    if re.search(rf"\b(?:visit\w*|visite\w*)\b.{{0,40}}\b{target}\b", norm):
        return True
    if re.search(rf"\b{target}\b", norm) and re.search(r"\bplanete|spatial|extra[-\s]?terrestre\b", norm):
        return True
    return False


def _unsupported_prediction_response() -> Dict[str, Any]:
    return {
        "response": (
            "Je ne peux pas produire une prevision touristique fiable pour cette cible, "
            "car elle sort du perimetre des donnees STATOUR. Si vous parlez du mois de "
            "mars, precisez la metrique touristique souhaitee (APF, MRE/TES, nuitees, "
            "region ou pays de residence) et je lancerai l'analyse adaptee."
        ),
        "chart_path": None,
        "agent": AGENT_KEY,
        "agent_name": AGENT_NAME,
        "agent_icon": AGENT_ICON,
        "prediction_context": {"blocked_prediction": True},
    }

# ── Agent identity ──
AGENT_KEY   = "prediction"
AGENT_NAME  = "Prévisionniste STATOUR"
AGENT_ICON  = "🔮"

# ──────────────────────────────────────────────────────────────────────────────
# Business rules (adjustable)
# ──────────────────────────────────────────────────────────────────────────────

# Historical annual growth rates by mode of entry (conservative defaults)
_BASE_GROWTH = {
    "aerien":     0.085,  # +8.5 %/an
    "maritime":   0.050,  # +5.0 %/an
    "terrestre":  0.030,  # +3.0 %/an
    "total":      0.065,  # +6.5 %/an  (overall, when voie breakdown not available)
}

# Scenario multipliers applied on top of base growth
_SCENARIO_MULT = {
    "baseline":   1.00,
    "optimiste":  1.15,
    "pessimiste": 0.82,
}

# External factor adjustments (multiplicative on the year total)
_EXTERNAL_FACTORS: Dict[str, float] = {
    "guerre_regionale":     -0.15,
    "conflit_proche_orient":-0.10,
    "pandemie":             -0.80,
    "crise_economique":     -0.20,
    "recession_europe":     -0.12,
    "evenement_maroc":      +0.12,   # major positive event (CAN, Coupe du Monde…)
    "coupe_du_monde":       +0.18,
    "can":                  +0.08,
    "croissance_europe":    +0.06,
    "reprise_post_covid":   +0.25,
}

# Keywords that map user text to factor keys
_FACTOR_KEYWORDS: Dict[str, List[str]] = {
    "guerre_regionale":      ["guerre", "conflit", "war", "crise sécuritaire"],
    "conflit_proche_orient": ["proche orient", "moyen orient", "israel", "gaza", "liban"],
    "pandemie":              ["pandémie", "pandemic", "covid", "épidémie"],
    "crise_economique":      ["crise économique", "récession", "inflation élevée"],
    "recession_europe":      ["récession europe", "recession europe"],
    "evenement_maroc":       ["événement maroc", "grand événement"],
    "coupe_du_monde":        ["coupe du monde", "mondial", "world cup", "2030"],
    "can":                   ["can 2025", "coupe d'afrique", "can 2026"],
    "croissance_europe":     ["croissance europe", "reprise europe"],
    "reprise_post_covid":    ["reprise post-covid", "post covid", "relance tourisme"],
}

# Default monthly seasonality weights (relative, sum ≠ 1; normalized internally)
# Derived from historical APF data: summer months are heaviest.
_DEFAULT_SEASONALITY = {
    1:  0.055,   # Janvier
    2:  0.058,
    3:  0.072,
    4:  0.078,
    5:  0.082,
    6:  0.085,
    7:  0.120,   # Juillet (pic)
    8:  0.115,   # Août
    9:  0.088,
    10: 0.082,
    11: 0.065,
    12: 0.100,   # Décembre (fêtes + MRE)
}

# Month labels (French)
_MONTHS_FR = {
    1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Aoû", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc",
}

# Long French month names → number — for parsing user questions
# ("février et mai" → [2, 5])
_MONTH_PARSE = {
    "janvier": 1, "january": 1, "jan": 1,
    "février": 2, "fevrier": 2, "february": 2, "feb": 2, "fev": 2,
    "mars": 3, "march": 3, "mar": 3,
    "avril": 4, "april": 4, "avr": 4,
    "mai": 5, "may": 5,
    "juin": 6, "june": 6, "jun": 6,
    "juillet": 7, "july": 7, "juil": 7, "jul": 7,
    "août": 8, "aout": 8, "august": 8, "aug": 8,
    "septembre": 9, "september": 9, "sept": 9, "sep": 9,
    "octobre": 10, "october": 10, "oct": 10,
    "novembre": 11, "november": 11, "nov": 11,
    "décembre": 12, "decembre": 12, "december": 12, "dec": 12,
}

# ──────────────────────────────────────────────────────────────────────────────
# PredictionEngine — pure computation, no I/O
# ──────────────────────────────────────────────────────────────────────────────

class PredictionEngine:
    """
    Stateless forecasting engine.  All methods are pure functions of the
    DataFrame and the request parameters.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._seasonality = self._compute_seasonality(df)
        self._growth_rates = self._compute_growth_rates(df)

    # ── Public API ────────────────────────────────────────────────────────

    def predict(
        self,
        target_year: int,
        scenario: str = "baseline",
        external_factors: Optional[List[str]] = None,
        voie: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Estimate annual tourism arrivals for `target_year`.

        Returns:
            {
                "target_year": int,
                "scenario": str,
                "baseline_total": int,
                "optimiste_total": int,
                "pessimiste_total": int,
                "selected_total": int,
                "monthly_breakdown": {month: arrivals},
                "last_actual_year": int,
                "last_actual_total": int,
                "applied_factors": [str],
                "factor_adjustment": float,   # combined multiplicative adjustment
                "growth_rate_used": float,
                "explanation": str,
            }
        """
        external_factors = external_factors or []

        # Last complete actual data point. APF 2026 is partial in the current
        # build, so annual projections must use 2025 as the base until 2026
        # has 12 reported months.
        last_year, last_total = self._last_actual(voie, complete_only=True)

        # Years to project forward
        n_years = target_year - last_year
        if n_years <= 0:
            # Target is in the past — return actual data
            actual = self._actual_for_year(target_year, voie)
            return {
                "target_year": target_year,
                "scenario": "actuel",
                "baseline_total": actual,
                "optimiste_total": actual,
                "pessimiste_total": actual,
                "selected_total": actual,
                "monthly_breakdown": self._actual_monthly(target_year),
                "last_actual_year": last_year,
                "last_actual_total": last_total,
                "applied_factors": [],
                "factor_adjustment": 1.0,
                "growth_rate_used": 0.0,
                "explanation": f"Données réelles disponibles pour {target_year}.",
            }

        # Growth rate
        growth = self._growth_rates.get(voie or "total", _BASE_GROWTH["total"])

        # External factor adjustments
        factor_adj = 1.0
        applied = []
        for f in external_factors:
            if f in _EXTERNAL_FACTORS:
                factor_adj += _EXTERNAL_FACTORS[f]
                applied.append(f)

        # Compound growth over n_years, then apply factor adjustment
        scenario_mult = _SCENARIO_MULT.get(scenario, 1.0)
        base = last_total * ((1 + growth) ** n_years) * factor_adj
        baseline = int(base)
        optimiste = int(base * _SCENARIO_MULT["optimiste"])
        pessimiste = int(base * _SCENARIO_MULT["pessimiste"])
        selected = int(base * scenario_mult)

        # Monthly breakdown for selected scenario
        monthly = self._monthly_from_annual(selected)

        # Explanation text
        pct_change = round((selected / last_total - 1) * 100, 1) if last_total else 0
        explanation = (
            f"Projection {target_year} (scénario **{scenario}**) basée sur "
            f"**{n_years} an{'s' if n_years > 1 else ''}** de croissance à "
            f"**{growth*100:.1f}%/an** depuis {last_year} ({last_total:,} arrivées). "
            f"Variation estimée vs {last_year}: **{pct_change:+.1f}%**."
        )
        if applied:
            factors_str = ", ".join(applied)
            explanation += f" Facteurs externes appliqués: {factors_str} ({factor_adj-1:+.0%})."

        return {
            "target_year": target_year,
            "scenario": scenario,
            "baseline_total": baseline,
            "optimiste_total": optimiste,
            "pessimiste_total": pessimiste,
            "selected_total": selected,
            "monthly_breakdown": monthly,
            "last_actual_year": last_year,
            "last_actual_total": last_total,
            "applied_factors": applied,
            "factor_adjustment": round(factor_adj, 4),
            "growth_rate_used": growth,
            "explanation": explanation,
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _compute_seasonality(self, df: pd.DataFrame) -> Dict[int, float]:
        """Compute normalized monthly seasonality coefficients from history."""
        if "date_stat_month" not in df.columns or "total" not in df.columns:
            return _DEFAULT_SEASONALITY

        try:
            monthly = df.groupby("date_stat_month")["total"].mean()
            total = monthly.sum()
            if total == 0:
                return _DEFAULT_SEASONALITY
            # Normalize: coefficients sum to 1
            return {int(m): float(v / total) for m, v in monthly.items()}
        except Exception:
            return _DEFAULT_SEASONALITY

    def _compute_growth_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate CAGR per voie from the last 3-4 COMPLETE years of available
        data. Falls back to _BASE_GROWTH when data is insufficient.

        Why exclude the latest partial year: APF data for the current year
        often has only Jan-Feb. Including it as v1 in CAGR collapses the
        growth rate to ~0% (or negative), producing degenerate forecasts
        like "0.0%/an" seen in production.
        """
        rates: Dict[str, float] = dict(_BASE_GROWTH)

        if "date_stat_year" not in df.columns or "total" not in df.columns:
            return rates

        try:
            years = sorted(df["date_stat_year"].dropna().unique().astype(int).tolist())
            if len(years) < 2:
                return rates

            # Detect & drop a partially-reported latest year. We treat a year
            # as "complete" only if it has 12 distinct months in the dataset.
            if "date_stat_month" in df.columns:
                months_per_year = (
                    df.groupby("date_stat_year")["date_stat_month"]
                    .nunique()
                    .to_dict()
                )
                while years and months_per_year.get(years[-1], 0) < 12:
                    years.pop()
                # Safety: if we removed the latest year, log it clearly
                if years and df['date_stat_year'].max() not in years:
                    excluded_year = int(df['date_stat_year'].max())
                    logger.info(
                        "CAGR calculation: excluded partial year %d (only %d months available). "
                        "Using %d→%d window instead.",
                        excluded_year,
                        months_per_year.get(excluded_year, 0),
                        years[0] if years else "?",
                        years[-1] if years else "?",
                    )
                if len(years) < 2:
                    return rates

            # Use last 3-4 complete years for CAGR
            window_years = years[-min(4, len(years)):]
            first_y, last_y = window_years[0], window_years[-1]
            n = last_y - first_y
            if n == 0:
                return rates

            by_year = df.groupby("date_stat_year")["total"].sum()

            def cagr(start_val, end_val, periods):
                if start_val <= 0 or end_val <= 0:
                    return _BASE_GROWTH["total"]
                return (end_val / start_val) ** (1 / periods) - 1

            # Overall CAGR
            v0 = by_year.get(first_y, 0)
            v1 = by_year.get(last_y, 0)
            if v0 and v1:
                rates["total"] = max(0.0, min(0.30, cagr(v0, v1, n)))

            # By voie if available
            if "voie" in df.columns:
                for voie_val in df["voie"].dropna().unique():
                    sub = df[df["voie"] == voie_val].groupby("date_stat_year")["total"].sum()
                    sv0 = sub.get(first_y, 0)
                    sv1 = sub.get(last_y, 0)
                    if sv0 and sv1:
                        key = str(voie_val).lower()
                        rates[key] = max(0.0, min(0.40, cagr(sv0, sv1, n)))

        except Exception as e:
            logger.debug("Growth rate computation failed: %s", e)

        return rates

    def _last_actual(self, voie: Optional[str] = None, complete_only: bool = False):
        """Return (last_year, total) from the DataFrame.

        complete_only=True excludes the latest partial year, which is critical
        for APF forecasting because 2026 currently contains only Jan-Feb.
        """
        if "date_stat_year" not in self.df.columns or "total" not in self.df.columns:
            return (2024, 14_000_000)  # safe fallback

        sub = self.df
        if voie and "voie" in self.df.columns:
            sub = self.df[self.df["voie"].str.lower() == voie.lower()]

        if complete_only and "date_stat_month" in sub.columns:
            months_per_year = sub.groupby("date_stat_year")["date_stat_month"].nunique()
            complete_years = months_per_year[months_per_year >= 12].index
            if len(complete_years) > 0:
                sub = sub[sub["date_stat_year"].isin(complete_years)]

        by_year = sub.groupby("date_stat_year")["total"].sum()
        if by_year.empty:
            return (2024, 14_000_000)

        last_year = int(by_year.index.max())
        last_total = int(by_year[last_year])
        return last_year, last_total

    def _actual_for_year(self, year: int, voie: Optional[str] = None) -> int:
        """Return the actual total for a specific year (0 if not found)."""
        sub = self.df
        if voie and "voie" in self.df.columns:
            sub = self.df[self.df["voie"].str.lower() == voie.lower()]

        if "date_stat_year" not in sub.columns:
            return 0

        filtered = sub[sub["date_stat_year"] == year]
        if filtered.empty:
            return 0
        return int(filtered["total"].sum())

    def _actual_monthly(self, year: int) -> Dict[int, int]:
        """Return monthly breakdown from actual data for a given year."""
        if "date_stat_year" not in self.df.columns or "date_stat_month" not in self.df.columns:
            return {}

        sub = self.df[self.df["date_stat_year"] == year]
        if sub.empty:
            return {}

        monthly = sub.groupby("date_stat_month")["total"].sum().astype(int)
        return {int(m): int(v) for m, v in monthly.items()}

    def _monthly_from_annual(self, annual_total: int) -> Dict[int, int]:
        """Distribute an annual total across months using seasonality coefficients."""
        total_coeff = sum(self._seasonality.values()) or 1.0
        result = {}
        for m in range(1, 13):
            coeff = self._seasonality.get(m, _DEFAULT_SEASONALITY.get(m, 1 / 12))
            result[m] = int(annual_total * coeff / total_coeff)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Chart generator
# ──────────────────────────────────────────────────────────────────────────────

def _build_projection_chart(
    prediction: Dict[str, Any],
    historical_by_year: Dict[int, int],
    charts_dir: str,
) -> Optional[str]:
    """
    Build a Plotly HTML chart showing historical data + three projection scenarios.
    Returns the file path (or None on failure).
    """
    try:
        import plotly.graph_objects as go

        target_year = prediction["target_year"]
        last_actual_year = prediction["last_actual_year"]

        # Historical series
        hist_years = sorted(historical_by_year.keys())
        hist_vals = [historical_by_year[y] for y in hist_years]

        # Projection pivot point: connect last actual to target
        proj_x = [last_actual_year, target_year]

        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=hist_years, y=hist_vals,
            mode="lines+markers",
            name="Données réelles",
            line=dict(color="#2563EB", width=3),
            marker=dict(size=7),
        ))

        # Three scenarios
        colors = {"baseline": "#F59E0B", "optimiste": "#10B981", "pessimiste": "#EF4444"}
        labels = {"baseline": "Scénario de base", "optimiste": "Scénario optimiste", "pessimiste": "Scénario pessimiste"}

        last_actual = historical_by_year.get(last_actual_year, 0)
        for sc in ["pessimiste", "baseline", "optimiste"]:
            key = f"{sc}_total"
            y_proj = [last_actual, prediction[key]]
            fig.add_trace(go.Scatter(
                x=proj_x, y=y_proj,
                mode="lines+markers",
                name=labels[sc],
                line=dict(color=colors[sc], width=2, dash="dash"),
                marker=dict(size=8, symbol="diamond"),
            ))

        # Confidence band (pessimiste ↔ optimiste)
        fig.add_trace(go.Scatter(
            x=proj_x + proj_x[::-1],
            y=[last_actual, prediction["optimiste_total"],
               last_actual, prediction["pessimiste_total"]],
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Intervalle de confiance",
        ))

        fig.update_layout(
            title=dict(
                text=f"Projection du flux touristique {target_year} — Maroc",
                font=dict(size=18, color="#1E3A5F"),
            ),
            xaxis=dict(title="Année", tickmode="linear", dtick=1),
            yaxis=dict(title="Arrivées aux postes frontières", tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="#FFFFFF",
            hovermode="x unified",
            height=480,
            margin=dict(l=60, r=30, t=70, b=80),
        )

        # Save
        os.makedirs(charts_dir, exist_ok=True)
        filename = f"prediction_{target_year}.html"
        filepath = os.path.join(charts_dir, filename)
        fig.write_html(filepath, include_plotlyjs="cdn", full_html=True)
        logger.info("Prediction chart saved: %s", filepath)
        return filepath

    except Exception as e:
        logger.warning("Prediction chart generation failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# PredictionAgent — orchestration + NLP parsing
# ──────────────────────────────────────────────────────────────────────────────

class PredictionAgent:
    """
    Agent wrapper for PredictionEngine.

    Parses user questions to extract:
    - target year (default: last_actual_year + 1)
    - scenario (baseline / optimiste / pessimiste)
    - external factors (detected from keywords in the question)

    Formats the result as Markdown text + generates a Plotly chart.
    """

    def __init__(self, df: pd.DataFrame, charts_dir: Optional[str] = None):
        self.df = df
        self.engine = PredictionEngine(df)
        self._lock = threading.Lock()

        if charts_dir is None:
            from config.settings import CHARTS_DIR
            charts_dir = CHARTS_DIR
        self.charts_dir = charts_dir

        # Web factor inference — optional (silent fail if missing)
        try:
            from tools.search_tools import TourismSearchTool
            self._searcher = TourismSearchTool()
        except Exception as e:
            logger.warning("Prediction searcher unavailable: %s", e)
            self._searcher = None

        try:
            from utils.base_agent import get_shared_client
            from config.settings import AZURE_OPENAI_DEPLOYMENT
            self._llm = get_shared_client()
            self._deployment = AZURE_OPENAI_DEPLOYMENT
        except Exception:
            self._llm = None
            self._deployment = None

        # Last web context snippet — appended to the response when external
        # factor inference was triggered.
        self._last_web_context: str = ""

        # Pre-compute historical by_year for charts
        self._hist_by_year: Dict[int, int] = {}
        if "date_stat_year" in df.columns and "total" in df.columns:
            by_year = df.groupby("date_stat_year")["total"].sum().astype(int)
            self._hist_by_year = {int(y): int(v) for y, v in by_year.items()}

        logger.info(
            "%s initialized — %d years of data (%s–%s)",
            AGENT_NAME,
            len(self._hist_by_year),
            min(self._hist_by_year) if self._hist_by_year else "?",
            max(self._hist_by_year) if self._hist_by_year else "?",
        )

    # ── Public API ────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process a prediction request.

        Returns:
            {
                "response": str,       — formatted Markdown response
                "chart_path": str|None,— path to Plotly HTML file
                "agent": "prediction",
                "agent_name": AGENT_NAME,
                "agent_icon": AGENT_ICON,
            }
        """
        with self._lock:
            parts = re.split(r"\n\s*\[Historique|\n\s*\[Contexte", user_message, maxsplit=1)
            clean_message = parts[0].strip()
            history_block = parts[1] if len(parts) > 1 else ""

            clean_message = self._resolve_implicit_refs(clean_message, history_block)
            if _unsupported_prediction_target(clean_message):
                return _unsupported_prediction_response()
            return self._chat_internal(clean_message)

    def _resolve_implicit_refs(self, message: str, history_block: str) -> str:
        """Resolve implicit follow-up references using conversation history.

        Handles cases like "Et pour 2028 ?" after a 2027 prediction by
        carrying over context (voie, metric type) from the previous turn.
        Only injects missing parameters — never overwrites explicit ones.
        """
        if not history_block:
            return message
        if not self._is_implicit_followup(message):
            return message

        q = _norm_text(message)
        has_year = bool(re.search(r"\b20[2-9]\d\b", q))

        has_voie = any(
            w in q
            for w in ["aerien", "avion", "vol", "aeroport", "maritime",
                       "mer", "bateau", "ferry", "terrestre", "route",
                       "frontiere terrestre"]
        )

        if has_year and has_voie:
            return message

        hist_lower = _norm_text(history_block)

        additions = []
        if not has_voie:
            if any(w in hist_lower for w in ["aerien", "avion", "vol"]):
                additions.append("voie aerienne")
            elif any(w in hist_lower for w in ["maritime", "mer", "bateau"]):
                additions.append("voie maritime")
            elif any(w in hist_lower for w in ["terrestre"]):
                additions.append("voie terrestre")

        if additions:
            message = f"{message} ({', '.join(additions)})"

        return message

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_implicit_followup(message: str) -> bool:
        """Return True for short turns that depend on prior context."""
        norm = _norm_text(message).strip()
        words = norm.split()
        if not words or len(words) > 7:
            return False
        if norm.startswith(("et ", "et pour", "pour ", "meme ", "idem", "pareil", "aussi")):
            return True
        if re.fullmatch(r"(?:20[2-9]\d|\?|\s)+", norm):
            return True
        return False

    def _extract_target_year(self, user_message: str) -> int:
        """Extract the forecast horizon from a potentially compound request."""
        norm = _norm_text(user_message)
        matches = list(re.finditer(r"\b(20[2-9]\d)\b", norm))
        if not matches:
            last_y, _ = self.engine._last_actual(complete_only=True)
            return last_y + 1
        if len(matches) == 1:
            return int(matches[0].group(1))

        forecast_terms = [
            "prediction", "prevision", "prevoir", "prevois", "projection",
            "estimation", "estime", "estimer", "forecast", "predict",
            "scenario", "optimiste", "pessimiste",
        ]
        forecast_positions = [
            m.start()
            for term in forecast_terms
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", norm)
        ]
        first_forecast_pos = min(forecast_positions) if forecast_positions else None
        last_actual_year, _ = self.engine._last_actual(complete_only=True)

        scored: List[Tuple[int, int, int]] = []
        for match in matches:
            year = int(match.group(1))
            pos = match.start()
            before = norm[max(0, pos - 100):pos]
            after = norm[pos:pos + 40]
            score = 0

            if year > last_actual_year:
                score += 8
            if first_forecast_pos is not None and pos > first_forecast_pos:
                score += 10
            if first_forecast_pos is not None and pos < first_forecast_pos:
                score -= 10
            if re.search(
                r"(prediction|prevision|prevoir|prevois|projection|estimation|estime|estimer|forecast|predict)"
                r"[^0-9]{0,80}$",
                before,
            ):
                score += 18
            if re.search(r"(pour|en|annee|horizon|cible)[^0-9]{0,25}$", before):
                score += 5
            if re.search(r"\b(de|depuis|entre)\s+(?:19|20)\d{2}[^.]{0,80}$", before):
                score -= 8
            if re.search(r"^\s*(?:jusqu|a|et|-)", after):
                score -= 4

            scored.append((score, pos, year))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    def _extract_months(self, message: str) -> List[int]:
        """Return the list of month numbers explicitly mentioned in the message.

        Handles "février et mai", "feb may", "mars/avril", "Q2", etc. Empty
        list means "no specific month requested → annual projection".
        """
        msg = message.lower()
        found: List[int] = []
        for name, num in _MONTH_PARSE.items():
            if re.search(r'\b' + re.escape(name) + r'\b', msg):
                if num not in found:
                    found.append(num)
        # M/YYYY or MM/YYYY pattern
        for m in re.finditer(r'\b(1[0-2]|[1-9])/20[2-9]\d\b', message):
            num = int(m.group(1))
            if num not in found:
                found.append(num)
        return sorted(found)

    def _dynamic_search(self, queries: List[Tuple[str, str]]) -> Tuple[str, Dict]:
        """Execute targeted searches from IntentExtractor queries.

        Replaces the legacy 8-query fan-out with a dynamic plan tailored to
        the question (period, metric, entities, geo scope). Returns
        ("", {}) when no searcher / no queries — caller falls back to the
        rule-based engine alone.
        """
        if not self._searcher or not queries:
            return "", {}

        try:
            grouped = self._searcher.search_multi(
                queries=queries,
                max_results_per_query=2,
                use_trusted_only=False,
                max_workers=4,
            )
        except Exception as e:
            logger.debug("Dynamic search failed: %s", e)
            return "", {}

        if not grouped:
            return "", {}

        sections = []
        for label, items in grouped.items():
            lines = [f"**{label}**"]
            for it in items[:2]:
                title = (it.get("title") or "")[:120]
                content = (it.get("content") or "")[:200]
                url = it.get("url") or ""
                if title and content:
                    lines.append(f"- {title}: {content} ({url})")
            if len(lines) > 1:
                sections.append("\n".join(lines))

        return ("\n\n".join(sections), grouped)

    def _infer_factors_from_web(
        self,
        user_message: str,
        target_year: int,
        context_block: str,
    ) -> List[str]:
        """Map the gathered web context to known factor keys via the LLM.

        Always called now — the rule-based keyword matcher alone misses
        most real-world signals. Returns the empty list silently if any
        step fails so the prediction still goes through.
        """
        if not self._llm or not context_block:
            return []

        factor_list = ", ".join(_EXTERNAL_FACTORS.keys())
        prompt = (
            f"Question utilisateur : {user_message}\n\n"
            f"Contexte web (recherche multi-source pour le tourisme Maroc "
            f"{target_year}) :\n\n{context_block[:3500]}\n\n"
            f"Sélectionne MAX 3 facteurs parmi cette liste, uniquement ceux "
            f"qui sont DIRECTEMENT et EXPLICITEMENT évoqués dans les extraits "
            f"ci-dessus avec un impact réel sur le tourisme Maroc en "
            f"{target_year}. N'invente rien. Si le contexte ne mentionne pas "
            f"explicitement un facteur (ex: pas de pandémie active, pas de "
            f"crise économique majeure, pas d'événement sportif imminent), "
            f"NE LE SÉLECTIONNE PAS.\n\n"
            f"Facteurs disponibles : {factor_list}\n\n"
            f"Réponds UNIQUEMENT par un JSON array (max 3 éléments), "
            f'ex: ["croissance_europe"]. Si rien n\'est clairement applicable, '
            f"renvoie []."
        )

        try:
            kwargs = {
                "model": self._deployment,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 200,
            }
            try:
                response = self._llm.chat.completions.create(
                    **kwargs, reasoning_effort="minimal"
                )
            except TypeError:
                response = self._llm.chat.completions.create(**kwargs)
            except Exception as e:
                if "reasoning_effort" in str(e).lower():
                    response = self._llm.chat.completions.create(**kwargs)
                else:
                    raise

            if not response.choices:
                return []
            raw = (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.debug("Factor inference LLM call failed: %s", e)
            return []

        try:
            cleaned = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            match = re.search(r"\[.*?\]", cleaned, flags=re.DOTALL)
            if not match:
                return []
            arr = json.loads(match.group(0))
            # Cap at 3 — protects against runaway adjustments when the LLM
            # ignores the instruction and returns 7+ keys (observed in tests).
            valid = [k for k in arr if k in _EXTERNAL_FACTORS]
            return valid[:3]
        except Exception as e:
            logger.debug("Factor JSON parse failed (%s) raw=%r", e, raw[:120])
            return []

    def _chat_internal(self, user_message: str) -> Dict[str, Any]:
        q = user_message.lower()

        # ── Parse target year ──
        target_year = self._extract_target_year(user_message)

        # ── Parse target months (e.g. "février et mai") ──
        target_months = self._extract_months(user_message)

        # ── Parse scenario (word-boundary regex — see "basé"/"bas" bug) ──
        def _has_word(pattern: str) -> bool:
            return bool(re.search(r'\b' + pattern + r'\b', q))

        optimistic_pat = r"(optimiste|meilleur\s+cas|best\s+case|maximum|haut)"
        pessimistic_pat = r"(pessimiste|pire\s+cas|worst\s+case|minimum|bas)"

        if _has_word(optimistic_pat):
            scenario = "optimiste"
        elif _has_word(pessimistic_pat):
            scenario = "pessimiste"
        else:
            scenario = "baseline"

        # ── Dynamic factor research using IntentExtractor ──
        # The rule-based keyword matcher alone misses real-world signals like
        # current wars, recessions, airline route changes. The IntentExtractor
        # generates a TARGETED query plan (period + metric + entities + geo)
        # instead of a fixed 8-query fan-out — fewer wasted queries, better
        # signal-to-noise on Tavily/Brave.
        needs_external_context = any(
            token in q
            for token in [
                "facteur", "facteurs", "contexte", "impact", "risque", "guerre",
                "conflit", "crise", "pandemie", "pandémie", "covid", "recession",
                "récession", "inflation", "coupe du monde", "mondial", "can ",
                "connectivite", "connectivité", "aerien", "aérien",
            ]
        )
        web_block = ""
        if needs_external_context:
            from utils.intent_extractor import IntentExtractor
            intent_ext = IntentExtractor()
            intent = intent_ext.extract(user_message)
            intent.analysis_type = "causal"
            intent.external_factors_categories = intent_ext._build_factor_categories(intent, user_message.lower())
            targeted_queries = intent_ext.build_search_queries(intent)
            web_block, _raw_grouped = self._dynamic_search(targeted_queries)

        # ── Combine keyword-matched factors with LLM-inferred ones ──
        detected_factors: List[str] = []
        for factor_key, keywords in _FACTOR_KEYWORDS.items():
            if factor_key == "coupe_du_monde" and not any(
                kw in q for kw in ["coupe du monde", "mondial", "world cup"]
            ):
                continue
            if any(kw in q for kw in keywords):
                detected_factors.append(factor_key)

        if web_block:
            inferred = self._infer_factors_from_web(user_message, target_year, web_block)
            negative_or_historical = {
                "pandemie", "crise_economique", "recession_europe",
                "guerre_regionale", "conflit_proche_orient", "reprise_post_covid",
            }
            for k in inferred:
                if k in negative_or_historical and not any(kw in q for kw in _FACTOR_KEYWORDS.get(k, [])):
                    continue
                if k not in detected_factors:
                    detected_factors.append(k)
            self._last_web_context = web_block
        else:
            self._last_web_context = ""

        # ── Parse voie ──
        voie = None
        if any(w in q for w in ["aérien", "avion", "vol", "aeroport"]):
            voie = "aerien"
        elif any(w in q for w in ["maritime", "mer", "bateau", "ferry"]):
            voie = "maritime"
        elif any(w in q for w in ["terrestre", "route", "frontière terrestre"]):
            voie = "terrestre"

        # ── Run prediction ──
        pred = self.engine.predict(
            target_year=target_year,
            scenario=scenario,
            external_factors=detected_factors,
            voie=voie,
        )

        # If specific months were requested, restrict the displayed monthly
        # breakdown to those months and adjust the headline number.
        if target_months:
            pred["target_months"] = target_months
            full_monthly = pred["monthly_breakdown"]
            pred["selected_months_total"] = sum(
                full_monthly.get(m, 0) for m in target_months
            )

        # ── Build response text ──
        response = self._format_response(pred, voie)

        # ── Generate chart ──
        chart_path = _build_projection_chart(pred, self._hist_by_year, self.charts_dir)

        return {
            "response": response,
            "chart_path": chart_path,
            "agent": AGENT_KEY,
            "agent_name": AGENT_NAME,
            "agent_icon": AGENT_ICON,
            "prediction_context": {
                "target_year": target_year,
                "scenario": scenario,
                "voie": voie,
            },
        }

    def _format_response(self, pred: Dict[str, Any], voie: Optional[str]) -> str:
        """Format prediction result as Markdown."""
        target_year = pred["target_year"]
        scenario = pred["scenario"]
        baseline = pred["baseline_total"]
        optimiste = pred["optimiste_total"]
        pessimiste = pred["pessimiste_total"]
        last_year = pred["last_actual_year"]
        last_total = pred["last_actual_total"]
        factors = pred["applied_factors"]
        explanation = pred["explanation"]
        target_months: List[int] = pred.get("target_months", [])

        voie_label = f" (voie {voie})" if voie else ""

        # ── Headline depends on whether specific months were requested ──
        if target_months:
            month_names = ", ".join(_MONTHS_FR[m] for m in target_months)
            months_total = pred.get("selected_months_total", 0)
            title = (
                f"## 🔮 Estimation touristique {target_year} — "
                f"{month_names}{voie_label}"
            )
            headline = (
                f"Estimation pour **{month_names} {target_year}** : "
                f"**{months_total:,}** arrivées (scénario *{scenario}*).\n\n"
                f"{explanation}"
            )
        else:
            title = f"## 🔮 Projection touristique {target_year}{voie_label}"
            headline = explanation

        lines = [title, "", headline, ""]

        # ── Scenarios — when months are requested, show their per-month total
        # under each scenario instead of the full-year number ──
        if target_months:
            full_monthly = pred["monthly_breakdown"]
            full_year = pred["selected_total"] or 1  # avoid /0 for ratios
            month_share = sum(full_monthly.get(m, 0) for m in target_months) / full_year
            opt_months = int(optimiste * month_share)
            base_months = int(baseline * month_share)
            pes_months = int(pessimiste * month_share)
            month_label = ", ".join(_MONTHS_FR[m] for m in target_months)
            lines += [
                f"### Estimations pour {month_label} {target_year}",
                "",
                "| Scénario | Arrivées estimées |",
                "|----------|-------------------|",
                f"| 🟢 Optimiste  | **{opt_months:,}** |",
                f"| 🟡 De base    | **{base_months:,}** |",
                f"| 🔴 Pessimiste | **{pes_months:,}** |",
                "",
            ]
        else:
            lines += [
                "### Estimations par scénario",
                "",
                f"| Scénario | Arrivées estimées | Variation vs {last_year} |",
                "|----------|-------------------|--------------------------|",
                f"| 🟢 Optimiste  | **{optimiste:,}** | {_pct(optimiste, last_total):+.1f}% |",
                f"| 🟡 De base    | **{baseline:,}** | {_pct(baseline, last_total):+.1f}% |",
                f"| 🔴 Pessimiste | **{pessimiste:,}** | {_pct(pessimiste, last_total):+.1f}% |",
                "",
            ]

        if factors:
            factor_details = []
            for f in factors:
                adj = _EXTERNAL_FACTORS.get(f, 0)
                factor_details.append(f"**{f.replace('_', ' ').title()}** ({adj:+.0%})")
            lines += [
                "### Facteurs externes appliqués",
                "",
                ", ".join(factor_details),
                "",
            ]

        # ── Monthly breakdown — restricted to requested months when applicable ──
        months_to_show = target_months if target_months else list(range(1, 13))
        lines += [
            "### Décomposition mensuelle"
            + (" (mois demandés)" if target_months else " (scénario sélectionné)"),
            "",
            "| Mois | Arrivées estimées |",
            "|------|-------------------|",
        ]
        for m in months_to_show:
            val = pred["monthly_breakdown"].get(m, 0)
            lines.append(f"| {_MONTHS_FR[m]} | {val:,} |")

        lines += [
            "",
            "> *Estimation basée sur les tendances historiques APF + "
            "recherche multi-source du contexte actuel. "
            "Pour des prévisions officielles, consulter les publications ONMT/MTAESS.*",
        ]

        if self._last_web_context:
            lines += [
                "",
                "### 🌐 Contexte externe analysé (multi-sources)",
                "",
                self._last_web_context,
            ]
            self._last_web_context = ""

        return "\n".join(lines)


def _pct(val: int, base: int) -> float:
    if base == 0:
        return 0.0
    return (val / base - 1) * 100
