"""
STATOUR KPI Cache
==================
Pre-computes common KPIs at startup so the analytics agent can answer
frequent questions (total arrivals, top nationalities, monthly trends…)
in milliseconds without calling the LLM or executing sandbox code.

Usage:
    from utils.kpi_cache import KPICache

    cache = KPICache(df)
    result = cache.try_answer("Combien de touristes en 2024?")
    if result:
        # Fast-path: KPI found, return directly
        return result
    # Slow-path: route to LLM analytics agent
"""

import re
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger("statour.kpi_cache")

# ── French month names for display ──
MONTHS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
}

# ── Month name → number mapping (FR + EN + short variants) ──
# Used by try_answer() to detect "février 2026", "february", "feb 2026", etc.
# Keys ordered by length DESC so longer names match before shorter ones
# (e.g. "février" before "fev", "january" before "jan").
MONTH_NAMES_MAP = {
    "janvier": 1, "january": 1, "jan": 1,
    "février": 2, "fevrier": 2, "february": 2, "fev": 2, "feb": 2,
    "mars": 3, "march": 3, "mar": 3,
    "avril": 4, "april": 4, "avr": 4, "apr": 4,
    "mai": 5, "may": 5,
    "juin": 6, "june": 6, "jun": 6,
    "juillet": 7, "july": 7, "juil": 7, "jul": 7,
    "août": 8, "aout": 8, "august": 8, "aug": 8,
    "septembre": 9, "september": 9, "sept": 9, "sep": 9,
    "octobre": 10, "october": 10, "oct": 10,
    "novembre": 11, "november": 11, "nov": 11,
    "décembre": 12, "decembre": 12, "december": 12, "dec": 12,
}


class KPICache:
    """
    Pre-computes and caches common KPIs from the active DataFrame.
    Provides a try_answer() method for fast-path responses.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._kpis: Dict[str, Any] = {}
        self._build(df)

    # ──────────────────────────────────────────────────────────────────────
    # Build the cache
    # ──────────────────────────────────────────────────────────────────────

    def _build(self, df: pd.DataFrame) -> None:
        """Pre-compute all KPIs. Called once at startup."""
        try:
            self._build_core(df)
            logger.info(
                "KPI cache built: %d years, years=%s",
                len(self._kpis.get("years_available", [])),
                self._kpis.get("years_available", []),
            )
        except Exception as e:
            logger.warning("KPI cache build failed: %s", e)

    def _build_core(self, df: pd.DataFrame) -> None:
        k = self._kpis

        # ── Year range ──
        if "date_stat_year" in df.columns:
            years = sorted(df["date_stat_year"].dropna().unique().astype(int).tolist())
            k["years_available"] = years
            k["min_year"] = years[0] if years else None
            k["max_year"] = years[-1] if years else None

            # ── Totals by year ──
            if "total" in df.columns:
                by_year = df.groupby("date_stat_year")["total"].sum().astype(int)
                k["total_by_year"] = by_year.to_dict()

                # MRE + TES by year
                if "mre" in df.columns and "tes" in df.columns:
                    k["mre_by_year"] = df.groupby("date_stat_year")["mre"].sum().astype(int).to_dict()
                    k["tes_by_year"] = df.groupby("date_stat_year")["tes"].sum().astype(int).to_dict()

                # Latest year/month
                last_year = years[-1]
                k["last_year"] = last_year
                df_last = df[df["date_stat_year"] == last_year]
                if "date_stat_month" in df.columns:
                    last_month = int(df_last["date_stat_month"].max())
                    k["last_month"] = last_month
                    k["last_month_name"] = MONTHS_FR.get(last_month, str(last_month))

                    df_last_month = df_last[df_last["date_stat_month"] == last_month]
                    k["last_month_total"] = int(df_last_month["total"].sum())
                    if "mre" in df.columns:
                        k["last_month_mre"] = int(df_last_month["mre"].sum())
                        k["last_month_tes"] = int(df_last_month["tes"].sum())

                # Grand total (all time)
                k["grand_total"] = int(df["total"].sum())
                if "mre" in df.columns:
                    k["grand_mre"] = int(df["mre"].sum())
                    k["grand_tes"] = int(df["tes"].sum())

            # ── Monthly breakdown by year ──
            if "total" in df.columns and "date_stat_month" in df.columns:
                monthly = (
                    df.groupby(["date_stat_year", "date_stat_month"])["total"]
                    .sum().astype(int)
                )
                # groupby(level=0) sub-series keeps MultiIndex → use xs() to strip year level
                k["monthly_by_year"] = {
                    int(yr): {int(idx[1] if isinstance(idx, tuple) else idx): int(val)
                               for idx, val in months.items()}
                    for yr, months in monthly.groupby(level=0)
                }

        # ── Top nationalities (all time) ──
        if "nationalite" in df.columns and "total" in df.columns:
            top_nat = (
                df.groupby("nationalite")["total"]
                .sum().nlargest(20).astype(int)
            )
            k["top_nationalities_20"] = top_nat.to_dict()
            k["top_nationalities_5"] = dict(list(top_nat.items())[:5])
            k["top_nationalities_10"] = dict(list(top_nat.items())[:10])

        # ── Voie (entry mode) breakdown ──
        if "voie" in df.columns and "total" in df.columns:
            by_voie = df.groupby("voie")["total"].sum().astype(int).to_dict()
            k["by_voie"] = by_voie
            total_voie = sum(by_voie.values())
            k["by_voie_pct"] = {
                v: round(cnt / total_voie * 100, 1) if total_voie else 0
                for v, cnt in by_voie.items()
            }

        # ── Regional breakdown ──
        if "region" in df.columns and "total" in df.columns:
            k["by_region"] = df.groupby("region")["total"].sum().astype(int).to_dict()

        # ── Continent breakdown ──
        if "continent" in df.columns and "total" in df.columns:
            k["by_continent"] = df.groupby("continent")["total"].sum().astype(int).to_dict()

    # ──────────────────────────────────────────────────────────────────────
    # Fast-path answering
    # ──────────────────────────────────────────────────────────────────────

    def _extract_month(self, q: str) -> Optional[int]:
        """Return month number (1-12) if the question mentions a month, else None.

        Why: try_answer() must not return annual totals when the user asks
        about a specific month. Previously, "février 2026" matched year 2026
        and returned the full-year total, ignoring the month.
        """
        for name, num in MONTH_NAMES_MAP.items():
            if re.search(r'\b' + re.escape(name) + r'\b', q):
                return num
        # Also match "M/YYYY" or "MM/YYYY" patterns
        m = re.search(r'\b(1[0-2]|[1-9])/20[12]\d\b', q)
        if m:
            return int(m.group(1))
        return None

    def try_answer(self, question: str) -> Optional[str]:
        """
        Try to answer a question from pre-computed KPIs.
        Returns a formatted response string if matched, or None to route to LLM.

        Only handles simple, unambiguous KPI queries.
        Complex analysis, charts, and multi-dimensional queries go to LLM.
        """
        if not self._kpis:
            return None

        q = question.lower().strip()

        # ── Don't intercept chart/graph requests ──
        chart_words = ["graphique", "chart", "graph", "courbe", "diagramme",
                       "histogramme", "visualise", "affiche", "trace", "plot", "heatmap"]
        if any(w in q for w in chart_words):
            return None

        # ── Don't intercept comparison or multi-dimensional queries ──
        complex_words = ["compar", "évolution", "tendance", "trend", "par rapport",
                         "versus", " vs ", "entre", "top 5", "top 10", "top 20",
                         "croissance", "variation", "pourcentage", "breakdown",
                         "par mois", "mensuel", "par voie", "par région", "par continent",
                         "estimation", "prévision", "prévisio", "prédi", "futur", "2027", "2028"]
        if any(w in q for w in complex_words):
            return None

        # ── PRIORITÉ : un mois spécifique est mentionné → laisser le LLM ──
        # Why: previously "février 2026" matched on year 2026 and returned the
        # full-year total, ignoring the month. Monthly questions always route
        # to the LLM so the generated code filters by year+month exactly.
        if self._extract_month(q) is not None:
            return None

        # ── DOMAINE HÉBERGEMENT → bypass cache (KPI cache ne couvre que APF) ──
        # Le cache est construit uniquement depuis fact_statistiques_apf (MRE+TES).
        # Toute question relative aux nuitées, hébergement, EHTC ou STDN doit aller
        # au LLM analytics pour qu'il interroge fact_statistiqueshebergementnationaliteestimees.
        _HEBERGEMENT_KW = [
            "nuitée", "nuitee", "nuitées", "nuitees",
            "hébergement", "hebergement",
            "hôtel", "hotel",
            "ehtc", "stdn",
            "établissement", "etablissement",
            "maison d'hôte", "maison hôte", "maison hotes",
            "camping", "riad",
            "chambre", "capacit",
            "arrivée hôt", "arrivee hôt", "arrivées hôt",
            "type d'héberg", "type d heberg",
            "délégation", "delegation",
            "télédéclaration", "teledeclaration",
        ]
        if any(kw in q for kw in _HEBERGEMENT_KW):
            return None

        # ── Year total query ──
        year_match = re.search(r'\b(20[12]\d)\b', q)
        if year_match:
            year = int(year_match.group(1))
            if year in self._kpis.get("total_by_year", {}):
                total = self._kpis["total_by_year"][year]
                mre = self._kpis.get("mre_by_year", {}).get(year, 0)
                tes = self._kpis.get("tes_by_year", {}).get(year, 0)

                # Check for MRE/TES specific question
                # Use word-boundary check so "touristes" does not match "tes"
                _tes_exact = bool(re.search(r'\btes\b', q))
                if "mre" in q or "marocain résidant" in q or "diaspora" in q:
                    return f"En **{year}**, les **MRE** (Marocains Résidant à l'Étranger) représentaient **{mre:,}** arrivées."
                if _tes_exact or "touriste étranger" in q or "étranger séjournant" in q:
                    return f"En **{year}**, les **TES** (Touristes Étrangers Séjournistes) représentaient **{tes:,}** arrivées."

                # Generic total
                response = f"En **{year}**, le Maroc a enregistré **{total:,}** arrivées aux postes frontières"
                if mre and tes:
                    mre_pct = round(mre / total * 100, 1) if total else 0
                    tes_pct = round(tes / total * 100, 1) if total else 0
                    response += (
                        f", dont **{mre:,} MRE** ({mre_pct}%) "
                        f"et **{tes:,} TES** ({tes_pct}%)."
                    )
                else:
                    response += "."
                return response

        # ── Grand total query ──
        grand_total_kws = ["total arrivées", "total des arrivées", "total global",
                           "grand total", "toute la période", "depuis 2019", "depuis le début"]
        if any(kw in q for kw in grand_total_kws):
            total = self._kpis.get("grand_total")
            if total:
                mre = self._kpis.get("grand_mre", 0)
                tes = self._kpis.get("grand_tes", 0)
                years = self._kpis.get("years_available", [])
                period = f"{years[0]}–{years[-1]}" if len(years) >= 2 else str(years[0]) if years else "?"
                return (
                    f"Sur toute la période disponible ({period}), le Maroc a enregistré "
                    f"**{total:,}** arrivées aux postes frontières, dont "
                    f"**{mre:,} MRE** ({round(mre/total*100,1) if total else 0}%) "
                    f"et **{tes:,} TES** ({round(tes/total*100,1) if total else 0}%)."
                )

        # ── Last available month ──
        last_month_kws = ["dernier mois", "dernière période", "derniere periode",
                          "données les plus récentes", "donnees les plus recentes",
                          "dernier enregistrement", "most recent", "latest",
                          "données récentes", "donnees recentes"]
        if any(kw in q for kw in last_month_kws):
            lm_total = self._kpis.get("last_month_total")
            if lm_total is not None:
                lm_name = self._kpis.get("last_month_name", "?")
                lm_year = self._kpis.get("last_year", "?")
                mre = self._kpis.get("last_month_mre", 0)
                tes = self._kpis.get("last_month_tes", 0)
                return (
                    f"La dernière période disponible est **{lm_name} {lm_year}** avec "
                    f"**{lm_total:,}** arrivées "
                    f"(**{mre:,} MRE** + **{tes:,} TES**)."
                )

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Accessors
    # ──────────────────────────────────────────────────────────────────────

    def get(self, key: str, default=None):
        return self._kpis.get(key, default)

    def years_available(self):
        return self._kpis.get("years_available", [])

    def total_for_year(self, year: int) -> Optional[int]:
        return self._kpis.get("total_by_year", {}).get(year)

    def refresh(self, df: pd.DataFrame) -> None:
        """Rebuild cache with a new DataFrame (e.g., after dataset switch)."""
        self.df = df
        self._kpis = {}
        self._build(df)
        logger.info("KPI cache refreshed")
