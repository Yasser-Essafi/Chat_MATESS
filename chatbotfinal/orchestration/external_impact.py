"""Deterministic external-impact analysis for tourism questions.

This path handles questions such as:
- "la guerre au Moyen-Orient a-t-elle impacte le flux ?"
- "quels facteurs externes impactent ce flux ?"

The generic planner can mix SQL, RAG and web search in a plausible but loose
order. This module forces a tourism analyst workflow: resolve the event first,
turn it into testable hypotheses, query APF and hebergement, then synthesize
with explicit confidence and limitations.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config.settings import CHARTS_DIR
from utils.fabric_catalog import (
    APF_SCOPE_LABEL,
    APF_TABLE,
    HEBERGEMENT_SCOPE_LABEL,
    HEBERGEMENT_TABLE,
    month_range_label,
)
from utils.logger import get_logger


logger = get_logger("statour.external_impact")


MONTHS_FR = {
    1: "Janvier", 2: "Fevrier", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Aout",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Decembre",
}

MONTH_LOOKUP = {
    "janvier": 1, "janv": 1, "january": 1,
    "fevrier": 2, "février": 2, "fev": 2, "fév": 2, "february": 2,
    "mars": 3, "march": 3,
    "avril": 4, "april": 4,
    "mai": 5, "may": 5,
    "juin": 6, "june": 6,
    "juillet": 7, "july": 7,
    "aout": 8, "août": 8, "august": 8,
    "septembre": 9, "sept": 9, "september": 9,
    "octobre": 10, "october": 10,
    "novembre": 11, "november": 11,
    "decembre": 12, "décembre": 12, "december": 12,
}

MIDDLE_EAST_TERMS = [
    "MOYEN", "ORIENT", "MIDDLE EAST", "ARABIE", "SAOUD", "EMIRAT", "ÉMIRAT",
    "QATAR", "KOWEIT", "KUWAIT", "BAHREIN", "BAHRAIN", "OMAN", "IRAN", "IRAK",
    "IRAQ", "LIBAN", "LEBANON", "JORDAN", "JORDANIE", "ISRAEL", "ISRAËL",
    "PALEST", "SYRIE", "SYRIA", "YEMEN", "EGYPTE", "ÉGYPTE", "TURQUIE", "TURKEY",
]

EVENT_KEYWORDS = [
    "guerre", "conflit", "crise", "tension", "seisme", "séisme", "pandemie",
    "pandémie", "covid", "visa", "greve", "grève", "petrole", "pétrole",
    "inflation", "ramadan", "can", "coupe du monde", "aerien", "aérien",
    "fermeture", "espace aerien", "espace aérien",
]

FACTOR_KEYWORDS = [
    "facteur", "facteurs", "cause", "causes", "raison", "raisons", "pourquoi",
    "explique", "impactent", "impacte", "ce flux", "cette evolution", "cette évolution",
    "hypothese", "hypotheses", "hypothèse", "hypothèses", "tester", "stagne",
    "stagnent", "augmente", "baisse", "mecanisme", "mécanisme",
]

RAMADAN_STARTS = {
    2024: date(2024, 3, 11),
    2025: date(2025, 3, 1),
    2026: date(2026, 2, 18),
}

COUNTRY_ALIASES = {
    "France": ["france", "francais", "français"],
    "Chine": ["chine", "chinois", "chinoise"],
    "Royaume-Uni": ["royaume uni", "royaume-uni", "anglais", "britannique", "uk"],
    "Espagne": ["espagne", "espagnol", "espagnols"],
    "Allemagne": ["allemagne", "allemand", "allemands"],
    "Italie": ["italie", "italien", "italiens"],
    "Etats-Unis": ["etats unis", "états-unis", "usa", "americain", "américain"],
}

COUNTRY_SQL_TERMS = {
    "France": ["FRANCE"],
    "Chine": ["CHINE", "CHINA"],
    "Royaume-Uni": ["ROYAUME", "UNITED KINGDOM", "UK"],
    "Espagne": ["ESPAGNE", "SPAIN"],
    "Allemagne": ["ALLEMAGNE", "GERMANY"],
    "Italie": ["ITALIE", "ITALY"],
    "Etats-Unis": ["ETATS", "UNITED STATES", "USA"],
}


@dataclass
class SourceItem:
    title: str
    url: str
    content: str = ""


@dataclass
class EventProfile:
    mode: str
    label: str
    search_query: str
    start_date: Optional[date] = None
    first_observed_year: Optional[int] = None
    first_observed_month: Optional[int] = None
    analysis_year: Optional[int] = None
    analysis_month: Optional[int] = None
    test_months: List[int] = field(default_factory=list)
    exposed_label: str = "marches exposes"
    exposed_filter_apf: str = ""
    exposed_filter_hbg: str = ""
    hbg_geo_filter: str = ""
    geo_label: str = ""
    focus_countries: List[str] = field(default_factory=list)
    voie_requested: bool = False
    segment_requested: bool = False
    market_requested: bool = False
    hypotheses: List[str] = field(default_factory=list)
    sources: List[SourceItem] = field(default_factory=list)
    confidence: float = 0.55


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[-‐‑‒–—_/']", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def _fmt_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _month_label(year: int, month: int) -> str:
    return f"{MONTHS_FR.get(int(month), str(month))} {int(year)}"


def _next_month(year: int, month: int) -> Tuple[int, int]:
    if month >= 12:
        return year + 1, 1
    return year, month + 1


def _event_first_observable(start: date) -> Tuple[int, int]:
    # date_stat is monthly. If an event starts in the final third of a month,
    # the first clean observation is the next month.
    if start.day >= 20:
        return _next_month(start.year, start.month)
    return start.year, start.month


def _like_filter(column: str, terms: Iterable[str]) -> str:
    clauses = []
    for term in terms:
        clean = term.replace("'", "''").upper()
        clauses.append(f"UPPER({column}) LIKE '%{clean}%'")
    return "(" + " OR ".join(clauses) + ")"


def _extract_sources(results: List[Dict[str, Any]], limit: int = 5) -> List[SourceItem]:
    sources: List[SourceItem] = []
    for r in (results or [])[:limit]:
        title = str(r.get("title") or "").strip()
        url = str(r.get("url") or "").strip()
        content = str(r.get("content") or r.get("snippet") or "").strip()
        if title or url or content:
            sources.append(SourceItem(title=title, url=url, content=content))
    return sources


def _parse_dates_from_text(text: str) -> List[date]:
    raw = text or ""
    norm = _norm(raw)
    dates: List[date] = []

    explicit = re.finditer(
        r"\b(\d{1,2})\s+"
        r"(janvier|janv|fevrier|février|fev|fév|mars|avril|mai|juin|juillet|"
        r"aout|août|septembre|sept|octobre|novembre|decembre|décembre|"
        r"january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(20\d{2})\b",
        norm,
    )
    for m in explicit:
        day = int(m.group(1))
        month = MONTH_LOOKUP.get(m.group(2), 0)
        year = int(m.group(3))
        if month:
            try:
                dates.append(date(year, month, day))
            except ValueError:
                pass

    relative = re.finditer(
        r"\b(fin|debut|début|mi)\s+"
        r"(janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|"
        r"septembre|octobre|novembre|decembre|décembre)\s+(20\d{2})\b",
        norm,
    )
    for m in relative:
        pos, month_name, year_s = m.groups()
        month = MONTH_LOOKUP.get(month_name, 0)
        year = int(year_s)
        if not month:
            continue
        day = 1 if pos in {"debut", "début"} else 15 if pos == "mi" else 28
        try:
            dates.append(date(year, month, day))
        except ValueError:
            pass

    return dates


def _parse_relative_dates_with_year(text: str, fallback_year: int) -> List[date]:
    norm = _norm(text)
    dates: List[date] = []
    relative = re.finditer(
        r"\b(fin|debut|mi)\s+"
        r"(janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre)"
        r"(?!\s+20\d{2})\b",
        norm,
    )
    for m in relative:
        pos, month_name = m.groups()
        month = MONTH_LOOKUP.get(month_name, 0)
        if not month:
            continue
        day = 1 if pos == "debut" else 15 if pos == "mi" else 28
        try:
            dates.append(date(fallback_year, month, day))
        except ValueError:
            pass
    return dates


def _requested_analysis_months(text: str) -> List[int]:
    norm = _norm(text)
    month_names = sorted({_norm(k) for k in MONTH_LOOKUP}, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(m) for m in month_names) + r")\b"
    months: List[int] = []
    for match in re.finditer(pattern, norm):
        prefix = norm[max(0, match.start() - 12):match.start()]
        if re.search(r"\b(fin|debut|mi)\s+$", prefix):
            continue
        month = MONTH_LOOKUP.get(match.group(1), 0)
        if month and month not in months:
            months.append(month)
    return months


def _focus_countries_from_message(message: str) -> List[str]:
    norm = _norm(message)
    countries: List[str] = []
    for label, aliases in COUNTRY_ALIASES.items():
        if any(re.search(r"\b" + re.escape(_norm(alias)) + r"\b", norm) for alias in aliases):
            countries.append(label)
    return countries


def _country_sql_filter(column: str, countries: List[str]) -> str:
    terms: List[str] = []
    for country in countries:
        terms.extend(COUNTRY_SQL_TERMS.get(country, [country.upper()]))
    return _like_filter(column, terms) if terms else "1=0"


def _is_voie_requested(message: str) -> bool:
    norm = _norm(message)
    return any(k in norm for k in ["voie", "aerienne", "maritime", "terrestre", "aerien"])


def _is_segment_requested(message: str) -> bool:
    norm = _norm(message)
    return any(re.search(r"\b" + term + r"\b", norm) for term in ["mre", "tes"])


def _is_hypothesis_scenario(message: str) -> bool:
    norm = _norm(message)
    has_apf = "apf" in norm or "frontiere" in norm or "arrivee" in norm
    has_hbg = "nuitee" in norm or "hebergement" in norm or "hotel" in norm
    has_hypothesis = any(k in norm for k in ["hypothese", "tester", "si ", "stagn", "diverg", "ecart"])
    has_contrast = any(k in norm for k in ["mais", "alors que", "pourtant", "stagn", "baisse"])
    return bool(has_apf and has_hbg and has_hypothesis and has_contrast)


def _source_event_start(sources: List[SourceItem], fallback_year: Optional[int]) -> Tuple[Optional[date], float]:
    text = "\n".join(f"{s.title}\n{s.content}" for s in sources)
    norm = _norm(text)

    # Prefer dates that appear in an event-start wording over incidental dates
    # such as publication dates or economic baseline references.
    if fallback_year and re.search(r"(declench\w*|depuis|lance\w*|frappe\w*|attaque\w*).{0,140}fin\s+fevrier\s+" + str(fallback_year), norm):
        return date(fallback_year, 2, 28), 0.92
    if fallback_year and re.search(r"(declench\w*|depuis|lance\w*|frappe\w*|attaque\w*).{0,140}28\s+fevrier\s+" + str(fallback_year), norm):
        return date(fallback_year, 2, 28), 0.92

    start_cue = re.search(
        r"(declench\w*|debut\w*|début\w*|depuis|lance\w*|frappe\w*|attaque\w*|commenc\w*)"
        r".{0,100}"
        r"(fin|debut|début|mi)?\s*"
        r"(\d{1,2})?\s*"
        r"(janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|septembre|octobre|novembre|decembre|décembre)"
        r"\s+(20\d{2})",
        norm,
    )
    if start_cue:
        pos = start_cue.group(2)
        day_s = start_cue.group(3)
        month = MONTH_LOOKUP.get(start_cue.group(4), 0)
        year = int(start_cue.group(5))
        if not fallback_year or year == fallback_year:
            day = int(day_s) if day_s else 1 if pos in {"debut", "début"} else 15 if pos == "mi" else 28 if pos == "fin" else 1
            try:
                return date(year, month, day), 0.9
            except ValueError:
                pass

    dates = _parse_dates_from_text(text)
    if fallback_year:
        dates = [d for d in dates if d.year == fallback_year] or dates
    if not dates:
        return None, 0.0
    dates = sorted(dates)
    return dates[0], 0.75


def _years_in_text(text: str) -> List[int]:
    return [int(y) for y in re.findall(r"\b(20[12]\d)\b", text or "")]


def should_handle_external_impact(message: str, conversation_context: str = "") -> bool:
    norm = _norm(message)
    if _is_hypothesis_scenario(message):
        return True
    has_metric = any(k in norm for k in [
        "flux", "tourisme", "touriste", "arrive", "nuitee", "hebergement", "apf",
        "mre", "tes", "hotel", "pays de residence", "marche", "marches", "mix",
        "golfe", "moyen orient",
    ])
    has_event = any(k in norm for k in [_norm(x) for x in EVENT_KEYWORDS])
    has_factor = any(k in norm for k in [_norm(x) for x in FACTOR_KEYWORDS])
    asks_impact = any(k in norm for k in ["impact", "effet", "consequence", "influence", "perturb", "affect", "choc", "rupture", "change", "modifie"])
    return bool(has_metric and ((has_event and asks_impact) or has_factor))


def needs_external_event_search(message: str) -> bool:
    """Return True only when the controlled impact path lacks a testable event date."""
    if not should_handle_external_impact(message, ""):
        return False
    if _is_hypothesis_scenario(message):
        return False
    profile = _build_event_profile(message, search_tool=None)
    return profile.mode == "event_impact" and profile.start_date is None


def _event_label_from_message(message: str) -> str:
    norm = _norm(message)
    if "seisme" in norm and "al haouz" in norm:
        return "seisme d'Al Haouz"
    if "moyen orient" in norm or "iran" in norm or "golfe" in norm:
        return "guerre au Moyen-Orient"
    if "seisme" in norm:
        return "seisme"
    if "ramadan" in norm:
        return "Ramadan"
    if "petrole" in norm:
        return "hausse du petrole"
    if "visa" in norm:
        return "changement visa/reglementaire"
    if "inflation" in norm:
        return "inflation et pouvoir d'achat"
    if "aerien" in norm or "vol" in norm:
        return "perturbations aeriennes"
    return "evenement externe mentionne"


def _search_query_for_event(message: str, label: str, current_year: int) -> str:
    norm = _norm(message)
    if label == "guerre au Moyen-Orient":
        return (
            f"guerre Moyen-Orient fin fevrier {current_year} tourisme Maroc "
            "date debut impact flux touristique annulations vols"
        )
    if "facteur" in norm or "pourquoi" in norm or "cause" in norm:
        return (
            f"Maroc tourisme {current_year} facteurs externes impact flux touristiques "
            "guerre Moyen-Orient petrole inflation vols Ramadan"
        )
    return f"{label} Maroc tourisme {current_year} date debut impact flux touristique"


def _hbg_geo_filter_from_message(message: str) -> Tuple[str, str]:
    norm = _norm(message)
    mapping = {
        "marrakech": ("Marrakech", "MARRAKECH"),
        "casablanca": ("Casablanca", "CASABLANCA"),
        "agadir": ("Agadir", "AGADIR"),
        "tanger": ("Tanger", "TANGER"),
        "rabat": ("Rabat", "RABAT"),
        "fes": ("Fes", "FES"),
        "fès": ("Fes", "FES"),
        "essaouira": ("Essaouira", "ESSAOUIRA"),
    }
    for key, (label, sql_term) in mapping.items():
        if key in norm:
            return f"UPPER(province_name) LIKE '%{sql_term}%'", label
    return "", ""


def _build_event_profile(message: str, search_tool=None) -> EventProfile:
    years = _years_in_text(message)
    current_year = years[0] if years else datetime.now().year
    norm = _norm(message)
    mode = "factor_discovery" if any(k in norm for k in ["facteur", "pourquoi", "cause", "raison"]) and not any(
        k in norm for k in ["guerre", "conflit", "seisme", "crise", "visa", "petrole", "inflation", "ramadan", "aerien"]
    ) else "event_impact"
    label = _event_label_from_message(message)
    if label in {"hausse du petrole", "inflation et pouvoir d'achat", "changement visa/reglementaire", "perturbations aeriennes"} and re.search(r"\b(peut|pourrait|possible|risque|mecanisme|mécanisme)\b", norm):
        mode = "factor_discovery"
    query = _search_query_for_event(message, label, current_year)

    start_date: Optional[date] = None
    date_conf = 0.0
    user_dates = _parse_dates_from_text(message) + _parse_relative_dates_with_year(message, current_year)
    if user_dates:
        start_date = sorted(user_dates)[0]
        date_conf = 0.9

    known_start: Optional[date] = None
    if label == "seisme d'Al Haouz":
        known_start = date(2023, 9, 8)
    if label == "Ramadan":
        known_start = RAMADAN_STARTS.get(current_year)
    if label == "guerre au Moyen-Orient" and current_year == 2026:
        known_start = date(2026, 2, 28)
    if known_start:
        start_date = known_start
        date_conf = 0.95

    results: List[Dict[str, Any]] = []
    needs_search = search_tool is not None and mode != "factor_discovery" and date_conf < 0.85
    if needs_search:
        try:
            results = search_tool.smart_search(query, analysis_type="causal", max_results=5)
            if not results:
                results = search_tool.smart_search(query, analysis_type="news", max_results=5)
        except Exception as e:
            logger.warning("External impact search failed: %s", str(e)[:120])

    sources = _extract_sources(results)
    if needs_search:
        source_start, source_conf = _source_event_start(sources, current_year)
        if source_start and source_conf > date_conf:
            start_date = source_start
            date_conf = source_conf

    if mode == "factor_discovery":
        start_date = None
        date_conf = 0.0
        if label == "evenement externe mentionne":
            label = "cartographie des facteurs externes"

    if label == "guerre au Moyen-Orient":
        exposed_filter_apf = (
            f"({_like_filter('nationalite', MIDDLE_EAST_TERMS)} OR "
            f"{_like_filter('continent', ['MOYEN', 'ORIENT', 'ARAB', 'MIDDLE'])})"
        )
        exposed_filter_hbg = _like_filter("nationalite_name", MIDDLE_EAST_TERMS)
        hypotheses = [
            "baisse possible des voyages depuis les marches directement exposes au conflit",
            "report possible de demande depuis les destinations concurrentes du Moyen-Orient vers le Maroc",
            "effet indirect via fermeture d'espaces aeriens, prix du petrole et couts des billets",
            "effet different selon APF, arrivees hotelieres et nuitees: les perimetres ne mesurent pas la meme chose",
        ]
        exposed_label = "pays de residence Moyen-Orient / Golfe elargi"
    else:
        exposed_filter_apf = "1=0"
        exposed_filter_hbg = "1=0"
        hypotheses = [
            "variation possible de la demande selon les marches emetteurs",
            "impact possible via transport aerien, pouvoir d'achat, calendrier et perception securitaire",
            "necessite de distinguer APF, arrivees hotelieres et nuitees avant de conclure",
        ]
        exposed_label = "segment expose a definir"

    first_year = first_month = None
    if start_date:
        first_year, first_month = _event_first_observable(start_date)

    hbg_geo_filter, geo_label = _hbg_geo_filter_from_message(message)
    analysis_months = _requested_analysis_months(message)
    analysis_year = first_year
    analysis_month = first_month
    test_months = [first_month] if first_month else []
    if analysis_months and years:
        analysis_year = years[0]
        analysis_month = max(analysis_months)
        test_months = analysis_months
    focus_countries = _focus_countries_from_message(message)
    market_requested = (
        mode == "event_impact"
        or bool(focus_countries)
        or any(k in norm for k in ["pays", "marche", "marches", "nationalite", "residence"])
    )

    profile = EventProfile(
        mode=mode,
        label=label,
        search_query=query,
        start_date=start_date,
        first_observed_year=first_year,
        first_observed_month=first_month,
        analysis_year=analysis_year,
        analysis_month=analysis_month,
        test_months=test_months,
        exposed_label=exposed_label,
        exposed_filter_apf=exposed_filter_apf,
        exposed_filter_hbg=exposed_filter_hbg,
        hbg_geo_filter=hbg_geo_filter,
        geo_label=geo_label,
        focus_countries=focus_countries,
        voie_requested=_is_voie_requested(message),
        segment_requested=_is_segment_requested(message),
        market_requested=market_requested,
        hypotheses=hypotheses,
        sources=sources,
        confidence=0.55 + min(date_conf, 0.35),
    )
    return profile


def _safe_query(db_layer, sql: str) -> pd.DataFrame:
    if not db_layer or getattr(db_layer, "source", None) != "fabric":
        return pd.DataFrame()
    return db_layer.safe_query(sql)


def _latest_month(db_layer, table: str) -> Optional[Tuple[int, int]]:
    tbl = db_layer._qualify(table)
    df = _safe_query(
        db_layer,
        f"SELECT TOP 1 YEAR(date_stat) AS annee, MONTH(date_stat) AS mois "
        f"FROM {tbl} WHERE date_stat IS NOT NULL "
        f"GROUP BY YEAR(date_stat), MONTH(date_stat) "
        f"ORDER BY annee DESC, mois DESC",
    )
    if df.empty:
        return None
    return int(df.iloc[0]["annee"]), int(df.iloc[0]["mois"])


def _impact_sql(db_layer, profile: EventProfile) -> Dict[str, pd.DataFrame]:
    if not db_layer or getattr(db_layer, "source", None) != "fabric":
        return {}

    apf = db_layer._qualify(APF_TABLE)
    hbg = db_layer._qualify(HEBERGEMENT_TABLE)

    target_year = profile.analysis_year or profile.first_observed_year
    target_month = profile.analysis_month or profile.first_observed_month
    if not target_year or not target_month:
        # Use latest common coverage for factor discovery or unresolved events.
        latest_apf = _latest_month(db_layer, APF_TABLE)
        latest_hbg = _latest_month(db_layer, HEBERGEMENT_TABLE)
        candidates = [x for x in [latest_apf, latest_hbg] if x]
        if not candidates:
            return {}
        target_year, target_month = min(candidates)

    year_prev = target_year - 1
    hbg_filter = profile.exposed_filter_hbg or "1=0"
    apf_filter = profile.exposed_filter_apf or "1=0"
    hbg_geo_clause = f"AND ({profile.hbg_geo_filter}) " if profile.hbg_geo_filter else ""
    market_apf_clause = f"AND ({apf_filter}) " if profile.label == "guerre au Moyen-Orient" else ""
    market_hbg_clause = f"AND ({hbg_filter}) " if profile.label == "guerre au Moyen-Orient" else ""

    monthly_apf = _safe_query(
        db_layer,
        f"SELECT YEAR(date_stat) AS annee, MONTH(date_stat) AS mois, "
        f"SUM(mre + tes) AS arrivees_apf, "
        f"SUM(CASE WHEN {apf_filter} THEN mre + tes ELSE 0 END) AS arrivees_apf_exposees "
        f"FROM {apf} WHERE date_stat IS NOT NULL AND YEAR(date_stat) IN ({year_prev}, {target_year}) "
        f"AND MONTH(date_stat) <= {target_month} "
        f"GROUP BY YEAR(date_stat), MONTH(date_stat) ORDER BY annee, mois",
    )

    monthly_hbg = _safe_query(
        db_layer,
        f"SELECT YEAR(date_stat) AS annee, MONTH(date_stat) AS mois, "
        f"SUM(arrivees) AS arrivees_hotelieres, SUM(nuitees) AS nuitees, "
        f"COUNT_BIG(*) AS lignes "
        f"FROM {hbg} WHERE date_stat IS NOT NULL AND YEAR(date_stat) IN ({year_prev}, {target_year}) "
        f"AND MONTH(date_stat) <= {target_month} "
        f"{hbg_geo_clause}"
        f"GROUP BY YEAR(date_stat), MONTH(date_stat) ORDER BY annee, mois",
    )

    market_apf = pd.DataFrame()
    market_hbg = pd.DataFrame()
    if profile.market_requested:
        market_apf = _safe_query(
            db_layer,
            f"WITH x AS ("
            f"SELECT nationalite AS pays, "
            f"SUM(CASE WHEN YEAR(date_stat) = {target_year} AND MONTH(date_stat) = {target_month} THEN mre + tes ELSE 0 END) AS courant, "
            f"SUM(CASE WHEN YEAR(date_stat) = {year_prev} AND MONTH(date_stat) = {target_month} THEN mre + tes ELSE 0 END) AS precedent "
            f"FROM {apf} WHERE date_stat IS NOT NULL AND YEAR(date_stat) IN ({year_prev}, {target_year}) "
            f"AND MONTH(date_stat) = {target_month} "
            f"{market_apf_clause}"
            f"GROUP BY nationalite) "
            f"SELECT TOP 8 pays, courant, precedent, courant - precedent AS delta "
            f"FROM x WHERE courant > 0 OR precedent > 0 ORDER BY ABS(courant - precedent) DESC",
        )

        market_hbg = _safe_query(
            db_layer,
            f"WITH x AS ("
            f"SELECT nationalite_name AS pays, "
            f"SUM(CASE WHEN YEAR(date_stat) = {target_year} AND MONTH(date_stat) = {target_month} THEN arrivees ELSE 0 END) AS courant, "
            f"SUM(CASE WHEN YEAR(date_stat) = {year_prev} AND MONTH(date_stat) = {target_month} THEN arrivees ELSE 0 END) AS precedent "
            f"FROM {hbg} WHERE date_stat IS NOT NULL AND YEAR(date_stat) IN ({year_prev}, {target_year}) "
            f"AND MONTH(date_stat) = {target_month} "
            f"{market_hbg_clause}"
            f"{hbg_geo_clause}"
            f"GROUP BY nationalite_name) "
            f"SELECT TOP 8 pays, courant, precedent, courant - precedent AS delta "
            f"FROM x WHERE courant > 0 OR precedent > 0 ORDER BY ABS(courant - precedent) DESC",
        )

    apf_mre_tes = pd.DataFrame()
    if profile.segment_requested:
        apf_mre_tes = _safe_query(
            db_layer,
            f"WITH x AS ("
            f"SELECT "
            f"SUM(CASE WHEN YEAR(date_stat) = {target_year} AND MONTH(date_stat) = {target_month} THEN mre ELSE 0 END) AS courant_mre, "
            f"SUM(CASE WHEN YEAR(date_stat) = {year_prev} AND MONTH(date_stat) = {target_month} THEN mre ELSE 0 END) AS precedent_mre, "
            f"SUM(CASE WHEN YEAR(date_stat) = {target_year} AND MONTH(date_stat) = {target_month} THEN tes ELSE 0 END) AS courant_tes, "
            f"SUM(CASE WHEN YEAR(date_stat) = {year_prev} AND MONTH(date_stat) = {target_month} THEN tes ELSE 0 END) AS precedent_tes "
            f"FROM {apf} WHERE date_stat IS NOT NULL AND YEAR(date_stat) IN ({year_prev}, {target_year}) "
            f"AND MONTH(date_stat) = {target_month}) "
            f"SELECT 'MRE' AS segment, courant_mre AS courant, precedent_mre AS precedent, courant_mre - precedent_mre AS delta FROM x "
            f"UNION ALL "
            f"SELECT 'TES' AS segment, courant_tes AS courant, precedent_tes AS precedent, courant_tes - precedent_tes AS delta FROM x",
        )

    voie_apf = pd.DataFrame()
    if profile.voie_requested:
        voie_apf = _safe_query(
            db_layer,
            f"WITH x AS ("
            f"SELECT COALESCE(voie, 'Non renseignee') AS voie, "
            f"SUM(CASE WHEN YEAR(date_stat) = {target_year} AND MONTH(date_stat) = {target_month} THEN mre + tes ELSE 0 END) AS courant, "
            f"SUM(CASE WHEN YEAR(date_stat) = {year_prev} AND MONTH(date_stat) = {target_month} THEN mre + tes ELSE 0 END) AS precedent "
            f"FROM {apf} WHERE date_stat IS NOT NULL AND YEAR(date_stat) IN ({year_prev}, {target_year}) "
            f"AND MONTH(date_stat) = {target_month} "
            f"GROUP BY COALESCE(voie, 'Non renseignee')) "
            f"SELECT voie, courant, precedent, courant - precedent AS delta "
            f"FROM x WHERE courant > 0 OR precedent > 0 ORDER BY ABS(courant - precedent) DESC",
        )

    focus_apf = pd.DataFrame()
    focus_hbg = pd.DataFrame()
    if profile.focus_countries:
        focus_filter_apf = _country_sql_filter("nationalite", profile.focus_countries)
        focus_filter_hbg = _country_sql_filter("nationalite_name", profile.focus_countries)
        focus_apf = _safe_query(
            db_layer,
            f"WITH x AS ("
            f"SELECT nationalite AS pays, "
            f"SUM(CASE WHEN YEAR(date_stat) = {target_year} AND MONTH(date_stat) = {target_month} THEN mre + tes ELSE 0 END) AS courant, "
            f"SUM(CASE WHEN YEAR(date_stat) = {year_prev} AND MONTH(date_stat) = {target_month} THEN mre + tes ELSE 0 END) AS precedent "
            f"FROM {apf} WHERE date_stat IS NOT NULL AND YEAR(date_stat) IN ({year_prev}, {target_year}) "
            f"AND MONTH(date_stat) = {target_month} AND ({focus_filter_apf}) "
            f"GROUP BY nationalite) "
            f"SELECT pays, courant, precedent, courant - precedent AS delta "
            f"FROM x WHERE courant > 0 OR precedent > 0 ORDER BY pays",
        )
        focus_hbg = _safe_query(
            db_layer,
            f"WITH x AS ("
            f"SELECT nationalite_name AS pays, "
            f"SUM(CASE WHEN YEAR(date_stat) = {target_year} AND MONTH(date_stat) = {target_month} THEN arrivees ELSE 0 END) AS courant, "
            f"SUM(CASE WHEN YEAR(date_stat) = {year_prev} AND MONTH(date_stat) = {target_month} THEN arrivees ELSE 0 END) AS precedent "
            f"FROM {hbg} WHERE date_stat IS NOT NULL AND YEAR(date_stat) IN ({year_prev}, {target_year}) "
            f"AND MONTH(date_stat) = {target_month} AND ({focus_filter_hbg}) "
            f"{hbg_geo_clause}"
            f"GROUP BY nationalite_name) "
            f"SELECT pays, courant, precedent, courant - precedent AS delta "
            f"FROM x WHERE courant > 0 OR precedent > 0 ORDER BY pays",
        )

    return {
        "monthly_apf": monthly_apf,
        "monthly_hbg": monthly_hbg,
        "market_apf": market_apf,
        "market_hbg": market_hbg,
        "apf_mre_tes": apf_mre_tes,
        "voie_apf": voie_apf,
        "focus_apf": focus_apf,
        "focus_hbg": focus_hbg,
        "target": pd.DataFrame([{"annee": target_year, "mois": target_month}]),
    }


def _row_for(df: pd.DataFrame, year: int, month: int) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    sub = df[(df["annee"].astype(int) == int(year)) & (df["mois"].astype(int) == int(month))]
    if sub.empty:
        return None
    return sub.iloc[0]


def _metric_change(df: pd.DataFrame, current_year: int, month: int, metric: str) -> Dict[str, Any]:
    current = _row_for(df, current_year, month)
    previous = _row_for(df, current_year - 1, month)
    cur = float(current[metric]) if current is not None and metric in current else None
    prev = float(previous[metric]) if previous is not None and metric in previous else None
    pct = ((cur - prev) / prev * 100) if cur is not None and prev not in (None, 0) else None
    return {"current": cur, "previous": prev, "delta": None if cur is None or prev is None else cur - prev, "pct": pct}


def _ytd_change(df: pd.DataFrame, current_year: int, month: int, metric: str) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"current": None, "previous": None, "pct": None}
    cur_df = df[(df["annee"].astype(int) == current_year) & (df["mois"].astype(int) <= month)]
    prev_df = df[(df["annee"].astype(int) == current_year - 1) & (df["mois"].astype(int) <= month)]
    cur = float(pd.to_numeric(cur_df[metric], errors="coerce").fillna(0).sum()) if metric in df else None
    prev = float(pd.to_numeric(prev_df[metric], errors="coerce").fillna(0).sum()) if metric in df else None
    pct = ((cur - prev) / prev * 100) if cur is not None and prev not in (None, 0) else None
    return {"current": cur, "previous": prev, "pct": pct}


def _make_chart(sql_data: Dict[str, pd.DataFrame], profile: EventProfile) -> List[str]:
    hbg = sql_data.get("monthly_hbg")
    apf = sql_data.get("monthly_apf")
    target = sql_data.get("target")
    if hbg is None or hbg.empty or apf is None or apf.empty or target is None or target.empty:
        return []
    try:
        import plotly.graph_objects as go

        target_year = int(target.iloc[0]["annee"])
        target_month = int(target.iloc[0]["mois"])
        combined = pd.merge(
            apf[["annee", "mois", "arrivees_apf", "arrivees_apf_exposees"]],
            hbg[["annee", "mois", "arrivees_hotelieres", "nuitees"]],
            on=["annee", "mois"],
            how="outer",
        ).sort_values(["annee", "mois"])
        combined["periode"] = combined["annee"].astype(int).astype(str) + "-" + combined["mois"].astype(int).astype(str).str.zfill(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined["periode"], y=combined["arrivees_apf"], mode="lines+markers", name="Arrivees APF"))
        fig.add_trace(go.Scatter(x=combined["periode"], y=combined["arrivees_hotelieres"], mode="lines+markers", name="Arrivees hotelieres"))
        if "arrivees_apf_exposees" in combined:
            fig.add_trace(go.Bar(x=combined["periode"], y=combined["arrivees_apf_exposees"], name="APF segment expose", opacity=0.45))
        event_x = f"{target_year}-{target_month:02d}"
        fig.add_vline(x=event_x, line_width=2, line_dash="dash", line_color="#b91c1c")
        fig.update_layout(
            title=f"Test d'impact: {profile.label}",
            xaxis_title="Mois",
            yaxis_title="Volume",
            legend={"orientation": "h", "y": 1.08},
            template="plotly_white",
        )
        os.makedirs(CHARTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", profile.label).strip("_")[:36] or "impact_externe"
        path = os.path.join(CHARTS_DIR, f"chart_{ts}_{safe}.html").replace("\\", "/")
        fig.write_html(path, include_plotlyjs="cdn", full_html=True, config={"responsive": True, "displaylogo": False})
        return [path]
    except Exception as e:
        logger.debug("External impact chart failed: %s", e)
        return []


def _sources_markdown(sources: List[SourceItem]) -> str:
    if not sources:
        return "Aucune source web exploitable n'a ete retrouvee pendant la resolution de l'evenement."
    lines = []
    for s in sources[:5]:
        title = s.title or s.url or "Source"
        if s.url:
            lines.append(f"- [{title}]({s.url})")
        else:
            lines.append(f"- {title}")
    return "\n".join(lines)


def _market_table(df: pd.DataFrame, metric_name: str) -> str:
    if df is None or df.empty:
        return "Aucun detail pays disponible."
    out = df.copy()
    for col in ["courant", "precedent", "delta"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).round(0).astype(int)
    out = out.rename(columns={
        "pays": "Pays de residence",
        "courant": f"{metric_name} courant",
        "precedent": f"{metric_name} N-1",
        "delta": "Delta",
    })
    return out.to_markdown(index=False)


def _delta_table(df: pd.DataFrame, label_col: str, label_name: str, metric_name: str) -> str:
    if df is None or df.empty:
        return "Aucun detail disponible."
    out = df.copy()
    for col in ["courant", "precedent", "delta"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).round(0).astype(int)
    out = out.rename(columns={
        label_col: label_name,
        "courant": f"{metric_name} courant",
        "precedent": f"{metric_name} N-1",
        "delta": "Delta",
    })
    return out.to_markdown(index=False)


def _specialized_sections(profile: EventProfile, sql_data: Dict[str, pd.DataFrame]) -> List[str]:
    sections: List[str] = []
    if profile.segment_requested:
        sections.extend([
            "",
            "## MRE versus TES (APF)",
            _delta_table(sql_data.get("apf_mre_tes", pd.DataFrame()), "segment", "Segment", "Arrivees APF"),
        ])
    if profile.voie_requested:
        sections.extend([
            "",
            "## Sensibilite par voie APF",
            "La table hebergement ne contient pas la voie d'entree; cette decomposition est donc limitee aux arrivees APF.",
            _delta_table(sql_data.get("voie_apf", pd.DataFrame()), "voie", "Voie", "Arrivees APF"),
        ])
    if profile.focus_countries:
        countries = ", ".join(profile.focus_countries)
        sections.extend([
            "",
            f"## Pays de residence demandes: {countries}",
            "APF:",
            _market_table(sql_data.get("focus_apf", pd.DataFrame()), "APF"),
            "",
            "Hebergement:",
            _market_table(sql_data.get("focus_hbg", pd.DataFrame()), "Arrivees hotelieres"),
        ])
    return sections


def _synthesize_event(profile: EventProfile, sql_data: Dict[str, pd.DataFrame], chart_paths: List[str]) -> str:
    target = sql_data.get("target")
    if target is None or target.empty:
        return "Je n'ai pas pu etablir une fenetre d'analyse exploitable dans les donnees Fabric."
    year = int(target.iloc[0]["annee"])
    month = int(target.iloc[0]["mois"])
    month_name = _month_label(year, month)

    apf = sql_data.get("monthly_apf", pd.DataFrame())
    hbg = sql_data.get("monthly_hbg", pd.DataFrame())
    apf_total = _metric_change(apf, year, month, "arrivees_apf")
    apf_exposed = _metric_change(apf, year, month, "arrivees_apf_exposees")
    hbg_arr = _metric_change(hbg, year, month, "arrivees_hotelieres")
    hbg_nights = _metric_change(hbg, year, month, "nuitees")
    ytd_apf = _ytd_change(apf, year, month, "arrivees_apf")
    ytd_hbg = _ytd_change(hbg, year, month, "arrivees_hotelieres")
    test_months = [m for m in (profile.test_months or [month]) if 1 <= int(m) <= int(month)] or [month]

    start = profile.start_date.isoformat() if profile.start_date else "date non confirmee"
    if profile.start_date:
        first_label = (
            _month_label(profile.first_observed_year, profile.first_observed_month)
            if profile.first_observed_year and profile.first_observed_month
            else month_name
        )
        requested_line = (
            f" Mois analyses: **{', '.join(_month_label(year, m) for m in test_months)}**."
            if len(test_months) > 1 else ""
        )
        event_line = (
            f"Evenement retenu: **{profile.label}**, date de debut detectee autour du "
            f"**{start}**. Avec des donnees mensuelles, le premier mois proprement testable est **{first_label}**."
            f"{requested_line}"
        )
    else:
        if profile.analysis_year and profile.analysis_month:
            event_line = (
                f"Evenement retenu: **{profile.label}**, mais la date de debut n'a pas ete extraite avec assez de certitude. "
                f"J'utilise le mois demande pour le test: **{month_name}**."
            )
        else:
            event_line = (
                f"Evenement retenu: **{profile.label}**, mais la date de debut n'a pas ete extraite avec assez de certitude. "
                f"J'utilise donc le dernier mois commun disponible: **{month_name}**."
            )
    geo_line = (
        f"\nFiltre geographique applique aux donnees d'hebergement: **province_name ~ {profile.geo_label}**. "
        "Les donnees APF restent nationales, car l'APF ne mesure pas les nuitees par province."
        if profile.geo_label else ""
    )

    verdict_bits = []
    if apf_total["pct"] is not None:
        verdict_bits.append(f"APF total {month_name}: {_fmt_pct(apf_total['pct'])} vs N-1")
    if hbg_arr["pct"] is not None:
        verdict_bits.append(f"arrivees hotelieres {month_name}: {_fmt_pct(hbg_arr['pct'])} vs N-1")
    if apf_exposed["pct"] is not None:
        verdict_bits.append(f"segment expose APF: {_fmt_pct(apf_exposed['pct'])} vs N-1")

    has_yoy = any(v is not None for v in [apf_total["pct"], hbg_arr["pct"], hbg_nights["pct"], apf_exposed["pct"]])
    if not has_yoy:
        conclusion = "Impact non mesurable en YoY avec la couverture disponible: la base ne fournit pas de reference N-1 exploitable pour ce test."
    elif any(v and v < -5 for v in [apf_total["pct"], hbg_arr["pct"], apf_exposed["pct"]]):
        conclusion = "Signal negatif mesurable sur au moins un indicateur."
    elif any(v and v > 5 for v in [apf_total["pct"], hbg_arr["pct"], apf_exposed["pct"]]):
        conclusion = "Pas de choc negatif agrege; les donnees montrent plutot une resistance ou un report positif sur certains indicateurs."
    else:
        conclusion = "Signal agrege limite: aucune rupture forte ne ressort du premier test mensuel."

    metric_rows: List[str] = []
    for m in test_months:
        m_name = _month_label(year, m)
        apf_total_m = _metric_change(apf, year, m, "arrivees_apf")
        apf_exposed_m = _metric_change(apf, year, m, "arrivees_apf_exposees")
        hbg_arr_m = _metric_change(hbg, year, m, "arrivees_hotelieres")
        hbg_nights_m = _metric_change(hbg, year, m, "nuitees")
        metric_rows.extend([
            f"| Arrivees APF | {m_name} | {_fmt_num(apf_total_m['current'])} | {_fmt_num(apf_total_m['previous'])} | {_fmt_pct(apf_total_m['pct'])} |",
            f"| Arrivees APF - segment expose | {m_name} | {_fmt_num(apf_exposed_m['current'])} | {_fmt_num(apf_exposed_m['previous'])} | {_fmt_pct(apf_exposed_m['pct'])} |",
            f"| Arrivees hotelieres | {m_name} | {_fmt_num(hbg_arr_m['current'])} | {_fmt_num(hbg_arr_m['previous'])} | {_fmt_pct(hbg_arr_m['pct'])} |",
            f"| Nuitees | {m_name} | {_fmt_num(hbg_nights_m['current'])} | {_fmt_num(hbg_nights_m['previous'])} | {_fmt_pct(hbg_nights_m['pct'])} |",
        ])

    lines = [
        "## Analyse d'impact externe",
        event_line + geo_line,
        "",
        f"**Verdict analytique:** {conclusion} Ce resultat reste une lecture de signal, pas une preuve causale definitive.",
        "",
        "## Test statistique minimal",
        "| Indicateur | Mois teste | Valeur | N-1 | Variation |",
        "| :--- | :--- | ---: | ---: | ---: |",
        *metric_rows,
        "",
        f"En cumul Janvier-{MONTHS_FR.get(month, month)} {year}, les arrivees APF sont a **{_fmt_pct(ytd_apf['pct'])}** vs N-1 et les arrivees hotelieres a **{_fmt_pct(ytd_hbg['pct'])}** vs N-1.",
        *_specialized_sections(profile, sql_data),
        "",
        "## Hypotheses metier testees",
    ]
    lines.extend(f"- {h}" for h in profile.hypotheses)
    lines.extend([
        "",
        "## Marches qui bougent le plus",
        "APF par pays de residence:",
        _market_table(sql_data.get("market_apf", pd.DataFrame()), "APF"),
        "",
        "Hebergement par pays de residence:",
        _market_table(sql_data.get("market_hbg", pd.DataFrame()), "Arrivees hotelieres"),
        "",
        "## Limites",
        f"- `date_stat` est mensuel: un evenement en fin de mois se voit surtout le mois suivant.",
        "- L'impact causal exige de suivre plusieurs mois et de comparer a des marches temoins.",
        "- APF et hebergement mesurent deux perimetres distincts; ils peuvent diverger.",
        "",
        "## Sources web utilisees pour dater/contextualiser l'evenement",
        _sources_markdown(profile.sources),
        "",
        f"Perimetres: {APF_SCOPE_LABEL}; {HEBERGEMENT_SCOPE_LABEL}.",
    ])
    if chart_paths:
        lines.append(f"\nChart: {chart_paths[0]}")
    return "\n".join(lines)


def _synthesize_factors(profile: EventProfile, sql_data: Dict[str, pd.DataFrame], chart_paths: List[str]) -> str:
    target = sql_data.get("target")
    if target is None or target.empty:
        return "Je n'ai pas pu recuperer la couverture Fabric necessaire pour diagnostiquer les facteurs externes."
    year = int(target.iloc[0]["annee"])
    month = int(target.iloc[0]["mois"])
    month_name = _month_label(year, month)

    apf = sql_data.get("monthly_apf", pd.DataFrame())
    hbg = sql_data.get("monthly_hbg", pd.DataFrame())
    apf_total = _metric_change(apf, year, month, "arrivees_apf")
    hbg_arr = _metric_change(hbg, year, month, "arrivees_hotelieres")
    hbg_nights = _metric_change(hbg, year, month, "nuitees")

    factors = [
        ("Geopolitique et securite regionale", "guerre/conflit, perception de risque, reallocation des destinations concurrentes"),
        ("Transport aerien", "fermeture d'espaces aeriens, annulations, capacite sieges, prix des billets"),
        ("Macro-economie", "inflation, change, pouvoir d'achat, prix du petrole"),
        ("Calendrier et saisonnalite", "Ramadan, vacances scolaires, evenements et salons"),
        ("Concurrence mediterraneenne", "Egypte, Turquie, Espagne, Portugal, Tunisie selon les marches sources"),
        ("Offre locale", "capacite, prix hotelier, qualite de service, connectivite region/province"),
    ]

    lines = [
        "## Diagnostic des facteurs externes possibles",
        f"Point de depart quantitatif: dernier mois commun analyse = **{month_name}**.",
        "",
        "| Indicateur | Valeur | N-1 | Variation |",
        "| :--- | ---: | ---: | ---: |",
        f"| Arrivees APF | {_fmt_num(apf_total['current'])} | {_fmt_num(apf_total['previous'])} | {_fmt_pct(apf_total['pct'])} |",
        f"| Arrivees hotelieres | {_fmt_num(hbg_arr['current'])} | {_fmt_num(hbg_arr['previous'])} | {_fmt_pct(hbg_arr['pct'])} |",
        f"| Nuitees | {_fmt_num(hbg_nights['current'])} | {_fmt_num(hbg_nights['previous'])} | {_fmt_pct(hbg_nights['pct'])} |",
        *_specialized_sections(profile, sql_data),
        "",
        "## Grille metier a appliquer",
    ]
    lines.extend(f"- **{name}**: {why}." for name, why in factors)
    if profile.market_requested:
        lines.extend([
            "",
            "## Marches a examiner en priorite",
            "APF par pays de residence:",
            _market_table(sql_data.get("market_apf", pd.DataFrame()), "APF"),
            "",
            "Hebergement par pays de residence:",
            _market_table(sql_data.get("market_hbg", pd.DataFrame()), "Arrivees hotelieres"),
        ])
    lines.extend([
        "",
        "## Lecture",
        "Un facteur externe ne doit etre retenu que s'il coche trois cases: proximite temporelle, mecanisme plausible, et signal observe dans APF ou hebergement. Sinon il reste une hypothese de veille.",
        "",
        "## Sources web pour la veille facteurs",
        _sources_markdown(profile.sources),
        "",
        f"Perimetres: {APF_SCOPE_LABEL}; {HEBERGEMENT_SCOPE_LABEL}.",
    ])
    if chart_paths:
        lines.append(f"\nChart: {chart_paths[0]}")
    return "\n".join(lines)


def _synthesize_hypothesis_scenario(message: str) -> str:
    return "\n".join([
        "## Hypotheses metier a tester",
        "La question decrit une divergence de perimetres: les arrivees APF peuvent augmenter sans que les nuitees progressent. Je ne traite donc pas cela comme un KPI simple, mais comme un diagnostic causal.",
        "",
        "## Hypotheses prioritaires",
        "- **Sejours plus courts**: les arrivees augmentent, mais la DMS baisse; test = nuitees / arrivees hotelieres par mois, region et pays de residence.",
        "- **Fuite hors hebergement classe**: plus de visiteurs logent en famille, location courte duree ou non classe; test = APF vs arrivees hotelieres par pays de residence et region.",
        "- **Poids MRE ou transit**: une hausse APF peut venir de retours MRE, visites familiales ou transit sans consommation hoteliere; test = MRE/TES, voie et poste frontiere.",
        "- **Mix geographique defavorable**: les points d'entree progressent, mais les destinations hotelieres principales ne captent pas le flux; test = regions/provinces hebergement vs postes/voies APF.",
        "- **Contraintes d'offre et prix**: capacite, prix hotelier ou disponibilite limitent la conversion en nuitees; test = categories, type d'hebergement, capacite et DMS.",
        "- **Effet calendrier**: Ramadan, vacances, salons ou jours feries peuvent deplacer les arrivees sans allonger les sejours; test = comparaison mois a mois vs N-1.",
        "",
        "## Ordre d'analyse recommande",
        "1. Confirmer le perimetre: APF = franchissements frontiere; hebergement = check-ins et nuitees en etablissements classes.",
        "2. Comparer le meme mois en N et N-1, puis le cumul YTD sur les mois disponibles.",
        "3. Segmenter par pays de residence, MRE/TES, voie APF, region/province, categorie et type d'hebergement.",
        "4. Conclure seulement si le signal est coherent dans le temps et dans au moins deux dimensions metier.",
        "",
        f"Perimetres: {APF_SCOPE_LABEL}; {HEBERGEMENT_SCOPE_LABEL}.",
    ])


def run_external_impact_analysis(
    message: str,
    db_layer=None,
    search_tool=None,
    conversation_context: str = "",
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    start = time.time()
    rid = run_id or f"impact_{datetime.now().strftime('%H%M%S')}"
    trace: List[Dict[str, Any]] = []

    if _is_hypothesis_scenario(message):
        t_h = time.time()
        response = _synthesize_hypothesis_scenario(message)
        trace.append({
            "stage": "hypothesis_framework",
            "label": "Diagnostic causal sans requete lourde",
            "status": "done",
            "agent": "external_impact",
            "detail": "Question hypothetique APF vs hebergement; grille metier directe.",
            "duration_ms": round((time.time() - t_h) * 1000, 1),
        })
        total_ms = (time.time() - start) * 1000
        return {
            "agent": "researcher",
            "agent_icon": "",
            "agent_name": "Analyste d'Impact Externe",
            "response": response,
            "chart_path": None,
            "chart_paths": [],
            "sources": [],
            "confidence": "85%",
            "data_freshness": {},
            "rerouted": False,
            "classification_time_ms": 0.0,
            "total_time_ms": round(total_ms, 1),
            "metric_context": "external_impact",
            "period": {"event_start": None, "first_observed_month": None},
            "data_scope_note": f"{APF_SCOPE_LABEL}; {HEBERGEMENT_SCOPE_LABEL}; diagnostic metier",
            "run_id": rid,
            "trace": trace,
            "fallbacks": [],
            "errors": [],
        }

    t0 = time.time()
    profile = _build_event_profile(message, search_tool=search_tool)
    trace.append({
        "stage": "event_resolution",
        "label": "Resolution de l'evenement externe",
        "status": "done",
        "agent": "external_impact",
        "detail": (
            f"{profile.label}; debut="
            f"{profile.start_date.isoformat() if profile.start_date else 'non confirme'}; "
            f"requete web: {profile.search_query[:120]}"
        ),
        "duration_ms": round((time.time() - t0) * 1000, 1),
    })

    t1 = time.time()
    sql_data = _impact_sql(db_layer, profile)
    trace.append({
        "stage": "impact_sql",
        "label": "Tests SQL APF et hebergement",
        "status": "done" if sql_data else "partial",
        "agent": "analytics",
        "detail": "Comparaison YoY, cumul YTD et marches de residence les plus contributeurs.",
        "duration_ms": round((time.time() - t1) * 1000, 1),
    })

    t2 = time.time()
    chart_paths = _make_chart(sql_data, profile)
    trace.append({
        "stage": "impact_chart",
        "label": "Visualisation du signal",
        "status": "done" if chart_paths else "skipped",
        "agent": "analytics",
        "detail": "Courbe APF/hebergement autour de la fenetre d'impact.",
        "duration_ms": round((time.time() - t2) * 1000, 1),
        "artifact_count": len(chart_paths),
    })

    t3 = time.time()
    if profile.mode == "factor_discovery":
        response = _synthesize_factors(profile, sql_data, chart_paths)
    else:
        response = _synthesize_event(profile, sql_data, chart_paths)
    trace.append({
        "stage": "impact_synthesis",
        "label": "Synthese metier controlee",
        "status": "done",
        "agent": "external_impact",
        "detail": "Signal observe, hypotheses, limites et sources.",
        "duration_ms": round((time.time() - t3) * 1000, 1),
    })

    total_ms = (time.time() - start) * 1000
    return {
        "agent": "researcher" if profile.mode == "factor_discovery" else "analytics",
        "agent_icon": "",
        "agent_name": "Analyste d'Impact Externe",
        "response": response,
        "chart_path": chart_paths[0] if chart_paths else None,
        "chart_paths": chart_paths,
        "sources": [{"title": s.title, "url": s.url} for s in profile.sources if s.url],
        "confidence": f"{profile.confidence:.0%}",
        "data_freshness": {},
        "rerouted": False,
        "classification_time_ms": 0.0,
        "total_time_ms": round(total_ms, 1),
        "metric_context": "external_impact",
        "period": {
            "event_start": profile.start_date.isoformat() if profile.start_date else None,
            "first_observed_month": (
                f"{profile.first_observed_year}-{profile.first_observed_month:02d}"
                if profile.first_observed_year and profile.first_observed_month else None
            ),
        },
        "data_scope_note": f"{APF_SCOPE_LABEL}; {HEBERGEMENT_SCOPE_LABEL}; veille externe",
        "run_id": rid,
        "trace": trace,
        "fallbacks": [],
        "errors": [],
    }
