"""
STATOUR Intent Extractor
=========================
Pure-Python intent analyzer for tourism chatbot user messages.

No LLM calls. No external dependencies (stdlib only). Sub-10ms execution.

Public API:
    extractor = IntentExtractor()
    intent = extractor.extract("pourquoi la baisse des arrivées en juillet 2025")
    queries = extractor.build_search_queries(intent)

Used by:
    - PredictionAgent: dynamic factor-research query generation
    - (Future) ResearcherAgent: targeted multi-query specialization
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Public dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentContext:
    period_month: Optional[int] = None
    period_year: Optional[int] = None
    metric_type: str = "general"
    analysis_type: str = "factual"
    geo_scope: str = "national"
    detected_entities: List[str] = field(default_factory=list)
    external_factors_categories: List[str] = field(default_factory=list)
    comparison_years: Optional[Tuple[int, int]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MONTH_NAMES_FR: Dict[int, str] = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "août",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre",
}

# ── Data currency — update when new data months are loaded ──
# APF (fact_statistiques_apf): Jan–Feb 2026 confirmed (knowledge base 2026-05-05).
# Hébergement (fact_statistiqueshebergementnationaliteestimees): may have more 2026 months.
# _LAST_DATA_MONTH reflects APF coverage only — not hébergement.
_CURRENT_YEAR: int = 2026
_LAST_DATA_MONTH: int = 2   # February — APF confirmed cutoff
_LAST_DATA_YEAR: int = 2026

_MONTH_PARSE: Dict[str, int] = {
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

# Metric types — order matters: first match wins (most specific first).
# "taux d'occupation" beats "hôtel" so taux_occupation is checked before hebergement_*.
_METRIC_KEYWORDS: Dict[str, List[str]] = {
    "taux_occupation": [
        r"taux\s+d['’]\s*occupation",
        r"taux\s+remplissage",
        r"occupancy",
        r"\bto\b",
    ],
    "hebergement_arrivees": [
        r"arriv[ée]es?\s+h[ôo]teli[èe]res?",
        r"arriv[ée]es?\s+h[ée]bergement",
        r"check[\s\-]?in",
        r"d[ée]claration",
        r"ehtc\s+arriv[ée]e",
    ],
    "hebergement_nuitees": [
        r"nuit[ée]es?",
        r"h[ée]bergement",
        r"\bh[ôo]tel\b",
        r"\behtc\b",
        r"\bstdn\b",
        r"\bnuit\b",
        r"\bchambre\b",
        r"\bcapacit[ée]\b",
        r"maison\s+d['’]\s*h[ôo]te[s]?",
        r"\bcamping\b",
        r"\briad\b",
    ],
    "recettes": [
        r"\brecettes?\b",
        r"\brevenus?\b",
        r"recette\s+touristique",
        r"office\s+des\s+changes",
    ],
    "apf_arrivees": [
        r"\bmre\b",
        r"\btes\b",
        r"postes?\s+fronti[èe]res?",
        r"\bapf\b",
        r"\bvoie\b",
        r"fronti[èe]re",
        r"voie\s+a[ée]rienne",
        r"voie\s+maritime",
        r"voie\s+terrestre",
        r"\bcontinent\b",
        r"\bdgsn\b",
    ],
}

# Analysis types — priority: causal > comparative > forecasting > trend > factual.
# Verified by test case 7 ("impact coupe du monde 2030" → causal, not forecasting).
_ANALYSIS_KEYWORDS: Dict[str, List[str]] = {
    "causal": [
        r"\bpourquoi\b",
        r"expliqu",
        r"\bfacteur",
        r"\bcause",
        r"\braison",
        r"\bcontexte\b",
        r"\bimpact",
        r"\binfluence",
        r"qu['’]\s*est[\s\-]ce\s+qui",
    ],
    "comparative": [
        r"\bcompar",
        r"\bversus\b",
        r"\bvs\b",
        r"\bentre\b",
        r"\bdiff[ée]rence",
        r"par\s+rapport",
        r"[ée]volution\s+relative",
    ],
    "forecasting": [
        r"\bestim",
        r"\bpr[ée]vision",
        r"\bpr[ée]voir",
        r"\bforecast",
        r"\b202[7-9]\b",
        r"\b2030\b",
        r"\bfutur",
        r"\bprojection",
    ],
    "trend": [
        r"\b[ée]volution\b",
        r"\btendance",
        r"\btrend\b",
        r"\bhistorique\b",
        r"\bdepuis\b",
        r"\bprogression\b",
        r"\bcroissance\b",
        r"\bbaisse\b",
        r"\bhausse\b",
    ],
}

# Geo scope — priority: regional > provincial > by_voie > by_nationality > national.
_GEO_KEYWORDS: Dict[str, List[str]] = {
    "regional": [
        r"\br[ée]gion",
        r"\bmarrakech\b",
        r"\bcasablanca\b",
        r"\btanger\b",
        r"\bagadir\b",
        r"\bf[èe]s\b",
        r"\brabat\b",
        r"\bsouss\b",
        r"\boriental\b",
        r"\bdakhla\b",
        r"\blaayoune\b",
    ],
    "provincial": [
        r"\bprovince",
        r"\bd[ée]l[ée]gation",
        r"\bprovincial",
    ],
    "by_voie": [
        r"\bvoie\b",
        r"\ba[ée]rien",
        r"\bmaritime\b",
        r"\bterrestre\b",
        r"\ba[ée]roport",
        r"\bport\b",
        r"fronti[èe]re\s+terrestre",
    ],
    "by_nationality": [
        r"pays\s+de\s+r[ée]sidence",
        r"\bnationalit[ée]",
    ],
}

# Named entities — value list = surface forms; key = canonical name stored in detected_entities.
_ENTITY_PATTERNS: Dict[str, List[str]] = {
    # Pays émetteurs
    "france":       [r"\bfrance\b", r"\bfran[çc]ais(e|es)?\b"],
    "espagne":      [r"\bespagne\b", r"\bespagnol(e|es|s)?\b"],
    "royaume-uni":  [r"\broyaume[\s\-]uni\b", r"\bbritannique[s]?\b", r"\banglais(e|es)?\b", r"\buk\b"],
    "italie":       [r"\bitalie\b", r"\bitalien(ne|nes|s)?\b"],
    "belgique":     [r"\bbelgique\b", r"\bbelge[s]?\b"],
    "allemagne":    [r"\ballemagne\b", r"\ballemand(e|es|s)?\b"],
    "usa":          [r"\busa\b", r"[ée]tats[\s\-]unis", r"\bam[ée]ricain(e|es|s)?\b"],
    "canada":       [r"\bcanada\b", r"\bcanadien(ne|nes|s)?\b"],
    "pays-bas":     [r"pays[\s\-]bas", r"\bhollande\b", r"\bn[ée]erlandais(e|es)?\b"],
    # Régions Maroc
    "marrakech":    [r"\bmarrakech\b"],
    "casablanca":   [r"\bcasablanca\b"],
    "tanger":       [r"\btanger\b"],
    "agadir":       [r"\bagadir\b"],
    "fès":          [r"\bf[èe]s\b"],
    "rabat":        [r"\brabat\b"],
    "souss":        [r"\bsouss\b"],
    "oriental":     [r"\boriental\b"],
    "dakhla":       [r"\bdakhla\b"],
    "laayoune":     [r"\blaayoune\b"],
    # Voies
    "aérienne":     [r"\ba[ée]rien(ne)?\b"],
    "maritime":     [r"\bmaritime\b"],
    "terrestre":    [r"\bterrestre\b"],
    # Compagnies
    "ryanair":          [r"\bryanair\b"],
    "royal air maroc":  [r"royal\s+air\s+maroc", r"\bram\b"],
    "easyjet":          [r"\beasyjet\b"],
    "transavia":        [r"\btransavia\b"],
    # Postes frontières
    "mohammed v":   [r"mohammed\s+v"],
    "ménara":       [r"\bm[ée]nara\b"],
    "tanger med":   [r"tanger\s+med"],
    "bab sebta":    [r"bab\s+sebta"],
    "béni anzar":   [r"b[ée]ni\s+anzar"],
    "nador":        [r"\bnador\b"],
}

# Country entities (used for by_nationality fallback)
_COUNTRY_ENTITIES = frozenset({
    "france", "espagne", "royaume-uni", "italie", "belgique", "allemagne",
    "usa", "canada", "pays-bas",
})

# Region entities (used for regional geo_scope fallback)
_REGION_ENTITIES = frozenset({
    "marrakech", "casablanca", "tanger", "agadir", "fès", "rabat",
    "souss", "oriental", "dakhla", "laayoune",
})

# Category → (display label, query template with {period} placeholder)
CATEGORY_TO_QUERY: Dict[str, Tuple[str, str]] = {
    "conjoncture_economique_europe":     ("Conjoncture économique EU",   "conjoncture économique europe {period} impact tourisme"),
    "connectivite_aerienne_maroc":       ("Connectivité aérienne Maroc", "connectivité aérienne maroc {period} nouvelles routes compagnies"),
    "operation_marhaba_mre":             ("Opération Marhaba MRE",       "opération marhaba maroc {year} mre marocains résidant étranger"),
    "trafic_aerien_ete":                 ("Trafic aérien été",           "trafic aérien maroc été {year} ryanair royal air maroc easyjet"),
    "marches_emetteurs_vacances":        ("Marchés émetteurs vacances",  "tourisme france espagne maroc vacances {period}"),
    "tourisme_hivernal_maroc":           ("Tourisme hivernal",           "tourisme maroc hiver {period} marrakech agadir destinations"),
    "destinations_concurrentes_mediterranee": ("Concurrence méditerranée", "tourisme méditerranée {year} concurrence turquie egypte maroc"),
    "arrivees_postes_frontieres_maroc":  ("APF Arrivées",                "arrivées postes frontières maroc {period} statistiques"),
    "taux_occupation_hotels_maroc":      ("Taux occupation hôtels",      "taux occupation hôtels maroc {period} hébergement touristique"),
    "declarations_ehtc":                 ("Déclarations EHTC",           "etablissements hébergement touristique classés maroc {year}"),
    "touristes_francais_maroc":          ("Touristes français",          "touristes français maroc {period} séjours arrivées"),
    "liaisons_aeriennes_france_maroc":   ("Liaisons FR-MA",              "liaisons aériennes france maroc {period} vols"),
    "touristes_espagnols_maroc":         ("Touristes espagnols",         "touristes espagnols maroc {period}"),
    "ferries_espagne_maroc":             ("Ferries Espagne-Maroc",       "ferries espagne maroc {period} trafic maritime"),
    "diaspora_marocaine":                ("Diaspora marocaine",          "diaspora marocaine {year} transferts retours maroc"),
    "operation_marhaba":                 ("Opération Marhaba",           "opération marhaba {year} maroc bilan résultats"),
    "vision_2030_tourisme_maroc":        ("Vision 2030",                 "vision 2030 tourisme maroc objectifs stratégie"),
    "objectifs_mtaess":                  ("Objectifs MTAESS",            "ministère tourisme maroc mtaess {year} objectifs tourisme"),
    "coupe_monde_2030_maroc":            ("Coupe du Monde 2030",         "coupe monde 2030 maroc espagne portugal impact tourisme"),
    "can_maroc":                         ("CAN Maroc",                   "coupe afrique nations maroc impact tourisme arrivées"),
    "tourisme_marrakech_tendances":      ("Tourisme Marrakech",          "tourisme marrakech {period} tendances arrivées hôtels"),
    "tourisme_agadir_balneaire":         ("Tourisme Agadir",             "tourisme balnéaire agadir {period} arrivées saison"),
}

# Maximum queries returned by build_search_queries()
_MAX_QUERIES = 5


# ──────────────────────────────────────────────────────────────────────────────
# Compile patterns once at module load
# ──────────────────────────────────────────────────────────────────────────────

def _compile_dict(d: Dict[str, List[str]]) -> Dict[str, List[Pattern[str]]]:
    return {key: [re.compile(p, re.IGNORECASE) for p in patterns]
            for key, patterns in d.items()}


_METRIC_RE   = _compile_dict(_METRIC_KEYWORDS)
_ANALYSIS_RE = _compile_dict(_ANALYSIS_KEYWORDS)
_GEO_RE      = _compile_dict(_GEO_KEYWORDS)
_ENTITY_RE   = _compile_dict(_ENTITY_PATTERNS)

_YEAR_RE = re.compile(r"\b(20[2-9]\d)\b")
_COMPARISON_YEARS_RE = re.compile(
    r"\b(20[2-9]\d)\s*(?:vs|versus|/|à|et|-|–|—)\s*(20[2-9]\d)\b",
    re.IGNORECASE,
)
_NUMERIC_MONTH_YEAR_RE = re.compile(r"\b(1[0-2]|[1-9])/20[2-9]\d\b")
_MONTH_RES = [(re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE), num)
              for name, num in _MONTH_PARSE.items()]


def _matches_any(text: str, patterns: List[Pattern[str]]) -> bool:
    return any(p.search(text) for p in patterns)


# ──────────────────────────────────────────────────────────────────────────────
# IntentExtractor
# ──────────────────────────────────────────────────────────────────────────────

class IntentExtractor:
    """Stateless rule-based intent classifier for tourism queries."""

    # ── Public API ──────────────────────────────────────────────────────────

    def extract(self, message: str) -> IntentContext:
        msg = message.lower()

        period_month, period_year = self._extract_period(msg, message)
        comparison_years = self._extract_comparison_years(message)
        entities = self._extract_entities(msg)
        metric_type = self._extract_metric_type(msg)
        analysis_type = self._extract_analysis_type(msg)
        geo_scope = self._extract_geo_scope(msg, entities)

        intent = IntentContext(
            period_month=period_month,
            period_year=period_year,
            metric_type=metric_type,
            analysis_type=analysis_type,
            geo_scope=geo_scope,
            detected_entities=entities,
            comparison_years=comparison_years,
        )
        intent.external_factors_categories = self._build_factor_categories(intent, msg)
        return intent

    def build_search_queries(self, intent: IntentContext) -> List[Tuple[str, str]]:
        """Map factor categories → (label, query_string) pairs, capped at 5."""
        period = self._format_period(intent.period_month, intent.period_year)
        year_str = str(intent.period_year) if intent.period_year else "récent"

        queries: List[Tuple[str, str]] = []
        for category in intent.external_factors_categories:
            spec = CATEGORY_TO_QUERY.get(category)
            if not spec:
                continue
            label, template = spec
            query = template.format(period=period, year=year_str)
            queries.append((label, query))

        return queries[:_MAX_QUERIES]

    # ── Period extraction ───────────────────────────────────────────────────

    def _extract_period(self, msg_lower: str, msg_raw: str) -> Tuple[Optional[int], Optional[int]]:
        # Step 1: Relative temporal phrases — pure substring lookup, no LLM.
        # Relative phrase wins over any bare 20XX year in the same message.
        _REL_MAP = {
            "cette année":        (_CURRENT_YEAR,         None),
            "cette annee":        (_CURRENT_YEAR,         None),
            "l'année en cours":   (_CURRENT_YEAR,         None),
            "l annee en cours":   (_CURRENT_YEAR,         None),
            "l'année dernière":   (_CURRENT_YEAR - 1,     None),
            "l annee derniere":   (_CURRENT_YEAR - 1,     None),
            "année dernière":     (_CURRENT_YEAR - 1,     None),
            "annee derniere":     (_CURRENT_YEAR - 1,     None),
            "l'an dernier":       (_CURRENT_YEAR - 1,     None),
            "an dernier":         (_CURRENT_YEAR - 1,     None),
            "l'année prochaine":  (_CURRENT_YEAR + 1,     None),
            "l annee prochaine":  (_CURRENT_YEAR + 1,     None),
            "année prochaine":    (_CURRENT_YEAR + 1,     None),
            "annee prochaine":    (_CURRENT_YEAR + 1,     None),
            "l'an prochain":      (_CURRENT_YEAR + 1,     None),
            "an prochain":        (_CURRENT_YEAR + 1,     None),
            "dernier mois":       (_LAST_DATA_YEAR,  _LAST_DATA_MONTH),
            "mois dernier":       (_LAST_DATA_YEAR,  _LAST_DATA_MONTH),
            "le mois dernier":    (_LAST_DATA_YEAR,  _LAST_DATA_MONTH),
            "mois précédent":     (_LAST_DATA_YEAR,  _LAST_DATA_MONTH),
            "mois precedent":     (_LAST_DATA_YEAR,  _LAST_DATA_MONTH),
            "dernière période":   (_LAST_DATA_YEAR,  _LAST_DATA_MONTH),
            "derniere periode":   (_LAST_DATA_YEAR,  _LAST_DATA_MONTH),
            "données récentes":   (_LAST_DATA_YEAR,  _LAST_DATA_MONTH),
            "donnees recentes":   (_LAST_DATA_YEAR,  _LAST_DATA_MONTH),
        }
        for phrase, (yr, mo) in _REL_MAP.items():
            if phrase in msg_lower:
                return mo, yr  # early return; relative phrase wins

        # Step 2: Bare 20XX scan (existing logic, unchanged)
        # Year — pick the LAST 20XX mentioned (most relevant when comparing)
        years = _YEAR_RE.findall(msg_raw)
        period_year = int(years[-1]) if years else None

        # Month — first month name match (word-boundary)
        period_month: Optional[int] = None
        for pattern, num in _MONTH_RES:
            if pattern.search(msg_lower):
                period_month = num
                break

        # Numeric month/year fallback ("7/2025")
        if period_month is None:
            m = _NUMERIC_MONTH_YEAR_RE.search(msg_raw)
            if m:
                period_month = int(m.group(1))

        return period_month, period_year

    def _extract_comparison_years(self, msg_raw: str) -> Optional[Tuple[int, int]]:
        m = _COMPARISON_YEARS_RE.search(msg_raw)
        if not m:
            return None
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y1 == y2:
            return None
        return (min(y1, y2), max(y1, y2))

    # ── Categorical extractors ──────────────────────────────────────────────

    def _extract_metric_type(self, msg: str) -> str:
        for metric in ("taux_occupation", "hebergement_arrivees",
                       "hebergement_nuitees", "recettes", "apf_arrivees"):
            if _matches_any(msg, _METRIC_RE[metric]):
                return metric
        return "general"

    def _extract_analysis_type(self, msg: str) -> str:
        # Priority: causal > comparative > forecasting > trend > factual
        for kind in ("causal", "comparative", "forecasting", "trend"):
            if _matches_any(msg, _ANALYSIS_RE[kind]):
                return kind
        return "factual"

    def _extract_geo_scope(self, msg: str, entities: List[str]) -> str:
        # Priority: regional > provincial > by_voie > by_nationality > national
        if _matches_any(msg, _GEO_RE["regional"]):
            return "regional"
        if any(e in _REGION_ENTITIES for e in entities):
            return "regional"
        if _matches_any(msg, _GEO_RE["provincial"]):
            return "provincial"
        if _matches_any(msg, _GEO_RE["by_voie"]):
            return "by_voie"
        if _matches_any(msg, _GEO_RE["by_nationality"]):
            return "by_nationality"
        if any(e in _COUNTRY_ENTITIES for e in entities):
            return "by_nationality"
        return "national"

    def _extract_entities(self, msg: str) -> List[str]:
        found: List[str] = []
        for canonical, patterns in _ENTITY_RE.items():
            if _matches_any(msg, patterns):
                if canonical not in found:
                    found.append(canonical)
        return found

    # ── Factor category builder (the 11-rule pipeline) ──────────────────────

    def _build_factor_categories(self, intent: IntentContext, msg: str) -> List[str]:
        categories: List[str] = []

        def add(*items: str) -> None:
            for item in items:
                if item not in categories:
                    categories.append(item)

        analysis = intent.analysis_type
        month = intent.period_month
        year = intent.period_year
        metric = intent.metric_type
        entities = intent.detected_entities

        # Rule 1 — causal/trend always get base economic + connectivity context
        if analysis in ("causal", "trend"):
            add("conjoncture_economique_europe", "connectivite_aerienne_maroc")

        # Rule 2 — causal + summer months
        if analysis == "causal" and month in (6, 7, 8):
            add("operation_marhaba_mre", "trafic_aerien_ete", "marches_emetteurs_vacances")

        # Rule 3 — causal + winter months
        if analysis == "causal" and month in (12, 1, 2, 3):
            add("tourisme_hivernal_maroc", "destinations_concurrentes_mediterranee")

        # Rule 4 — APF metric
        if metric == "apf_arrivees":
            add("arrivees_postes_frontieres_maroc")

        # Rule 5 — Hébergement metrics
        if metric in ("hebergement_nuitees", "hebergement_arrivees"):
            add("taux_occupation_hotels_maroc", "declarations_ehtc")

        # Rule 6 — France entity
        if "france" in entities:
            add("touristes_francais_maroc", "liaisons_aeriennes_france_maroc")

        # Rule 7 — Espagne entity
        if "espagne" in entities:
            add("touristes_espagnols_maroc", "ferries_espagne_maroc")

        # Rule 8 — MRE mentioned (or APF + MRE detected)
        if re.search(r"\bmre\b", msg) or (metric == "apf_arrivees" and re.search(r"\bmre\b", msg)):
            add("diaspora_marocaine", "operation_marhaba")

        # Rule 9 — Forecasting
        if analysis == "forecasting":
            add("vision_2030_tourisme_maroc", "objectifs_mtaess")
            if year == 2030 or "2030" in msg:
                add("coupe_monde_2030_maroc")
            if re.search(r"\bcan\b", msg) or "coupe afrique" in msg or "coupe d'afrique" in msg:
                add("can_maroc")

        # Rule 10 — Marrakech regional
        if intent.geo_scope == "regional" and "marrakech" in entities:
            add("tourisme_marrakech_tendances")

        # Rule 11 — Agadir regional
        if intent.geo_scope == "regional" and "agadir" in entities:
            add("tourisme_agadir_balneaire")

        return categories

    # ── Period formatter ────────────────────────────────────────────────────

    def _format_period(self, month: Optional[int], year: Optional[int]) -> str:
        if month and year:
            return f"{MONTH_NAMES_FR[month]} {year}"
        if year:
            return str(year)
        return "récent"


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("pourquoi la baisse des arrivées en juillet 2025",
         {"analysis_type": "causal", "period_month": 7, "period_year": 2025}),

        ("combien de touristes français en 2024",
         {"analysis_type": "factual", "metric_type": "general", "period_year": 2024}),

        ("nuitées par région en 2024",
         {"metric_type": "hebergement_nuitees", "geo_scope": "regional"}),

        ("estimation flux touristique 2027",
         {"analysis_type": "forecasting", "period_year": 2027}),

        ("MRE en juillet 2025 opération marhaba",
         {"period_month": 7, "period_year": 2025}),

        ("compare 2024 vs 2025 arrivées APF",
         {"analysis_type": "comparative", "metric_type": "apf_arrivees"}),

        ("impact coupe du monde 2030 sur le tourisme",
         {"analysis_type": "causal", "period_year": 2030}),

        ("bonjour comment ça va",
         {"analysis_type": "factual", "metric_type": "general"}),
    ]

    extractor = IntentExtractor()
    failures = 0

    for msg, expected in test_cases:
        intent = extractor.extract(msg)
        queries = extractor.build_search_queries(intent)
        print(f"\nMESSAGE: {msg}")
        print(f"  metric_type={intent.metric_type}, analysis_type={intent.analysis_type}")
        print(f"  period={intent.period_month}/{intent.period_year}")
        print(f"  geo_scope={intent.geo_scope}")
        print(f"  entities={intent.detected_entities}")
        print(f"  factors={intent.external_factors_categories}")
        print(f"  queries={len(queries)} generated")
        for label, q in queries:
            print(f"    [{label}] {q}")
        for key, val in expected.items():
            actual = getattr(intent, key)
            if actual == val:
                print(f"  CHECK {key}={val}: PASS")
            else:
                print(f"  CHECK {key}={val}: FAIL (got {actual})")
                failures += 1

    print(f"\n{'=' * 60}")
    print(f"Total failures: {failures}")
    print('=' * 60)
