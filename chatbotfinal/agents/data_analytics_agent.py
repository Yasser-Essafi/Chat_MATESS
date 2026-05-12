"""
STATOUR Data Analytics Agent — Fixed
======================================
Catalogs Microsoft Fabric Gold tables and analyzes them through SQL-on-demand.
Connected to Web Search for context enrichment.
Generates interactive Plotly charts.

Fixes from original:
- Uses BaseAgent (eliminates duplicated code)
- Secure exec() with timeout, restricted builtins, and safe print
- Thread-safe stdout capture (no more sys.stdout swap)
- Fixed retry loop off-by-one (was range(1, MAX_RETRIES) = only 2 retries)
- Proper temperature (0.1, not 1.0) for code generation
- Proper max_completion_tokens usage for reasoning-model calls
- DataFrame copy in exec to prevent mutation of source data
- Structured logging throughout
- Integrated with shared cache and logger

Usage:
    from agents.data_analytics_agent import DataAnalyticsAgent

    agent = DataAnalyticsAgent()
    response = agent.chat("Top 5 pays de résidence en 2024")
    response = agent.chat("Montre l'évolution mensuelle")
"""

import os
import sys
import re
import glob
import signal
import contextlib
import unicodedata
from io import StringIO
from datetime import datetime
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    PLOTLY_AVAILABLE = False

# Premium chart engine
from utils.chart_engine import (
    ChartEngine, get_chart_namespace, CHART_TYPES, PREMIUM_COLORS,
    chart_bar, chart_line, chart_area, chart_pie, chart_donut,
    chart_treemap, chart_sunburst, chart_heatmap, chart_scatter,
    chart_combo, chart_choropleth, chart_save,
)

from config.settings import (
    SYSTEM_PROMPTS,
    ANALYTICS_TEMPERATURE,
    ANALYTICS_MAX_COMPLETION_TOKENS,
    ANALYTICS_REASONING_EFFORT,
    DATA_DIR,
    CHARTS_DIR,
    MAX_CODE_RETRIES,
    EXEC_TIMEOUT_SECONDS,
    DATA_SKIP_FILES,
    FABRIC_ENABLED,
    FABRIC_TABLES,
)
from utils.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger("statour.analytics")

MAX_RESPONSE_CHARTS = 4


# ══════════════════════════════════════════════════════════════════════════════
# Markdown Table Helper (injected into sandbox as `to_md`)
# ══════════════════════════════════════════════════════════════════════════════

def _df_to_markdown(df) -> str:
    """
    Convert a DataFrame to a clean markdown table string.
    Automatically formats numeric columns with thousands separators.
    Trailing newline ensures consecutive tables are visually separated.
    Called inside sandboxed code via: print(to_md(result))
    """
    import pandas as _pd
    df = df.copy()

    # Format numeric columns with commas
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].apply(lambda x: f"{int(x):,}" if _pd.notna(x) else "")

    # Build markdown table
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join(":---" for _ in df.columns) + " |"
    rows = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for _, row in df.iterrows()
    )
    # Trailing \n so that consecutive print(to_md(...)) calls have a blank line between them
    return f"{headers}\n{separator}\n{rows}\n"


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# History limit for analytics (system prompt can be large with schema)
ANALYTICS_MAX_HISTORY = 20

# French month names for date enrichment
MONTH_NAMES_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
}

# ══════════════════════════════════════════════════════════════════════════════
# Code Security
# ══════════════════════════════════════════════════════════════════════════════

# Blocked keywords in generated code — much stronger than original
BLOCKED_KEYWORDS = [
    # OS / filesystem
    "import os", "import sys", "import subprocess", "import shutil",
    "subprocess", "shutil", "rmdir", "unlink", "rmtree",
    "os.system", "os.popen", "os.exec", "os.remove", "os.unlink",
    "os.path", "os.environ",
    # Dangerous builtins
    "__import__", "eval(", "exec(", "compile(",
    "__builtins__", "__class__", "__subclasses__",
    "globals(", "locals(",
    "getattr(", "setattr(", "delattr(",
    # File I/O
    "open(", "write(", ".read(",
    "read_csv", "read_excel", "to_csv", "to_excel",
    "read_json", "read_sql", "read_html",
    # Interactive / debug
    "breakpoint(", "input(",
    # Network
    "import requests", "import urllib", "import http",
    "requests.get", "requests.post",
]

# Safe import whitelist — only data-science modules allowed in sandbox
_ALLOWED_IMPORTS = {
    "pandas", "numpy", "plotly", "plotly.express", "plotly.graph_objects",
    "plotly.graph_objs", "math", "statistics", "datetime", "collections",
    "itertools", "functools", "re", "json", "csv", "io", "copy",
}

def _safe_import(name, *args, **kwargs):
    """Restricted __import__ that only allows safe data-science modules."""
    top = name.split(".")[0]
    if top in _ALLOWED_IMPORTS or name in _ALLOWED_IMPORTS:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"Module '{name}' is not allowed in the analytics sandbox.")

# Restricted builtins whitelist for exec sandbox
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "enumerate": enumerate, "filter": filter,
    "float": float, "format": format, "frozenset": frozenset,
    "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "range": range, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "type": type,
    "zip": zip,
    "True": True, "False": False, "None": None,
    # Exception types — required for try/except blocks in generated code
    "Exception": Exception, "BaseException": BaseException,
    "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "AttributeError": AttributeError, "NameError": NameError,
    "ZeroDivisionError": ZeroDivisionError, "RuntimeError": RuntimeError,
    "StopIteration": StopIteration, "OverflowError": OverflowError,
    "__import__": _safe_import,  # allow safe imports (pandas, numpy, plotly...)
    # Note: print is injected separately as safe_print
}


# ══════════════════════════════════════════════════════════════════════════════
# Execution Timeout
# ══════════════════════════════════════════════════════════════════════════════

class ExecutionTimeout(Exception):
    """Raised when code execution exceeds the timeout."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise ExecutionTimeout(
        f"Code execution timed out ({EXEC_TIMEOUT_SECONDS}s)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Safe Code Execution
# ══════════════════════════════════════════════════════════════════════════════

def _check_code_safety(code: str) -> Tuple[bool, str]:
    """
    Check if generated code contains blocked keywords.

    Args:
        code: Python code string.

    Returns:
        (is_safe, reason) — True if safe, False with reason if blocked.
    """
    code_lower = code.lower()
    for kw in BLOCKED_KEYWORDS:
        if kw.lower() in code_lower:
            return False, f"Blocked keyword: {kw}"
    return True, ""


def _clean_code(code: str) -> str:
    """
    Remove import statements that the LLM might sneak in.
    All libraries are pre-loaded in the execution namespace.

    Args:
        code: Raw Python code.

    Returns:
        Cleaned code with imports removed.
    """
    lines = code.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            logger.debug("Removed import line: %s", stripped[:60])
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)


def execute_code_safe(
    code: str,
    exec_globals: Dict,
    timeout_seconds: int = EXEC_TIMEOUT_SECONDS,
) -> Dict:
    """
    Execute Python code in a sandboxed environment.

    Security measures:
        - Blocked keyword check
        - Restricted __builtins__ (no open, eval, exec, etc.)
        - Custom safe_print replaces print (thread-safe, no sys.stdout swap)
        - Execution timeout via SIGALRM (Unix) or no timeout (Windows)
        - DataFrame copies prevent mutation of source data

    Args:
        code: Python code to execute.
        exec_globals: Execution namespace (contains df, pd, np, px, go, etc.).
        timeout_seconds: Maximum execution time in seconds.

    Returns:
        Dict with keys:
            - output (str): Captured print output
            - chart_path (str|None): Path to generated chart, if any
            - error (str|None): Error message, if any
    """
    result = {"output": "", "chart_path": None, "chart_paths": [], "error": None}

    # ── Security check ──
    is_safe, reason = _check_code_safety(code)
    if not is_safe:
        result["error"] = f"Security: {reason}"
        logger.warning("Code blocked: %s", reason)
        return result

    # ── Thread-safe print capture ──
    # Instead of swapping sys.stdout (which is NOT thread-safe),
    # we inject a safe_print function into the exec namespace.
    output_lines = []

    def safe_print(*args, **kwargs):
        """Thread-safe print replacement for sandboxed execution."""
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(a) for a in args) + end
        output_lines.append(text)

    # ── Build restricted namespace ──
    exec_globals["print"] = safe_print
    exec_globals["__builtins__"] = SAFE_BUILTINS.copy()

    # ── Set timeout (Unix only) ──
    has_alarm = hasattr(signal, "SIGALRM")
    old_handler = None

    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    # ── Execute ──
    try:
        # Use one namespace for globals and locals. With separate namespaces,
        # Python treats exec like a class body: lambdas/comprehensions/functions
        # cannot reliably see variables assigned earlier in the generated code.
        # A single namespace matches normal script execution while preserving
        # restricted builtins and injected helpers.
        exec(code, exec_globals, exec_globals)
        result["output"] = "".join(output_lines)

    except ExecutionTimeout:
        result["error"] = f"Execution timed out ({timeout_seconds}s limit)"
        logger.warning("Code execution timed out")

    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
        logger.warning("Code syntax error: %s", e)

    except KeyError as e:
        result["error"] = f"KeyError: {e} — check column names"
        logger.warning("Code key error (likely wrong column): %s", e)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        logger.warning("Code execution error: %s: %s", type(e).__name__, e)

    finally:
        if has_alarm:
            signal.alarm(0)  # Cancel the alarm
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Data Analytics Agent
# ══════════════════════════════════════════════════════════════════════════════

class DataAnalyticsAgent(BaseAgent):
    """
    Data analytics agent for STATOUR.

    Features:
        - Catalogs Fabric Gold tables without bulk-loading their contents
        - Auto-detects schema and low-cardinality sample values
        - Generates T-SQL/Python code via LLM for data analysis
        - Executes code in secure sandbox with timeout
        - Creates interactive Plotly charts
        - Auto-retries with error context when code fails
        - Web search for external context when needed
        - Multi-dataset support with switching

    Inherits from BaseAgent:
        - Shared AzureOpenAI client
        - History trimming
        - Thread-safe LLM calls
        - Reset/length methods
    """

    def __init__(self):
        # ── Pre-init: Load datasets (needed for system prompt) ──
        self.datasets: Dict[str, Dict] = {}
        self.active_dataset_name: Optional[str] = None
        self.chart_count = 0
        self.last_chart_paths: List[str] = []
        self.kpi_cache = None   # Will be set after data loads
        self._apf_df = None     # APF DataFrame kept in memory for KPI/prediction
        self._db = None         # DBLayer reference for SQL-on-demand queries

        logger.info("Cataloging Fabric Gold metadata")
        self._auto_load_all_data()

        # ── Build KPI cache from active dataset ──
        self._rebuild_kpi_cache()

        # ── Initialize web search ──
        self._init_web_search()

        # ── Build system prompt with schema info ──
        system_prompt = self._build_system_prompt()

        # ── Initialize BaseAgent ──
        super().__init__(
            system_prompt=system_prompt,
            agent_name="Analyste de Données STATOUR",
            temperature=ANALYTICS_TEMPERATURE,              # 0.1, not 1.0!
            max_tokens=ANALYTICS_MAX_COMPLETION_TOKENS,     # 4096, not 1024!
            max_history_messages=ANALYTICS_MAX_HISTORY,
            reasoning_effort=ANALYTICS_REASONING_EFFORT,
        )

        logger.info(
            "%s ready — %d dataset(s), active: %s",
            self.agent_name,
            len(self.datasets),
            self.active_dataset_name or "None",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Temporal Constraint Injection
    # ──────────────────────────────────────────────────────────────────────

    # Month mapping for temporal constraint extraction.
    # Keys are normalized lowercase tokens matched against the user message.
    _MONTHS_CONSTRAINT = {
        "janvier": (1, "Janvier"), "january": (1, "Janvier"), "jan": (1, "Janvier"),
        "février": (2, "Février"), "fevrier": (2, "Février"),
        "february": (2, "Février"), "feb": (2, "Février"), "fev": (2, "Février"),
        "mars": (3, "Mars"), "march": (3, "Mars"),
        "avril": (4, "Avril"), "april": (4, "Avril"), "avr": (4, "Avril"),
        "mai": (5, "Mai"), "may": (5, "Mai"),
        "juin": (6, "Juin"), "june": (6, "Juin"),
        "juillet": (7, "Juillet"), "july": (7, "Juillet"),
        "août": (8, "Août"), "aout": (8, "Août"), "august": (8, "Août"),
        "septembre": (9, "Septembre"), "september": (9, "Septembre"),
        "octobre": (10, "Octobre"), "october": (10, "Octobre"),
        "novembre": (11, "Novembre"), "november": (11, "Novembre"),
        "décembre": (12, "Décembre"), "decembre": (12, "Décembre"),
        "december": (12, "Décembre"),
    }

    def _build_temporal_constraint(self, message: str) -> str:
        """Return a soft hint about the requested time period for the LLM.

        Now SQL-flavoured since the LLM generates T-SQL queries against
        Fabric instead of pandas filters.
        """
        msg_lower = message.lower()

        # Collect ALL years mentioned (e.g. "2024 vs 2025" → [2024, 2025])
        year_matches = re.findall(r'\b(20[12]\d)\b', message)
        years = list(dict.fromkeys(int(y) for y in year_matches))  # dedup, preserve order

        month_num = None
        month_name = None
        for name, (num, display) in self._MONTHS_CONSTRAINT.items():
            if re.search(r'\b' + re.escape(name) + r'\b', msg_lower):
                month_num = num
                month_name = display
                break

        if not years and not month_num:
            return ""

        # Multi-year comparison: emit IN(...) so neither year is filtered out
        if len(years) > 1:
            years_sql = ", ".join(str(y) for y in years)
            years_label = " vs ".join(str(y) for y in years)
            base_constraint = (
                f"\n\nPériode demandée : comparaison {years_label}.\n"
                f"Filtre SQL attendu : WHERE YEAR(date_stat) IN ({years_sql})"
            )
            # When 2026 is in an APF comparison, force month-alignment (Jan–Feb only
            # for APF — confirmed partial year). Hébergement may have more 2026 months,
            # so we only apply this constraint when the query is clearly about APF.
            PARTIAL_YEAR = 2026
            _APF_SIGNALS = ["apf", "frontière", "frontiere", "poste", "mre", "tes",
                            "voie", "continent", "diaspora"]
            is_apf_query = any(kw in msg_lower for kw in _APF_SIGNALS)
            if PARTIAL_YEAR in years and is_apf_query:
                last_month = self.kpi_cache.get("last_month", 2) if self.kpi_cache else 2
                months_list = ", ".join(str(m) for m in range(1, last_month + 1))
                _MO_NAMES = {
                    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
                    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
                    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
                }
                last_month_name = _MO_NAMES.get(last_month, str(last_month))
                base_constraint += (
                    f" AND MONTH(date_stat) IN ({months_list})\n\n"
                    f"⚠️ RÈGLE APF : la table APF 2026 ne contient que Janvier–{last_month_name}. "
                    f"Filtrer LES DEUX années sur MONTH(date_stat) IN ({months_list}). "
                    f"Indiquer dans la réponse : 'Comparaison Janvier–{last_month_name} uniquement "
                    f"(APF — données partielles 2026)'."
                )
            elif PARTIAL_YEAR in years:
                # Ambiguous or hébergement query — advise to verify coverage first
                base_constraint += (
                    f"\n\n⚠️ NOTE : pour 2026, vérifier la couverture réelle avec "
                    f"SELECT DISTINCT MONTH(date_stat) avant de comparer les années. "
                    f"L'APF s'arrête à Février 2026 ; l'hébergement peut avoir plus de mois."
                )
            return base_constraint

        year = years[0] if years else None

        if year and month_num:
            return (
                f"\n\nPériode demandée : {month_name} {year} uniquement "
                f"(pas de total annuel).\n"
                f"Filtre SQL attendu : "
                f"WHERE YEAR(date_stat) = {year} AND MONTH(date_stat) = {month_num}"
            )

        if year:
            return (
                f"\n\nPériode demandée : année {year}.\n"
                f"Filtre SQL attendu : WHERE YEAR(date_stat) = {year}"
            )

        latest_year = self.kpi_cache.get("last_year") if self.kpi_cache else 2026
        return (
            f"\n\nPériode demandée : {month_name} (année non précisée — "
            f"utiliser {latest_year}, la dernière disponible).\n"
            f"Filtre SQL attendu : "
            f"WHERE YEAR(date_stat) = {latest_year} AND MONTH(date_stat) = {month_num}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Web Search Initialization
    # ──────────────────────────────────────────────────────────────────────

    def _rebuild_kpi_cache(self) -> None:
        """Build the KPI cache from SQL aggregations against Fabric.

        Loads the APF fact table once (101 K rows, ~10 s) for the existing
        KPICache pandas logic. The same DataFrame is also kept on
        ``self._apf_df`` so the PredictionAgent and the orchestrator's
        year-range discovery can reuse it without a second query.
        """
        if not getattr(self, "_db", None) or self._db.source != "fabric":
            self.kpi_cache = None
            return

        if "fact_statistiques_apf" not in self.datasets:
            self.kpi_cache = None
            return

        try:
            from utils.kpi_cache import KPICache
            qualified = self.datasets["fact_statistiques_apf"]["qualified_name"]
            df = self._db.query_df(f"SELECT * FROM {qualified}")
            df = self._auto_enrich(df)
            self._apf_df = df  # exposed for PredictionAgent + year-range
            self.kpi_cache = KPICache(df)
            logger.info("KPI cache built from APF (%d rows via SQL)", len(df))
        except Exception as e:
            logger.warning("KPI cache build failed (non-critical): %s", e)
            self.kpi_cache = None
            self._apf_df = None

    def _init_web_search(self) -> None:
        """Initialize web search tool (optional — doesn't block agent startup)."""
        try:
            from tools.search_tools import TourismSearchTool
            self.searcher = TourismSearchTool()
            self.searcher_available = True
            logger.info("Web search: available")
        except Exception as e:
            self.searcher = None
            self.searcher_available = False
            logger.warning("Web search unavailable: %s", e)

    # ──────────────────────────────────────────────────────────────────────
    # Data Loading
    # ──────────────────────────────────────────────────────────────────────

    def _auto_load_all_data(self) -> None:
        """Catalog Fabric tables WITHOUT loading their contents.

        SQL-on-demand mode: tables stay in Fabric. We pull metadata only —
        column names + types via INFORMATION_SCHEMA, plus a small sample of
        distinct values for low-cardinality text columns. The LLM uses this
        catalog to write T-SQL queries; the sandbox runs them via `sql()`.

        Startup cost: ~5-15 seconds (was ~31 minutes for the bulk load).
        Memory cost: a few KB per table (was ~1 GB for the 7 M-row table).
        """
        if not FABRIC_ENABLED:
            logger.error(
                "Fabric not configured — set FABRIC_SQL_ENDPOINT, FABRIC_DATABASE, "
                "AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET in .env"
            )
            return

        try:
            from utils.db_layer import DBLayer
            self._db = DBLayer()
        except Exception as e:
            logger.error("Fabric: DBLayer init failed: %s", e)
            self._db = None
            return

        if self._db.source != "fabric":
            logger.error("Fabric connection failed — %s", self._db.status)
            return

        logger.info(
            "Fabric: cataloguing %d table(s) — schemas only, no bulk load",
            len(FABRIC_TABLES),
        )

        for table_name in FABRIC_TABLES:
            try:
                catalog = self._catalog_table(self._db, table_name)
                if catalog:
                    self.datasets[table_name] = catalog
                    logger.info(
                        "  ✅ Catalogued %s.%s: %d cols, ~%s rows",
                        self._db.schema, table_name,
                        catalog["col_count"],
                        f"{catalog['row_count']:,}" if catalog["row_count"] else "?",
                    )
            except Exception as e:
                logger.error("  ❌ Catalog %s: %s", table_name, str(e)[:200])

        if self.datasets and not self.active_dataset_name:
            facts = [n for n in self.datasets.keys() if n.startswith("fact_")]
            self.active_dataset_name = facts[0] if facts else list(self.datasets.keys())[0]

    def _catalog_table(self, db, table_name: str) -> Optional[Dict]:
        """Inspect a Fabric table: columns, types, row count, sample values
        for low-cardinality text columns. Returns a metadata dict (no df)."""
        cols = db.query_df(
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE "
            "FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
            "ORDER BY ORDINAL_POSITION",
            (db.schema, table_name),
        )
        if cols.empty:
            return None

        qualified = f"[{db.schema}].[{table_name}]"
        try:
            n = int(db.query_df(f"SELECT COUNT(*) AS n FROM {qualified}").iloc[0, 0])
        except Exception:
            n = 0

        # Sample 30 distinct values for each text column with cardinality <= 1000.
        # Numeric and date columns get min/max instead.
        sample_values: Dict[str, List] = {}
        column_summary: List[Dict] = []
        for _, row in cols.iterrows():
            col_name = row["COLUMN_NAME"]
            dtype = row["DATA_TYPE"]
            entry = {"name": col_name, "type": dtype}
            if dtype in ("varchar", "nvarchar", "char", "nchar"):
                try:
                    distinct_n = int(db.query_df(
                        f"SELECT COUNT(DISTINCT [{col_name}]) AS n FROM {qualified}"
                    ).iloc[0, 0])
                    entry["unique"] = distinct_n
                    if 0 < distinct_n <= 1000:
                        vals_df = db.query_df(
                            f"SELECT DISTINCT TOP 30 [{col_name}] AS v FROM {qualified} "
                            f"WHERE [{col_name}] IS NOT NULL ORDER BY [{col_name}]"
                        )
                        sample_values[col_name] = vals_df["v"].dropna().tolist()
                except Exception:
                    pass
            elif dtype in ("int", "bigint", "smallint", "tinyint", "decimal",
                           "numeric", "float", "real", "money"):
                try:
                    mm = db.query_df(
                        f"SELECT MIN([{col_name}]) AS mn, MAX([{col_name}]) AS mx, "
                        f"SUM(CAST([{col_name}] AS BIGINT)) AS sm FROM {qualified}"
                    )
                    entry["min"] = mm.iloc[0, 0]
                    entry["max"] = mm.iloc[0, 1]
                    entry["sum"] = mm.iloc[0, 2]
                except Exception:
                    pass
            elif dtype in ("date", "datetime", "datetime2", "smalldatetime"):
                try:
                    mm = db.query_df(
                        f"SELECT MIN([{col_name}]) AS mn, MAX([{col_name}]) AS mx FROM {qualified}"
                    )
                    entry["min"] = str(mm.iloc[0, 0])
                    entry["max"] = str(mm.iloc[0, 1])
                except Exception:
                    pass
            column_summary.append(entry)

        schema_text = self._format_catalog_text(table_name, qualified, column_summary, n)

        return {
            "qualified_name": qualified,
            "schema": schema_text,
            "columns": column_summary,
            "sample_values": sample_values,
            "col_count": len(column_summary),
            "row_count": n,
            "source": "fabric",
        }

    @staticmethod
    def _format_catalog_text(table_name: str, qualified: str,
                              cols: List[Dict], row_count: int) -> str:
        """Render the per-table catalog as a string for the LLM prompt."""
        lines = [
            f"TABLE: {qualified}  ({row_count:,} rows)",
            "COLUMNS:",
        ]
        for c in cols:
            extras = []
            if "unique" in c:
                extras.append(f"unique={c['unique']:,}")
            if "min" in c and "max" in c:
                extras.append(f"min={c['min']}")
                extras.append(f"max={c['max']}")
            if "sum" in c and c["sum"] is not None:
                extras.append(f"sum={c['sum']:,}")
            tail = " | " + " ".join(extras) if extras else ""
            lines.append(f"  {c['name']:35s} {c['type']:12s}{tail}")
        return "\n".join(lines)

    def _auto_enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-enrich DataFrame:
        - Parse date columns and create year/month/quarter columns
        - Fill numeric NaN with 0
        - Compute APF "total" = mre + tes when both exist
        - Compute hebergement "total" = arrivees when present (so analytics
          questions phrased around "total" / "arrivées" work uniformly)
        - Alias nationalite_name → nationalite for cross-table consistency
        """
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["date", "dat", "periode", "period"]):
                try:
                    df[col] = pd.to_datetime(df[col])
                    c = col.replace(" ", "_").lower()
                    df[f"{c}_year"] = df[col].dt.year
                    df[f"{c}_month"] = df[col].dt.month
                    df[f"{c}_month_name"] = df[col].dt.month.map(MONTH_NAMES_FR)
                    df[f"{c}_quarter"] = df[col].dt.quarter
                    df[f"{c}_year_month"] = df[col].dt.strftime("%Y-%m")
                    logger.debug("Enriched date column: %s", col)
                except Exception:
                    pass

        # Fill numeric NaN
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(0)

        col_lower_map = {c.lower(): c for c in df.columns}

        # APF fact table: total = mre + tes
        if "mre" in col_lower_map and "tes" in col_lower_map and "total" not in col_lower_map:
            mre_col = col_lower_map["mre"]
            tes_col = col_lower_map["tes"]
            df["total"] = df[mre_col] + df[tes_col]
            logger.debug("APF: created 'total' = %s + %s", mre_col, tes_col)

        # Hebergement fact: expose arrivees as 'total' so common analytics
        # questions work uniformly across the two fact tables.
        elif "arrivees" in col_lower_map and "total" not in col_lower_map:
            df["total"] = df[col_lower_map["arrivees"]]
            logger.debug("Hebergement: aliased arrivees → total")

        # Cross-table column alias: 'nationalite_name' (hebergement) ⇆
        # 'nationalite' (apf) — let the LLM use either name.
        if "nationalite_name" in col_lower_map and "nationalite" not in col_lower_map:
            df["nationalite"] = df[col_lower_map["nationalite_name"]]
        if "region_name" in col_lower_map and "region" not in col_lower_map:
            df["region"] = df[col_lower_map["region_name"]]

        return df

    # ──────────────────────────────────────────────────────────────────────
    # Schema Detection
    # ──────────────────────────────────────────────────────────────────────

    def _auto_detect_schema(self, df: pd.DataFrame, name: str) -> str:
        """Generate a detailed schema description for the LLM."""
        lines = [
            f"DATASET: {name}",
            f"ROWS: {len(df):,}",
            f"COLUMNS: {len(df.columns)}",
            "",
        ]

        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()

            if pd.api.types.is_numeric_dtype(df[col]):
                lines.append(
                    f"  {col:30s} : {dtype:10s} | "
                    f"unique={nunique:,} | "
                    f"min={df[col].min():,} | "
                    f"max={df[col].max():,} | "
                    f"sum={df[col].sum():,}"
                )
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                lines.append(
                    f"  {col:30s} : {dtype:10s} | "
                    f"{df[col].min()} to {df[col].max()}"
                )
            else:
                samples = df[col].dropna().unique()[:8]
                s = ", ".join([str(v) for v in samples])
                if len(s) > 80:
                    s = s[:80] + "..."
                lines.append(
                    f"  {col:30s} : {dtype:10s} | "
                    f"unique={nunique:,} | ex: {s}"
                )

        return "\n".join(lines)

    def _get_value_samples_from_catalog(self, table_name: str, info: Dict) -> str:
        """Render the cached sample values for one table (catalog-only mode)."""
        sv = info.get("sample_values") or {}
        if not sv:
            return ""
        lines = [f"\nSAMPLE VALUES — {table_name} (use these EXACT values in WHERE clauses):"]
        for col, vals in sv.items():
            preview = ", ".join(f'"{v}"' for v in vals[:30])
            lines.append(f"  {col}: [{preview}]")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # System Prompt Builder
    # ──────────────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Build the system prompt — SQL-on-demand mode.

        The LLM is instructed to compose T-SQL queries against Fabric and
        execute them via the sandbox `sql()` helper. No DataFrames are
        pre-loaded into memory.
        """
        schemas = []
        samples = []
        for name, info in self.datasets.items():
            schemas.append(info["schema"])
            sv_text = self._get_value_samples_from_catalog(name, info)
            if sv_text:
                samples.append(sv_text)

        schemas_text = "\n\n".join(schemas) if schemas else "No tables catalogued."
        values_text = "\n".join(samples) if samples else ""
        ds_list = ", ".join(self.datasets.keys()) if self.datasets else "None"
        active = self.active_dataset_name or "None"

        base_intro = (
            "Tu génères du T-SQL exécuté sur Microsoft Fabric Lakehouse Gold via la fonction sql().\n"
            "Réponds UNIQUEMENT avec du code Python dans un bloc ```python```.\n\n"
            "━━ RÈGLE D'OR : AMBIGUÏTÉ \"ARRIVÉES\" ━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "\"arrivées\" signifie 2 choses différentes :\n"
            "- APF (postes frontières) → fact_statistiques_apf (mre + tes)\n"
            "- Hébergement (hotels) → fact_statistiqueshebergementnationaliteestimees (arrivees)\n"
            "Si ambigu → fournir LES DEUX métriques dans la réponse.\n\n"
            "━━ ROUTAGE OBLIGATOIRE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "HÉBERGEMENT → [dbo_GOLD].[fact_statistiqueshebergementnationaliteestimees]\n"
            "  Mots-clés : nuitées, hébergement, hôtel, EHTC, STDN, TO, taux occupation,\n"
            "               établissement, délégation, maison d'hôtes, capacité, chambre\n"
            "  Métriques : nuitees, arrivees\n\n"
            "FRONTIÈRES → [dbo_GOLD].[fact_statistiques_apf]\n"
            "  Mots-clés : postes frontières, APF, MRE, TES, voie, frontière, pays de résidence, continent\n"
            "  Métriques : mre, tes (total = mre + tes)\n\n"
            "JAMAIS croiser les métriques entre les deux tables.\n\n"
            "━━ CONTEXTE MÉTIER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "- \"nationalite\" en SQL = pays de résidence du voyageur (PAS nationalité ethnique)\n"
            "- Renommer dans l'output : nationalite → \"Pays de résidence\"\n"
            "- MRE = Marocains diaspora | TES = Touristes étrangers\n"
            "- date_stat = toujours 1er du mois (mensuel, pas quotidien)\n"
            "- COUVERTURE 2026 PAR TABLE :\n"
            "  • fact_statistiques_apf (APF) : UNIQUEMENT Janvier–Février 2026 confirmés.\n"
            "    → Toute comparaison APF 2026 vs N-1 DOIT restreindre les deux années à Jan–Fév.\n"
            "    → Signaler dans la réponse : 'Comparaison Jan–Fév uniquement (APF, données partielles 2026)'.\n"
            "  • fact_statistiqueshebergementnationaliteestimees (hébergement) : mois 2026 potentiellement\n"
            "    supérieurs à Février. Toujours vérifier avec SELECT DISTINCT MONTH(date_stat) avant de conclure.\n"
            "    → Ne jamais supposer que l'hébergement s'arrête à Février 2026.\n\n"
            f"━━ TABLES DISPONIBLES (Fabric SQL Analytics Endpoint) ━━━━━━━━\n"
            f"Tables : {ds_list}\n"
            f"TABLE PAR DÉFAUT : [dbo_GOLD].[{active}]\n\n"
        )

        sql_rules = (
            "━━ RÈGLES SQL/SANDBOX ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "1. Utiliser `sql(query)` — JAMAIS `pd.read_sql`, `read_excel`, `read_csv` (bloqués).\n"
            "2. Préfixer toutes les tables avec [dbo_GOLD].\n"
            "3. AGRÉGER côté SQL (GROUP BY, SUM, COUNT, TOP N) — jamais ramener millions de lignes.\n"
            "4. Filtres temporels : `WHERE YEAR(date_stat) = 2025 AND MONTH(date_stat) = 7`.\n"
            "5. Pré-loaded : pd, np, px, go, to_md(df), MONTH_NAMES_FR, chart_* functions, save_chart.\n"
            "6. Tableaux : `print(to_md(result))`.\n"
            "7. ⚠️ GRAPHIQUES OBLIGATOIRES : Si l'utilisateur mentionne 'graphique', 'graph', 'chart',\n"
            "   'visualisation', 'courbe', 'évolution', 'comparaison', 'compare', ou demande de\n"
            "   'créer'/'crée'/'génère'/'montre'/'affiche' un graphique → tu DOIS générer le(s)\n"
            "   graphique(s) avec les fonctions chart_* premium ci-dessous, PAS juste les recommander.\n"
            "   JAMAIS écrire 'Graphes recommandés' ou 'pour créer le graphique vous pouvez...'.\n"
            "   GÉNÈRE TOUJOURS les graphiques demandés en code Python EXÉCUTABLE.\n"
            "   Maximum 4 graphiques; au-delà, résumer en tableau.\n"
            "   ⚠️ TOUJOURS appeler `save_chart(fig, 'titre_court')` pour persister chaque graphique\n"
            "   et l'afficher dans la conversation. Sans save_chart, le graphique n'est PAS affiché.\n"
            "8. NOMS DE MOIS : JAMAIS écrire un CASE SQL pour les noms de mois.\n"
            "   Utiliser Python post-traitement : `df['mois_fr'] = df['mois'].map(MONTH_NAMES_FR)`\n"
            "   MONTH_NAMES_FR = {1:'Janvier', 2:'Février', ..., 12:'Décembre'} (clés entières 1-12).\n\n"
        )

        chart_docs = (
            "━━ FONCTIONS GRAPHIQUES PREMIUM (ChartEngine) ━━━━━━━━━━━━━━━━━━\n"
            "Toutes ces fonctions retournent un objet Figure Plotly avec style premium MTAESS.\n"
            "⚠️ TOUJOURS appeler `save_chart(fig, 'titre_court')` (PAS chart_save!) pour\n"
            "que le graphique soit affiché dans la conversation. save_chart enregistre ET\n"
            "publie le graphique. Maximum 4 graphiques par réponse.\n\n"
            "TYPES DE GRAPHIQUES DISPONIBLES (chacun retourne une Figure):\n"
            "  chart_bar(df, x, y, color=None, title='', subtitle='', horizontal=False, height=600)\n"
            "    → Barres verticales/horizontales. Ex: top pays, comparaisons par catégorie.\n\n"
            "  chart_line(df, x, y, color=None, title='', subtitle='', height=600, markers=True)\n"
            "    → Courbes temporelles. Ex: évolution mensuelle/annuelle.\n"
            "    → Pour séries multiples (plusieurs villes/pays), utiliser color='ville'.\n\n"
            "  chart_area(df, x, y, color=None, title='', subtitle='', height=600)\n"
            "    → Aire empilée pour séries temporelles.\n\n"
            "  chart_pie(df, names, values, title='', subtitle='', height=550)\n"
            "    → Camembert pour répartitions.\n\n"
            "  chart_donut(df, names, values, title='', subtitle='', height=550)\n"
            "    → Donut avec total au centre.\n\n"
            "  chart_treemap(df, path, values, title='', subtitle='', height=600)\n"
            "    → Treemap hiérarchique. path = ['continent', 'pays']\n\n"
            "  chart_sunburst(df, path, values, title='', subtitle='', height=600)\n"
            "    → Sunburst radial hiérarchique.\n\n"
            "  chart_heatmap(df, x, y, z, title='', subtitle='', height=600)\n"
            "    → Heatmap. Ex: mois × année × arrivées.\n\n"
            "  chart_scatter(df, x, y, color=None, title='', subtitle='', height=600, trendline=False)\n"
            "    → Nuage de points. trendline=True ajoute régression.\n\n"
            "  chart_combo(df, x, bar_y, line_y, title='', subtitle='', height=600)\n"
            "    → Barres + ligne sur 2 axes Y. Ex: arrivées (bars) vs croissance (ligne).\n\n"
            "  chart_choropleth(df, locations, z, title='', subtitle='', height=650)\n"
            "    → Carte choroplèthe des régions du Maroc. locations = colonne avec noms de régions.\n\n"
            "  chart_waterfall(df, x, y, title='', subtitle='', height=550)\n"
            "    → Cascade pour variations cumulées.\n\n"
            "  chart_funnel(df, x, y, title='', subtitle='', height=550)\n"
            "    → Entonnoir pour conversions.\n\n"
            "EXEMPLE 1 — UN graphique (top pays):\n"
            "```python\n"
            "df = sql(\"SELECT TOP 10 nationalite, SUM(mre+tes) AS arrivees FROM [dbo_GOLD].[fact_statistiques_apf] WHERE YEAR(date_stat)=2024 GROUP BY nationalite ORDER BY arrivees DESC\")\n"
            "fig = chart_bar(df, x='nationalite', y='arrivees', title='Top 10 pays de résidence APF', subtitle='Année 2024')\n"
            "save_chart(fig, 'top_pays_2024')  # ← affiche dans la conversation\n"
            "```\n\n"
            "EXEMPLE 2 — DEUX graphiques (évolution + comparaison):\n"
            "```python\n"
            "# Graphique 1 : Évolution mensuelle Casablanca\n"
            "evo = sql(\"SELECT YEAR(date_stat) AS annee, MONTH(date_stat) AS mois, SUM(nuitees) AS nuitees FROM [dbo_GOLD].[fact_statistiqueshebergementnationaliteestimees] WHERE ville_name='Casablanca' AND YEAR(date_stat) BETWEEN 2024 AND 2026 GROUP BY YEAR(date_stat), MONTH(date_stat) ORDER BY annee, mois\")\n"
            "evo['periode'] = evo['annee'].astype(str) + '-' + evo['mois'].astype(str).str.zfill(2)\n"
            "fig1 = chart_line(evo, x='periode', y='nuitees', title='Évolution mensuelle des nuitées Casablanca', subtitle='2024-2026')\n"
            "save_chart(fig1, 'evolution_casablanca')\n\n"
            "# Graphique 2 : Comparaison entre villes\n"
            "comp = sql(\"SELECT ville_name AS ville, YEAR(date_stat) AS annee, SUM(nuitees) AS nuitees FROM [dbo_GOLD].[fact_statistiqueshebergementnationaliteestimees] WHERE ville_name IN ('Casablanca','Marrakech','Agadir','Rabat') AND YEAR(date_stat) BETWEEN 2024 AND 2026 GROUP BY ville_name, YEAR(date_stat) ORDER BY annee, ville\")\n"
            "fig2 = chart_bar(comp, x='annee', y='nuitees', color='ville', title='Comparaison nuitées par ville', subtitle='Casablanca vs Marrakech vs Agadir vs Rabat')\n"
            "save_chart(fig2, 'comparaison_villes')\n"
            "```\n\n"
            "PALETTES DISPONIBLES: PREMIUM_COLORS['primary'], ['morocco_gradient'], ['ocean_gradient'], ['sunset_gradient']\n\n"
            "━━ FORMAT DE RÉPONSE NARRATIF ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Ta réponse doit être un RÉCIT ANALYTIQUE structuré, PAS un dump de données brutes.\n"
            "Structure attendue :\n"
            "1. Paragraphe d'introduction et contexte\n"
            "2. [GRAPHIQUE_1] ← marqueur où le 1er graphique sera inséré\n"
            "3. Paragraphe commentant le graphique (tendances, chiffres clés)\n"
            "4. [GRAPHIQUE_2] ← marqueur pour le 2e graphique si applicable\n"
            "5. Paragraphe comparatif / facteurs explicatifs\n"
            "6. Conclusion / recommandations\n\n"
            "RÈGLES :\n"
            "- Place [GRAPHIQUE_N] (N=1,2,3,4) dans ton texte là où chaque graphique doit apparaître\n"
            "- N correspond à l'ordre d'appel de save_chart() dans le code\n"
            "- JAMAIS afficher de noms de tables SQL, chemins de fichiers, ou noms de colonnes internes\n"
            "- JAMAIS inclure de lignes 'NOTE:' techniques\n"
            "- Utilise 'données d'hébergement' ou 'données aux postes frontières' au lieu des noms de tables\n"
            "- Les tableaux de données (to_md) sont OK mais doivent être commentés dans le récit\n\n"
        )

        examples = (
            "━━ EXEMPLES T-SQL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "# Total APF d'une année\n"
            "SELECT SUM(mre)+SUM(tes) AS total_arrivees\n"
            "  FROM [dbo_GOLD].[fact_statistiques_apf]\n"
            "  WHERE YEAR(date_stat) = 2025;\n\n"
            "# Top 5 pays de résidence en 2024 (APF)\n"
            "SELECT TOP 5 nationalite AS [Pays de résidence],\n"
            "             SUM(mre+tes) AS [Arrivées]\n"
            "  FROM [dbo_GOLD].[fact_statistiques_apf]\n"
            "  WHERE YEAR(date_stat) = 2024\n"
            "  GROUP BY nationalite ORDER BY [Arrivées] DESC;\n\n"
            "# Nuitées par type d'hébergement en 2024 (jointure dim)\n"
            "SELECT dc.type_eht_libelle AS [Type], SUM(f.nuitees) AS [Nuitées]\n"
            "  FROM [dbo_GOLD].[fact_statistiqueshebergementnationaliteestimees] f\n"
            "  JOIN [dbo_GOLD].[gld_dim_categories_classements] dc\n"
            "    ON dc.categorie_name = f.categorie_name\n"
            "  WHERE YEAR(f.date_stat) = 2024\n"
            "  GROUP BY dc.type_eht_libelle ORDER BY [Nuitées] DESC;\n\n"
            "# Mois disponibles pour 2026 (TOUJOURS utiliser pour vérifier la couverture)\n"
            "# APF :\n"
            "apf_months = sql(\"SELECT DISTINCT YEAR(date_stat) AS annee, MONTH(date_stat) AS mois FROM [dbo_GOLD].[fact_statistiques_apf] WHERE YEAR(date_stat) = 2026 ORDER BY mois\")\n"
            "apf_months['Mois (FR)'] = apf_months['mois'].map(MONTH_NAMES_FR)  # ← utiliser MONTH_NAMES_FR, JAMAIS CASE SQL\n"
            "print(to_md(apf_months))\n"
            "# Hébergement :\n"
            "eht_months = sql(\"SELECT DISTINCT YEAR(date_stat) AS annee, MONTH(date_stat) AS mois FROM [dbo_GOLD].[fact_statistiqueshebergementnationaliteestimees] WHERE YEAR(date_stat) = 2026 ORDER BY mois\")\n"
            "eht_months['Mois (FR)'] = eht_months['mois'].map(MONTH_NAMES_FR)\n"
            "print(to_md(eht_months))\n"
        )

        return (
            base_intro
            + sql_rules
            + chart_docs
            + f"━━ SCHÉMAS DÉTAILLÉS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{schemas_text}\n\n"
            + (f"{values_text}\n\n" if values_text else "")
            + examples
        )

    def _refresh_system_prompt(self) -> None:
        """Rebuild and update system prompt (e.g., after dataset switch)."""
        new_prompt = self._build_system_prompt()
        self.update_system_prompt(new_prompt)

    # ──────────────────────────────────────────────────────────────────────
    # Dataset Management
    # ──────────────────────────────────────────────────────────────────────

    def load_dataset(self, filepath: str) -> str:
        """File-based dataset loading is disabled in SQL-on-demand mode."""
        return ("ℹ️  File-based loading is disabled. The chatbot now runs "
                "SQL on Fabric Lakehouse Gold. Add tables to FABRIC_TABLES "
                "in .env to expose them.")

    def switch_dataset(self, name: str) -> str:
        """Mark `name` as the default table for the LLM (used in the prompt)."""
        if name in self.datasets:
            self.active_dataset_name = name
            self._refresh_system_prompt()
            row_count = self.datasets[name].get("row_count", "?")
            logger.info("Active table set to: %s", name)
            return f"📌 Active: {name} ({row_count:,} rows)"

        available = ", ".join(self.datasets.keys())
        return f"❌ Not found: '{name}'. Available: {available}"

    # ──────────────────────────────────────────────────────────────────────
    # Code Extraction
    # ──────────────────────────────────────────────────────────────────────

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            return None

        code = matches[0].strip()
        code = _clean_code(code)

        if not code.strip():
            return None

        return code

    def _extract_text(self, response: str) -> str:
        """Extract text part only (remove code blocks)."""
        text = re.sub(
            r"```python\s*\n.*?```", "", response, flags=re.DOTALL
        ).strip()
        return text

    # ──────────────────────────────────────────────────────────────────────
    # Code Execution
    # ──────────────────────────────────────────────────────────────────────

    def _cleanup_old_charts(self, keep_last: int = 20) -> None:
        """Delete all but the `keep_last` most recent chart files.

        Why: the charts/ folder accumulates an HTML file per analytics run
        and grows unbounded on a long-lived server. Keeps the 20 most recent
        so the last few conversation turns can still link to their chart.
        """
        try:
            chart_files = sorted(
                glob.glob(os.path.join(CHARTS_DIR, "chart_*.html")),
                key=os.path.getmtime,
            )
            to_delete = chart_files[:-keep_last] if len(chart_files) > keep_last else []
            for path in to_delete:
                try:
                    os.remove(path)
                except OSError:
                    pass
        except Exception as e:
            logger.debug("Chart cleanup skipped (non-critical): %s", e)

    def _execute_analysis(self, code: str) -> Dict:
        """
        Execute analysis code in secure sandbox.

        Creates a unique chart path, builds execution namespace
        with DataFrame copies, and runs via execute_code_safe().

        Args:
            code: Python code to execute.

        Returns:
            Dict with output, chart_path, error.
        """
        if not self.datasets:
            return {
                "output": "",
                "chart_path": None,
                "chart_paths": [],
                "error": "No tables catalogued — Fabric connection may be down.",
            }

        if not getattr(self, "_db", None) or self._db.source != "fabric":
            return {
                "output": "",
                "chart_path": None,
                "chart_paths": [],
                "error": "Fabric connection unavailable — cannot run SQL.",
            }

        # ── Prune old chart files before creating a new one ──
        self._cleanup_old_charts()

        # ── Generate unique chart path ──
        self.chart_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_run_id = f"{timestamp}_{self.chart_count}"
        chart_filename = f"chart_{chart_run_id}_1.html"
        chart_path = os.path.join(CHARTS_DIR, chart_filename).replace("\\", "/")
        chart_paths: List[str] = []

        def _register_chart(path: str) -> str:
            if path and path not in chart_paths:
                chart_paths.append(path)
            return path

        # Replace placeholder in code
        code = code.replace("CHART_PATH", chart_path)

        # ── Build execution namespace ──
        # SQL-on-demand: the LLM gets a `sql(query)` function instead of huge
        # in-memory DataFrames. Returned DataFrames are typically <10k rows.
        db_ref = self._db  # captured for the closure below

        def _sandbox_sql(query: str) -> "pd.DataFrame":
            """Execute a read-only T-SQL query against Fabric. Capped at
            DBLayer.SAFE_QUERY_ROW_LIMIT rows."""
            return db_ref.safe_query(query)

        def _save_chart(fig, label: Optional[str] = None) -> Optional[str]:
            """Persist a Plotly figure and register it as a response artifact."""
            if len(chart_paths) >= MAX_RESPONSE_CHARTS:
                print(
                    f"Limite graphiques atteinte ({MAX_RESPONSE_CHARTS}). "
                    "Graphique supplementaire ignore; presenter les donnees en tableau."
                )
                return None
            index = len(chart_paths) + 1
            safe_label = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                str(label or f"chart_{index}"),
            ).strip("_")[:36]
            suffix = f"_{safe_label}" if safe_label else ""
            path = os.path.join(
                CHARTS_DIR,
                f"chart_{chart_run_id}_{index}{suffix}.html",
            ).replace("\\", "/")
            fig.write_html(
                path,
                include_plotlyjs="cdn",
                full_html=True,
                config={"responsive": True, "displaylogo": False},
            )
            return _register_chart(path)

        exec_globals = {
            "sql": _sandbox_sql,
            "pd": pd,
            "np": np,
            "MONTH_NAMES_FR": MONTH_NAMES_FR,
            "months_fr": MONTH_NAMES_FR,
            "save_chart": _save_chart,
            # Helper: renders a DataFrame as a markdown table string
            "to_md": _df_to_markdown,
        }

        # Always inject plotly directly — don't rely on module-level PLOTLY_AVAILABLE flag
        # which can be False due to import timing issues or environment quirks.
        try:
            import plotly.express as _px
            import plotly.graph_objects as _go
            exec_globals["px"] = _px
            exec_globals["go"] = _go
        except ImportError:
            logger.warning("Plotly not available; chart generation will fail")

        # ── Inject premium chart engine namespace ──
        # Provides chart_bar, chart_line, chart_pie, chart_save, etc.
        exec_globals.update(get_chart_namespace())

        # ── Prepend critical helpers to the code so they're always defined ──
        # Prevents NameError on months_fr even if the model forgets to define it
        code_preamble = (
            "months_fr = MONTH_NAMES_FR\n"
            "MONTHS = MONTH_NAMES_FR\n"
        )
        code = code_preamble + code

        # ── Execute ──
        result = execute_code_safe(code, exec_globals, EXEC_TIMEOUT_SECONDS)

        # ── Check for chart file ──
        if os.path.exists(chart_path):
            _register_chart(chart_path)
        result["chart_paths"] = chart_paths[:MAX_RESPONSE_CHARTS]
        if result["chart_paths"]:
            result["chart_path"] = result["chart_paths"][0]
            logger.debug("Chart generated: %s", chart_filename)
            # Chart was generated successfully — clear the error (it may be from
            # post-chart code like a summary print that doesn't affect the output)
            if result.get("error"):
                logger.info(
                    "Chart generated despite error '%s' — treating as success",
                    result["error"][:80],
                )
                result["error"] = None

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Web Context (optional enrichment)
    # ──────────────────────────────────────────────────────────────────────

    def _needs_web_search(self, message: str) -> bool:
        """Analytics agent uses only the local DataFrame — never web search."""
        return False

    def _get_web_context(self, query: str) -> str:
        """Get web context if searcher is available."""
        if not self.searcher_available or not self.searcher:
            return ""
        try:
            logger.debug("Fetching web context for: %s", query[:50])
            results = self.searcher.search(query, max_results=3)
            if results:
                parts = [
                    f"[{r['title']}] {r['content'][:150]}"
                    for r in results
                ]
                return "WEB CONTEXT:\n" + "\n".join(parts)
        except Exception as e:
            logger.warning("Web context fetch failed: %s", e)
        return ""

    # ──────────────────────────────────────────────────────────────────────
    # Error Fix Message Builder
    # ──────────────────────────────────────────────────────────────────────

    def _build_fix_message(self, error: str, failed_code: str = "", original_question: str = "") -> str:
        """Build a detailed error fix message with catalog schema context (SQL-on-demand)."""
        if not self.active_dataset_name or self.active_dataset_name not in self.datasets:
            return f"Error: {error}\nFix the code."

        dataset_info = self.datasets[self.active_dataset_name]

        fix_msg = f"The generated analytics code failed.\n\nOriginal user question:\n{original_question}\n\nError:\n{error}\n\n"
        if failed_code:
            fix_msg += (
                "Failed code that must be repaired completely:\n"
                "```python\n"
                f"{failed_code[:6000]}\n"
                "```\n\n"
            )

        # ── NameError: spell out the undefined variable explicitly ──
        import re as _re
        name_match = _re.search(r"NameError: name '(\w+)' is not defined", error)
        if name_match:
            undef_var = name_match.group(1)
            fix_msg += (
                f"⚠️  '{undef_var}' does not exist in the execution environment.\n"
                f"    Each run starts FRESH — use sql('SELECT ...') to get data.\n\n"
            )

        # ── Schema from catalog (no df available — SQL-on-demand mode) ──
        cols = dataset_info.get("columns", [])
        col_names = [c["name"] for c in cols]
        fix_msg += f"Table: {dataset_info.get('qualified_name', self.active_dataset_name)}\n"
        fix_msg += f"Available columns: {col_names}\n\n"

        # Sample values for text columns
        sv = dataset_info.get("sample_values", {})
        if sv:
            fix_msg += "Sample values for text columns (use in WHERE clauses):\n"
            for col, vals in sv.items():
                fix_msg += f"  {col}: {vals[:10]}\n"

        fix_msg += (
            "\nReturn the COMPLETE corrected code as a single ```python block.\n"
            "CRITICAL: Each execution starts FRESH — no intermediate variables survive between attempts.\n"
            "You MUST redefine ALL variables from scratch in the new block.\n"
            "Do NOT write partial fixes — write the full code from start to finish.\n"
            "\nRules (SQL-on-demand mode):\n"
            "- Use sql('SELECT ...') to query Fabric — NEVER use df, pd.read_sql, read_excel.\n"
            "- Prefix all tables with [dbo_GOLD]: sql('SELECT ... FROM [dbo_GOLD].[table_name] ...').\n"
            "- Always AGGREGATE in SQL (GROUP BY, SUM, COUNT, TOP N) — never load millions of rows.\n"
            "- Pre-loaded: sql, pd, np, px (plotly.express), go (plotly.graph_objects), to_md, MONTH_NAMES_FR.\n"
            "- Do NOT use import statements — all libraries are already available.\n"
            "- Print DataFrames with: print(to_md(result)) — never use to_string().\n"
            "- Save charts with: save_chart(fig, 'titre court') (max 4). For one legacy chart, fig.write_html('CHART_PATH') is also accepted.\n"
            "- Always call print() on results.\n"
        )

        return fix_msg

    # ──────────────────────────────────────────────────────────────────────
    # Format Success Response
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_response_text(text: str, is_exec_output: bool = False) -> str:
        """
        Clean LLM text or exec output before showing to user.
        Removes script/code references and fixes display labels.
        Applied to BOTH text_part and exec_output.
        """
        import re

        if not text:
            return text

        # ── A. Fix column label: any variant of Nationalité → Pays de résidence ──
        # Applied to both text and exec output (tables have these headers)
        text = re.sub(r"Nationalit[eéè]s?", "Pays de résidence", text, flags=re.IGNORECASE)
        text = text.replace("nationalité", "pays de résidence")
        text = text.replace("nationalités", "pays de résidence")
        text = text.replace("Nationalité", "Pays de résidence")
        text = text.replace("Nationalités", "Pays de résidence")

        if is_exec_output:
            # exec output — fix column labels + replace NaN values
            text = re.sub(r'\+?nan%?', 'N/A', text, flags=re.IGNORECASE)
            text = re.sub(r'\bNaN\b', 'N/A', text)
            return text.strip()

        # ── B. Remove sentences/lines containing script/code references ──
        # Strategy: split into sentences, filter out any that mention the internal mechanism.
        # IMPORTANT: keep keywords specific — avoid broad words that cause false positives.
        # "ci-dessous" alone is NOT in the list (it appears in "(chiffres clés ci-dessous)" legitimately).
        SCRIPT_KEYWORDS = [
            "script",            # matches "le script", "un script", "ce script"
            "le code",           # matches "le code ci-dessous", "le code suivant"
            "ce code",           # "ce code calcule"
            "the code",          # English variant
            "exécutez",          # "exécutez-le", "exécutez ce code"
            "lancez",            # "lancez le script"
            "run this",          # English variant
            "calculé par le",    # "calculé par le code"
            "le programme",      # "le programme calcule"
            "ce programme",      # "ce programme affiche"
            "le calcul suivant", # formal precede-code phrase
            "voici le calcul",   # precedes code block
            "code ci-dessous",   # compound phrase — specific
            "script ci-dessous", # compound phrase — specific
            "code suivant",      # "le code suivant calcule"
            "script suivant",    # "le script suivant"
            "voici un code",     # "voici un code qui"
            "voici le code",     # "voici le code:"
            "le script ci",      # "le script ci-dessous"
            "effectue le calcul",# "effectue le calcul suivant"
        ]

        # Split by sentence-ending punctuation followed by space/newline
        sentences = re.split(r'(?<=[.!?])\s+', text)
        clean_sentences = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(kw in lower for kw in SCRIPT_KEYWORDS):
                continue  # drop this sentence entirely
            clean_sentences.append(sentence)
        text = " ".join(clean_sentences)

        # ── C. Strip "Dans le dataset apf_data" / "Dans le dataset X" ──
        text = re.sub(r"(?i)dans le dataset\s+\S+[,.]?\s*", "", text)

        # ── D. Strip label lines (📋 Résultats:, etc.) ──
        text = re.sub(r"(?im)^[📋📊✅]\s*(résultats?|results?|analyse)\s*:?\s*$\n?", "", text)
        text = re.sub(r"(?im)^(résultats?\s*:)\s*$\n?", "", text)

        # ── E. Strip PART N — labels ──
        text = re.sub(r"(?m)^PART\s+\d+\s*[—–\-]+\s*[^\n]*\n?", "", text)

        # ── F-bis. Strip NOTE/NB sentences with internal terms (before table stripping) ──
        sentences2 = re.split(r'(?<=[.!?])\s+', text)
        _internal_kw = ["fact_", "dbo_GOLD", "eht_id", "nationalite_name",
                        "hebergementnationalite", "table fact"]
        text = " ".join(
            s for s in sentences2
            if not (s.lstrip().upper().startswith(("NOTE:", "NB:"))
                    and any(k in s for k in _internal_kw))
        )

        # ── F-ter. Strip internal table/schema references ──
        text = re.sub(r"\[dbo_GOLD\]\.\[[^\]]+\]", "", text)
        text = re.sub(r"(?i)\b(table\s+)?fact_statistiques\w*", "", text)
        text = re.sub(r"(?i)\bgld_dim_\w+", "", text)

        # ── F-quater. Strip chart path lines ──
        text = re.sub(r"(?m)^.*(?:Chart:|charts[/\\])\S*\.html.*$\n?", "", text)

        # ── F. Collapse multiple blank lines ──
        text = re.sub(r"\n{3,}", "\n\n", text)

        # ── G. Discard text containing unresolved Python f-string placeholders ──
        # Pattern: {variable_name:format_spec} e.g. {total:,} {pct_change:+.1%} {fr:,}
        # These appear when the model writes its narrative with Python format strings
        # instead of actual computed values. The exec_output has the real values.
        if re.search(r'\{[a-z_][a-z_0-9]*:[^}]{1,20}\}', text):
            return ""  # suppress — exec_output will display the real numbers

        return text.strip()

    def _format_success(self, assistant_message: str, exec_result: Dict) -> str:
        """Format a successful execution response."""
        parts = []

        text_part = self._extract_text(assistant_message)
        if text_part:
            text_part = self._clean_response_text(text_part, is_exec_output=False)
            if text_part:
                parts.append(text_part)

        if exec_result.get("output"):
            output = exec_result["output"].strip()
            if output:
                clean_lines = [
                    line for line in output.split("\n")
                    if not ((".html" in line or os.sep in line)
                            and any(c in line for c in ["charts/", "charts\\", "chart_", ".html"]))
                ]
                output = "\n".join(clean_lines).strip()
                output = self._clean_response_text(output, is_exec_output=True)
            if output:
                parts.append(f"\n{output}")

        result = "\n".join(parts) if parts else "Analyse terminée."

        result = re.sub(
            r"\[GRAPHIQUE[_ ](\d+)\]",
            lambda m: f"<!-- chart:{m.group(1)} -->",
            result,
        )

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Main Chat — Thread-safe
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _norm_text(text: str) -> str:
        """Lowercase ASCII-ish text for robust French intent matching."""
        if any(marker in (text or "") for marker in ("Ã", "Â", "â")):
            try:
                text = text.encode("latin1").decode("utf-8")
            except Exception:
                pass
        text = unicodedata.normalize("NFKD", text or "")
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()

    @staticmethod
    def _fmt_num(value) -> str:
        if value is None or pd.isna(value):
            return "-"
        return f"{int(round(float(value))):,}"

    @staticmethod
    def _extract_year_from_text(message: str, default: Optional[int] = None) -> Optional[int]:
        m = re.search(r"\b(20[12]\d)\b", message or "")
        return int(m.group(1)) if m else default

    def _extract_month_from_text(self, message: str) -> Optional[Tuple[int, str]]:
        norm = self._norm_text(message)
        months = [
            ("janvier", 1, "Janvier"), ("janv", 1, "Janvier"), ("jan", 1, "Janvier"),
            ("fevrier", 2, "Fevrier"), ("fev", 2, "Fevrier"), ("feb", 2, "Fevrier"),
            ("mars", 3, "Mars"), ("avril", 4, "Avril"), ("avr", 4, "Avril"),
            ("mai", 5, "Mai"), ("juin", 6, "Juin"), ("juillet", 7, "Juillet"),
            ("juil", 7, "Juillet"), ("aout", 8, "Aout"), ("septembre", 9, "Septembre"),
            ("sept", 9, "Septembre"), ("octobre", 10, "Octobre"),
            ("novembre", 11, "Novembre"), ("decembre", 12, "Decembre"),
            ("dec", 12, "Decembre"),
        ]
        for token, num, label in months:
            if re.search(r"\b" + re.escape(token) + r"\b", norm):
                return num, label
        return None

    def _latest_complete_hebergement_year(self) -> Optional[int]:
        if not getattr(self, "_db", None) or self._db.source != "fabric":
            return None
        try:
            hbg = self._db._qualify("fact_statistiqueshebergementnationaliteestimees")
            df = self._db.safe_query(
                f"SELECT YEAR(date_stat) AS annee, COUNT(DISTINCT MONTH(date_stat)) AS mois_count "
                f"FROM {hbg} WHERE date_stat IS NOT NULL "
                f"GROUP BY YEAR(date_stat) HAVING COUNT(DISTINCT MONTH(date_stat)) = 12 "
                f"ORDER BY annee DESC"
            )
            if not df.empty:
                return int(df.iloc[0]["annee"])
        except Exception as e:
                logger.debug("Latest complete hebergement year lookup failed: %s", e)
        return None

    def _stamp_period(self, response: str, user_message: str) -> str:
        years = list(dict.fromkeys(re.findall(r"\b(20[12]\d)\b", user_message or "")))
        if len(years) == 1 and years[0] not in (response or ""):
            return f"Periode demandee: {years[0]}.\n\n{response}"
        return response

    def _write_monthly_chart(self, df: pd.DataFrame, title: str, x_col: str, y_col: str) -> Optional[str]:
        """Generate a monthly chart using the premium ChartEngine."""
        try:
            os.makedirs(CHARTS_DIR, exist_ok=True)
            self._cleanup_old_charts()
            self.chart_count += 1
            
            # Use premium chart engine for consistent styling
            fig = chart_line(df, x=x_col, y=y_col, title=title, markers=True, height=500)
            
            # Save with premium settings
            path = chart_save(fig, prefix=f"monthly_{self.chart_count}")
            self.last_chart_paths = [path]
            return path
        except Exception as e:
            logger.warning("Deterministic chart generation failed: %s", e)
            return None

    def _try_city_comparison_answer(self, user_message: str, hbg_table: str) -> Optional[str]:
        """Deterministic city evolution/comparison answer with contract charts."""
        msg = self._norm_text(user_message)
        asks_city = "casablanca" in msg
        asks_chart = any(k in msg for k in ["graph", "graphe", "graphique", "chart", "courbe", "visualis"])
        asks_comparison = any(k in msg for k in ["compare", "comparaison", "autre ville", "autres villes", "marrakech", "tanger"])
        if not (asks_city and asks_chart and asks_comparison):
            return None

        latest_year = self._latest_complete_hebergement_year()
        if not latest_year:
            latest_year = 2025
        start_year = latest_year - 2
        case_sql = (
            "CASE "
            "WHEN UPPER(province_name) LIKE '%CASABLANCA%' THEN 'Casablanca' "
            "WHEN UPPER(province_name) LIKE '%MARRAKECH%' THEN 'Marrakech' "
            "WHEN UPPER(province_name) LIKE '%TANGER%' THEN 'Tanger' "
            "ELSE 'Autre' END"
        )
        city_filter = (
            "UPPER(province_name) LIKE '%CASABLANCA%' OR "
            "UPPER(province_name) LIKE '%MARRAKECH%' OR "
            "UPPER(province_name) LIKE '%TANGER%'"
        )
        try:
            df = self._db.safe_query(
                f"""
                SELECT YEAR(date_stat) AS annee,
                       {case_sql} AS ville,
                       SUM(arrivees) AS arrivees,
                       SUM(nuitees) AS nuitees
                FROM {hbg_table}
                WHERE YEAR(date_stat) BETWEEN {start_year} AND {latest_year}
                  AND ({city_filter})
                GROUP BY YEAR(date_stat), {case_sql}
                ORDER BY annee, ville
                """
            )
            if df.empty:
                return None

            df["annee"] = df["annee"].astype(int)
            df["arrivees"] = pd.to_numeric(df["arrivees"], errors="coerce").fillna(0)
            df["nuitees"] = pd.to_numeric(df["nuitees"], errors="coerce").fillna(0)
            os.makedirs(CHARTS_DIR, exist_ok=True)
            self._cleanup_old_charts()

            paths: List[str] = []
            evo = df[df["ville"] == "Casablanca"].copy()
            if not evo.empty:
                self.chart_count += 1
                fig = chart_line(
                    evo,
                    x="annee",
                    y="arrivees",
                    title=f"Casablanca - evolution des arrivees hotelieres ({start_year}-{latest_year})",
                    markers=True,
                    height=500,
                )
                paths.append(chart_save(fig, prefix=f"casablanca_evolution_{self.chart_count}"))

            comp = df[df["annee"] == latest_year].copy()
            if not comp.empty:
                comp = comp.sort_values("arrivees", ascending=False)
                self.chart_count += 1
                fig = chart_bar(
                    comp,
                    x="ville",
                    y="arrivees",
                    title=f"Comparaison des arrivees hotelieres - {latest_year}",
                    height=500,
                )
                paths.append(chart_save(fig, prefix=f"city_comparison_{self.chart_count}"))

            self.last_chart_paths = [p for p in paths if p]
            if not self.last_chart_paths:
                return None

            comp_display = comp[["ville", "arrivees", "nuitees"]].rename(
                columns={"ville": "Ville", "arrivees": "Arrivees", "nuitees": "Nuitees"}
            )
            evo_display = evo[["annee", "arrivees", "nuitees"]].rename(
                columns={"annee": "Annee", "arrivees": "Arrivees Casablanca", "nuitees": "Nuitees Casablanca"}
            )
            casa_latest = evo[evo["annee"] == latest_year]
            casa_first = evo[evo["annee"] == start_year]
            growth = None
            if not casa_latest.empty and not casa_first.empty:
                first = float(casa_first.iloc[0]["arrivees"] or 0)
                latest = float(casa_latest.iloc[0]["arrivees"] or 0)
                if first:
                    growth = (latest / first - 1) * 100
            growth_line = (
                f"Casablanca progresse de {growth:+.1f}% sur {start_year}-{latest_year}."
                if growth is not None
                else "Casablanca dispose d'une serie exploitable sur la periode retenue."
            )
            return (
                f"## Synthese executive - Casablanca ({start_year}-{latest_year})\n"
                f"- {growth_line}\n"
                "- Marrakech reste le benchmark loisirs le plus puissant; Tanger capte une dynamique affaires/industrie/portuaire.\n"
                "- Casablanca doit etre lue comme destination business, MICE et transit: la performance depend davantage des salons, de la connectivite aerienne, des taux d'occupation semaine et de la conversion leisure que d'un seul volume d'arrivees.\n\n"
                "## Decisions pour top management\n"
                "1. Piloter Casablanca avec un tableau de bord MICE/business separe du benchmark loisirs Marrakech.\n"
                "2. Comparer mensuellement Casablanca a Marrakech et Tanger sur arrivees, nuitees et DMS, pas uniquement les arrivees.\n"
                "3. Concentrer les actions sur connectivite, congresses, produits city-break et conversion des passagers en nuitees.\n\n"
                "## Evolution Casablanca\n"
                + _df_to_markdown(evo_display)
                + "\n\n## Comparaison villes\n"
                + _df_to_markdown(comp_display)
                + "\n\nPerimetre: hebergement classe EHTC/STDN + estimatif; villes/provinces rapprochees via province_name. "
                "Les arrivees APF ne sont pas un perimetre ville."
            )
        except Exception as e:
            logger.debug("Deterministic city comparison failed: %s", e)
            return None

    def try_official_kpi_answer(self, user_message: str, domain_context: Optional[str] = None) -> Optional[str]:
        """Answer high-frequency official KPI prompts with fixed SQL/templates."""
        if not getattr(self, "_db", None) or self._db.source != "fabric":
            return None

        msg = self._norm_text(user_message)
        years_all = list(dict.fromkeys(int(y) for y in re.findall(r"\b(20[12]\d)\b", user_message or "")))
        year = years_all[0] if len(years_all) == 1 else None
        month_info = self._extract_month_from_text(user_message)
        apf = self._db._qualify("fact_statistiques_apf")
        hbg = self._db._qualify("fact_statistiqueshebergementnationaliteestimees")

        has_chart = any(k in msg for k in ["graphique", "graphe", "chart", "courbe", "visualis", "plot"])
        asks_arrivals = any(k in msg for k in ["arrivee", "arrivees", "touriste", "touristes"])
        asks_hotel = any(k in msg for k in ["hoteliere", "hotelieres", "hotel", "hebergement", "ehtc"])
        asks_nights = any(k in msg for k in ["nuitee", "nuitees", "stdn"])
        asks_dms = bool(re.search(r"\bdms\b", msg)) or "duree moyenne de sejour" in msg
        asks_apf = any(k in msg for k in ["apf", "frontiere", "frontieres", "poste", "mre", "tes", "dgsn"])
        asks_region = "region" in msg
        asks_top_country = "top" in msg and any(k in msg for k in ["pays", "residence", "nationalite"])
        asks_monthly = any(k in msg for k in ["par mois", "mensuel", "mensuelle", "mois"])
        asks_voie = any(k in msg for k in ["voie", "aerien", "aeroport", "maritime", "terrestre"])

        city_answer = self._try_city_comparison_answer(user_message, hbg)
        if city_answer:
            return city_answer

        if month_info and asks_arrivals and not has_chart:
            month_num, month_name = month_info
            query_year = year or (self.kpi_cache.get("last_year") if self.kpi_cache else None)
            if query_year:
                try:
                    sections = []
                    if asks_apf or (not asks_hotel and domain_context != "hebergement"):
                        apf_df = self._db.safe_query(
                            f"SELECT SUM(mre) AS [MRE], SUM(tes) AS [TES], SUM(mre + tes) AS [Arrivees APF] "
                            f"FROM {apf} WHERE YEAR(date_stat) = {query_year} AND MONTH(date_stat) = {month_num}"
                        )
                        if not apf_df.empty:
                            sections.append("APF/DGSN:\n" + _df_to_markdown(apf_df))
                    if asks_hotel or (not asks_apf and domain_context != "apf"):
                        hbg_df = self._db.safe_query(
                            f"SELECT SUM(arrivees) AS [Arrivees hotelieres], SUM(nuitees) AS [Nuitees] "
                            f"FROM {hbg} WHERE YEAR(date_stat) = {query_year} AND MONTH(date_stat) = {month_num}"
                        )
                        if not hbg_df.empty:
                            sections.append("Hebergement classe:\n" + _df_to_markdown(hbg_df))
                    if sections:
                        default_note = "" if year else " (annee la plus recente disponible)"
                        return (
                            f"## Arriv\u00e9es - {month_name} {query_year}{default_note}\n"
                            + "\n\n".join(sections)
                            + "\nPerimetres separes: APF = postes frontieres; hebergement = etablissements classes."
                        )
                except Exception as e:
                    logger.debug("Deterministic monthly arrivals failed: %s", e)

        if year and asks_apf and asks_arrivals and not has_chart:
            total = self.kpi_cache.total_for_year(year) if self.kpi_cache else None
            if total is not None:
                mre = self.kpi_cache.get("mre_by_year", {}).get(year, 0)
                tes = self.kpi_cache.get("tes_by_year", {}).get(year, 0)
                response = f"En **{year}**, le Maroc a enregistre **{self._fmt_num(total)}** arrivees aux postes frontieres (APF)."
                if mre or tes:
                    mre_pct = round(mre / total * 100, 1) if total else 0
                    tes_pct = round(tes / total * 100, 1) if total else 0
                    response += f"\n\nDetail: **{self._fmt_num(mre)} MRE** ({mre_pct}%) et **{self._fmt_num(tes)} TES** ({tes_pct}%)."
                return response + "\n\nPerimetre: APF/DGSN, arrivees aux postes frontieres."

        if year and asks_hotel and asks_arrivals and not has_chart:
            try:
                df = self._db.safe_query(
                    f"SELECT SUM(arrivees) AS arrivees_hotelieres, SUM(nuitees) AS nuitees "
                    f"FROM {hbg} WHERE YEAR(date_stat) = {year}"
                )
                if not df.empty:
                    row = df.iloc[0]
                    return (
                        f"En **{year}**, les **arrivees hotelieres** dans les etablissements d'hebergement classes "
                        f"s'elevent a **{self._fmt_num(row['arrivees_hotelieres'])}**.\n\n"
                        f"Nuitees associees: **{self._fmt_num(row['nuitees'])}**.\n\n"
                        "Perimetre: hebergement classe EHTC/STDN + estimatif."
                    )
            except Exception as e:
                logger.debug("Deterministic hotel arrivals failed: %s", e)

        if year and asks_arrivals and not asks_apf and not asks_hotel and not has_chart:
            try:
                apf_total = self.kpi_cache.total_for_year(year) if self.kpi_cache else None
                hbg_df = self._db.safe_query(
                    f"SELECT SUM(arrivees) AS [Arrivees hotelieres], SUM(nuitees) AS [Nuitees] "
                    f"FROM {hbg} WHERE YEAR(date_stat) = {year}"
                )
                sections = []
                if apf_total is not None:
                    sections.append(f"APF/DGSN: **{self._fmt_num(apf_total)}** arrivees aux postes frontieres.")
                if not hbg_df.empty:
                    sections.append("Hebergement classe:\n" + _df_to_markdown(hbg_df))
                if sections:
                    return (
                        f"## Arriv\u00e9es - {year}\n"
                        + "\n\n".join(sections)
                        + "\n\nNote: la question est ambigue; les arriv\u00e9es APF et hebergement sont deux perimetres distincts."
                    )
            except Exception as e:
                logger.debug("Deterministic ambiguous arrivals failed: %s", e)

        if year and asks_nights and asks_region:
            try:
                df = self._db.safe_query(
                    f"SELECT region_name AS [Region], SUM(nuitees) AS [Nuitees] "
                    f"FROM {hbg} WHERE YEAR(date_stat) = {year} "
                    f"GROUP BY region_name ORDER BY [Nuitees] DESC"
                )
                if not df.empty:
                    return f"## Nuitees par region - {year}\n" + _df_to_markdown(df) + "\nPerimetre: hebergement classe EHTC/STDN + estimatif."
            except Exception as e:
                logger.debug("Deterministic nuitées by region failed: %s", e)

        if year and asks_voie and not has_chart:
            try:
                df = self._db.safe_query(
                    f"SELECT voie AS [Voie], SUM(mre) AS [MRE], SUM(tes) AS [TES], "
                    f"SUM(mre + tes) AS [Arrivees APF] "
                    f"FROM {apf} WHERE YEAR(date_stat) = {year} "
                    f"GROUP BY voie ORDER BY [Arrivees APF] DESC"
                )
                if not df.empty:
                    total = float(df["Arrivees APF"].sum() or 0)
                    if total:
                        df["Part (%)"] = (df["Arrivees APF"] / total * 100).round(1)
                    return f"## Repartition par voie APF - {year}\n" + _df_to_markdown(df) + "\nPerimetre: APF/DGSN."
            except Exception as e:
                logger.debug("Deterministic APF by voie failed: %s", e)

        if year and asks_top_country and (asks_apf or domain_context == "apf" or (not asks_hotel and not asks_nights)):
            try:
                df = self._db.safe_query(
                    f"SELECT TOP 10 nationalite AS [Pays de residence], "
                    f"SUM(mre) AS [MRE], SUM(tes) AS [TES], SUM(mre + tes) AS [Arrivees APF] "
                    f"FROM {apf} WHERE YEAR(date_stat) = {year} "
                    f"GROUP BY nationalite ORDER BY [Arrivees APF] DESC"
                )
                if not df.empty:
                    return f"## Top 10 pays de residence APF - {year}\n" + _df_to_markdown(df)
            except Exception as e:
                logger.debug("Deterministic APF top countries failed: %s", e)

        if year and asks_top_country and (asks_hotel or asks_nights or domain_context == "hebergement"):
            try:
                df = self._db.safe_query(
                    f"SELECT TOP 10 nationalite_name AS [Pays de residence], "
                    f"SUM(arrivees) AS [Arrivees hotelieres], SUM(nuitees) AS [Nuitees] "
                    f"FROM {hbg} WHERE YEAR(date_stat) = {year} "
                    f"GROUP BY nationalite_name ORDER BY [Arrivees hotelieres] DESC"
                )
                if not df.empty:
                    return f"## Top 10 pays de residence hebergement - {year}\n" + _df_to_markdown(df)
            except Exception as e:
                logger.debug("Deterministic HBG top countries failed: %s", e)

        if year and has_chart and asks_arrivals and asks_monthly:
            try:
                if asks_hotel or domain_context == "hebergement":
                    df = self._db.safe_query(
                        f"SELECT MONTH(date_stat) AS mois, SUM(arrivees) AS arrivees "
                        f"FROM {hbg} WHERE YEAR(date_stat) = {year} "
                        f"GROUP BY MONTH(date_stat) ORDER BY mois"
                    )
                    title = f"Arrivees hotelieres par mois - {year}"
                    scope = "hebergement classe EHTC/STDN + estimatif"
                else:
                    df = self._db.safe_query(
                        f"SELECT MONTH(date_stat) AS mois, SUM(mre + tes) AS arrivees "
                        f"FROM {apf} WHERE YEAR(date_stat) = {year} "
                        f"GROUP BY MONTH(date_stat) ORDER BY mois"
                    )
                    title = f"Arrivees APF par mois - {year}"
                    scope = "APF/DGSN, postes frontieres"
                if not df.empty:
                    df["mois_fr"] = df["mois"].astype(int).map(MONTH_NAMES_FR)
                    chart_path = self._write_monthly_chart(df, title, "mois_fr", "arrivees")
                    if chart_path:
                        self.last_chart_paths = [chart_path]
                    if not chart_path:
                        return "⚠️ Graphique demande, mais la generation Plotly a echoue. Les donnees sont disponibles:\n\n" + _df_to_markdown(df)
                    display = df[["mois_fr", "arrivees"]].rename(columns={"mois_fr": "Mois", "arrivees": "Arrivees"})
                    return f"## Graphique créé\n{title}.\n\n" + _df_to_markdown(display) + f"\nPérimètre : {scope}."
            except Exception as e:
                logger.debug("Deterministic monthly chart failed: %s", e)

        if asks_dms:
            dms_year = year or self._latest_complete_hebergement_year()
            if dms_year:
                try:
                    df = self._db.safe_query(
                        f"SELECT SUM(nuitees) AS nuitees, SUM(arrivees) AS arrivees "
                        f"FROM {hbg} WHERE YEAR(date_stat) = {dms_year}"
                    )
                    if not df.empty:
                        row = df.iloc[0]
                        arrivals = float(row["arrivees"] or 0)
                        nights = float(row["nuitees"] or 0)
                        dms = nights / arrivals if arrivals else 0
                        default_note = "" if year else " (derniere annee complete disponible)"
                        return (
                            f"## DMS - {dms_year}{default_note}\n"
                            f"La **DMS (Duree Moyenne de Sejour)** est de **{dms:.2f} nuits**.\n\n"
                            f"Calcul: **{self._fmt_num(nights)} nuitees** / **{self._fmt_num(arrivals)} arrivees hotelieres**.\n\n"
                            "Perimetre: hebergement classe EHTC/STDN + estimatif. Cette DMS ne doit pas etre calculee avec les arrivees APF."
                        )
                except Exception as e:
                    logger.debug("Deterministic DMS failed: %s", e)

        return None

    def chat(self, user_message: str, domain_context: Optional[str] = None) -> str:
        """
        Process a data analytics request.

        domain_context: active conversation domain from the orchestrator.
          "hebergement" → skip APF KPI cache, hint LLM to use hébergement table.
          "apf"         → normal APF fast-path.
          None          → unknown, fall through to per-message detection.

        Workflow:
            1. Check datasets available
            2. Try KPI cache (skipped if domain_context == "hebergement")
            3. Optionally gather web context
            4. Call LLM with domain hint injected
            5. Execute sandbox code, retry on failure
            6. Format and return results
        """
        with self._lock:
            return self._chat_internal(user_message, domain_context)

    def _chat_internal(self, user_message: str, domain_context: Optional[str] = None) -> str:
        """Internal chat logic (called with self._lock held)."""
        self.last_chart_paths = []

        if not self.datasets:
            return (
                "❌ Aucune table Fabric n'est cataloguée. Vérifiez la configuration "
                "Fabric (endpoint, base, schéma et authentification Azure)."
            )

        # ── Fast-path: KPI cache (APF only) ──
        # domain_context="hebergement" → cache is APF-only, skip it entirely.
        # Per-message hébergement keywords are also checked inside try_answer().
        official_answer = self.try_official_kpi_answer(user_message, domain_context=domain_context)
        if official_answer:
            logger.debug("Official KPI fast path hit for: %s", user_message[:60])
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": official_answer})
            return official_answer

        if self.kpi_cache:
            fast_answer = self.kpi_cache.try_answer(user_message, domain_context=domain_context)
            if fast_answer:
                logger.debug("KPI cache hit for: %s", user_message[:60])
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": fast_answer})
                return fast_answer

        self._trim_history()

        # ── Gather optional web context ──
        web_context = ""
        if self._needs_web_search(user_message):
            web_context = self._get_web_context(user_message)

        # ── Build augmented message ──
        # Inject a mandatory temporal filter when the user mentions a year/month.
        # This is appended to the user turn so the model sees it, but we restore
        # the clean message after the call so it doesn't pollute future turns.
        temporal_constraint = self._build_temporal_constraint(user_message)
        augmented = user_message + temporal_constraint

        # Inject domain hint so the LLM picks the right table on follow-ups
        # (e.g. "et en 2025" after a hébergement question stays on hébergement).
        if domain_context == "hebergement":
            augmented += (
                "\n\n[DOMAINE ACTIF: hébergement — interroger "
                "fact_statistiqueshebergementnationaliteestimees (nuitées/arrivées hôtelières).]"
            )
        elif domain_context == "apf":
            augmented += (
                "\n\n[DOMAINE ACTIF: APF — interroger fact_statistiques_apf (MRE/TES/arrivées frontières).]"
            )

        if web_context:
            augmented = f"{augmented}\n\n---\n{web_context}"

        # ── Add to history and call LLM ──
        self.conversation_history.append({
            "role": "user", "content": augmented
        })

        assistant_message = self._call_llm()

        if assistant_message is None:
            # Connection errors are transient — retry the full LLM call once
            logger.warning("Analytics LLM returned None — retrying full call...")
            assistant_message = self._call_llm()

        if assistant_message is None:
            self._pop_last_user_message()
            return "⚠️ Impossible de joindre le service d'analyse. Veuillez réessayer dans quelques instants."

        # ── Store clean user message in history (not augmented) ──
        self.conversation_history[-1] = {
            "role": "user", "content": user_message
        }
        self.conversation_history.append({
            "role": "assistant", "content": assistant_message
        })

        # ── Extract code ──
        code = self._extract_code(assistant_message)
        if not code:
            # No code block — return text response as-is
            return self._stamp_period(assistant_message, user_message)

        # ── Execute code ──
        logger.info("Executing analysis code...")
        current_code = code
        exec_result = self._execute_analysis(code)

        # ── Success ──
        if not exec_result.get("error"):
            self.last_chart_paths = exec_result.get("chart_paths") or []
            return self._stamp_period(self._format_success(assistant_message, exec_result), user_message)

        # ── Failure → Retry loop ──
        # FIX: Was range(1, MAX_RETRIES) which gave only MAX_RETRIES-1 attempts
        for attempt in range(1, MAX_CODE_RETRIES + 1):
            logger.warning(
                "Code error (attempt %d/%d): %s",
                attempt, MAX_CODE_RETRIES,
                exec_result["error"][:100],
            )

            fix_msg = self._build_fix_message(
                exec_result["error"],
                failed_code=current_code,
                original_question=user_message,
            )

            self.conversation_history.append({
                "role": "user", "content": fix_msg
            })

            fixed_response = self._call_llm()

            if fixed_response is None:
                logger.warning("Fix attempt %d: no LLM response", attempt)
                self.conversation_history.pop()  # rollback orphan user message
                continue

            self.conversation_history.append({
                "role": "assistant", "content": fixed_response
            })

            fixed_code = self._extract_code(fixed_response)
            if not fixed_code:
                logger.warning("Fix attempt %d: no code block in response", attempt)
                self.conversation_history.pop()  # rollback assistant (no code)
                self.conversation_history.pop()  # rollback user fix_msg
                continue

            logger.info("Re-executing (attempt %d/%d)...", attempt + 1, MAX_CODE_RETRIES + 1)
            current_code = fixed_code
            exec_result = self._execute_analysis(fixed_code)

            if not exec_result.get("error"):
                logger.info("Fix attempt %d succeeded", attempt)
                self.last_chart_paths = exec_result.get("chart_paths") or []
                return self._stamp_period(self._format_success(assistant_message, exec_result), user_message)

        # ── All retries failed ──
        text_part = self._extract_text(assistant_message)
        logger.error(
            "Code execution failed after %d attempts: %s",
            MAX_CODE_RETRIES, exec_result.get("error", "unknown"),
        )
        return (
            f"{text_part}\n\n"
            f"⚠️ Code execution failed after {MAX_CODE_RETRIES + 1} attempts: "
            f"{exec_result.get('error', 'Unknown error')}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Quick Commands
    # ──────────────────────────────────────────────────────────────────────

    def quick_stats(self, dataset_name: Optional[str] = None) -> str:
        """Get quick statistics for a dataset."""
        name = dataset_name or self.active_dataset_name
        if not name or name not in self.datasets:
            available = ", ".join(self.datasets.keys())
            return f"❌ Not found. Available: {available}"

        df = self.datasets[name]["df"]
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        txt_cols = df.select_dtypes(include=["object"]).columns.tolist()
        date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        lines = [
            f"📊 {name}",
            "=" * 50,
            f"Rows: {len(df):,} | Cols: {len(df.columns)}",
        ]

        if date_cols:
            for col in date_cols:
                lines.append(
                    f"Period: {df[col].min().strftime('%Y-%m')} → "
                    f"{df[col].max().strftime('%Y-%m')}"
                )

        if num_cols:
            lines.append("\nNumeric:")
            for col in num_cols[:8]:
                lines.append(f"  {col:25s}: {df[col].sum():>15,.0f}")

        if txt_cols:
            lines.append("\nCategories:")
            for col in txt_cols[:8]:
                lines.append(f"  {col:25s}: {df[col].nunique():,} unique")

        lines.append("=" * 50)
        return "\n".join(lines)

    def list_datasets(self) -> str:
        """List all loaded datasets."""
        if not self.datasets:
            return "❌ Aucune table Fabric cataloguée."

        lines = ["📁 DATASETS", "=" * 50]
        for name, info in self.datasets.items():
            df = info["df"]
            mark = " ◀ ACTIVE" if name == self.active_dataset_name else ""
            lines.append(
                f"  {name:25s} | {len(df):>8,} rows | "
                f"{len(df.columns):>3} cols{mark}"
            )
        lines.append("=" * 50)
        return "\n".join(lines)

    def get_schema(self, dataset_name: Optional[str] = None) -> str:
        """Get schema for a dataset."""
        name = dataset_name or self.active_dataset_name
        if not name or name not in self.datasets:
            available = ", ".join(self.datasets.keys())
            return f"❌ Not found. Available: {available}"
        return self.datasets[name]["schema"]

    def get_sample(self, n: int = 5, dataset_name: Optional[str] = None) -> str:
        """Get sample rows from a dataset."""
        name = dataset_name or self.active_dataset_name
        if not name or name not in self.datasets:
            available = ", ".join(self.datasets.keys())
            return f"❌ Not found. Available: {available}"
        df = self.datasets[name]["df"]
        return f"📄 {name} (first {n} rows)\n{df.head(n).to_string()}"

    def get_columns(self, dataset_name: Optional[str] = None) -> str:
        """List columns for a dataset."""
        name = dataset_name or self.active_dataset_name
        if not name or name not in self.datasets:
            available = ", ".join(self.datasets.keys())
            return f"❌ Not found. Available: {available}"

        df = self.datasets[name]["df"]
        lines = [f"📋 {name} — Columns", "=" * 50]
        for i, col in enumerate(df.columns, 1):
            lines.append(
                f"  {i:3d}. {col:30s} "
                f"{str(df[col].dtype):15s} "
                f"{df[col].nunique():,} unique"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Interactive CLI for testing the Data Analytics Agent."""
    try:
        from config.settings import validate_config
        validate_config(require_tavily=False)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return

    print()
    print("=" * 60)
    print("📊  STATOUR — Data Analytics Agent")
    print("    Ministère du Tourisme du Maroc")
    print("=" * 60)
    print()
    print("Commands:")
    print("  /datasets        — List loaded datasets")
    print("  /switch <name>   — Switch active dataset")
    print("  /load <file>     — Load new file")
    print("  /stats [name]    — Quick stats")
    print("  /schema [name]   — Dataset schema")
    print("  /columns [name]  — List columns")
    print("  /sample [name]   — Sample rows")
    print("  /reset           — Reset conversation")
    print("  /quit            — Exit")
    print()
    print("=" * 60)
    print()

    agent = DataAnalyticsAgent()

    while True:
        try:
            user_input = input("👤 You: ").strip()
            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd == "/quit":
                print("\n👋 Goodbye!")
                break

            if cmd == "/reset":
                print(f"\n{agent.reset_conversation()}\n")
                continue

            if cmd == "/datasets":
                print(f"\n{agent.list_datasets()}\n")
                continue

            if cmd.startswith("/switch"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    print(f"\n{agent.switch_dataset(parts[1].strip())}\n")
                else:
                    print("\n❌ Usage: /switch <name>\n")
                continue

            if cmd.startswith("/load"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    print(f"\n{agent.load_dataset(parts[1].strip())}\n")
                else:
                    print("\n❌ Usage: /load <filename>\n")
                continue

            if cmd.startswith("/stats"):
                parts = user_input.split(maxsplit=1)
                name = parts[1].strip() if len(parts) > 1 else None
                print(f"\n{agent.quick_stats(name)}\n")
                continue

            if cmd.startswith("/schema"):
                parts = user_input.split(maxsplit=1)
                name = parts[1].strip() if len(parts) > 1 else None
                print(f"\n{agent.get_schema(name)}\n")
                continue

            if cmd.startswith("/columns"):
                parts = user_input.split(maxsplit=1)
                name = parts[1].strip() if len(parts) > 1 else None
                print(f"\n{agent.get_columns(name)}\n")
                continue

            if cmd.startswith("/sample"):
                parts = user_input.split(maxsplit=1)
                name = parts[1].strip() if len(parts) > 1 else None
                print(f"\n{agent.get_sample(5, name)}\n")
                continue

            print("\n⏳ Analyzing...")
            response = agent.chat(user_input)
            print(f"\n🤖 {agent.agent_name}:\n{response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}\n")
            logger.error("CLI error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
