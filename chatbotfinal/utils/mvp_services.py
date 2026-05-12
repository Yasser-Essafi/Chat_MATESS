"""
STATOUR MVP executive services.

Small service layer used by the Flask API:
- readiness checks for demo safety
- dashboard KPI/chart summaries
- cached-plus-live external signals for executive insights
"""

from __future__ import annotations

import importlib
import copy
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import FABRIC_TABLES, CHARTS_DIR
from utils.logger import get_logger
from utils.chart_engine import ChartEngine, chart_bar, chart_line, chart_save

logger = get_logger("statour.mvp")


DEPENDENCY_IMPORTS = {
    "flask": "flask",
    "openai": "openai",
    "pandas": "pandas",
    "plotly": "plotly",
    "sqlalchemy": "sqlalchemy",
    "pyodbc": "pyodbc",
    "azure-identity": "azure.identity",
    "tavily-python": "tavily",
    "exa-py": "exa_py",
}


STATIC_SIGNALS = [
    {
        "title": "Maroc, destination leader en Afrique",
        "content": (
            "UN Tourism indique que le Maroc a accueilli 17,4 millions de "
            "touristes internationaux en 2024, soit +20% vs 2023."
        ),
        "source": "UN Tourism",
        "url": "https://www.unwto.org/news/un-tourism-in-morocco-driving-investments-and-celebrating-innovation-in-africa-s-most-visited-destination",
    },
    {
        "title": "Risques macro et géopolitiques",
        "content": (
            "UN Tourism signale que les coûts de voyage, les tensions "
            "géopolitiques et l'incertitude économique restent des risques "
            "pour la demande touristique internationale."
        ),
        "source": "UN Tourism",
        "url": "https://www.unwto.org/news/international-tourist-arrivals-grew-5-in-q1-2025",
    },
    {
        "title": "Vision 2030 et investissement",
        "content": (
            "Le profil investissement d'UN Tourism met en avant le poids "
            "économique du tourisme marocain et l'opportunité 2030."
        ),
        "source": "UN Tourism",
        "url": "https://www.unwto.org/fr/investissements/tourisme-faire-des-affaires-investir-en-maroc",
    },
]


READINESS_CACHE_TTL_SECONDS = int(os.getenv("READINESS_CACHE_TTL_SECONDS", "15"))
DASHBOARD_CACHE_TTL_SECONDS = int(os.getenv("DASHBOARD_CACHE_TTL_SECONDS", "15"))
_CACHE_LOCK = threading.Lock()
_CACHE: Dict[Any, Dict[str, Any]] = {}


def _cache_key(kind: str, orch=None) -> tuple[str, Any]:
    return (kind, id(orch) if orch is not None else "standalone")


def _get_cached(kind: str, orch, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    if ttl_seconds <= 0:
        return None
    key = _cache_key(kind, orch)
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if not cached:
            return None
        if now - float(cached.get("stored_at", 0)) > ttl_seconds:
            _CACHE.pop(key, None)
            return None
        return copy.deepcopy(cached.get("value"))


def _set_cached(kind: str, orch, value: Dict[str, Any]) -> Dict[str, Any]:
    with _CACHE_LOCK:
        _CACHE[_cache_key(kind, orch)] = {
            "stored_at": time.monotonic(),
            "value": copy.deepcopy(value),
        }
    return value


def clear_service_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _fmt_int(value: Optional[float]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return f"{int(round(float(value))):,}".replace(",", " ")


def _pct(value: Optional[float]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return f"{float(value):+.1f}%"


def dependency_status() -> Dict[str, Any]:
    deps = {name: _module_available(mod) for name, mod in DEPENDENCY_IMPORTS.items()}
    drivers: List[str] = []
    if deps.get("pyodbc"):
        try:
            import pyodbc  # type: ignore

            drivers = list(pyodbc.drivers())
        except Exception:
            drivers = []
    odbc_ok = any(
        d in drivers for d in ["ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"]
    )
    missing = [name for name, ok in deps.items() if not ok]
    if deps.get("pyodbc") and not odbc_ok:
        missing.append("ODBC Driver 17/18 for SQL Server")
    return {
        "dependencies": deps,
        "odbc_drivers": drivers,
        "odbc_driver_ok": odbc_ok,
        "missing": missing,
    }


def _table_month_coverage(db, table_name: str) -> Optional[Dict[str, Any]]:
    qualified = db._qualify(table_name)
    try:
        df = db.safe_query(
            f"""
            SELECT YEAR(date_stat) AS annee, MONTH(date_stat) AS mois
            FROM {qualified}
            WHERE date_stat IS NOT NULL
            GROUP BY YEAR(date_stat), MONTH(date_stat)
            ORDER BY annee DESC, mois DESC
            """
        )
        if df.empty:
            return None
        latest = df.iloc[0]
        months_latest = (
            df[df["annee"] == latest["annee"]]["mois"].astype(int).sort_values().tolist()
        )
        return {
            "table": table_name,
            "latest_year": int(latest["annee"]),
            "latest_month": int(latest["mois"]),
            "months_latest_year": months_latest,
        }
    except Exception as e:
        logger.debug("Coverage check failed for %s: %s", table_name, e)
        return None


def _compute_readiness(orch=None) -> Dict[str, Any]:
    dep = dependency_status()
    fabric = {
        "connected": False,
        "source": "unavailable",
        "status": "not checked",
        "tables": {},
    }
    search = {"available": False, "exa_available": False}
    rag = {"available": False, "chunks": 0}
    latest = {"apf": None, "hebergement": None}

    if orch is not None:
        analytics = getattr(orch, "analytics_agent", None)
        db = getattr(analytics, "_db", None)
        fabric["connected"] = bool(db and db.source == "fabric")
        fabric["source"] = getattr(db, "source", "unavailable")
        fabric["status"] = getattr(db, "status", "")
        fabric["tables"] = {
            name: {
                "catalogued": True,
                "row_count": info.get("row_count"),
                "columns": info.get("col_count"),
            }
            for name, info in getattr(analytics, "datasets", {}).items()
        }

        if db and db.source == "fabric":
            latest["apf"] = _table_month_coverage(db, "fact_statistiques_apf")
            latest["hebergement"] = _table_month_coverage(
                db, "fact_statistiqueshebergementnationaliteestimees"
            )

        researcher = getattr(orch, "researcher_agent", None)
        search = {
            "available": bool(getattr(researcher, "_search_available", False)),
            "exa_available": bool(getattr(researcher, "_exa_available", False)),
        }
        rag_stats = {}
        try:
            if getattr(researcher, "_rag_available", False) and getattr(researcher, "rag", None):
                rag_stats = researcher.rag.get_stats()
        except Exception:
            rag_stats = {}
        rag = {
            "available": bool(getattr(researcher, "_rag_available", False)),
            "chunks": int(rag_stats.get("total_chunks", 0) or 0),
        }
    else:
        try:
            from utils.db_layer import DBLayer

            db = DBLayer()
            fabric["connected"] = db.source == "fabric"
            fabric["source"] = db.source
            fabric["status"] = db.status
            if db.source == "fabric":
                for table in FABRIC_TABLES:
                    try:
                        n = db.safe_query(f"SELECT COUNT(*) AS n FROM {db._qualify(table)}")
                        fabric["tables"][table] = {
                            "catalogued": True,
                            "row_count": int(n.iloc[0, 0]),
                        }
                    except Exception as e:
                        fabric["tables"][table] = {
                            "catalogued": False,
                            "error": str(e)[:200],
                        }
                latest["apf"] = _table_month_coverage(db, "fact_statistiques_apf")
                latest["hebergement"] = _table_month_coverage(
                    db, "fact_statistiqueshebergementnationaliteestimees"
                )
        except Exception as e:
            fabric["status"] = str(e)[:200]

        try:
            from tools.rag_tools import RAGManager

            stats = RAGManager().get_stats()
            rag = {"available": True, "chunks": int(stats.get("total_chunks", 0) or 0)}
        except Exception:
            pass

        try:
            from tools.search_tools import TourismSearchTool

            searcher = TourismSearchTool()
            search = {
                "available": True,
                "exa_available": bool(getattr(searcher, "_exa_available", False)),
            }
        except Exception:
            pass

    ready = (
        fabric["connected"]
        and bool(fabric["tables"])
        and search["available"]
        and rag["available"]
        and not dep["missing"]
    )
    blockers = []
    if not fabric["connected"] or not fabric["tables"]:
        blockers.append("Fabric Gold non connecté ou aucune table cataloguée")
    if not search["available"]:
        blockers.append("Recherche web indisponible pour les insights")
    if not rag["available"]:
        blockers.append("Base RAG indisponible")
    blockers.extend(dep["missing"])

    return {
        "ready": ready,
        "checked_at": datetime.now().isoformat(),
        "fabric": fabric,
        "latest_data": latest,
        "rag": rag,
        "search": search,
        "dependency_status": dep,
        "blockers": blockers,
    }


def get_readiness(orch=None, force_refresh: bool = False) -> Dict[str, Any]:
    if not force_refresh:
        cached = _get_cached("readiness", orch, READINESS_CACHE_TTL_SECONDS)
        if cached is not None:
            return cached
    return _set_cached("readiness", orch, _compute_readiness(orch))


def _safe_scalar(db, sql: str, col: str) -> Optional[float]:
    try:
        df = db.safe_query(sql)
        if df.empty:
            return None
        value = df.iloc[0][col]
        return None if pd.isna(value) else float(value)
    except Exception as e:
        logger.debug("Scalar query failed: %s", e)
        return None


def _build_plotly_chart(df: pd.DataFrame, kind: str, title: str, x: str, y: str, color: str = "") -> Optional[str]:
    """Build a premium chart for the dashboard using ChartEngine.
    
    Args:
        df: DataFrame with the data
        kind: Chart type ('bar' or 'line')
        title: Chart title
        x: X-axis column name
        y: Y-axis column name
        color: Optional color grouping column
    
    Returns:
        URL path to the chart ('/charts/filename.html') or None
    """
    if df.empty:
        return None
    try:
        os.makedirs(CHARTS_DIR, exist_ok=True)
        
        # Use premium ChartEngine for consistent styling
        if kind == "bar":
            fig = chart_bar(df, x=x, y=y, color=color or None, title=title, height=400)
        else:
            fig = chart_line(df, x=x, y=y, color=color or None, title=title, height=400, markers=True)
        
        # Save with premium settings
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = chart_save(fig, prefix=f"dashboard_{kind}_{stamp}")
        
        # Return URL-style path for API
        filename = os.path.basename(path)
        return "/charts/" + filename
    except Exception as e:
        logger.debug("Dashboard chart failed: %s", e)
        return None


def _compute_dashboard_summary(orch=None) -> Dict[str, Any]:
    readiness = get_readiness(orch)
    analytics = getattr(orch, "analytics_agent", None) if orch is not None else None
    db = getattr(analytics, "_db", None)
    kpis: List[Dict[str, Any]] = []
    charts: List[Dict[str, Any]] = []
    tables: Dict[str, Any] = {}

    apf_cov = readiness["latest_data"].get("apf")
    hbg_cov = readiness["latest_data"].get("hebergement")

    if db and getattr(db, "source", None) == "fabric":
        apf = db._qualify("fact_statistiques_apf")
        hbg = db._qualify("fact_statistiqueshebergementnationaliteestimees")

        if apf_cov:
            y, m = apf_cov["latest_year"], apf_cov["latest_month"]
            prev_y = y - 1
            cur_total = _safe_scalar(
                db,
                f"SELECT SUM(mre + tes) AS v FROM {apf} WHERE YEAR(date_stat)={y} AND MONTH(date_stat)={m}",
                "v",
            )
            prev_total = _safe_scalar(
                db,
                f"SELECT SUM(mre + tes) AS v FROM {apf} WHERE YEAR(date_stat)={prev_y} AND MONTH(date_stat)={m}",
                "v",
            )
            cur_tes = _safe_scalar(
                db,
                f"SELECT SUM(tes) AS v FROM {apf} WHERE YEAR(date_stat)={y} AND MONTH(date_stat)={m}",
                "v",
            )
            cur_mre = _safe_scalar(
                db,
                f"SELECT SUM(mre) AS v FROM {apf} WHERE YEAR(date_stat)={y} AND MONTH(date_stat)={m}",
                "v",
            )
            yoy = ((cur_total / prev_total - 1) * 100) if cur_total and prev_total else None
            kpis.extend(
                [
                    {"label": "Arrivées APF", "value": _fmt_int(cur_total), "delta": _pct(yoy), "period": f"{m:02d}/{y}"},
                    {"label": "TES", "value": _fmt_int(cur_tes), "delta": None, "period": f"{m:02d}/{y}"},
                    {"label": "MRE", "value": _fmt_int(cur_mre), "delta": None, "period": f"{m:02d}/{y}"},
                ]
            )

            try:
                trend = db.safe_query(
                    f"""
                    SELECT YEAR(date_stat) AS annee, MONTH(date_stat) AS mois,
                           SUM(mre + tes) AS arrivees
                    FROM {apf}
                    GROUP BY YEAR(date_stat), MONTH(date_stat)
                    ORDER BY annee, mois
                    """
                )
                if not trend.empty:
                    trend["periode"] = trend["annee"].astype(str) + "-" + trend["mois"].astype(int).astype(str).str.zfill(2)
                    url = _build_plotly_chart(trend.tail(24), "line", "Arrivées APF mensuelles", "periode", "arrivees")
                    if url:
                        charts.append({"title": "Arrivées APF mensuelles", "url": url, "type": "line"})
            except Exception:
                pass

            try:
                top = db.safe_query(
                    f"""
                    SELECT TOP 8 nationalite AS pays_residence, SUM(mre + tes) AS arrivees
                    FROM {apf}
                    WHERE YEAR(date_stat)={y}
                    GROUP BY nationalite
                    ORDER BY arrivees DESC
                    """
                )
                if not top.empty:
                    tables["top_markets"] = top.to_dict(orient="records")
                    url = _build_plotly_chart(top, "bar", f"Top pays de résidence {y}", "pays_residence", "arrivees")
                    if url:
                        charts.append({"title": f"Top pays de résidence {y}", "url": url, "type": "bar"})
            except Exception:
                pass

        if hbg_cov:
            y, m = hbg_cov["latest_year"], hbg_cov["latest_month"]
            nuits = _safe_scalar(
                db,
                f"SELECT SUM(nuitees) AS v FROM {hbg} WHERE YEAR(date_stat)={y} AND MONTH(date_stat)={m}",
                "v",
            )
            arrivees_h = _safe_scalar(
                db,
                f"SELECT SUM(arrivees) AS v FROM {hbg} WHERE YEAR(date_stat)={y} AND MONTH(date_stat)={m}",
                "v",
            )
            kpis.extend(
                [
                    {"label": "Nuitées EHTC", "value": _fmt_int(nuits), "delta": None, "period": f"{m:02d}/{y}"},
                    {"label": "Arrivées hôtelières", "value": _fmt_int(arrivees_h), "delta": None, "period": f"{m:02d}/{y}"},
                ]
            )
    else:
        kpis = [
            {"label": "Arrivées Maroc 2024", "value": "17,4 M", "delta": "+20%", "period": "UN Tourism"},
            {"label": "Statut Fabric", "value": "Hors ligne", "delta": None, "period": "readiness"},
            {"label": "Recherche web", "value": "À vérifier", "delta": None, "period": "readiness"},
            {"label": "RAG interne", "value": f"{readiness['rag']['chunks']} chunks", "delta": None, "period": "base interne"},
        ]

    return {
        "status": "ok" if readiness["ready"] else "degraded",
        "generated_at": datetime.now().isoformat(),
        "kpis": kpis,
        "charts": charts,
        "signals": STATIC_SIGNALS,
        "tables": tables,
        "data_freshness": readiness["latest_data"],
        "readiness": readiness,
    }


def get_dashboard_summary(orch=None, force_refresh: bool = False) -> Dict[str, Any]:
    if not force_refresh:
        cached = _get_cached("dashboard", orch, DASHBOARD_CACHE_TTL_SECONDS)
        if cached is not None:
            return cached
    return _set_cached("dashboard", orch, _compute_dashboard_summary(orch))


def external_signal_context(searcher, message: str, max_results: int = 3) -> Dict[str, Any]:
    results = []
    if searcher:
        try:
            query = (
                f"{message} Maroc tourisme impact facteurs externes "
                "géopolitique connectivité aérienne marchés émetteurs"
            )
            results = searcher.search(query, max_results=max_results, use_trusted_only=True)
        except Exception as e:
            logger.debug("Live signal search failed: %s", e)
            results = []
    if not results:
        results = STATIC_SIGNALS
    sources = []
    context_lines = []
    for item in results[:max_results]:
        title = item.get("title") or item.get("source") or "Source"
        url = item.get("url") or ""
        content = item.get("content") or ""
        sources.append({"title": title, "url": url, "source": item.get("source") or title})
        context_lines.append(f"- {title}: {content[:350]} ({url})")
    return {"context": "\n".join(context_lines), "sources": sources}
