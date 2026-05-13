"""
Executor Node — Tool-Calling Execution Loop
=============================================
Executes plan steps by calling the appropriate tools.
Each step produces structured Evidence that feeds into the Reviewer/Synthesizer.

Tools wrapped:
- sql: Fabric Lakehouse Gold T-SQL via DBLayer
- web_search: Tavily/Exa/Brave via TourismSearchTool
- rag: ChromaDB vector search via RAGManager
- prediction: PredictionEngine (CAGR + seasonality)
- chart: LLM-generated Plotly code via sandboxed execution
"""

import os
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from utils.base_agent import get_shared_client, _prepare_messages, _should_skip_reasoning_effort
from utils.logger import get_logger
from config.settings import (
    AZURE_OPENAI_DEPLOYMENT,
    MODEL_IS_REASONING,
    ANALYTICS_REASONING_EFFORT,
    ANALYTICS_MAX_COMPLETION_TOKENS,
    CHARTS_DIR,
    MAX_CODE_RETRIES,
)
from .planner import PlanStep, ExecutionPlan

logger = get_logger("statour.orchestration.executor")


_CITY_OR_PROVINCE_MARKERS = re.compile(
    r"\b(casablanca|marrakech|agadir|tanger|rabat|fes|fès|meknes|meknès|"
    r"oujda|essaouira|tetouan|tétouan|laayoune|laâyoune|dakhla|safi|el\s+jadida)\b",
    re.IGNORECASE,
)


def _result_looks_empty(result: Dict[str, Any]) -> bool:
    output = str(result.get("output") or "")
    return (
        result.get("data") is None
        and not result.get("error")
        and (
            not output.strip()
            or "aucune donnée" in output.lower()
            or "aucune donnee" in output.lower()
            or "no data" in output.lower()
        )
    )


def _should_retry_location_query(message: str, code: str, result: Dict[str, Any]) -> bool:
    if not _result_looks_empty(result):
        return False
    if not _CITY_OR_PROVINCE_MARKERS.search(message or ""):
        return False

    code_lower = (code or "").lower()
    return (
        "fact_statistiqueshebergementnationaliteestimees" in code_lower
        and (
            "province_name" not in code_lower
            or "region_name = 'casablanca'" in code_lower
            or 'region_name = "casablanca"' in code_lower
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# Evidence Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Evidence:
    """Output from a single execution step."""
    step_id: int
    tool: str
    success: bool
    data: Any = None  # DataFrame, str, dict — depends on tool
    text_summary: str = ""  # human-readable summary of findings
    chart_paths: List[str] = field(default_factory=list)
    error: str = ""
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Aggregated evidence from all execution steps."""
    evidence: List[Evidence] = field(default_factory=list)
    chart_paths: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0
    all_successful: bool = True
    errors: List[str] = field(default_factory=list)

    def text_context(self) -> str:
        """Combine all evidence text summaries into a single context string."""
        parts = []
        for ev in self.evidence:
            if ev.text_summary:
                label = f"[{ev.tool.upper()}]"
                parts.append(f"{label} {ev.text_summary}")
        return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# SQL Code Generation Prompt
# ══════════════════════════════════════════════════════════════════════════════

_SQL_GEN_PROMPT = """Tu génères du code Python qui utilise la fonction sql() pour interroger Microsoft Fabric.

ENVIRONNEMENT SANDBOX:
- sql(query) → pd.DataFrame (exécute T-SQL sur Fabric, max 50000 lignes)
- pd, np (pandas, numpy disponibles)
- save_chart(fig, label) → str (sauvegarde un graphique Plotly, retourne le path)
- to_md(df) → str (convertit un DataFrame en tableau Markdown)
- MONTH_NAMES_FR = {1: 'Janvier', ..., 12: 'Décembre'}
- import plotly.express as px, plotly.graph_objects as go (disponibles)
- print() pour afficher les résultats textuels

TABLES DISPONIBLES:
- [dbo_GOLD].[fact_statistiques_apf] — Arrivées aux postes frontières
  Colonnes: statistiques_apf_id, nationalite, poste_frontiere, region, continent, voie, date_stat, mre, tes
  Note: "nationalite" = pays de résidence (PAS la nationalité ethnique)
  Métriques: mre (Marocains résidant à l'étranger), tes (Touristes étrangers), total = mre + tes
  Données mensuelles: vérifier les mois disponibles par année avant toute comparaison partielle

- [dbo_GOLD].[fact_statistiqueshebergementnationaliteestimees] — Nuitées/arrivées hôtelières estimées
  Colonnes: eht_id, nationalite_name, categorie_name, province_name, region_name, date_stat, nuitees, arrivees
  Note: nationalite_name = pays de résidence; date_stat = granularité mensuelle
  Géographie: province_name = ville/province (ex. Casablanca, Marrakech, Tanger), region_name = région administrative (ex. Casablanca-Settat).
  Si l'utilisateur dit seulement "Casablanca", "Marrakech", "Tanger", etc., filtrer province_name avec UPPER(province_name) LIKE '%CASABLANCA%'.

JOINTURES:
- gld_dim_categories_classements ON categorie_name → type_eht_libelle
- gld_dim_etablissements_hebergements ON CAST(etablissement_id_genere AS VARCHAR) = eht_id
- gld_dim_delegations ON delegation_bk = delegation_id → delegation_name

RÈGLES:
- TOUJOURS préfixer les tables: [dbo_GOLD].[nom_table]
- TOUJOURS agréger en SQL (GROUP BY, SUM, COUNT, TOP N) — JAMAIS tirer des millions de lignes
- date_stat est mensuel: filtrer avec date_stat IS NOT NULL AND YEAR(date_stat) = X AND MONTH(date_stat) = Y
- Si 2026 est partielle, comparer seulement les mêmes mois disponibles entre années
- Les séries historiques internes ne couvrent pas forcément toutes les années demandées: si une plage inclut 2018 ou 2020-2022, interroger les années disponibles et imprimer une note de couverture/missing years, sans conclure que toute la période est vide.
- Utiliser print(to_md(df)) pour afficher les résultats
- Si graphique demandé: utiliser save_chart(fig, "label") — PAS fig.write_html()

GÉNÈRE UNIQUEMENT LE CODE PYTHON (pas de ```python, pas d'explications):"""


# ══════════════════════════════════════════════════════════════════════════════
# Executor Class
# ══════════════════════════════════════════════════════════════════════════════

class Executor:
    """Executes plan steps using available tools."""

    def __init__(self, db_layer=None, search_tool=None, rag_manager=None, prediction_engine=None):
        self._db = db_layer
        self._search = search_tool
        self._rag = rag_manager
        self._prediction_engine = prediction_engine
        self._chart_count = 0

    def execute_plan(self, plan: ExecutionPlan, message: str, conversation_context: str = "") -> ExecutionResult:
        """
        Execute all steps in a plan sequentially, collecting evidence.

        Args:
            plan: The execution plan from the Planner
            message: Original user message
            conversation_context: Recent conversation context

        Returns:
            ExecutionResult with all collected evidence
        """
        start = time.time()
        result = ExecutionResult()
        step_results: Dict[int, Evidence] = {}

        for step in plan.steps:
            ev = self._execute_step(step, message, step_results, conversation_context)
            step_results[step.step_id] = ev
            result.evidence.append(ev)

            if ev.chart_paths:
                result.chart_paths.extend(ev.chart_paths)

            if not ev.success:
                result.all_successful = False
                if ev.error:
                    result.errors.append(f"Step {step.step_id} ({step.tool}): {ev.error}")

        result.total_duration_ms = (time.time() - start) * 1000
        logger.info(
            "Execution complete: %d steps, %d charts, %.0fms, success=%s",
            len(result.evidence), len(result.chart_paths),
            result.total_duration_ms, result.all_successful,
        )
        return result

    def _execute_step(
        self,
        step: PlanStep,
        message: str,
        prior_results: Dict[int, Evidence],
        conversation_context: str,
    ) -> Evidence:
        """Execute a single plan step."""
        start = time.time()

        try:
            if step.tool == "sql":
                ev = self._exec_sql(step, message, prior_results, conversation_context)
            elif step.tool == "web_search":
                ev = self._exec_web_search(step, message, prior_results)
            elif step.tool == "rag":
                ev = self._exec_rag(step, message)
            elif step.tool == "prediction":
                ev = self._exec_prediction(step, message, prior_results)
            elif step.tool == "chart":
                ev = self._exec_chart(step, message, prior_results, conversation_context)
            else:
                ev = Evidence(
                    step_id=step.step_id,
                    tool=step.tool,
                    success=False,
                    error=f"Unknown tool: {step.tool}",
                )
        except Exception as e:
            logger.error("Step %d (%s) failed: %s", step.step_id, step.tool, str(e)[:200])
            ev = Evidence(
                step_id=step.step_id,
                tool=step.tool,
                success=False,
                error=str(e)[:500],
            )

        ev.duration_ms = (time.time() - start) * 1000
        return ev

    # ──────────────────────────────────────────────────────────────────────
    # SQL Tool
    # ──────────────────────────────────────────────────────────────────────

    def _exec_sql(self, step: PlanStep, message: str, prior: Dict[int, Evidence], context: str) -> Evidence:
        """Generate and execute SQL via LLM code generation."""
        if not self._db or self._db.source != "fabric":
            return Evidence(step_id=step.step_id, tool="sql", success=False,
                           error="Fabric connection unavailable")

        code = self._generate_sql_code(step, message, prior, context)
        if not code:
            return Evidence(step_id=step.step_id, tool="sql", success=False,
                           error="Failed to generate SQL code")

        result = self._execute_sandbox(code)

        if result.get("error"):
            # Retry with error context
            for retry in range(MAX_CODE_RETRIES):
                fix_prompt = f"Le code précédent a échoué avec: {result['error']}\n\nCorrige le code."
                code = self._generate_sql_code(step, message, prior, context, error_context=fix_prompt)
                if code:
                    result = self._execute_sandbox(code)
                    if not result.get("error"):
                        break

        if result.get("error"):
            return Evidence(step_id=step.step_id, tool="sql", success=False,
                           error=result["error"][:500])

        if _should_retry_location_query(message, code, result):
            fix_prompt = (
                "La requete a retourne vide pour une ville/province marocaine. "
                "Pour la table hebergement, utilise province_name pour les villes/provinces "
                "(Casablanca, Marrakech, Tanger, etc.) et reserve region_name aux regions "
                "administratives comme Casablanca-Settat. Relance une requete agregee par annee "
                "sur les annees disponibles, puis imprime une note de couverture si certaines "
                "annees demandees sont absentes."
            )
            retry_code = self._generate_sql_code(step, message, prior, context, error_context=fix_prompt)
            if retry_code:
                retry_result = self._execute_sandbox(retry_code)
                if not retry_result.get("error") and not _result_looks_empty(retry_result):
                    logger.info("SQL location retry recovered non-empty results")
                    result = retry_result

        return Evidence(
            step_id=step.step_id,
            tool="sql",
            success=True,
            data=result.get("data"),
            text_summary=result.get("output", ""),
            chart_paths=result.get("chart_paths", []),
        )

    def _generate_sql_code(
        self, step: PlanStep, message: str, prior: Dict[int, Evidence],
        context: str, error_context: str = ""
    ) -> Optional[str]:
        """Use LLM to generate Python+SQL code for the analytics query."""
        client = get_shared_client()

        user_parts = [f"Question utilisateur: {message}"]
        if step.description:
            user_parts.append(f"Objectif de cette étape: {step.description}")
        if context:
            user_parts.append(f"Contexte conversation: {context[:300]}")

        # Include prior evidence as context
        for sid, ev in prior.items():
            if ev.success and ev.text_summary:
                user_parts.append(f"Résultat étape {sid}: {ev.text_summary[:200]}")

        if error_context:
            user_parts.append(error_context)

        messages = [
            {"role": "system", "content": _SQL_GEN_PROMPT},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

        kwargs: Dict[str, Any] = {
            "model": AZURE_OPENAI_DEPLOYMENT,
            "messages": _prepare_messages(messages),
            "max_completion_tokens": ANALYTICS_MAX_COMPLETION_TOKENS,
        }
        if MODEL_IS_REASONING and not _should_skip_reasoning_effort():
            kwargs["reasoning_effort"] = ANALYTICS_REASONING_EFFORT

        try:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip() if response.choices else ""
            # Strip markdown code fences if present
            content = re.sub(r"^```(?:python)?\s*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content)
            return content if content else None
        except Exception as e:
            logger.error("SQL code generation failed: %s", str(e)[:200])
            return None

    def _execute_sandbox(self, code: str) -> Dict[str, Any]:
        """Execute Python code in a sandboxed environment."""
        import io
        import contextlib
        import pandas as pd
        import numpy as np

        MONTH_NAMES_FR = {
            1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
            5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
            9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
        }

        chart_paths: List[str] = []

        def _sandbox_sql(query: str) -> pd.DataFrame:
            return self._db.safe_query(query)

        def _save_chart(fig, label: Optional[str] = None) -> Optional[str]:
            self._chart_count += 1
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            index = len(chart_paths) + 1
            safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label or f"chart_{index}")).strip("_")[:36]
            suffix = f"_{safe_label}" if safe_label else ""
            path = os.path.join(CHARTS_DIR, f"chart_{ts}_{self._chart_count}_{index}{suffix}.html").replace("\\", "/")
            os.makedirs(CHARTS_DIR, exist_ok=True)
            fig.write_html(path, include_plotlyjs="cdn", full_html=True,
                          config={"responsive": True, "displaylogo": False})
            chart_paths.append(path)
            return path

        def _to_md(df: pd.DataFrame) -> str:
            if df is None or df.empty:
                return "(Aucune donnée)"
            try:
                return df.to_markdown(index=False)
            except ImportError:
                return df.to_string(index=False)

        exec_globals: Dict[str, Any] = {
            "sql": _sandbox_sql,
            "pd": pd,
            "np": np,
            "MONTH_NAMES_FR": MONTH_NAMES_FR,
            "months_fr": MONTH_NAMES_FR,
            "save_chart": _save_chart,
            "to_md": _to_md,
        }

        try:
            import plotly.express as px
            import plotly.graph_objects as go
            exec_globals["px"] = px
            exec_globals["go"] = go
        except ImportError:
            pass

        output_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(code, exec_globals)  # noqa: S102

            output_text = output_buffer.getvalue()
            dataframes = [
                (name, value)
                for name, value in exec_globals.items()
                if isinstance(value, pd.DataFrame) and not value.empty
            ]
            if not output_text.strip() and dataframes:
                sections = []
                for name, df in dataframes[:5]:
                    sections.append(
                        f"[DataFrame {name}: {len(df)} lignes x {len(df.columns)} colonnes]\n"
                        f"{_to_md(df.head(50))}"
                    )
                output_text = "\n\n".join(sections)
            return {
                "output": output_text,
                "chart_paths": chart_paths,
                "error": None,
                "data": dataframes[0][1] if dataframes else None,
            }
        except Exception as e:
            return {
                "output": output_buffer.getvalue(),
                "chart_paths": chart_paths,
                "error": f"{type(e).__name__}: {str(e)}",
                "data": None,
            }

    # ──────────────────────────────────────────────────────────────────────
    # Web Search Tool
    # ──────────────────────────────────────────────────────────────────────

    def _exec_web_search(self, step: PlanStep, message: str, prior: Dict[int, Evidence]) -> Evidence:
        """Execute web search."""
        if not self._search:
            return Evidence(step_id=step.step_id, tool="web_search", success=False,
                           error="Search tool not available")

        query = step.parameters.get("query", message)

        try:
            results = self._search.smart_search(query, analysis_type="factual", max_results=4)
            if not results:
                results = self._search.smart_search(message[:100], analysis_type="news", max_results=3)

            if results:
                formatted_parts = []
                for r in results[:4]:
                    title = r.get("title", "")
                    snippet = r.get("content", r.get("snippet", ""))[:300]
                    url = r.get("url", "")
                    formatted_parts.append(f"• {title}\n  {snippet}\n  Source: {url}")

                text_summary = "\n\n".join(formatted_parts)
                return Evidence(
                    step_id=step.step_id, tool="web_search", success=True,
                    data=results, text_summary=text_summary,
                    metadata={"sources": [{"title": r.get("title"), "url": r.get("url")} for r in results[:4]]},
                )
            else:
                return Evidence(step_id=step.step_id, tool="web_search", success=True,
                               text_summary="Aucun résultat pertinent trouvé.")

        except Exception as e:
            return Evidence(step_id=step.step_id, tool="web_search", success=False,
                           error=str(e)[:300])

    # ──────────────────────────────────────────────────────────────────────
    # RAG Tool
    # ──────────────────────────────────────────────────────────────────────

    def _exec_rag(self, step: PlanStep, message: str) -> Evidence:
        """Execute RAG search on internal knowledge base."""
        if not self._rag:
            return Evidence(step_id=step.step_id, tool="rag", success=False,
                           error="RAG manager not available")

        query = step.parameters.get("query", message)

        try:
            results = self._rag.search_formatted(query, n_results=3)
            if results:
                return Evidence(step_id=step.step_id, tool="rag", success=True,
                               data=results, text_summary=results)
            else:
                return Evidence(step_id=step.step_id, tool="rag", success=True,
                               text_summary="Aucun document pertinent dans la base de connaissances.")
        except Exception as e:
            return Evidence(step_id=step.step_id, tool="rag", success=False,
                           error=str(e)[:300])

    # ──────────────────────────────────────────────────────────────────────
    # Prediction Tool
    # ──────────────────────────────────────────────────────────────────────

    def _exec_prediction(self, step: PlanStep, message: str, prior: Dict[int, Evidence]) -> Evidence:
        """Execute prediction using PredictionEngine."""
        if not self._prediction_engine:
            return Evidence(step_id=step.step_id, tool="prediction", success=False,
                           error="Prediction engine not available")

        try:
            result = self._prediction_engine.chat(message)
            if isinstance(result, dict):
                response_text = result.get("response", "")
                chart_path = result.get("chart_path")
                chart_paths = [chart_path] if chart_path else []
                return Evidence(
                    step_id=step.step_id, tool="prediction", success=True,
                    data=result, text_summary=response_text,
                    chart_paths=chart_paths,
                )
            else:
                return Evidence(step_id=step.step_id, tool="prediction", success=True,
                               text_summary=str(result))
        except Exception as e:
            return Evidence(step_id=step.step_id, tool="prediction", success=False,
                           error=str(e)[:300])

    # ──────────────────────────────────────────────────────────────────────
    # Chart Tool (depends on prior SQL data)
    # ──────────────────────────────────────────────────────────────────────

    def _exec_chart(self, step: PlanStep, message: str, prior: Dict[int, Evidence], context: str) -> Evidence:
        """Generate chart from prior SQL/prediction evidence."""
        # Collect data context from dependencies
        data_context_parts = []
        for dep_id in step.depends_on:
            if dep_id in prior and prior[dep_id].success:
                data_context_parts.append(prior[dep_id].text_summary[:500])

        if not data_context_parts:
            # If no explicit dependencies, use all prior SQL results
            for ev in prior.values():
                if ev.tool == "sql" and ev.success and ev.text_summary:
                    data_context_parts.append(ev.text_summary[:500])

        if not data_context_parts:
            return Evidence(step_id=step.step_id, tool="chart", success=False,
                           error="No data available to chart")

        # Generate chart code via LLM
        chart_step = PlanStep(
            step_id=step.step_id,
            tool="sql",
            description=f"Génère un graphique pour: {step.description or message}. "
                       f"Données disponibles:\n{chr(10).join(data_context_parts[:2])}",
            parameters=step.parameters,
        )
        return self._exec_sql(chart_step, message, prior, context)
