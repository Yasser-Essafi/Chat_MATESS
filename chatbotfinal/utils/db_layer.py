"""
STATOUR Database Layer — Microsoft Fabric Lakehouse Gold (Fabric-only)
======================================================================
Connects to the Fabric SQL Analytics Endpoint with an Azure AD Service
Principal. The Excel fallback was removed — the chatbot now points
exclusively to the Lakehouse Gold.

Required env vars:
    FABRIC_SQL_ENDPOINT    e.g. xxx.datawarehouse.fabric.microsoft.com
    FABRIC_DATABASE        e.g. LH_03_MTAESS_GOLD
    FABRIC_SCHEMA          e.g. dbo_GOLD  (default: dbo)
    AZURE_TENANT_ID
    AZURE_CLIENT_ID
    AZURE_CLIENT_SECRET

Usage:
    from utils.db_layer import DBLayer

    db = DBLayer()
    df = db.query_df("SELECT TOP 10 * FROM [dbo_GOLD].[fact_statistiques_apf]")
    print(db.source)   # "fabric" or "unavailable"
    print(db.status)
"""

import os
import urllib.parse
from typing import Optional, Dict, Any, List

import pandas as pd

from utils.logger import get_logger

logger = get_logger("statour.db_layer")

# Try ODBC Driver 18 first (newer, preferred), fall back to 17 (older Windows
# installs). Either is supported by Fabric SQL Analytics Endpoint.
_FABRIC_ODBC_DRIVERS = [
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
]

# Fabric scope for AAD token
_FABRIC_AAD_SCOPE = "https://database.windows.net/.default"

# Default table name (used when callers don't specify one)
_DEFAULT_TABLE = "fact_statistiques_apf"


class DBLayer:
    """
    Fabric Lakehouse Gold data access layer.

    Attributes:
        source (str): "fabric" if connected, "unavailable" otherwise
        status (str): Human-readable connection status
        schema (str): Default schema for table references (dbo / dbo_GOLD / ...)
    """

    def __init__(self):
        self.engine = None
        self.source = "unavailable"
        self.status = "Fabric non configuré"
        self.schema = os.getenv("FABRIC_SCHEMA", "dbo").strip() or "dbo"
        self._driver_used: Optional[str] = None

        self._try_connect_fabric()

    # ──────────────────────────────────────────────────────────────────────
    # Fabric connection
    # ──────────────────────────────────────────────────────────────────────

    def _pick_odbc_driver(self) -> Optional[str]:
        """Return the first installed ODBC driver from our preference list."""
        try:
            import pyodbc
        except ImportError:
            return None
        installed = set(pyodbc.drivers())
        for d in _FABRIC_ODBC_DRIVERS:
            if d in installed:
                return d
        return None

    def _try_connect_fabric(self) -> None:
        """Attempt to connect to Microsoft Fabric Lakehouse Gold."""
        fabric_server = os.getenv("FABRIC_SQL_ENDPOINT", "").strip()
        fabric_db = os.getenv("FABRIC_DATABASE", "").strip()

        if not fabric_server or not fabric_db:
            logger.info(
                "DBLayer: Fabric not configured "
                "(FABRIC_SQL_ENDPOINT / FABRIC_DATABASE missing)"
            )
            self.status = "Fabric non configuré"
            return

        driver = self._pick_odbc_driver()
        if not driver:
            self.status = (
                "ODBC Driver 17/18 for SQL Server non installé. "
                "Windows: https://aka.ms/odbc18 — Linux: msodbcsql18"
            )
            logger.error("DBLayer: %s", self.status)
            return

        token = self._get_aad_token()
        if not token:
            self.status = "Token AAD Fabric indisponible (vérifier AZURE_* env vars)"
            logger.warning("DBLayer: %s", self.status)
            return

        try:
            from sqlalchemy import create_engine, event, text
            import struct

            SQL_COPT_SS_ACCESS_TOKEN = 1256

            conn_str = (
                f"Driver={{{driver}}};"
                f"Server={fabric_server},1433;"
                f"Database={fabric_db};"
                "Encrypt=yes;"
                "TrustServerCertificate=no;"
                "Connection Timeout=30;"
            )
            encoded = urllib.parse.quote_plus(conn_str)
            engine = create_engine(
                f"mssql+pyodbc:///?odbc_connect={encoded}",
                pool_pre_ping=True,
            )

            # Refresh the AAD token on every new physical connection so that
            # tokens expiring after ~1 hour never cause error 18456.
            @event.listens_for(engine, "do_connect")
            def provide_token(dialect, conn_rec, cargs, cparams):
                fresh_token = self._get_aad_token()
                if not fresh_token:
                    raise RuntimeError(
                        "DBLayer: Could not refresh AAD token for Fabric connection"
                    )
                token_bytes = fresh_token.encode("utf-16-le")
                token_struct = struct.pack(
                    f"=I{len(token_bytes)}s", len(token_bytes), token_bytes
                )
                cparams["attrs_before"] = {SQL_COPT_SS_ACCESS_TOKEN: token_struct}

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            self.engine = engine
            self.source = "fabric"
            self._driver_used = driver
            self.status = (
                f"Fabric Lakehouse Gold ({fabric_db}@"
                f"{fabric_server.split('.')[0]}, schema={self.schema}, "
                f"driver={driver})"
            )
            logger.info("DBLayer: Connected — %s", self.status)

        except ImportError as e:
            self.status = f"Dépendance manquante: {e} (pip install sqlalchemy pyodbc azure-identity)"
            logger.error("DBLayer: %s", self.status)
        except Exception as e:
            self.status = f"Fabric connexion échouée: {str(e)[:200]}"
            logger.error("DBLayer: %s", self.status)

    def _get_aad_token(self) -> Optional[str]:
        """
        Obtain an Azure AD access token for Fabric SQL via Service Principal.

        Reads: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
        """
        tenant_id = os.getenv("AZURE_TENANT_ID", "").strip()
        client_id = os.getenv("AZURE_CLIENT_ID", "").strip()
        client_secret = os.getenv("AZURE_CLIENT_SECRET", "").strip()

        if not all([tenant_id, client_id, client_secret]):
            logger.debug(
                "DBLayer: Missing AAD env vars (AZURE_TENANT_ID / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET)"
            )
            return None

        try:
            from azure.identity import ClientSecretCredential  # type: ignore

            cred = ClientSecretCredential(tenant_id, client_id, client_secret)
            token_obj = cred.get_token(_FABRIC_AAD_SCOPE)
            logger.debug("DBLayer: AAD token acquired (expires_on=%s)", token_obj.expires_on)
            return token_obj.token

        except ImportError:
            logger.warning(
                "DBLayer: azure-identity not installed — "
                "install with: pip install azure-identity"
            )
        except Exception as e:
            logger.warning("DBLayer: AAD token error: %s", e)

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def _qualify(self, table: str) -> str:
        """Return the fully-qualified, bracketed table name with schema prefix.

        Skips qualification if the caller already passed a `[schema].[table]`
        or `schema.table` form.
        """
        if "." in table:
            return table  # caller already qualified it
        return f"[{self.schema}].[{table}]"

    # Forbidden SQL keywords for the user-facing safe_query() helper.
    # Anything that mutates data, alters schema, or invokes DBA-level commands
    # is rejected before the query reaches Fabric.
    _SQL_FORBIDDEN = (
        r"\b(insert|update|delete|drop|alter|truncate|create|merge|grant|"
        r"revoke|exec|execute|sp_executesql|xp_cmdshell|backup|restore|"
        r"shutdown|kill|bulk|openrowset|opendatasource|use|declare|set)\b"
    )

    # Hard cap on rows returned to the analytics sandbox — prevents the LLM
    # from accidentally pulling millions of rows into pandas.
    SAFE_QUERY_ROW_LIMIT = 100_000

    def query_df(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute T-SQL and return a DataFrame. Fabric-only.

        Raises:
            RuntimeError: If Fabric is not connected.
        """
        if self.engine is None:
            raise RuntimeError(
                f"DBLayer: Fabric Lakehouse not connected — {self.status}"
            )
        return self._fabric_query(sql, params)

    def safe_query(self, sql_text: str) -> pd.DataFrame:
        """Execute a read-only T-SQL query from the analytics sandbox.

        Hardening:
        - Reject any statement containing write/DDL keywords (case-insensitive).
        - Reject queries that include a semicolon followed by another statement
          (a basic guard against query stacking).
        - Cap returned rows at SAFE_QUERY_ROW_LIMIT (post-execution slice).
        """
        import re as _re
        if not sql_text or not sql_text.strip():
            raise ValueError("Empty SQL query")

        s = sql_text.strip().rstrip(";")
        # Only one explicit read statement is allowed from the LLM sandbox.
        # This blocks SELECT INTO, comment-smuggling, session changes and
        # anything that is not clearly a SELECT / CTE query.
        if not _re.match(r"^\s*(select|with)\b", s, _re.IGNORECASE):
            raise PermissionError("Only SELECT/CTE read queries are allowed.")
        if "--" in s or "/*" in s or "*/" in s:
            raise PermissionError("SQL comments are not allowed in sandbox queries.")
        if _re.search(r"\bselect\b[\s\S]{0,2000}\binto\b", s, _re.IGNORECASE):
            raise PermissionError("SELECT INTO is forbidden in read-only mode.")
        if _re.search(self._SQL_FORBIDDEN, s, _re.IGNORECASE):
            raise PermissionError(
                "Read-only SQL only — INSERT/UPDATE/DELETE/DDL forbidden."
            )
        # Block stacked queries (e.g. "SELECT ...; DROP TABLE ...")
        if ";" in s:
            raise PermissionError("Multi-statement queries are not allowed.")

        # Guard against accidental raw fact-table scans. The post-execution
        # row cap below protects Python memory, but it cannot save Fabric from
        # doing the full scan first. Analytical queries should aggregate in SQL
        # or explicitly use TOP for inspection.
        if (
            _re.search(r"\bselect\s+\*", s, _re.IGNORECASE)
            and not _re.search(r"\btop\s*(?:\(|\d)", s, _re.IGNORECASE)
            and not _re.search(r"\bcount\s*\(", s, _re.IGNORECASE)
        ):
            raise PermissionError(
                "Raw SELECT * without TOP is not allowed. Aggregate in SQL or use SELECT TOP N."
            )

        df = self.query_df(s)
        if len(df) > self.SAFE_QUERY_ROW_LIMIT:
            logger.warning(
                "safe_query truncated %d rows → %d (hint: add aggregation/TOP)",
                len(df), self.SAFE_QUERY_ROW_LIMIT,
            )
            df = df.head(self.SAFE_QUERY_ROW_LIMIT).copy()
        return df

    def get_df(self, table: Optional[str] = None,
               date_column: Optional[str] = None) -> pd.DataFrame:
        """Return the full contents of `table` (default: APF fact table).

        When `date_column` is provided, the load is split year-by-year to avoid
        the TCP timeout that the SQL Analytics Endpoint imposes on long-running
        SELECT * scans (~17 min observed on a 7M-row table). For each year we
        issue a separate, smaller query and concatenate the results.
        """
        tbl = self._qualify(table or _DEFAULT_TABLE)

        if not date_column:
            return self.query_df(f"SELECT * FROM {tbl}")

        # Discover available years on this column, then load one chunk per year
        years_df = self.query_df(
            f"SELECT DISTINCT YEAR([{date_column}]) AS yr FROM {tbl} "
            f"WHERE [{date_column}] IS NOT NULL ORDER BY yr"
        )
        years = [int(y) for y in years_df["yr"].tolist() if pd.notna(y)]
        if not years:
            return self.query_df(f"SELECT * FROM {tbl}")

        chunks: List[pd.DataFrame] = []
        for yr in years:
            chunk = self.query_df(
                f"SELECT * FROM {tbl} WHERE YEAR([{date_column}]) = {yr}"
            )
            if not chunk.empty:
                chunks.append(chunk)
                logger.info("  [chunk] %s year %d: %s rows",
                            tbl, yr, f"{len(chunk):,}")
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)

    def list_tables(self) -> List[str]:
        """List all tables in the Fabric default schema (or all schemas if
        FABRIC_SCHEMA was not specified)."""
        if self.engine is None:
            return []
        try:
            sql = (
                "SELECT TABLE_SCHEMA, TABLE_NAME "
                "FROM INFORMATION_SCHEMA.TABLES "
                "WHERE TABLE_TYPE = 'BASE TABLE' "
                "ORDER BY TABLE_SCHEMA, TABLE_NAME"
            )
            df = self.query_df(sql)
            return [f"{r['TABLE_SCHEMA']}.{r['TABLE_NAME']}" for _, r in df.iterrows()]
        except Exception as e:
            logger.warning("DBLayer.list_tables failed: %s", e)
            return []

    def get_schema(self) -> Dict[str, Any]:
        """Return schema information from Fabric."""
        if self.engine is None:
            return {"source": "unavailable", "tables": [], "status": self.status}
        return self._fabric_schema()

    def ping(self) -> bool:
        """Return True if the Fabric connection is alive."""
        if self.engine is None:
            return False
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def reconnect(self) -> bool:
        """Re-attempt Fabric connection. Returns True if now connected."""
        self.engine = None
        self.source = "unavailable"
        self.status = "Reconnexion Fabric en cours…"
        self._try_connect_fabric()
        return self.source == "fabric"

    # ──────────────────────────────────────────────────────────────────────
    # Fabric helpers
    # ──────────────────────────────────────────────────────────────────────

    def _fabric_query(self, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Run SQL against the Fabric endpoint.

        Uses an explicit connection context manager so SQLAlchemy always
        rolls back and returns the connection to the pool cleanly — prevents
        PendingRollbackError on subsequent queries when a prior query fails.
        """
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(sql, conn, params=params)
        except Exception as e:
            logger.error("DBLayer: Fabric query failed: %s\nSQL: %s", e, sql[:200])
            raise

    def _fabric_schema(self) -> Dict[str, Any]:
        """Fetch table schemas from Fabric via INFORMATION_SCHEMA, filtered
        on the configured FABRIC_SCHEMA."""
        try:
            with self.engine.connect() as conn:
                cols_df = pd.read_sql(
                    """
                    SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = ?
                    ORDER BY TABLE_NAME, ORDINAL_POSITION
                    """,
                    conn,
                    params=(self.schema,),
                )
            tables: Dict[str, List[Dict[str, str]]] = {}
            for _, row in cols_df.iterrows():
                tbl = row["TABLE_NAME"]
                tables.setdefault(tbl, []).append(
                    {"name": row["COLUMN_NAME"], "type": row["DATA_TYPE"]}
                )

            return {
                "source": "fabric",
                "schema": self.schema,
                "tables": [{"name": t, "columns": c} for t, c in tables.items()],
            }
        except Exception as e:
            logger.warning("DBLayer: Schema fetch failed: %s", e)
            return {"source": "fabric", "schema": self.schema,
                    "tables": [], "error": str(e)}

    # ──────────────────────────────────────────────────────────────────────
    # Representation
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"<DBLayer source={self.source!r} status={self.status!r}>"
