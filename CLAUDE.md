# Chatbot Tourisme MTAESS — CLAUDE.md

## What this project is
Flask REST API + custom HTML/JS frontend for a multi-agent tourism chatbot.
Client: Morocco's Ministry of Tourism (MTAESS / STATOUR platform).
Stack: Python 3.11, Flask, Azure OpenAI (deployment `gpt-5-mini` reasoning model), pandas, Plotly, ChromaDB, Tavily/Brave Search, Microsoft Fabric Lakehouse Gold (SQL Analytics Endpoint).

## Project layout
```
CHATBOT_TOURISME/
├── server.py                   # Flask entry point — runs on port 5000
├── tmp_test.py                 # 34-test regression suite (run from this dir)
└── chatbotfinal/
    ├── .env                    # API keys (NEVER read or modify)
    ├── config/settings.py      # All constants, prompts, env vars
    ├── agents/
    │   ├── orchestrator.py     # Routes messages to the right agent (3-layer)
    │   ├── data_analytics_agent.py  # Catalogs Fabric tables → LLM generates T-SQL → sandboxed exec
    │   ├── researcher_agent.py      # Tavily/Brave web search + ChromaDB RAG
    │   ├── normal_agent.py          # Greetings / general questions
    │   └── prediction_agent.py      # Rule-based forecasting with external factors
    ├── utils/
    │   ├── base_agent.py       # Shared AzureOpenAI client + history trimming + reasoning_effort
    │   ├── db_layer.py         # Fabric SQL Analytics Endpoint ONLY (no Excel fallback)
    │   ├── kpi_cache.py        # Pre-computed KPI answers (APF data, month-aware)
    │   └── cache.py            # Search result cache (TTL-based)
    ├── tools/
    │   ├── rag_tools.py        # ChromaDB vector store (84 chunks)
    │   └── search_tools.py     # Tavily + Brave Search (auto-select via SEARCH_BACKEND env)
    ├── data/                   # Legacy dir — NOT used for data anymore (Fabric is the source)
    ├── frontend/index.html     # Single-page app (marked.js for markdown)
    ├── knowledge_base/documents/  # RAG source docs (markdown)
    └── charts/                 # Generated Plotly HTML charts (runtime, auto-pruned to 20)
```

## Run commands
```bash
# From CHATBOT_TOURISME/ directory
python server.py                  # Start server on http://localhost:5000
python tmp_test.py                # Run 34-test regression suite (server must be running)
pip install -r chatbotfinal/requirements.txt  # Install deps
```

## Architecture — critical facts

### Data source: Microsoft Fabric Lakehouse Gold
- **Connection**: SQL Analytics Endpoint via ODBC Driver 18, Service Principal auth (SQL_COPT_SS_ACCESS_TOKEN struct)
- **Schema**: `dbo_GOLD` (env var `FABRIC_SCHEMA`)
- **5 catalogued tables** (metadata-only at startup — no bulk load):
  - `fact_statistiques_apf` — 9 cols, ~101,519 rows (APF border arrivals, years 2019–2026)
  - `fact_statistiqueshebergementnationaliteestimees` — 9 cols, ~7.3M rows (accommodation data)
  - `gld_dim_categories_classements` — 8 cols, ~61 rows
  - `gld_dim_etablissements_hebergements` — 20 cols, ~5,843 rows
  - `gld_dim_delegations` — 4 cols, ~26 rows
- **Startup time**: ~75 seconds (catalog + KPI cache build)
- **Excel fallback**: PERMANENTLY REMOVED — Fabric is the only data source

### SQL-on-demand analytics
- Analytics agent catalogs tables via INFORMATION_SCHEMA at startup (no bulk load)
- LLM generates T-SQL queries → sandbox executes via `sql("SELECT ...")` against Fabric
- The sandbox has NO `df` variable — only `sql()`, `pd`, `np`, `px`, `go`, `to_md()`, `MONTH_NAMES_FR`
- `safe_query()` in DBLayer guards against SQL injection (read-only, 100k row cap)
- `_fabric_query()` uses `with engine.connect() as conn:` + `pool_pre_ping=True` to prevent PendingRollbackError

### KPI cache
- Built at startup from full APF SELECT (~10s) → pandas DataFrame → `_apf_df` in memory
- Used by KPICache for instant answers to common questions (total, year, MRE/TES)
- Month-aware: `try_answer()` returns None when a month is detected (prevents annual-total bug)
- Exposed on `analytics_agent._apf_df` for PredictionAgent and orchestrator year-range discovery

### Data flow
```
user → Flask /api/chat → Orchestrator.route() → agent.chat() → response
```
- **Orchestrator**: 3-layer routing — Layer 1 instant rules (0ms) → Layer 2 LLM (1-2s) → Layer 3 keyword fallback
- **Analytics**: catalog metadata → LLM writes T-SQL → exec sandbox (sql() → Fabric) → stdout markdown
- **Charts**: Plotly HTML files saved to `chatbotfinal/charts/`, served via `/charts/<filename>`, auto-pruned to 20 files
- **No streaming** — permanently removed. Only `/api/chat` (POST, JSON response).

### Performance settings
- `reasoning_effort` per agent: classifier=minimal, normal=minimal, analytics=low, researcher=low, prediction=minimal
- `MAX_CODE_RETRIES = 0` — no retry loops; if SQL fails, return error directly
- `CLASSIFIER_MAX_TOKENS = 150`

### Security (OWASP hardening)
- Rate limiting: 20 requests/60s per IP (sliding window)
- CORS allowlist: `ALLOWED_ORIGINS` env var
- MAX_MESSAGE_CHARS: 4000 chars per message
- MAX_CONTENT_LENGTH: 64KB
- Security headers: X-Content-Type-Options, X-Frame-Options, Referrer-Policy, X-Request-ID
- Error handlers return opaque `{"error":"chat_failed","request_id":...}` (no stack traces)

### Search backends
- Tavily (default) + Brave Search (faster, set `SEARCH_BACKEND=brave` or `auto`)
- Parallel multi-query via ThreadPoolExecutor (`search_multi()`)

## What Claude MUST NOT do — violations from past sessions
- **NEVER re-add streaming.** Permanently removed. `/api/chat/stream` must not exist.
- **NEVER use `df` in the analytics sandbox** — use `sql("SELECT ...")` instead. There is no `df` variable.
- **NEVER try to load Excel files** — data is exclusively in Fabric Lakehouse Gold.
- **NEVER add `temperature` parameter** to Azure OpenAI calls — reasoning model, temperature forbidden.
- **NEVER modify `data/apf_data.xlsx`** or any file in `data/` (legacy, ignored).
- **NEVER modify `knowledge_base/vectorstore/`** (binary ChromaDB files).
- **NEVER read or display `.env`** content.
- **NEVER add new dependencies** without checking `requirements.txt` first.
- **NEVER use `pd.read_sql(sql, self.engine)`** directly — always use `with self.engine.connect() as conn: pd.read_sql(sql, conn)` to prevent PendingRollbackError.
- The `to_md()` function in the analytics sandbox is a pandas→markdown helper, not a file operation.

## Required env vars
```
# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://llm-mtaess.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini

# Microsoft Fabric
FABRIC_SQL_ENDPOINT=4ql4saucbfcutnpn6ladpjqh5q-ycdaqpdqiubuxb5fdqh3gzxnuu.datawarehouse.fabric.microsoft.com
FABRIC_DATABASE=LH_03_MTAESS_GOLD
FABRIC_SCHEMA=dbo_GOLD
FABRIC_TABLES=fact_statistiques_apf,fact_statistiqueshebergementnationaliteestimees,gld_dim_categories_classements,gld_dim_etablissements_hebergements,gld_dim_delegations

# Azure Service Principal (for Fabric auth)
AZURE_TENANT_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...

# Search (optional)
TAVILY_API_KEY=...
BRAVE_API_KEY=...
SEARCH_BACKEND=auto   # auto | tavily | brave
```

## Regression test suite
- File: `tmp_test.py` at project root
- 34 tests: analytics (A1-A13), prediction (P1-P6), researcher (R1-R3), normal (N1-N4), edge cases (E1-E3), follow-ups (F1a/b, F2a/b), historical (O1)
- Run: `python tmp_test.py` (server must be on port 5000)
- Last result: **34/34 pass**, median wall time 2.7s, max 25s (complex year comparison)

## Domain terminology
- **APF**: Arrivées aux Postes Frontières (border arrivals data)
- **STATOUR**: Ministry tourism data collection platform
- **STDN**: Système de Télé-déclaration des Nuitées (hotel night stay declarations)
- **MRE**: Marocains Résidant à l'Étranger (Moroccans abroad)
- **TES**: Touristes Étrangers de Séjour (foreign tourists)
- **EHTC**: Établissements d'Hébergement Touristique Classés (classified accommodation)
- **dbo_GOLD**: Fabric schema containing all production gold-layer tables
- **sql()**: The sandbox function that runs T-SQL against Fabric (LLM must use this, NOT df)
