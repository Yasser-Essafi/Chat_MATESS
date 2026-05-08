# CHATBOT_TOURISME — MTAESS Tourism Analytics Chatbot

## What this project is

Flask REST API + custom HTML/JS frontend for a multi-agent tourism analytics chatbot.
**Client**: Ministry of Tourism, Handicraft and Social Economy of Morocco (MTAESS / STATOUR).
**Stack**: Python 3.11, Flask, Azure OpenAI (`gpt-5-mini`, reasoning model), pandas, Plotly,
ChromaDB (RAG), Tavily + Exa.ai + Brave (web search), Microsoft Fabric Lakehouse Gold
(SQL Analytics Endpoint via pyodbc + SQLAlchemy T-SQL).

---

## Project layout

```
CHATBOT_TOURISME/
├── server.py                         # Flask entry point — port 5000, threaded
├── tmp_test.py                       # Regression suite (run with server running)
└── chatbotfinal/
    ├── .env                          # API keys — NEVER read, modify, or display
    ├── config/settings.py            # All constants, prompts, env vars
    ├── agents/
    │   ├── orchestrator.py           # 3-layer routing: instant rules → LLM → fallback
    │   ├── data_analytics_agent.py   # T-SQL generation → sandboxed exec on Fabric
    │   ├── researcher_agent.py       # Agentic web search (tool calling) + RAG
    │   ├── normal_agent.py           # Greetings / platform Q&A
    │   └── prediction_agent.py       # Rule-based forecasting + web context
    ├── utils/
    │   ├── base_agent.py             # Shared Azure OpenAI client + history management
    │   ├── db_layer.py               # Fabric SQL Analytics Endpoint ONLY (no Excel)
    │   ├── kpi_cache.py              # Pre-computed KPIs (APF data, month-aware)
    │   ├── intent_extractor.py       # Dynamic intent + targeted search query builder
    │   └── cache.py                  # Search result TTL cache (shared across agents)
    ├── tools/
    │   ├── rag_tools.py              # ChromaDB vector store (84+ chunks)
    │   └── search_tools.py           # Tavily + Brave + Exa.ai (smart_search by type)
    ├── knowledge_base/documents/     # RAG source docs (markdown, auto-indexed)
    └── charts/                       # Generated Plotly HTML charts (runtime, pruned)
```

---

## Run commands

```bash
python server.py                                   # Start server: http://localhost:5000
python tmp_test.py                                 # Regression suite (server must be running)
pip install -r chatbotfinal/requirements.txt       # Install dependencies
python chatbotfinal/rebuild_knowledge_base.py      # Rebuild RAG vectorstore after doc changes
```

---

## Azure OpenAI — gpt-5-mini CRITICAL facts

- **Model**: `gpt-5-mini` (reasoning model) accessed via Azure Chat Completions API
- **NEVER** pass `temperature` — reasoning models reject it (auto-handled in `base_agent.py`)
- **NEVER** use `max_tokens` — use `max_completion_tokens` instead
- Use `reasoning_effort` (`"minimal"/"low"/"medium"/"high"`) to control latency vs quality
- **Function/tool calling IS supported** — use it for web search (agentic pattern)
- `base_agent.py` handles all model compatibility issues automatically (reasoning_effort rejection, system→developer role conversion)
- Responses API also available at `/openai/v1/` endpoint — currently using Chat Completions
- **Bing Grounding** (Azure AI Agents): NOT available for gpt-5 series — use Tavily/Exa instead

---

## Fabric Tables — BOTH fact tables are critical

### Schema: `[dbo_GOLD]` on workspace `WS_01_MTAESS_MVP_DEV`

### Table 1: `[dbo_GOLD].[fact_statistiques_apf]` (~101,519 rows)

| Column | Type | Description |
|--------|------|-------------|
| `statistiques_apf_id` | UUID | Primary key |
| `nationalite` | String | **Pays de résidence** (NOT ethnic nationality — see CRITICAL RULES) |
| `poste_frontiere` | String | Border post (45 posts, prefix: A=Airport, P=Port, T=Terrestrial) |
| `region` | String | Moroccan region (12 regions) |
| `continent` | String | Continent of origin (14 values) |
| `voie` | String | Entry mode: V Aérienne / V Maritime / V Terrestre |
| `date_stat` | Date | 1st of month (monthly granularity, NO daily data) |
| `mre` | Integer | Marocains Résidant à l'Étranger (diaspora returning) |
| `tes` | Float | Touristes Étrangers Séjournistes (foreign tourists) |

**Derived metrics**: `total = mre + tes`
**Coverage**: 2019 + 2023–2026 (Jan–Feb 2026 latest as of build)

### Table 2: `[dbo_GOLD].[fact_statistiqueshebergementnationaliteestimees]` (~7.3M rows)

| Column | Type | Description |
|--------|------|-------------|
| `eht_id` | String | Establishment ID |
| `nationalite_name` | String | Pays de résidence of tourist |
| `categorie_name` | String | Hotel category (FK → gld_dim_categories_classements) |
| `region_name` | String | Region |
| `date_stat` | Date | 1st of month |
| `nuitees` | Integer | **Nights spent** (accommodation metric) |
| `arrivees` | Integer | **Check-ins** at establishments (≠ APF border arrivals) |

**Key JOINs**:
- `gld_dim_categories_classements` ON `categorie_name` → exposes `type_eht_libelle` (Hôtels, Maisons d'hôtes, Campings...)
- `gld_dim_etablissements_hebergements` ON `CAST(etablissement_id_genere AS VARCHAR) = eht_id` → `etablissement_libelle_fr`, `capacite_nbr_chambre_actuel`, `delegation_id`
- `gld_dim_delegations` ON `delegation_bk = delegation_id` → `delegation_name`

**Coverage**: data from STDN (télédéclarations) + STATOUR (manual) + Estimatif model

---

## CRITICAL Business Rules — Read before ANY analytics or prompt work

### Rule 1: The "arrivées" Ambiguity — Most common source of errors

The word **"arrivées"** means TWO completely different things:

| Context | Table | SQL metric | Meaning |
|---------|-------|-----------|---------|
| **APF** (border) | `fact_statistiques_apf` | `mre + tes` | People crossing a border post |
| **Hébergement** (hotel) | `fact_statistiqueshebergementnationaliteestimees` | `arrivees` | Check-ins at classified hotels |

→ **ALWAYS clarify which** in responses. If ambiguous → provide BOTH metrics.

### Rule 2: SQL column naming traps

- `nationalite` in APF table = **pays de résidence** (country of residence of traveler) — NOT ethnic nationality
- `nationalite_name` in hébergement table = also pays de résidence
- **ALWAYS rename these columns in output**: `nationalite` → "Pays de résidence"
- `arrivees` (hébergement) ≠ `mre+tes` (APF) — completely different metrics, different tables

### Rule 3: Data granularity

- `date_stat` = ALWAYS 1st of month — monthly data only, NO daily breakdown
- Filter temporally with: `WHERE YEAR(date_stat) = 2025 AND MONTH(date_stat) = 7`
- Partial year 2026: only Jan–Feb available — NEVER compare full-year 2026 with earlier full years

### Rule 4: Key domain concepts

| Term | Definition |
|------|-----------|
| **MRE** | Marocains Résidant à l'Étranger — Moroccan diaspora returning home |
| **TES** | Touristes Étrangers Séjournistes — Foreign non-Moroccan tourists |
| **EHTC** | Établissements d'Hébergement Touristique Classés (~5000 classified hotels) |
| **STDN** | Système de Télé-déclaration des Nuitées — electronic hotel declarations |
| **STATOUR** | Manual entry platform used by regional delegations (backup if STDN insufficient) |
| **Estimatif** | Statistical model estimating nuitées for non-declaring establishments (based on provincial average occupancy rate × available rooms × days) |
| **TO** | Taux d'Occupation = occupied rooms / available rooms × 100 |
| **Opération Marhaba** | Annual June–September MRE return program (causes maritime arrival peaks) |
| **DMS** | Durée Moyenne de Séjour (average length of stay) |
| **Pays de résidence** | Country of residence of the traveler (stored as "nationalite" in SQL — naming mismatch in schema) |

---

## Architecture — Critical facts

### SQL-on-demand (NO bulk loading into Python memory)

- Analytics agent catalogs table metadata ONLY at startup (column names, types, row counts, sample values)
- LLM generates T-SQL → sandbox executes via `sql("SELECT ...")` against Fabric
- **Sandbox environment**: `sql()`, `pd`, `np`, `px` (plotly.express), `go` (plotly.graph_objects), `to_md(df)`, `MONTH_NAMES_FR`
- **NO `df` variable exists in sandbox** — use `sql()` every time
- ALWAYS prefix tables: `[dbo_GOLD].[table_name]`
- ALWAYS aggregate in SQL (GROUP BY, SUM, COUNT, TOP N) — NEVER pull millions of rows into pandas

### Web search architecture (target: native function calling)

The researcher_agent uses **native function calling** — gpt-5-mini decides when and what to search:

```python
# Tool schemas defined in search_tools.py:
SEARCH_TOOLS_SCHEMA = [
    {
        "type": "function", "name": "web_search",
        "description": "Factual/news search via Tavily. Use for: recent tourism news, official stats, current events Morocco.",
    },
    {
        "type": "function", "name": "semantic_search",
        "description": "Semantic/causal search via Exa.ai. Use for: 'why' questions, trend analysis, contextual factors.",
    }
]
# LLM calls tools autonomously, Python executes and returns results, LLM synthesizes
```

### IntentExtractor (utils/intent_extractor.py)

Pure Python module — NO LLM calls, NO external deps, executes in <10ms.
Extracts from user message: `period_month`, `period_year`, `metric_type`, `analysis_type`,
`geo_scope`, `detected_entities`, `external_factors_categories`.
Used by prediction_agent to build targeted search queries dynamically.

### KPI cache

- Built at startup from full APF SELECT → pandas DataFrame → `_apf_df` kept in memory
- `kpi_cache.try_answer()` returns fast pre-computed answers for common year/total questions
- Month-aware: returns `None` if a month is detected (routes to LLM for precision)
- Does NOT cover hébergement data (routes hébergement questions to LLM analytics)

### Agents and routing

```
user → Flask /api/chat → Orchestrator.route()
  ├── "analytics"   → DataAnalyticsAgent (Fabric T-SQL)
  ├── "researcher"  → ResearcherAgent (web search + RAG)
  ├── "prediction"  → PredictionAgent (rule-based + web context)
  └── "normal"      → NormalAgent (greetings / platform Q&A)
```

**Orchestrator 3-layer classification**: instant rules (0ms) → LLM (1–2s) → keyword fallback

### Performance settings (do not change without understanding impact)

| Setting | Value | Why |
|---------|-------|-----|
| `MAX_CODE_RETRIES` | `2` | SQL almost always fails once; retries with error context |
| `CLASSIFIER_MAX_TOKENS` | `150` | One-word output only |
| `reasoning_effort` classifier | `"minimal"` | Routing needs near-zero latency |
| `reasoning_effort` analytics | `"low"` | Code gen needs some reasoning, not max |
| Chart pruning | 20 most recent | Prevents unbounded charts/ folder growth |

---

## What Codex MUST NOT do — violations from past sessions

```
NEVER re-add streaming — permanently removed. /api/chat/stream must not exist.
NEVER use `df` in the analytics sandbox — use sql("SELECT ...") instead.
NEVER try to load Excel files — Fabric Lakehouse Gold is the only data source.
NEVER add `temperature` to Azure OpenAI calls — reasoning model rejects it.
NEVER set MAX_CODE_RETRIES = 0 — SQL fails on first try regularly, needs 2 retries.
NEVER use pd.read_sql(sql, engine) — always use: with engine.connect() as conn: pd.read_sql(sql, conn)
NEVER generate static/hardcoded external factor queries (8-query fixed pattern is wrong)
  → external factor queries must be DYNAMIC: generated by LLM via tool calling
NEVER answer "arrivées" questions without clarifying APF vs hébergement context
NEVER treat `nationalite` column as ethnic nationality — it's always pays de résidence
NEVER pull millions of rows from hébergement table (7.3M rows) — ALWAYS aggregate in SQL
NEVER modify knowledge_base/vectorstore/ (binary ChromaDB — rebuild_knowledge_base.py instead)
NEVER read, display, or log .env content
NEVER use `pool_pre_ping=False` on Fabric engine — always pool_pre_ping=True
NEVER use `max_tokens` parameter — use `max_completion_tokens` for reasoning models
NEVER compare full-year 2026 with full-year 2025 — 2026 has only Jan-Feb data
NEVER use the Responses API `web_search` tool for gpt-5-mini — not yet supported on Azure
NEVER use Azure AI Agents Bing Grounding with gpt-5-mini — explicitly excluded for gpt-5 series
```

---

## Required env vars

```bash
# Azure OpenAI (reasoning model — Chat Completions API)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://llm-mtaess.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini

# Microsoft Fabric Lakehouse Gold (SQL Analytics Endpoint)
FABRIC_SQL_ENDPOINT=4ql4saucbfcutnpn6ladpjqh5q-ycdaqpdqiubuxb5fdqh3gzxnuu.datawarehouse.fabric.microsoft.com
FABRIC_DATABASE=LH_03_MTAESS_GOLD
FABRIC_SCHEMA=dbo_GOLD
FABRIC_TABLES=fact_statistiques_apf,fact_statistiqueshebergementnationaliteestimees,gld_dim_categories_classements,gld_dim_etablissements_hebergements,gld_dim_delegations

# Azure Service Principal (Fabric auth)
AZURE_TENANT_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...

# Web Search tools
TAVILY_API_KEY=...        # factual/news search — primary
EXA_API_KEY=...           # semantic/causal research — for "pourquoi" questions
BRAVE_API_KEY=...         # volume fallback
SEARCH_BACKEND=auto       # auto | tavily | brave
```

---

## Regression test suite

- **File**: `tmp_test.py` at project root
- **Coverage**: analytics (A1–A13), hébergement (H1–H4), prediction (P1–P7),
  researcher (R1–R3), normal (N1–N4), causal (X1–X3), edge cases (E1–E3), follow-ups (F1–F2)
- **Run**: `python tmp_test.py` (server must be on port 5000)
- **Target**: All tests pass, no 0.0%/an in prediction, no APF metrics in hébergement answers

---

## Domain terminology

| Term | Full name | Notes |
|------|-----------|-------|
| APF | Arrivées aux Postes Frontières | Border arrivals from DGSN (police) |
| STATOUR | Plateforme de saisie statistiques tourisme | Manual hotel declaration platform |
| STDN | Système Télé-Déclaration Nuitées | Electronic hotel night declarations |
| MRE | Marocains Résidant à l'Étranger | Moroccan diaspora (~5M people) |
| TES | Touristes Étrangers Séjournistes | Foreign non-Moroccan tourists |
| EHTC | Établissements Hébergement Touristique Classés | ~5000 classified hotels |
| DGSN | Direction Générale Sûreté Nationale | Source of APF border data |
| ONMT | Office National Marocain du Tourisme | Tourism promotion body |
| MTAESS | Ministère Tourisme, Artisanat, Économie Sociale et Solidaire | The client |
| dbo_GOLD | Gold schema on Fabric | All production tables |
| sql() | Sandbox SQL function | Runs T-SQL against Fabric (NOT df, NOT read_sql) |
| TO | Taux d'Occupation | Occupancy rate metric |
| DMS | Durée Moyenne de Séjour | Average length of stay |
| Estimatif | Statistical estimation model | Fills gaps for non-declaring EHTC |
| Vision 2030 | Tourism target | 26M TES arrivals + 120Mrd MAD revenue |