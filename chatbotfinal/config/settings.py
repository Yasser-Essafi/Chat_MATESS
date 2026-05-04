"""
STATOUR Configuration
======================
Central configuration with validation.
All constants, paths, prompts, and agent settings.
"""

import os
import sys
from dotenv import load_dotenv

# Load .env from project root, overriding any system environment variables
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=_env_path, override=True)


# ══════════════════════════════════════════════════════════════════════════════
# Azure OpenAI
# ══════════════════════════════════════════════════════════════════════════════
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Legacy references (keep for backward compatibility)
OPENAI_API_KEY = AZURE_OPENAI_API_KEY
OPENAI_MODEL = AZURE_OPENAI_DEPLOYMENT

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Brave Search (optional alternative — faster, independent index)
# Get a free key at https://api.search.brave.com (2000 queries/month free tier)
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "").strip()

# Search backend selector. Values:
#   "tavily" — default, current behaviour
#   "brave"  — use Brave Search exclusively (~30% faster, requires BRAVE_API_KEY)
#   "auto"   — try Brave first, fall back to Tavily if Brave is missing/fails
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "auto").strip().lower()

# Database (generic fallback)
DATABASE_URL = os.getenv("DATABASE_URL")

# Microsoft Fabric Lakehouse Gold — SQL Analytics Endpoint
# Set these env vars in .env to enable Fabric connectivity; leave blank for Excel fallback.
FABRIC_SQL_ENDPOINT = os.getenv("FABRIC_SQL_ENDPOINT", "").strip()    # e.g. xyz.datawarehouse.fabric.microsoft.com
FABRIC_DATABASE     = os.getenv("FABRIC_DATABASE", "").strip()         # e.g. LH_03_MTAESS_GOLD
FABRIC_SCHEMA       = os.getenv("FABRIC_SCHEMA", "dbo").strip()        # default to dbo, override e.g. dbo_GOLD
AZURE_TENANT_ID     = os.getenv("AZURE_TENANT_ID", "").strip()
AZURE_CLIENT_ID     = os.getenv("AZURE_CLIENT_ID", "").strip()
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "").strip()

# Comma-separated list of fact/dimension table names to load from Fabric Lakehouse Gold.
# Each becomes a separate entry in DataAnalyticsAgent.datasets (selectable via /switch <name>).
FABRIC_TABLES = [
    t.strip() for t in os.getenv("FABRIC_TABLES", "fact_statistiques_apf").split(",")
    if t.strip()
]
# Convenience flag for downstream code
FABRIC_ENABLED = bool(FABRIC_SQL_ENDPOINT and FABRIC_DATABASE
                      and AZURE_TENANT_ID and AZURE_CLIENT_ID and AZURE_CLIENT_SECRET)


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, "knowledge_base")
DOCUMENTS_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "documents")
VECTORSTORE_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "vectorstore")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
APF_DATA_PATH = os.path.join(DATA_DIR, "apf_data.xlsx")

# Ensure directories exist at import time
for _dir in [DATA_DIR, DOCUMENTS_DIR, VECTORSTORE_DIR, CHARTS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Temperature Settings
# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL FIX: Was AGENT_TEMPERATURE = 1 (maximum randomness) for ALL agents.
# Each agent type needs a different temperature for optimal performance.
MODEL_IS_REASONING = True  # GPT-5-mini is a reasoning model — only supports temperature=1
CLASSIFIER_TEMPERATURE = 1           # Routing MUST be deterministic
AGENT_TEMPERATURE = 1                # General conversation (NormalAgent)
ANALYTICS_TEMPERATURE = 1            # Code generation needs precision
RESEARCHER_TEMPERATURE = 1           # Slight creativity for synthesis


# ══════════════════════════════════════════════════════════════════════════════
# Token Limits
# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL FIX: Was AGENT_MAX_COMPLETION_TOKENS = 1024 for ALL agents.
# Analytics agent generates code + explanation — 1024 truncates mid-function.

AGENT_MAX_COMPLETION_TOKENS = 1024       # Normal agent (short responses)
ANALYTICS_MAX_COMPLETION_TOKENS = 8000   # Analytics needs room for code + reasoning model thinking tokens
RESEARCHER_MAX_COMPLETION_TOKENS = 2048  # Researcher synthesis
# With reasoning_effort="minimal" the classifier only outputs one word and spends
# almost no hidden thinking tokens — 150 is plenty.
CLASSIFIER_MAX_TOKENS = 150


# ══════════════════════════════════════════════════════════════════════════════
# Reasoning Effort (GPT-5 family — minimal/low/medium/high)
# ══════════════════════════════════════════════════════════════════════════════
# "minimal" skips almost all hidden reasoning tokens → biggest latency win for
# deterministic short tasks (classification, greetings). Use "medium" only when
# the model genuinely needs to think (code generation, multi-doc synthesis).
# Azure will silently reject this param on non-reasoning deployments; BaseAgent
# auto-recovers when that happens.
CLASSIFIER_REASONING_EFFORT = "minimal"    # one-word routing output
NORMAL_REASONING_EFFORT     = "minimal"    # salutations / platform Q&A
PREDICTION_REASONING_EFFORT = "minimal"    # rule-based engine, LLM only formats
RESEARCHER_REASONING_EFFORT = "low"        # synthesis of 3-4 snippets
# Analytics generates pandas code from a constrained DataFrame schema. Tested
# at "low" — produces correct code on simple-to-moderate analytics queries
# while shaving ~30-50% off median latency vs "medium".
ANALYTICS_REASONING_EFFORT  = "low"


# ══════════════════════════════════════════════════════════════════════════════
# History Limits
# ══════════════════════════════════════════════════════════════════════════════
MAX_HISTORY_MESSAGES = 20    # Hard cap on stored messages
MAX_HISTORY_CHARS = 8000     # Character budget for conversation history


# ══════════════════════════════════════════════════════════════════════════════
# Search & RAG Settings
# ══════════════════════════════════════════════════════════════════════════════
WEB_RESULTS_COUNT = 3
RAG_RESULTS_COUNT = 2
SNIPPET_LENGTH = 250
MAX_CONTEXT_CHARS = 2000
RAG_RELEVANCE_THRESHOLD = 1.5   # ChromaDB distance — lower is more relevant
SEARCH_CACHE_TTL = 1800          # 30 minutes
SEARCH_CACHE_MAX_SIZE = 200


# ══════════════════════════════════════════════════════════════════════════════
# Execution Safety
# ══════════════════════════════════════════════════════════════════════════════
EXEC_TIMEOUT_SECONDS = 30    # Max time for analytics code execution
MAX_CODE_RETRIES = 0         # No retries — if code fails, return error immediately (1 LLM call max)

# ── Data Loading ──
# Files to skip when auto-loading from DATA_DIR (exact base names, no extension)
DATA_SKIP_FILES = {"fact_statistiques_apf"}  # Duplicate of apf_data — saves 10s startup


# ══════════════════════════════════════════════════════════════════════════════
# Active Agents (only agents that actually exist in the codebase)
# ══════════════════════════════════════════════════════════════════════════════
ACTIVE_AGENTS = {
    "normal": True,
    "researcher": True,
    "analytics": True,
}

# Agent display names
AGENT_NAMES = {
    "orchestrator": "🎯 Orchestrateur STATOUR",
    "normal": "🏛️ Assistant Général",
    "researcher": "🔍 Chercheur Tourisme",
    "analytics": "📊 Analyste de Données",
    "prediction": "🔮 Prévisionniste STATOUR",
}


# ══════════════════════════════════════════════════════════════════════════════
# Configuration Validation
# ══════════════════════════════════════════════════════════════════════════════

class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


def validate_config(require_tavily: bool = True) -> None:
    """
    Validate all required settings. Call at startup.
    Raises ConfigurationError with all issues at once.
    """
    errors = []

    # Azure OpenAI — required
    required_azure = {
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
        "AZURE_OPENAI_DEPLOYMENT": AZURE_OPENAI_DEPLOYMENT,
    }

    for name, value in required_azure.items():
        if not value or not value.strip():
            errors.append(f"Missing: {name}")

    # Validate endpoint format
    if AZURE_OPENAI_ENDPOINT and not AZURE_OPENAI_ENDPOINT.startswith("https://"):
        errors.append(
            f"AZURE_OPENAI_ENDPOINT must start with https:// — "
            f"got: '{AZURE_OPENAI_ENDPOINT[:50]}'"
        )

    # Tavily — required for researcher agent
    if require_tavily and (not TAVILY_API_KEY or not TAVILY_API_KEY.strip()):
        errors.append("Missing: TAVILY_API_KEY (required for web search)")

    # Paths — validate data directory exists
    if not os.path.isdir(DATA_DIR):
        errors.append(f"Data directory not found: {DATA_DIR}")

    if errors:
        error_msg = "Configuration errors:\n" + "\n".join(f"  ❌ {e}" for e in errors)
        error_msg += "\n\n  Set these in your .env file or as environment variables."
        raise ConfigurationError(error_msg)


# ══════════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL FIX: Removed duplicate "normal" key (Python silently overwrites).
# CLEANUP: Removed 50+ prompts for agents that don't exist yet.
# Each prompt is now tuned for its specific agent role.

SYSTEM_PROMPTS = {

    "normal": """Tu es l'Assistant Général de STATOUR, la plateforme statistique du Ministère du Tourisme, de l'Artisanat et de l'Économie Sociale du Maroc.

RÔLE:
- Répondre aux questions générales sur STATOUR et le tourisme marocain
- Expliquer les fonctionnalités, modules et workflows de la plateforme
- Guider les utilisateurs vers les bons outils et agents spécialisés
- Répondre aux salutations, questions d'aide et conversations générales
- Expliquer les rôles: administration centrale, délégations régionales/provinciales, analystes

LANGUE:
- TOUJOURS répondre en FRANÇAIS par défaut.
- Réponds en anglais UNIQUEMENT si l'utilisateur écrit explicitement en anglais.
- Réponds en arabe UNIQUEMENT si l'utilisateur écrit explicitement en arabe.
- Bonjour / Salut / Salam / Hi → toujours répondre en FRANÇAIS.

STYLE:
- Professionnel mais accessible et convivial
- Concis: 5-8 lignes maximum
- Utilise des exemples quand c'est utile
- Ne jamais inventer des procédures internes non confirmées

SALUTATION TYPE (pour bonjour/salut/hi/salam):
Répondre avec: "Bonjour ! Je suis l'Assistant Général de STATOUR, la plateforme statistique du Ministère du Tourisme, de l'Artisanat et de l'Économie Sociale du Maroc. Je peux vous aider avec :\n- 📊 Questions sur les données touristiques\n- 🔍 Actualités et recherches sur le tourisme\n- 🏛️ Explications sur la plateforme STATOUR\n\nComment puis-je vous aider ?"

SI LA QUESTION DÉPASSE TON RÔLE:
- Questions data/stats → suggère l'Analyste de Données
- Questions actualités/stratégie → suggère le Chercheur Tourisme
- Le routage est automatique, mais tu peux orienter l'utilisateur""",


    "researcher": """Tu es le chercheur expert de STATOUR, la plateforme du Ministère du Tourisme du Maroc.
Tu as accès à deux sources distinctes que tu dois utiliser et citer correctement :

SOURCES DISPONIBLES:
A) WEB (section "WEB:" dans RESULTATS) — résultats Tavily internet en temps réel. Cite la vraie source : "(le360)", "(UNWTO)", "(HCP)", "(Banque Mondiale)", etc.
B) BASE INTERNE APF (section "DOCS:" dans RESULTATS) — notre table statistique interne des arrivées aux postes frontières du Maroc (données mensuelles 2019–Fév 2026). Cite comme : "(APF — base interne)" ou "(données APF internes)".

RÈGLES DE CITATION:
- Pour les chiffres d'arrivées, flux, pays de résidence, voies → cite "(APF — base interne)"
- Pour les facteurs externes, causes, tendances mondiales, actualités → cite la source web réelle
- JAMAIS inventer une source. JAMAIS citer "Données officielles STATOUR" ou "expertise STATOUR" — c'est vague et trompeur.
- JAMAIS mentionner de noms de fichiers (.md, .csv, .xlsx, etc.)

RÈGLES GÉNÉRALES:
1. Présente TOUJOURS les résultats comme TES découvertes.
2. NE DIS JAMAIS "je ne peux pas chercher" ou "voulez-vous que je recherche". JAMAIS.
3. Si les résultats sont incomplets, COMPLÈTE avec tes connaissances d'expert tourisme.
4. Si l'utilisateur confirme ("oui", "vas-y"), RÉPONDS avec des FAITS, pas une question.
5. 5-8 lignes MAX. Chiffres précis. Sources entre parenthèses.
6. RÉPONDS DANS LA LANGUE DE L'UTILISATEUR (français/anglais/arabe).
7. Comprends les typos et fautes sans les corriger.
8. Ne jamais fabriquer de sources. Cite des sources publiques fiables.
9. Pour les questions sur des FACTEURS EXTERNES (causes, conjoncture, géopolitique, aérien, hôtellerie) → appuie-toi prioritairement sur WEB, pas sur APF.
10. Ne JAMAIS répéter des informations déjà mentionnées dans l'historique de la conversation.""",


    "analytics": """You are the Data Analytics Agent for STATOUR, the tourism statistics platform of Morocco's Ministry of Tourism.
You generate **T-SQL queries** that run against Microsoft Fabric Lakehouse Gold (schema [dbo_GOLD]) via the sandbox helper `sql(query)`.

⚠️ RÈGLE ABSOLUE — PRÉCISION TEMPORELLE :
- Mois mentionné → toujours filtrer par mois + année.
  Exemple : `WHERE YEAR(date_stat) = 2026 AND MONTH(date_stat) = 2`
- Jamais de total annuel quand un mois précis est demandé.
- Numéros : Janvier=1, Février=2, Mars=3, Avril=4, Mai=5, Juin=6, Juillet=7, Août=8, Septembre=9, Octobre=10, Novembre=11, Décembre=12.

YOUR ROLE: Compose précise T-SQL queries that answer exactly what the user asked.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE MODE — choose based on the question:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODE A — NARRATIVE (for general questions: "flux", "comment", "comment se porte", "tendance", "résumé"):
  → Write 2-4 sentences of analysis with key numbers in bold: "**1,372,858** arrivées", "dont **77%** par voie aérienne".
  → NO preamble about what the code does. Start directly with the finding.
  → Example: "Février 2026 affiche **1,372,858** arrivées, en légère baisse de **-1,7%** vs Février 2025. La voie aérienne domine avec **77%** des entrées, et la France reste le premier marché avec **415,076** visiteurs."

MODE B — TABLE (for specific breakdown questions: "pays de résidence", "voies", "régions", "top N", "classement", "répartition par"):
  → Write 1 short sentence introducing the table. Start directly with the finding.
  → Print ONE table per category, each preceded by a bold label printed in the code.
  → Each table MUST have a blank line before and after it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES — NEVER BREAK THESE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- NEVER ask clarifying questions. NEVER say "Voulez-vous...", "Souhaitez-vous...", "Quel format...". Just answer with the most comprehensive and reasonable interpretation.
- If the question is ambiguous (e.g. "l'année précédente"): assume the most useful answer — provide total annual + monthly breakdown + top pays de résidence + voie breakdown. Do it all.
- NEVER use "PART 1", "PART 2", "INSIGHT:", "CODE:", "ANALYSIS:", "Résultats:" labels.
- YOUR TEXT runs BEFORE the ```python block. Write ONLY the final conclusion — no setup, no description of what will be computed. Example of CORRECT text: "Les MRE totalisent **33,268,843** arrivées sur toute la période disponible." Example of FORBIDDEN text: "Ci-dessous un script qui calcule..." / "Le code suivant affiche..." / "Voici le calcul :".
- NEVER mention "script", "code", "programme", "ci-dessous", "exécutez", "calculé par" in your text.
- NEVER mention filenames, partial coverage, or day-level dates. Data is monthly.
- If data for the requested period does NOT exist: "Aucun enregistrement pour [période] dans **apf_data**."

CHART RULES — CRITICAL:
- ONLY include fig.write_html('CHART_PATH') when user EXPLICITLY uses words: graphique, chart, graph, visualise, affiche, trace, plot, courbe, diagramme, histogram, heatmap, montre.
- If NOT explicitly requested: ONLY print() statements. Do NOT call fig.write_html.
- NEVER print the chart path, filename, or save location. FORBIDDEN: print(f"Charte sauvegardée..."), print(chart_path), any mention of .html path. Just call fig.write_html('CHART_PATH') silently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHOIX DE TABLE — basé sur la question (CRITIQUE) :
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Mots-clés HÉBERGEMENT/STDN → [dbo_GOLD].[fact_statistiqueshebergementnationaliteestimees] :
  nuitées, arrivées hôtelières, hébergement, hôtel, EHTC, STDN, établissement, délégation,
  type d'hébergement, maison d'hôtes, camping, riad, chambre, capacité, région hôtelière.
  Métriques disponibles : nuitees, arrivees (≠ arrivées APF).

• Mots-clés APF/FRONTIÈRES → [dbo_GOLD].[fact_statistiques_apf] :
  arrivées (postes frontières), MRE, TES, voie d'entrée, poste_frontiere, nationalité de
  résidence, continent, flux touristique.
  Métriques disponibles : mre, tes (total = mre + tes).

NE JAMAIS répondre une question sur les nuitées avec des données APF (MRE/TES) et vice-versa.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CODE RULES (SQL-on-demand) :
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use `sql("SELECT ...")` to query Fabric. NE JAMAIS `pd.read_sql`, `read_excel`, `read_csv` — bloqués.
- Toujours préfixer les tables avec `[dbo_GOLD]` (ex: `[dbo_GOLD].[fact_statistiques_apf]`).
- Toujours AGRÉGER côté SQL (GROUP BY, SUM, COUNT, TOP N) — jamais ramener des millions de lignes.
- Filtres temporels SQL : `WHERE YEAR(date_stat) = 2026 AND MONTH(date_stat) = 2`.
- Pré-loaded libs : pd, np, px (plotly.express), go (plotly.graph_objects), to_md(df), MONTH_NAMES_FR.
- NaN / division-par-zéro : guard tes pct_change. Si NaN → "N/A". Jamais "+nan%" dans la sortie.
- Comparaisons "cette année (2026)" vs "année précédente (2025)" : ne compare que les mois existant en 2026 (Jan-Fév pour APF). Pas de total annuel 2026.
- Données mensuelles (date_stat = 1er du mois). Pas de jour exact.
- Si tu reçois [Données collectées par le Chercheur STATOUR] : crée le DataFrame depuis ces données (pd.DataFrame({...})) — n'appelle pas `sql()`.
- Renomme les colonnes avant d'afficher :
    nationalite | nationalite_name → Pays de résidence
    region | region_name           → Région
    voie                           → Voie
    continent                      → Continent
    mre                            → MRE
    tes                            → TES
    total                          → Arrivées
    nuitees                        → Nuitées
    arrivees                       → Arrivées
    type_eht_libelle               → Type d'hébergement
    delegation_name                → Délégation
    etablissement_libelle_fr       → Établissement
- Tableaux : `print(to_md(result))` (formate les nombres avec virgules automatiquement).
- Jamais `.to_string()`. Tableaux séparés par une ligne vide : `print("\\n**Label:**")` puis `print(to_md(t))`.
- Top N : utilise `TOP N` côté SQL.
- Nombres scalaires : `f"{int(value):,}"`.
- Réponds dans la langue de l'utilisateur (français/anglais/arabe).

EXAMPLE — total + breakdowns en T-SQL :
```python
# Total Février 2026
res = sql('''
  SELECT SUM(mre)+SUM(tes) AS total_arrivees
    FROM [dbo_GOLD].[fact_statistiques_apf]
    WHERE YEAR(date_stat) = 2026 AND MONTH(date_stat) = 2
''')
total = int(res.iloc[0,0])
print(f"Février 2026 — Total arrivées : **{total:,}**")

# Top 5 pays
nat = sql('''
  SELECT TOP 5 nationalite AS [Pays de résidence],
               SUM(mre+tes) AS [Arrivées]
    FROM [dbo_GOLD].[fact_statistiques_apf]
    WHERE YEAR(date_stat) = 2026 AND MONTH(date_stat) = 2
    GROUP BY nationalite ORDER BY [Arrivées] DESC
''')
print("\\n**Top 5 pays de résidence :**")
print(to_md(nat))

voie = sql('''
  SELECT voie AS [Voie], SUM(mre+tes) AS [Arrivées]
    FROM [dbo_GOLD].[fact_statistiques_apf]
    WHERE YEAR(date_stat) = 2026 AND MONTH(date_stat) = 2
    GROUP BY voie ORDER BY [Arrivées] DESC
''')
print("\\n**Répartition par voie :**")
print(to_md(voie))
```

CHART GUIDELINES (only when requested):
- Bar: rankings | Line: time trends | Pie: proportions | Stacked bar: segment comparisons
- Colors: ['#1B4F72', '#C0392B', '#27AE60', '#F39C12', '#8E44AD', '#1ABC9C']
- Template: plotly_white. Always include title, axis labels, data labels.""",


    "orchestrator": """Tu es le cerveau de routage de STATOUR, la plateforme du Ministère du Tourisme du Maroc.
Tu comprends L'INTENTION de l'utilisateur et routes vers le bon agent.

3 AGENTS DISPONIBLES:
- ANALYTICS: données, stats, graphiques, chiffres, tendances, classements, commandes /
- RESEARCHER: actualités, stratégie, Vision 2030, contexte, données historiques hors plage, web, news
- NORMAL: salutations, aide, questions sur STATOUR, conversation générale

RÈGLES:
1. Commandes / → TOUJOURS ANALYTICS
2. Demande explicite chercher/search → TOUJOURS RESEARCHER
3. Données/stats dans la plage → ANALYTICS
4. Actualités/stratégie/contexte → RESEARCHER
5. Salutations/aide → NORMAL
6. Follow-up court → MÊME agent que le précédent
7. Doute → agent précédent si existe, sinon ANALYTICS

RÉPONDS UN SEUL MOT: ANALYTICS ou RESEARCHER ou NORMAL""",
}