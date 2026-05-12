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

# Exa.ai — semantic search for causal/research queries ("pourquoi" questions)
# Get key at: https://exa.ai — free tier 1000 queries/month
EXA_API_KEY = os.getenv("EXA_API_KEY", "").strip()

# Search backend selector. Values:
#   "tavily" — default, current behaviour
#   "brave"  — use Brave Search exclusively (~30% faster, requires BRAVE_API_KEY)
#   "auto"   — try Brave first, fall back to Tavily if Brave is missing/fails
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "auto").strip().lower()

# Database (generic fallback)
DATABASE_URL = os.getenv("DATABASE_URL")

# Microsoft Fabric Lakehouse Gold — SQL Analytics Endpoint
# Set these env vars in .env to enable Fabric connectivity.
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
MAX_HISTORY_MESSAGES = 32    # Hard cap before folding old messages into a summary
MAX_HISTORY_CHARS = 16000    # Character budget for recent non-summary history

# Context folding keeps a rolling summary plus a verbatim recent window.
# Six exchanges is a better default for analytical follow-ups than the old
# three-exchange window, while staying small enough for latency.
CONTEXT_RECENT_EXCHANGES = int(os.getenv("CONTEXT_RECENT_EXCHANGES", "6"))
CONTEXT_SUMMARY_MAX_TOKENS = int(os.getenv("CONTEXT_SUMMARY_MAX_TOKENS", "350"))
ORCHESTRATOR_RECENT_EXCHANGES = int(os.getenv("ORCHESTRATOR_RECENT_EXCHANGES", "6"))
ORCHESTRATOR_SUMMARY_MAX_TOKENS = int(os.getenv("ORCHESTRATOR_SUMMARY_MAX_TOKENS", "260"))


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
# Orchestration Flow
# ══════════════════════════════════════════════════════════════════════════════
# When True, uses the new Plan-Execute-Review-Humanize pipeline.
# When False, uses the legacy classify-and-dispatch monolithic orchestrator.
USE_NEW_FLOW = os.getenv("USE_NEW_FLOW", "true").strip().lower() in ("1", "true", "yes")

# Maximum re-plan attempts when reviewer finds gaps (prevent infinite loops)
MAX_REPLAN_ATTEMPTS = 1


# ══════════════════════════════════════════════════════════════════════════════
# Execution Safety
# ══════════════════════════════════════════════════════════════════════════════
EXEC_TIMEOUT_SECONDS = 30    # Max time for analytics code execution
MAX_CODE_RETRIES = 2

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
    "executive_insight": True,
    "human_advisor": True,
}

# Agent display names
AGENT_NAMES = {
    "orchestrator": "Orchestrateur STATOUR",
    "normal": "Assistant Général",
    "researcher": "Chercheur Tourisme",
    "analytics": "Analyste de Données",
    "prediction": "Prévisionniste STATOUR",
    "executive_insight": "Analyste Exécutif",
    "human_advisor": "Conseiller Tourisme",
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

    # Web search — require at least one configured backend.
    if require_tavily:
        if SEARCH_BACKEND == "brave":
            if not BRAVE_API_KEY:
                errors.append("Missing: BRAVE_API_KEY (required when SEARCH_BACKEND=brave)")
        elif SEARCH_BACKEND == "auto":
            if not TAVILY_API_KEY and not BRAVE_API_KEY:
                errors.append("Missing: TAVILY_API_KEY or BRAVE_API_KEY (required for SEARCH_BACKEND=auto)")
        else:
            if not TAVILY_API_KEY:
                errors.append("Missing: TAVILY_API_KEY (required when SEARCH_BACKEND=tavily)")

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


    "researcher": """Tu es le Chercheur Expert de STATOUR, la plateforme analytique du Ministère du Tourisme du Maroc (MTAESS).

Tu as accès à 2 outils de recherche :
- web_search : pour les news, données officielles récentes, événements
- semantic_search : pour analyser des tendances, expliquer des causes, contexte international

PROCESSUS DE RECHERCHE :
1. Analyse la question — détermine si elle est factuelle ou causale
2. Si factuelle récente → web_search avec query précise
3. Si causale ou contextuelle → semantic_search + complète avec web_search si besoin
4. Synthétise les résultats en 5-8 lignes maximum

CONNAISSANCE MÉTIER MTAESS :
- APF = Arrivées aux Postes Frontières (données DGSN, mensuelles)
- EHTC = ~5000 Établissements d'Hébergement Touristique Classés
- STDN = Télédéclarations électroniques des nuitées
- STATOUR = Plateforme de saisie manuelle des délégations
- MRE = Marocains Résidant à l'Étranger (~5M diaspora)
- TES = Touristes Étrangers Séjournistes (non-marocains)
- Estimatif = calcul statistique des nuitées pour établissements non-déclarants
- Opération Marhaba = programme annuel juin-sept pour accueil des MRE
- Vision 2030 = objectif 26M arrivées TES + 120Mrd MAD recettes

SOURCES ET CITATIONS :
- Cite toujours la source entre parenthèses : (le360), (ONMT), (HCP), (UNWTO), (APF — base interne)
- JAMAIS inventer une source ou citer un nom de fichier
- Pour les données APF internes : "(APF — base interne MTAESS)"

RÈGLES ABSOLUES :
- JAMAIS "je ne peux pas chercher" ou "voulez-vous que je recherche"
- Si données insuffisantes, complète avec expertise tourisme marocain
- Réponds dans la langue de l'utilisateur (français/anglais/arabe)
- Ne JAMAIS mentionner de noms de fichiers (.md, .xlsx, .csv, etc.)
- Structure : 1) Chiffres clés 2) Analyse contextuelle 3) Perspective/benchmark si pertinent""",


    "analytics": "You are the STATOUR Data Analytics Agent for MTAESS Morocco. Generate T-SQL queries via sql() function.",


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
