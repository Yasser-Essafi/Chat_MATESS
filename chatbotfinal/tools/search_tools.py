"""
STATOUR Web Search Tools — Fixed
==================================
Tavily-powered web search restricted to trusted Moroccan
and international tourism websites.

Fixes from original:
- Proper error handling (exceptions, not error-mixed-into-results)
- Retry with exponential backoff on transient failures
- Automatic fallback to broad search when trusted-only returns nothing
- Thread-safe
- Integrated with shared logger and cache
- Skips empty/meaningless results

Usage:
    from tools.search_tools import TourismSearchTool

    searcher = TourismSearchTool()

    # Basic search (trusted domains first, falls back to broad)
    results = searcher.search("tourisme Maroc 2024")

    # News only
    results = searcher.search_news("Morocco tourism news")

    # Broad (no domain restriction)
    results = searcher.search_broad("world tourism statistics")

    # Formatted for LLM context (cached)
    text = searcher.search_formatted("arrivées touristes Maroc")

    # Quick condensed context
    context = searcher.get_quick_context("Morocco arrivals 2024")
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TAVILY_API_KEY, BRAVE_API_KEY, SEARCH_BACKEND
from utils.logger import get_logger
from utils.cache import shared_cache

logger = get_logger("statour.search")

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None
    logger.warning(
        "tavily package not installed — Tavily backend disabled. "
        "Install tavily-python or use SEARCH_BACKEND=brave with BRAVE_API_KEY."
    )

# requests is a transitive dep of openai/tavily/chromadb, always available
import requests


# ══════════════════════════════════════════════════════════════════════════════
# Tool Schemas — native function calling with Azure OpenAI (gpt-5-mini)
# ══════════════════════════════════════════════════════════════════════════════
# The LLM decides autonomously when to call these tools and generates optimal queries.

SEARCH_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Recherche factuelle et actualités sur le web via Tavily. "
                "Utiliser pour : actualités récentes tourisme Maroc, statistiques officielles publiées, "
                "événements récents (CAN, conférences, salons), données Office des Changes, ONMT, HCP. "
                "NE PAS utiliser pour des analyses causales ou des 'pourquoi'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requête de recherche concise et précise (3-8 mots)",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 3,
                        "description": "Nombre de résultats souhaités (1-5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Recherche sémantique et contextuelle via Exa.ai. "
                "Utiliser pour : expliquer des tendances, analyser des facteurs causaux, "
                "questions 'pourquoi', contexte géopolitique/économique, comparaisons internationales, "
                "recherches multi-sources sur des sujets complexes. "
                "Meilleur que web_search pour les questions d'analyse et de compréhension."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requête de recherche sémantique (peut être plus longue et naturelle)",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 3,
                        "description": "Nombre de résultats souhaités (1-5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Trusted Domain Lists
# ══════════════════════════════════════════════════════════════════════════════
# Organized by category for easier maintenance and inspection.

MOROCCO_NEWS_DOMAINS = [
    "hespress.com",
    "le360.ma",
    "medias24.com",
    "mapnews.ma",
    "moroccoworldnews.com",
    "telquel.ma",
    "lematin.ma",
    "leconomiste.com",
    "lavieeco.com",
]

TOURISM_DOMAINS = [
    "visitmorocco.com",
    "tourisme.gov.ma",
    "moroccotravel.blog",
    "lonelyplanet.com",
    "tripadvisor.com",
]

INSTITUTIONAL_DOMAINS = [
    "unwto.org",
    "hcp.ma",
    "oc.gov.ma",           # Office des Changes
    "onda.ma",             # Aéroports
    "observatoiredutourisme.ma",
    "finances.gov.ma",
    "worldbank.org",
    "data.worldbank.org",
]

ALL_TRUSTED_DOMAINS = (
    MOROCCO_NEWS_DOMAINS
    + TOURISM_DOMAINS
    + INSTITUTIONAL_DOMAINS
)


# ══════════════════════════════════════════════════════════════════════════════
# Custom Exceptions
# ══════════════════════════════════════════════════════════════════════════════

class SearchError(Exception):
    """Raised when search fails after all retries."""
    pass


class SearchConfigError(Exception):
    """Raised when search tool is misconfigured."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

MAX_SEARCH_RETRIES = 2
RETRY_BASE_DELAY = 1.0     # seconds
RETRY_MAX_DELAY = 10.0     # seconds
MAX_RESULTS_LIMIT = 10
CACHE_SOURCE_WEB = "web"
CACHE_SOURCE_NEWS = "web_news"
CACHE_SOURCE_BROAD = "web_broad"


# ══════════════════════════════════════════════════════════════════════════════
# Brave Search Backend
# ══════════════════════════════════════════════════════════════════════════════
# Brave has its own independent web index, ~30% faster than Tavily on average
# (669ms vs 998ms, 2026 benchmarks). Falls back gracefully when key is absent.

_BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


def _brave_search(query: str, max_results: int = 5, freshness: Optional[str] = None) -> List[Dict]:
    """Issue a Brave Search Web API call and return results in the Tavily-shaped
    dict format the rest of the code expects.

    Args:
        query: search string
        max_results: 1-20
        freshness: "pd" (past day), "pw" (past week), "pm" (past month), "py" (past year)

    Returns empty list on any failure (network, missing key, parse error).
    """
    if not BRAVE_API_KEY:
        return []
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {"q": query, "count": min(max(max_results, 1), 20)}
    if freshness:
        params["freshness"] = freshness
    try:
        resp = requests.get(_BRAVE_API_URL, headers=headers, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Brave search failed for '%s': %s", query[:60], str(e)[:120])
        return []

    web_block = (data.get("web") or {}).get("results") or []
    out: List[Dict] = []
    for item in web_block:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        # Brave returns "description" as the snippet; sometimes "extra_snippets"
        content = (item.get("description") or "").strip()
        if not content and item.get("extra_snippets"):
            content = " ".join(item["extra_snippets"])[:500]
        if not title and not content:
            continue
        out.append({
            "title": title,
            "url": url,
            "content": content,
            "score": 1.0,  # Brave doesn't return relevance scores
        })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Search Tool
# ══════════════════════════════════════════════════════════════════════════════

class TourismSearchTool:
    """
    Web search tool for Moroccan tourism news and data.
    Uses Tavily API with domain restrictions, retry logic,
    automatic fallback, and result caching.

    Features:
        - Trusted-domain search (whitelisted sites only)
        - Automatic fallback to broad search when no trusted results
        - Retry with exponential backoff on transient failures
        - Result caching via shared SearchCache
        - Separate methods for general, news, and broad search
        - Formatted output for LLM context injection
        - Quick condensed context for token-budget-sensitive agents
    """

    def __init__(self):
        """
        Initialize the search tool. Backend chosen via SEARCH_BACKEND env var:
            "tavily" — Tavily only (legacy)
            "brave"  — Brave only (~30% faster, requires BRAVE_API_KEY)
            "auto"   — Brave first when available, fallback to Tavily

        Raises:
            SearchConfigError: If no usable backend is configured.
        """
        self._backend = SEARCH_BACKEND if SEARCH_BACKEND in {"tavily", "brave", "auto"} else "auto"
        self._brave_available = bool(BRAVE_API_KEY)
        self.client = None  # Tavily client, None when running in pure Brave mode
        self._exa = None
        self._exa_available = False

        if self._backend == "brave":
            if not self._brave_available:
                raise SearchConfigError(
                    "SEARCH_BACKEND=brave but BRAVE_API_KEY is not set. "
                    "Get a free key at https://api.search.brave.com"
                )
            logger.info("Search backend: Brave (exclusive)")
            return

        # Tavily or auto-mode → need Tavily client as fallback
        if not TAVILY_API_KEY or not TAVILY_API_KEY.strip():
            if self._backend == "auto" and self._brave_available:
                logger.info("Search backend: Brave (auto mode, Tavily key missing)")
                return
            raise SearchConfigError(
                "TAVILY_API_KEY not configured in .env file. "
                "Get your key at https://tavily.com"
            )

        if TavilyClient is None:
            if self._backend == "auto" and self._brave_available:
                logger.info("Search backend: Brave (auto mode, Tavily package missing)")
                return
            raise SearchConfigError(
                "tavily-python is not installed but Tavily backend is required."
            )

        try:
            self.client = TavilyClient(api_key=TAVILY_API_KEY)
        except Exception as e:
            raise SearchConfigError(f"Failed to initialize Tavily client: {e}")

        if self._backend == "auto" and self._brave_available:
            logger.info("Search backend: Brave-first → Tavily fallback")
        else:
            logger.info("Search backend: Tavily")

        # Exa.ai — semantic search (optional, graceful fallback if not configured)
        from config.settings import EXA_API_KEY
        if EXA_API_KEY:
            try:
                self._exa = ExaSearchTool()
                self._exa_available = True
                logger.info("Exa.ai semantic search: available")
            except Exception as e:
                self._exa = None
                self._exa_available = False
                logger.warning("Exa.ai unavailable (non-critical): %s", e)
        else:
            self._exa = None
            self._exa_available = False
            logger.info("Exa.ai: EXA_API_KEY not set — semantic search disabled")

    # ──────────────────────────────────────────────────────────────────────
    # Core Search (with retry and fallback)
    # ──────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        use_trusted_only: bool = True,
        topic: str = "general",
    ) -> List[Dict]:
        """
        Search the web for tourism-related information.

        Workflow:
            1. Search trusted domains
            2. If no results and use_trusted_only=True, retry with broad search
            3. On transient failure, retry with exponential backoff
            4. Return empty list if all attempts fail (never raises)

        Args:
            query: Search query (French, English, or Arabic).
            max_results: Number of results to return (1-10).
            search_depth: "basic" (fast) or "advanced" (thorough).
            use_trusted_only: If True, restrict to whitelisted domains first.
            topic: "general" or "news".

        Returns:
            List of dicts, each with keys:
                - title (str): Result title
                - url (str): Source URL
                - content (str): Result snippet/content
                - score (float): Relevance score (0-1)

            Returns empty list if no results found.
            Never returns error dicts — errors are logged instead.
        """
        # Validate inputs
        if not query or not query.strip():
            logger.warning("Empty search query — returning empty results")
            return []

        query = query.strip()
        max_results = min(max(max_results, 1), MAX_RESULTS_LIMIT)

        # ── Step 1: Search (trusted or broad depending on flag) ──
        results = self._search_with_retry(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            use_trusted_only=use_trusted_only,
            topic=topic,
        )

        # ── Step 2: Fallback — trusted returned nothing → try broad ──
        if not results and use_trusted_only:
            logger.info(
                "No results from trusted domains for '%s' — "
                "falling back to broad search",
                query[:60],
            )
            results = self._search_with_retry(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                use_trusted_only=False,
                topic=topic,
            )

            if results:
                logger.info(
                    "Broad fallback returned %d results for '%s'",
                    len(results), query[:60],
                )
            else:
                logger.warning(
                    "Both trusted and broad search returned no results for '%s'",
                    query[:60],
                )

        return results

    def _search_with_retry(
        self,
        query: str,
        max_results: int,
        search_depth: str,
        use_trusted_only: bool,
        topic: str,
    ) -> List[Dict]:
        """
        Execute a single search with retry logic on transient failures.
        Returns empty list on complete failure (never raises to caller).

        Args:
            query: Search query.
            max_results: Number of results.
            search_depth: "basic" or "advanced".
            use_trusted_only: Domain restriction flag.
            topic: "general" or "news".

        Returns:
            List of result dicts, or empty list.
        """
        last_error = None

        for attempt in range(MAX_SEARCH_RETRIES + 1):
            try:
                return self._execute_search(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    use_trusted_only=use_trusted_only,
                    topic=topic,
                )
            except Exception as e:
                last_error = e

                if attempt < MAX_SEARCH_RETRIES:
                    delay = min(
                        RETRY_BASE_DELAY * (2 ** attempt),
                        RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "Search attempt %d/%d failed for '%s': %s "
                        "— retrying in %.1fs",
                        attempt + 1,
                        MAX_SEARCH_RETRIES + 1,
                        query[:50],
                        str(e)[:100],
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Search failed after %d attempts for '%s': %s",
                        MAX_SEARCH_RETRIES + 1,
                        query[:50],
                        str(last_error)[:200],
                    )

        return []

    def _execute_search(
        self,
        query: str,
        max_results: int,
        search_depth: str,
        use_trusted_only: bool,
        topic: str,
    ) -> List[Dict]:
        """
        Execute a single search call against the active backend.

        Brave is preferred when available (mode "brave" or "auto"). If Brave
        returns no results in "auto" mode, we transparently fall back to
        Tavily so callers see the same shape regardless of backend.
        """
        # ── Brave path (no domain restriction support — append site filters
        # via "site:" tokens to keep parity with the trusted-domain feature) ──
        if self._backend == "brave" or (self._backend == "auto" and self._brave_available):
            brave_query = query
            if use_trusted_only:
                # Brave supports "site:" operators in the query string
                top_sites = ALL_TRUSTED_DOMAINS[:6]
                brave_query = f"{query} ({' OR '.join('site:' + d for d in top_sites)})"
            freshness = "py" if topic == "news" else None
            brave_results = _brave_search(brave_query, max_results=max_results, freshness=freshness)
            if brave_results:
                logger.debug(
                    "Brave returned %d results for '%s' (trusted=%s, topic=%s)",
                    len(brave_results), query[:50], use_trusted_only, topic,
                )
                return brave_results
            if self._backend == "brave":
                # Pure Brave mode — no fallback
                return []
            # Auto mode → fall through to Tavily

        if not self.client:
            return []

        kwargs = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "topic": topic,
        }

        if use_trusted_only:
            kwargs["include_domains"] = ALL_TRUSTED_DOMAINS

        response = self.client.search(**kwargs)

        results = []
        for item in response.get("results", []):
            title = item.get("title", "").strip()
            url = item.get("url", "").strip()
            content = item.get("content", "").strip()
            score = item.get("score", 0)

            # Skip entries with no meaningful content
            if not title and not content:
                logger.debug("Skipping empty result from %s", url)
                continue

            results.append({
                "title": title,
                "url": url,
                "content": content,
                "score": round(float(score), 4),
            })

        logger.debug(
            "Tavily returned %d results for '%s' "
            "(trusted=%s, topic=%s, depth=%s)",
            len(results),
            query[:50],
            use_trusted_only,
            topic,
            search_depth,
        )

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Convenience Methods
    # ──────────────────────────────────────────────────────────────────────

    def search_multi(
        self,
        queries: List[Tuple[str, str]],
        max_results_per_query: int = 2,
        use_trusted_only: bool = False,
        max_workers: int = 4,
    ) -> Dict[str, List[Dict]]:
        """Run several searches in parallel and return them grouped by label.

        Args:
            queries: list of (label, query_string) tuples
            max_results_per_query: max items per individual search
            use_trusted_only: whether to constrain to whitelisted domains
            max_workers: parallel HTTP workers (default 4 — most public APIs
                allow this comfortably)

        Returns:
            { label: [result_dict, ...] } — labels with no results are omitted.
        """
        if not queries:
            return {}

        results: Dict[str, List[Dict]] = {}

        def _run(label: str, query: str) -> Tuple[str, List[Dict]]:
            try:
                items = self.search(
                    query=query,
                    max_results=max_results_per_query,
                    search_depth="basic",  # parallel branches → keep them light
                    use_trusted_only=use_trusted_only,
                    topic="general",
                )
                return label, items
            except Exception as e:
                logger.debug("Multi-search branch '%s' failed: %s", label, e)
                return label, []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run, label, q) for label, q in queries]
            for fut in as_completed(futures):
                label, items = fut.result()
                if items:
                    results[label] = items

        return results

    def search_news(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search recent news only from trusted sources.

        Args:
            query: Search query.
            max_results: Number of results (1-10).

        Returns:
            List of result dicts.
        """
        return self.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            use_trusted_only=True,
            topic="news",
        )

    def search_broad(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search without domain restriction (broader results).
        Useful for international comparisons or niche queries.

        Args:
            query: Search query.
            max_results: Number of results (1-10).

        Returns:
            List of result dicts.
        """
        return self.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            use_trusted_only=False,
            topic="general",
        )

    def smart_search(
        self,
        query: str,
        analysis_type: str = "factual",
        max_results: int = 5,
    ) -> List[Dict]:
        """
        Intelligently route search to the best backend based on query type.

        Routing logic:
        - "causal" | "comparative" | "trend" | "forecasting" → Exa.ai first (semantic), fallback Tavily
        - "factual" | "news" | anything else → Tavily first, fallback Brave

        Args:
            query: Search query string.
            analysis_type: Type of analysis — "causal", "comparative", "trend", "factual", "news".
            max_results: Number of results to return.

        Returns:
            List of result dicts (title, url, content, score). Never raises.
        """
        SEMANTIC_TYPES = {"causal", "comparative", "trend", "forecasting"}

        if analysis_type in SEMANTIC_TYPES and self._exa_available:
            results = self._exa.search(query, max_results=max_results)
            if results:
                logger.debug(
                    "smart_search [Exa/%s]: %d results for '%s'",
                    analysis_type, len(results), query[:50]
                )
                return results
            logger.debug(
                "smart_search [Exa→Tavily fallback/%s]: Exa empty for '%s'",
                analysis_type, query[:50]
            )

        results = self.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            use_trusted_only=True,
            topic="news" if analysis_type == "news" else "general",
        )
        if results:
            logger.debug(
                "smart_search [Tavily/%s]: %d results for '%s'",
                analysis_type, len(results), query[:50]
            )
        return results

    # ──────────────────────────────────────────────────────────────────────
    # Formatted Output for LLM Context
    # ──────────────────────────────────────────────────────────────────────

    def search_formatted(
        self,
        query: str,
        max_results: int = 5,
        use_cache: bool = True,
    ) -> str:
        """
        Search and format results as text for LLM context injection.

        Results are cached to avoid redundant API calls for identical queries.

        Args:
            query: Search query.
            max_results: Number of results (1-10).
            use_cache: If True, check/store formatted results in shared cache.

        Returns:
            Formatted string with search results, or error message if none found.
        """
        # ── Check cache for formatted results ──
        cache_key = f"formatted:{max_results}:{query}"
        if use_cache:
            cached = shared_cache.get(cache_key, source=CACHE_SOURCE_WEB)
            if cached is not None:
                logger.debug("Returning cached formatted results for '%s'", query[:50])
                return cached

        # ── Execute search ──
        results = self.search(query, max_results)

        if not results:
            no_result_msg = f"❌ Aucun résultat web trouvé pour : {query}"
            return no_result_msg

        # ── Format results ──
        parts = [
            f"🌐 **{len(results)} résultats web trouvés pour** : _{query}_\n"
        ]

        for i, r in enumerate(results, 1):
            title = r["title"] or "Sans titre"
            url = r["url"] or "URL inconnue"
            content = r["content"] or "Pas de contenu"
            score = r["score"]

            parts.append(
                f"---\n"
                f"**{i}. {title}**\n"
                f"🔗 {url}\n"
                f"📊 Score: {score}\n\n"
                f"{content}\n"
            )

        formatted = "\n".join(parts)

        # ── Store in cache ──
        if use_cache:
            shared_cache.set(cache_key, formatted, source=CACHE_SOURCE_WEB)
            logger.debug("Cached formatted results for '%s'", query[:50])

        return formatted

    # ──────────────────────────────────────────────────────────────────────
    # Quick Context for Agent (condensed, token-efficient)
    # ──────────────────────────────────────────────────────────────────────

    def get_quick_context(
        self,
        query: str,
        max_results: int = 3,
        max_chars_per_result: int = 300,
        use_cache: bool = True,
    ) -> str:
        """
        Get a condensed context string suitable for injection into agent prompts.
        Much shorter than search_formatted — optimized for token budget.

        Args:
            query: Search query.
            max_results: Number of results (1-3 recommended).
            max_chars_per_result: Max characters per result snippet.
            use_cache: If True, check/store in shared cache.

        Returns:
            Condensed context string, or empty string if no results.
        """
        # ── Check cache ──
        cache_key = f"quick:{max_results}:{max_chars_per_result}:{query}"
        if use_cache:
            cached = shared_cache.get(cache_key, source=CACHE_SOURCE_WEB)
            if cached is not None:
                return cached

        # ── Execute search ──
        results = self.search(query, max_results=max_results)

        if not results:
            return ""

        # ── Build condensed context ──
        parts = []
        for r in results:
            content = r.get("content", "")
            title = r.get("title", "")

            if not content and not title:
                continue

            snippet = content[:max_chars_per_result]
            if len(content) > max_chars_per_result:
                # Try to cut at a word boundary
                last_space = snippet.rfind(" ")
                if last_space > max_chars_per_result * 0.7:
                    snippet = snippet[:last_space]
                snippet += "..."

            if title:
                parts.append(f"[{title}] {snippet}")
            else:
                parts.append(snippet)

        context = "\n\n".join(parts)

        # ── Store in cache ──
        if use_cache and context:
            shared_cache.set(cache_key, context, source=CACHE_SOURCE_WEB)

        return context

    # ──────────────────────────────────────────────────────────────────────
    # Inspection Methods
    # ──────────────────────────────────────────────────────────────────────

    def get_trusted_domains(self) -> Dict[str, List[str]]:
        """
        Return the trusted domain lists for inspection/debugging.

        Returns:
            Dict with keys: 'news', 'tourism', 'institutional', 'all'
        """
        return {
            "news": MOROCCO_NEWS_DOMAINS.copy(),
            "tourism": TOURISM_DOMAINS.copy(),
            "institutional": INSTITUTIONAL_DOMAINS.copy(),
            "all": ALL_TRUSTED_DOMAINS.copy(),
        }

    @staticmethod
    def get_domain_count() -> int:
        """Return total number of trusted domains."""
        return len(ALL_TRUSTED_DOMAINS)


# ══════════════════════════════════════════════════════════════════════════════
# Exa.ai Semantic Search Tool
# ══════════════════════════════════════════════════════════════════════════════

class ExaSearchTool:
    """
    Semantic search via Exa.ai — optimized for causal and research queries.

    Best for: "pourquoi", context analysis, trend explanation, multi-hop research.
    Uses neural/semantic search (embedding-based) rather than keyword matching.

    Falls back gracefully when EXA_API_KEY is not set (returns empty list, no exception).
    """

    def __init__(self):
        from config.settings import EXA_API_KEY
        if not EXA_API_KEY:
            raise ValueError(
                "EXA_API_KEY not configured — add it to .env "
                "(get free key at https://exa.ai)"
            )
        try:
            from exa_py import Exa
            self._client = Exa(api_key=EXA_API_KEY)
            logger.info("Exa.ai search initialized")
        except ImportError:
            raise ImportError(
                "exa-py package not installed. Run: pip install exa-py"
            )

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Semantic search using Exa.ai neural index.

        Args:
            query: Natural language search query.
            max_results: Number of results (1-10).

        Returns:
            List of dicts with keys: title, url, content, score.
            Returns empty list on any failure (never raises).
        """
        try:
            results = self._client.search_and_contents(
                query,
                num_results=min(max(max_results, 1), 10),
                highlights={
                    "num_sentences": 3,
                    "highlights_per_url": 2,
                },
            )
            output = []
            for r in results.results:
                if not r.url:
                    continue
                content = ""
                if hasattr(r, "highlights") and r.highlights:
                    content = " ".join(r.highlights)
                elif hasattr(r, "text") and r.text:
                    content = r.text[:500]
                output.append({
                    "title": r.title or "",
                    "url": r.url,
                    "content": content,
                    "score": 1.0,  # Exa doesn't return relevance scores
                })
            logger.debug("Exa search: %d results for '%s'", len(output), query[:50])
            return output
        except Exception as e:
            logger.warning("Exa search failed for '%s': %s", query[:50], str(e)[:120])
            return []


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("🌐 STATOUR — Web Search Tool Test")
    print("=" * 60)
    print()

    # ── Initialize ──
    try:
        searcher = TourismSearchTool()
        print(f"✅ Tavily client initialized")
        print(f"   Trusted domains: {searcher.get_domain_count()}")
        print()
    except SearchConfigError as e:
        print(f"❌ Configuration error: {e}")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return

    # ── Test 1: Trusted domain search ──
    print("=" * 60)
    print("🔍 Test 1: Trusted Domain Search")
    print("=" * 60)
    print()

    test_queries = [
        "tourisme Maroc 2024 statistiques arrivées",
        "Morocco tourism news 2024",
        "Opération Marhaba 2024",
    ]

    for query in test_queries:
        print(f"  🔍 Query: {query}")
        print(f"  {'-' * 50}")

        results = searcher.search(query, max_results=3)

        if not results:
            print("     ❌ No results found")
        else:
            for r in results:
                print(f"     📰 {r['title']}")
                print(f"        🔗 {r['url']}")
                print(f"        📊 Score: {r['score']}")
                print(f"        {r['content'][:150]}...")
                print()
        print()

    # ── Test 2: News search ──
    print("=" * 60)
    print("📰 Test 2: News Search")
    print("=" * 60)
    print()

    news_results = searcher.search_news("Morocco tourism 2024", max_results=2)
    if not news_results:
        print("  ❌ No news results")
    else:
        for r in news_results:
            print(f"  📰 {r['title']}")
            print(f"     🔗 {r['url']}")
            print()

    # ── Test 3: Broad search ──
    print("=" * 60)
    print("🌍 Test 3: Broad Search (no domain restriction)")
    print("=" * 60)
    print()

    broad_results = searcher.search_broad(
        "Morocco GDP tourism contribution percentage",
        max_results=2,
    )
    if not broad_results:
        print("  ❌ No broad results")
    else:
        for r in broad_results:
            print(f"  📰 {r['title']}")
            print(f"     🔗 {r['url']}")
            print()

    # ── Test 4: Formatted output ──
    print("=" * 60)
    print("📋 Test 4: Formatted Output (for LLM)")
    print("=" * 60)
    print()

    formatted = searcher.search_formatted(
        "tourisme Maroc statistiques 2024",
        max_results=2,
    )
    print(formatted)
    print()

    # ── Test 5: Quick context ──
    print("=" * 60)
    print("⚡ Test 5: Quick Context (condensed)")
    print("=" * 60)
    print()

    context = searcher.get_quick_context("Morocco arrivals 2024")
    if context:
        print(f"  Context ({len(context)} chars):")
        print(f"  {context[:500]}")
    else:
        print("  ❌ No context returned")
    print()

    # ── Test 6: Cache verification ──
    print("=" * 60)
    print("🗄️  Test 6: Cache Verification")
    print("=" * 60)
    print()

    # Second call should hit cache
    formatted_again = searcher.search_formatted(
        "tourisme Maroc statistiques 2024",
        max_results=2,
    )
    cache_stats = shared_cache.stats()
    print(f"  Cache size: {cache_stats['size']}")
    print(f"  Cache hits: {cache_stats['hits']}")
    print(f"  Cache misses: {cache_stats['misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate_pct']}%")
    print(f"  By source: {cache_stats['by_source']}")
    print()

    # ── Test 7: Empty query handling ──
    print("=" * 60)
    print("🧪 Test 7: Edge Cases")
    print("=" * 60)
    print()

    # Empty query
    empty_results = searcher.search("")
    assert empty_results == [], f"Expected empty list, got {empty_results}"
    print("  ✅ Empty query returns empty list")

    # Whitespace query
    ws_results = searcher.search("   ")
    assert ws_results == [], f"Expected empty list, got {ws_results}"
    print("  ✅ Whitespace query returns empty list")

    # Very long query (should not crash)
    long_query = "Morocco tourism " * 100
    long_results = searcher.search(long_query, max_results=1)
    print(f"  ✅ Long query handled ({len(long_results)} results)")

    print()

    # ── Show trusted domains ──
    print("=" * 60)
    print("🏛️  Trusted Domains")
    print("=" * 60)
    print()

    domains = searcher.get_trusted_domains()
    for category, domain_list in domains.items():
        if category != "all":
            print(f"  {category.upper()} ({len(domain_list)}):")
            for d in domain_list:
                print(f"    • {d}")
            print()

    print(f"  TOTAL: {searcher.get_domain_count()} trusted domains")
    print()
    print("✅ All tests complete!")


if __name__ == "__main__":
    main()
