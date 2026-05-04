"""
STATOUR Search Cache
=====================
Thread-safe, time-aware LRU cache shared across agents.
Prevents redundant API calls for identical queries within a TTL window.

Usage:
    from utils.cache import SearchCache
    
    cache = SearchCache(max_size=200, ttl_seconds=1800)
    
    # Check cache
    result = cache.get("tourism morocco 2024", source="web")
    if result is None:
        result = expensive_api_call(...)
        cache.set("tourism morocco 2024", result, source="web")
    
    # Stats
    print(cache.stats())
    
    # Clear
    cache.clear()
"""

import time
import threading
from hashlib import sha256
from typing import Optional, Any, Dict

# Import logger — handle both direct run and module import
try:
    from utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("statour.cache")


class SearchCache:
    """
    Thread-safe, time-aware LRU cache.

    Features:
        - TTL-based expiration (stale entries auto-removed on access)
        - LRU eviction when max_size is reached
        - Thread-safe via threading.Lock
        - Source-namespaced keys (same query, different source = different entry)
        - Hit/miss statistics for monitoring

    Args:
        max_size: Maximum number of cached entries.
        ttl_seconds: Time-to-live in seconds. Entries older than this are expired.
    """

    def __init__(self, max_size: int = 200, ttl_seconds: int = 1800):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    # ──────────────────────────────────────────────────────────────────────
    # Key Generation
    # ──────────────────────────────────────────────────────────────────────

    def _make_key(self, query: str, source: str) -> str:
        """
        Generate a deterministic cache key from query + source.
        Normalizes query (lowercase, stripped) so minor variations hit cache.
        """
        normalized = f"{source}:{query.lower().strip()}"
        return sha256(normalized.encode("utf-8")).hexdigest()

    # ──────────────────────────────────────────────────────────────────────
    # Core Operations
    # ──────────────────────────────────────────────────────────────────────

    def get(self, query: str, source: str = "default") -> Optional[str]:
        """
        Retrieve a cached result.

        Args:
            query: The search query string.
            source: Namespace for the cache entry (e.g., "web", "rag", "classify").

        Returns:
            Cached data string if found and not expired, else None.
        """
        key = self._make_key(query, source)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL expiration
            age = time.time() - entry["timestamp"]
            if age >= self._ttl:
                del self._cache[key]
                self._expirations += 1
                self._misses += 1
                logger.debug(
                    "Cache EXPIRED: [%s] '%s' (age: %.0fs > TTL: %ds)",
                    source, query[:50], age, self._ttl,
                )
                return None

            # Cache hit — update access time for LRU
            entry["last_access"] = time.time()
            self._hits += 1
            logger.debug(
                "Cache HIT: [%s] '%s' (age: %.0fs)",
                source, query[:50], age,
            )
            return entry["data"]

    def set(self, query: str, data: str, source: str = "default") -> None:
        """
        Store a result in the cache.

        Args:
            query: The search query string.
            data: The result data to cache.
            source: Namespace for the cache entry.
        """
        key = self._make_key(query, source)

        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_one()

            now = time.time()
            self._cache[key] = {
                "data": data,
                "timestamp": now,
                "last_access": now,
                "query": query[:100],  # Store truncated query for debugging
                "source": source,
            }
            logger.debug(
                "Cache SET: [%s] '%s' (%d bytes)",
                source, query[:50], len(data),
            )

    def has(self, query: str, source: str = "default") -> bool:
        """Check if a non-expired entry exists without updating access time."""
        key = self._make_key(query, source)

        with self._lock:
            if key not in self._cache:
                return False
            age = time.time() - self._cache[key]["timestamp"]
            if age >= self._ttl:
                del self._cache[key]
                self._expirations += 1
                return False
            return True

    def delete(self, query: str, source: str = "default") -> bool:
        """
        Explicitly remove a cache entry.

        Returns:
            True if the entry existed and was removed, False otherwise.
        """
        key = self._make_key(query, source)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Eviction
    # ──────────────────────────────────────────────────────────────────────

    def _evict_one(self) -> None:
        """
        Evict the least-recently-accessed entry.
        Must be called with self._lock held.
        """
        if not self._cache:
            return

        # First, try to evict expired entries
        now = time.time()
        expired_keys = [
            k for k, v in self._cache.items()
            if now - v["timestamp"] >= self._ttl
        ]
        if expired_keys:
            for k in expired_keys:
                del self._cache[k]
                self._expirations += 1
            logger.debug("Evicted %d expired entries", len(expired_keys))
            return

        # Otherwise, evict LRU (least recently accessed)
        lru_key = min(
            self._cache,
            key=lambda k: self._cache[k]["last_access"],
        )
        evicted = self._cache.pop(lru_key)
        self._evictions += 1
        logger.debug(
            "Cache LRU eviction: [%s] '%s'",
            evicted.get("source", "?"),
            evicted.get("query", "?")[:50],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Bulk Operations
    # ──────────────────────────────────────────────────────────────────────

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cache cleared: %d entries removed", count)
            return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries without waiting for access.

        Returns:
            Number of expired entries removed.
        """
        now = time.time()
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items()
                if now - v["timestamp"] >= self._ttl
            ]
            for k in expired_keys:
                del self._cache[k]
                self._expirations += 1

            if expired_keys:
                logger.debug("Cleanup: removed %d expired entries", len(expired_keys))

            return len(expired_keys)

    # ──────────────────────────────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            # Break down by source
            sources = {}
            for entry in self._cache.values():
                src = entry.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(hit_rate, 1),
                "evictions": self._evictions,
                "expirations": self._expirations,
                "by_source": sources,
            }

    def __len__(self) -> int:
        """Return current cache size."""
        return len(self._cache)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SearchCache(size={s['size']}/{s['max_size']}, "
            f"hit_rate={s['hit_rate_pct']}%, "
            f"ttl={s['ttl_seconds']}s)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module-level shared cache instance
# ══════════════════════════════════════════════════════════════════════════════
# All agents can import and share this single cache.
#
# Usage:
#     from utils.cache import shared_cache
#     result = shared_cache.get("query", source="web")

try:
    from config.settings import SEARCH_CACHE_MAX_SIZE, SEARCH_CACHE_TTL
    shared_cache = SearchCache(
        max_size=SEARCH_CACHE_MAX_SIZE,
        ttl_seconds=SEARCH_CACHE_TTL,
    )
except ImportError:
    shared_cache = SearchCache(max_size=200, ttl_seconds=1800)


# ══════════════════════════════════════════════════════════════════════════════
# CLI Test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 SearchCache Test")
    print("=" * 60)
    print()

    cache = SearchCache(max_size=5, ttl_seconds=3)

    # Test basic set/get
    cache.set("tourism morocco", "Result A", source="web")
    cache.set("tourism morocco", "Result B", source="rag")  # Different source = different entry

    assert cache.get("tourism morocco", "web") == "Result A"
    assert cache.get("tourism morocco", "rag") == "Result B"
    assert cache.get("tourism morocco", "other") is None  # Miss
    print("✅ Basic set/get works")

    # Test case-insensitive normalization
    assert cache.get("TOURISM MOROCCO", "web") == "Result A"
    assert cache.get("  Tourism Morocco  ", "web") == "Result A"
    print("✅ Case-insensitive normalization works")

    # Test has()
    assert cache.has("tourism morocco", "web") is True
    assert cache.has("nonexistent", "web") is False
    print("✅ has() works")

    # Test delete
    assert cache.delete("tourism morocco", "web") is True
    assert cache.get("tourism morocco", "web") is None
    assert cache.delete("tourism morocco", "web") is False  # Already deleted
    print("✅ delete() works")

    # Test TTL expiration
    cache.set("expiring query", "temp data", source="test")
    assert cache.get("expiring query", "test") == "temp data"
    print("⏳ Waiting for TTL expiration (3 seconds)...")
    time.sleep(3.5)
    assert cache.get("expiring query", "test") is None  # Expired
    print("✅ TTL expiration works")

    # Test LRU eviction (max_size=5)
    for i in range(6):
        cache.set(f"query_{i}", f"result_{i}", source="test")
    assert len(cache) <= 5
    print("✅ LRU eviction works")

    # Test cleanup
    cache.set("cleanup test", "data", source="test")
    time.sleep(3.5)
    removed = cache.cleanup_expired()
    assert removed > 0
    print(f"✅ cleanup_expired() removed {removed} entries")

    # Test stats
    s = cache.stats()
    print(f"\n📊 Stats: {s}")
    print(f"   {cache!r}")

    # Test clear
    cache.set("a", "1", "test")
    cache.set("b", "2", "test")
    cleared = cache.clear()
    assert cleared >= 2
    assert len(cache) == 0
    print(f"✅ clear() removed {cleared} entries")

    print("\n✅ All tests passed!")