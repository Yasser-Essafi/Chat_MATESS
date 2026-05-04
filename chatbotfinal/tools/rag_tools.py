"""
STATOUR RAG Tools — Fixed
===========================
Loads markdown knowledge documents into ChromaDB vector store
and provides semantic search retrieval.

Fixes from original:
- Relevance threshold filtering (no more garbage results)
- Error handling around all ChromaDB operations
- Integrated with shared logger and cache
- Python 3.8+ compatible type hints
- Graceful degradation when ChromaDB unavailable

Usage:
    from tools.rag_tools import RAGManager

    rag = RAGManager()
    rag.build_vectorstore()           # Run once to index documents
    results = rag.search("tourisme")  # Search anytime
    formatted = rag.search_formatted("tourisme Maroc")
    stats = rag.get_stats()
"""

import os
import sys
import glob
import hashlib
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    RAG_RELEVANCE_THRESHOLD,
)
from utils.logger import get_logger
from utils.cache import shared_cache

logger = get_logger("statour.rag")

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.error(
        "chromadb package not installed. Run: pip install chromadb"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

COLLECTION_NAME = "statour_knowledge"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
BATCH_SIZE = 500           # ChromaDB add batch limit
CACHE_SOURCE_RAG = "rag"
WORDS_PER_CHAR_ESTIMATE = 5  # Rough: 5 chars per word


# ══════════════════════════════════════════════════════════════════════════════
# Custom Exceptions
# ══════════════════════════════════════════════════════════════════════════════

class RAGError(Exception):
    """Raised when RAG operations fail."""
    pass


class RAGBuildError(RAGError):
    """Raised when vectorstore building fails."""
    pass


class RAGSearchError(RAGError):
    """Raised when search operations fail."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Text Splitter
# ══════════════════════════════════════════════════════════════════════════════

class MarkdownSplitter:
    """
    Splits markdown text into chunks by headers and size.
    Each chunk preserves its section header for context.

    Features:
        - Splits at ## and # headers first
        - Long sections are further split by size with overlap
        - Each chunk carries metadata (source file, section title)
        - Overlap ensures context continuity between chunks

    Args:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, source: str = "") -> List[Dict]:
        """
        Split markdown text into chunks with metadata.

        Args:
            text: Full markdown text content.
            source: Source filename for metadata.

        Returns:
            List of dicts, each with:
                - text (str): Chunk content with section header
                - metadata (dict): source, section
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to splitter (source: %s)", source)
            return []

        chunks = []
        sections = self._split_by_headers(text)

        for section_title, section_text in sections:
            # Skip empty sections
            if not section_text.strip():
                continue

            if len(section_text) <= self.chunk_size:
                # Section fits in one chunk
                chunk_text = self._format_chunk(section_title, section_text)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "section": section_title or "Introduction",
                    }
                })
            else:
                # Section needs to be split into sub-chunks
                sub_chunks = self._split_by_size(section_text)
                for i, sub in enumerate(sub_chunks):
                    part_label = f" (partie {i + 1}/{len(sub_chunks)})"
                    section_label = (section_title or "Introduction") + part_label

                    chunk_text = self._format_chunk(
                        section_title + part_label if section_title else None,
                        sub,
                    )
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "source": source,
                            "section": section_label,
                        }
                    })

        logger.debug(
            "Split '%s' into %d chunks (avg %.0f chars)",
            source,
            len(chunks),
            sum(len(c["text"]) for c in chunks) / max(len(chunks), 1),
        )

        return chunks

    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text at markdown headers (# and ##).

        Returns:
            List of (title, content) tuples.
        """
        lines = text.split("\n")
        sections = []
        current_title = ""
        current_lines = []

        for line in lines:
            stripped = line.strip()

            # Detect ## headers (level 2)
            if stripped.startswith("## ") and not stripped.startswith("###"):
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    sections.append((current_title, content))
                current_title = stripped.lstrip("# ").strip()
                current_lines = []

            # Detect # headers (level 1)
            elif stripped.startswith("# ") and not stripped.startswith("##"):
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    sections.append((current_title, content))
                current_title = stripped.lstrip("# ").strip()
                current_lines = []

            else:
                current_lines.append(line)

        # Don't forget the last section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_title, content))

        # If no headers found, return entire text as one section
        if not sections and text.strip():
            sections.append(("", text.strip()))

        return sections

    def _split_by_size(self, text: str) -> List[str]:
        """
        Split long text into sized chunks with overlap.
        Tries to split at word boundaries.

        Returns:
            List of text chunks.
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        approx_words = self.chunk_size // WORDS_PER_CHAR_ESTIMATE
        overlap_words = self.chunk_overlap // WORDS_PER_CHAR_ESTIMATE

        # Ensure minimum values
        approx_words = max(approx_words, 10)
        overlap_words = max(overlap_words, 0)

        start = 0
        while start < len(words):
            end = min(start + approx_words, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            # Move start forward, accounting for overlap
            next_start = end - overlap_words
            if next_start <= start:
                # Prevent infinite loop
                next_start = start + max(approx_words // 2, 1)
            start = next_start

        return chunks

    @staticmethod
    def _format_chunk(title: Optional[str], content: str) -> str:
        """Format a chunk with optional section header."""
        if title:
            return f"## {title}\n\n{content}"
        return content


# ══════════════════════════════════════════════════════════════════════════════
# RAG Manager
# ══════════════════════════════════════════════════════════════════════════════

class RAGManager:
    """
    Manages the ChromaDB vector store for STATOUR knowledge documents.
    Uses ChromaDB's built-in embedding model (default: all-MiniLM-L6-v2).

    Features:
        - Auto-loads and indexes markdown documents
        - Semantic search with relevance threshold filtering
        - Result caching via shared SearchCache
        - Graceful degradation when ChromaDB is unavailable
        - Formatted output for LLM context injection
        - Build/rebuild with progress logging

    Args:
        chunk_size: Target chunk size for text splitting.
        chunk_overlap: Overlap between chunks.
        relevance_threshold: Maximum distance for relevant results
            (lower = more relevant in ChromaDB's distance metric).
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        relevance_threshold: float = RAG_RELEVANCE_THRESHOLD,
    ):
        self.relevance_threshold = relevance_threshold
        self.splitter = MarkdownSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._collection = None
        self._client = None
        self._available = False

        # Initialize ChromaDB client
        self._init_chromadb()

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client with error handling."""
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available — RAG will return empty results")
            return

        try:
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=VECTORSTORE_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._available = True
            logger.info(
                "ChromaDB initialized (path: %s)",
                VECTORSTORE_DIR,
            )
        except Exception as e:
            logger.error(
                "ChromaDB initialization failed: %s — RAG will return empty results",
                str(e)[:200],
            )
            self._available = False

    @property
    def collection(self):
        """
        Lazy-load the ChromaDB collection.
        Creates it if it doesn't exist.

        Returns:
            ChromaDB collection, or None if unavailable.
        """
        if not self._available or self._client is None:
            return None

        if self._collection is None:
            try:
                self._collection = self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"description": "STATOUR tourism knowledge base"},
                )
                logger.debug(
                    "Collection '%s' loaded (%d documents)",
                    COLLECTION_NAME,
                    self._collection.count(),
                )
            except Exception as e:
                logger.error("Failed to load collection: %s", e)
                return None

        return self._collection

    @property
    def is_available(self) -> bool:
        """Check if RAG system is operational."""
        return self._available and self.collection is not None

    # ──────────────────────────────────────────────────────────────────────
    # Build / Rebuild
    # ──────────────────────────────────────────────────────────────────────

    def build_vectorstore(self, force_rebuild: bool = False) -> Dict:
        """
        Load all .md files from DOCUMENTS_DIR, split, embed, and store.

        Args:
            force_rebuild: If True, delete existing collection and rebuild.
                           If False, skip if collection already has documents.

        Returns:
            Dict with keys: status, message, files_processed, total_chunks.

        Raises:
            RAGBuildError: If building fails critically.
        """
        if not self._available:
            return {
                "status": "error",
                "message": "ChromaDB not available",
                "files_processed": 0,
                "total_chunks": 0,
            }

        # ── Find markdown files ──
        if not os.path.isdir(DOCUMENTS_DIR):
            return {
                "status": "error",
                "message": f"Documents directory not found: {DOCUMENTS_DIR}",
                "files_processed": 0,
                "total_chunks": 0,
            }

        md_files = sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.md")))

        if not md_files:
            return {
                "status": "error",
                "message": f"No .md files found in {DOCUMENTS_DIR}",
                "files_processed": 0,
                "total_chunks": 0,
            }

        logger.info("Found %d markdown files in %s", len(md_files), DOCUMENTS_DIR)

        # ── Check if already built ──
        try:
            existing_count = self.collection.count() if self.collection else 0
        except Exception:
            existing_count = 0

        if existing_count > 0 and not force_rebuild:
            msg = (
                f"Vector store already has {existing_count} chunks. "
                f"Use force_rebuild=True to rebuild."
            )
            logger.info(msg)
            return {
                "status": "skipped",
                "message": msg,
                "files_processed": 0,
                "total_chunks": existing_count,
            }

        # ── Clear existing if rebuilding ──
        if existing_count > 0 and force_rebuild:
            try:
                self._client.delete_collection(COLLECTION_NAME)
                self._collection = None  # Reset lazy reference
                logger.info(
                    "Deleted existing collection (%d chunks)", existing_count
                )
            except Exception as e:
                logger.warning("Failed to delete existing collection: %s", e)
                self._collection = None

        # ── Process all files ──
        all_ids = []
        all_texts = []
        all_metadatas = []
        files_processed = 0
        files_failed = 0

        for fpath in md_files:
            fname = os.path.basename(fpath)

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()

                if not content.strip():
                    logger.warning("Skipping empty file: %s", fname)
                    continue

                chunks = self.splitter.split(content, source=fname)

                if not chunks:
                    logger.warning("No chunks produced from: %s", fname)
                    continue

                for i, chunk in enumerate(chunks):
                    # Deterministic ID based on file + index + content prefix
                    chunk_id = hashlib.md5(
                        f"{fname}_{i}_{chunk['text'][:50]}".encode("utf-8")
                    ).hexdigest()

                    all_ids.append(chunk_id)
                    all_texts.append(chunk["text"])
                    all_metadatas.append(chunk["metadata"])

                files_processed += 1
                logger.info("  📝 %s: %d chunks", fname, len(chunks))

            except UnicodeDecodeError:
                # Try with latin-1 encoding
                try:
                    with open(fpath, "r", encoding="latin-1") as f:
                        content = f.read()
                    chunks = self.splitter.split(content, source=fname)
                    for i, chunk in enumerate(chunks):
                        chunk_id = hashlib.md5(
                            f"{fname}_{i}_{chunk['text'][:50]}".encode("utf-8")
                        ).hexdigest()
                        all_ids.append(chunk_id)
                        all_texts.append(chunk["text"])
                        all_metadatas.append(chunk["metadata"])
                    files_processed += 1
                    logger.info("  📝 %s: %d chunks (latin-1)", fname, len(chunks))
                except Exception as e2:
                    files_failed += 1
                    logger.error("  ❌ %s: encoding error: %s", fname, e2)

            except Exception as e:
                files_failed += 1
                logger.error("  ❌ %s: %s", fname, str(e)[:100])

        if not all_ids:
            return {
                "status": "error",
                "message": "No chunks were produced from any file",
                "files_processed": 0,
                "total_chunks": 0,
            }

        # ── Add to ChromaDB in batches ──
        total_added = 0
        try:
            for i in range(0, len(all_ids), BATCH_SIZE):
                end = min(i + BATCH_SIZE, len(all_ids))
                self.collection.add(
                    ids=all_ids[i:end],
                    documents=all_texts[i:end],
                    metadatas=all_metadatas[i:end],
                )
                batch_count = end - i
                total_added += batch_count
                logger.debug("  Added batch: %d chunks (%d/%d)", batch_count, total_added, len(all_ids))

        except Exception as e:
            logger.error("Failed to add chunks to ChromaDB: %s", e)
            return {
                "status": "partial",
                "message": f"Added {total_added}/{len(all_ids)} chunks before error: {e}",
                "files_processed": files_processed,
                "total_chunks": total_added,
            }

        # ── Summary ──
        msg = (
            f"Indexed {total_added} chunks from {files_processed} files"
            f"{f' ({files_failed} failed)' if files_failed else ''}"
        )
        logger.info("✅ %s", msg)

        return {
            "status": "success",
            "message": msg,
            "files_processed": files_processed,
            "files_failed": files_failed,
            "total_chunks": total_added,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        use_cache: bool = True,
    ) -> List[Dict]:
        """
        Semantic search across the knowledge base.

        Features:
            - Relevance threshold filtering (removes low-quality results)
            - Result caching via shared SearchCache
            - Graceful degradation when ChromaDB unavailable

        Args:
            query: Search query in any language.
            n_results: Maximum number of results to return.
            use_cache: If True, check/store in shared cache.

        Returns:
            List of dicts, each with:
                - text (str): Full chunk text
                - content (str): Alias for text (compatibility)
                - source (str): Source filename
                - section (str): Section title
                - distance (float): ChromaDB distance (lower = more relevant)

            Returns empty list if no relevant results or RAG unavailable.
        """
        if not query or not query.strip():
            return []

        query = query.strip()

        # ── Check availability ──
        if not self.is_available:
            logger.debug("RAG unavailable — returning empty results")
            return []

        # ── Check cache ──
        cache_key = f"rag:{n_results}:{query}"
        if use_cache:
            cached = shared_cache.get(cache_key, source=CACHE_SOURCE_RAG)
            if cached is not None:
                # Cache stores formatted string, not raw results
                # For raw results, we skip cache (only formatted uses it)
                pass

        # ── Query ChromaDB ──
        try:
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.debug("Collection is empty — no results")
                return []

            actual_n = min(n_results, collection_count)

            results = self.collection.query(
                query_texts=[query],
                n_results=actual_n,
            )
        except Exception as e:
            logger.error(
                "ChromaDB query failed for '%s': %s",
                query[:50], str(e)[:200],
            )
            return []

        # ── Parse and filter results ──
        output = []

        if not results or not results.get("documents"):
            return []

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, doc in enumerate(documents):
            # Get distance (lower = more relevant)
            dist = distances[i] if i < len(distances) else None

            # ── FIX: Relevance threshold filtering ──
            # Skip results that are too far from the query
            if dist is not None and dist > self.relevance_threshold:
                logger.debug(
                    "Skipping irrelevant result (distance=%.4f > threshold=%.4f): %s",
                    dist, self.relevance_threshold, doc[:60],
                )
                continue

            # Get metadata
            meta = metadatas[i] if i < len(metadatas) else {}

            output.append({
                "text": doc,
                "content": doc,  # Alias for compatibility with researcher agent
                "source": meta.get("source", "unknown"),
                "section": meta.get("section", "unknown"),
                "distance": round(dist, 4) if dist is not None else None,
            })

        logger.debug(
            "RAG search for '%s': %d/%d results passed relevance filter (threshold=%.2f)",
            query[:50], len(output), len(documents), self.relevance_threshold,
        )

        return output

    def search_formatted(
        self,
        query: str,
        n_results: int = 5,
        use_cache: bool = True,
    ) -> str:
        """
        Search and return results as formatted text for LLM context.

        Args:
            query: Search query.
            n_results: Maximum number of results.
            use_cache: If True, check/store in shared cache.

        Returns:
            Formatted string with search results.
        """
        # ── Check cache ──
        cache_key = f"rag_fmt:{n_results}:{query}"
        if use_cache:
            cached = shared_cache.get(cache_key, source=CACHE_SOURCE_RAG)
            if cached is not None:
                logger.debug("Returning cached RAG formatted results for '%s'", query[:50])
                return cached

        # ── Execute search ──
        results = self.search(query, n_results, use_cache=False)

        if not results:
            return "Aucun résultat trouvé dans la base de connaissances."

        # ── Format ──
        parts = [
            f"📚 **{len(results)} résultats trouvés dans la base "
            f"de connaissances STATOUR :**\n"
        ]

        for i, r in enumerate(results, 1):
            source = r.get("source", "inconnu")
            section = r.get("section", "inconnue")
            distance = r.get("distance")
            text = r.get("text", "")

            distance_str = f"{distance:.4f}" if distance is not None else "N/A"

            parts.append(
                f"---\n"
                f"**Résultat {i}** "
                f"(source: `{source}` | section: {section} | "
                f"pertinence: {distance_str})\n\n"
                f"{text}\n"
            )

        formatted = "\n".join(parts)

        # ── Store in cache ──
        if use_cache:
            shared_cache.set(cache_key, formatted, source=CACHE_SOURCE_RAG)

        return formatted

    # ──────────────────────────────────────────────────────────────────────
    # Info & Stats
    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """
        Return stats about the current vector store.

        Returns:
            Dict with collection name, chunk count, paths, availability status.
        """
        chunk_count = 0

        if self.is_available:
            try:
                chunk_count = self.collection.count()
            except Exception as e:
                logger.warning("Failed to get collection count: %s", e)

        return {
            "collection": COLLECTION_NAME,
            "total_chunks": chunk_count,
            "vectorstore_path": VECTORSTORE_DIR,
            "documents_path": DOCUMENTS_DIR,
            "available": self._available,
            "relevance_threshold": self.relevance_threshold,
        }

    def list_sources(self) -> List[str]:
        """
        List all unique source files in the knowledge base.

        Returns:
            Sorted list of source filenames.
        """
        if not self.is_available:
            return []

        try:
            # Get all metadata
            all_data = self.collection.get()
            if all_data and all_data.get("metadatas"):
                sources = set()
                for meta in all_data["metadatas"]:
                    if meta and "source" in meta:
                        sources.add(meta["source"])
                return sorted(sources)
        except Exception as e:
            logger.warning("Failed to list sources: %s", e)

        return []

    def get_document_count(self) -> int:
        """Return the number of unique source documents."""
        return len(self.list_sources())


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("📚 STATOUR — RAG Vector Store Manager")
    print("=" * 60)
    print()

    rag = RAGManager()

    # ── Stats ──
    stats = rag.get_stats()
    print(f"📊 Current stats:")
    print(f"   Collection: {stats['collection']}")
    print(f"   Chunks: {stats['total_chunks']}")
    print(f"   Available: {stats['available']}")
    print(f"   Threshold: {stats['relevance_threshold']}")
    print(f"   Vectorstore: {stats['vectorstore_path']}")
    print(f"   Documents: {stats['documents_path']}")
    print()

    # ── Build ──
    print("🔨 Building vector store...")
    print()
    result = rag.build_vectorstore(force_rebuild=True)
    print(f"\n📋 Result: {result['status']}")
    print(f"   {result['message']}")
    print()

    # ── Sources ──
    sources = rag.list_sources()
    if sources:
        print(f"📁 Sources ({len(sources)}):")
        for s in sources:
            print(f"   • {s}")
        print()

    # ── Test searches ──
    test_queries = [
        "Combien de touristes français visitent le Maroc ?",
        "Quels sont les principaux aéroports du Maroc ?",
        "Qu'est-ce que l'Opération Marhaba ?",
        "Définition KPI tourisme",
        "STATOUR plateforme architecture",
        "Vision 2030 tourisme Maroc",
    ]

    print("=" * 60)
    print("🔍 Search Tests")
    print("=" * 60)
    print()

    for q in test_queries:
        print(f"❓ {q}")
        results = rag.search(q, n_results=3)
        if not results:
            print("   ❌ No relevant results")
        else:
            for r in results:
                print(
                    f"   → [{r['source']}] {r['section']} "
                    f"(distance: {r['distance']})"
                )
        print()

    # ── Test formatted output ──
    print("=" * 60)
    print("📋 Formatted Output Test")
    print("=" * 60)
    print()

    formatted = rag.search_formatted("tourisme Maroc statistiques", n_results=2)
    print(formatted)
    print()

    # ── Test cache ──
    print("=" * 60)
    print("🗄️  Cache Test")
    print("=" * 60)
    print()

    # Second call should hit cache
    formatted2 = rag.search_formatted("tourisme Maroc statistiques", n_results=2)
    cache_stats = shared_cache.stats()
    print(f"   Cache hits: {cache_stats['hits']}")
    print(f"   Cache misses: {cache_stats['misses']}")
    print(f"   Hit rate: {cache_stats['hit_rate_pct']}%")
    print()

    # ── Test edge cases ──
    print("=" * 60)
    print("🧪 Edge Case Tests")
    print("=" * 60)
    print()

    # Empty query
    empty = rag.search("")
    assert empty == [], f"Expected empty list for empty query, got {empty}"
    print("   ✅ Empty query returns empty list")

    # Whitespace query
    ws = rag.search("   ")
    assert ws == [], f"Expected empty list for whitespace query, got {ws}"
    print("   ✅ Whitespace query returns empty list")

    # Query with no matches (should return empty after threshold filter)
    garbage = rag.search("xyzzy foobar baz quantum entanglement", n_results=2)
    print(f"   ✅ Irrelevant query: {len(garbage)} results (filtered by threshold)")

    print()
    print("✅ All tests complete!")


if __name__ == "__main__":
    main()