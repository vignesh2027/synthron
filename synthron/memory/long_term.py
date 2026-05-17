"""Long-term vector memory — ChromaDB local + Pinecone cloud option."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from synthron.utils.config import settings
from synthron.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaLongTermMemory:
    """Local long-term vector memory using ChromaDB.

    Stores text as embeddings for semantic similarity search.
    No API key required — runs 100% locally.
    """

    def __init__(self, collection_name: str = "synthron_memory") -> None:
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embed_fn = None

    async def initialize(self) -> bool:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = await asyncio.to_thread(
                chromadb.PersistentClient,
                path=settings.memory.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"[long_term] ChromaDB initialized: {self.collection_name} "
                f"({await self.count()} entries)"
            )
            return True
        except ImportError:
            logger.warning("[long_term] chromadb not installed. Run: pip install chromadb")
            return False
        except Exception as exc:
            logger.error(f"[long_term] ChromaDB init failed: {exc}")
            return False

    async def store(
        self,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store content as an embedding.

        Args:
            key: Unique identifier for this memory.
            content: Text content to embed and store.
            metadata: Optional metadata tags (task_type, agent, etc.)

        Returns:
            True on success.
        """
        if not self._collection:
            logger.warning("[long_term] ChromaDB not initialized, skipping store.")
            return False

        doc_id = hashlib.md5(key.encode()).hexdigest()
        meta = {
            "key": key,
            "ts": time.time(),
            "length": len(content),
            **(metadata or {}),
        }
        # Convert all metadata values to strings/numbers for ChromaDB
        meta = {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in meta.items()}

        try:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=[doc_id],
                documents=[content[:8000]],  # ChromaDB limit
                metadatas=[meta],
            )
            logger.debug(f"[long_term] Stored: '{key}' ({len(content)} chars)")
            return True
        except Exception as exc:
            logger.error(f"[long_term] Store failed: {exc}")
            return False

    async def recall(
        self, query: str, top_k: int = 5, filter_meta: dict | None = None
    ) -> list[dict[str, Any]]:
        """Semantic search for relevant memories.

        Args:
            query: Natural language query.
            top_k: Number of top results to return.
            filter_meta: Optional ChromaDB where-filter dict.

        Returns:
            List of dicts with 'content', 'key', 'score', 'metadata'.
        """
        if not self._collection:
            return []

        try:
            results = await asyncio.to_thread(
                self._collection.query,
                query_texts=[query],
                n_results=min(top_k, await self.count() or 1),
                where=filter_meta,
                include=["documents", "metadatas", "distances"],
            )

            output = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                output.append({
                    "content": doc,
                    "key": meta.get("key", ""),
                    "score": 1 - dist,  # cosine similarity
                    "metadata": meta,
                })

            return output
        except Exception as exc:
            logger.debug(f"[long_term] Recall failed: {exc}")
            return []

    async def delete(self, key: str) -> bool:
        """Delete a memory entry by key."""
        if not self._collection:
            return False
        doc_id = hashlib.md5(key.encode()).hexdigest()
        try:
            await asyncio.to_thread(self._collection.delete, ids=[doc_id])
            return True
        except Exception:
            return False

    async def count(self) -> int:
        """Return total number of stored memories."""
        if not self._collection:
            return 0
        try:
            return await asyncio.to_thread(self._collection.count)
        except Exception:
            return 0

    async def list_all(self, limit: int = 100) -> list[dict]:
        """List all stored memory entries."""
        if not self._collection:
            return []
        try:
            results = await asyncio.to_thread(
                self._collection.get,
                limit=limit,
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            return [
                {"content": d[:200], "metadata": m}
                for d, m in zip(docs, metas)
            ]
        except Exception:
            return []


class PineconeLongTermMemory:
    """Cloud long-term memory using Pinecone (optional).

    Provides infinite scale for enterprise deployments.
    Requires PINECONE_API_KEY environment variable.
    """

    def __init__(self, index_name: str = "") -> None:
        self.index_name = index_name or settings.memory.pinecone_index_name
        self._index = None
        self._embed_model = "text-embedding-3-small"

    async def initialize(self) -> bool:
        """Initialize Pinecone connection."""
        if not settings.memory.pinecone_api_key:
            logger.debug("[long_term] Pinecone API key not set, skipping.")
            return False
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.memory.pinecone_api_key)
            self._index = pc.Index(self.index_name)
            logger.info(f"[long_term] Pinecone connected: index='{self.index_name}'")
            return True
        except ImportError:
            logger.debug("[long_term] pinecone-client not installed.")
            return False
        except Exception as exc:
            logger.error(f"[long_term] Pinecone init failed: {exc}")
            return False

    async def store(self, key: str, content: str, metadata: dict | None = None) -> bool:
        """Store content vector in Pinecone."""
        if not self._index:
            return False
        try:
            vector = await self._embed(content)
            meta = {"content": content[:40_000], "key": key, **(metadata or {})}
            await asyncio.to_thread(
                self._index.upsert,
                vectors=[{"id": key, "values": vector, "metadata": meta}],
            )
            return True
        except Exception as exc:
            logger.error(f"[long_term/pinecone] Store failed: {exc}")
            return False

    async def recall(self, query: str, top_k: int = 5, **_) -> list[dict]:
        """Semantic search in Pinecone."""
        if not self._index:
            return []
        try:
            vector = await self._embed(query)
            results = await asyncio.to_thread(
                self._index.query, vector=vector, top_k=top_k, include_metadata=True
            )
            return [
                {
                    "content": m.metadata.get("content", ""),
                    "key": m.metadata.get("key", m.id),
                    "score": m.score,
                    "metadata": dict(m.metadata),
                }
                for m in results.matches
            ]
        except Exception as exc:
            logger.debug(f"[long_term/pinecone] Recall failed: {exc}")
            return []

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding using Gemini or fallback."""
        try:
            from synthron.utils.config import settings as s
            import google.generativeai as genai
            genai.configure(api_key=s.providers.gemini_api_key)
            result = await asyncio.to_thread(
                genai.embed_content,
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
            )
            return result["embedding"]
        except Exception:
            # Simple hash-based fallback (768 dims)
            import hashlib, struct
            h = hashlib.sha256(text.encode()).digest()
            return [struct.unpack("f", h[i:i+4])[0] for i in range(0, min(len(h), 768*4), 4)]
