"""Retriever module for TrustYojana.

Implements Experiment 5:
- Connects to persistent ChromaDB collection
- retrieve(query, k, scheme=None) returns top-k chunks with text, metadata, and similarity score
- Supports optional metadata filtering by scheme
- Provides helper to inspect indexed schemes and documents
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.utils import embedding_functions

from src.config import get_config

logger = logging.getLogger(__name__)

_retriever_instance: Optional["SchemeRetriever"] = None


class SchemeRetriever:
    """Retriever for querying ChromaDB with optional scheme filtering."""

    def __init__(self):
        self.config = get_config()
        self.chroma_dir = self.config.paths.chroma_db_path
        self.collection_name = self.config.paths.collection_name
        self.default_k = self.config.rag.top_k

        # Embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embeddings.model_name,
            device=self.config.embeddings.device,
        )

        # Connect to Chroma
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return total number of chunks indexed in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error reading collection count: {e}")
            return 0

    def get_indexed_schemes(self) -> List[Dict[str, Any]]:
        """Return a list of indexed schemes with document details and access dates."""
        try:
            all_data = self.collection.get(include=["metadatas"])
            metadatas = all_data.get("metadatas", [])
            schemes_dict: Dict[str, Dict[str, Any]] = {}

            for m in metadatas:
                if not m:
                    continue
                s_name = m.get("scheme", "Unknown")
                if s_name not in schemes_dict:
                    schemes_dict[s_name] = {
                        "scheme": s_name,
                        "title": m.get("title", ""),
                        "source_file": m.get("source_file", ""),
                        "url": m.get("url", ""),
                        "doc_date": m.get("doc_date", "unknown"),
                        "date_accessed": m.get("date_accessed", "unknown"),
                        "chunks_count": 0,
                    }
                schemes_dict[s_name]["chunks_count"] += 1

            return list(schemes_dict.values())
        except Exception as e:
            logger.error(f"Error fetching indexed schemes: {e}")
            return []

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        scheme: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks matching the query with optional scheme filtering.
        
        Returns a list of chunk dictionaries with text, metadata, and similarity score.
        """
        top_k = k if k is not None else self.default_k
        total_chunks = self.count()
        if total_chunks == 0:
            logger.warning("Retriever called but collection is empty.")
            return []

        # Bound k by total available chunks
        effective_k = min(top_k, total_chunks)

        where_filter = None
        if scheme and scheme.lower() not in ("all", "none", "unknown", "general", ""):
            # Check if scheme matches any indexed scheme
            where_filter = {"scheme": scheme}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=effective_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Filter query for scheme='{scheme}' failed: {e}. Retrying without filter.")
            # Fallback to unfiltered query if filter failed (e.g. scheme name mismatch)
            results = self.collection.query(
                query_texts=[query],
                n_results=effective_k,
                include=["documents", "metadatas", "distances"],
            )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks: List[Dict[str, Any]] = []
        for doc_text, meta, dist in zip(documents, metadatas, distances):
            # For cosine distance in chromadb, similarity score = 1.0 - distance
            similarity = round(max(0.0, 1.0 - float(dist)), 4)
            chunks.append({
                "text": doc_text,
                "metadata": meta,
                "score": similarity,
                "scheme": meta.get("scheme", "Unknown"),
                "title": meta.get("title", ""),
                "source_file": meta.get("source_file", ""),
                "page": meta.get("page", 1),
                "url": meta.get("url", ""),
                "doc_date": meta.get("doc_date", "unknown"),
                "date_accessed": meta.get("date_accessed", "unknown"),
                "chunk_id": meta.get("chunk_id", ""),
            })

        return chunks


def get_retriever() -> SchemeRetriever:
    """Singleton getter for SchemeRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = SchemeRetriever()
    return _retriever_instance


def retrieve(
    query: str,
    k: Optional[int] = None,
    scheme: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Module-level retrieve function for Experiment 5."""
    return get_retriever().retrieve(query=query, k=k, scheme=scheme)
