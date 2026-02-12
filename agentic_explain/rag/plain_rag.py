"""
Plain RAG strategy: wraps LlamaIndex VectorStoreIndex in the RAGStrategy protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_explain.rag.strategy import RetrievedChunk, RAGStrategy
from agentic_explain.rag.build_index import build_rag_index, load_rag_index


def _rag_index_exists(persist_dir: str | Path) -> bool:
    """Return True if a persisted RAG index exists at persist_dir."""
    p = Path(persist_dir)
    return (p / "docstore.json").exists() and (p / "default__vector_store.json").exists()


class PlainRAGStrategy:
    """RAG strategy that uses the existing VectorStoreIndex (semantic search on formulation chunks)."""

    def __init__(self, index: Any) -> None:
        self._index = index

    @property
    def name(self) -> str:
        return "plain_rag"

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        return [
            RetrievedChunk(
                text=n.text,
                score=float(n.score) if n.score is not None else None,
                metadata=dict(n.metadata) if n.metadata else None,
            )
            for n in nodes
        ]


def build_plain_rag(
    py_path: str | Path,
    lp_path: str | Path | None = None,
    mps_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    persist_dir: str | Path = "outputs/rag_index",
    *,
    force_rebuild: bool = False,
) -> PlainRAGStrategy:
    """
    Build or load the plain RAG index and return a PlainRAGStrategy.

    If persist_dir already contains a valid index and force_rebuild is False, loads it.
    Otherwise builds from py_path, lp_path, mps_path, data_dir and persists to persist_dir.
    """
    persist_dir = Path(persist_dir)
    if not force_rebuild and _rag_index_exists(persist_dir):
        index = load_rag_index(persist_dir=persist_dir)
    else:
        index = build_rag_index(
            py_path=py_path,
            lp_path=lp_path,
            mps_path=mps_path,
            data_dir=data_dir,
            persist_dir=persist_dir,
        )
    return PlainRAGStrategy(index)
