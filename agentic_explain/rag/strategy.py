"""
RAG strategy protocol: common interface for Plain RAG, Graph RAG, and No-RAG retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class RetrievedChunk:
    """A single retrieved context chunk with optional score and metadata."""

    text: str
    score: float | None = None
    metadata: dict[str, Any] | None = None


@runtime_checkable
class RAGStrategy(Protocol):
    """Protocol for RAG retrieval. All strategies (plain, graph, no-RAG) implement this."""

    @property
    def name(self) -> str:
        """Strategy identifier for logging (e.g. 'plain_rag', 'graph_rag', 'no_rag')."""
        ...

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return top-k retrieved chunks for the query, with scores and metadata."""
        ...
