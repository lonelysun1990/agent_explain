"""
No-RAG strategy: returns the full FORMULATION_DOCS and index mapping as context (no retrieval).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_explain.rag.strategy import RetrievedChunk, RAGStrategy


def _build_index_mapping_text(data_dir: str | Path) -> str:
    """Build index mapping text from ds_list and project_list."""
    data_dir = Path(data_dir)
    with open(data_dir / "ds_list.json", "r", encoding="utf-8") as f:
        ds_list = json.load(f)
    with open(data_dir / "project_list.json", "r", encoding="utf-8") as f:
        project_list = json.load(f)
    lines = [
        "Index mapping (same model in .py, .lp, .mps):",
        "j = employee index (0 to n_employees-1):",
    ]
    for idx, ds in enumerate(ds_list):
        name = ds.get("name", "")
        title = ds.get("title", "")
        lines.append(f"  j={idx}: {name} ({title})")
    lines.append("d = project index (0 to n_projects-1):")
    for idx, p in enumerate(project_list):
        name = p.get("name", "")
        lines.append(f"  d={idx}: {name}")
    lines.append("t = week index (0 to horizon-1). Use 'Week N' in queries for t=N.")
    return "\n".join(lines)


class NoRAGStrategy:
    """RAG strategy that returns full formulation docs + index mapping (no retrieval)."""

    def __init__(self, formulation_docs: str, index_mapping_text: str) -> None:
        self._formulation_docs = formulation_docs
        self._index_mapping_text = index_mapping_text
        self._full_context = f"{formulation_docs}\n\n---\n{index_mapping_text}"

    @property
    def name(self) -> str:
        return "no_rag"

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        del query, top_k
        return [
            RetrievedChunk(
                text=self._full_context,
                score=1.0,
                metadata={"source": "no_rag", "strategy": "full_context"},
            )
        ]


def build_no_rag(
    py_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    *,
    formulation_docs: str | None = None,
) -> NoRAGStrategy:
    """
    Build a NoRAGStrategy with full FORMULATION_DOCS and index mapping.

    Either pass formulation_docs directly, or pass py_path to load from staffing_model.
    data_dir is required for index mapping (ds_list.json, project_list.json).
    """
    if formulation_docs is None:
        if py_path is None:
            raise ValueError("Provide formulation_docs or py_path to load FORMULATION_DOCS.")
        # Load from use-case formulation (e.g. use_case.staffing_model)
        from use_case.staffing_model import FORMULATION_DOCS
        formulation_docs = FORMULATION_DOCS
    if data_dir is None:
        raise ValueError("data_dir is required for index mapping.")
    index_mapping_text = _build_index_mapping_text(data_dir)
    return NoRAGStrategy(formulation_docs, index_mapping_text)
