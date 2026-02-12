"""
Build a unified LlamaIndex RAG index from:
  - Parsed .py formulation chunks
  - .lp file chunks
  - .mps file chunks
  - Optional index_mapping document (employee/project names -> indices)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llama_index.core import Document, VectorStoreIndex, StorageContext

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    _EMBED_MODEL = "openai"
except ImportError:
    _EMBED_MODEL = None

from agentic_explain.parsers.py_formulation_parser import parse_py_formulation
from agentic_explain.rag.lp_parser import parse_lp_file
from agentic_explain.rag.mps_parser import parse_mps_file


def _chunk_to_document(chunk: dict[str, Any]) -> Document:
    text = chunk.get("text", "")
    meta = chunk.get("metadata", {})
    return Document(text=text, metadata=meta)


def build_index_mapping_doc(data_dir: str | Path) -> Document:
    """Build a single document mapping entity names to indices (j, d, t) from ds_list and project_list."""
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
    text = "\n".join(lines)
    return Document(
        text=text,
        metadata={"source": "index_mapping", "section": "index_mapping", "path": str(data_dir)},
    )


def collect_raw_chunks(
    py_path: str | Path,
    lp_path: str | Path | None = None,
    mps_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Collect all raw chunks (before embedding) from all sources.

    Returns a flat list of dicts: { "text": str, "metadata": { "source": ..., "section": ..., ... } }
    Useful for inspecting what goes into the RAG index.
    """
    chunks: list[dict[str, Any]] = []
    chunks.extend(parse_py_formulation(py_path))
    if lp_path and Path(lp_path).exists():
        chunks.extend(parse_lp_file(lp_path))
    if mps_path and Path(mps_path).exists():
        chunks.extend(parse_mps_file(mps_path))
    if data_dir and Path(data_dir).exists():
        doc = build_index_mapping_doc(data_dir)
        chunks.append({"text": doc.text, "metadata": doc.metadata})
    return chunks


def build_rag_index(
    py_path: str | Path,
    lp_path: str | Path | None = None,
    mps_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    persist_dir: str | Path = "outputs/rag_index",
    embed_model: str = "default",
) -> VectorStoreIndex:
    """
    Build and persist a unified RAG index from .py, .lp, .mps, and optional index mapping.

    Parameters
    ----------
    py_path : path to formulation .py
    lp_path : path to model.lp (optional)
    mps_path : path to model.mps (optional)
    data_dir : path to data folder for index_mapping (optional)
    persist_dir : where to save the index
    embed_model : "default" (OpenAI if OPENAI_API_KEY set, else simple hash), or "openai"

    Returns
    -------
    VectorStoreIndex (also persisted to persist_dir)
    """
    documents: list[Document] = []

    # Parsed .py
    py_chunks = parse_py_formulation(py_path)
    for ch in py_chunks:
        documents.append(_chunk_to_document(ch))

    # .lp
    if lp_path and Path(lp_path).exists():
        for ch in parse_lp_file(lp_path):
            documents.append(_chunk_to_document(ch))

    # .mps
    if mps_path and Path(mps_path).exists():
        for ch in parse_mps_file(mps_path):
            documents.append(_chunk_to_document(ch))

    # Index mapping from data
    if data_dir and Path(data_dir).exists():
        documents.append(build_index_mapping_doc(data_dir))

    if not documents:
        raise ValueError("No documents to index. Provide at least py_path.")

    # Embedding model
    if embed_model == "openai" or (embed_model == "default" and _EMBED_MODEL == "openai"):
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
            embed = OpenAIEmbedding()
        except Exception:
            embed = None
    else:
        embed = None

    if embed is None:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from llama_index.embeddings.langchain import LangchainEmbedding
            embed = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )
        except ImportError:
            from llama_index.embeddings.openai import OpenAIEmbedding
            embed = OpenAIEmbedding()

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        documents, embed_model=embed, storage_context=storage_context
    )
    index.storage_context.persist(persist_dir=str(persist_path))

    # Re-write docstore and index_store with indentation for human readability.
    # Skip default__vector_store.json (too large — pretty-printing would 3-4x the size).
    for fname in ("docstore.json", "index_store.json"):
        fpath = persist_path / fname
        if fpath.exists():
            data = json.loads(fpath.read_text(encoding="utf-8"))
            fpath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return index


def load_rag_index(
    persist_dir: str | Path = "outputs/rag_index",
    embed_model: Any = None,
) -> VectorStoreIndex:
    """Load a persisted RAG index."""
    from llama_index.core import load_index_from_storage

    persist_dir = Path(persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(f"RAG index not found at {persist_dir}. Run build_rag_index first.")

    if embed_model is None:
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
            embed_model = OpenAIEmbedding()
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from llama_index.embeddings.langchain import LangchainEmbedding
            embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )

    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage_context, embed_model=embed_model)
