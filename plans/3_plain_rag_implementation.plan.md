---
name: Plain RAG Implementation
overview: Specification for the Plain RAG strategy that wraps the existing LlamaIndex VectorStoreIndex and implements the RAGStrategy protocol. This is Section 2 of the Graph RAG Comparison Framework plan.
isProject: false
---

# Plain RAG Strategy (Implementation Spec)

## Purpose

The Plain RAG strategy is the baseline retrieval approach: it uses the existing LlamaIndex `VectorStoreIndex` built from formulation documents (.py, .lp, .mps) and index mapping. It implements the common `RAGStrategy` protocol so the workflow can call `strategy.retrieve(query, top_k)` regardless of backend.

## Implementation

Create `agentic_explain/rag/plain_rag.py`:

1. **Wrap VectorStoreIndex in RAGStrategy protocol**
   - Class `PlainRAGStrategy` implements `RAGStrategy`.
   - Attribute: holds a `VectorStoreIndex` instance (built/loaded via existing `build_index.py`).

2. **`retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]`**
   - Call `index.as_retriever(similarity_top_k=top_k).retrieve(query)`.
   - Convert each LlamaIndex node to `RetrievedChunk(text=node.text, score=node.score, metadata=node.metadata)`.
   - Return the list of `RetrievedChunk`.

3. **`name` property**
   - Return `"plain_rag"` for logging and debug traceability.

4. **Factory: `build_plain_rag(py_path, lp_path=None, mps_path=None, data_dir=None, persist_dir="outputs/rag_index")`**
   - If index exists at `persist_dir` (via `rag_index_exists()` or similar), call `load_rag_index(persist_dir)` and wrap in `PlainRAGStrategy`.
   - Otherwise call `build_rag_index(py_path, lp_path, mps_path, data_dir, persist_dir)` then wrap.
   - Return a `PlainRAGStrategy` instance.

## Dependencies

- Reuses `agentic_explain/rag/build_index.py`: `build_rag_index`, `load_rag_index`, and (if present) `rag_index_exists`.
- Depends on `agentic_explain/rag/strategy.py`: `RetrievedChunk` dataclass and `RAGStrategy` protocol.

## Integration

- Workflow nodes (`make_constraint_generation_node`, `make_ilp_analysis_node`) receive `rag_strategy: RAGStrategy`. When using Plain RAG, pass `build_plain_rag(...)` as the strategy.
- Debug capture: `rag_retrieval_debug` stores chunks with `text`, `score`, `metadata` (already compatible with `RetrievedChunk`); also store `rag_strategy.name` (e.g. `"plain_rag"`).
