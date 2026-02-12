"""
Graph RAG strategy: two-level hierarchical graph (type nodes + entity-group nodes) with 1-hop expansion retrieval.

All use-case-specific logic (entity names, objective/variable mapping, constraint grouping)
is provided via GraphRAGConfig; this module stays generic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agentic_explain.rag.strategy import RetrievedChunk, RAGStrategy
from agentic_explain.rag.lp_parser import (
    extract_constraint_instances,
    extract_objective_variable_families,
)
from agentic_explain.rag.graph_rag_config import GraphRAGConfig


def _parse_formulation_sections(docs: str) -> dict[str, dict[str, str]]:
    """Parse FORMULATION_DOCS into variables, constraints, objectives subsection text."""
    sections: dict[str, dict[str, str]] = {"variables": {}, "constraints": {}, "objectives": {}}
    current_top = None
    current_sub = None
    buffer: list[str] = []
    for line in docs.split("\n"):
        if line.startswith("## VARIABLES"):
            if buffer and current_top and current_sub:
                sections[current_top][current_sub] = "\n".join(buffer).strip()
            current_top = "variables"
            current_sub = None
            buffer = []
            continue
        if line.startswith("## CONSTRAINTS"):
            if buffer and current_top and current_sub:
                sections[current_top][current_sub] = "\n".join(buffer).strip()
            current_top = "constraints"
            current_sub = None
            buffer = []
            continue
        if line.startswith("## OBJECTIVES"):
            if buffer and current_top and current_sub:
                sections[current_top][current_sub] = "\n".join(buffer).strip()
            current_top = "objectives"
            current_sub = None
            buffer = []
            continue
        if line.startswith("### "):
            if buffer and current_top and current_sub:
                sections[current_top][current_sub] = "\n".join(buffer).strip()
            current_sub = line[4:].strip().split()[0]
            buffer = []
            continue
        if current_top and current_sub is not None:
            buffer.append(line)
    if buffer and current_top and current_sub:
        sections[current_top][current_sub] = "\n".join(buffer).strip()
    return sections


def _build_hierarchical_graph(
    config: GraphRAGConfig,
    formulation_docs: str,
    constraint_instances: dict[str, list[dict[str, Any]]],
    objective_var_families: set[str],
) -> tuple[Any, dict[str, str]]:
    """Build networkx graph and node_id -> text mapping. Returns (graph, node_texts)."""
    import networkx as nx

    sections = _parse_formulation_sections(formulation_docs)
    G = nx.Graph()
    node_texts: dict[str, str] = {}

    # ----- Level 1: type nodes -----
    var_families = list(sections.get("variables", {}).keys())
    for v in var_families:
        nid = f"var:{v}"
        G.add_node(nid, level=1, kind="variable")
        node_texts[nid] = sections["variables"].get(v, f"Variable {v}.")

    constraint_types = list(sections.get("constraints", {}).keys())
    for c in constraint_types:
        nid = f"constr:{c}"
        G.add_node(nid, level=1, kind="constraint")
        node_texts[nid] = sections["constraints"].get(c, f"Constraint {c}.")

    obj_terms = list(sections.get("objectives", {}).keys())
    for o in obj_terms:
        nid = f"obj:{o}"
        G.add_node(nid, level=1, kind="objective")
        node_texts[nid] = sections["objectives"].get(o, f"Objective {o}.")

    # Type-level edges (var <-> constr, var <-> obj)
    for ctype, instances in constraint_instances.items():
        if not instances:
            continue
        all_families = set()
        for inst in instances:
            all_families |= inst.get("variable_families", set())
        constr_nid = f"constr:{ctype}"
        if not G.has_node(constr_nid):
            continue
        for vf in all_families:
            var_nid = f"var:{vf}"
            if G.has_node(var_nid):
                G.add_edge(var_nid, constr_nid)

    obj_var_map = config.get_objective_variable_map()
    for o in obj_terms:
        for vf in obj_var_map.get(o, objective_var_families):
            var_nid = f"var:{vf}"
            if G.has_node(var_nid):
                G.add_edge(var_nid, f"obj:{o}")

    # ----- Level 2: entity-group nodes -----
    l2_by_entity: dict[tuple[str, int], list[str]] = {}
    for ctype, instances in constraint_instances.items():
        type_desc = sections["constraints"].get(ctype, ctype)
        grouped: dict[tuple[str, int], list[dict]] = {}
        for inst in instances:
            ent = config.constraint_name_to_entity(ctype, inst["name"])
            if ent is None:
                continue
            grouped.setdefault(ent, []).append(inst)
        for (entity_type, eid), group in grouped.items():
            name = config.get_entity_name(entity_type, eid)
            nid = f"constr:{ctype}:{entity_type}_{eid}:{name}"
            G.add_node(nid, level=2, kind="constraint", entity_type=entity_type, entity_id=eid, entity_name=name)
            sample = group[0].get("sample", group[0].get("expr", "")) if group else ""
            text = (
                f"Entity: {ctype} for {entity_type} {eid} ({name})\n"
                f"Type: {type_desc[:300]}\n"
                f"Instance count: {len(group)}\n"
                f"Sample: {sample}"
            )
            node_texts[nid] = text
            G.add_edge(nid, f"constr:{ctype}")
            l2_by_entity.setdefault((entity_type, eid), []).append(nid)

    # Variable L2 nodes (by entity scope from config)
    for vf in var_families:
        scope = config.get_variable_entity_scope(vf)
        n_entities = config.get_entity_count(scope)
        for eid in range(n_entities):
            name = config.get_entity_name(scope, eid)
            nid = f"var:{vf}:{scope}_{eid}:{name}"
            G.add_node(nid, level=2, kind="variable", entity_type=scope, entity_id=eid, entity_name=name)
            node_texts[nid] = f"Variable {vf} for {scope} {eid} ({name}). {sections['variables'].get(vf, '')[:200]}"
            G.add_edge(nid, f"var:{vf}")
            l2_by_entity.setdefault((scope, eid), []).append(nid)

    # Sibling edges (L2 <-> L2 same entity)
    for node_ids in l2_by_entity.values():
        for i, nid1 in enumerate(node_ids):
            for nid2 in node_ids[i + 1 :]:
                G.add_edge(nid1, nid2)

    return G, node_texts


def _embed_graph_nodes(
    node_texts: dict[str, str],
    embed_model: Any,
) -> dict[str, np.ndarray]:
    """Embed each node's text; return node_id -> vector."""
    node_ids = list(node_texts.keys())
    texts = [node_texts[nid] for nid in node_ids]
    if hasattr(embed_model, "get_text_embedding_batch"):
        vectors = embed_model.get_text_embedding_batch(texts)
    else:
        vectors = [embed_model.get_text_embedding(t) for t in texts]
    return dict(zip(node_ids, vectors))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


class GraphRAGStrategy:
    """RAG strategy using a two-level hierarchical graph with 1-hop expansion."""

    def __init__(
        self,
        graph: Any,
        node_texts: dict[str, str],
        embeddings: dict[str, np.ndarray],
        embed_model: Any,
    ) -> None:
        self._graph = graph
        self._node_texts = node_texts
        self._embeddings = embeddings
        self._embed_model = embed_model
        self._node_ids = list(embeddings.keys())

    @property
    def name(self) -> str:
        return "graph_rag"

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if hasattr(self._embed_model, "get_query_embedding"):
            qvec = np.asarray(self._embed_model.get_query_embedding(query), dtype=float)
        else:
            qvec = np.asarray(self._embed_model.get_text_embedding(query), dtype=float)
        scores_list = [
            (_cosine_similarity(qvec, self._embeddings[nid]), nid)
            for nid in self._node_ids
        ]
        scores_list.sort(key=lambda x: -x[0])
        top = scores_list[:top_k]
        seen = set()
        chunks: list[tuple[float, str, dict]] = []
        for score, nid in top:
            if nid not in seen:
                seen.add(nid)
                chunks.append((score, self._node_texts[nid], {"node_id": nid}))
            for neighbor in self._graph.neighbors(nid):
                if neighbor not in seen:
                    seen.add(neighbor)
                    nb_score = next((s for s, n in scores_list if n == neighbor), 0.0)
                    nb_score = 0.5 * (nb_score + score)
                    chunks.append((nb_score, self._node_texts[neighbor], {"node_id": neighbor}))
        chunks.sort(key=lambda x: -x[0])
        return [
            RetrievedChunk(text=text, score=s, metadata=meta)
            for s, text, meta in chunks[: top_k * 3]
        ]


def build_graph_rag(
    config: GraphRAGConfig,
    lp_path: str | Path,
    persist_dir: str | Path = "outputs/graph_rag_index",
    embed_model: Any = None,
    *,
    force_rebuild: bool = False,
) -> GraphRAGStrategy:
    """Build or load the graph RAG index using use-case config. Returns a GraphRAGStrategy."""
    import networkx as nx

    persist_dir = Path(persist_dir)
    graph_path = persist_dir / "graph.json"
    texts_path = persist_dir / "node_texts.json"
    embeddings_path = persist_dir / "embeddings.npz"

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

    if not force_rebuild and graph_path.exists() and texts_path.exists() and embeddings_path.exists():
        with open(graph_path, "r", encoding="utf-8") as f:
            G = nx.node_link_graph(json.load(f))
        with open(texts_path, "r", encoding="utf-8") as f:
            node_texts = json.load(f)
        data = np.load(embeddings_path)
        embeddings = {str(k): data[k] for k in data.files}
        return GraphRAGStrategy(G, node_texts, embeddings, embed_model)

    formulation_docs = config.get_formulation_docs()
    constraint_instances = extract_constraint_instances(
        lp_path, constraint_type_prefixes=config.get_constraint_type_prefixes()
    )
    objective_var_families = extract_objective_variable_families(lp_path)
    G, node_texts = _build_hierarchical_graph(
        config, formulation_docs, constraint_instances, objective_var_families
    )
    embeddings = _embed_graph_nodes(node_texts, embed_model)

    persist_dir.mkdir(parents=True, exist_ok=True)
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G), f, indent=2)
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(node_texts, f, indent=2)
    np.savez_compressed(embeddings_path, **embeddings)

    return GraphRAGStrategy(G, node_texts, embeddings, embed_model)
