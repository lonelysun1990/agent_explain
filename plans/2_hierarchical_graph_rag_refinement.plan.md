---
name: Graph RAG Comparison
overview: "Update the Graph RAG strategy to use a two-level hierarchical graph: Level 1 type nodes (~16 abstract concepts) and Level 2 entity-group nodes (~208 nodes grouped by employee/project), connected by parent, sibling, and cross-entity edges. This replaces the flat ~16-node graph in the existing plan."
todos:
  - id: lp-helper
    content: Add extract_constraint_instances() to lp_parser.py for grouping constraint instances by type with sample expressions
    status: pending
  - id: graph-rag
    content: "Create GraphRAGStrategy with two-level hierarchical graph (~212 nodes): Level 1 type nodes + Level 2 entity-group nodes, three edge types, 1-hop expansion retrieval"
    status: pending
isProject: false
---

# Hierarchical Graph RAG Update

## The Problem

The original graph RAG design has only ~15-20 type-level nodes (variable families, constraint types, objective terms). When a user asks "Why was Josh not staffed on Ipp IO Pilot in week 6?", there is nothing entity-specific to match against — `constr:demand_balance` is too abstract to distinguish project 0 from project 10.

## Proposed: Two-Level Hierarchy

### Level 1: Type Nodes (~16 nodes)

- 5 variable families: `var:x`, `var:x_ind`, `var:x_p_ind`, `var:d_miss`, `var:x_idle`
- 7 constraint types: `constr:demand_balance`, `constr:employee_allocation`, `constr:indicator_constraint`, `constr:staffed_indicator`, `constr:max_concurrency`, `constr:specific_employee_staffing`, `constr:oversight_employee`
- 4 objective terms: `obj:cost_of_missing_demand`, `obj:idle_time`, `obj:staffing_consistency`, `obj:out_of_cohort_penalty`

Text: FORMULATION_DOCS description for the type. Embedded.

### Level 2: Entity-Group Nodes (~208 nodes)

**Constraint groups:** demand_balance x project (22), employee_allocation x employee (14), max_concurrency x employee (14), indicator_constraint x employee (14), staffed_indicator x employee (14), specific_employee_staffing x employee (14), oversight_employee x project (22).

**Variable groups:** x/x_ind/x_p_ind/x_idle x employee (14 each), d_miss x project (22).

**Objective groups:** 4 nodes (one per term).

**Total: ~16 (L1) + ~196 (L2) = ~212 nodes**

### Edge Types

1. **Parent edges (Level 2 -> Level 1):** Every Level 2 node -> its parent type node.
2. **Type-level edges (Level 1 <-> Level 1):** Derived from LP file (var <-> constr, var <-> obj).
3. **Sibling/entity edges (Level 2 <-> Level 2):** Same employee or same project.

### Retrieval

1. Embed query; cosine similarity -> top-k nodes.
2. 1-hop expansion; neighbor score = 0.5 * matched score.
3. Assemble text from matched + expanded nodes; return `list[RetrievedChunk]`.

### Implementation

- `lp_parser.py`: `extract_constraint_instances(lp_path) -> dict[str, list[dict]]` — constraint instances grouped by type with name, variable families, sample expression.
- `graph_rag.py`: `_build_hierarchical_graph(formulation_docs, lp_path, data_dir)`, `_embed_graph_nodes(graph, embed_model)`, `GraphRAGStrategy.retrieve(query, top_k)`.
