"""LangGraph state for the agentic explainability workflow."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_query: str
    reformulated_query: str
    resolved_entities: dict[str, Any]
    constraint_expressions: list[str]
    counterfactual_status: str  # "feasible" | "infeasible" | "error"
    counterfactual_result: dict[str, Any]
    baseline_result: dict[str, Any]
    comparison_summary: str
    ilp_path: str
    conflict_explanation: str
    final_summary: str
    retry_count: int
    # --- Debug / visibility fields ---
    rag_retrieval_debug: dict[str, Any]  # chunks, scores, query used for retrieval
    llm_messages_debug: dict[str, Any]   # system + user messages sent to each LLM call
