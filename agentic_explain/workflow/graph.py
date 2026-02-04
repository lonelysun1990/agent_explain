"""
LangGraph workflow: query -> entity_resolution -> constraint_generation -> counterfactual_run -> (feasible -> compare | infeasible -> ilp_analysis) -> summarize.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agentic_explain.workflow.state import AgentState
from agentic_explain.workflow import nodes


def create_workflow(
    *,
    openai_client: Any,
    rag_index: Any,
    baseline_result: dict[str, Any],
    data_dir: str,
    build_model_fn: Callable,
    inputs: dict,
    env_kwargs: dict,
    outputs_dir: str = "outputs",
) -> Any:
    """
    Build the agentic explainability graph.

    Parameters
    ----------
    openai_client : openai.OpenAI() (or compatible)
    rag_index : LlamaIndex VectorStoreIndex
    baseline_result : dict with objective_value, decision_variables
    data_dir : path to data folder (ds_list.json, project_list.json)
    build_model_fn : (inputs, env_kwargs) -> Gurobi model
    inputs : processed model inputs
    env_kwargs : for Gurobi env
    outputs_dir : where to write counterfactual.ilp

    Returns
    -------
    compiled LangGraph
    """
    variable_names = ["x", "x_ind", "x_p_ind", "d_miss", "x_idle"]

    def _query_node(s):
        return nodes.make_query_node()(s, openai_client=openai_client)

    entity_n = nodes.make_entity_resolution_node(data_dir)
    constraint_n = nodes.make_constraint_generation_node(rag_index, openai_client, variable_names)
    counterfactual_n = nodes.make_counterfactual_run_node(build_model_fn, inputs, env_kwargs, outputs_dir)
    compare_n = nodes.make_compare_node()
    ilp_n = nodes.make_ilp_analysis_node(rag_index, openai_client)
    summarize_n = nodes.make_summarize_node(openai_client)

    graph = StateGraph(AgentState)

    graph.add_node("query", _query_node)
    graph.add_node("entity_resolution", entity_n)
    graph.add_node("constraint_generation", constraint_n)
    graph.add_node("counterfactual_run", counterfactual_n)
    graph.add_node("compare", compare_n)
    graph.add_node("ilp_analysis", ilp_n)
    graph.add_node("summarize", summarize_n)

    graph.set_entry_point("query")
    graph.add_edge("query", "entity_resolution")
    graph.add_edge("entity_resolution", "constraint_generation")
    graph.add_edge("constraint_generation", "counterfactual_run")

    def route_after_run(state: AgentState) -> str:
        status = state.get("counterfactual_status", "error")
        if status == "feasible":
            return "compare"
        if status == "infeasible":
            return "ilp_analysis"
        return "summarize"

    graph.add_conditional_edges("counterfactual_run", route_after_run)
    graph.add_edge("compare", "summarize")
    graph.add_edge("ilp_analysis", "summarize")
    graph.add_edge("summarize", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def invoke_workflow(
    workflow: Any,
    user_query: str,
    *,
    baseline_result: dict[str, Any],
    rag_index: Any,
) -> dict[str, Any]:
    """Run the workflow and return the final state."""
    config = {"configurable": {"thread_id": "default"}}
    initial: AgentState = {
        "user_query": user_query,
        "baseline_result": baseline_result,
        "rag_index": rag_index,
    }
    final = workflow.invoke(initial, config=config)
    return final
