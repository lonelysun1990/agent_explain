"""
Evaluation helpers: load queries, run single/batch workflow, format results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_eval_queries(path: str | Path) -> list[dict[str, Any]]:
    """Load evaluation queries from a JSON file (e.g. queries.json)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("queries", data.get("items", []))


def run_single_query(
    workflow: Any,
    query: str,
    baseline_result: dict[str, Any],
    *,
    rag_strategy: Any = None,
) -> dict[str, Any]:
    """
    Run the workflow for one query and return the final state.

    The workflow must have been built with the same rag_strategy (passed at create_workflow time).
    rag_strategy is not passed to invoke; it is only for documentation.
    """
    from agentic_explain.workflow.graph import invoke_workflow

    return invoke_workflow(workflow, query, baseline_result=baseline_result)


def format_query_result(query: dict[str, Any], final_state: dict[str, Any]) -> str:
    """Format a single query result for display (expected vs actual path, constraint, answer theme)."""
    expected_path = query.get("expected_path", "?")
    actual_path = final_state.get("counterfactual_status", "?")
    path_match = actual_path == expected_path
    lines = [
        f"Query: {query.get('query', '')[:80]}...",
        f"  Expected path: {expected_path}  Actual: {actual_path}  Match: {path_match}",
        f"  Expected constraint: {query.get('expected_constraint_expr', '?')}",
        f"  Generated: {final_state.get('constraint_expressions', [])}",
        f"  Summary: {final_state.get('final_summary', '')[:200]}...",
    ]
    return "\n".join(lines)


def run_batch_evaluation(
    workflow: Any,
    queries: list[dict[str, Any]],
    baseline_result: dict[str, Any],
    output_path: str | Path,
    *,
    rag_strategy_name: str = "unknown",
) -> list[dict[str, Any]]:
    """
    Run the workflow for all queries and save results to a JSON file.

    Returns the list of result dicts (query_id, query, expected_path, actual_path, path_match, etc.).
    """
    from agentic_explain.workflow.graph import invoke_workflow

    results = []
    for i, q in enumerate(queries):
        query_text = q.get("query", "")
        print(f"[{i + 1}/{len(queries)}] {q.get('id', i)}: {query_text[:60]}...")
        state = invoke_workflow(workflow, query_text, baseline_result=baseline_result)
        results.append({
            "query_id": q.get("id"),
            "query": query_text,
            "expected_path": q.get("expected_path"),
            "actual_path": state.get("counterfactual_status", "unknown"),
            "path_match": state.get("counterfactual_status") == q.get("expected_path"),
            "expected_constraint_expr": q.get("expected_constraint_expr"),
            "constraint_expressions": state.get("constraint_expressions", []),
            "final_summary": state.get("final_summary", ""),
            "rag_strategy": rag_strategy_name,
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return results
