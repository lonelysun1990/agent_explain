"""
Evaluation helpers: load queries, run single/batch workflow, format results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from agentic_explain.evaluation.run_storage import (
    create_run_dir,
    get_completed_tasks,
    save_result,
    write_run_dir_metadata,
)


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


def run_batch_multi_strategy(
    create_workflow_fn: Callable[..., Any],
    strategies: dict[str, Any],
    queries: list[dict[str, Any]],
    baseline_result: dict[str, Any],
    outputs_dir: str | Path,
    *,
    run_dir: Path | None = None,
    resume: bool = False,
    openai_client: Any = None,
    data_dir: str = "",
    build_model_fn: Callable | None = None,
    inputs: dict | None = None,
    env_kwargs: dict | None = None,
    temperature: float = 0,
) -> Path:
    """
    Run all (strategy x query) combinations, save each result to run_dir.
    Progress: prints which strategy/query finished, failed, or was skipped (resume).

    Parameters
    ----------
    create_workflow_fn : callable that returns a workflow (e.g. create_workflow from graph)
    strategies : dict name -> rag_strategy
    queries : list of query dicts (query, expected_path, expected_constraint_expr, etc.)
    baseline_result : baseline result dict
    outputs_dir : parent for runs (e.g. OUTPUTS_DIR)
    run_dir : if None, create new run dir; if Path and resume=True, skip existing files
    resume : if True and run_dir has existing result files, skip those (strategy, query_index)
    openai_client, data_dir, build_model_fn, inputs, env_kwargs, temperature : passed to create_workflow_fn

    Returns
    -------
    Path to the run directory.
    """
    from agentic_explain.workflow.graph import invoke_workflow

    outputs_dir = Path(outputs_dir)
    if run_dir is None:
        run_dir = create_run_dir(outputs_dir)
        resume = False
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    strategy_names = list(strategies.keys())
    completed = get_completed_tasks(run_dir, strategy_names, len(queries)) if resume else set()
    if resume and completed:
        print(f"Resume: skipping {len(completed)} already-completed (strategy, query_index) pairs.")

    write_run_dir_metadata(run_dir, {
        "strategies": strategy_names,
        "query_count": len(queries),
        "resume": resume,
    })

    total = len(strategy_names) * len(queries)
    done = 0
    failed: list[tuple[str, int, str]] = []  # (strategy, query_index, error_msg)

    for sname in strategy_names:
        strategy = strategies[sname]
        workflow = create_workflow_fn(
            openai_client=openai_client,
            rag_strategy=strategy,
            baseline_result=baseline_result,
            data_dir=data_dir,
            build_model_fn=build_model_fn,
            inputs=inputs or {},
            env_kwargs=env_kwargs or {},
            outputs_dir=str(outputs_dir),
            temperature=temperature,
        )
        for qi, q in enumerate(queries):
            if (sname, qi) in completed:
                print(f"  [skip] {sname} query {qi} (already done)")
                done += 1
                continue
            query_text = q.get("query", "")
            try:
                state = invoke_workflow(workflow, query_text, baseline_result=baseline_result)
                save_result(
                    run_dir, sname, qi, state,
                    query_meta={
                        "query": query_text,
                        "query_id": q.get("id"),
                        "expected_path": q.get("expected_path"),
                        "expected_constraint_expr": q.get("expected_constraint_expr"),
                        "reference_answer": q.get("reference_answer"),
                    },
                )
                done += 1
                status = state.get("counterfactual_status", "?")
                print(f"  [ok] {sname} query {qi} -> {status} ({done}/{total})")
            except Exception as e:
                failed.append((sname, qi, str(e)))
                print(f"  [FAIL] {sname} query {qi}: {e}")
    if failed:
        print(f"\nFailed: {len(failed)} — {failed}")
    print(f"Run dir: {run_dir}")
    return run_dir
