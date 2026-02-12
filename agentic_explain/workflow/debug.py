"""
Debug helpers: print retrieval, LLM messages, applied constraints, and comparison from workflow state.
"""

from __future__ import annotations

import textwrap
from typing import Any


def print_retrieval_debug(final_state: dict[str, Any], *, text_truncate: int = 500) -> None:
    """Print RAG retrieval debug info (query, top-k, chunks with scores and text)."""
    rag_debug = final_state.get("rag_retrieval_debug", {})
    strategy = rag_debug.get("strategy", "?")
    print(f"RAG strategy: {strategy}\n")
    for stage_name, info in rag_debug.items():
        if stage_name == "strategy":
            continue
        print("=" * 90)
        print(f"  Stage: {stage_name}")
        print(f"  Retrieval query: {info.get('query', '?')}")
        print(f"  Top-k: {info.get('top_k', '?')}")
        if "iis_constraint_names" in info:
            print(f"  IIS constraints: {info['iis_constraint_names']}")
        print("=" * 90)
        for i, chunk in enumerate(info.get("chunks", [])):
            score = chunk.get("score")
            meta = chunk.get("metadata", {})
            text = chunk.get("text", "")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"\n  —— Chunk {i}  score={score_str}")
            print(f"     source={meta.get('source', '?')}  section={meta.get('section', '?')}")
            extra = {k: v for k, v in meta.items() if k not in ("source", "section", "path")}
            if extra:
                print(f"     {extra}")
            display = text if len(text) <= text_truncate else text[:text_truncate] + "\n     ... [truncated]"
            wrapped = textwrap.fill(display, width=95, initial_indent="     ", subsequent_indent="     ")
            print(wrapped)
        print()


def print_llm_messages(final_state: dict[str, Any], *, chars_to_show: int = 3000) -> None:
    """Print system/user messages and raw LLM response for each RAG-augmented stage."""
    llm_debug = final_state.get("llm_messages_debug", {})
    for stage_name, info in llm_debug.items():
        print("=" * 90)
        print(f"  Stage: {stage_name}")
        print("=" * 90)
        print("\n  [SYSTEM MESSAGE]")
        sys_msg = info.get("system", "")
        print(textwrap.fill(sys_msg, width=95, initial_indent="  ", subsequent_indent="  "))
        print(f"\n  [USER MESSAGE]  (first {chars_to_show} chars)")
        user_msg = info.get("user", "")
        display_user = user_msg if len(user_msg) <= chars_to_show else user_msg[:chars_to_show] + "\n  ... [truncated]"
        print(textwrap.fill(display_user, width=95, initial_indent="  ", subsequent_indent="  "))
        print("\n  [RAW LLM RESPONSE]")
        raw = info.get("raw_response", "")
        print(textwrap.fill(raw, width=95, initial_indent="  ", subsequent_indent="  "))
        print()


def print_applied_constraints(final_state: dict[str, Any]) -> None:
    """Print which constraints were added to the counterfactual model and baseline/forced values."""
    cf_result = final_state.get("counterfactual_result", {})
    applied = cf_result.get("applied_constraints", [])
    if not applied:
        print("No constraints were applied (check counterfactual_result for errors).")
        if cf_result.get("error"):
            print(f"  Error: {cf_result['error']}")
        return
    print("=" * 90)
    print(f"  {len(applied)} constraint(s) added to the counterfactual Gurobi model")
    print("=" * 90)
    for i, ac in enumerate(applied):
        baseline_val = ac.get("baseline_value")
        forced_val = ac.get("forced_value")
        bv_str = f"{baseline_val:.4f}" if baseline_val is not None else "N/A"
        direction = ""
        if baseline_val is not None and forced_val is not None:
            if abs(forced_val - baseline_val) < 1e-8:
                direction = "(no change from baseline)"
            elif forced_val > baseline_val:
                direction = f"(forcing UP from {bv_str})"
            else:
                direction = f"(forcing DOWN from {bv_str})"
        print(f"\n  Constraint {i}:")
        print(f"    Expression:      {ac.get('expr')}")
        print(f"    Gurobi var:      {ac.get('gurobi_var_name')}")
        print(f"    Forced value:    {forced_val}  {direction}")
        print(f"    Baseline value:  {bv_str}")
        print(f"    Var type:        {ac.get('var_type')}  bounds=[{ac.get('var_lb')}, {ac.get('var_ub')}]")
        print(f"    Constr name:     {ac.get('constraint_name')}")
    print(f"\n  Counterfactual status: {final_state.get('counterfactual_status')}")
    cf_obj = cf_result.get("objective_value")
    base_obj = final_state.get("baseline_result", {}).get("objective_value")
    if cf_obj is not None and base_obj is not None:
        print(f"  Baseline obj:          {base_obj:.4f}")
        print(f"  Counterfactual obj:    {cf_obj:.4f}")
        print(f"  Delta:                 {cf_obj - base_obj:+.4f}")


def print_comparison(final_state: dict[str, Any]) -> None:
    """Print the comparison summary (objective breakdown and variable changes) from the compare node."""
    comparison = final_state.get("comparison_summary", "(no comparison)")
    print(comparison)


def print_workflow_summary(final_state: dict[str, Any]) -> None:
    """Print retrieval debug, LLM messages, applied constraints, comparison, and final summary."""
    print("\n" + "=" * 90)
    print("  RAG RETRIEVAL DEBUG")
    print("=" * 90)
    print_retrieval_debug(final_state)
    print("=" * 90)
    print("  LLM MESSAGES")
    print("=" * 90)
    print_llm_messages(final_state)
    print("=" * 90)
    print("  APPLIED CONSTRAINTS")
    print("=" * 90)
    print_applied_constraints(final_state)
    print("\n" + "=" * 90)
    print("  OBJECTIVE COMPARISON & VARIABLE CHANGES")
    print("=" * 90)
    print_comparison(final_state)
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)
    print(final_state.get("final_summary", "(no summary)"))
