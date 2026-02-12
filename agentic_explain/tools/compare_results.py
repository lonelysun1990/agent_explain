"""
Compare baseline and counterfactual optimization results.

Produces a structured comparison dict with objective breakdown and variable
diffs, persists it to disk, and returns it for the workflow to consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run_comparison(
    baseline_variables: dict[str, float],
    counterfactual_variables: dict[str, float],
    inputs: dict[str, Any],
    variable_names: list[str],
    counterfactual_breakdown: dict[str, Any] | None = None,
    persist_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compare baseline and counterfactual solutions.

    Parameters
    ----------
    baseline_variables : flat dict var_name -> value from baseline solve.
    counterfactual_variables : flat dict var_name -> value from counterfactual solve.
    inputs : processed model inputs (needed by compute_objective_breakdown).
    variable_names : list of variable family names (e.g. ["x", "x_ind", ...]).
    counterfactual_breakdown : pre-computed breakdown from counterfactual run (optional).
    persist_path : if set, write the comparison JSON here.

    Returns
    -------
    dict with keys:
        objective_comparison : list of dicts per term
        objective_totals : { baseline, counterfactual, delta, pct_change }
        variable_diffs : dict family_name -> list of { var, baseline, counterfactual, delta }
        summary_text : formatted text suitable for LLM consumption
    """
    from use_case.staffing_model import compute_objective_breakdown

    # --- Objective breakdown ---
    base_breakdown = compute_objective_breakdown(baseline_variables, inputs)
    cf_breakdown = (
        counterfactual_breakdown
        if counterfactual_breakdown and not counterfactual_breakdown.get("error")
        else compute_objective_breakdown(counterfactual_variables, inputs)
    )

    base_terms = base_breakdown.get("terms", {})
    cf_terms = cf_breakdown.get("terms", {}) if isinstance(cf_breakdown, dict) else {}

    # Iterate over the union of term names (no hardcoded names)
    all_term_names = list(dict.fromkeys(list(base_terms.keys()) + list(cf_terms.keys())))

    objective_comparison = []
    for term_name in all_term_names:
        bt = base_terms.get(term_name, {})
        ct = cf_terms.get(term_name, {})
        bv = bt.get("value", 0)
        cv = ct.get("value", 0)
        delta = cv - bv
        pct = (delta / bv * 100) if abs(bv) > 1e-8 else 0.0
        objective_comparison.append({
            "term": term_name,
            "description": bt.get("description", ct.get("description", "")),
            "formula": bt.get("formula", ct.get("formula", "")),
            "baseline": bv,
            "counterfactual": cv,
            "delta": delta,
            "pct_change": pct,
        })

    base_total = base_breakdown.get("total", 0)
    cf_total = cf_breakdown.get("total", 0)
    delta_total = cf_total - base_total
    pct_total = (delta_total / base_total * 100) if abs(base_total) > 1e-8 else 0.0

    objective_totals = {
        "baseline": base_total,
        "counterfactual": cf_total,
        "delta": delta_total,
        "pct_change": pct_total,
    }

    # --- Variable diffs grouped by family ---
    variable_diffs: dict[str, list[dict[str, Any]]] = {}
    for family in variable_names:
        prefix = family + "["
        diffs = []
        for k in counterfactual_variables:
            if k.startswith(prefix):
                bv = baseline_variables.get(k, 0.0)
                cv = counterfactual_variables[k]
                # Use 0.5 threshold for binary-like vars, 1e-4 for continuous
                threshold = 0.5 if cv in (0, 1) and bv in (0, 1) else 1e-4
                if abs(cv - bv) > threshold:
                    diffs.append({
                        "var": k,
                        "baseline": bv,
                        "counterfactual": cv,
                        "delta": cv - bv,
                    })
        if diffs:
            diffs.sort(key=lambda x: abs(x["delta"]), reverse=True)
            variable_diffs[family] = diffs

    # --- Build summary text (for LLM consumption) ---
    lines = ["=== OBJECTIVE COMPARISON ==="]
    lines.append(f"{'Term':<35s} {'Baseline':>12s} {'Counter.':>12s} {'Delta':>12s} {'%Change':>10s}")
    lines.append("─" * 85)
    for oc in objective_comparison:
        lines.append(
            f"{oc['term']:<35s} {oc['baseline']:12.4f} {oc['counterfactual']:12.4f} "
            f"{oc['delta']:+12.4f} {oc['pct_change']:+9.1f}%"
        )
        if oc["description"]:
            lines.append(f"  ({oc['description']})")
    lines.append("─" * 85)
    lines.append(
        f"{'TOTAL':<35s} {objective_totals['baseline']:12.4f} "
        f"{objective_totals['counterfactual']:12.4f} "
        f"{objective_totals['delta']:+12.4f} {objective_totals['pct_change']:+9.1f}%"
    )

    lines.append("")
    lines.append("=== VARIABLE CHANGES BY FAMILY ===")
    for family, diffs in variable_diffs.items():
        lines.append(f"\n{family} — {len(diffs)} variables changed (top 15 by magnitude):")
        for d in diffs[:15]:
            direction = "▲" if d["delta"] > 0 else "▼"
            lines.append(
                f"  {direction} {d['var']}: {d['baseline']:.4f} → "
                f"{d['counterfactual']:.4f} ({d['delta']:+.4f})"
            )
        if len(diffs) > 15:
            lines.append(f"  ... and {len(diffs) - 15} more")

    summary_text = "\n".join(lines)

    # --- Assemble result ---
    result = {
        "objective_comparison": objective_comparison,
        "objective_totals": objective_totals,
        "variable_diffs": {
            # Only persist top 50 per family to keep the file readable
            family: diffs[:50]
            for family, diffs in variable_diffs.items()
        },
        "variable_diff_counts": {family: len(diffs) for family, diffs in variable_diffs.items()},
        "summary_text": summary_text,
    }

    # --- Persist ---
    if persist_path:
        try:
            path = Path(persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # best-effort

    return result
