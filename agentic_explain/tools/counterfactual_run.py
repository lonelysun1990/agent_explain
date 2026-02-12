"""
Run a counterfactual optimization: build baseline model, add user constraints, optimize.
Returns result dict or writes .ilp on infeasibility.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB


def _parse_constraint_expression(expr: str, model: "gp.Model") -> dict[str, Any]:
    """
    Parse a simple constraint expression like 'x_ind[0,6,10] == 1' or 'd_miss[23,21] == 0',
    add it to the model, and return a debug dict with details.

    Returns
    -------
    dict with: expr, gurobi_var_name, forced_value, baseline_value, constraint_name
    """
    expr = expr.strip()
    # Match: var_name[i,j,k] == value  or  var_name[i, j, k] == value
    m = re.match(r"(\w+)\s*\[\s*([^\]]+)\s*\]\s*==\s*([\d.]+)", expr)
    if not m:
        raise ValueError(f"Cannot parse constraint expression: {expr}")
    var_name = m.group(1)
    indices_str = m.group(2)
    value = float(m.group(3))

    indices = [int(s.strip()) for s in indices_str.split(",")]
    gurobi_var_name = f"{var_name}[{','.join(map(str, indices))}]"
    var = model.getVarByName(gurobi_var_name)
    if var is None:
        # Fallback: Gurobi may not have name index until model.update(); search by VarName
        for v in model.getVars():
            if v.VarName == gurobi_var_name:
                var = v
                break
    if var is None:
        raise ValueError(f"Variable not found in model: {gurobi_var_name}")

    # Capture the variable's current bounds/start for debug context
    constr_name = f"user_constr_{var_name}_{'_'.join(map(str, indices))}"
    model.addConstr(var == value, name=constr_name)

    return {
        "expr": expr,
        "gurobi_var_name": gurobi_var_name,
        "forced_value": value,
        "var_lb": var.LB,
        "var_ub": var.UB,
        "var_type": var.VType,
        "constraint_name": constr_name,
    }


def run_counterfactual(
    build_model_fn: Callable,
    inputs: dict,
    env_kwargs: dict,
    constraint_expressions: list[str],
    time_limit: int = 100,
    ilp_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build a fresh model, add the given constraint expressions, optimize.

    Parameters
    ----------
    build_model_fn : callable(inputs, env_kwargs) -> gurobipy.Model
        e.g. build_gurobi_model from staffing_model
    inputs : processed data dict for the model
    env_kwargs : passed to gp.Env()
    constraint_expressions : list of strings like "x_ind[0,6,10] == 1"
    time_limit : seconds
    ilp_path : if set and model is infeasible, write IIS to this path

    Returns
    -------
    dict with:
        status: "feasible" | "infeasible" | "error"
        objective_value: float or None
        decision_variables: dict var_name -> value (if feasible)
        ilp_path: str or None (if infeasible and written)
        error: str (if status == "error")
    """
    result: dict[str, Any] = {
        "status": "error",
        "objective_value": None,
        "decision_variables": {},
        "ilp_path": None,
        "error": None,
        "applied_constraints": [],  # debug: details of each constraint added
    }

    try:
        model = build_model_fn(inputs, env_kwargs)
    except Exception as e:
        result["error"] = str(e)
        return result

    # Ensure variable names are available for getVarByName (required by some Gurobi builds).
    model.update()

    try:
        for expr in constraint_expressions:
            info = _parse_constraint_expression(expr, model)
            result["applied_constraints"].append(info)
    except Exception as e:
        result["error"] = f"Failed to add constraint: {e}"
        return result

    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        result["status"] = "feasible"
        result["objective_value"] = model.ObjVal
        result["decision_variables"] = {v.VarName: v.X for v in model.getVars()}
        # Compute objective breakdown using the solved variables
        try:
            from use_case.staffing_model import compute_objective_breakdown
            result["objective_breakdown"] = compute_objective_breakdown(
                result["decision_variables"], inputs
            )
        except Exception as e:
            result["objective_breakdown"] = {"error": str(e)}
        return result

    if model.status == GRB.INFEASIBLE:
        result["status"] = "infeasible"
        if ilp_path:
            try:
                model.computeIIS()
                path = Path(ilp_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                model.write(str(path))
                result["ilp_path"] = str(path)
            except Exception as e:
                result["error"] = f"IIS write failed: {e}"
        return result

    result["error"] = f"Unexpected status: {model.status}"
    return result
