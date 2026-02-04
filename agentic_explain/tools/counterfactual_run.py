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


def _parse_constraint_expression(expr: str, model: "gp.Model") -> "gp.Constr":
    """
    Parse a simple constraint expression like 'x_ind[0,6,10] == 1' or 'd_miss[23,21] == 0'
    and return the corresponding Gurobi constraint (to be added).

    Returns the Constr object after model.addConstr(...).
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
        raise ValueError(f"Variable not found in model: {gurobi_var_name}")

    return model.addConstr(var == value, name=f"user_constr_{var_name}_{'_'.join(map(str, indices))}")


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
    result = {
        "status": "error",
        "objective_value": None,
        "decision_variables": {},
        "ilp_path": None,
        "error": None,
    }

    try:
        model = build_model_fn(inputs, env_kwargs)
    except Exception as e:
        result["error"] = str(e)
        return result

    try:
        for expr in constraint_expressions:
            _parse_constraint_expression(expr, model)
    except Exception as e:
        result["error"] = f"Failed to add constraint: {e}"
        return result

    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        result["status"] = "feasible"
        result["objective_value"] = model.ObjVal
        result["decision_variables"] = {v.VarName: v.X for v in model.getVars()}
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
