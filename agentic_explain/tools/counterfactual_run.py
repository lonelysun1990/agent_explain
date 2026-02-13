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


def _resolve_rhs_value(rhs: str, inputs: dict[str, Any]) -> float:
    """Resolve RHS to a number. rhs is either a numeric string or param[indices] (e.g. D[8,10])."""
    rhs = rhs.strip()
    try:
        return float(rhs)
    except ValueError:
        pass
    m = re.match(r"(\w+)\s*\[\s*([^\]]+)\s*\]", rhs)
    if not m:
        raise ValueError(f"Cannot resolve RHS as number or param[indices]: {rhs!r}")
    param_name = m.group(1)
    indices = [int(s.strip()) for s in m.group(2).split(",")]
    # Map formulation param names to inputs keys (e.g. D -> demand array in staffing)
    if param_name == "D":
        arr = inputs.get("demand")
    else:
        arr = inputs.get(param_name) or inputs.get(param_name.lower())
    if arr is None:
        raise ValueError(f"Unknown parameter in inputs: {param_name!r}")
    return float(arr[tuple(indices)])


def _get_var_by_name(model: "gp.Model", var_name: str, indices: list[int]) -> "gp.Var":
    gurobi_var_name = f"{var_name}[{','.join(map(str, indices))}]"
    var = model.getVarByName(gurobi_var_name)
    if var is None:
        for v in model.getVars():
            if v.VarName == gurobi_var_name:
                return v
        raise ValueError(f"Variable not found in model: {gurobi_var_name}")
    return var


def _parse_constraint_expression(expr: str, model: "gp.Model", inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Parse a constraint expression, add it to the model, and return a debug dict.

    Accepts:
      - var[indices] == number  or  var[indices] == param[indices]
      - var[indices] >=/<= number  or  var[indices] >=/<= param[indices]
      - sum of var[indices] (with +) >=/<=/== number  or  param[indices]
    """
    expr = expr.strip()

    # --- Linear: lhs (sum of vars) >=/<=/== rhs (number or param[indices]) ---
    m_linear = re.match(r"^(.+?)\s*(>=|<=|==)\s*([\d.]+|\w+\s*\[\s*[^\]]+\s*\])\s*$", expr)
    if m_linear and "+" in expr:
        lhs_str, op, rhs_str = m_linear.group(1), m_linear.group(2), m_linear.group(3).strip()
        rhs_val = _resolve_rhs_value(rhs_str, inputs)
        # Collect all var[indices] from lhs
        terms = re.findall(r"(\w+)\s*\[\s*([^\]]+)\s*\]", lhs_str)
        if not terms:
            raise ValueError(f"No variable terms found in LHS: {expr!r}")
        lin_expr = gp.LinExpr(0.0)
        for var_name, indices_str in terms:
            indices = [int(s.strip()) for s in indices_str.split(",")]
            v = _get_var_by_name(model, var_name, indices)
            lin_expr += v
        constr_name = f"user_constr_linear_{len(model.getConstrs())}"
        if op == ">=":
            model.addConstr(lin_expr >= rhs_val, name=constr_name)
        elif op == "<=":
            model.addConstr(lin_expr <= rhs_val, name=constr_name)
        else:
            model.addConstr(lin_expr == rhs_val, name=constr_name)
        return {
            "expr": expr,
            "gurobi_var_name": None,
            "forced_value": rhs_val,
            "constraint_name": constr_name,
            "linear": True,
        }

    # --- Simple: single var op number or param ---
    m_simple = re.match(
        r"(\w+)\s*\[\s*([^\]]+)\s*\]\s*(==|>=|<=)\s*([\d.]+|\w+\s*\[\s*[^\]]+\s*\])\s*$",
        expr,
    )
    if m_simple:
        var_name = m_simple.group(1)
        indices_str = m_simple.group(2)
        op = m_simple.group(3)
        rhs_str = m_simple.group(4).strip()
        indices = [int(s.strip()) for s in indices_str.split(",")]
        value = _resolve_rhs_value(rhs_str, inputs)
        var = _get_var_by_name(model, var_name, indices)
        gurobi_var_name = f"{var_name}[{','.join(map(str, indices))}]"
        constr_name = f"user_constr_{var_name}_{'_'.join(map(str, indices))}"
        if op == "==":
            model.addConstr(var == value, name=constr_name)
        elif op == ">=":
            model.addConstr(var >= value, name=constr_name)
        else:
            model.addConstr(var <= value, name=constr_name)
        return {
            "expr": expr,
            "gurobi_var_name": gurobi_var_name,
            "forced_value": value,
            "var_lb": var.LB,
            "var_ub": var.UB,
            "var_type": var.VType,
            "constraint_name": constr_name,
            "linear": False,
        }
    raise ValueError(f"Cannot parse constraint expression: {expr!r}")


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
            info = _parse_constraint_expression(expr, model, inputs)
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
