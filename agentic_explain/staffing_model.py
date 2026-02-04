"""
Staffing optimization model: Gurobi formulation and data loading.

This module is the canonical .py source for the staffing optimization model.
It can be replaced by your own implementation; the agentic workflow expects
build_gurobi_model(inputs, env_kwargs) and load_raw_data(data_dir).
"""

# Structured formulation documentation for RAG. Sections use names that match
# model variables and constraint name patterns (x, x_ind, d_miss, demand_balance, etc.).
FORMULATION_DOCS = r"""
## PROBLEM_OVERVIEW
Name: Staffing Optimization.
Description: Data science organization wants to staff its employees efficiently and dynamically to handle challenging staffing situations with multiple pilots and projects. The DS leadership wants a tool that can help them with explicit trade-offs during staffing decision making.
Requirements:
- Account for different starting times for each project
- Account for different FTE requirements and timelines for each project
- Account for missed staffing demand, and minimize it
- Account for employee idling, and minimize it
- Max number of concurrent projects per employee at any given time
- Allow employee preferences for different cohorts of projects/products
- Avoid too much back-and-forth staffing assignments
- Allow special requirements for projects (oversight, specific employees)

## INDEX_MAPPING
Indices link model variables to business entities. Employee and project names come from ds_list and project_list in data; use j=0,1,..., d=0,1,..., t=0,1,... in constraints.
### j
Employee index. Symbol: j in J. Range: 0 to n_employees-1. Each employee is identified by position in ds_list. When explaining, translate j to employee name (e.g. j=0 is first employee in ds_list).
### d
Project index. Symbol: d in D. Range: 0 to n_projects-1. Each project is identified by position in project_list. When explaining, translate d to project name (e.g. d=0 is first project in project_list).
### t
Time index (week). Symbol: t in T. Range: 0 to horizon-1. Each period is one week. Use "Week N" in explanations for t=N.

## VARIABLES
### x
Name: Staffing Allocation. Symbol: x[j,t,d]. Dimensions: (n_employees, horizon, n_projects). Bounds: [0,1]. Unit: FTE.
Description: The fraction of employee j's time allocated to project d in week t. A value of 0.5 means 50% of that employee's time on that project that week.
Business context: Represents how much of employee j's capacity is assigned to project d in week t. Used in demand_balance and employee_allocation constraints.
Math: continuous, lb=0, ub=1. In code: model.addVars(..., name="x").

### x_ind
Name: Allocation Indicator. Symbol: x_ind[j,t,d]. Dimensions: (n_employees, horizon, n_projects). Binary.
Description: 1 if employee j is allocated to project d in week t (any positive allocation); 0 otherwise.
Business context: Used to enforce max concurrent projects per employee (max_concurrency_constraint). Links to x via indicator_constraint.
Math: x_ind >= x and x_ind <= M*x. In code: model.addVars(..., name="x_ind").

### x_p_ind
Name: Project Staffing Indicator. Symbol: x_p_ind[j,d]. Dimensions: (n_employees, n_projects). Binary.
Description: 1 if employee j ever works on project d over the horizon; 0 otherwise.
Business context: Minimizing sum of x_p_ind reduces staffing fragmentation. Used in out-of-cohort penalty objective. Links to x via staffed_indicator constraints.
Math: x_p_ind <= sum_t x[j,t,d] and M*x_p_ind >= sum_t x[j,t,d]. In code: model.addVars(..., name="x_p_ind").

### d_miss
Name: Unmet Staffing Demand (Under Staffing). Symbol: d_miss[t,d]. Dimensions: (horizon, n_projects). Bounds: [0, inf). Unit: FTE.
Description: The amount of project d's demand that cannot be met in week t.
Business context: Staffing shortage. If a project needs 2.0 FTE in a week but we provide 1.5 FTE, d_miss = 0.5. Minimized in objective (cost_of_missing_demand). Appears in demand_balance: sum_j x[j,t,d]*F_j + d_miss[t,d] = D[d,t].
Math: continuous, lb=0. In code: model.addVars(..., name="d_miss").

### x_idle
Name: Employee Idle Time. Symbol: x_idle[j,t]. Dimensions: (n_employees, horizon). Bounds: [0,1]. Unit: FTE.
Description: The fraction of employee j's time that is idle (not assigned to any project) in week t.
Business context: Unutilized capacity. Minimized in objective (idle_time). Employee allocation: sum_d x[j,t,d] + x_idle[j,t] = 1 for all j,t.
Math: continuous, lb=0, ub=1. In code: model.addVars(..., name="x_idle").

## CONSTRAINTS
Constraint names in the model follow patterns: demand_balance_project_{d}_week_{t}, employee_allocation_constraint_{j}_{t}, indicator_constraint_1_{j}_{t}_{d}, indicator_constraint_0_{j}_{t}_{d}, staffed_indicator_1_{j}_{d}, staffed_indicator_0_{j}_{d}, max_concurrency_constraint_{t}_{j}, specific_employee_staffing_constraint_{j}_{d}, oversight_employee_constraint_{d}.
### demand_balance
Name: Demand Balance. For each project d and week t: sum_j x[j,t,d] * F_j + d_miss[t,d] = D[d,t].
Description: Total staffing (allocations weighted by employee FTE) plus unmet demand equals required demand.
Business context: Core balance of supply and demand. If we cannot meet demand, d_miss captures the shortage. Variables: x, d_miss.

### employee_allocation
Name: Employee Allocation. For each employee j and week t: sum_d x[j,t,d] + x_idle[j,t] = 1.
Description: Each employee's allocation across projects plus idle time equals 100% per week.
Business context: Capacity constraint; cannot work more than 100%. Variables: x, x_idle.

### indicator_constraint
Name: Indicator Tracking. For each j,t,d: x_ind[j,t,d] >= x[j,t,d] and x_ind[j,t,d] <= M*x[j,t,d].
Description: Links x and x_ind so that x_ind=1 when x>0.
Business context: Ensures x_ind correctly indicates whether employee is on project (for max concurrent projects). Variables: x, x_ind.

### staffed_indicator
Name: Project Staffing Indicator. For each j,d: x_p_ind[j,d] <= sum_t x[j,t,d] and M*x_p_ind[j,d] >= sum_t x[j,t,d].
Description: Links x_p_ind to whether employee j ever works on project d.
Business context: Tracks employee-project pairings for consistency and out-of-cohort penalty. Variables: x, x_p_ind.

### max_concurrency
Name: Max Concurrent Projects. For each j,t: sum_d x_ind[j,t,d] <= P_j.
Description: Limits how many projects an employee can work on in one week.
Business context: Prevents excessive context switching. Variables: x_ind. Constraint name pattern: max_concurrency_constraint_{t}_{j}.

### specific_employee_staffing
Name: Specific Employee Staffing. For each j,d: sum_t x[j,t,d] >= d_p[j,d].
Description: Ensures required specific employee hours per project are met.
Business context: Some projects require specific employees (e.g. domain experts). Variables: x. Constraint name pattern: specific_employee_staffing_constraint_{j}_{d}.

### oversight_employee
Name: Oversight Employee Requirement. For each d: sum over oversight employees j and t of x[j,t,d] >= d_oversight_p[d].
Description: Projects with oversight requirements get sufficient oversight FTE.
Business context: Some projects need senior/oversight roles. Variables: x. Constraint name pattern: oversight_employee_constraint_{d}.

## OBJECTIVES
Total objective: minimize (obj1 + obj2 + obj3 + obj4).
### cost_of_missing_demand
Sum over t,d of C_miss[d,t] * d_miss[t,d]. Minimizes weighted unmet demand; cost typically increases over time.
Variables: d_miss.

### idle_time
Sum over j,t of x_idle[j,t]. Minimizes total employee idle time.
Variables: x_idle.

### staffing_consistency
Sum over j,d of x_p_ind[j,d]. Minimizes number of employee-project pairings to reduce fragmentation.
Variables: x_p_ind.

### out_of_cohort_penalty
Sum over j,d of x_p_ind[j,d] * project_cohort_penalty[j,d]. Penalizes assigning employees to projects outside their cohort.
Variables: x_p_ind.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np


# -------- Data structures --------

class DotDict(dict):
    """A dictionary with dot notation access and recursive conversion."""

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, attr):
        if attr in self:
            return self.get(attr)
        raise AttributeError(f"Attribute {attr} not found in {self}")

    def __setattr__(self, attr, value):
        if isinstance(value, dict):
            value = DotDict(value)
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]


# -------- Data loading --------

def load_raw_data(data_dir: str | Path) -> dict[str, Any]:
    """Load raw data from JSON and txt files in data_dir (Option 2 style)."""
    data_dir = Path(data_dir)
    with open(data_dir / "fte_mapping.json", "r", encoding="utf-8") as f:
        fte_mapping = json.load(f)
    with open(data_dir / "concurrent_projects.json", "r", encoding="utf-8") as f:
        concurrent_projects = json.load(f)
    with open(data_dir / "oversight_ds_list.txt", "r", encoding="utf-8") as f:
        oversight_ds_list = [line.strip() for line in f if line.strip()]
    with open(data_dir / "ds_list.json", "r", encoding="utf-8") as f:
        ds_list = json.load(f)
    with open(data_dir / "project_list.json", "r", encoding="utf-8") as f:
        project_list = json.load(f)
    return {
        "fte_mapping": fte_mapping,
        "concurrent_projects": concurrent_projects,
        "oversight_ds_list": oversight_ds_list,
        "ds_list": ds_list,
        "project_list": project_list,
    }


def process_data(
    fte_mapping: dict,
    concurrent_projects: dict,
    oversight_ds_list: list,
    ds_list: list,
    project_list: list,
) -> dict[str, Any]:
    """Process raw data into model inputs (indices, demand, costs, etc.)."""

    def find_idx(list_dict: list, name: str, key: str = "name") -> int:
        for idx, value in enumerate(list_dict):
            if value[key] == name:
                return idx
        return -1

    fte_list = [fte_mapping[ds["title"]] * ds["performance"] for ds in ds_list]
    max_proj_list = [concurrent_projects[ds["title"]] for ds in ds_list]

    n_projects = len(project_list)
    D = range(n_projects)
    n_employees = len(ds_list)
    J = range(n_employees)

    horizon = int(
        np.max(
            [
                p["duration"] + p["starting offset"] + p["max delay start"]
                for p in project_list
            ]
        )
    )
    T = range(horizon)

    demand = np.array(
        [
            [0] * p["starting offset"]
            + [p["fte"]] * p["duration"]
            + [0] * (horizon - p["starting offset"] - p["duration"])
            for p in project_list
        ]
    )

    C_miss = np.array(
        [
            list(
                np.linspace(
                    p["under staffing cost"][0], p["under staffing cost"][1], horizon
                )
            )
            for p in project_list
        ]
    )

    d_p = np.zeros((n_employees, n_projects))
    d_oversight_p = np.zeros(n_projects)

    for p_idx, p in enumerate(project_list):
        if p.get("special requirements") is not None:
            req = p["special requirements"]
            if "oversight" in req:
                d_oversight_p[p_idx] = p["duration"] * req["oversight"]
            if "employees" in req:
                for name, fte in req["employees"].items():
                    ds_idx = find_idx(ds_list, name)
                    if ds_idx >= 0:
                        d_p[ds_idx, p_idx] = fte * p["duration"]

    oversight_idx_list = [
        ds_idx for ds_idx, ds in enumerate(ds_list)
        if ds["title"] in oversight_ds_list
    ]

    project_cohort_penalty = np.zeros((len(ds_list), len(project_list)))
    for ds_idx, ds in enumerate(ds_list):
        for p_idx, p in enumerate(project_list):
            project_cohort_penalty[ds_idx, p_idx] = 1 - any(
                cohort in p["tags"] for cohort in ds["cohort"]
            )

    return {
        "fte_list": fte_list,
        "max_proj_list": max_proj_list,
        "n_projects": n_projects,
        "D": D,
        "n_employees": n_employees,
        "J": J,
        "horizon": horizon,
        "T": T,
        "demand": demand,
        "C_miss": C_miss,
        "d_p": d_p,
        "d_oversight_p": d_oversight_p,
        "oversight_idx_list": oversight_idx_list,
        "project_cohort_penalty": project_cohort_penalty,
    }


# -------- Model building --------

def build_gurobi_model(inputs: dict, env_kwargs: dict | None = None):
    """
    Build the staffing optimization Gurobi model.

    Parameters
    ----------
    inputs : dict
        Processed data from process_data().
    env_kwargs : dict, optional
        Passed to gp.Env(). If None or empty, gp.Env() uses default (env vars like GRB_LICENSE_FILE).

    Returns
    -------
    model : gurobipy.Model
    """
    inputs = DotDict(inputs)
    env_kwargs = env_kwargs or {}
    env = gp.Env(**env_kwargs)
    model = gp.Model(name="Staffing_Optimization", env=env)

    # Decision variables
    x = model.addVars(
        inputs.n_employees,
        inputs.horizon,
        inputs.n_projects,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="x",
    )
    x_ind = model.addVars(
        inputs.n_employees,
        inputs.horizon,
        inputs.n_projects,
        vtype=GRB.BINARY,
        name="x_ind",
    )
    x_p_ind = model.addVars(
        inputs.n_employees, inputs.n_projects, vtype=GRB.BINARY, name="x_p_ind"
    )
    d_miss = model.addVars(
        inputs.horizon, inputs.n_projects, vtype=GRB.CONTINUOUS, lb=0, name="d_miss"
    )
    x_idle = model.addVars(
        inputs.n_employees,
        inputs.horizon,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="x_idle",
    )

    # Objectives
    obj1 = gp.quicksum(
        inputs.C_miss[d, t] * d_miss[t, d] for t in inputs.T for d in inputs.D
    )
    obj2 = gp.quicksum(x_idle[j, t] for t in inputs.T for j in inputs.J)
    obj3 = gp.quicksum(x_p_ind[j, d] for j in inputs.J for d in inputs.D)
    obj4 = gp.quicksum(
        x_p_ind[j, d] * inputs.project_cohort_penalty[j, d]
        for j in inputs.J
        for d in inputs.D
    )
    model.setObjective(obj1 + obj2 + obj3 + obj4, GRB.MINIMIZE)

    # Constraints
    for t in inputs.T:
        for d in inputs.D:
            model.addConstr(
                gp.quicksum(x[j, t, d] * inputs.fte_list[j] for j in inputs.J)
                + d_miss[t, d]
                == inputs.demand[d, t],
                name=f"demand_balance_project_{d}_week_{t}",
            )

    for j in inputs.J:
        for t in inputs.T:
            model.addConstr(
                gp.quicksum(x[j, t, d] for d in inputs.D) + x_idle[j, t] == 1,
                name=f"employee_allocation_constraint_{j}_{t}",
            )

    for j in inputs.J:
        for t in inputs.T:
            for d in inputs.D:
                model.addConstr(
                    x_ind[j, t, d] >= x[j, t, d],
                    name=f"indicator_constraint_1_{j}_{t}_{d}",
                )
                model.addConstr(
                    x_ind[j, t, d] <= 1e8 * x[j, t, d],
                    name=f"indicator_constraint_0_{j}_{t}_{d}",
                )

    for j in inputs.J:
        for d in inputs.D:
            staffed_sum = gp.quicksum(x[j, t, d] for t in inputs.T)
            model.addConstr(
                100 * x_p_ind[j, d] >= staffed_sum, name=f"staffed_indicator_1_{j}_{d}"
            )
            model.addConstr(
                x_p_ind[j, d] <= staffed_sum, name=f"staffed_indicator_0_{j}_{d}"
            )

    for t in inputs.T:
        for j in inputs.J:
            model.addConstr(
                gp.quicksum(x_ind[j, t, d] for d in inputs.D)
                <= inputs.max_proj_list[j],
                name=f"max_concurrency_constraint_{t}_{j}",
            )

    for j in inputs.J:
        for d in inputs.D:
            model.addConstr(
                gp.quicksum(x[j, t, d] for t in inputs.T) >= inputs.d_p[j, d],
                name=f"specific_employee_staffing_constraint_{j}_{d}",
            )

    for d in inputs.D:
        model.addConstr(
            gp.quicksum(
                x[j, t, d]
                for t in inputs.T
                for j in inputs.J
                if j in inputs.oversight_idx_list
            )
            >= inputs.d_oversight_p[d],
            name=f"oversight_employee_constraint_{d}",
        )

    return model
