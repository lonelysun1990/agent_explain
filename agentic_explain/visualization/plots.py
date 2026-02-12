"""
Visualization utilities for staffing optimization results.

All plotting functions accept numpy arrays and metadata lists (ds_list, project_list).
Use `parse_decision_variables()` to convert a Gurobi decision-variable dict
(var_name -> value) into the numpy arrays expected by the plot functions.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_decision_variables(
    decision_variables: dict[str, float],
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Convert a flat {var_name: value} dict into named numpy arrays.

    Parameters
    ----------
    decision_variables : dict
        Keys like ``"x[0,1,2]"`` and float values.
    eps : float
        Values below *eps* are snapped to 0; values within *eps* of 1 are
        snapped to 1.

    Returns
    -------
    dict[str, np.ndarray]
        One array per variable family (``x``, ``x_ind``, ``x_p_ind``,
        ``d_miss``, ``x_idle``).
    """
    grouped: dict[str, dict[tuple[int, ...], float]] = {}
    for key, value in decision_variables.items():
        var_name, idx_str = key.split("[")
        indices = tuple(int(i) for i in idx_str.rstrip("]").split(","))
        grouped.setdefault(var_name, {})[indices] = value

    result: dict[str, np.ndarray] = {}
    for var_name, data in grouped.items():
        shape = tuple(max(idx[dim] for idx in data) + 1 for dim in range(len(next(iter(data)))))
        arr = np.zeros(shape)
        for indices, value in data.items():
            v = 0.0 if abs(value) < eps else (1.0 if abs(1.0 - value) < eps else value)
            arr[indices] = v
        result[var_name] = arr

    # Clean up indicator variables from continuous values
    if "x" in result:
        result["x_ind"] = (result["x"] > 0).astype(int)
        result["x_p_ind"] = (result["x"].sum(axis=1) > 0).astype(int)

    return result


# ---------------------------------------------------------------------------
# Objective summary
# ---------------------------------------------------------------------------

def print_objective_summary(
    result: dict[str, np.ndarray],
    inputs: dict[str, Any],
) -> dict[str, float]:
    """Print and return the four objective components.

    Parameters
    ----------
    result : dict from ``parse_decision_variables``
    inputs : processed model inputs (from ``process_data``)

    Returns
    -------
    dict with keys ``missing_demand``, ``idle_time``, ``staffing_consistency``,
    ``out_of_cohort_penalty``, ``total``.
    """
    d_miss = result["d_miss"]
    x_idle = result["x_idle"]
    x_p_ind = result["x_p_ind"]

    C_miss = inputs["C_miss"]
    cohort_pen = inputs["project_cohort_penalty"]

    obj1 = float(np.sum(C_miss * d_miss.T))       # C_miss is (D,T), d_miss is (T,D)
    obj2 = float(np.sum(x_idle))
    obj3 = float(np.sum(x_p_ind))
    obj4 = float(np.sum(x_p_ind * cohort_pen))
    total = obj1 + obj2 + obj3 + obj4

    print(f"  Missing demand cost : {obj1:10.4f}")
    print(f"  Idle time           : {obj2:10.4f}")
    print(f"  Staffing consistency: {obj3:10.4f}")
    print(f"  Out-of-cohort pen.  : {obj4:10.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  Total               : {total:10.4f}")

    return {
        "missing_demand": obj1,
        "idle_time": obj2,
        "staffing_consistency": obj3,
        "out_of_cohort_penalty": obj4,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Input visualization
# ---------------------------------------------------------------------------

def plot_demand(
    demand: np.ndarray,
    project_list: list[dict],
    *,
    ylabel: str = "FTE Demand",
    title: str = "Project Demand Over Time",
    figsize: tuple[float, float] = (12, 3.5),
) -> plt.Figure:
    """Line chart of per-project demand over weeks.

    Parameters
    ----------
    demand : np.ndarray, shape (n_projects, horizon)
    project_list : list of dicts with ``"name"`` key
    """
    fig, ax = plt.subplots(figsize=figsize)
    for d_idx in range(len(project_list)):
        ax.plot(demand[d_idx, :], label=project_list[d_idx]["name"], linewidth=1.4)
    ax.set_xlabel("Week")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        fancybox=True, shadow=True, ncol=4, fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Stacked bar – per-employee allocation
# ---------------------------------------------------------------------------

def plot_employee_allocations(
    allocation: np.ndarray,
    ds_list: list[dict],
    project_list: list[dict],
    *,
    figsize_per_row: float = 2.5,
    ncols: int = 2,
) -> plt.Figure:
    """Stacked bar chart of each employee's weekly allocation across projects.

    Parameters
    ----------
    allocation : np.ndarray, shape (n_employees, horizon, n_projects)
    ds_list : list of dicts with ``"name"``
    project_list : list of dicts with ``"name"``
    """
    n_employees, n_weeks, n_projects = allocation.shape
    nrows = (n_employees + ncols - 1) // ncols

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(6 * ncols, figsize_per_row * nrows),
        squeeze=False,
    )

    palette = sns.color_palette("husl", n_projects)

    for emp_id, ax in enumerate(axs.flat):
        if emp_id >= n_employees:
            ax.axis("off")
            continue
        emp_alloc = allocation[emp_id]  # (horizon, n_projects)
        bottom = np.zeros(n_weeks)
        for p_id in range(n_projects):
            proj_data = emp_alloc[:, p_id]
            if proj_data.sum() > 0:
                ax.bar(
                    range(n_weeks), proj_data, bottom=bottom,
                    color=palette[p_id], alpha=0.85,
                    label=project_list[p_id]["name"],
                )
            bottom += proj_data
        ax.set_title(f"{ds_list[emp_id]['name']}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Week", fontsize=8)
        ax.set_ylabel("Allocation", fontsize=8)
        ax.set_xticks(range(n_weeks))
        ax.tick_params(labelsize=7)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels, loc="upper center",
            bbox_to_anchor=(0.5, -0.55), fancybox=True, shadow=True,
            ncol=2, fontsize=6,
        )

    fig.suptitle("Employee Allocation by Project", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    plt.subplots_adjust(hspace=2.2)
    return fig


# ---------------------------------------------------------------------------
# Gantt chart
# ---------------------------------------------------------------------------

def plot_gantt_chart(
    allocation: np.ndarray,
    ds_list: list[dict],
    project_list: list[dict],
    max_concurrent: int | None = None,
    *,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Gantt-style chart showing staffing allocation across employees and weeks.

    Parameters
    ----------
    allocation : np.ndarray, shape (n_employees, horizon, n_projects)
    ds_list : list of dicts with ``"name"``
    project_list : list of dicts with ``"name"``
    max_concurrent : max rows per employee (auto-detected if None)
    """
    n_employees, n_weeks, n_projects = allocation.shape
    palette = sns.color_palette("husl", n_projects)

    if max_concurrent is None:
        x_p_ind = (allocation.sum(axis=1) > 0).astype(int)  # (emp, proj)
        max_concurrent = int(np.max(x_p_ind.sum(axis=1)))
    row_max = max(max_concurrent, 1)

    if figsize is None:
        figsize = (max(12, n_weeks * 0.5), max(8, n_employees * row_max * 0.35))

    fig, ax = plt.subplots(figsize=figsize)

    row_allocation: dict[int, dict[int, int]] = {}
    for i in range(n_employees):
        row_allocation[i] = {}
        for j in range(n_projects):
            for k in range(n_weeks):
                if allocation[i, k, j] > 0:
                    if j in row_allocation[i]:
                        plotting_row = row_allocation[i][j]
                    else:
                        used = set(row_allocation[i].values())
                        available = [r for r in range(n_projects) if r not in used]
                        plotting_row = available[0] if available else 0
                        row_allocation[i][j] = plotting_row
                    ax.barh(
                        row_max * i + plotting_row, 1, left=k,
                        color=palette[j], edgecolor="white", linewidth=0.3,
                    )

    # Y-axis labels
    label_list = []
    for i in range(n_employees):
        for r in range(row_max):
            label_list.append(ds_list[i]["name"] if r == 0 else "")
    ax.set_yticks(range(n_employees * row_max))
    ax.set_yticklabels(label_list, fontsize=8)
    ax.set_xlabel("Week")
    ax.set_title("Staffing Allocation Gantt Chart", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    # Legend
    for i in range(n_projects):
        ax.bar(0, 0, color=palette[i], label=project_list[i]["name"], edgecolor="black")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Staffing shortage line chart
# ---------------------------------------------------------------------------

def plot_staffing_shortage(
    d_miss: np.ndarray,
    project_list: list[dict],
    *,
    figsize: tuple[float, float] = (12, 3),
) -> plt.Figure:
    """Line chart of unmet staffing demand per project over weeks.

    Parameters
    ----------
    d_miss : np.ndarray, shape (horizon, n_projects)
    project_list : list of dicts with ``"name"``
    """
    fig, ax = plt.subplots(figsize=figsize)
    n_projects = d_miss.shape[1]
    for p_idx in range(n_projects):
        if d_miss[:, p_idx].sum() > 0:
            ax.plot(d_miss[:, p_idx], label=project_list[p_idx]["name"], linewidth=1.4)
    ax.set_xlabel("Week")
    ax.set_ylabel("Staffing Shortage (FTE)")
    ax.set_title("Unmet Staffing Demand by Project", fontsize=12, fontweight="bold")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.2),
        fancybox=True, shadow=True, ncol=4, fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Staffing idle line chart
# ---------------------------------------------------------------------------

def plot_staffing_idle(
    x_idle: np.ndarray,
    ds_list: list[dict],
    *,
    figsize: tuple[float, float] = (12, 3),
) -> plt.Figure:
    """Line chart of employee idle time over weeks.

    Parameters
    ----------
    x_idle : np.ndarray, shape (n_employees, horizon)
    ds_list : list of dicts with ``"name"``
    """
    fig, ax = plt.subplots(figsize=figsize)
    n_employees = x_idle.shape[0]
    for emp_idx in range(n_employees):
        ax.plot(x_idle[emp_idx, :], label=ds_list[emp_idx]["name"], linewidth=1.4)
    ax.set_xlabel("Week")
    ax.set_ylabel("Idle Time (%)")
    ax.set_title("Employee Idle Time", fontsize=12, fontweight="bold")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.2),
        fancybox=True, shadow=True, ncol=5, fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Demand vs. supply (understaffing cost) heatmap
# ---------------------------------------------------------------------------

def plot_understaffing_cost(
    C_miss: np.ndarray,
    project_list: list[dict],
    *,
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Heatmap of the understaffing penalty C_miss[project, week].

    Parameters
    ----------
    C_miss : np.ndarray, shape (n_projects, horizon)
    project_list : list of dicts with ``"name"``
    """
    fig, ax = plt.subplots(figsize=figsize)
    proj_names = [p["name"] for p in project_list]
    sns.heatmap(
        C_miss, ax=ax, cmap="YlOrRd", linewidths=0.3,
        yticklabels=proj_names, xticklabels=range(C_miss.shape[1]),
        cbar_kws={"label": "Penalty"},
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Project")
    ax.set_title("Understaffing Cost Penalty (C_miss)", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig
