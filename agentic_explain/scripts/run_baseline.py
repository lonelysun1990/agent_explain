"""
Run the staffing optimization once and save baseline result, model.lp, and model.mps.
Run from project root: python -m agentic_explain.scripts.run_baseline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from gurobipy import GRB

from config.load_secrets import get_gurobi_env_kwargs
from use_case.staffing_model import (
    STAFFING_DATA_DIR,
    STAFFING_OUTPUTS_DIR,
    load_raw_data,
    process_data,
    build_gurobi_model,
)


def main(
    data_dir: str | Path | None = None,
    outputs_dir: str | Path | None = None,
    time_limit: int = 100,
) -> None:
    data_dir = Path(data_dir if data_dir is not None else STAFFING_DATA_DIR)
    outputs_dir = Path(outputs_dir if outputs_dir is not None else STAFFING_OUTPUTS_DIR)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(data_dir)
    inputs = process_data(
        raw["fte_mapping"],
        raw["concurrent_projects"],
        raw["oversight_ds_list"],
        raw["ds_list"],
        raw["project_list"],
    )

    env_kwargs = get_gurobi_env_kwargs()
    model = build_gurobi_model(inputs, env_kwargs)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.optimize()

    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print("Model did not reach optimal or time limit. Status:", model.status)
        sys.exit(1)

    baseline = {
        "status": "optimal" if model.status == GRB.OPTIMAL else "time_limit",
        "objective_value": model.ObjVal,
        "decision_variables": {v.VarName: v.X for v in model.getVars()},
    }

    baseline_path = outputs_dir / "baseline_result.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print("Wrote", baseline_path)

    lp_path = outputs_dir / "model.lp"
    mps_path = outputs_dir / "model.mps"
    model.write(str(lp_path))
    model.write(str(mps_path))
    print("Wrote", lp_path, mps_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="Default: use_case/staffing_model/data")
    p.add_argument("--outputs-dir", default=None, help="Default: use_case/staffing_model/outputs")
    p.add_argument("--time-limit", type=int, default=100)
    args = p.parse_args()
    main(data_dir=args.data_dir or STAFFING_DATA_DIR, outputs_dir=args.outputs_dir or STAFFING_OUTPUTS_DIR, time_limit=args.time_limit)
