"""
Run the agentic explainability workflow for one user query.
Prerequisites: baseline run and RAG index already built.

  python -m agentic_explain.scripts.run_workflow "Why was Josh not staffed on Ipp IO Pilot in week 6?"

Expects:
  - config/secrets.env with OPENAI_API_KEY and GUROBI_LICENSE_FILE
  - use_case/staffing_model/outputs/baseline_result.json (from run_baseline)
  - use_case/staffing_model/outputs/rag_index/ (from build_rag_index)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.load_secrets import load_secrets, get_gurobi_env_kwargs
from openai import OpenAI

from use_case.staffing_model import (
    STAFFING_DATA_DIR,
    STAFFING_OUTPUTS_DIR,
    load_raw_data,
    process_data,
    build_gurobi_model,
)
from agentic_explain.rag.plain_rag import build_plain_rag
from agentic_explain.workflow.graph import create_workflow, invoke_workflow


def main(
    user_query: str,
    *,
    data_dir: str | Path | None = None,
    outputs_dir: str | Path | None = None,
) -> None:
    load_secrets()
    data_dir = Path(data_dir if data_dir is not None else STAFFING_DATA_DIR)
    outputs_dir = Path(outputs_dir if outputs_dir is not None else STAFFING_OUTPUTS_DIR)

    baseline_path = outputs_dir / "baseline_result.json"
    if not baseline_path.exists():
        print("Run run_baseline first to create", baseline_path)
        sys.exit(1)
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_result = json.load(f)

    rag_dir = outputs_dir / "rag_index"
    py_path = _project_root / "use_case" / "staffing_model" / "staffing_model.py"
    lp_path = outputs_dir / "model.lp"
    mps_path = outputs_dir / "model.mps"
    rag_strategy = build_plain_rag(
        py_path, lp_path=lp_path, mps_path=mps_path, data_dir=data_dir, persist_dir=rag_dir
    )

    raw = load_raw_data(data_dir)
    inputs = process_data(
        raw["fte_mapping"],
        raw["concurrent_projects"],
        raw["oversight_ds_list"],
        raw["ds_list"],
        raw["project_list"],
    )
    env_kwargs = get_gurobi_env_kwargs()

    openai_client = OpenAI()

    workflow = create_workflow(
        openai_client=openai_client,
        rag_strategy=rag_strategy,
        baseline_result=baseline_result,
        data_dir=str(data_dir),
        build_model_fn=build_gurobi_model,
        inputs=inputs,
        env_kwargs=env_kwargs,
        outputs_dir=str(outputs_dir),
    )

    final = invoke_workflow(
        workflow,
        user_query,
        baseline_result=baseline_result,
    )

    print("\n--- Summary ---\n")
    print(final.get("final_summary", "No summary produced."))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run agentic explainability workflow")
    p.add_argument("query", nargs="?", default="Why was Josh not staffed on Ipp IO Pilot in week 6?")
    p.add_argument("--data-dir", default=None, help="Default: use_case/staffing_model/data")
    p.add_argument("--outputs-dir", default=None, help="Default: use_case/staffing_model/outputs")
    args = p.parse_args()
    main(args.query, data_dir=args.data_dir, outputs_dir=args.outputs_dir)
