"""
Run the agentic explainability workflow for one user query.
Prerequisites: baseline run and RAG index already built.

  python -m agentic_explain.scripts.run_workflow "Why was Josh not staffed on Ipp IO Pilot in week 6?"

Expects from project root:
  - config/secrets.env (or secrets.env) with OPENAI_API_KEY and GUROBI_LICENSE_FILE
  - outputs/baseline_result.json (from run_baseline)
  - outputs/rag_index/ (from build_rag_index)
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

from agentic_explain.staffing_model import load_raw_data, process_data, build_gurobi_model
from agentic_explain.rag.build_index import load_rag_index
from agentic_explain.workflow.graph import create_workflow, invoke_workflow


def main(
    user_query: str,
    *,
    data_dir: str | Path = "data",
    outputs_dir: str | Path = "outputs",
) -> None:
    load_secrets()
    outputs_dir = Path(outputs_dir)
    data_dir = Path(data_dir)

    baseline_path = outputs_dir / "baseline_result.json"
    if not baseline_path.exists():
        print("Run run_baseline first to create", baseline_path)
        sys.exit(1)
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_result = json.load(f)

    rag_dir = outputs_dir / "rag_index"
    if not rag_dir.exists():
        print("Build RAG index first (e.g. run build_rag_index). Expected at", rag_dir)
        sys.exit(1)
    rag_index = load_rag_index(persist_dir=rag_dir)

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
        rag_index=rag_index,
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
        rag_index=rag_index,
    )

    print("\n--- Summary ---\n")
    print(final.get("final_summary", "No summary produced."))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run agentic explainability workflow")
    p.add_argument("query", nargs="?", default="Why was Josh not staffed on Ipp IO Pilot in week 6?")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--outputs-dir", default="outputs")
    args = p.parse_args()
    main(args.query, data_dir=args.data_dir, outputs_dir=args.outputs_dir)
