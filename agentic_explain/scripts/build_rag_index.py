"""
Build the unified RAG index from .py formulation, .lp, .mps, and data index mapping.

Run after run_baseline so that use_case/staffing_model/outputs/model.lp and model.mps exist.

  python -m agentic_explain.scripts.build_rag_index

Uses OPENAI_API_KEY from config/secrets.env if set (else HuggingFace fallback).
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.load_secrets import load_secrets
from use_case.staffing_model import STAFFING_DATA_DIR, STAFFING_OUTPUTS_DIR
from agentic_explain.rag.build_index import build_rag_index


def main(
    py_path: str | Path | None = None,
    lp_path: str | Path | None = None,
    mps_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    outputs_dir: str | Path | None = None,
) -> None:
    load_secrets()
    data_dir = Path(data_dir if data_dir is not None else STAFFING_DATA_DIR)
    outputs_dir = Path(outputs_dir if outputs_dir is not None else STAFFING_OUTPUTS_DIR)

    if py_path is None:
        py_path = _project_root / "use_case" / "staffing_model" / "staffing_model.py"
    py_path = Path(py_path)
    if lp_path is None:
        lp_path = outputs_dir / "model.lp"
    if mps_path is None:
        mps_path = outputs_dir / "model.mps"

    build_rag_index(
        py_path=py_path,
        lp_path=lp_path if Path(lp_path).exists() else None,
        mps_path=mps_path if Path(mps_path).exists() else None,
        data_dir=data_dir,
        persist_dir=outputs_dir / "rag_index",
    )
    print("RAG index built at", outputs_dir / "rag_index")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--py-path", default=None)
    p.add_argument("--lp-path", default=None)
    p.add_argument("--mps-path", default=None)
    p.add_argument("--data-dir", default=None, help="Default: use_case/staffing_model/data")
    p.add_argument("--outputs-dir", default=None, help="Default: use_case/staffing_model/outputs")
    args = p.parse_args()
    main(
        py_path=args.py_path,
        lp_path=args.lp_path,
        mps_path=args.mps_path,
        data_dir=args.data_dir or STAFFING_DATA_DIR,
        outputs_dir=args.outputs_dir or STAFFING_OUTPUTS_DIR,
    )
