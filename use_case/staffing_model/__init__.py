"""Staffing optimization use case: model, data, outputs, and Graph RAG config."""

from pathlib import Path

from use_case.staffing_model.staffing_model import (
    FORMULATION_DOCS,
    load_raw_data,
    process_data,
    build_gurobi_model,
    compute_objective_breakdown,
)
from use_case.staffing_model.rag_config import StaffingGraphRAGConfig

_ROOT = Path(__file__).resolve().parent
STAFFING_DATA_DIR = _ROOT / "data"
STAFFING_OUTPUTS_DIR = _ROOT / "outputs"

__all__ = [
    "FORMULATION_DOCS",
    "load_raw_data",
    "process_data",
    "build_gurobi_model",
    "compute_objective_breakdown",
    "StaffingGraphRAGConfig",
    "STAFFING_DATA_DIR",
    "STAFFING_OUTPUTS_DIR",
]
