"""
Staffing use-case implementation of GraphRAGConfig.

Loads entity names from ds_list.json and project_list.json in data_dir;
provides constraint-type prefixes, objective–variable mapping, and entity
extraction from LP constraint names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agentic_explain.rag.graph_rag_config import GraphRAGConfig

# Formulation docs come from the staffing model module
from use_case.staffing_model.staffing_model import FORMULATION_DOCS


class StaffingGraphRAGConfig:
    """GraphRAGConfig for the staffing optimization use case."""

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._employee_names: list[str] = []
        self._project_names: list[str] = []
        self._load_entity_lists()

    def _load_entity_lists(self) -> None:
        with open(self._data_dir / "ds_list.json", "r", encoding="utf-8") as f:
            ds_list = json.load(f)
        with open(self._data_dir / "project_list.json", "r", encoding="utf-8") as f:
            project_list = json.load(f)
        self._employee_names = [ds.get("name", f"employee_{i}") for i, ds in enumerate(ds_list)]
        self._project_names = [p.get("name", f"project_{d}") for d, p in enumerate(project_list)]

    def get_formulation_docs(self) -> str:
        return FORMULATION_DOCS

    def get_entity_name(self, entity_type: str, entity_id: int) -> str:
        if entity_type == "employee":
            if 0 <= entity_id < len(self._employee_names):
                return self._employee_names[entity_id]
            return f"employee_{entity_id}"
        if entity_type == "project":
            if 0 <= entity_id < len(self._project_names):
                return self._project_names[entity_id]
            return f"project_{entity_id}"
        return f"{entity_type}_{entity_id}"

    def get_entity_count(self, entity_type: str) -> int:
        if entity_type == "employee":
            return len(self._employee_names)
        if entity_type == "project":
            return len(self._project_names)
        return 0

    def constraint_name_to_entity(
        self, constraint_type: str, constraint_instance_name: str
    ) -> tuple[str, int] | None:
        ctype = constraint_type
        cname = constraint_instance_name
        if ctype == "demand_balance":
            m = re.match(r"demand_balance_project_(\d+)_week_\d+", cname)
            return ("project", int(m.group(1))) if m else None
        if ctype == "employee_allocation":
            m = re.match(r"employee_allocation_constraint_(\d+)_\d+", cname)
            return ("employee", int(m.group(1))) if m else None
        if ctype == "indicator_constraint":
            m = re.match(r"indicator_constraint_\d+_(\d+)_\d+_\d+", cname)
            return ("employee", int(m.group(1))) if m else None
        if ctype == "staffed_indicator":
            m = re.match(r"staffed_indicator_\d+_(\d+)_\d+", cname)
            return ("employee", int(m.group(1))) if m else None
        if ctype == "max_concurrency":
            m = re.match(r"max_concurrency_constraint_\d+_(\d+)", cname)
            return ("employee", int(m.group(1))) if m else None
        if ctype == "specific_employee_staffing":
            m = re.match(r"specific_employee_staffing_constraint_(\d+)_\d+", cname)
            return ("employee", int(m.group(1))) if m else None
        if ctype == "oversight_employee":
            m = re.match(r"oversight_employee_constraint_(\d+)", cname)
            return ("project", int(m.group(1))) if m else None
        return None

    def get_objective_variable_map(self) -> dict[str, set[str]]:
        return {
            "cost_of_missing_demand": {"d_miss"},
            "idle_time": {"x_idle"},
            "staffing_consistency": {"x_p_ind"},
            "out_of_cohort_penalty": {"x_p_ind"},
        }

    def get_variable_entity_scope(self, variable_family: str) -> str:
        if variable_family == "d_miss":
            return "project"
        return "employee"

    def get_constraint_type_prefixes(self) -> list[tuple[str, str]]:
        return [
            ("demand_balance_project_", "demand_balance"),
            ("employee_allocation_constraint_", "employee_allocation"),
            ("indicator_constraint_", "indicator_constraint"),
            ("staffed_indicator_", "staffed_indicator"),
            ("max_concurrency_constraint_", "max_concurrency"),
            ("specific_employee_staffing_constraint_", "specific_employee_staffing"),
            ("oversight_employee_constraint_", "oversight_employee"),
        ]
