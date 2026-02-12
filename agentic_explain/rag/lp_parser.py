"""
Parse Gurobi .lp file into RAG-ready chunks.

LP format: Minimize/Maximize, Subject To (constraint_name: expr), Bounds, Binaries/Generals, End.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_lp_file(lp_path: str | Path) -> list[dict[str, Any]]:
    """
    Parse a .lp file into chunks with metadata.

    Returns
    -------
    list of dicts: { "text": str, "metadata": { "source": "lp", "section", "constraint_name" or "variable_name" } }
    """
    path = Path(lp_path)
    content = path.read_text(encoding="utf-8")
    chunks = []
    lines = content.split("\n")

    section = None
    buffer = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("\\"):
            continue
        if stripped.startswith("Minimize") or stripped.startswith("Maximize"):
            if buffer and section:
                _flush_section(buffer, section, chunks, path)
            section = "objective"
            buffer = [stripped]
        elif stripped == "Subject To":
            if buffer and section:
                _flush_section(buffer, section, chunks, path)
            section = "constraints"
            buffer = []
        elif section == "constraints" and ":" in stripped:
            # Constraint: name: expr
            m = re.match(r"^([^:]+):\s*(.+)", stripped)
            if m:
                cname, expr = m.group(1).strip(), m.group(2).strip()
                chunks.append({
                    "text": f"Constraint: {cname}\n{expr}",
                    "metadata": {"source": "lp", "section": "constraints", "constraint_name": cname, "path": str(path)},
                })
        elif stripped == "Bounds" or stripped.startswith("Bounds"):
            if buffer and section == "constraints":
                _flush_section(buffer, section, chunks, path)
            section = "bounds"
            buffer = [stripped]
        elif stripped in ("Binaries", "Generals", "End") or stripped.startswith("Binaries") or stripped.startswith("Generals"):
            if buffer and section:
                _flush_section(buffer, section, chunks, path)
            if "Binaries" in stripped or "Generals" in stripped:
                section = "variables"
                buffer = [stripped]
            else:
                section = None
                buffer = []
        else:
            buffer.append(stripped)

    if buffer and section:
        _flush_section(buffer, section, chunks, path)

    return chunks


def _flush_section(buffer: list[str], section: str, chunks: list, path: Path) -> None:
    text = "\n".join(buffer).strip()
    if len(text) < 10:
        return
    if section == "objective":
        chunks.append({
            "text": f"Objective:\n{text}",
            "metadata": {"source": "lp", "section": "objective", "path": str(path)},
        })
    elif section == "constraints" and buffer:
        # Multi-line constraint block
        for line in buffer:
            if ":" in line:
                m = re.match(r"^([^:]+):\s*(.+)", line.strip())
                if m:
                    cname, expr = m.group(1).strip(), m.group(2).strip()
                    chunks.append({
                        "text": f"Constraint: {cname}\n{expr}",
                        "metadata": {"source": "lp", "section": "constraints", "constraint_name": cname, "path": str(path)},
                    })
    elif section == "bounds" or section == "variables":
        chunks.append({
            "text": text,
            "metadata": {"source": "lp", "section": section, "path": str(path)},
        })


# LP constraint name prefix -> semantic type (for grouping and graph nodes)
_CONSTRAINT_TYPE_PREFIXES: list[tuple[str, str]] = [
    ("demand_balance_project_", "demand_balance"),
    ("employee_allocation_constraint_", "employee_allocation"),
    ("indicator_constraint_", "indicator_constraint"),
    ("staffed_indicator_", "staffed_indicator"),
    ("max_concurrency_constraint_", "max_concurrency"),
    ("specific_employee_staffing_constraint_", "specific_employee_staffing"),
    ("oversight_employee_constraint_", "oversight_employee"),
]


def _constraint_name_to_type(cname: str, prefixes: list[tuple[str, str]] | None = None) -> str:
    """
    Map LP constraint name to semantic constraint type.
    E.g. demand_balance_project_0_week_0 -> demand_balance.
    If prefixes is None, uses default _CONSTRAINT_TYPE_PREFIXES.
    """
    use = prefixes if prefixes is not None else _CONSTRAINT_TYPE_PREFIXES
    for prefix, ctype in use:
        if cname.startswith(prefix):
            return ctype
    # Fallback: strip trailing _digits
    return re.sub(r"_\d+$", "", re.sub(r"_\d+_.*$", "", cname))


def _variable_families_from_expr(expr: str) -> set[str]:
    """Extract variable family names from LP expression (e.g. x[0,0,0] -> x)."""
    return set(re.findall(r"([a-zA-Z_]+)\s*\[", expr))


def extract_constraint_instances(
    lp_path: str | Path,
    constraint_type_prefixes: list[tuple[str, str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Parse LP file and return constraint instances grouped by constraint type.

    constraint_type_prefixes: optional list of (LP_prefix, semantic_type). If None, uses
        default staffing-style prefixes (for backward compatibility).

    Returns
    -------
    dict mapping constraint_type -> list of:
        - name: full constraint name
        - variable_families: set of variable names (e.g. {"x", "d_miss"})
        - expr: full expression string (LHS and RHS)
        - sample: short sample for display (first 200 chars of expr)
    """
    path = Path(lp_path)
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")
    result: dict[str, list[dict[str, Any]]] = {}
    in_constraints = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Subject To":
            in_constraints = True
            continue
        if stripped in ("Bounds", "Binaries", "Generals", "End"):
            in_constraints = False
            continue
        if not in_constraints or ":" not in stripped:
            continue
        m = re.match(r"^([^:]+):\s*(.+)", stripped)
        if not m:
            continue
        cname, expr = m.group(1).strip(), m.group(2).strip()
        ctype = _constraint_name_to_type(cname, constraint_type_prefixes)
        families = _variable_families_from_expr(expr)
        entry = {
            "name": cname,
            "variable_families": families,
            "expr": expr,
            "sample": (expr[:200] + "..." if len(expr) > 200 else expr),
        }
        result.setdefault(ctype, []).append(entry)
    return result


def extract_objective_variable_families(lp_path: str | Path) -> set[str]:
    """Parse LP objective section and return variable families that appear in it."""
    path = Path(lp_path)
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")
    families: set[str] = set()
    in_objective = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Minimize") or stripped.startswith("Maximize"):
            in_objective = True
            families |= _variable_families_from_expr(stripped)
            continue
        if in_objective:
            if stripped == "Subject To":
                break
            families |= _variable_families_from_expr(stripped)
    return families
