"""
Parse Gurobi .ilp (IIS) file to extract constraint and variable names for conflict analysis.

.ilp format is similar to .lp: Subject To, Bounds, etc., but only the subset in the IIS.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_ilp_file(ilp_path: str | Path) -> dict[str, Any]:
    """
    Parse an .ilp file (IIS) into constraint names and variable names.

    Returns
    -------
    dict with keys: constraint_names (list), variable_names (list), raw_lines (list of str).
    """
    path = Path(ilp_path)
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")

    constraint_names = []
    variable_names = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("\\"):
            continue
        if stripped == "Subject To":
            continue
        if ":" in stripped:
            m = re.match(r"^([^:]+):\s*(.+)", stripped)
            if m:
                cname = m.group(1).strip()
                constraint_names.append(cname)
                # Extract variable names from RHS (e.g. x[0,6,10], d_miss[1,2])
                expr = m.group(2)
                for var in re.findall(r"\b([a-zA-Z_]\w*)\s*\[", expr):
                    variable_names.add(var)
                for var in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", expr):
                    if not var.upper() in ("E", "L", "G", "N", "INF"):
                        variable_names.add(var)
        if stripped == "Bounds":
            continue
        # Bounds section: variable name
        parts = stripped.split()
        for p in parts:
            if "[" in p:
                base = p.split("[")[0]
                if base and base not in ("<", ">", "="):
                    variable_names.add(base)

    return {
        "constraint_names": constraint_names,
        "variable_names": list(variable_names),
        "raw_lines": lines,
        "path": str(path),
    }
