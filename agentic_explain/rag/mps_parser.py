"""
Parse Gurobi .mps file to extract variable and constraint names for RAG.

MPS is column/row oriented; we extract names from ROWS and COLUMNS sections.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_mps_file(mps_path: str | Path) -> list[dict[str, Any]]:
    """
    Parse .mps file into chunks (name lists and optional short descriptions).

    Returns
    -------
    list of dicts with text and metadata source=mps, section=rows|columns.
    """
    path = Path(mps_path)
    content = path.read_text(encoding="utf-8")
    chunks = []
    lines = content.split("\n")

    section = None
    row_names = []
    col_names = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[0] == "ROWS":
            section = "rows"
            continue
        if parts[0] == "COLUMNS":
            section = "columns"
            continue
        if parts[0] in ("RHS", "RANGES", "BOUNDS", "ENDATA"):
            if section == "rows" and row_names:
                text = "Constraint (row) names in MPS:\n" + "\n".join(sorted(set(row_names))[:200])
                chunks.append({
                    "text": text,
                    "metadata": {"source": "mps", "section": "constraints", "path": str(path)},
                })
            if section == "columns" and col_names:
                text = "Variable (column) names in MPS:\n" + "\n".join(sorted(set(col_names))[:200])
                chunks.append({
                    "text": text,
                    "metadata": {"source": "mps", "section": "variables", "path": str(path)},
                })
            section = None
            continue
        if section == "rows" and len(parts) >= 2:
            # N row_name (N = type, row_name is constraint name)
            row_names.append(parts[1])
        if section == "columns" and len(parts) >= 2:
            col_names.append(parts[1])

    if section == "rows" and row_names:
        text = "Constraint (row) names in MPS:\n" + "\n".join(sorted(set(row_names))[:200])
        chunks.append({
            "text": text,
            "metadata": {"source": "mps", "section": "constraints", "path": str(path)},
        })
    if section == "columns" and col_names:
        text = "Variable (column) names in MPS:\n" + "\n".join(sorted(set(col_names))[:200])
        chunks.append({
            "text": text,
            "metadata": {"source": "mps", "section": "variables", "path": str(path)},
        })

    return chunks
