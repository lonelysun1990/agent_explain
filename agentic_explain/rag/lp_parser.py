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
