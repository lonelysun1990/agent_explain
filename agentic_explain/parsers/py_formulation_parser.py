"""
Parse a Python file containing a Gurobi formulation into RAG-ready text chunks.

Extracts: FORMULATION_DOCS (problem overview, index mapping, variables, constraints,
objectives with business context), docstrings, variable definitions (addVars),
constraint naming patterns (addConstr name=). Chunks are tagged so retrieval can
match variable/constraint names to the math formulation.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def _extract_formulation_docs(source: str, path: Path) -> list[dict[str, Any]]:
    """Extract FORMULATION_DOCS constant and split into section chunks with metadata."""
    chunks = []
    # Match FORMULATION_DOCS = r"""...""" or FORMULATION_DOCS = """..."""
    match = re.search(
        r"FORMULATION_DOCS\s*=\s*r?[\"']{3}([\s\S]*?)[\"']{3}",
        source,
    )
    if not match:
        return chunks
    doc_text = match.group(1).strip()
    if not doc_text:
        return chunks

    # Normalize: ensure we have newlines before ## (raw string may have literal \n)
    doc_text = doc_text.replace("\\n", "\n")
    # Split by ## SECTION (allow start of string or newline before ##; capture section name)
    section_blocks = re.split(r"(?:^|\n)##\s+([A-Z_][A-Za-z_0-9]*)\s*\n", doc_text)
    if not section_blocks:
        return chunks
    # section_blocks[0] = leading text; [1]=section name, [2]=content, [3]=name, [4]=content, ...
    if section_blocks[0].strip() and not section_blocks[0].strip().startswith("##"):
        chunks.append({
            "text": section_blocks[0].strip(),
            "metadata": {"source": "py", "section": "formulation_docs", "path": str(path)},
        })
    # Iterate (name, content) pairs: (1,2), (3,4), (5,6), ...
    i = 1
    while i + 2 <= len(section_blocks):
        section_name = section_blocks[i].strip()
        section_content = section_blocks[i + 1].replace("\r\n", "\n").strip()
        section_key = section_name.lower()
        # Split by ### subsection (^ matches line start with MULTILINE)
        sub_blocks = re.split(r"^###\s+([^\n]+)\s*\n", section_content, flags=re.MULTILINE)
        if len(sub_blocks) <= 1:
            chunks.append({
                "text": section_content,
                "metadata": {"source": "py", "section": section_key, "path": str(path)},
            })
        else:
            # First element may be intro before any ###
            if sub_blocks[0].strip():
                chunks.append({
                    "text": sub_blocks[0].strip(),
                    "metadata": {"source": "py", "section": section_key, "path": str(path)},
                })
            for j in range(1, len(sub_blocks), 2):
                if j + 1 >= len(sub_blocks):
                    break
                entity_name = sub_blocks[j].strip()
                entity_content = sub_blocks[j + 1].strip()
                meta = {"source": "py", "section": section_key, "path": str(path)}
                entity_key = entity_name.split()[0] if entity_name else ""
                if section_key == "variables":
                    meta["variable_name"] = entity_key
                elif section_key == "constraints":
                    meta["constraint_name"] = entity_key
                elif section_key == "objectives":
                    meta["objective_name"] = entity_key
                elif section_key == "index_mapping":
                    meta["index_name"] = entity_key
                chunks.append({"text": f"### {entity_name}\n{entity_content}", "metadata": meta})
        i += 2

    return chunks


def parse_py_formulation(py_path: str | Path) -> list[dict[str, Any]]:
    """
    Parse a .py formulation file into chunks for RAG.

    Returns
    -------
    list of dicts with keys: text, metadata (source=py, section, variable_name, constraint_name, etc.)
    """
    path = Path(py_path)
    source = path.read_text(encoding="utf-8")
    chunks = []

    # 0) FORMULATION_DOCS: problem overview, index mapping, variables, constraints, objectives
    chunks.extend(_extract_formulation_docs(source, path))

    # 1) Module-level docstring as overview (if not already covered)
    try:
        tree = ast.parse(source)
        if ast.get_docstring(tree):
            doc = ast.get_docstring(tree)
            if doc and not any(c.get("metadata", {}).get("section") == "overview" for c in chunks):
                chunks.append({
                    "text": f"Formulation overview:\n{doc}",
                    "metadata": {"source": "py", "section": "overview", "path": str(path)},
                })
    except SyntaxError:
        pass

    # 2) Extract addVars / addConstr from source (regex for robustness)
    # Variables: model.addVars(..., name="x") or name='x'
    var_pattern = re.compile(
        r"model\.addVars\s*\([^)]*\)\s*(?:,|)\s*name\s*=\s*[\"'](\w+)[\"']",
        re.DOTALL,
    )
    for m in var_pattern.finditer(source):
        var_name = m.group(1)
        # Get a few lines of context (comment above or line)
        start = max(0, source.rfind("\n", 0, m.start()) - 1)
        block = source[start : m.end() + 200].split("\n")[:6]
        context = "\n".join(block).strip()
        chunks.append({
            "text": f"Variable: {var_name}\n\nCode context:\n{context}",
            "metadata": {"source": "py", "section": "variables", "variable_name": var_name, "path": str(path)},
        })

    # Constraints: model.addConstr(..., name=f"..." or name="...")
    constr_name_pattern = re.compile(
        r"model\.addConstr\s*\([^)]+\)\s*,\s*name\s*=\s*(?:f)?[\"']([^\"']+)[\"']",
        re.DOTALL,
    )
    for m in constr_name_pattern.finditer(source):
        name_expr = m.group(1)
        # If it's f"demand_balance_project_{d}_week_{t}", store pattern
        if "{" in name_expr:
            pattern = name_expr.replace("{", "").replace("}", "").replace("_project_", "_project_%d_").replace("_week_", "_week_%d_")
            # Simplify to a readable pattern
            pattern_clean = re.sub(r"%d", "N", pattern)
        else:
            pattern_clean = name_expr
        start = max(0, source.rfind("\n", 0, m.start()))
        block = source[start : m.end() + 150].split("\n")[:5]
        context = "\n".join(block).strip()
        chunks.append({
            "text": f"Constraint name pattern: {pattern_clean}\n\nCode context:\n{context}",
            "metadata": {
                "source": "py",
                "section": "constraints",
                "constraint_name_pattern": pattern_clean,
                "path": str(path),
            },
        })

    # 3) Section headers / comments as documentation (markdown-style or # blocks)
    lines = source.split("\n")
    current_section = None
    buffer = []
    for i, line in enumerate(lines):
        if re.match(r"^\s*#+\s*(Variables|Constraints|Objectives|Data|Model building)", line, re.I):
            if buffer and current_section:
                text = "\n".join(buffer).strip()
                if len(text) > 80:
                    chunks.append({
                        "text": text,
                        "metadata": {"source": "py", "section": current_section, "path": str(path)},
                    })
            current_section = re.search(r"#+\s*(\w+)", line, re.I)
            current_section = current_section.group(1).lower() if current_section else "other"
            buffer = [line]
        elif line.strip().startswith("#") or (buffer and not line.strip().startswith("def ") and not line.strip().startswith("class ")):
            buffer.append(line)
        elif buffer and current_section and line.strip() and not line.strip().startswith("#"):
            buffer.append(line)
            if len(buffer) >= 15:
                text = "\n".join(buffer).strip()
                if len(text) > 80:
                    chunks.append({
                        "text": text,
                        "metadata": {"source": "py", "section": current_section or "code", "path": str(path)},
                    })
                buffer = []
    if buffer and current_section:
        text = "\n".join(buffer).strip()
        if len(text) > 80:
            chunks.append({
                "text": text,
                "metadata": {"source": "py", "section": current_section, "path": str(path)},
            })

    # 4) Function docstrings for process_data and build_gurobi_model
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node)
                if doc and node.name in ("process_data", "build_gurobi_model", "load_raw_data"):
                    chunks.append({
                        "text": f"Function {node.name}:\n{doc}",
                        "metadata": {"source": "py", "section": "functions", "function": node.name, "path": str(path)},
                    })
    except SyntaxError:
        pass

    return chunks
