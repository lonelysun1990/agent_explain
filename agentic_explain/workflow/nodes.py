"""
Workflow nodes: query, entity resolution, constraint generation, counterfactual run, compare, ilp analysis, summarize.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_explain.workflow.state import AgentState


def make_query_node():
    def query_node(state: "AgentState", *, openai_client) -> "AgentState":
        user_query = state.get("user_query", "")
        if not user_query:
            state["reformulated_query"] = ""
            return state
        sys = (
            "You are a reasoning assistant. Identify whether the user's question about an optimization result "
            "is a counterfactual ('why not') or a direct instruction ('force'). "
            "If counterfactual, rephrase as a forcing instruction (e.g. 'Force X to happen'). "
            "If direct, keep as is. Output only the rephrased or unchanged query, one line."
        )
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user_query}],
            max_tokens=200,
        )
        reformulated = (r.choices[0].message.content or "").strip()
        state["reformulated_query"] = reformulated or user_query
        return state
    return query_node


def make_entity_resolution_node(data_dir: str | Path):
    data_dir = Path(data_dir)
    with open(data_dir / "ds_list.json", "r", encoding="utf-8") as f:
        ds_list = json.load(f)
    with open(data_dir / "project_list.json", "r", encoding="utf-8") as f:
        project_list = json.load(f)
    name_to_j = {ds["name"]: j for j, ds in enumerate(ds_list)}
    name_to_d = {p["name"]: d for d, p in enumerate(project_list)}

    def entity_resolution_node(state: "AgentState") -> "AgentState":
        query = state.get("reformulated_query", "")
        resolved = {}
        rewritten = query
        for name, j in name_to_j.items():
            if name in rewritten:
                rewritten = rewritten.replace(name, f"j={j}")
                resolved["j"] = resolved.get("j", {})
                resolved["j"][name] = j
        for name, d in name_to_d.items():
            if name in rewritten:
                rewritten = rewritten.replace(name, f"d={d}")
                resolved["d"] = resolved.get("d", {})
                resolved["d"][name] = d
        week_m = re.search(r"[Ww]eek\s+(\d+)", rewritten)
        if week_m:
            t_val = int(week_m.group(1))
            rewritten = re.sub(r"[Ww]eek\s+\d+", f"t={t_val}", rewritten)
            resolved["t"] = {"Week " + week_m.group(1): t_val}
        state["resolved_entities"] = resolved
        state["reformulated_query"] = rewritten
        return state
    return entity_resolution_node


def make_constraint_generation_node(rag_index, openai_client, variable_names: list[str]):
    def constraint_generation_node(state: "AgentState") -> "AgentState":
        query = state.get("reformulated_query", "")
        if not query:
            state["constraint_expressions"] = []
            return state
        retriever = rag_index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)
        context = "\n".join([n.text for n in nodes])
        vars_str = ", ".join(variable_names)
        sys = (
            f"You translate a user request into one or more Python constraint expressions for a Gurobi model. "
            f"Available variables (use exact names): {vars_str}. "
            f"Format: variable_name[i,j,k] == value (integer indices, no spaces in brackets). "
            f"Examples: x_ind[0,6,10] == 1 (force employee 0 on project 10 in week 6), d_miss[23,21] == 0 (no unmet demand for project 21 in week 23). "
            f"Output only the constraint line(s), one per line, no explanation."
        )
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": f"RAG context:\n{context}\n\nUser request:\n{query}"},
            ],
            max_tokens=300,
        )
        raw = (r.choices[0].message.content or "").strip()
        exprs = re.findall(r"\w+\s*\[\s*[^\]]+\s*\]\s*==\s*[\d.]+", raw)
        valid = [e for e in exprs if e.split("[")[0].strip() in variable_names]
        state["constraint_expressions"] = valid
        return state
    return constraint_generation_node


def make_counterfactual_run_node(build_model_fn, inputs: dict, env_kwargs: dict, outputs_dir: str | Path):
    from agentic_explain.tools.counterfactual_run import run_counterfactual

    outputs_dir = Path(outputs_dir)

    def counterfactual_run_node(state: "AgentState") -> "AgentState":
        exprs = state.get("constraint_expressions", [])
        if not exprs:
            state["counterfactual_status"] = "error"
            state["counterfactual_result"] = {"error": "No constraint expressions"}
            return state
        ilp_path = outputs_dir / "counterfactual.ilp"
        result = run_counterfactual(
            build_model_fn=build_model_fn,
            inputs=inputs,
            env_kwargs=env_kwargs,
            constraint_expressions=exprs,
            ilp_path=ilp_path,
        )
        state["counterfactual_status"] = result["status"]
        state["counterfactual_result"] = result
        if result.get("ilp_path"):
            state["ilp_path"] = result["ilp_path"]
        return state
    return counterfactual_run_node


def make_compare_node():
    def compare_node(state: "AgentState") -> "AgentState":
        baseline = state.get("baseline_result", {})
        cf = state.get("counterfactual_result", {})
        if not baseline or not cf.get("decision_variables"):
            state["comparison_summary"] = "Cannot compare: missing baseline or counterfactual solution."
            return state
        base_obj = baseline.get("objective_value")
        cf_obj = cf.get("objective_value")
        base_vars = baseline.get("decision_variables", {})
        cf_vars = cf.get("decision_variables", {})
        lines = [
            f"Baseline objective: {base_obj}",
            f"Counterfactual objective: {cf_obj}",
            f"Difference: {cf_obj - base_obj if base_obj is not None and cf_obj is not None else 'N/A'}",
        ]
        key_vars = [k for k in cf_vars if k.startswith("d_miss") or k.startswith("x_idle")]
        for k in key_vars[:20]:
            bv = base_vars.get(k, 0)
            cv = cf_vars.get(k, 0)
            if abs(cv - bv) > 1e-6:
                lines.append(f"  {k}: baseline={bv:.4f} counterfactual={cv:.4f}")
        state["comparison_summary"] = "\n".join(lines)
        return state
    return compare_node


def make_ilp_analysis_node(rag_index, openai_client):
    from agentic_explain.rag.ilp_parser import parse_ilp_file

    def ilp_analysis_node(state: "AgentState") -> "AgentState":
        ilp_path = state.get("ilp_path")
        if not ilp_path or not Path(ilp_path).exists():
            state["conflict_explanation"] = "No IIS file available."
            return state
        parsed = parse_ilp_file(ilp_path)
        constraint_names = parsed.get("constraint_names", [])
        if not constraint_names:
            state["conflict_explanation"] = "IIS parsed but no constraint names extracted."
            return state
        retriever = rag_index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve("constraint " + " ".join(constraint_names[:5]))
        context = "\n".join([n.text for n in nodes])
        sys = (
            "You explain in 2–4 sentences which existing model constraints conflict with the user's requested constraint, "
            "based on the IIS constraint names and the formulation context. Use business terms (e.g. demand balance, employee allocation)."
        )
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": f"IIS constraints: {constraint_names}\n\nContext:\n{context}"},
            ],
            max_tokens=250,
        )
        state["conflict_explanation"] = (r.choices[0].message.content or "").strip()
        return state
    return ilp_analysis_node


def make_summarize_node(openai_client):
    def summarize_node(state: "AgentState") -> "AgentState":
        user_query = state.get("user_query", "")
        status = state.get("counterfactual_status", "")
        if status == "feasible":
            comp = state.get("comparison_summary", "")
            sys = "Summarize the trade-off between the baseline and the user's requested scenario in 2–4 sentences. Mention objective and key variable changes."
            r = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": f"User asked: {user_query}\n\nComparison:\n{comp}"},
                ],
                max_tokens=300,
            )
            state["final_summary"] = (r.choices[0].message.content or comp).strip()
        elif status == "infeasible":
            conflict = state.get("conflict_explanation", "")
            sys = "In one or two sentences, tell the user that their requested change is infeasible and which constraints conflict."
            r = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": f"User asked: {user_query}\n\nConflict analysis:\n{conflict}"},
                ],
                max_tokens=200,
            )
            state["final_summary"] = (r.choices[0].message.content or conflict).strip()
        else:
            err = state.get("counterfactual_result", {}).get("error", "Unknown error")
            state["final_summary"] = f"The counterfactual run failed: {err}"
        return state
    return summarize_node
