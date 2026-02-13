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


def make_query_node(temperature: float = 0):
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
            temperature=temperature,
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


def make_constraint_generation_node(rag_strategy, openai_client, variable_names: list[str], temperature: float = 0):
    def constraint_generation_node(state: "AgentState") -> "AgentState":
        query = state.get("reformulated_query", "")
        if not query:
            state["constraint_expressions"] = []
            return state
        chunks = rag_strategy.retrieve(query, top_k=5)
        context = "\n".join([c.text for c in chunks])

        # --- Capture retrieval debug info ---
        rag_debug = state.get("rag_retrieval_debug", {})
        rag_debug["strategy"] = getattr(rag_strategy, "name", "unknown")
        rag_debug["constraint_generation"] = {
            "query": query,
            "top_k": 5,
            "chunks": [
                {
                    "text": c.text,
                    "score": c.score,
                    "metadata": {k: str(v) for k, v in (c.metadata or {}).items()},
                }
                for c in chunks
            ],
        }
        state["rag_retrieval_debug"] = rag_debug

        vars_str = ", ".join(variable_names)
        # Build generic examples from the actual variable names (no hardcoded model-specific info)
        examples = []
        for vn in variable_names[:3]:
            examples.append(f"{vn}[0,0] == 1")
        examples_str = ", ".join(examples)
        sys_msg = (
            f"You translate a user request into one or more constraint expressions for a Gurobi optimization model. "
            f"Available decision variables (use exact names): {vars_str}. "
            f"Format: variable_name[index1,index2,...] == value or >= value or <= value. "
            f"Value can be a number (e.g. 0, 1) or a data parameter (e.g. D[t,d] for demand). "
            f"You may use a single variable or a sum of variables (e.g. x[j,t,d] + x[j+1,t,d] + ... >= D[d,t]). "
            f"Example format: {examples_str}. For 'force no unmet demand': d_miss[t,d] == 0. "
            f"Use the RAG context below to understand variable dimensions and index meanings. "
            f"Output only the constraint line(s), one per line, no explanation."
        )
        user_msg = f"RAG context:\n{context}\n\nUser request:\n{query}"
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
            temperature=temperature,
        )
        raw = (r.choices[0].message.content or "").strip()

        # --- Capture LLM messages debug ---
        llm_debug = state.get("llm_messages_debug", {})
        llm_debug["constraint_generation"] = {
            "system": sys_msg,
            "user": user_msg,
            "raw_response": raw,
        }
        state["llm_messages_debug"] = llm_debug

        # --- Parse constraint expressions from LLM output ---
        # Accept: (1) single variable[indices] == number or == param[indices];
        #         (2) single variable >=/<= number or param[indices];
        #         (3) sum of variables >=/==/<= number or param[indices] (data params treated like numbers).
        def _parse_exprs(text: str) -> list[str]:
            results: list[str] = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Linear (sum) first: full line "term + term + ... op value"
                if "+" in line and any(op in line for op in (">=", "<=", "==")):
                    m = re.match(r"^(.+?)\s*(>=|<=|==)\s*([\d.]+|\w+\s*\[\s*[^\]]+\s*\])\s*$", line)
                    if m:
                        lhs = m.group(1)
                        terms = re.findall(r"(\w+)\s*\[\s*[^\]]+\s*\]", lhs)
                        if terms and all(t in variable_names for t in terms):
                            results.append(line)
                            continue
                # Simple: single var op number or param
                for pat in (
                    r"\w+\s*\[\s*[^\]]+\s*\]\s*==\s*[\d.]+",
                    r"\w+\s*\[\s*[^\]]+\s*\]\s*=\s*[\d.]+",
                    r"\w+\s*\[\s*[^\]]+\s*\]\s*[<>]=?\s*[\d.]+",
                    r"\w+\s*\[\s*[^\]]+\s*\]\s*==\s*\w+\s*\[\s*[^\]]+\s*\]",
                    r"\w+\s*\[\s*[^\]]+\s*\]\s*[<>]=?\s*\w+\s*\[\s*[^\]]+\s*\]",
                ):
                    for m in re.finditer(pat, line):
                        e = m.group(0)
                        if "=" in pat and "==" not in e and re.search(r"=\s*[\d.]", e):
                            e = re.sub(r"(?<!=)=(?!=)", "==", e)
                        results.append(e)
            return list(dict.fromkeys(results))

        exprs = _parse_exprs(raw)

        def _expr_uses_valid_vars(expr: str) -> bool:
            """True if every variable on the LHS is in variable_names (RHS can be number or param)."""
            for op in ("==", ">=", "<="):
                if op in expr:
                    lhs = expr.split(op)[0]
                    break
            else:
                return False
            for m in re.finditer(r"(\w+)\s*\[", lhs):
                if m.group(1) not in variable_names:
                    return False
            return True

        valid = [e for e in exprs if _expr_uses_valid_vars(e)]
        rejected = [e for e in exprs if e not in valid]

        # --- Retry if no valid expressions found ---
        retry_raw = None
        if not valid and raw:
            retry_prompt = (
                f"Your previous response was:\n{raw}\n\n"
                f"Rewrite as one or more constraints we can parse. "
                f"Use: variable_name[indices] == or >= or <= number or data parameter (e.g. D[d,t]). "
                f"Or a sum of variables: e.g. x[8,0,10] + x[8,1,10] + ... >= D[8,10]. "
                f"Available variables: {vars_str}. Output only the constraint line(s), one per line."
            )
            r2 = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": retry_prompt},
                ],
                max_tokens=300,
                temperature=temperature,
            )
            retry_raw = (r2.choices[0].message.content or "").strip()
            exprs_retry = _parse_exprs(retry_raw)
            valid_retry = [e for e in exprs_retry if e.split("[")[0].strip() in variable_names]
            rejected_retry = [e for e in exprs_retry if e.split("[")[0].strip() not in variable_names]
            if valid_retry:
                exprs = exprs_retry
                valid = valid_retry
                rejected = rejected_retry

        # --- Capture constraint parsing debug ---
        parsing_debug: dict = {
            "raw_response": raw,
            "regex_matches_attempt1": _parse_exprs(raw),
            "valid_expressions": valid,
            "rejected_expressions": rejected,
            "rejected_reason": (
                f"variable name not in {variable_names}" if rejected else None
            ),
        }
        if retry_raw is not None:
            parsing_debug["retry_raw_response"] = retry_raw
            parsing_debug["regex_matches_retry"] = _parse_exprs(retry_raw)
        # Check if LLM produced something the regex couldn't parse at all
        if not valid and raw:
            parsing_debug["parse_failure"] = (
                "Regex found no valid constraints. We accept: single var or sum of vars, "
                "with ==/>=/<= and numeric or data-parameter RHS (e.g. D[8,10])."
            )
        llm_debug["constraint_generation"]["parsing"] = parsing_debug
        state["llm_messages_debug"] = llm_debug

        state["constraint_expressions"] = valid
        return state
    return constraint_generation_node


def make_counterfactual_run_node(build_model_fn, inputs: dict, env_kwargs: dict, outputs_dir: str | Path):
    import agentic_explain.tools.counterfactual_run as _cf_module

    outputs_dir = Path(outputs_dir)

    def counterfactual_run_node(state: "AgentState") -> "AgentState":
        exprs = state.get("constraint_expressions", [])
        if not exprs:
            state["counterfactual_status"] = "error"
            state["counterfactual_result"] = {"error": "No constraint expressions"}
            return state
        ilp_path = outputs_dir / "counterfactual.ilp"
        # Use module-level reference so autoreload picks up changes
        result = _cf_module.run_counterfactual(
            build_model_fn=build_model_fn,
            inputs=inputs,
            env_kwargs=env_kwargs,
            constraint_expressions=exprs,
            ilp_path=ilp_path,
        )
        # Enrich applied_constraints with baseline values
        baseline_vars = state.get("baseline_result", {}).get("decision_variables", {})
        for ac in result.get("applied_constraints", []):
            vname = ac.get("gurobi_var_name", "")
            ac["baseline_value"] = baseline_vars.get(vname)

        state["counterfactual_status"] = result["status"]
        state["counterfactual_result"] = result
        if result.get("ilp_path"):
            state["ilp_path"] = result["ilp_path"]

        # --- Persist counterfactual result for debugging ---
        try:
            persist_dir = outputs_dir / "counterfactual_runs"
            persist_dir.mkdir(parents=True, exist_ok=True)
            # Save a readable summary (without the huge decision_variables dict)
            summary = {k: v for k, v in result.items() if k != "decision_variables"}
            summary["user_query"] = state.get("user_query", "")
            summary["constraint_expressions"] = exprs
            with open(persist_dir / "last_run.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            # Also save the full decision variables separately (for detailed analysis)
            if result.get("decision_variables"):
                with open(persist_dir / "last_run_variables.json", "w", encoding="utf-8") as f:
                    json.dump(result["decision_variables"], f)
        except Exception:
            pass  # persistence is best-effort; don't break the workflow

        return state
    return counterfactual_run_node


def make_compare_node(inputs: dict, variable_names: list[str], outputs_dir: str | Path = "outputs"):
    import agentic_explain.tools.compare_results as _cmp_module

    outputs_dir = Path(outputs_dir)

    def compare_node(state: "AgentState") -> "AgentState":
        baseline = state.get("baseline_result", {})
        cf = state.get("counterfactual_result", {})
        if not baseline or not cf.get("decision_variables"):
            state["comparison_summary"] = "Cannot compare: missing baseline or counterfactual solution."
            return state

        persist_path = outputs_dir / "counterfactual_runs" / "last_comparison.json"

        comparison = _cmp_module.run_comparison(
            baseline_variables=baseline.get("decision_variables", {}),
            counterfactual_variables=cf.get("decision_variables", {}),
            inputs=inputs,
            variable_names=variable_names,
            counterfactual_breakdown=cf.get("objective_breakdown"),
            persist_path=persist_path,
        )

        state["comparison_summary"] = comparison.get("summary_text", "")
        return state
    return compare_node


def make_ilp_analysis_node(rag_strategy, openai_client, temperature: float = 0):
    import agentic_explain.rag.ilp_parser as _ilp_module

    def ilp_analysis_node(state: "AgentState") -> "AgentState":
        ilp_path = state.get("ilp_path")
        if not ilp_path or not Path(ilp_path).exists():
            state["conflict_explanation"] = "No IIS file available."
            return state
        parsed = _ilp_module.parse_ilp_file(ilp_path)
        constraint_names = parsed.get("constraint_names", [])
        if not constraint_names:
            state["conflict_explanation"] = "IIS parsed but no constraint names extracted."
            return state
        rag_query = "constraint " + " ".join(constraint_names[:5])
        chunks = rag_strategy.retrieve(rag_query, top_k=5)
        context = "\n".join([c.text for c in chunks])

        # --- Capture retrieval debug info ---
        rag_debug = state.get("rag_retrieval_debug", {})
        rag_debug["strategy"] = getattr(rag_strategy, "name", "unknown")
        rag_debug["ilp_analysis"] = {
            "query": rag_query,
            "top_k": 5,
            "iis_constraint_names": constraint_names,
            "chunks": [
                {
                    "text": c.text,
                    "score": c.score,
                    "metadata": {k: str(v) for k, v in (c.metadata or {}).items()},
                }
                for c in chunks
            ],
        }
        state["rag_retrieval_debug"] = rag_debug

        sys_msg = (
            "You explain in 2–4 sentences which existing model constraints conflict with the user's requested constraint, "
            "based on the IIS constraint names and the formulation context. Use business terms rather than raw variable names where possible."
        )
        user_msg = f"IIS constraints: {constraint_names}\n\nContext:\n{context}"
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=250,
            temperature=temperature,
        )

        # --- Capture LLM messages debug ---
        llm_debug = state.get("llm_messages_debug", {})
        llm_debug["ilp_analysis"] = {
            "system": sys_msg,
            "user": user_msg,
            "raw_response": (r.choices[0].message.content or "").strip(),
        }
        state["llm_messages_debug"] = llm_debug

        state["conflict_explanation"] = (r.choices[0].message.content or "").strip()
        return state
    return ilp_analysis_node


def make_summarize_node(openai_client, temperature: float = 0):
    def summarize_node(state: "AgentState") -> "AgentState":
        user_query = state.get("user_query", "")
        status = state.get("counterfactual_status", "")
        if status == "feasible":
            comp = state.get("comparison_summary", "")
            sys_msg = (
                "You are an optimization analyst. The user asked a what-if question about an optimization model. "
                "A counterfactual run was performed (forcing their requested change) and compared to the baseline. "
                "You are given a detailed objective breakdown and variable changes.\n\n"
                "Your response should:\n"
                "1. Start with the bottom line: did the user's change improve or worsen the total objective? By how much?\n"
                "2. Explain which objective terms changed the most and why (referencing the specific variable changes).\n"
                "3. Highlight trade-offs: if one term improved but another worsened, explain the mechanism.\n"
                "4. Use business language where possible, referencing the variable descriptions from the comparison data.\n"
                "5. Keep it to 3-5 sentences."
            )
            r = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"User asked: {user_query}\n\nDetailed comparison:\n{comp}"},
                ],
                max_tokens=500,
                temperature=temperature,
            )
            state["final_summary"] = (r.choices[0].message.content or comp).strip()
        elif status == "infeasible":
            conflict = state.get("conflict_explanation", "")
            sys_msg = (
                "You are an optimization analyst. The user asked a what-if question but the requested change "
                "made the model infeasible — no solution exists that satisfies all constraints.\n\n"
                "Explain in 2-3 sentences:\n"
                "1. That the change is infeasible.\n"
                "2. Which constraints conflict and why (use business language).\n"
                "3. What the user could do instead (relax a constraint, try a different week, etc.)."
            )
            r = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"User asked: {user_query}\n\nConflict analysis:\n{conflict}"},
                ],
                max_tokens=300,
                temperature=temperature,
            )
            state["final_summary"] = (r.choices[0].message.content or conflict).strip()
        else:
            err = state.get("counterfactual_result", {}).get("error", "Unknown error")
            state["final_summary"] = f"The counterfactual run failed: {err}"
        return state
    return summarize_node
