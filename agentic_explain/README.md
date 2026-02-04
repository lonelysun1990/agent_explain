# Agentic Explainability Workflow

Explain optimization run results via natural language: ask "why wasn't X?" or "force X" and get a counterfactual run plus a summary (trade-offs or infeasibility conflict).

## Setup

1. **Secrets**  
   Copy `config/secrets.env.example` to `config/secrets.env` (or project root `secrets.env`) and set:
   - `OPENAI_API_KEY` – for the LLM
   - `GUROBI_LICENSE_FILE` – path to your Gurobi license (e.g. `config/WLS-dev-key.lic`)

2. **Install**  
   From project root:
   ```bash
   pip install -r requirements-agentic.txt
   ```

## Steps

**Option A: Jupyter notebook (recommended)**  
Open and run `agentic_explain/notebooks/AgenticExplainability_Usage.ipynb`. It sets up paths, runs baseline, builds the RAG index, and runs the workflow with example queries.

**Option B: Command line** (run from project root)

1. **Baseline run** (one-time; produces baseline result and model files):
   ```bash
   python -m agentic_explain.scripts.run_baseline
   ```
   Writes `outputs/baseline_result.json`, `outputs/model.lp`, `outputs/model.mps`.

2. **Build RAG index** (one-time; uses .py, .lp, .mps, and data):
   ```bash
   python -m agentic_explain.scripts.build_rag_index
   ```
   Writes `outputs/rag_index/`.

3. **Run workflow** for a query:
   ```bash
   python -m agentic_explain.scripts.run_workflow "Why was Josh not staffed on Ipp IO Pilot in week 6?"
   ```
   Uses baseline and RAG index from `outputs/` and prints the final summary.

## Layout

- `staffing_model.py` – Gurobi formulation and data loading (replace with your own .py if needed).
- `notebooks/AgenticExplainability_Usage.ipynb` – Usage examples (setup, baseline, RAG build, workflow runs).
- `parsers/py_formulation_parser.py` – extracts RAG chunks from the formulation .py.
- `rag/` – .lp, .mps, .ilp parsers and unified index build/load.
- `tools/counterfactual_run.py` – build model, add user constraints, optimize (or write .ilp on infeasibility).
- `workflow/` – LangGraph: query → entity resolution → constraint generation → counterfactual run → compare or ILP analysis → summarize.

## Data

Uses `data/` at project root (ds_list.json, project_list.json, etc.). Legacy notebooks and reference files live in `legacy/` and are not used by this workflow.
