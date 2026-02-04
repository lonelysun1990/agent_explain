# Config and secrets

1. Copy `secrets.env.example` to `secrets.env` in this directory (or project root).
2. Set:
   - `OPENAI_API_KEY` – your OpenAI API key for the agentic workflow.
   - `GUROBI_LICENSE_FILE` – path to your Gurobi license file. The license file lives in this folder (e.g. `config/WLS-dev-key.lic` from project root, or `WLS-dev-key.lic` if `secrets.env` is in `config/`).
3. Do not commit `secrets.env`.

The workflow loads secrets via `config.load_secrets.load_secrets()` and `get_gurobi_env_kwargs()`.
