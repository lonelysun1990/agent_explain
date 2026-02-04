"""
Load secrets from secrets.env into os.environ.
Call this at the start of scripts that need OpenAI or Gurobi.
"""
import os
from pathlib import Path


def load_secrets(env_file: str = "secrets.env", search_parents: bool = True) -> None:
    """
    Load KEY=VALUE lines from env_file into os.environ.
    If env_file is a relative path, search from cwd and then parent directories
    until the file is found. Also tries config/secrets.env (e.g. when secrets live in config/).
    """
    path = Path(env_file)
    if not path.is_absolute():
        start = Path.cwd()
        candidates = [start / env_file]
        if search_parents:
            candidates += [d / env_file for d in start.parents]
        # Also try config/secrets.env when env_file is "secrets.env"
        if path.name == "secrets.env":
            candidates += [start / "config" / "secrets.env"]
            if search_parents:
                candidates += [d / "config" / "secrets.env" for d in start.parents]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and value and key not in os.environ:
                    os.environ[key] = value


def get_gurobi_env_kwargs():
    """
    Return env_kwargs for gurobipy.Env.
    If GUROBI_LICENSE_FILE is set, set GRB_LICENSE_FILE to its absolute path.
    Tries: cwd/lic, parents/lic, then cwd/config/lic (so WLS-dev-key.lic finds config/WLS-dev-key.lic).
    Returns empty dict (gurobipy.Env() uses env vars by default).
    """
    load_secrets()
    lic = os.environ.get("GUROBI_LICENSE_FILE")
    if lic:
        p = Path(lic)
        if not p.is_absolute():
            candidates = [Path.cwd() / lic]
            for d in Path.cwd().parents:
                candidates.append(d / lic)
            # If license lives in config/ (e.g. config/WLS-dev-key.lic), try config/<basename>
            base = Path(lic).name
            candidates.append(Path.cwd() / "config" / base)
            for d in Path.cwd().parents:
                candidates.append(d / "config" / base)
            for c in candidates:
                if c.exists():
                    p = c.resolve()
                    break
        os.environ["GRB_LICENSE_FILE"] = str(p)
    return {}
