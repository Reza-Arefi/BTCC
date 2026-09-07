"""Load repo `.env` into process env without printing values.

Does not override variables already set in the environment.
Never logs secret values.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Parse KEY=VALUE lines from a .env file into os.environ.

    Returns the path loaded, or None if missing.
    """
    if path is None:
        # Prefer repo-root .env (parent of package)
        pkg = Path(__file__).resolve().parent
        candidates = [
            pkg.parent / ".env",
            pkg / ".env",
            Path.cwd() / ".env",
        ]
    else:
        candidates = [Path(path)]

    env_file = next((p for p in candidates if p.is_file()), None)
    if env_file is None:
        return None

    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
            value = value.replace("\\n", "\n").replace("\\\"", '"').replace("\\\\", "\\")
        if not override and key in os.environ:
            continue
        os.environ[key] = value
    return env_file
