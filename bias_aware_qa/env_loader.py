"""Minimal .env loader (no external dependency).

Loads KEY=VALUE lines from ``bias_aware_qa/.env`` (which holds the copied API
key) into ``os.environ`` so both this package and the COMPASS backend see the
credentials. Never prints secret values.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """Load a .env file into os.environ; returns the (masked) keys loaded."""
    if path is None:
        # bias_aware_qa/.env lives two levels up from this file's package dir
        path = Path(__file__).resolve().parents[1] / ".env"
    path = Path(path)
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = "***" if "KEY" in key or "SECRET" in key else value
    return loaded


def openai_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))
