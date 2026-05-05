"""Utilities for loading runtime environment variables from the project .env file."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_runtime_env(*, override: bool = True) -> list[Path]:
    """Load environment variables from the project root .env and current working directory."""

    loaded_paths: list[Path] = []
    repo_root = Path(__file__).resolve().parents[3]
    repo_env = repo_root / ".env"
    cwd_env = Path.cwd() / ".env"
    candidates: list[Path] = [repo_env]
    if cwd_env != repo_env:
        candidates.append(cwd_env)

    for candidate in candidates:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=override)
            loaded_paths.append(candidate)
    return loaded_paths
