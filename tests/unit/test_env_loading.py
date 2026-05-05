from __future__ import annotations

import os
from pathlib import Path

from trajpatch.utils.env import load_runtime_env


def test_load_runtime_env_loads_cwd_dotenv_with_override(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("TRAJPATCH_DOTENV_TEST_VAR=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAJPATCH_DOTENV_TEST_VAR", "from-shell")

    loaded_paths = load_runtime_env(override=True)

    assert str(Path(dotenv_path).resolve()) in {str(path.resolve()) for path in loaded_paths}
    assert Path.cwd().joinpath(".env").resolve() in {path.resolve() for path in loaded_paths}
    assert os.environ["TRAJPATCH_DOTENV_TEST_VAR"] == "from-dotenv"
