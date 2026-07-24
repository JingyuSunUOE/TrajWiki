from __future__ import annotations

import json
import time
from pathlib import Path

from trajpatch.experiments.progress import ExperimentProgress


def test_progress_emits_periodic_snapshot_without_completed_work(
    tmp_path: Path,
) -> None:
    progress_path = tmp_path / "progress.json"
    progress = ExperimentProgress(
        progress_path,
        enabled=True,
        interval_seconds=1,
    )
    try:
        progress.start_stage("slow_provider_calls", total=2)
        first = json.loads(progress_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 3.0
        second = first
        while (
            second["updated_at_unix"] <= first["updated_at_unix"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
            second = json.loads(progress_path.read_text(encoding="utf-8"))
        assert second["updated_at_unix"] > first["updated_at_unix"]
        assert second["stage"] == "slow_provider_calls"
        assert second["completed"] == 0
        assert second["status"] == "running"
    finally:
        progress.close()
