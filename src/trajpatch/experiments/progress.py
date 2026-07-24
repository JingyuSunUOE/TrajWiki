"""Thread-safe progress snapshots for long-running rebuttal experiments."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


class ExperimentProgress:
    """Aggregate progress in the coordinator thread and persist it atomically."""

    def __init__(
        self,
        path: Path,
        *,
        enabled: bool = True,
        interval_seconds: int = 30,
    ) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.interval_seconds = max(1, int(interval_seconds))
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._stage_started_at = self._started_at
        self._last_report_at = 0.0
        self._stage = "initializing"
        self._total = 0
        self._completed = 0
        self._succeeded = 0
        self._failed = 0
        self._reused = 0
        self._status = "running"
        self._last_error: str | None = None
        self._stages: dict[str, dict[str, Any]] = {}
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                self._stages = dict(existing.get("stages") or {})
            except (OSError, ValueError):
                self._stages = {}
        self._write(force=True)
        if self.enabled:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="trajwiki-experiment-progress",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def start_stage(self, stage: str, *, total: int, reused: int = 0) -> None:
        with self._lock:
            self._stage = str(stage)
            self._total = max(0, int(total))
            self._completed = 0
            self._succeeded = 0
            self._failed = 0
            self._reused = max(0, int(reused))
            self._stage_started_at = time.time()
            self._status = "running"
            self._last_error = None
            self._write_locked(force=True)

    def advance(self, *, succeeded: bool = True, error: str | None = None) -> None:
        with self._lock:
            self._completed += 1
            if succeeded:
                self._succeeded += 1
            else:
                self._failed += 1
                self._last_error = " ".join(str(error or "unknown error").split())[:500]
            self._write_locked(force=self._completed >= self._total)

    def finish_stage(self, *, status: str = "complete") -> None:
        with self._lock:
            self._status = str(status)
            self._write_locked(force=True)

    def fail(self, error: BaseException | str) -> None:
        with self._lock:
            self._status = "error"
            self._last_error = " ".join(str(error).split())[:500]
            self._write_locked(force=True)
        self._stop_event.set()

    def close(self) -> None:
        """Stop periodic reporting without blocking process shutdown."""

        self._stop_event.set()
        heartbeat = self._heartbeat_thread
        if (
            heartbeat is not None
            and heartbeat.is_alive()
            and heartbeat is not threading.current_thread()
        ):
            heartbeat.join(timeout=1.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._payload(time.time())

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            with self._lock:
                if self._status == "running":
                    self._write_locked(force=True)

    def _write(self, *, force: bool) -> None:
        with self._lock:
            self._write_locked(force=force)

    def _write_locked(self, *, force: bool) -> None:
        now = time.time()
        if (
            not force
            and now - self._last_report_at < self.interval_seconds
        ):
            return
        payload = self._payload(now)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(self.path)
        self._last_report_at = now
        if self.enabled:
            rate = float(payload["rate_per_second"])
            eta = payload["eta_seconds"]
            eta_text = "unknown" if eta is None else f"{float(eta):.0f}s"
            print(
                "[trajwiki-progress] "
                f"stage={self._stage} status={self._status} "
                f"completed={self._completed}/{self._total} "
                f"success={self._succeeded} failed={self._failed} "
                f"reused={self._reused} rate={rate:.2f}/s eta={eta_text}",
                file=sys.stderr,
                flush=True,
            )

    def _payload(self, now: float) -> dict[str, Any]:
        elapsed = max(0.0, now - self._stage_started_at)
        rate = self._completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self._total - self._completed)
        eta = remaining / rate if rate > 0 and remaining else (0.0 if not remaining else None)
        stage_payload = {
            "status": self._status,
            "total": self._total,
            "completed": self._completed,
            "succeeded": self._succeeded,
            "failed": self._failed,
            "reused": self._reused,
            "rate_per_second": rate,
            "eta_seconds": eta,
            "elapsed_seconds": elapsed,
            "last_error": self._last_error,
            "updated_at_unix": now,
        }
        self._stages[self._stage] = stage_payload
        return {
            "schema_version": "answer_ablation_progress_v1",
            "status": self._status,
            "stage": self._stage,
            "total": self._total,
            "completed": self._completed,
            "succeeded": self._succeeded,
            "failed": self._failed,
            "reused": self._reused,
            "rate_per_second": rate,
            "eta_seconds": eta,
            "stage_elapsed_seconds": elapsed,
            "run_elapsed_seconds": max(0.0, now - self._started_at),
            "last_error": self._last_error,
            "updated_at_unix": now,
            "stages": self._stages,
        }
