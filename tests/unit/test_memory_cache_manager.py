from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path

from trajpatch.cache.manager import MemoryCacheManager
from trajpatch.cache.models import MemoryCacheBundle
from trajpatch.config import RunConfig


class _DummyAdapter:
    adapter_version = "test-v1"

    def history_fingerprint_payload(self, sample):
        return {"sample_id": sample.sample_id}


class _DummySample:
    sample_id = "sample-a"
    dataset_name = "locomo"


def _manager(tmp_path: Path) -> MemoryCacheManager:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")
    config = RunConfig(
        dataset="locomo",
        dataset_path=dataset_path,
        output_dir=tmp_path / "output",
        database_path=tmp_path / "output" / "run.sqlite",
        memory_cache_dir=tmp_path / ".trajpatch_cache",
    )
    return MemoryCacheManager(config, _DummyAdapter())


def _bundle() -> MemoryCacheBundle:
    return MemoryCacheBundle(
        sample_meta={"sample_id": "sample-a", "dataset_name": "locomo"},
        memory_stats={},
    )


def test_try_file_lock_retries_until_lock_is_released(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache.lock"
    lock_path.write_text("held", encoding="utf-8")

    def release_lock() -> None:
        time.sleep(0.01)
        lock_path.unlink()

    thread = threading.Thread(target=release_lock)
    thread.start()
    try:
        with MemoryCacheManager._try_file_lock(  # noqa: SLF001
            lock_path,
            attempts=100,
            min_sleep=0.001,
            max_sleep=0.001,
        ) as acquired:
            assert acquired is True
    finally:
        thread.join(timeout=1)


def test_try_file_lock_returns_false_after_retry_exhaustion(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache.lock"
    lock_path.write_text("held", encoding="utf-8")

    with MemoryCacheManager._try_file_lock(  # noqa: SLF001
        lock_path,
        attempts=2,
        min_sleep=0.0,
        max_sleep=0.0,
    ) as acquired:
        assert acquired is False

    assert lock_path.exists()


def test_try_file_lock_removes_stale_lock_and_records_stat(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache.lock"
    lock_path.write_text("held", encoding="utf-8")
    old_time = time.time() - 3600
    lock_path.touch()
    import os

    os.utime(lock_path, (old_time, old_time))
    stats: dict[str, float] = {}

    with MemoryCacheManager._try_file_lock(  # noqa: SLF001
        lock_path,
        attempts=2,
        min_sleep=0.0,
        max_sleep=0.0,
        stale_after_seconds=1.0,
        stats=stats,
    ) as acquired:
        assert acquired is True

    assert stats["cache_lock_stale_removed"] == 1.0
    assert not lock_path.exists()


def test_try_file_lock_records_acquire_failure_stat(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache.lock"
    lock_path.write_text("held", encoding="utf-8")
    stats: dict[str, float] = {}

    with MemoryCacheManager._try_file_lock(  # noqa: SLF001
        lock_path,
        attempts=1,
        min_sleep=0.0,
        max_sleep=0.0,
        stale_after_seconds=600.0,
        stats=stats,
    ) as acquired:
        assert acquired is False

    assert stats["cache_lock_acquire_failed"] == 1.0


def test_save_sample_cache_reports_false_when_sample_lock_is_not_acquired(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    @contextmanager
    def always_locked(lock_path, **kwargs):
        yield False

    manager._try_file_lock = always_locked  # type: ignore[method-assign]  # noqa: SLF001

    cache_path, written = manager.save_sample_cache(_DummySample(), _bundle(), "fingerprint-a")

    assert cache_path == manager.sample_cache_path("fingerprint-a")
    assert written is False
    assert not cache_path.exists()
