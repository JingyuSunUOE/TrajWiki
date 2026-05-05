"""Cache manager for sample-level memory bundles."""

from __future__ import annotations

import os
import random
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from trajpatch.cache.fingerprints import build_memory_fingerprint, build_sample_history_fingerprint
from trajpatch.cache.models import CacheManifest, CacheManifestEntry, MemoryCacheBundle
from trajpatch.ids import snapshot_id, trajectory_id, wiki_page_id
from trajpatch.utils.json_utils import dumps_json


class MemoryCacheManager:
    def __init__(self, config, adapter) -> None:
        self.config = config
        self.adapter = adapter
        self.build_fingerprint, self.build_payload = build_memory_fingerprint(config, adapter)
        self.lock_stats: dict[str, float] = {
            "cache_lock_stale_removed": 0.0,
            "cache_lock_acquire_failed": 0.0,
            "cache_write_skipped_due_to_lock": 0.0,
        }

    @property
    def enabled(self) -> bool:
        return bool(self.config.memory_cache_enabled)

    def global_cache_dir(self) -> Path:
        return self.config.memory_cache_dir / "memory" / self.build_fingerprint

    def samples_dir(self) -> Path:
        return self.global_cache_dir() / "samples"

    def manifest_path(self) -> Path:
        return self.global_cache_dir() / "manifest.json"

    def sample_fingerprint(self, sample) -> tuple[str, object]:
        return build_sample_history_fingerprint(sample, self.adapter)

    def sample_cache_path(self, sample_fingerprint: str) -> Path:
        return self.samples_dir() / f"{sample_fingerprint}.json"

    def load_manifest(self) -> CacheManifest:
        manifest_path = self.manifest_path()
        if manifest_path.exists():
            return CacheManifest.parse_raw(manifest_path.read_text(encoding="utf-8"))
        return CacheManifest(
            schema_version=str(self.build_payload["schema_version"]),
            build_fingerprint=self.build_fingerprint,
            dataset_name=self.config.dataset,
            adapter_version=str(self.build_payload["adapter_version"]),
            provider_kind=self.config.provider_kind,
            backbone_model=self.config.backbone_model,
            embedding_model=self.config.embedding_model,
            created_at=datetime.utcnow().isoformat(),
            prompt_hashes=dict(self.build_payload["prompt_hashes"]),
            source_hashes=dict(self.build_payload["source_hashes"]),
            stats={"cache_hits": 0.0, "cache_writes": 0.0},
        )

    def save_manifest(self, manifest: CacheManifest) -> None:
        self._atomic_write_json(self.manifest_path(), manifest.dict(), indent=True)

    def load_sample_cache(self, sample) -> tuple[MemoryCacheBundle | None, str]:
        sample_fingerprint, _ = self.sample_fingerprint(sample)
        cache_path = self.sample_cache_path(sample_fingerprint)
        if not self.enabled or not cache_path.exists():
            return None, sample_fingerprint
        bundle = MemoryCacheBundle.parse_raw(cache_path.read_text(encoding="utf-8"))
        if not self._bundle_matches_expected(bundle, sample_fingerprint):
            return None, sample_fingerprint
        manifest = self.load_manifest()
        entry = manifest.sample_index.get(sample_fingerprint)
        now = datetime.utcnow().isoformat()
        if entry is not None:
            entry.hit_count += 1
            entry.last_accessed_at = now
            manifest.stats["cache_hits"] = float(manifest.stats.get("cache_hits", 0.0) + 1.0)
            with self._try_file_lock(self.manifest_path().with_suffix(".json.lock"), stats=self.lock_stats) as acquired:
                if acquired:
                    self.save_manifest(manifest)
        return bundle, sample_fingerprint

    def save_sample_cache(self, sample, bundle: MemoryCacheBundle, sample_fingerprint: str) -> tuple[Path, bool]:
        cache_path = self.sample_cache_path(sample_fingerprint)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._try_file_lock(cache_path.with_suffix(cache_path.suffix + ".lock"), stats=self.lock_stats) as acquired:
            if not acquired:
                self.lock_stats["cache_write_skipped_due_to_lock"] += 1.0
                return cache_path, False
            self._atomic_write_json(cache_path, bundle.dict(), indent=True)
        with self._try_file_lock(self.manifest_path().with_suffix(".json.lock"), stats=self.lock_stats) as acquired:
            if acquired:
                manifest = self.load_manifest()
                now = datetime.utcnow().isoformat()
                existing = manifest.sample_index.get(sample_fingerprint)
                sample_ids = list(existing.sample_ids) if existing is not None else []
                if sample.sample_id not in sample_ids:
                    sample_ids.append(sample.sample_id)
                manifest.sample_index[sample_fingerprint] = CacheManifestEntry(
                    sample_id=sample.sample_id,
                    sample_ids=sample_ids or [sample.sample_id],
                    sample_fingerprint=sample_fingerprint,
                    history_fingerprint=str(bundle.sample_meta.get("history_fingerprint", sample_fingerprint)),
                    bundle_path=str(cache_path),
                    created_at=existing.created_at if existing is not None else now,
                    last_accessed_at=existing.last_accessed_at if existing is not None else None,
                    hit_count=existing.hit_count if existing is not None else 0,
                )
                manifest.stats["cache_writes"] = float(manifest.stats.get("cache_writes", 0.0) + 1.0)
                self.save_manifest(manifest)
        return cache_path, True

    @staticmethod
    def _atomic_write_json(path: Path, data: Any, *, indent: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temp_path.write_text(dumps_json(data, indent=indent), encoding="utf-8")
        os.replace(temp_path, path)

    @staticmethod
    @contextmanager
    def _try_file_lock(
        lock_path: Path,
        *,
        attempts: int = 20,
        min_sleep: float = 0.01,
        max_sleep: float = 0.05,
        stale_after_seconds: float = 600.0,
        stats: dict[str, float] | None = None,
    ):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd: int | None = None
        acquired = False
        attempt_count = max(1, int(attempts))
        try:
            for attempt_index in range(attempt_count):
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    payload = {
                        "pid": os.getpid(),
                        "thread_id": threading.get_ident(),
                        "created_at": datetime.utcnow().isoformat(),
                    }
                    os.write(fd, dumps_json(payload, indent=False).encode("utf-8"))
                    acquired = True
                    break
                except FileExistsError:
                    if MemoryCacheManager._remove_stale_lock(
                        lock_path,
                        stale_after_seconds=stale_after_seconds,
                    ):
                        if stats is not None:
                            stats["cache_lock_stale_removed"] = float(
                                stats.get("cache_lock_stale_removed", 0.0) + 1.0
                            )
                        continue
                    if attempt_index == attempt_count - 1:
                        break
                    time.sleep(random.uniform(min_sleep, max_sleep))
            if not acquired and stats is not None:
                stats["cache_lock_acquire_failed"] = float(
                    stats.get("cache_lock_acquire_failed", 0.0) + 1.0
                )
            yield acquired
        finally:
            if fd is not None:
                os.close(fd)
            if acquired:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    @staticmethod
    def _remove_stale_lock(lock_path: Path, *, stale_after_seconds: float) -> bool:
        if stale_after_seconds <= 0:
            return False
        try:
            raw = lock_path.read_text(encoding="utf-8")
            payload = {}
            if raw.strip().startswith("{"):
                import json

                payload = json.loads(raw)
            created_at = payload.get("created_at") if isinstance(payload, dict) else None
            if created_at:
                created = datetime.fromisoformat(str(created_at))
                age_seconds = (datetime.utcnow() - created).total_seconds()
            else:
                age_seconds = time.time() - lock_path.stat().st_mtime
            if age_seconds < stale_after_seconds:
                return False
            lock_path.unlink()
            return True
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            return False

    def hydrate_sample_cache(self, store, bundle: MemoryCacheBundle, sample) -> None:
        store.import_sample_memory_bundle(self._remap_bundle_for_sample(bundle, sample))

    def _remap_bundle_for_sample(self, bundle: MemoryCacheBundle, sample) -> MemoryCacheBundle:
        original_sample_id = str(bundle.sample_meta.get("sample_id", ""))
        original_dataset_name = str(bundle.sample_meta.get("dataset_name", ""))
        if original_sample_id == sample.sample_id and original_dataset_name == sample.dataset_name:
            return bundle

        remapped = bundle.copy(deep=True)
        trajectory_map: dict[str, str] = {}
        snapshot_map: dict[str, str] = {}
        claim_map: dict[str, str] = {}
        page_map: dict[str, str] = {}
        embedding_id_map: dict[str, str] = {}

        for row in remapped.trajectories:
            old_id = str(row["id"])
            ordinal = int(old_id.rsplit("-", 1)[-1])
            new_id = trajectory_id(sample.sample_id, ordinal)
            row["_old_latest_snapshot_id"] = row.get("latest_snapshot_id")
            row["_old_metadata_json"] = dict(row.get("metadata_json") or {})
            row["id"] = new_id
            row["sample_id"] = sample.sample_id
            row["dataset_name"] = sample.dataset_name
            trajectory_map[old_id] = new_id

        for row in remapped.episodic_snapshots:
            old_id = str(row["id"])
            old_trajectory_id = str(row["trajectory_id"])
            new_trajectory_id = trajectory_map[old_trajectory_id]
            new_id = snapshot_id(new_trajectory_id, int(row["version"]))
            row["id"] = new_id
            row["trajectory_id"] = new_trajectory_id
            row["links_json"] = self._remap_message_ids(row.get("links_json", []), original_sample_id, sample.sample_id)
            if "embedding_ref" in row and row.get("embedding_ref"):
                old_embedding_ref = str(row["embedding_ref"])
                row["embedding_ref"] = f"{new_id}-emb"
                embedding_id_map[old_embedding_ref] = str(row["embedding_ref"])
            snapshot_map[old_id] = new_id

        for index, row in enumerate(remapped.wiki_pages, start=1):
            old_id = str(row["id"])
            ordinal = self._trailing_ordinal(old_id, fallback=index)
            new_id = wiki_page_id(sample.sample_id, str(row.get("page_type") or "page"), ordinal)
            page_map[old_id] = new_id
            if row.get("embedding_id"):
                embedding_id_map[str(row["embedding_id"])] = f"{new_id}-emb"

        for row in remapped.wiki_pages:
            old_id = str(row["id"])
            new_id = page_map[old_id]
            row["id"] = new_id
            row["sample_id"] = sample.sample_id
            row["dataset_name"] = sample.dataset_name
            row["trajectory_ids_json"] = [trajectory_map.get(str(value), str(value)) for value in list(row.get("trajectory_ids_json", []))]
            row["linked_page_ids_json"] = [page_map.get(str(value), str(value)) for value in list(row.get("linked_page_ids_json", []))]
            row["embedding_id"] = f"{new_id}-emb" if row.get("embedding_id") else None

        for row in remapped.claims:
            old_trajectory_id = str(row["trajectory_id"])
            new_trajectory_id = trajectory_map[old_trajectory_id]
            old_claim_id = str(row["claim_id"])
            new_claim_id = self._replace_prefix(old_claim_id, old_trajectory_id, new_trajectory_id)
            claim_map[old_claim_id] = new_claim_id
            row["trajectory_id"] = new_trajectory_id
            row["snapshot_id"] = snapshot_map[str(row["snapshot_id"])]
            row["claim_id"] = new_claim_id
            row["id"] = self._replace_prefix(str(row["id"]), old_trajectory_id, new_trajectory_id)
            row["source_message_ids_json"] = self._remap_message_ids(
                row.get("source_message_ids_json", []), original_sample_id, sample.sample_id
            )

        for row in remapped.claims:
            if row.get("parent_claim_id"):
                row["parent_claim_id"] = claim_map.get(str(row["parent_claim_id"]), row["parent_claim_id"])
            if row.get("revised_from_claim_id"):
                row["revised_from_claim_id"] = claim_map.get(
                    str(row["revised_from_claim_id"]), row["revised_from_claim_id"]
                )

        for row in remapped.claim_ops:
            old_trajectory_id = str(row["trajectory_id"])
            new_trajectory_id = trajectory_map[old_trajectory_id]
            row["trajectory_id"] = new_trajectory_id
            row["snapshot_id"] = snapshot_map[str(row["snapshot_id"])]
            row["id"] = self._replace_prefix(str(row["id"]), old_trajectory_id, new_trajectory_id)
            row["target_claim_id"] = claim_map.get(
                str(row["target_claim_id"]),
                self._replace_prefix(str(row["target_claim_id"]), old_trajectory_id, new_trajectory_id),
            )
            if row.get("new_claim_id"):
                row["new_claim_id"] = claim_map.get(
                    str(row["new_claim_id"]),
                    self._replace_prefix(str(row["new_claim_id"]), old_trajectory_id, new_trajectory_id),
                )
            row["source_message_ids_json"] = self._remap_message_ids(
                row.get("source_message_ids_json", []), original_sample_id, sample.sample_id
            )

        for row in remapped.embeddings:
            owner_type = str(row["owner_type"])
            old_owner_id = str(row["owner_id"])
            old_embedding_id = str(row["id"])
            if owner_type in {"trajectory_summary", "trajectory"}:
                new_owner_id = trajectory_map[old_owner_id]
                new_id = self._replace_prefix(old_embedding_id, old_owner_id, new_owner_id)
            elif owner_type == "wiki_page":
                new_owner_id = page_map[old_owner_id]
                new_id = f"{new_owner_id}-emb"
            elif owner_type == "snapshot":
                new_owner_id = snapshot_map[old_owner_id]
                new_id = self._replace_prefix(old_embedding_id, old_owner_id, new_owner_id)
            else:
                raise ValueError(
                    "Unsupported cached embedding owner_type during remap: "
                    f"embedding_id={old_embedding_id} owner_type={owner_type} owner_id={old_owner_id}"
                )
            row["owner_id"] = new_owner_id
            row["id"] = new_id
            embedding_id_map[old_embedding_id] = new_id

        for row in remapped.trajectories:
            old_latest_snapshot_id = row.pop("_old_latest_snapshot_id", None)
            row["latest_snapshot_id"] = snapshot_map.get(str(old_latest_snapshot_id), None) if old_latest_snapshot_id else None
            old_metadata = dict(row.pop("_old_metadata_json", row.get("metadata_json") or {}))
            row["metadata_json"] = self._remap_json_refs(
                old_metadata,
                trajectory_map=trajectory_map,
                snapshot_map=snapshot_map,
                claim_map=claim_map,
                page_map=page_map,
                embedding_id_map=embedding_id_map,
                old_sample_id=original_sample_id,
                new_sample_id=sample.sample_id,
            )
            if row["latest_snapshot_id"]:
                row["metadata_json"]["latest_snapshot_id"] = row["latest_snapshot_id"]

        for rows in (
            remapped.episodic_snapshots,
            remapped.wiki_pages,
            remapped.claims,
            remapped.claim_ops,
            remapped.embeddings,
        ):
            for row in rows:
                for text_key in (
                    "title",
                    "slug",
                    "markdown_text",
                    "semantic_text",
                    "summary_content",
                    "context",
                    "raw_text",
                    "text",
                    "rationale",
                ):
                    if isinstance(row.get(text_key), str):
                        row[text_key] = self._remap_text_refs(
                            str(row[text_key]),
                            trajectory_map=trajectory_map,
                            snapshot_map=snapshot_map,
                            claim_map=claim_map,
                            page_map=page_map,
                            embedding_id_map=embedding_id_map,
                            old_sample_id=original_sample_id,
                            new_sample_id=sample.sample_id,
                        )
                if isinstance(row.get("metadata_json"), dict):
                    row["metadata_json"] = self._remap_json_refs(
                        row["metadata_json"],
                        trajectory_map=trajectory_map,
                        snapshot_map=snapshot_map,
                        claim_map=claim_map,
                        page_map=page_map,
                        embedding_id_map=embedding_id_map,
                        old_sample_id=original_sample_id,
                        new_sample_id=sample.sample_id,
                    )

        remapped.sample_meta["sample_id"] = sample.sample_id
        remapped.sample_meta["dataset_name"] = sample.dataset_name
        return remapped

    def _bundle_matches_expected(self, bundle: MemoryCacheBundle, sample_fingerprint: str) -> bool:
        history_fingerprint = str(bundle.sample_meta.get("history_fingerprint", ""))
        build_fingerprint = str(bundle.sample_meta.get("build_fingerprint", ""))
        if not history_fingerprint or history_fingerprint != sample_fingerprint:
            return False
        if not build_fingerprint or build_fingerprint != self.build_fingerprint:
            return False
        return True

    @staticmethod
    def _replace_prefix(value: str, old_prefix: str, new_prefix: str) -> str:
        return value.replace(old_prefix, new_prefix, 1) if value.startswith(old_prefix) else value

    @staticmethod
    def _trailing_ordinal(value: str, *, fallback: int) -> int:
        try:
            return int(value.rsplit("-", 1)[-1])
        except (TypeError, ValueError):
            return fallback

    @classmethod
    def _remap_json_refs(
        cls,
        value: Any,
        *,
        trajectory_map: dict[str, str],
        snapshot_map: dict[str, str],
        claim_map: dict[str, str],
        page_map: dict[str, str],
        embedding_id_map: dict[str, str],
        old_sample_id: str,
        new_sample_id: str,
    ) -> Any:
        if isinstance(value, dict):
            return {
                key: cls._remap_json_refs(
                    nested,
                    trajectory_map=trajectory_map,
                    snapshot_map=snapshot_map,
                    claim_map=claim_map,
                    page_map=page_map,
                    embedding_id_map=embedding_id_map,
                    old_sample_id=old_sample_id,
                    new_sample_id=new_sample_id,
                )
                for key, nested in value.items()
            }
        if isinstance(value, list):
            return [
                cls._remap_json_refs(
                    nested,
                    trajectory_map=trajectory_map,
                    snapshot_map=snapshot_map,
                    claim_map=claim_map,
                    page_map=page_map,
                    embedding_id_map=embedding_id_map,
                    old_sample_id=old_sample_id,
                    new_sample_id=new_sample_id,
                )
                for nested in value
            ]
        if isinstance(value, str):
            exact_maps = (trajectory_map, snapshot_map, claim_map, page_map, embedding_id_map)
            for mapping in exact_maps:
                if value in mapping:
                    return mapping[value]
            return cls._remap_text_refs(
                value,
                trajectory_map=trajectory_map,
                snapshot_map=snapshot_map,
                claim_map=claim_map,
                page_map=page_map,
                embedding_id_map=embedding_id_map,
                old_sample_id=old_sample_id,
                new_sample_id=new_sample_id,
            )
        return value

    @staticmethod
    def _remap_text_refs(
        value: str,
        *,
        trajectory_map: dict[str, str],
        snapshot_map: dict[str, str],
        claim_map: dict[str, str],
        page_map: dict[str, str],
        embedding_id_map: dict[str, str],
        old_sample_id: str,
        new_sample_id: str,
    ) -> str:
        remapped = value
        for mapping in (trajectory_map, snapshot_map, claim_map, page_map, embedding_id_map):
            for old_id, new_id in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
                if old_id and old_id in remapped:
                    remapped = remapped.replace(old_id, new_id)
        if old_sample_id and old_sample_id != new_sample_id:
            remapped = remapped.replace(f"{old_sample_id}-m", f"{new_sample_id}-m")
            remapped = remapped.replace(old_sample_id, new_sample_id)
        return remapped

    @staticmethod
    def _remap_message_id(value: str, old_sample_id: str, new_sample_id: str) -> str:
        if not old_sample_id or old_sample_id == new_sample_id:
            return value
        prefix = f"{old_sample_id}-m"
        replacement = f"{new_sample_id}-m"
        return value.replace(prefix, replacement, 1) if value.startswith(prefix) else value

    @staticmethod
    def _remap_message_ids(message_ids: list[Any], old_sample_id: str, new_sample_id: str) -> list[Any]:
        if not old_sample_id or old_sample_id == new_sample_id:
            return list(message_ids)
        return [
            MemoryCacheManager._remap_message_id(str(item), old_sample_id, new_sample_id)
            if isinstance(item, str)
            else item
            for item in message_ids
        ]
