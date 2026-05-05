"""Artifact export helpers."""

from __future__ import annotations

from pathlib import Path

from trajpatch.storage.repository import TrajPatchStore
from trajpatch.utils.json_utils import append_jsonl, dumps_json, write_json


class ArtifactExporter:
    def __init__(self, output_dir: Path, store: TrajPatchStore, *, run_dir: Path | None = None) -> None:
        self.output_dir = output_dir
        self.run_dir = run_dir or output_dir
        self.store = store

    @staticmethod
    def _trajectory_manifest_row(payload: dict, *, json_path: Path, summary_path: Path | None) -> dict:
        trajectory = dict(payload.get("trajectory") or {})
        metadata = dict(trajectory.get("metadata_json") or {})
        snapshots = list(payload.get("snapshots") or [])
        source_refs: list[str] = []
        for snapshot in snapshots:
            for claim in list((snapshot or {}).get("claims") or []):
                for ref in list(claim.get("source_message_ids_json") or []):
                    text = str(ref).strip()
                    if text and text not in source_refs:
                        source_refs.append(text)
                        if len(source_refs) >= 24:
                            break
                if len(source_refs) >= 24:
                    break
            if len(source_refs) >= 24:
                break
        summary_text = str(metadata.get("retrieval_summary_text") or "").strip()
        return {
            "schema_version": "trajectory_export_manifest_v1",
            "trajectory_id": trajectory.get("id"),
            "sample_id": trajectory.get("sample_id"),
            "dataset_name": trajectory.get("dataset_name"),
            "label": trajectory.get("label"),
            "is_open": trajectory.get("is_open"),
            "snapshot_count": trajectory.get("snapshot_count"),
            "latest_snapshot_id": trajectory.get("latest_snapshot_id"),
            "json_path": json_path.name,
            "summary_path": summary_path.name if summary_path is not None else None,
            "summary_preview": summary_text[:1000],
            "source_anchor_refs": source_refs,
            "metadata": {
                "trajectory_identity_summary_v1": metadata.get("trajectory_identity_summary_v1"),
                "trajectory_recent_update_v1": metadata.get("trajectory_recent_update_v1"),
                "trajectory_historical_item_terms_v1": list(
                    metadata.get("trajectory_historical_item_terms_v1") or []
                )[:50],
                "trajectory_historical_item_terms_v2": list(
                    metadata.get("trajectory_historical_item_terms_v2") or []
                )[:50],
                "source_surface_terms_v1": list(metadata.get("source_surface_terms_v1") or [])[:50],
                "source_surface_raw_terms_v1": list(metadata.get("source_surface_raw_terms_v1") or [])[:50],
                "facet_tags": list(metadata.get("facet_tags") or []),
                "facet_values": list(metadata.get("facet_values") or [])[:50],
                "entity_mentions": list(metadata.get("entity_mentions") or [])[:50],
            },
        }

    def export_sample_trajectories(self, sample_id: str) -> dict:
        rows = []
        memory_dir = self.run_dir / "memories" / sample_id
        for trajectory in self.store.list_trajectories(sample_id):
            payload = self.store.inspect_trajectory(trajectory.id)
            if payload:
                path = memory_dir / f"{trajectory.id}.json"
                write_json(path, payload)
                summary_text = str((trajectory.metadata_json or {}).get("retrieval_summary_text") or "").strip()
                summary_path: Path | None = None
                if summary_text:
                    summary_path = memory_dir / f"{trajectory.id}.summary.md"
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary_path.write_text(summary_text + "\n", encoding="utf-8")
                rows.append(self._trajectory_manifest_row(payload, json_path=path, summary_path=summary_path))
        if rows:
            manifest_path = memory_dir / "trajectories.jsonl"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(dumps_json(row, indent=False))
                    handle.write("\n")
            return {
                "trajectory_count": len(rows),
                "manifest_path": str(manifest_path),
                "manifest_bytes": manifest_path.stat().st_size if manifest_path.exists() else 0,
            }
        return {"trajectory_count": 0, "manifest_path": str(memory_dir / "trajectories.jsonl"), "manifest_bytes": 0}

    def export_sample_wiki(self, sample_id: str) -> None:
        pages = self.store.list_wiki_pages(sample_id)
        if not pages:
            return
        wiki_dir = self.run_dir / "memories" / sample_id / "wiki"
        rows = []
        for page in pages:
            payload = {
                "id": page.id,
                "sample_id": page.sample_id,
                "dataset_name": page.dataset_name,
                "page_type": page.page_type,
                "title": page.title,
                "slug": page.slug,
                "keywords": list(page.keywords_json or []),
                "trajectory_ids": list(page.trajectory_ids_json or []),
                "linked_page_ids": list(page.linked_page_ids_json or []),
                "entity_names": list(page.entity_names_json or []),
                "metadata": dict(page.metadata_json or {}),
            }
            write_json(wiki_dir / f"{page.id}.json", payload)
            (wiki_dir / f"{page.id}.md").write_text((page.markdown_text or "").strip() + "\n", encoding="utf-8")
            rows.append(payload)
        index_page = next((page for page in pages if page.page_type == "index"), None)
        if index_page is not None:
            (wiki_dir / "index.md").write_text((index_page.markdown_text or "").strip() + "\n", encoding="utf-8")
        append_jsonl(wiki_dir / "pages.jsonl", rows)

    def export_report(self, payload: dict) -> None:
        write_json(self.run_dir / "reports" / "run_summary.json", payload)

    def export_benchmark_details(self, payload: dict) -> None:
        write_json(self.run_dir / "details.json", payload)

    def export_benchmark_summary(self, payload: dict) -> None:
        write_json(self.run_dir / "summary.json", payload)
