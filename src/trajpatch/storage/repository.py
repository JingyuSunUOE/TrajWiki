"""Repository helpers for persisting and querying TrajWiki state."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from trajpatch.cache.models import MemoryCacheBundle
from trajpatch.ids import trajectory_id
from trajpatch.storage.models import (
    AggregateMetricRecord,
    AnswerRecord,
    ClaimOpRecord,
    ClaimRecord,
    EmbeddingRecord,
    EpisodicMemorySnapshot,
    EvaluationRecord,
    RawMessageRecord,
    RetrievalEvent,
    RunMetaRecord,
    TrajectoryRecord,
    WikiPageRecord,
)
from trajpatch.types import NormalizedMessage

DATETIME_FIELDS = {"created_at", "closed_at"}


def _record_to_dict(record) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column in record.__table__.columns:  # type: ignore[attr-defined]
        value = getattr(record, column.name)
        payload[column.name] = value.isoformat() if isinstance(value, datetime) else value
    return payload


def _restore_datetime_fields(payload: dict[str, Any]) -> dict[str, Any]:
    restored = dict(payload)
    for field_name in DATETIME_FIELDS:
        if isinstance(restored.get(field_name), str):
            restored[field_name] = datetime.fromisoformat(restored[field_name])
    return restored


class TrajWikiStore:
    """Thin repository layer around the SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_raw_message(
        self, sample_id: str, dataset_name: str, message: NormalizedMessage
    ) -> RawMessageRecord:
        record = RawMessageRecord(
            id=f"{sample_id}-m{message.turn_index:04d}",
            sample_id=sample_id,
            dataset_name=dataset_name,
            turn_index=message.turn_index,
            role=message.role,
            speaker_name=message.speaker_name,
            content=message.content,
            source_ref=message.source_ref,
            occurred_at=message.occurred_at,
            metadata_json=message.metadata,
        )
        self.session.merge(record)
        return record

    def list_raw_message_ids(self, sample_id: str) -> set[str]:
        rows = self.session.execute(
            select(RawMessageRecord.id).where(RawMessageRecord.sample_id == sample_id)
        ).all()
        return {row[0] for row in rows}

    def list_sample_ids(self) -> list[str]:
        rows = self.session.execute(select(RawMessageRecord.sample_id).distinct()).all()
        return [row[0] for row in rows]

    def fetch_raw_messages(self, message_ids: Iterable[str]) -> list[RawMessageRecord]:
        ids = list(dict.fromkeys(message_ids))
        if not ids:
            return []
        rows = self.session.execute(select(RawMessageRecord).where(RawMessageRecord.id.in_(ids))).scalars()
        indexed = {row.id: row for row in rows}
        return [indexed[item] for item in ids if item in indexed]

    def list_raw_messages_for_sample(self, sample_id: str) -> list[RawMessageRecord]:
        rows = self.session.execute(
            select(RawMessageRecord)
            .where(RawMessageRecord.sample_id == sample_id)
            .order_by(RawMessageRecord.turn_index.asc(), RawMessageRecord.id.asc())
        ).scalars()
        return list(rows)

    def next_trajectory_ordinal(self, sample_id: str) -> int:
        count = self.session.scalar(
            select(func.count()).select_from(TrajectoryRecord).where(TrajectoryRecord.sample_id == sample_id)
        )
        return int(count or 0) + 1

    def create_trajectory(
        self,
        *,
        sample_id: str,
        dataset_name: str,
        label: str,
        strict_matching: bool,
        max_length: int | None,
        metadata: dict,
    ) -> TrajectoryRecord:
        record = TrajectoryRecord(
            id=trajectory_id(sample_id, self.next_trajectory_ordinal(sample_id)),
            sample_id=sample_id,
            dataset_name=dataset_name,
            label=label,
            strict_matching=strict_matching,
            is_open=True,
            snapshot_count=0,
            max_length=max_length,
            metadata_json=metadata,
        )
        self.session.add(record)
        return record

    def list_trajectories(self, sample_id: str, *, open_only: bool = False) -> list[TrajectoryRecord]:
        clauses = [TrajectoryRecord.sample_id == sample_id]
        if open_only:
            clauses.append(TrajectoryRecord.is_open.is_(True))
        rows = self.session.execute(select(TrajectoryRecord).where(*clauses)).scalars().all()
        return list(rows)

    def get_trajectory(self, trajectory_id: str) -> TrajectoryRecord | None:
        return self.session.get(TrajectoryRecord, trajectory_id)

    def latest_snapshot(self, trajectory_id: str) -> EpisodicMemorySnapshot | None:
        return self.session.execute(
            select(EpisodicMemorySnapshot)
            .where(EpisodicMemorySnapshot.trajectory_id == trajectory_id)
            .order_by(EpisodicMemorySnapshot.version.desc())
            .limit(1)
        ).scalar_one_or_none()

    def list_snapshots(self, trajectory_id: str) -> list[EpisodicMemorySnapshot]:
        rows = self.session.execute(
            select(EpisodicMemorySnapshot)
            .where(EpisodicMemorySnapshot.trajectory_id == trajectory_id)
            .order_by(EpisodicMemorySnapshot.version.asc())
        ).scalars()
        return list(rows)

    def list_snapshots_for_trajectories(self, trajectory_ids: Iterable[str]) -> list[EpisodicMemorySnapshot]:
        ids = list(dict.fromkeys(trajectory_ids))
        if not ids:
            return []
        rows = self.session.execute(
            select(EpisodicMemorySnapshot)
            .where(EpisodicMemorySnapshot.trajectory_id.in_(ids))
            .order_by(EpisodicMemorySnapshot.trajectory_id.asc(), EpisodicMemorySnapshot.version.asc())
        ).scalars()
        return list(rows)

    def save_episodic_snapshot(self, snapshot: EpisodicMemorySnapshot) -> EpisodicMemorySnapshot:
        trajectory = self.get_trajectory(snapshot.trajectory_id)
        self.session.add(snapshot)
        if trajectory is not None:
            trajectory.snapshot_count += 1
            trajectory.latest_snapshot_id = snapshot.id
        return snapshot

    def replace_claims_for_snapshot(self, claims: list[ClaimRecord]) -> None:
        for claim in claims:
            self.session.add(claim)

    def add_claim_ops(self, ops: list[ClaimOpRecord]) -> None:
        for op in ops:
            self.session.add(op)

    def list_claims_for_snapshot(self, snapshot_id: str) -> list[ClaimRecord]:
        rows = self.session.execute(select(ClaimRecord).where(ClaimRecord.snapshot_id == snapshot_id)).scalars()
        return list(rows)

    def list_claim_ops_for_snapshot(self, snapshot_id: str) -> list[ClaimOpRecord]:
        rows = self.session.execute(select(ClaimOpRecord).where(ClaimOpRecord.snapshot_id == snapshot_id)).scalars()
        return list(rows)

    def list_claims_for_snapshots(self, snapshot_ids: Iterable[str]) -> dict[str, list[ClaimRecord]]:
        ids = list(dict.fromkeys(snapshot_ids))
        grouped: dict[str, list[ClaimRecord]] = defaultdict(list)
        if not ids:
            return grouped
        rows = self.session.execute(select(ClaimRecord).where(ClaimRecord.snapshot_id.in_(ids))).scalars()
        for row in rows:
            grouped[row.snapshot_id].append(row)
        return grouped

    def list_claim_ops_for_snapshots(self, snapshot_ids: Iterable[str]) -> dict[str, list[ClaimOpRecord]]:
        ids = list(dict.fromkeys(snapshot_ids))
        grouped: dict[str, list[ClaimOpRecord]] = defaultdict(list)
        if not ids:
            return grouped
        rows = self.session.execute(select(ClaimOpRecord).where(ClaimOpRecord.snapshot_id.in_(ids))).scalars()
        for row in rows:
            grouped[row.snapshot_id].append(row)
        return grouped

    def latest_claims(self, trajectory_id: str) -> list[ClaimRecord]:
        trajectory = self.get_trajectory(trajectory_id)
        if trajectory is None or not trajectory.latest_snapshot_id:
            return []
        rows = self.session.execute(
            select(ClaimRecord).where(ClaimRecord.snapshot_id == trajectory.latest_snapshot_id)
        ).scalars()
        return list(rows)

    def next_claim_ordinal(self, trajectory_id: str) -> int:
        rows = self.session.execute(
            select(ClaimRecord.claim_id).where(ClaimRecord.trajectory_id == trajectory_id)
        ).scalars()
        highest = 0
        for claim_id in rows:
            match = re.search(r"-c(\d+)$", str(claim_id or ""))
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def next_op_ordinal(self, trajectory_id: str) -> int:
        count = self.session.scalar(
            select(func.count()).select_from(ClaimOpRecord).where(ClaimOpRecord.trajectory_id == trajectory_id)
        )
        return int(count or 0) + 1

    def save_embedding(
        self,
        *,
        embedding_id: str,
        owner_type: str,
        owner_id: str,
        model_name: str,
        vector: list[float],
        semantic_text: str,
        metadata: dict,
    ) -> EmbeddingRecord:
        norm = sum(value * value for value in vector) ** 0.5
        record = EmbeddingRecord(
            id=embedding_id,
            owner_type=owner_type,
            owner_id=owner_id,
            model_name=model_name,
            vector_json=vector,
            semantic_text=semantic_text,
            norm=norm,
            metadata_json=metadata,
        )
        self.session.merge(record)
        return record

    def fetch_embedding(self, owner_id: str, owner_type: str | None = None) -> EmbeddingRecord | None:
        clauses = [EmbeddingRecord.owner_id == owner_id]
        if owner_type is not None:
            clauses.append(EmbeddingRecord.owner_type == owner_type)
        return self.session.execute(select(EmbeddingRecord).where(*clauses)).scalar_one_or_none()

    def fetch_embeddings_by_owner_ids(
        self, owner_ids: Iterable[str], owner_type: str | None = None
    ) -> dict[str, EmbeddingRecord]:
        ids = list(dict.fromkeys(owner_ids))
        if not ids:
            return {}
        clauses = [EmbeddingRecord.owner_id.in_(ids)]
        if owner_type is not None:
            clauses.append(EmbeddingRecord.owner_type == owner_type)
        rows = self.session.execute(select(EmbeddingRecord).where(*clauses)).scalars().all()
        return {row.owner_id: row for row in rows}

    def snapshot_embedding(self, snapshot_id: str) -> EmbeddingRecord | None:
        return self.fetch_embedding(snapshot_id, "snapshot")

    def save_wiki_page(self, page: WikiPageRecord) -> WikiPageRecord:
        self.session.merge(page)
        return page

    def replace_wiki_pages_for_sample(self, sample_id: str, pages: list[WikiPageRecord]) -> None:
        self.session.execute(delete(WikiPageRecord).where(WikiPageRecord.sample_id == sample_id))
        for page in pages:
            self.session.add(page)

    def list_wiki_pages(self, sample_id: str) -> list[WikiPageRecord]:
        rows = self.session.execute(
            select(WikiPageRecord)
            .where(WikiPageRecord.sample_id == sample_id)
            .order_by(WikiPageRecord.page_type.asc(), WikiPageRecord.title.asc(), WikiPageRecord.id.asc())
        ).scalars()
        return list(rows)

    def get_wiki_page(self, page_id: str) -> WikiPageRecord | None:
        return self.session.get(WikiPageRecord, page_id)

    def record_retrieval_event(self, event: RetrievalEvent) -> RetrievalEvent:
        self.session.add(event)
        return event

    def record_answer(self, answer: AnswerRecord) -> AnswerRecord:
        self.session.merge(answer)
        return answer

    def record_evaluations(self, rows: list[EvaluationRecord]) -> list[EvaluationRecord]:
        for row in rows:
            self.session.merge(row)
        return rows

    def record_run_meta(self, record: RunMetaRecord) -> RunMetaRecord:
        self.session.merge(record)
        return record

    def replace_aggregate_metrics(self, run_id: str, rows: list[AggregateMetricRecord]) -> list[AggregateMetricRecord]:
        self.session.execute(delete(AggregateMetricRecord).where(AggregateMetricRecord.run_id == run_id))
        for row in rows:
            self.session.add(row)
        return rows

    def fetch_run_meta(self) -> RunMetaRecord | None:
        return self.session.execute(select(RunMetaRecord).limit(1)).scalar_one_or_none()

    def list_aggregate_metrics(self, run_id: str | None = None) -> list[AggregateMetricRecord]:
        stmt = select(AggregateMetricRecord)
        if run_id is not None:
            stmt = stmt.where(AggregateMetricRecord.run_id == run_id)
        return list(self.session.execute(stmt).scalars().all())

    def export_sample_memory_bundle(self, sample_id: str) -> MemoryCacheBundle:
        trajectories = self.list_trajectories(sample_id)
        trajectory_ids = [trajectory.id for trajectory in trajectories]
        episodic_snapshots = (
            self.session.execute(
                select(EpisodicMemorySnapshot).where(EpisodicMemorySnapshot.trajectory_id.in_(trajectory_ids))
            ).scalars().all()
            if trajectory_ids
            else []
        )
        snapshot_ids = [snapshot.id for snapshot in episodic_snapshots]
        claims = (
            self.session.execute(select(ClaimRecord).where(ClaimRecord.snapshot_id.in_(snapshot_ids))).scalars().all()
            if snapshot_ids
            else []
        )
        claim_ops = (
            self.session.execute(select(ClaimOpRecord).where(ClaimOpRecord.snapshot_id.in_(snapshot_ids))).scalars().all()
            if snapshot_ids
            else []
        )
        wiki_pages = self.list_wiki_pages(sample_id)
        embedding_owner_ids = [*trajectory_ids, *snapshot_ids, *[page.id for page in wiki_pages]]
        embeddings = (
            self.session.execute(
                select(EmbeddingRecord).where(EmbeddingRecord.owner_id.in_(embedding_owner_ids))
            ).scalars().all()
            if embedding_owner_ids
            else []
        )
        dataset_name = self.session.scalar(
            select(RawMessageRecord.dataset_name).where(RawMessageRecord.sample_id == sample_id).limit(1)
        ) or (trajectories[0].dataset_name if trajectories else "")
        return MemoryCacheBundle(
            sample_meta={
                "sample_id": sample_id,
                "dataset_name": dataset_name,
                "created_at": datetime.utcnow().isoformat(),
            },
            trajectories=[_record_to_dict(row) for row in trajectories],
            episodic_snapshots=[_record_to_dict(row) for row in episodic_snapshots],
            wiki_pages=[_record_to_dict(row) for row in wiki_pages],
            claims=[_record_to_dict(row) for row in claims],
            claim_ops=[_record_to_dict(row) for row in claim_ops],
            embeddings=[_record_to_dict(row) for row in embeddings],
            memory_stats={
                "trajectory_count": len(trajectories),
                "episodic_snapshot_count": len(episodic_snapshots),
                "wiki_page_count": len(wiki_pages),
                "claim_count": len(claims),
                "claim_op_count": len(claim_ops),
                "embedding_count": len(embeddings),
            },
        )

    def import_sample_memory_bundle(self, bundle: MemoryCacheBundle | dict[str, Any]) -> None:
        parsed = bundle if isinstance(bundle, MemoryCacheBundle) else MemoryCacheBundle.parse_obj(bundle)
        self._merge_rows(TrajectoryRecord, parsed.trajectories)
        self._merge_rows(
            EpisodicMemorySnapshot,
            sorted(parsed.episodic_snapshots, key=lambda row: (str(row["trajectory_id"]), int(row["version"]))),
        )
        self._merge_rows(WikiPageRecord, parsed.wiki_pages)
        self._merge_rows(ClaimRecord, parsed.claims)
        self._merge_rows(ClaimOpRecord, parsed.claim_ops)
        self._merge_rows(EmbeddingRecord, parsed.embeddings)

    def _merge_rows(self, model, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.session.merge(model(**_restore_datetime_fields(dict(row))))

    def inspect_trajectory(self, trajectory_id: str) -> dict:
        trajectory = self.get_trajectory(trajectory_id)
        if trajectory is None:
            return {}
        snapshots = self.list_snapshots(trajectory_id)
        payload = {"trajectory": trajectory.__dict__.copy(), "snapshots": []}
        payload["trajectory"].pop("_sa_instance_state", None)
        for snapshot in snapshots:
            snapshot_dict = snapshot.__dict__.copy()
            snapshot_dict.pop("_sa_instance_state", None)
            snapshot_dict["claims"] = [claim.__dict__.copy() for claim in self.list_claims_for_snapshot(snapshot.id)]
            snapshot_dict["ops"] = [op.__dict__.copy() for op in self.list_claim_ops_for_snapshot(snapshot.id)]
            for item in snapshot_dict["claims"] + snapshot_dict["ops"]:
                item.pop("_sa_instance_state", None)
            payload["snapshots"].append(snapshot_dict)
        return payload

    def metric_summary(self) -> dict[str, float]:
        rows = self.session.execute(select(EvaluationRecord.metric_name, EvaluationRecord.metric_value)).all()
        grouped: dict[str, list[float]] = defaultdict(list)
        for name, value in rows:
            if value is not None:
                grouped[name].append(float(value))
        return {name: (sum(values) / len(values) if values else 0.0) for name, values in grouped.items()}
