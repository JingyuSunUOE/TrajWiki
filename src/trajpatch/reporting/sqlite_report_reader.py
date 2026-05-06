"""Read run summaries from per-run or indexed SQLite databases."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from sqlalchemy import inspect, select

from trajpatch.exceptions import ParserValidationError
from trajpatch.storage.database import create_index_schema, create_schema
from trajpatch.storage.models import (
    AggregateMetricRecord,
    IndexedAggregateMetricRecord,
    IndexedRunRecord,
    RunMetaRecord,
)

DEFAULT_SORT_BY = "completed_at_desc"


class SQLiteReportReader:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def report_index(
        self,
        *,
        index_database_path: Path,
        dataset: str | None = None,
        subset: str | None = None,
        backbone_model: str | None = None,
        judge_model: str | None = None,
        embedding_model: str | None = None,
        backbone_provider_kind: str | None = None,
        judge_provider_kind: str | None = None,
        m: int | None = None,
        t_pages: int | None = None,
        k: int | None = None,
        neighbor_radius: int | None = None,
        retrieval_expansion_mode: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
        sort_by: str = DEFAULT_SORT_BY,
    ) -> list[dict[str, Any]]:
        if not index_database_path.exists():
            return []
        session = create_index_schema(index_database_path)()
        try:
            self._ensure_schema_compatibility(session, table_name="run_registry", database_path=index_database_path)
            stmt = select(IndexedRunRecord)
            if dataset is not None:
                stmt = stmt.where(IndexedRunRecord.dataset == dataset)
            if subset is not None:
                stmt = stmt.where(IndexedRunRecord.dataset_scope_key == subset)
            if backbone_model is not None:
                stmt = stmt.where(IndexedRunRecord.backbone_model == backbone_model)
            if judge_model is not None:
                stmt = stmt.where(IndexedRunRecord.judge_model == judge_model)
            if embedding_model is not None:
                stmt = stmt.where(IndexedRunRecord.embedding_model == embedding_model)
            if backbone_provider_kind is not None:
                stmt = stmt.where(IndexedRunRecord.backbone_provider_kind == backbone_provider_kind)
            if judge_provider_kind is not None:
                stmt = stmt.where(IndexedRunRecord.judge_provider_kind == judge_provider_kind)
            if m is not None:
                stmt = stmt.where(IndexedRunRecord.m == m)
            if t_pages is not None:
                stmt = stmt.where(IndexedRunRecord.t_pages == t_pages)
            if k is not None:
                stmt = stmt.where(IndexedRunRecord.k == k)
            if neighbor_radius is not None:
                stmt = stmt.where(IndexedRunRecord.neighbor_radius == neighbor_radius)
            if retrieval_expansion_mode is not None:
                stmt = stmt.where(
                    IndexedRunRecord.retrieval_expansion_mode == retrieval_expansion_mode
                )
            if run_id is not None:
                stmt = stmt.where(IndexedRunRecord.run_id == run_id)
            runs = list(session.execute(stmt).scalars().all())
            if not runs:
                return []
            run_ids = [row.run_id for row in runs]
            metrics_stmt = select(IndexedAggregateMetricRecord).where(
                IndexedAggregateMetricRecord.run_id.in_(run_ids),
                IndexedAggregateMetricRecord.group_level == "overall",
            )
            metrics_by_run = self._group_metrics(session.execute(metrics_stmt).scalars().all())
            rows = [self._row_from_run(row, metrics_by_run.get(row.run_id, {})) for row in runs]
            return self._finalize_rows(rows, sort_by=sort_by, limit=limit)
        finally:
            session.close()

    def report_single_run(self, *, database_path: Path, sort_by: str = DEFAULT_SORT_BY) -> list[dict[str, Any]]:
        if not database_path.exists():
            return []
        session = create_schema(database_path)()
        try:
            self._ensure_schema_compatibility(session, table_name="run_meta", database_path=database_path)
            run_meta = session.execute(select(RunMetaRecord).limit(1)).scalar_one_or_none()
            if run_meta is None:
                return []
            metrics_stmt = select(AggregateMetricRecord).where(
                AggregateMetricRecord.run_id == run_meta.run_id,
                AggregateMetricRecord.group_level == "overall",
            )
            metrics = self._group_metrics(session.execute(metrics_stmt).scalars().all())
            row = self._row_from_run(run_meta, metrics.get(run_meta.run_id, {}))
            return self._finalize_rows([row], sort_by=sort_by, limit=1)
        finally:
            session.close()

    def print_tables(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            self.console.print("No matching runs found.")
            return
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["dataset"])].append(row)
        for dataset_name, dataset_rows in grouped.items():
            if dataset_name == "locomo":
                self.console.print(self._build_locomo_table(dataset_rows))
            else:
                self.console.print(self._build_medmt_table(dataset_rows))

    def _group_metrics(self, rows: list[AggregateMetricRecord | IndexedAggregateMetricRecord]) -> dict[str, dict[str, float]]:
        grouped: dict[str, dict[str, float]] = defaultdict(dict)
        for row in rows:
            grouped[str(row.run_id)][str(row.metric_name)] = float(row.metric_value)
        return grouped

    def _row_from_run(self, row: RunMetaRecord | IndexedRunRecord, metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "run_id": row.run_id,
            "dataset": row.dataset,
            "dataset_scope_key": row.dataset_scope_key or "all",
            "backbone_model": row.backbone_model,
            "backbone_provider_kind": row.backbone_provider_kind,
            "judge_model": row.judge_model,
            "judge_provider_kind": row.judge_provider_kind,
            "embedding_model": row.embedding_model,
            "m": row.m,
            "t_pages": row.t_pages,
            "k": row.k,
            "neighbor_radius": row.neighbor_radius,
            "retrieval_expansion_mode": row.retrieval_expansion_mode,
            "completed_at": row.completed_at,
            "total_runtime_s": float(row.total_runtime_s or 0.0),
            "prompt_tokens": int(row.prompt_tokens or 0),
            "completion_tokens": int(row.completion_tokens or 0),
            "total_tokens": int(row.total_tokens or 0),
            "run_dir": row.run_dir,
            "run_database_path": row.run_database_path,
            "processed_samples": int(row.processed_samples or 0),
            "processed_queries": int(row.processed_queries or 0),
            "excluded_count": int(row.excluded_count or 0),
            "metrics": metrics,
        }

    def _finalize_rows(
        self, rows: list[dict[str, Any]], *, sort_by: str, limit: int | None
    ) -> list[dict[str, Any]]:
        reverse = True
        field_name = sort_by
        if sort_by.endswith("_asc"):
            reverse = False
            field_name = sort_by[:-4]
        elif sort_by.endswith("_desc"):
            reverse = True
            field_name = sort_by[:-5]

        def sort_key(row: dict[str, Any]) -> Any:
            if field_name in row.get("metrics", {}):
                return row["metrics"].get(field_name, 0.0)
            value = row.get(field_name)
            if value is None:
                return 0 if reverse else ""
            return value

        ordered = sorted(rows, key=sort_key, reverse=reverse)
        if limit is not None:
            return ordered[:limit]
        return ordered

    def _build_locomo_table(self, rows: list[dict[str, Any]]) -> Table:
        table = Table(title="TrajWiki LOCOMO Runs")
        for column in [
            "run_id",
            "dataset",
            "scope",
            "backbone",
            "judge",
            "m/tp/k/nr/mode",
            "F1",
            "BLEU-1",
            "judge_acc",
            "total_runtime_s",
            "total_tokens",
            "run_dir",
        ]:
            justify = "right" if column in {"F1", "BLEU-1", "judge_acc", "total_runtime_s", "total_tokens"} else "left"
            table.add_column(column, justify=justify)
        for row in rows:
            metrics = row["metrics"]
            table.add_row(
                str(row["run_id"]),
                str(row["dataset"]),
                str(row["dataset_scope_key"]),
                str(row["backbone_model"]),
                str(row["judge_model"]),
                f"{row['m']}/{row['t_pages']}/{row['k']}/{row['neighbor_radius']}/{row['retrieval_expansion_mode']}",
                self._render_metric(metrics.get("F1")),
                self._render_metric(metrics.get("BLEU-1")),
                self._render_metric(metrics.get("judge_acc")),
                f"{row['total_runtime_s']:.2f}",
                str(row["total_tokens"]),
                str(row["run_dir"]),
            )
        return table

    def _build_medmt_table(self, rows: list[dict[str, Any]]) -> Table:
        table = Table(title="TrajWiki MedMT Runs")
        for column in [
            "run_id",
            "dataset",
            "scope",
            "backbone",
            "judge",
            "m/tp/k/nr/mode",
            "judge_acc",
            "total_runtime_s",
            "total_tokens",
            "run_dir",
        ]:
            justify = "right" if column in {"judge_acc", "total_runtime_s", "total_tokens"} else "left"
            table.add_column(column, justify=justify)
        for row in rows:
            metrics = row["metrics"]
            table.add_row(
                str(row["run_id"]),
                str(row["dataset"]),
                str(row["dataset_scope_key"]),
                str(row["backbone_model"]),
                str(row["judge_model"]),
                f"{row['m']}/{row['t_pages']}/{row['k']}/{row['neighbor_radius']}/{row['retrieval_expansion_mode']}",
                self._render_metric(metrics.get("judge_acc")),
                f"{row['total_runtime_s']:.2f}",
                str(row["total_tokens"]),
                str(row["run_dir"]),
            )
        return table

    @staticmethod
    def _render_metric(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{float(value):.4f}"

    @staticmethod
    def _ensure_schema_compatibility(session, *, table_name: str, database_path: Path) -> None:
        inspector = inspect(session.bind)
        existing_tables = set(inspector.get_table_names())
        if table_name not in existing_tables:
            return
        columns = {column["name"] for column in inspector.get_columns(table_name)}
        missing = sorted({"neighbor_radius", "dataset_scope_key", "retrieval_expansion_mode", "t_pages"} - columns)
        if missing:
            raise ParserValidationError(
                f"SQLite schema at {database_path} is missing required columns {missing} in table '{table_name}'. "
                "Delete the old SQLite file and rerun to rebuild the schema."
            )
