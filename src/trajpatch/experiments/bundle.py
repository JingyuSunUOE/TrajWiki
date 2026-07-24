"""Portable, checksummed rebuttal bundle packaging."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from trajpatch.experiments.progress import ExperimentProgress

_FORBIDDEN_NAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "secrets",
    "secrets.json",
}
_SECRET_MARKERS = [
    b"OPENAI_API_KEY=",
    b"ANTHROPIC_API_KEY=",
    b"GOOGLE_API_KEY=",
    b"GEMINI_API_KEY=",
    b"sk-proj-",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_source_file(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    lowered_parts = {part.casefold() for part in path.parts}
    if lowered_parts & _FORBIDDEN_NAMES:
        return False
    if path.name.endswith((".tmp", ".lock", ".tar.zst", ".pyc")):
        return False
    if "__pycache__" in path.parts or ".git" in path.parts:
        return False
    return True


def _assert_no_secret_marker(path: Path) -> None:
    if path.suffix.lower() in {
        ".sqlite",
        ".db",
        ".npz",
        ".npy",
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gz",
        ".zst",
    }:
        return
    with path.open("rb") as handle:
        tail = b""
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            payload = tail + chunk
            marker = next(
                (value for value in _SECRET_MARKERS if value in payload),
                None,
            )
            if marker is not None:
                raise ValueError(
                    f"Refusing to package a possible secret from {path.name}: "
                    f"marker={marker.decode('ascii', errors='replace')}"
                )
            tail = payload[-64:]


def _add_tree(
    files: dict[str, Path],
    *,
    source: Path,
    archive_prefix: str,
    excluded_parts: set[str] | None = None,
) -> None:
    if not source.exists():
        return
    excluded = excluded_parts or set()
    candidates = [source] if source.is_file() else sorted(source.rglob("*"))
    for path in candidates:
        if not _safe_source_file(path):
            continue
        try:
            relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        except ValueError:
            continue
        if set(relative.parts) & excluded:
            continue
        archive_path = (Path(archive_prefix) / relative).as_posix()
        files.setdefault(archive_path, path)


def _repository_git_state(repository: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        commit = ""
        status = ""
    return {
        "commit": commit or None,
        "dirty": bool(status.strip()),
        "status_sha256": (
            hashlib.sha256(status.encode("utf-8")).hexdigest()
            if status
            else None
        ),
    }


def _generated_environment_files(repository: Path) -> dict[str, bytes]:
    try:
        pip_freeze = subprocess.run(
            ["python", "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pip_freeze = f"pip freeze unavailable: {exc}\n"
    git_state = _repository_git_state(repository)
    return {
        "environment/pip_freeze.txt": pip_freeze.encode("utf-8"),
        "environment/git_state.json": (
            json.dumps(git_state, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }


def package_rebuttal_bundle(
    experiment_path: Path | str,
    *,
    output_path: Path | str | None = None,
    workflow_logs_path: Path | str | None = None,
    repository_path: Path | str | None = None,
) -> dict[str, Any]:
    """Create one relocatable archive without changing original manifests."""

    try:
        import zstandard
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "package-rebuttal-bundle requires the zstandard package."
        ) from exc

    experiment_dir = Path(experiment_path).expanduser().resolve()
    manifest_path = experiment_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    sampling_path = experiment_dir / "sampling_manifest.json"
    if not sampling_path.exists():
        raise FileNotFoundError(sampling_path)
    sampling = _read_json(sampling_path)
    if manifest.get("status") != "complete":
        raise ValueError("Only a complete answer-ablation experiment can be packaged.")
    for report_name in ["integrity_report.json", "validation_report.json"]:
        report_path = experiment_dir / report_name
        if not report_path.exists():
            raise FileNotFoundError(
                f"A validated experiment must contain {report_name}."
            )
        report = _read_json(report_path)
        if int(report.get("error_count") or 0):
            raise ValueError(
                f"Refusing to package an experiment with errors in {report_name}."
            )
    progress = ExperimentProgress(
        experiment_dir / "progress.json",
        enabled=False,
    )
    progress.start_stage("packaging", total=1)
    source_run_dir = Path(str(manifest.get("source_run_dir") or "")).expanduser().resolve()
    source_database = Path(
        str(manifest.get("source_database_path") or "")
    ).expanduser().resolve()
    if not source_run_dir.exists() or not source_database.exists():
        raise FileNotFoundError(
            "The experiment source run or source database is unavailable."
        )
    for source_name in ["details.json", "summary.json"]:
        source_path = source_run_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(
                f"The experiment source run is missing {source_name}."
            )
    for source_path, manifest_key in [
        (source_run_dir / "details.json", "source_details_sha256"),
        (source_database, "source_database_sha256"),
    ]:
        expected_sha256 = str(manifest.get(manifest_key) or "").strip()
        if expected_sha256 and _sha256_file(source_path) != expected_sha256:
            raise ValueError(
                "The immutable source artifact no longer matches "
                f"{manifest_key}: {source_path}."
            )
    sampling_profile = str(sampling.get("sampling_profile") or "")
    if sampling_profile == "rebuttal_200_v1":
        required_experiment_files = [
            "answer_ablation_rows.jsonl",
            "independent_judgment_rows.jsonl",
            "provider_call_rows.jsonl",
            "variant_context_rows.jsonl",
            "token_budget_audit.csv",
            "call_plan.json",
            "parent_experiment_reuse.json",
            "prevalence_weighted_paired_statistics.csv",
            "dialogue_cluster_paired_statistics.csv",
            "matched_budget_compliance.csv",
        ]
        missing_experiment_files = [
            name
            for name in required_experiment_files
            if not (experiment_dir / name).exists()
        ]
        if missing_experiment_files:
            raise FileNotFoundError(
                "The 200-query experiment is missing required artifacts: "
                f"{missing_experiment_files}."
            )
        parent_reuse = _read_json(
            experiment_dir / "parent_experiment_reuse.json"
        )
        expected_parent_counts = {
            "status": "verified",
            "generation_job_count": 540,
            "judgment_job_count": 660,
            "provider_call_row_count": 1200,
            "parent_query_overlap": 60,
        }
        mismatched_parent_counts = {
            key: {
                "expected": expected,
                "observed": parent_reuse.get(key),
            }
            for key, expected in expected_parent_counts.items()
            if parent_reuse.get(key) != expected
        }
        if mismatched_parent_counts:
            raise ValueError(
                "The 200-query parent-reuse proof is incomplete: "
                f"{mismatched_parent_counts}."
            )
        required_offline_files = [
            "offline_ablation_summary.json",
            "cost_benefit_summary.json",
            "auditability_summary.json",
            "direct_retrieval_ablation.json",
            "ranking_robustness_summary.json",
        ]
        missing_offline_files = [
            name
            for name in required_offline_files
            if not (experiment_dir / "offline_analysis" / name).exists()
        ]
        if missing_offline_files:
            raise FileNotFoundError(
                "The 200-query experiment is missing scoped offline analyses: "
                f"{missing_offline_files}."
            )
        baseline_methods = set(manifest.get("baseline_methods") or [])
        attachments = list(
            manifest.get("external_baseline_attachments") or []
        )
        if "mem0_saved" not in baseline_methods or not any(
            "mem0_saved" in set(attachment.get("methods") or [])
            and bool(str(attachment.get("path") or "").strip())
            and Path(str(attachment.get("path") or "")).expanduser().exists()
            for attachment in attachments
        ):
            raise FileNotFoundError(
                "The 200-query experiment is missing its normalized saved "
                "Mem0 attachment."
            )
    repository = (
        Path(repository_path).expanduser().resolve()
        if repository_path is not None
        else Path(__file__).resolve().parents[3]
    )
    repository_git = _repository_git_state(repository)
    expected_commit = str(
        dict(manifest.get("git") or {}).get("commit") or ""
    ).strip()
    if expected_commit:
        if repository_git.get("commit") != expected_commit:
            raise ValueError(
                "Repository commit does not match the immutable experiment image."
            )
        if repository_git.get("dirty") is True:
            raise ValueError(
                "Repository became dirty after the experiment; refusing to "
                "package a mismatched code snapshot."
            )
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else experiment_dir.parent
        / f"verification_bundle_{experiment_dir.name}.tar.zst"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    for name in [
        "details.json",
        "summary.json",
        "status.json",
        "run_manifest.json",
    ]:
        path = source_run_dir / name
        if path.exists():
            files[f"source_run/{name}"] = path
    files[f"source_run/{source_database.name}"] = source_database
    for analysis_name in ["analysis", "analysis_v2"]:
        _add_tree(
            files,
            source=source_run_dir / analysis_name,
            archive_prefix=f"source_run/{analysis_name}",
        )
    _add_tree(
        files,
        source=experiment_dir,
        archive_prefix="experiment",
        excluded_parts={"cache"},
    )

    parent_reuse_path = experiment_dir / "parent_experiment_reuse.json"
    if parent_reuse_path.exists():
        parent_reuse = _read_json(parent_reuse_path)
        parent_value = str(parent_reuse.get("parent_experiment_path") or "")
        if parent_value:
            parent_dir = Path(parent_value).expanduser().resolve()
            for name in [
                "experiment_manifest.json",
                "sampling_manifest.json",
                "integrity_report.json",
                "validation_report.json",
                "call_plan.json",
                "provider_call_rows.jsonl",
            ]:
                path = parent_dir / name
                if path.exists():
                    files[f"parent_experiment/{name}"] = path
            parent_sampling_path = parent_dir / "sampling_manifest.json"
            if parent_sampling_path.exists():
                parent_queries = {
                    str(row.get("query_task_id") or "")
                    for row in list(
                        _read_json(parent_sampling_path).get("selected_queries")
                        or []
                    )
                }
                for stage, variants_key in [
                    ("generation", "reusable_generation_variants"),
                    ("judge", "reusable_judgment_variants"),
                ]:
                    for variant in list(parent_reuse.get(variants_key) or []):
                        for query_task_id in parent_queries:
                            path = (
                                parent_dir
                                / "jobs"
                                / stage
                                / variant
                                / f"{query_task_id}.json"
                            )
                            if path.exists():
                                files[
                                    f"parent_experiment/jobs/{stage}/"
                                    f"{variant}/{path.name}"
                                ] = path

    for index, attachment in enumerate(
        list(manifest.get("external_baseline_attachments") or [])
    ):
        value = str(attachment.get("path") or "")
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if path.exists():
            files[
                f"external_baselines/{index:02d}_{path.name}"
            ] = path

    experiment_suffix = experiment_dir.name
    study_dirs = sorted(
        (source_run_dir / "rebuttal_experiments").glob(
            f"audit_study_*_{experiment_suffix}"
        )
    )
    for study_dir in study_dirs:
        _add_tree(
            files,
            source=study_dir,
            archive_prefix=f"audit_study/{study_dir.name}",
        )
    if sampling_profile == "rebuttal_200_v1" and not any(
        archive_path.startswith("audit_study/")
        and archive_path.endswith("/audit_study_manifest.json")
        for archive_path in files
    ):
        raise FileNotFoundError(
            "The 200-query experiment is missing its 24-case audit study."
        )
    if sampling_profile == "rebuttal_200_v1" and not any(
        int(
            _read_json(study_dir / "audit_study_manifest.json").get(
                "case_count"
            )
            or 0
        )
        == 24
        for study_dir in study_dirs
        if (study_dir / "audit_study_manifest.json").exists()
    ):
        raise ValueError(
            "The 200-query audit study does not contain the required 24 cases."
        )
    if workflow_logs_path is not None:
        _add_tree(
            files,
            source=Path(workflow_logs_path).expanduser().resolve(),
            archive_prefix="workflow_logs",
            excluded_parts={"cache"},
        )

    for name in [
        "pyproject.toml",
        "README.md",
        "Dockerfile",
        "job.yml",
        "job-package.yml",
        ".dockerignore",
        ".gitignore",
        "rebuttal.md",
        "neurips_2026.pdf",
    ]:
        path = repository / name
        if path.exists() and _safe_source_file(path):
            files[f"code_snapshot/{name}"] = path
    for directory in ["src", "tests", "scripts", "plot"]:
        _add_tree(
            files,
            source=repository / directory,
            archive_prefix=f"code_snapshot/{directory}",
        )

    for archive_path, path in files.items():
        if not archive_path.startswith("code_snapshot/"):
            _assert_no_secret_marker(path)
    generated = _generated_environment_files(repository)
    entries = [
        {
            "archive_path": archive_path,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for archive_path, path in sorted(files.items())
    ]
    entries.extend(
        {
            "archive_path": archive_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for archive_path, payload in sorted(generated.items())
    )
    bundle_manifest = {
        "schema_version": "trajwiki_rebuttal_bundle_v1",
        "contains_sensitive_text": True,
        "portable_relative_paths": True,
        "original_manifests_preserved": True,
        "relocation": {
            "source_run": "source_run/",
            "answer_experiment": "experiment/",
            "parent_reuse_evidence": "parent_experiment/",
            "external_baselines": "external_baselines/",
            "audit_study": "audit_study/",
            "workflow_logs": "workflow_logs/",
            "code_snapshot": "code_snapshot/",
            "note": (
                "Resolve original absolute paths by artifact role; original "
                "server manifests are retained verbatim for provenance."
            ),
        },
        "experiment_config_hash": manifest.get("config_hash"),
        "sampling_profile": sampling_profile,
        "sampling_status": sampling.get("sampling_status"),
        "file_count": len(entries),
        "files": entries,
        "excluded": [
            ".env and credentials",
            "model caches",
            "experiment embedding cache",
            "temporary and lock files",
        ],
    }
    manifest_bytes = (
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temp_path = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    with temp_path.open("wb") as raw_handle:
        compressor = zstandard.ZstdCompressor(level=9)
        with compressor.stream_writer(raw_handle, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                info = tarfile.TarInfo("bundle_manifest.json")
                info.size = len(manifest_bytes)
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(manifest_bytes))
                for archive_path, payload in sorted(generated.items()):
                    info = tarfile.TarInfo(archive_path)
                    info.size = len(payload)
                    info.mode = 0o600
                    archive.addfile(info, io.BytesIO(payload))
                for archive_path, path in sorted(files.items()):
                    info = archive.gettarinfo(str(path), arcname=archive_path)
                    info.mode = 0o600
                    with path.open("rb") as source_handle:
                        archive.addfile(info, source_handle)
    temp_path.replace(destination)
    destination.chmod(0o600)
    archive_sha256 = _sha256_file(destination)
    sha_path = destination.with_suffix(destination.suffix + ".sha256")
    sha_path.write_text(
        f"{archive_sha256}  {destination.name}\n",
        encoding="utf-8",
    )
    sha_path.chmod(0o600)
    progress.advance()
    progress.finish_stage()
    return {
        "schema_version": "trajwiki_rebuttal_bundle_report_v1",
        "bundle_path": str(destination),
        "bundle_sha256": archive_sha256,
        "sha256_path": str(sha_path),
        "file_count": len(entries),
        "contains_sensitive_text": True,
    }


def validate_rebuttal_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Stream-validate paths, required files, sizes, and SHA256 values."""

    try:
        import zstandard
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "validate-rebuttal-bundle requires the zstandard package."
        ) from exc

    path = Path(bundle_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    observed: dict[str, dict[str, Any]] = {}
    bundle_manifest: dict[str, Any] | None = None
    captured_json: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    captured_json_names = {
        "experiment/experiment_manifest.json",
        "experiment/integrity_report.json",
        "experiment/validation_report.json",
    }
    with path.open("rb") as raw_handle:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(raw_handle) as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|") as archive:
                for member in archive:
                    member_path = Path(member.name)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        errors.append(f"unsafe archive path: {member.name}")
                        continue
                    if member.name.casefold().endswith("/.env") or (
                        member_path.name.casefold() in _FORBIDDEN_NAMES
                    ):
                        errors.append(f"forbidden sensitive path: {member.name}")
                    if not member.isfile():
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        errors.append(f"unable to read archive member: {member.name}")
                        continue
                    digest = hashlib.sha256()
                    chunks: list[bytes] = []
                    size = 0
                    while True:
                        chunk = extracted.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        size += len(chunk)
                        if (
                            member.name == "bundle_manifest.json"
                            or member.name in captured_json_names
                        ):
                            chunks.append(chunk)
                    observed[member.name] = {
                        "sha256": digest.hexdigest(),
                        "size_bytes": size,
                    }
                    if member.name == "bundle_manifest.json":
                        bundle_manifest = json.loads(b"".join(chunks))
                    elif member.name in captured_json_names:
                        try:
                            captured_json[member.name] = json.loads(
                                b"".join(chunks)
                            )
                        except (TypeError, ValueError):
                            errors.append(
                                f"invalid JSON report in bundle: {member.name}"
                            )
    if bundle_manifest is None:
        errors.append("bundle_manifest.json is missing")
        expected: dict[str, dict[str, Any]] = {}
    else:
        expected = {
            str(row.get("archive_path") or ""): row
            for row in list(bundle_manifest.get("files") or [])
        }
        if bundle_manifest.get("portable_relative_paths") is not True:
            errors.append("bundle does not declare portable relative paths")
    for archive_path, row in expected.items():
        actual = observed.get(archive_path)
        if actual is None:
            errors.append(f"missing checksummed member: {archive_path}")
            continue
        if (
            actual["sha256"] != row.get("sha256")
            or actual["size_bytes"] != row.get("size_bytes")
        ):
            errors.append(f"checksum or size mismatch: {archive_path}")
    unexpected = sorted(
        set(observed) - set(expected) - {"bundle_manifest.json"}
    )
    if unexpected:
        errors.append(f"unmanifested archive members: {unexpected[:5]}")
    for required in [
        "source_run/details.json",
        "source_run/summary.json",
        "experiment/experiment_manifest.json",
        "experiment/sampling_manifest.json",
        "experiment/integrity_report.json",
        "experiment/validation_report.json",
        "experiment/provider_call_rows.jsonl",
        "experiment/variant_context_rows.jsonl",
        "code_snapshot/pyproject.toml",
    ]:
        if required not in observed:
            errors.append(f"required bundle member is missing: {required}")
    if (
        bundle_manifest
        and bundle_manifest.get("sampling_profile") == "rebuttal_200_v1"
    ):
        for required in [
            "experiment/answer_ablation_rows.jsonl",
            "experiment/independent_judgment_rows.jsonl",
            "experiment/token_budget_audit.csv",
            "experiment/call_plan.json",
            "experiment/parent_experiment_reuse.json",
            "experiment/offline_analysis/offline_ablation_summary.json",
            "experiment/offline_analysis/cost_benefit_summary.json",
            "experiment/offline_analysis/auditability_summary.json",
            "experiment/offline_analysis/direct_retrieval_ablation.json",
            "experiment/offline_analysis/ranking_robustness_summary.json",
            "parent_experiment/provider_call_rows.jsonl",
        ]:
            if required not in observed:
                errors.append(
                    f"required 200-query bundle member is missing: {required}"
                )
        if not any(name.startswith("external_baselines/") for name in observed):
            errors.append("the normalized external baseline is missing")
        if not any(
            name.startswith("audit_study/")
            and name.endswith("/audit_study_manifest.json")
            for name in observed
        ):
            errors.append("the 24-case audit study is missing")
    if not any(
        name.startswith("source_run/") and name.endswith(".sqlite")
        for name in observed
    ):
        errors.append("source SQLite database is missing")
    for report_name in [
        "experiment/integrity_report.json",
        "experiment/validation_report.json",
    ]:
        report = captured_json.get(report_name)
        if report is not None and int(report.get("error_count") or 0):
            errors.append(f"bundle contains a failed report: {report_name}")
    experiment_manifest = captured_json.get(
        "experiment/experiment_manifest.json"
    )
    if experiment_manifest is not None:
        expected_details_sha256 = str(
            experiment_manifest.get("source_details_sha256") or ""
        ).strip()
        actual_details = observed.get("source_run/details.json")
        if (
            expected_details_sha256
            and actual_details is not None
            and actual_details["sha256"] != expected_details_sha256
        ):
            errors.append(
                "source_run/details.json does not match the experiment manifest"
            )
        expected_database_sha256 = str(
            experiment_manifest.get("source_database_sha256") or ""
        ).strip()
        database_rows = [
            row
            for name, row in observed.items()
            if name.startswith("source_run/") and name.endswith(".sqlite")
        ]
        if (
            expected_database_sha256
            and database_rows
            and database_rows[0]["sha256"] != expected_database_sha256
        ):
            errors.append(
                "source SQLite does not match the experiment manifest"
            )
    return {
        "schema_version": "trajwiki_rebuttal_bundle_validation_v1",
        "bundle_path": str(path),
        "bundle_sha256": _sha256_file(path),
        "error_count": len(errors),
        "errors": errors,
        "file_count": len(expected),
        "contains_sensitive_text": bool(
            bundle_manifest
            and bundle_manifest.get("contains_sensitive_text")
        ),
    }
