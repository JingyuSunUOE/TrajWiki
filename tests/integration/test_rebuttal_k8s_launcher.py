from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _launcher_fixture(
    tmp_path: Path,
    *,
    fail_first_formal: bool,
) -> tuple[Path, dict[str, str], Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    workspace = Path(__file__).resolve().parents[2]
    shutil.copy2(workspace / "job.yml", repository / "job.yml")
    shutil.copy2(workspace / "job-package.yml", repository / "job-package.yml")
    (repository / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (repository / "README.md").write_text("test\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text(
        "[project]\nname='trajwiki-test'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    (repository / ".env").write_text(
        "OPENAI_API_KEY=test\nANTHROPIC_API_KEY=test\n",
        encoding="utf-8",
    )
    run_dir = repository / "run"
    run_dir.mkdir()
    (run_dir / "details.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "trajpatch.sqlite").write_bytes(b"SQLite format 3\x00")
    mem0_path = repository / "mem0.jsonl"
    mem0_path.write_text("{}\n", encoding="utf-8")
    parent = repository / "parent"
    parent.mkdir()
    details_sha256 = hashlib.sha256(
        (run_dir / "details.json").read_bytes()
    ).hexdigest()
    database_sha256 = hashlib.sha256(
        (run_dir / "trajpatch.sqlite").read_bytes()
    ).hexdigest()
    parent_config = {
        "variants": [
            "full",
            "direct_trajectory",
            "latest_snapshot",
            "hybrid_raw_rag",
            "wiki_summaries",
            "no_claim_state",
            "no_source_constraint",
            "full_context",
            "naive_dense_rag",
        ],
        "sample_size": 60,
        "sample_seed": 7,
        "max_total_tokens": 32000,
        "max_output_tokens": 512,
        "token_counter_requested": "tiktoken",
        "require_exact_token_counter": True,
        "token_safety_margin": 128,
        "rag_chunk_size": 384,
        "rag_chunk_overlap": 64,
        "rag_top_k": 4,
        "backbone_provider_kind": "remote",
        "backbone_model": "gpt-4o-mini",
        "independent_judge_provider_kind": "remote",
        "independent_judge_model": "claude-sonnet-4-6",
        "generation_temperature": 0.0,
        "generation_seed": 7,
        "full_answer_policy": "rerun_with_shared_neutral_prompt_v1",
        "answer_prompt_version": "answer_ablation_neutral_v1",
        "judge_prompt_version": "independent_answer_judge_v1",
    }
    (parent / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_details_sha256": details_sha256,
                "source_database_sha256": database_sha256,
                "config": parent_config,
                "baseline_methods": ["mem0_saved"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (parent / "sampling_manifest.json").write_text(
        '{"selected_queries":['
        + ",".join(
            f'{{"query_task_id":"query-{index}"}}'
            for index in range(60)
        )
        + "]}\n",
        encoding="utf-8",
    )
    for name in ("integrity_report.json", "validation_report.json"):
        (parent / name).write_text('{"error_count":0}\n', encoding="utf-8")
    (parent / "provider_call_rows.jsonl").write_text(
        "{}\n" * 1200,
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    kubectl_log = tmp_path / "kubectl.log"
    kubectl_state = tmp_path / "kubectl.state"
    log_probe_state = tmp_path / "log-probe.state"
    log_probe_ready = tmp_path / "log-probe.ready"
    log_follow_state = tmp_path / "log-follow.state"
    log_follow_ready = tmp_path / "log-follow.ready"
    _write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "status" ]]; then
    exit 0
fi
if [[ "$1" == "rev-parse" ]]; then
    printf '%040d\n' 1
    exit 0
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$FAKE_DOCKER_LOG"
if [[ "$1" == "image" && "$2" == "inspect" ]]; then
    if [[ "$*" == *"Architecture"* ]]; then
        printf 'amd64/linux\n'
    else
        printf 'example/traj@sha256:abcdef\n'
    fi
fi
""",
    )
    _write_executable(
        fake_bin / "kubectl",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$FAKE_KUBECTL_LOG"
if [[ "$1" == "auth" ]]; then
    printf 'yes\n'
    exit 0
fi
if [[ "$1" == "create" ]]; then
    if [[ "$*" == *"--dry-run=client"* ]]; then
        exit 0
    fi
    rendered=""
    previous=""
    for argument in "$@"; do
        if [[ "$previous" == "-f" ]]; then
            rendered="$argument"
            break
        fi
        previous="$argument"
    done
    stem="$(basename "$rendered" _rendered.yaml)"
    job="job-${stem//_/-}"
    printf '%s\n' "$job" >"$FAKE_KUBECTL_STATE"
    printf '%s' "$job"
    exit 0
fi
if [[ "$1" == "get" && "$2" == "pods" ]]; then
    job=""
    for argument in "$@"; do
        if [[ "$argument" == job-name=* ]]; then
            job="${argument#job-name=}"
        fi
    done
    printf 'pod-%s' "$job"
    exit 0
fi
if [[ "$1" == "get" && "$2" == "jobs" ]]; then
    exit 0
fi
if [[ "$1" == "get" && "$2" == "job" ]]; then
    job="$3"
    if [[ "$*" == *".status.succeeded"* ]]; then
        if (( FAKE_LOG_PROBE_FAILURES > 0 )) && [[ ! -e "$FAKE_LOG_PROBE_READY" ]]; then
            exit 0
        fi
        if (( FAKE_LOG_FOLLOW_FAILURES > 0 )) && [[ ! -e "$FAKE_LOG_FOLLOW_READY" ]]; then
            exit 0
        fi
        if [[ "$FAKE_FAIL_FIRST_FORMAL" == "1" && "$job" == *"formal-initial"* ]]; then
            exit 0
        fi
        printf '1'
        exit 0
    fi
    if [[ "$*" == *".status.failed"* ]]; then
        if [[ "$FAKE_FAIL_FIRST_FORMAL" == "1" && "$job" == *"formal-initial"* ]]; then
            printf '1'
        fi
        exit 0
    fi
    printf 'kind: Job\nmetadata:\n  name: %s\n' "$job"
    exit 0
fi
if [[ "$1" == "get" && "$2" == "pod" ]]; then
    printf 'pod ready\n'
    exit 0
fi
if [[ "$1" == "describe" ]]; then
    printf 'description\n'
    exit 0
fi
if [[ "$1" == "logs" ]]; then
    if [[ "$*" == *"--tail=1"* ]]; then
        probe_count=0
        if [[ -r "$FAKE_LOG_PROBE_STATE" ]]; then
            probe_count="$(cat "$FAKE_LOG_PROBE_STATE")"
        fi
        if (( probe_count < FAKE_LOG_PROBE_FAILURES )); then
            printf '%s\n' "$(( probe_count + 1 ))" >"$FAKE_LOG_PROBE_STATE"
            printf 'ContainerCreating\n' >&2
            exit 1
        fi
        : >"$FAKE_LOG_PROBE_READY"
    else
        follow_count=0
        if [[ -r "$FAKE_LOG_FOLLOW_STATE" ]]; then
            follow_count="$(cat "$FAKE_LOG_FOLLOW_STATE")"
        fi
        if (( follow_count < FAKE_LOG_FOLLOW_FAILURES )); then
            printf '%s\n' "$(( follow_count + 1 ))" >"$FAKE_LOG_FOLLOW_STATE"
            printf 'transient log stream failure\n' >&2
            exit 1
        fi
        : >"$FAKE_LOG_FOLLOW_READY"
    fi
    printf 'application log\n'
    exit 0
fi
if [[ "$1" == "delete" ]]; then
    : >"$FAKE_KUBECTL_STATE"
    printf 'deleted\n'
    exit 0
fi
exit 0
""",
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "TRAJWIKI_PROJECT_DIR": str(repository),
        "TRAJWIKI_RUN_DIR": str(run_dir),
        "TRAJWIKI_MEM0_SOURCE": str(mem0_path),
        "TRAJWIKI_PARENT_60_EXPERIMENT": str(parent),
        "TRAJWIKI_WORKFLOW_ID": "testworkflow",
        "TRAJWIKI_CONFIRM_FORMAL": "RUN",
        "TRAJWIKI_IMAGE_REPO": "example/traj",
        "FAKE_DOCKER_LOG": str(docker_log),
        "FAKE_KUBECTL_LOG": str(kubectl_log),
        "FAKE_KUBECTL_STATE": str(kubectl_state),
        "FAKE_LOG_PROBE_STATE": str(log_probe_state),
        "FAKE_LOG_PROBE_READY": str(log_probe_ready),
        "FAKE_LOG_PROBE_FAILURES": "0",
        "FAKE_LOG_FOLLOW_STATE": str(log_follow_state),
        "FAKE_LOG_FOLLOW_READY": str(log_follow_ready),
        "FAKE_LOG_FOLLOW_FAILURES": "0",
        "FAKE_FAIL_FIRST_FORMAL": "1" if fail_first_formal else "0",
        "TRAJWIKI_POD_START_TIMEOUT_SECONDS": "10",
        "TRAJWIKI_POD_STATUS_INTERVAL_SECONDS": "1",
    }
    return repository, env, docker_log, kubectl_log


def _line_index(lines: list[str], *parts: str) -> int:
    return next(
        index
        for index, line in enumerate(lines)
        if all(part in line for part in parts)
    )


@pytest.mark.parametrize(
    ("mode", "fail_first_formal"),
    [("all", False), ("formal", True)],
)
def test_kubernetes_launcher_archives_deletes_and_resumes_in_order(
    tmp_path: Path,
    mode: str,
    fail_first_formal: bool,
) -> None:
    repository, env, docker_log, kubectl_log = _launcher_fixture(
        tmp_path,
        fail_first_formal=fail_first_formal,
    )
    launcher = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "launch_rebuttal_k8s.sh"
    )
    result = subprocess.run(
        ["bash", str(launcher), mode],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    kubectl_lines = kubectl_log.read_text(encoding="utf-8").splitlines()
    assert not any("delete jobs --all" in line for line in kubectl_lines)
    assert "build " in docker_log.read_text(encoding="utf-8")
    rendered_files = sorted(
        (
            repository
            / "private"
            / "rebuttal_200"
            / "testworkflow"
            / "kubernetes"
        ).glob("*_rendered.yaml")
    )
    assert rendered_files
    rendered_text = "\n".join(
        path.read_text(encoding="utf-8") for path in rendered_files
    )
    assert str(repository / "run") in rendered_text
    assert str(repository / "mem0.jsonl") in rendered_text
    assert str(repository / "parent") in rendered_text
    assert str(repository) in rendered_text
    assert (
        str(
            repository
            / "private"
            / "rebuttal_200"
            / "testworkflow"
        )
        in rendered_text
    )
    assert (
        str(repository / "verification_bundle_testworkflow.tar.zst")
        in rendered_text
    )

    initial_create = _line_index(
        kubectl_lines,
        "create -f",
        "formal_initial_rendered.yaml",
    )
    initial_archive = _line_index(
        kubectl_lines,
        "get job job-formal-initial -n",
        "-o yaml",
    )
    initial_delete = _line_index(
        kubectl_lines,
        "delete job job-formal-initial",
    )
    assert initial_create < initial_archive < initial_delete

    if fail_first_formal:
        resume_create = _line_index(
            kubectl_lines,
            "create -f",
            "formal_resume_rendered.yaml",
        )
        resume_delete = _line_index(
            kubectl_lines,
            "delete job job-formal-resume",
        )
        assert initial_delete < resume_create < resume_delete
    else:
        dry_delete = _line_index(
            kubectl_lines,
            "delete job job-dry-run-initial",
        )
        package_create = _line_index(
            kubectl_lines,
            "create -f",
            "package_initial_rendered.yaml",
        )
        package_delete = _line_index(
            kubectl_lines,
            "delete job job-package-initial",
        )
        assert dry_delete < initial_create < initial_delete < package_create
        assert package_create < package_delete


def test_kubernetes_launcher_waits_for_container_logs(
    tmp_path: Path,
) -> None:
    repository, env, _, _ = _launcher_fixture(
        tmp_path,
        fail_first_formal=False,
    )
    env["FAKE_LOG_PROBE_FAILURES"] = "2"
    launcher = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "launch_rebuttal_k8s.sh"
    )

    result = subprocess.run(
        ["bash", str(launcher), "dry-run"],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("Waiting for container start:") == 2
    assert "Container logs available:" in result.stdout
    assert Path(env["FAKE_LOG_PROBE_STATE"]).read_text(encoding="utf-8").strip() == "2"


def test_kubernetes_launcher_reattaches_interrupted_log_stream(
    tmp_path: Path,
) -> None:
    repository, env, _, kubectl_log = _launcher_fixture(
        tmp_path,
        fail_first_formal=False,
    )
    env["FAKE_LOG_FOLLOW_FAILURES"] = "1"
    launcher = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "launch_rebuttal_k8s.sh"
    )

    result = subprocess.run(
        ["bash", str(launcher), "dry-run"],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Pod log stream ended with status 1 while Job is active;" in result.stdout
    assert "Reattaching to Pod logs:" in result.stdout
    assert any(
        "logs -f" in line and "--since=30s" in line
        for line in kubectl_log.read_text(encoding="utf-8").splitlines()
    )
