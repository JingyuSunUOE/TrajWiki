#!/usr/bin/env bash

set -euo pipefail
umask 077

MODE="${1:-all}"
case "$MODE" in
    dry-run|formal|package|all|both)
        ;;
    *)
        echo "Usage: $0 {dry-run|formal|package|all}" >&2
        exit 2
        ;;
esac
if [[ "$MODE" == "both" ]]; then
    MODE="all"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${TRAJWIKI_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
NS="${TRAJWIKI_K8S_NAMESPACE:-eidf098ns}"
IMAGE_REPO="${TRAJWIKI_IMAGE_REPO:-jingyusun/traj}"
RUN_DIR="${TRAJWIKI_RUN_DIR:-$PROJECT_DIR/output/remote_gpt4omini/locomo/multi_hop/20260507_001755_559270_rc900bf_locomo_multi-hop_gpt-4o-mini_gpt-4o-mini_qwen3-embedding-8b_m15_tp15_k15_nr1_remupdate-linked-plus-neighbors}"
MEM0_SOURCE="${TRAJWIKI_MEM0_SOURCE:-$PROJECT_DIR/locomo_eval/outputs/runs/mem0__gpt-4o-mini__category_1_multi_hop__20260406-010812/predictions.jsonl}"
PARENT_DIR_FILE="${TRAJWIKI_PARENT_60_FILE:-$PROJECT_DIR/private/rebuttal/answer_ablation_experiment_dir.txt}"
PARENT_60_EXPERIMENT="${TRAJWIKI_PARENT_60_EXPERIMENT:-}"
GPU_JOB_TEMPLATE="$PROJECT_DIR/job.yml"
PACKAGE_JOB_TEMPLATE="$PROJECT_DIR/job-package.yml"
WORKFLOW_ID="${TRAJWIKI_WORKFLOW_ID:-rw$(date -u +%Y%m%d%H%M%S)}"
WORKFLOW_DIR="$PROJECT_DIR/private/rebuttal_200/$WORKFLOW_ID"
K8S_LOG_DIR="$WORKFLOW_DIR/kubernetes"
BUNDLE_PATH="${TRAJWIKI_BUNDLE_PATH:-$PROJECT_DIR/verification_bundle_${WORKFLOW_ID}.tar.zst}"

mkdir -p "$K8S_LOG_DIR"
cd "$PROJECT_DIR"

if [[ -z "$PARENT_60_EXPERIMENT" && -r "$PARENT_DIR_FILE" ]]; then
    PARENT_60_EXPERIMENT="$(tr -d '\r\n' <"$PARENT_DIR_FILE")"
fi
if [[ -z "$PARENT_60_EXPERIMENT" ]]; then
    echo "The 60-query parent experiment path is not configured." >&2
    echo "Set TRAJWIKI_PARENT_60_EXPERIMENT or provide $PARENT_DIR_FILE." >&2
    exit 1
fi

for command_name in docker kubectl python3 sed git tee; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command is unavailable: $command_name" >&2
        exit 1
    fi
done

for required_path in \
    "$PROJECT_DIR/.env" \
    "$PROJECT_DIR/Dockerfile" \
    "$GPU_JOB_TEMPLATE" \
    "$PACKAGE_JOB_TEMPLATE" \
    "$RUN_DIR/details.json" \
    "$RUN_DIR/trajpatch.sqlite" \
    "$MEM0_SOURCE" \
    "$PARENT_60_EXPERIMENT/experiment_manifest.json" \
    "$PARENT_60_EXPERIMENT/sampling_manifest.json" \
    "$PARENT_60_EXPERIMENT/provider_call_rows.jsonl"; do
    if [[ ! -r "$required_path" ]]; then
        echo "Required input is missing or unreadable: $required_path" >&2
        exit 1
    fi
done

python3 - "$RUN_DIR" "$PARENT_60_EXPERIMENT" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
parent = Path(sys.argv[2]).resolve()
manifest = json.loads(
    (parent / "experiment_manifest.json").read_text(encoding="utf-8")
)
sampling = json.loads(
    (parent / "sampling_manifest.json").read_text(encoding="utf-8")
)
if manifest.get("status") != "complete":
    raise SystemExit("The 60-query parent experiment is not complete.")
selected = list(sampling.get("selected_queries") or [])
if len(selected) != 60:
    raise SystemExit(
        f"The parent sampling manifest has {len(selected)} queries, expected 60."
    )
expected_details = str(manifest.get("source_details_sha256") or "")
actual_details = hashlib.sha256(
    (run_dir / "details.json").read_bytes()
).hexdigest()
if not expected_details or expected_details != actual_details:
    raise SystemExit(
        "The parent experiment and requested source run have different "
        "details.json hashes."
    )
expected_database = str(manifest.get("source_database_sha256") or "")
actual_database = hashlib.sha256(
    (run_dir / "trajpatch.sqlite").read_bytes()
).hexdigest()
if not expected_database or expected_database != actual_database:
    raise SystemExit(
        "The parent experiment and requested source run have different "
        "SQLite hashes."
    )
config = dict(manifest.get("config") or {})
expected_protocol = {
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
mismatches = {
    key: {"expected": expected, "observed": config.get(key)}
    for key, expected in expected_protocol.items()
    if config.get(key) != expected
}
if mismatches:
    raise SystemExit(f"The parent protocol is incompatible: {mismatches}.")
if set(manifest.get("baseline_methods") or []) != {"mem0_saved"}:
    raise SystemExit(
        "The parent experiment must contain the saved Mem0 baseline."
    )
for report_name in ("integrity_report.json", "validation_report.json"):
    report_path = parent / report_name
    if not report_path.is_file():
        raise SystemExit(f"The parent experiment is missing {report_name}.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report.get("error_count") or 0):
        raise SystemExit(f"The parent experiment failed {report_name}.")
provider_rows = sum(
    1
    for line in (parent / "provider_call_rows.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
)
if provider_rows != 1200:
    raise SystemExit(
        f"The parent provider ledger has {provider_rows} rows, expected 1200."
    )
print(
    {
        "parent_experiment": str(parent),
        "parent_query_count": len(selected),
        "parent_provider_call_rows": provider_rows,
        "source_details_sha256": actual_details,
        "source_database_sha256": actual_database,
    }
)
PY

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Formal rebuttal images must be built from a clean committed tree." >&2
    echo "Commit or intentionally ignore local changes before launching." >&2
    git status --short >&2
    exit 1
fi
GIT_COMMIT="$(git rev-parse HEAD)"

python3 - "$PROJECT_DIR/.env" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

names: set[str] = set()
for raw_line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    names.add(line.split("=", 1)[0].removeprefix("export ").strip())
missing = sorted({"OPENAI_API_KEY", "ANTHROPIC_API_KEY"} - names)
if missing:
    legacy = "Claud_API_Key" in names
    suffix = (
        " Rename Claud_API_Key to ANTHROPIC_API_KEY."
        if legacy and "ANTHROPIC_API_KEY" in missing
        else ""
    )
    raise SystemExit(f"Missing required API key names: {missing}.{suffix}")
print("Required API key names are present; values were not printed.")
PY

if [[ "$(kubectl auth can-i create jobs --namespace "$NS")" != "yes" ]]; then
    echo "Current kubectl identity cannot create Jobs in namespace $NS." >&2
    exit 1
fi
if [[ "$(kubectl auth can-i delete jobs --namespace "$NS")" != "yes" ]]; then
    echo "Current kubectl identity cannot delete Jobs in namespace $NS." >&2
    exit 1
fi

SOURCE_HASH="$(
    python3 - "$PROJECT_DIR" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = [
    root / "Dockerfile",
    root / "pyproject.toml",
    root / "README.md",
    root / "job.yml",
    root / "job-package.yml",
    *sorted((root / "src").rglob("*.py")),
    *sorted((root / "scripts").glob("*.sh")),
    *sorted((root / "tests").rglob("*.py")),
]
digest = hashlib.sha256()
for path in paths:
    relative = path.relative_to(root).as_posix()
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest()[:12])
PY
)"

if [[ "${TRAJWIKI_SKIP_IMAGE_BUILD:-0}" == "1" ]]; then
    IMAGE_REF="${TRAJWIKI_IMAGE_REF:-}"
    if [[ -z "$IMAGE_REF" || "$IMAGE_REF" != *@sha256:* ]]; then
        echo "TRAJWIKI_IMAGE_REF must be an immutable @sha256 digest." >&2
        exit 1
    fi
else
    IMAGE_TAG="${TRAJWIKI_IMAGE_TAG:-rebuttal-${SOURCE_HASH}-$(date -u +%Y%m%d%H%M%S)}"
    IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
    echo "Building Linux/amd64 image: $IMAGE"
    docker build \
        --pull \
        --platform linux/amd64 \
        --build-arg "TRAJWIKI_GIT_COMMIT=$GIT_COMMIT" \
        --build-arg "TRAJWIKI_GIT_DIRTY=0" \
        --build-arg "TRAJWIKI_SOURCE_HASH=$SOURCE_HASH" \
        -t traj:latest \
        -t "$IMAGE" \
        .
    IMAGE_PLATFORM="$(docker image inspect "$IMAGE" --format '{{.Architecture}}/{{.Os}}')"
    if [[ "$IMAGE_PLATFORM" != "amd64/linux" ]]; then
        echo "Unexpected image platform: $IMAGE_PLATFORM" >&2
        exit 1
    fi
    docker push "$IMAGE"
    docker pull "$IMAGE" >/dev/null
    IMAGE_REF="$(docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}')"
    if [[ -z "$IMAGE_REF" || "$IMAGE_REF" != *@sha256:* ]]; then
        echo "Unable to resolve an immutable pushed image digest." >&2
        exit 1
    fi
fi

printf '%s\n' "$IMAGE_REF" >"$WORKFLOW_DIR/image_digest.txt"
printf '%s\n' "$GIT_COMMIT" >"$WORKFLOW_DIR/git_commit.txt"
printf '%s\n' "$SOURCE_HASH" >"$WORKFLOW_DIR/source_hash.txt"
python3 -m pip freeze >"$WORKFLOW_DIR/controller_pip_freeze.txt"
echo "Workflow: $WORKFLOW_ID"
echo "Kubernetes image: $IMAGE_REF"

render_job() {
    local mode="$1"
    local template="$GPU_JOB_TEMPLATE"
    if [[ "$mode" == "package" ]]; then
        template="$PACKAGE_JOB_TEMPLATE"
    fi
    sed \
        -e "s#image: jingyusun/traj:latest#image: ${IMAGE_REF}#" \
        -e "s#args: \\[ \"formal\" \\]#args: [ \"${mode}\" ]#" \
        -e "s#WORKFLOW_ID#${WORKFLOW_ID}#g" \
        -e "s#JOB_MODE#${mode}#g" \
        -e "s#IMAGE_SOURCE#${SOURCE_HASH}#g" \
        -e "s#IMAGE_REF_VALUE#${IMAGE_REF}#g" \
        -e "s#RUN_DIR_VALUE#${RUN_DIR}#g" \
        -e "s#MEM0_SOURCE_VALUE#${MEM0_SOURCE}#g" \
        -e "s#PARENT_60_VALUE#${PARENT_60_EXPERIMENT}#g" \
        -e "s#PROJECT_DIR_VALUE#${PROJECT_DIR}#g" \
        -e "s#PRIVATE_DIR_VALUE#${WORKFLOW_DIR}#g" \
        -e "s#BUNDLE_PATH_VALUE#${BUNDLE_PATH}#g" \
        "$template"
}

CURRENT_JOB=""
CURRENT_POD=""
CURRENT_MODE=""

archive_job_state() {
    local job="$1"
    local pod="$2"
    local mode="$3"
    local prefix="$K8S_LOG_DIR/${mode}_${job}"
    kubectl get job "$job" -n "$NS" -o yaml >"${prefix}_job.yaml" 2>&1 || true
    kubectl describe job "$job" -n "$NS" >"${prefix}_job_describe.txt" 2>&1 || true
    if [[ -n "$pod" ]]; then
        kubectl get pod "$pod" -n "$NS" -o yaml >"${prefix}_pod.yaml" 2>&1 || true
        kubectl describe pod "$pod" -n "$NS" >"${prefix}_pod_describe.txt" 2>&1 || true
        kubectl logs "$pod" -n "$NS" >"${prefix}_final.log" 2>&1 || true
    fi
}

delete_exact_job() {
    local job="$1"
    if [[ -n "$job" ]] && kubectl get job "$job" -n "$NS" >/dev/null 2>&1; then
        kubectl delete job "$job" -n "$NS" --wait=true
    fi
}

cleanup_current_job() {
    if [[ -n "$CURRENT_JOB" ]]; then
        archive_job_state "$CURRENT_JOB" "$CURRENT_POD" "$CURRENT_MODE"
        delete_exact_job "$CURRENT_JOB"
        CURRENT_JOB=""
        CURRENT_POD=""
        CURRENT_MODE=""
    fi
}
trap cleanup_current_job EXIT INT TERM

run_job() {
    local mode="$1"
    local attempt_label="${2:-initial}"
    local rendered="$K8S_LOG_DIR/${mode}_${attempt_label}_rendered.yaml"
    local job=""
    local pod=""
    local log_path=""
    local succeeded=""
    local failed=""

    render_job "$mode" >"$rendered"
    kubectl create --dry-run=client -f "$rendered" -n "$NS" -o yaml >/dev/null
    echo "Rendered Job validated: mode=$mode attempt=$attempt_label"
    job="$(kubectl create -f "$rendered" -n "$NS" -o jsonpath='{.metadata.name}')"
    CURRENT_JOB="$job"
    CURRENT_MODE="$mode"
    echo "Created Job: $job"

    while [[ -z "$pod" ]]; do
        pod="$(
            kubectl get pods \
                -n "$NS" \
                -l "job-name=$job" \
                -o jsonpath='{.items[0].metadata.name}' \
                2>/dev/null \
                || true
        )"
        if [[ -z "$pod" ]]; then
            kubectl get job "$job" -n "$NS"
            sleep 5
        fi
    done
    CURRENT_POD="$pod"
    echo "Pod: $pod"
    kubectl get pod "$pod" -n "$NS" -o wide

    log_path="$K8S_LOG_DIR/${mode}_${attempt_label}_${job}.log"
    set +e
    kubectl logs -f "$pod" -n "$NS" 2>&1 | tee "$log_path"
    set -e

    while true; do
        succeeded="$(
            kubectl get job "$job" -n "$NS" \
                -o jsonpath='{.status.succeeded}' 2>/dev/null || true
        )"
        failed="$(
            kubectl get job "$job" -n "$NS" \
                -o jsonpath='{.status.failed}' 2>/dev/null || true
        )"
        if [[ "$succeeded" == "1" ]]; then
            break
        fi
        if [[ -n "$failed" && "$failed" != "0" ]]; then
            break
        fi
        sleep 5
    done

    archive_job_state "$job" "$pod" "$mode"
    delete_exact_job "$job"
    CURRENT_JOB=""
    CURRENT_POD=""
    CURRENT_MODE=""
    if [[ "$succeeded" == "1" ]]; then
        echo "Job completed and deleted: $job"
        return 0
    fi
    echo "Job failed and was archived/deleted: $job" >&2
    return 1
}

if [[ "$MODE" == "dry-run" ]]; then
    run_job dry-run
elif [[ "$MODE" == "formal" ]]; then
    if ! run_job formal initial; then
        echo "Creating the single allowed resume Job." >&2
        run_job formal resume
    fi
elif [[ "$MODE" == "package" ]]; then
    run_job package
else
    run_job dry-run
    if [[ "${TRAJWIKI_CONFIRM_FORMAL:-}" != "RUN" ]]; then
        read -r -p "Dry-run passed. Type RUN to create the paid formal Job: " answer
        if [[ "$answer" != "RUN" ]]; then
            echo "Formal Job was not created."
            exit 0
        fi
    fi
    if ! run_job formal initial; then
        echo "Creating the single allowed resume Job." >&2
        run_job formal resume
    fi
    run_job package
fi

residual_jobs=()
while IFS= read -r job; do
    if [[ -n "$job" ]]; then
        residual_jobs+=("$job")
    fi
done < <(
    kubectl get jobs -n "$NS" \
        -l "trajwiki.org/workflow=$WORKFLOW_ID" \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
)
for job in "${residual_jobs[@]}"; do
    if [[ -n "$job" ]]; then
        echo "Deleting residual workflow Job by exact name: $job" >&2
        delete_exact_job "$job"
    fi
done
remaining="$(
    kubectl get jobs -n "$NS" \
        -l "trajwiki.org/workflow=$WORKFLOW_ID" \
        -o jsonpath='{.items[*].metadata.name}'
)"
if [[ -n "$remaining" ]]; then
    echo "Residual workflow Jobs remain: $remaining" >&2
    exit 1
fi

trap - EXIT INT TERM
echo "Workflow $WORKFLOW_ID completed with no residual Jobs."
echo "Logs: $K8S_LOG_DIR"
if [[ -r "$WORKFLOW_DIR/rebuttal_bundle_path.txt" ]]; then
    echo "Bundle: $(tr -d '\r\n' <"$WORKFLOW_DIR/rebuttal_bundle_path.txt")"
fi
