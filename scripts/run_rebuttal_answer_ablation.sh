#!/usr/bin/env bash

set -euo pipefail
umask 077

MODE="${1:-formal}"
case "$MODE" in
    preflight|dry-run|formal|package)
        ;;
    *)
        echo "Usage: $0 {preflight|dry-run|formal|package}" >&2
        exit 2
        ;;
esac

PROJECT_DIR="${TRAJWIKI_PROJECT_DIR:-/data/users/jingyu/TrajWiki}"
RUN_DIR="${TRAJWIKI_RUN_DIR:-$PROJECT_DIR/output/remote_gpt4omini/locomo/multi_hop/20260507_001755_559270_rc900bf_locomo_multi-hop_gpt-4o-mini_gpt-4o-mini_qwen3-embedding-8b_m15_tp15_k15_nr1_remupdate-linked-plus-neighbors}"
MEM0_SOURCE="${TRAJWIKI_MEM0_SOURCE:-$PROJECT_DIR/locomo_eval/outputs/runs/mem0__gpt-4o-mini__category_1_multi_hop__20260406-010812/predictions.jsonl}"
PRIVATE_DIR="${TRAJWIKI_REBUTTAL_PRIVATE_DIR:-$PROJECT_DIR/private/rebuttal_200}"
LEGACY_PRIVATE_DIR="${TRAJWIKI_REBUTTAL_PILOT_DIR:-$PROJECT_DIR/private/rebuttal}"
MEM0_NORMALIZED="$PRIVATE_DIR/mem0_saved.normalized.jsonl"
EXPERIMENT_DIR_FILE="$PRIVATE_DIR/answer_ablation_200_experiment_dir.txt"
PARENT_DIR_FILE="${TRAJWIKI_PARENT_60_FILE:-$LEGACY_PRIVATE_DIR/answer_ablation_experiment_dir.txt}"
CLI="${TRAJWIKI_CLI:-trajwiki}"

export NO_COLOR=1
export PYTHONUNBUFFERED=1

mkdir -p "$PRIVATE_DIR"
cd "$PROJECT_DIR"

if ! command -v "$CLI" >/dev/null 2>&1; then
    echo "TrajWiki CLI is not installed in the image: $CLI" >&2
    exit 1
fi

if [[ "$MODE" == "package" ]]; then
    if [[ ! -r "$EXPERIMENT_DIR_FILE" ]]; then
        echo "Missing 200-query experiment pointer: $EXPERIMENT_DIR_FILE" >&2
        exit 1
    fi
    EXPERIMENT_DIR="$(tr -d '\r\n' <"$EXPERIMENT_DIR_FILE")"
    SAMPLING_MANIFEST="$EXPERIMENT_DIR/sampling_manifest.json"
    for required_path in \
        "$RUN_DIR/details.json" \
        "$RUN_DIR/trajpatch.sqlite" \
        "$EXPERIMENT_DIR/experiment_manifest.json" \
        "$SAMPLING_MANIFEST"; do
        if [[ ! -r "$required_path" ]]; then
            echo "Package input is missing or unreadable: $required_path" >&2
            exit 1
        fi
    done

    "$CLI" analyze-answer-ablation "$EXPERIMENT_DIR"
    "$CLI" analyze-offline-ablation "$RUN_DIR" \
        --variants full,no_wiki_direct,wiki_only,flat_raw,snapshot_m1,snapshot_m2,source_supported_only \
        --budgets 4000,8000,16000,32000 \
        --rank-cutoffs 5,10,15,20,30,50 \
        --sampling-manifest "$SAMPLING_MANIFEST"
    "$CLI" analyze-cost-benefit "$RUN_DIR" \
        --baselines trajwiki_observed,full_context_proxy,no_wiki_direct,flat_raw,wiki_only \
        --future-query-counts 1,2,5,10,20,50,100 \
        --sampling-manifest "$SAMPLING_MANIFEST"
    "$CLI" analyze-auditability "$RUN_DIR" \
        --baselines trajwiki_observed,full_context_proxy,no_wiki_direct,flat_raw,wiki_only \
        --sampling-manifest "$SAMPLING_MANIFEST"
    "$CLI" analyze-failures \
        --run-path "$RUN_DIR" \
        --sampling-manifest "$SAMPLING_MANIFEST"
    "$CLI" analyze-ranking-robustness "$RUN_DIR" \
        --relative-perturbation 0.20 \
        --random-draws 100 \
        --seed 7 \
        --cutoff 15 \
        --page-cutoffs 5,10,15 \
        --trajectory-cutoffs 5,10,15,20,30 \
        --sampling-manifest "$SAMPLING_MANIFEST"
    "$CLI" prepare-audit-study "$RUN_DIR" \
        --case-count 24 \
        --seed 7 \
        --answer-experiment "$EXPERIMENT_DIR"
    "$CLI" validate-run-artifacts "$EXPERIMENT_DIR"

    BUNDLE_PATH="${TRAJWIKI_BUNDLE_PATH:-$PROJECT_DIR/verification_bundle_$(date -u +%Y%m%d_%H%M%S).tar.zst}"
    "$CLI" package-rebuttal-bundle "$EXPERIMENT_DIR" \
        --output-path "$BUNDLE_PATH" \
        --workflow-logs "$PRIVATE_DIR" \
        --repository-path "$PROJECT_DIR"
    "$CLI" validate-rebuttal-bundle "$BUNDLE_PATH"
    printf '%s\n' "$BUNDLE_PATH" >"$PRIVATE_DIR/rebuttal_bundle_path.txt"
    echo "Offline analyses, audit study, and bundle completed."
    echo "Bundle: $BUNDLE_PATH"
    exit 0
fi

for required_path in \
    "$PROJECT_DIR/.env" \
    "$RUN_DIR/details.json" \
    "$RUN_DIR/trajpatch.sqlite" \
    "$MEM0_SOURCE"; do
    if [[ ! -r "$required_path" ]]; then
        echo "Required input is missing or unreadable: $required_path" >&2
        exit 1
    fi
done

PARENT_EXPERIMENT="${TRAJWIKI_PARENT_60_EXPERIMENT:-}"
if [[ -z "$PARENT_EXPERIMENT" && -r "$PARENT_DIR_FILE" ]]; then
    PARENT_EXPERIMENT="$(tr -d '\r\n' <"$PARENT_DIR_FILE")"
fi
if [[ -z "$PARENT_EXPERIMENT" || ! -r "$PARENT_EXPERIMENT/experiment_manifest.json" ]]; then
    echo "A complete 60-query parent experiment is required." >&2
    echo "Set TRAJWIKI_PARENT_60_EXPERIMENT or provide $PARENT_DIR_FILE." >&2
    exit 1
fi

python - "$PROJECT_DIR/.env" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values

env_path = Path(sys.argv[1])
values = {
    **{key: value for key, value in dotenv_values(env_path).items() if value},
    **{key: value for key, value in os.environ.items() if value},
}
required = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
missing = [key for key in required if not values.get(key)]
if missing:
    legacy = "Claud_API_Key" in values
    suffix = (
        " Rename Claud_API_Key to ANTHROPIC_API_KEY."
        if legacy and "ANTHROPIC_API_KEY" in missing
        else ""
    )
    raise SystemExit(f"Missing required API keys: {missing}.{suffix}")
print("API key names validated without printing secret values.")
PY

python - <<'PY'
from __future__ import annotations

from importlib.metadata import version

import torch

packages = [
    "trajwiki",
    "torch",
    "transformers",
    "sentence-transformers",
    "vllm",
    "tiktoken",
    "zstandard",
    "openai",
    "anthropic",
]
versions = {name: version(name) for name in packages}
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA inside the Pod.")
if not str(torch.version.cuda or "").startswith("12.6"):
    raise SystemExit(
        f"Expected a CUDA 12.6 PyTorch build, found {torch.version.cuda!r}."
    )
print(
    {
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_0": torch.cuda.get_device_name(0),
    }
)
PY

nvidia-smi --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader,nounits

"$CLI" validate-run-artifacts "$RUN_DIR"
"$CLI" import-baseline-answers "$MEM0_SOURCE" \
    --method mem0_saved \
    --run-path "$RUN_DIR" \
    --output-path "$MEM0_NORMALIZED"

python - "$MEM0_NORMALIZED" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != 282:
    raise SystemExit(f"Expected 282 normalized Mem0 rows, found {len(rows)}.")
if {str(row.get("method")) for row in rows} != {"mem0_saved"}:
    raise SystemExit("Normalized Mem0 method label is invalid.")
print({"baseline_rows": len(rows), "method": "mem0_saved"})
PY

if [[ "$MODE" == "preflight" ]]; then
    echo "Preflight completed. No model calls were made."
    exit 0
fi

ABLATION_ARGS=(
    "$RUN_DIR"
    --sample-size 200
    --sampling-profile rebuttal_200_v1
    --sample-seed 7
    --variants full,direct_trajectory,latest_snapshot,hybrid_raw_rag,wiki_summaries,no_claim_state,no_source_constraint,full_context,naive_dense_rag,full_context_matched,hybrid_raw_rag_matched
    --reuse-experiment "$PARENT_EXPERIMENT"
    --reuse-policy require
    --max-total-tokens 32000
    --max-output-tokens 512
    --token-counter tiktoken
    --require-exact-token-counter
    --token-safety-margin 128
    --rag-chunk-size 384
    --rag-chunk-overlap 64
    --rag-top-k 4
    --baseline-answers "$MEM0_NORMALIZED"
    --backbone-provider-kind remote
    --backbone-model gpt-4o-mini
    --independent-judge-provider-kind remote
    --independent-judge-model claude-sonnet-4-6
    --generation-max-concurrency 6
    --judge-max-concurrency 6
    --context-save-mode full
    --max-provider-calls 5300
    --progress
    --progress-interval-seconds 30
    --resume
)

DRY_RUN_REPORT="$PRIVATE_DIR/answer_ablation_200_dry_run_report.json"
"$CLI" run-answer-ablation "${ABLATION_ARGS[@]}" \
    --dry-run \
    --report-path "$DRY_RUN_REPORT"

python - "$DRY_RUN_REPORT" "$EXPERIMENT_DIR_FILE" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
experiment_dir = Path(report["experiment_dir"])
call_plan = json.loads(
    (experiment_dir / "call_plan.json").read_text(encoding="utf-8")
)
expected = {
    "selected_query_count": 200,
    "generated_variant_count": 11,
    "expected_variant_context_row_count": 2200,
    "expected_answer_row_count": 2600,
    "unavailable_stage_row_count": 400,
    "reused_generation_job_count": 540,
    "reused_judgment_job_count": 660,
    "reused_provider_call_row_count": 1200,
    "expected_successful_logical_provider_work": 4800,
    "token_counter_exact": True,
}
for field, value in expected.items():
    if call_plan.get(field) != value:
        raise SystemExit(
            f"Dry-run mismatch for {field}: "
            f"{call_plan.get(field)!r} != {value!r}"
        )
generation_calls = int(call_plan.get("generation_call_count") or 0)
judge_calls = int(call_plan.get("judge_call_count") or 0)
planned_calls = int(call_plan.get("provider_call_count") or 0)
existing_calls = int(call_plan.get("existing_provider_call_count") or 0)
existing_new_calls = int(
    call_plan.get("existing_new_provider_call_count") or 0
)
projected_calls = int(call_plan.get("projected_provider_call_count_total") or 0)
if planned_calls != generation_calls + judge_calls:
    raise SystemExit("Planned provider calls do not equal generation plus judge.")
if existing_calls == 1200:
    if existing_new_calls != 0:
        raise SystemExit(
            "A fresh nested extension unexpectedly contains new provider calls."
        )
    if (generation_calls, judge_calls, planned_calls, projected_calls) != (
        1660,
        1940,
        3600,
        4800,
    ):
        raise SystemExit(
            "Fresh nested extension must plan 1660/1940/3600 new calls "
            f"and 4800 total; observed "
            f"{generation_calls}/{judge_calls}/{planned_calls}/{projected_calls}."
        )
else:
    if not (
        1200 < existing_calls <= 5300
        and 0 < existing_new_calls <= 4100
        and 0 <= generation_calls <= 1660
        and 0 <= judge_calls <= 1940
        and projected_calls <= 5300
    ):
        raise SystemExit(
            "Resume call plan is outside the allowed protocol envelope: "
            f"existing={existing_calls}, generation={generation_calls}, "
            f"judge={judge_calls}, projected={projected_calls}."
        )
with (experiment_dir / "token_budget_audit.csv").open(
    "r",
    encoding="utf-8",
    newline="",
) as handle:
    budget_rows = list(csv.DictReader(handle))
if len(budget_rows) != 2200:
    raise SystemExit(f"Expected 2200 budget rows, found {len(budget_rows)}.")
bad = [
    f"{row['variant']}/{row['query_task_id']}"
    for row in budget_rows
    if row.get("token_counter_exact") != "True"
    or row.get("budget_passed_before_call") != "True"
    or row.get("matched_budget_passed_before_call") == "False"
]
if bad:
    raise SystemExit(f"Token-budget dry-run failed: {bad[:5]}")
Path(sys.argv[2]).write_text(str(experiment_dir) + "\n", encoding="utf-8")
print(json.dumps(call_plan, indent=2, sort_keys=True))
print(f"Experiment directory: {experiment_dir}")
PY

if [[ "$MODE" == "dry-run" ]]; then
    echo "Dry-run completed. No new provider calls were made."
    exit 0
fi

python - "$PROJECT_DIR/.env" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(sys.argv[1]), override=True)
openai_model = OpenAI().models.retrieve("gpt-4o-mini")
anthropic_model = Anthropic().models.retrieve(model_id="claude-sonnet-4-6")
print(
    {
        "remote_model_access_validated": True,
        "backbone_model": openai_model.id,
        "independent_judge_model": anthropic_model.id,
    }
)
PY

FORMAL_REPORT="$PRIVATE_DIR/answer_ablation_200_formal_report.json"
"$CLI" run-answer-ablation "${ABLATION_ARGS[@]}" \
    --report-path "$FORMAL_REPORT"

EXPERIMENT_DIR="$(tr -d '\r\n' <"$EXPERIMENT_DIR_FILE")"
"$CLI" analyze-answer-ablation "$EXPERIMENT_DIR"
"$CLI" validate-run-artifacts "$EXPERIMENT_DIR"

python - "$EXPERIMENT_DIR/validation_report.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(report.get("error_count") or 0):
    raise SystemExit(f"Formal experiment validation failed: {report.get('errors')}")
expected = {
    "selected_query_count": 200,
    "answer_row_count": 2600,
    "independent_judgment_row_count": 2600,
    "variant_context_row_count": 2200,
}
for field, value in expected.items():
    if int(report.get(field) or 0) != value:
        raise SystemExit(
            f"Formal artifact mismatch for {field}: "
            f"{report.get(field)!r} != {value}"
        )
print(json.dumps(report, indent=2, sort_keys=True))
PY

cp "$EXPERIMENT_DIR/validation_report.json" \
    "$PRIVATE_DIR/answer_ablation_200_validation.json"
echo "Formal 200-query answer-ablation experiment completed successfully."
echo "Experiment directory: $EXPERIMENT_DIR"
