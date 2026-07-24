# TrajWiki

TrajWiki is an offline benchmark framework for long-memory question answering. It builds
source-grounded episodic memory trajectories, compiles a per-sample wiki routing layer,
retrieves compact evidence, generates grounded answers, and stores detailed diagnostics for
failure analysis, offline counterfactual ablations, cost-benefit analysis, and auditability
evaluation.

## What The Pipeline Does

For each logical sample, TrajWiki runs a fixed memory-and-evaluation pipeline:

1. Load LOCOMO or MedMT data and group rows into logical samples.
2. Convert conversation history into episodic snapshots and source-linked claims.
3. Append snapshots to bounded memory trajectories controlled by `--m`.
4. Compile trajectory summaries and source-surface signals into wiki pages.
5. Route each query through wiki pages, then select top trajectories and snapshots.
6. Build answer context from compacted snapshots plus raw source messages.
7. Generate a LOCOMO free-form answer with `Answer:` and `Rationale:`.
8. Apply deterministic and type-specific validation for date, count, list, place, bridge,
   and abstention cases.
9. Evaluate answers with semantic `F1`, `BLEU-1`, and an LLM judge.
10. Persist SQLite records, memory exports, summary/detail JSON, and optional analysis
    artifacts for retrieval ablation, cost analysis, and provenance auditing.

The default LOCOMO answer path is `freeform_v2`. The older structured
`answer_evidence_synthesis` path remains as a legacy fallback, not the default.

## Supported Benchmarks

### LOCOMO

LOCOMO rows are grouped by conversation. Each conversation gets its own memory trajectories
and wiki pages.

Supported subset keys:

- `all`
- `multi_hop`
- `temporal`
- `open_domain`
- `single_hop`

Expected directory layout:

```text
locomo/
  category_1_multi_hop.jsonl
  category_2_temporal.jsonl
  category_3_open_domain.jsonl
  category_4_single_hop.jsonl
```

### MedMT

MedMT samples are treated independently. All turns except the final user turn become memory
history; the final user turn becomes the query.

Supported subset keys:

- `all`
- `long_context_memory_and_understanding`
- `resistance_to_contextual_interference`
- `information_contradiction`

Expected directory layout:

```text
medmt/
  long_context_memory_and_understanding.json
  resistance_to_contextual_interference.json
  information_contradiction.json
```

### Research And Deployment Boundaries

TrajWiki is a research prototype, not a clinical decision system or a compliance-ready
personal-memory service. MedMT evaluates long-context memory behavior only; generated
medical content must not be treated as medical advice. The reference implementation stores
run artifacts in local files and SQLite and does not provide encryption at rest, access
control, tenant isolation, consent management, retention policies, or HIPAA/GDPR
certification.

Append-only snapshots preserve an epistemic revision history, but they must not be
interpreted as a requirement to retain personal data indefinitely. A production deployment
needs an authenticated hard-deletion path that removes the affected raw messages,
snapshots, claims, embeddings, wiki links, caches, and exported diagnostics, followed by
index rebuilding and deletion verification. High-risk domains additionally require
freshness checks, conflict quarantine, source verification, and human oversight before
memory updates or answers can influence decisions.

## Installation

Use Python 3.11 or newer.

```bash
python -m pip install -e .
```

Install development and test tools when editing the project:

```bash
python -m pip install -e ".[dev]"
```

The default install includes vLLM support for `openai-compatible` local runs and
`--vllm-autostart`. vLLM is pinned to a CUDA 12-compatible release for the
`nvidia/cuda:12.6` runtime family, and Transformers is pinned to the matching tokenizer
stack used by vLLM 0.10.x. Avoid replacing these with unconstrained `vllm>=...` or
`transformers>=...` dependencies unless the CUDA/runtime stack is upgraded together. The
optional `quant` extra only adds quantization-specific packages such as `bitsandbytes`.

Check the CLI:

```bash
trajwiki --help
trajpatch --help
PYTHONPATH=src python -m trajpatch --help
```

`trajwiki` is the preferred console command. `trajpatch` and `python -m trajpatch` remain
available for backward compatibility with older scripts and cache/database paths.

Remote providers read API keys from environment variables or `.env`:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
HF_TOKEN=...
```

OpenAI-compatible local or private endpoints use separate settings:

```bash
OPENAI_COMPATIBLE_BASE_URL=http://localhost:8000/v1
OPENAI_COMPATIBLE_API_KEY=EMPTY
```

`OPENAI_COMPATIBLE_BASE_URL` points to your local or private OpenAI-compatible service. It
does not call OpenAI unless you explicitly point it at an OpenAI endpoint.

## Running Benchmarks

### Smoke Test

Use mock providers and hash embeddings to validate plumbing without model calls:

```bash
PYTHONPATH=src python -m trajpatch benchmark-locomo \
  --dataset-path data/locomo \
  --subset multi_hop \
  --backbone-provider-kind mock \
  --judge-provider-kind mock \
  --embedding-model hash-embedding \
  --max-samples 1 \
  --output-dir output/smoke
```

### LOCOMO With Remote Models

Remote mode only sends backbone and judge calls to the remote provider. The embedding
model below is still loaded locally; with CUDA enabled, `Qwen/Qwen3-Embedding-8B`
uses local GPU memory.

```bash
PYTHONPATH=src python -m trajpatch benchmark-locomo \
  --dataset-path data/locomo \
  --subset multi_hop \
  --output-dir output/remote \
  --backbone-provider-kind remote \
  --judge-provider-kind remote \
  --backbone-model gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --device-mode auto \
  --m 15 \
  --t-pages 15 \
  --k 15 \
  --neighbor-radius 1 \
  --retrieval-expansion-mode update_linked_plus_neighbors \
  --memory-extract-batch-size auto \
  --judge-max-concurrency 6 \
  --conv-workers 5 \
  --ablation-diagnostics \
  --retrieval-rank-save-mode full \
  --cost-diagnostics \
  --cost-call-save-mode compact \
  --auditability-diagnostics \
  --audit-packet-save-mode compact \
  --rebuild-memory-cache \
  --rebuild-semantic-metric-cache \
  --verbose
```

The diagnostic flags above do not add extra answer-generation calls. They write compact
JSON/JSONL/CSV artifacts that can later be analyzed offline. For quick smoke tests, leave
them disabled.

### LOCOMO With Local vLLM

This is the recommended local-model path for running Qwen or other local chat models.
vLLM exposes the model through an OpenAI-compatible API. For Qwen3 reasoning models,
prefer TrajWiki `text_json` mode: it avoids vLLM guided-JSON engine crashes and the
TrajWiki text parsers tolerate Qwen `<think>` reasoning preambles. Use `vllm` structured
mode only after validating that the specific model and vLLM release can handle guided JSON
schemas under your target concurrency.

Option 1: start vLLM yourself:

```bash
vllm serve Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen3-8b \
  --gpu-memory-utilization 0.70 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --trust-remote-code \
  --enforce-eager \
  --generation-config vllm
```

Then run TrajWiki against that endpoint:

```bash
PYTHONPATH=src python -m trajpatch benchmark-locomo \
  --dataset-path data/locomo \
  --subset multi_hop \
  --output-dir output/vllm_local \
  --index-database-path output/vllm_local/trajpatch_index.sqlite \
  --memory-cache-dir .trajpatch_cache_vllm_local \
  --backbone-provider-kind openai-compatible \
  --judge-provider-kind openai-compatible \
  --backbone-model qwen3-8b \
  --judge-model qwen3-8b \
  --openai-compatible-base-url http://localhost:8000/v1 \
  --openai-compatible-api-key EMPTY \
  --openai-compatible-structured-mode text_json \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --device-mode auto \
  --m 15 \
  --t-pages 20 \
  --k 20 \
  --neighbor-radius 1 \
  --retrieval-expansion-mode update_linked_plus_neighbors \
  --memory-extract-batch-size auto \
  --judge-max-concurrency 4 \
  --conv-workers 4 \
  --rebuild-memory-cache \
  --rebuild-semantic-metric-cache \
  --verbose
```

Option 2: let TrajWiki start and stop vLLM for this run:

```bash
PYTHONPATH=src python -m trajpatch benchmark-locomo \
  --dataset-path data/locomo \
  --subset multi_hop \
  --output-dir output/vllm_local \
  --index-database-path output/vllm_local/trajpatch_index.sqlite \
  --memory-cache-dir .trajpatch_cache_vllm_local \
  --backbone-provider-kind openai-compatible \
  --judge-provider-kind openai-compatible \
  --backbone-model qwen3-8b \
  --judge-model qwen3-8b \
  --openai-compatible-api-key EMPTY \
  --openai-compatible-structured-mode text_json \
  --vllm-autostart \
  --vllm-model Qwen/Qwen3-8B \
  --vllm-served-model-name qwen3-8b \
  --vllm-cuda-visible-devices 0 \
  --vllm-gpu-memory-utilization 0.70 \
  --vllm-dtype bfloat16 \
  --vllm-extra-args "--max-model-len 8192 --trust-remote-code --enforce-eager --generation-config vllm" \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --device-mode auto \
  --m 15 \
  --t-pages 20 \
  --k 20 \
  --neighbor-radius 1 \
  --retrieval-expansion-mode update_linked_plus_neighbors \
  --memory-extract-batch-size auto \
  --judge-max-concurrency 4 \
  --conv-workers 4 \
  --rebuild-memory-cache \
  --rebuild-semantic-metric-cache \
  --verbose
```

With `--vllm-autostart`, the runner checks `/v1/models`, reuses an existing compatible
server when available, otherwise starts `vllm serve`, waits for readiness, and stops only
the process it started. Use `--vllm-keep-alive` to leave the launched server running.

### LOCOMO With Bare Local Transformers

This mode loads the Hugging Face model inside the TrajWiki process. It is useful for
debugging but less reliable for structured outputs than `openai-compatible`.

```bash
PYTHONPATH=src python -m trajpatch benchmark-locomo \
  --dataset-path data/locomo \
  --subset multi_hop \
  --output-dir output/local \
  --backbone-provider-kind local \
  --judge-provider-kind local \
  --backbone-model Qwen/Qwen3-8B \
  --judge-model Qwen/Qwen3-8B \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --device-mode auto \
  --m 15 \
  --t-pages 20 \
  --k 20 \
  --neighbor-radius 1 \
  --retrieval-expansion-mode update_linked_plus_neighbors \
  --memory-extract-batch-size 16 \
  --rebuild-memory-cache \
  --rebuild-semantic-metric-cache \
  --verbose
```

Do not use `--conv-workers > 1` with `--backbone-provider-kind local`.

### MedMT

```bash
PYTHONPATH=src python -m trajpatch benchmark-medmt \
  --dataset-path data/medmt \
  --subset all \
  --output-dir output/remote \
  --backbone-provider-kind remote \
  --judge-provider-kind remote \
  --backbone-model gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --device-mode auto
```

### Generic Runner

```bash
PYTHONPATH=src python -m trajpatch run \
  --dataset locomo \
  --subset temporal \
  --dataset-path data/locomo \
  --output-dir output
```

## CLI Commands

- `benchmark-locomo`: run LOCOMO with LOCOMO-specific grouping, metrics, and analysis.
- `benchmark-medmt`: run MedMT with MedMT-specific adapter and judge rubric.
- `run`: generic runner for either dataset.
- `sweep`: run parameter grids over `m`, `k`, `t_pages`, neighbor radius, and expansion
  mode.
- `report`: read a run database or cross-run index and print aggregate metrics.
- `inspect`: print one trajectory or snapshot from a run database.
- `export`: export trajectory artifacts from a run database.
- `analyze-failures`: LOCOMO failure attribution for a run or subset directory.
- `analyze-failures-diff`: compare two LOCOMO failure reports.
- `analyze-offline-ablation`: build offline counterfactual retrieval/context ablation
  tables from one completed LOCOMO run.
- `analyze-cost-benefit`: build cost breakdown, amortization, and memory/candidate scaling
  tables from one completed LOCOMO run.
- `analyze-auditability`: build source-support, unsupported-risk, failure-localization,
  conflict/obsolete, and audit-packet tables from one completed LOCOMO run.
- `run-answer-ablation`: run the resumable, stratified answer-level rebuttal experiment
  over one immutable LOCOMO memory build.
- `analyze-answer-ablation`: rebuild answer-level tables and paired statistics without
  new provider calls.
- `analyze-ranking-robustness`: perturb saved deterministic retrieval weights without
  rebuilding memory or calling an LLM.
- `validate-run-artifacts`: check provenance scope, provider-call uniqueness, and
  run/experiment completeness.
- `prepare-audit-study`, `conduct-audit-study`, and `analyze-audit-study`: prepare and
  analyze the blinded human provenance study.
- `import-baseline-answers`: normalize saved baseline predictions for the same blinded
  independent judge used by the answer ablation.

Run any command with `--help` for the full option list.

## Important Runtime Options

### Dataset And Output

- `--dataset-path`: dataset file or directory.
- `--subset`: dataset subset key.
- `--output-dir`: root directory for run artifacts.
- `--index-database-path`: optional cross-run index path. Use separate paths for remote,
  local, and vLLM runs when running them concurrently.
- `--max-samples`: debug limit over logical samples.
- `--reset-run-db / --no-reset-run-db`: recreate or reuse the per-run database path.
- `--verbose`: print stage-level traces, timings, routing decisions, fallback metadata,
  worker placement, and provider-call summaries.

### Retrieval And Memory

- `--m`: maximum number of snapshots per trajectory.
- `--t-pages`: number of wiki pages selected by page routing.
- `--k`: number of trajectories selected for answer generation.
- `--neighbor-radius`: number of neighboring snapshots added around retrieved snapshots.
- `--retrieval-expansion-mode`: expansion strategy. Common values are
  `update_linked_plus_neighbors`, `neighbors_only`, and `none`.
- `--memory-cache / --no-memory-cache`: enable or disable reusable sample-level memory
  cache.
- `--memory-cache-dir`: memory cache root.
- `--rebuild-memory-cache`: ignore and overwrite existing memory cache entries.
- `--rebuild-semantic-metric-cache`: regenerate LOCOMO semantic metric cache entries.

### Parallelism And Devices

- `--conv-workers`: logical sample worker count. Remote and `openai-compatible` providers
  support sharded workers; bare `local` does not. Sharded workers share a single
  `SentenceTransformer` embedding model instance per `(model, device)` inside the same
  process, so repeated workers on one GPU do not duplicate embedding weights.
- `--memory-extract-batch-size`: memory-stage batch size or `auto`.
- `--judge-max-concurrency`: judge and semantic-metric concurrency, or `auto`.
- `--device-mode`: embedding/device placement mode.
- `--cuda-preflight-mode`: CUDA memory preflight behavior: `off`, `warn`, or `strict`.
  The default `warn` records risk without blocking; `strict` fails before model loading on
  unsafe GPU plans.
- `--cuda-preflight-reserve-gb`: per-GPU safety reserve used by preflight estimates.
- `--cuda-preflight-report/--no-cuda-preflight-report`: write or suppress
  `status/cuda_preflight.json`.

For remote-like providers, `--memory-extract-batch-size auto` is worker-aware. For example,
`--conv-workers 5` resolves the per-worker memory extraction batch conservatively instead
of multiplying into an excessive remote-call burst. Run-level analysis artifacts are written
only by the main process after worker SQLite shards have been merged, so sharded workers do
not concurrently append to the same analysis JSONL files.

### Offline Diagnostic Options

These options are disabled by default and are intended mainly for LOCOMO paper runs:

- `--ablation-diagnostics`: save gold labels and full retrieval ranking diagnostics.
- `--retrieval-rank-save-mode`: `top-n` or `full`; use `full` for offline ablations.
- `--retrieval-rank-save-limit`: top-N retention when rank save mode is `top-n`.
- `--offline-context-budgets`: comma-separated context budgets recorded in run metadata.
- `--offline-rank-cutoffs`: comma-separated rank cutoffs recorded in run metadata.
- `--cost-diagnostics`: save compact cost query/call rows for offline cost analysis.
- `--cost-call-save-mode`: `summary` or `compact`; use `compact` for paper runs.
- `--cost-price-config`: optional pricing JSON for dollar-cost estimates.
- `--future-query-counts`: comma-separated future query counts for amortization curves.
- `--auditability-diagnostics`: save provenance, answer support, claim lifecycle, and audit
  packet rows.
- `--audit-packet-save-mode`: `summary` or `compact`; `compact` includes short source and
  claim previews, never embedding vectors.

The LOCOMO-only diagnostic writers are skipped on MedMT runs even if the flags are present.
This lets shared scripts pass the same option set to both benchmarks without producing
misleading MedMT ablation/audit artifacts.

### Provider Modes

- `--backbone-provider-kind` and `--judge-provider-kind`: one of `mock`, `remote`,
  `openai-compatible`, or `local`.
- `--backbone-model` and `--judge-model`: model identifiers for answer/memory generation
  and judging.
- `--embedding-model`: embedding model id. Use `hash-embedding` for mock/smoke tests.

Provider behavior:

| Provider kind | Intended use | Structured output behavior |
| --- | --- | --- |
| `mock` | Fast deterministic tests | Deterministic mock structures |
| `remote` | Hosted APIs through LiteLLM/vendor SDKs | Uses supported remote structured paths |
| `openai-compatible` | vLLM or private OpenAI-compatible services | `text_json` for Qwen3-style reasoning models; guided JSON/schema in `vllm` mode for validated servers |
| `local` | Bare Hugging Face Transformers in-process | Text-only; structured JSON can be brittle |

### OpenAI-Compatible And vLLM Options

- `--openai-compatible-base-url`: base URL for a local/private OpenAI-compatible server,
  for example `http://localhost:8000/v1`.
- `--openai-compatible-api-key`: API key for that server. vLLM commonly accepts `EMPTY`.
- `--openai-compatible-structured-mode`: `vllm`, `openai_json_schema`, or `text_json`.
  For Qwen3 on vLLM, `text_json` is the safer default because guided JSON can be unstable
  with reasoning chat templates, complex schemas, and concurrent requests.
- `--vllm-autostart`: explicitly start a vLLM server before the benchmark.
- `--vllm-model`: model passed to `vllm serve`.
- `--vllm-served-model-name`: model name exposed by `/v1/models`.
- `--vllm-host` and `--vllm-port`: bind address and port for the vLLM subprocess.
- `--vllm-cuda-visible-devices`: CUDA devices visible to the vLLM subprocess only.
- `--vllm-tensor-parallel-size`: vLLM tensor-parallel size.
- `--vllm-gpu-memory-utilization`: vLLM GPU memory utilization setting.
- `--vllm-dtype`: optional vLLM dtype.
- `--vllm-extra-args`: additional arguments appended to `vllm serve`.
- `--vllm-startup-timeout-s`: readiness timeout.
- `--vllm-keep-alive`: keep an autostarted vLLM process alive after the benchmark.

Remote runs never need vLLM. Bare local Transformers runs also do not need vLLM, because
they load the model directly in the TrajWiki process.

## Outputs

Each run is written under:

```text
<output_dir>/<dataset>/<dataset_scope>/<run_id>/
```

Main artifacts:

```text
summary.json                 Aggregate metrics, run meta, costs, diagnostics
details.json                 Per-query answers, metrics, retrieval and answer metadata
fallback_repair_events.jsonl Compact fallback/repair events
trajpatch.sqlite             Per-run SQLite database
status/events.jsonl          Lifecycle events and heartbeat/debug milestones
status/cuda_preflight.json   CUDA inventory, reservations, assignments, warnings, errors
status/vllm_server.log       vLLM stdout/stderr when autostart launches a server
status/vllm_server.json      Compact vLLM autostart/reuse metadata
run_failed.json              Present only for failed/incomplete runs
failed_shards/               Preserved worker shard DBs for failed sharded runs
debug/                       Optional structured-output and repair diagnostics
memories/<sample_id>/        Exported trajectory and wiki memory artifacts
```

Memory exports:

```text
memories/<sample_id>/trajectories.jsonl
memories/<sample_id>/<trajectory_id>.json
memories/<sample_id>/<trajectory_id>.summary.md
memories/<sample_id>/wiki/index.md
memories/<sample_id>/wiki/pages.jsonl
memories/<sample_id>/wiki/<page_id>.md
memories/<sample_id>/wiki/<page_id>.json
```

LOCOMO analysis artifacts produced during evaluation or by offline analysis commands:

```text
analysis/text_only_filter_manifest.json
analysis/text_only_filtered_summary.json
analysis/gold_labels.jsonl
analysis/offline_ablation_rows.jsonl
analysis/offline_ablation_summary.json
analysis/offline_ablation_table.csv
analysis/evidence_funnel.csv
analysis/cost_recall_curve.csv
analysis/variant_examples.jsonl
analysis/cost_call_rows.jsonl
analysis/cost_query_rows.jsonl
analysis/cost_phase_summary.csv
analysis/cost_quality_table.csv
analysis/amortization_break_even.csv
analysis/amortized_cost_curve.csv
analysis/memory_scaling.csv
analysis/candidate_scaling.csv
analysis/cost_benefit_summary.json
analysis/answer_support_rows.jsonl
analysis/answer_context_claim_rows.jsonl
analysis/claim_lifecycle_rows.jsonl
analysis/audit_packet_rows.jsonl
analysis/auditability_rows.jsonl
analysis/auditability_summary.json
analysis/source_support_table.csv
analysis/unsupported_answer_table.csv
analysis/failure_localization_table.csv
analysis/conflict_obsolete_table.csv
analysis/audit_packet_cost.csv
analysis/error_propagation_funnel.csv
analysis/audit_examples.jsonl
analysis/direct_retrieval_ablation.json
analysis/direct_retrieval_rows.jsonl
analysis/trajectory_drift_diagnostics.json
analysis/trajectory_drift_rows.jsonl
analysis/trajectory_drift_query_rows.jsonl
```

If a run already contains a legacy artifact schema, rebuilt sample-scoped diagnostics are
written under `analysis_v2/` instead of overwriting the originals. Compact call rows use
globally unique run/worker/role/call identifiers and include
`cost_reconciliation.json` when cost diagnostics are enabled.

Rebuttal experiments are stored separately:

```text
rebuttal_experiments/answer_ablation_<config_hash>/
  experiment_manifest.json
  sampling_manifest.json
  gold_label_rows.jsonl
  call_plan.json
  external_baseline_manifest.json
  retrieval_plan_rows.jsonl
  variant_context_rows.jsonl
  token_budget_audit.csv
  unavailable_stage_rows.jsonl
  contexts/<variant>/<query>.json.gz
  jobs/generation/<variant>/<query>.json
  jobs/judge/<variant>/<query>.json
  answer_ablation_rows.jsonl
  answer_ablation_table.csv
  paired_statistics.csv
  stage_diagnostic_statistics.csv
  stratum_summary.csv
  variant_cost_summary.csv
  balanced_failure_examples.jsonl
  integrity_report.json
  offline_analysis/
rebuttal_experiments/audit_study_seed<seed>_<count>/
  audit_tasks.jsonl
  audit_study_key.jsonl
  audit_study_manifest.json
  audit_labels_template.csv
```

Full prompts and raw retrieved context are intentionally not copied into compact
diagnostics. Large text stays in controlled debug artifacts or SQLite records where needed.
Full answer-ablation contexts and human-audit packets do contain dialogue text. Treat those
directories as sensitive and review dataset and privacy constraints before publishing them.

## Evaluation And Analysis

LOCOMO reports:

- `F1`: semantic canonical set F1 with slot-aware soft matching.
- `BLEU-1`: clipped unigram precision over canonical answer text, without brevity penalty.
- `judge_acc`: partial-credit LLM judge score, where correct is `1.0`, partial is `0.5`,
  incorrect is `0.0`, and infrastructure failures are `null`.
- Evidence and retrieval diagnostics, including gold source coverage when references are
  available.
- Text-only filtered metrics that exclude clear visual/OCR-dependent questions while
  leaving the original summary unchanged.
- Metadata term diagnostics for trajectory keyword and historical item fields, including
  internal-template leakage checks for retrieval-facing memory signals.

Run failure analysis:

```bash
PYTHONPATH=src python -m trajpatch analyze-failures \
  --run-path output/remote/locomo/multi_hop \
  --show-ranks \
  --show-facets
```

Compare two runs:

```bash
PYTHONPATH=src python -m trajpatch analyze-failures-diff \
  --before output/baseline/locomo/multi_hop \
  --after output/experiment/locomo/multi_hop
```

Offline counterfactual retrieval/context ablation:

```bash
trajwiki analyze-offline-ablation output/remote/locomo/multi_hop/<run_id> \
  --variants full,no_wiki_direct,wiki_only,flat_raw,snapshot_m1,snapshot_m2,source_supported_only \
  --budgets 4000,8000,16000,32000 \
  --rank-cutoffs 5,10,15,20,30,50
```

Offline cost-benefit and scalability analysis:

```bash
trajwiki analyze-cost-benefit output/remote/locomo/multi_hop/<run_id> \
  --baselines trajwiki_observed,full_context_proxy,no_wiki_direct,flat_raw,wiki_only \
  --price-config configs/pricing/gpt-4o-mini.example.json \
  --future-query-counts 1,2,5,10,20,50,100
```

The example pricing file intentionally contains `null` prices. Fill in the exact provider
prices for the run date before reporting dollar costs; token and latency tables are still
generated without prices.

Offline auditability and interpretability analysis:

```bash
trajwiki analyze-auditability output/remote/locomo/multi_hop/<run_id> \
  --baselines trajwiki_observed,full_context_proxy,no_wiki_direct,flat_raw,wiki_only \
  --audit-labels audit_labels.csv
```

`--audit-labels` is optional. Without labels, TrajWiki reports proxy or observable metrics
such as source-supported answer rate proxy, unsupported-answer risk, failure-localization
distribution, deprecated-claim leakage proxy, and audit packet cost. With labels, it also
reports accuracy/F1 for labeled conflict, obsolete-claim, failure-localization, and human
audit fields. `error_propagation_funnel.csv` additionally traces gold-source coverage from
storage through trajectory/page retrieval, final context, and answer support. Its
conditional correctness columns use the observed benchmark judge and are not human
accuracy estimates.

## Rebuttal 200-Query Extension Workflow

The current rebuttal experiment is a **post-hoc nested extension** of the completed
60-query pilot. It reuses the same immutable 282-query LoCoMo multi-hop memory build and
contains every pilot query; it is not a preregistered or independent replication. The
fixed `rebuttal_200_v1` sample uses seed 7 and selects:

```text
strict deep-history:   9 / 9
update-sensitive:    132 / 188
ordinary:             59 / 85
total:               200 / 282
```

The experiment generates 11 same-model answer configurations:

```text
full
direct_trajectory
latest_snapshot
hybrid_raw_rag
wiki_summaries
no_claim_state
no_source_constraint
full_context
naive_dense_rag
full_context_matched
hybrid_raw_rag_matched
```

`Full Context (32K)` and `Flat Raw Memory (32K)` are upper-bound context comparisons.
Their matched counterparts receive, query by query, no more complete-prompt tokens than
regenerated Full TrajWiki. Full Context truncates only at complete message boundaries;
the two Flat Raw variants share the exact same cached lexical+dense ranking. Naive Dense
RAG uses Qwen embeddings, 384-token chunks, 64-token overlap, and top-4 retrieval.

The 60-query parent experiment must be complete and use the same source hashes, models,
prompt versions, tokenizer, budgets, RAG settings, generation seed, and imported Mem0
answers. `--reuse-policy require` verifies every reusable generation and judge job,
including its prompt SHA, before any API call. Parent attachment is separate from the
core experiment hash.

```bash
trajwiki run-answer-ablation RUN_DIR \
  --sample-size 200 \
  --sampling-profile rebuttal_200_v1 \
  --sample-seed 7 \
  --variants full,direct_trajectory,latest_snapshot,hybrid_raw_rag,wiki_summaries,no_claim_state,no_source_constraint,full_context,naive_dense_rag,full_context_matched,hybrid_raw_rag_matched \
  --reuse-experiment PARENT_60_EXPERIMENT \
  --reuse-policy require \
  --baseline-answers PRIVATE_DIR/mem0_saved.normalized.jsonl \
  --max-total-tokens 32000 \
  --max-output-tokens 512 \
  --token-counter tiktoken \
  --require-exact-token-counter \
  --token-safety-margin 128 \
  --rag-chunk-size 384 \
  --rag-chunk-overlap 64 \
  --rag-top-k 4 \
  --backbone-provider-kind remote \
  --backbone-model gpt-4o-mini \
  --independent-judge-provider-kind remote \
  --independent-judge-model claude-sonnet-4-6 \
  --generation-max-concurrency 6 \
  --judge-max-concurrency 6 \
  --context-save-mode full \
  --max-provider-calls 5300 \
  --progress \
  --progress-interval-seconds 30 \
  --report-path PRIVATE_DIR/answer_ablation_200_report.json \
  --resume \
  --dry-run
```

On a fresh extension, dry-run must report:

```text
selected queries:                 200
variant contexts:               2,200
answers / independent judgments: 2,600 / 2,600
unavailable saved stages:          400
reused GPT / Claude jobs:           540 / 660
new GPT / Claude calls:            1,660 / 1,940
new provider calls:                3,600
successful logical provider work: 4,800
```

The 5,300 cap counts metered application-level provider attempts, including recoverable
failed attempts recorded by TrajWiki; retries hidden inside a provider SDK are not
separately observable.
After an interruption, the same command may report fewer remaining calls; `--resume`
accepts only complete jobs whose schema, experiment hash, prompt hash, variant, and query
all match. `progress.json` and stderr show stage progress, rate, errors, and ETA.
Machine-readable command output is written atomically through `--report-path`, so CUDA
startup output cannot corrupt JSON reports.

Results are separated into the controlled Full TrajWiki reference, component ablations,
32K upper bounds, token-matched baselines, strong RAG, observed pipeline, and historical
external baseline groups.
Analysis emits ordinary paired bootstrap intervals, population-prevalence-weighted
intervals, and dialogue-cluster bootstrap sensitivity intervals. These intervals
represent query-sampling uncertainty, not repeated model-generation variance.
`matched_budget_compliance.csv`, `completion_truncation_summary.csv`, and
`provider_fingerprint_summary.csv` make budget and provider drift auditable.

The following remain outside the evidence supported by this run:

- `Lifecycle State Hidden` is not a memory rebuild without ADD/REVISE/DEPRECATE.
- Missing saved answer stages do not support a true no-repair or no-retry ablation.
- Flat Raw Memory is not GraphRAG or RAPTOR.
- The experiment does not rerun the Qwen3-8B regression.
- It does not establish token break-even, medical safety, privacy deletion, or regulatory
  compliance.

The Kubernetes launcher requires a clean committed tree, builds and pushes one immutable
Linux/amd64 image digest, then runs a GPU dry-run Job, a GPU formal Job, and a CPU package
Job. A failed formal Job gets at most one resume Job. Each Job's rendered YAML, status,
logs, and `describe` output are archived before that exact Job is deleted; no
`kubectl delete jobs --all` operation is used.
Before building the image, the controller verifies that the 60-query parent is complete,
has zero integrity/validation errors, contains exactly 60 sampled queries and 1,200
provider-ledger rows, and shares the source `details.json` hash. The validated source-run,
SQLite, Mem0, parent, workflow-private, and bundle paths are injected explicitly into
every Pod rather than inferred again inside the container. Set
`TRAJWIKI_PARENT_60_EXPERIMENT` when the legacy private pointer file is unavailable.
With `TRAJWIKI_WORKFLOW_ID=<id>`, the default final archive is
`verification_bundle_<id>.tar.zst`.

```bash
TRAJWIKI_PARENT_60_EXPERIMENT=/absolute/path/to/parent_60 \
TRAJWIKI_CONFIRM_FORMAL=RUN \
scripts/launch_rebuttal_k8s.sh all
```

The CPU package Job runs answer analysis, sample-scoped offline ablation, cost,
auditability, failure, and ranking-robustness analyses, then prepares the blinded
24-case audit study. Packaging is refused unless both experiment validators report zero
errors and all required 200-query protocol artifacts are present. It creates one
mode-`0600` `.tar.zst` archive containing the source details/summary/SQLite, experiment
jobs and contexts, parent reuse proof, normalized Mem0 answers, offline reports, audit
files, code snapshot, dependency records, and Kubernetes logs. `.env`, credentials,
model caches, and temporary files are excluded.

```bash
trajwiki package-rebuttal-bundle EXPERIMENT_DIR \
  --output-path verification_bundle_<timestamp>.tar.zst \
  --workflow-logs PRIVATE_WORKFLOW_DIR

trajwiki validate-rebuttal-bundle \
  verification_bundle_<timestamp>.tar.zst
```

Every bundle uses relative archive paths and a SHA256 manifest. Original server manifests
are preserved unchanged even though they contain original absolute paths.

## Completed 60-Query Pilot

The following section documents the already completed pilot protocol for provenance and
comparison. The current Kubernetes scripts implement the 200-query extension above; do
not use the archived command below as the formal extension command.

The rebuttal workflow reuses one completed LOCOMO multi-hop memory build. It does not
rebuild snapshots, trajectories, or wiki pages for every answer variant. The completed
pilot used a fixed 60-query answer-level comparison. Mem0 was not called; only its saved
answers were independently judged.

First validate the source run:

```bash
trajwiki validate-run-artifacts RUN_DIR --json
```

Normalize the already saved Mem0 answers against all 282 source-run queries. Missing or
duplicate IDs, or mismatched sample/question/gold hashes, fail the import:

```bash
trajwiki import-baseline-answers \
  locomo_eval/outputs/runs/<saved_mem0_run>/predictions.jsonl \
  --method mem0_saved \
  --run-path RUN_DIR \
  --output-path PRIVATE_DIR/mem0_saved.normalized.jsonl
```

Set credentials for the answer generator and independent judge. Run the exact command
once with `--dry-run` first, inspect `call_plan.json` and `token_budget_audit.csv`, then
rerun the same command without `--dry-run`:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...

trajwiki run-answer-ablation RUN_DIR \
  --sample-size 60 \
  --sample-seed 7 \
  --variants full,direct_trajectory,latest_snapshot,hybrid_raw_rag,wiki_summaries,no_claim_state,no_source_constraint,full_context,naive_dense_rag \
  --max-total-tokens 32000 \
  --max-output-tokens 512 \
  --token-counter tiktoken \
  --require-exact-token-counter \
  --token-safety-margin 128 \
  --rag-chunk-size 384 \
  --rag-chunk-overlap 64 \
  --rag-top-k 4 \
  --baseline-answers PRIVATE_DIR/mem0_saved.normalized.jsonl \
  --backbone-provider-kind remote \
  --backbone-model gpt-4o-mini \
  --independent-judge-provider-kind remote \
  --independent-judge-model claude-sonnet-4-6 \
  --generation-max-concurrency 6 \
  --judge-max-concurrency 6 \
  --context-save-mode full \
  --max-provider-calls 1300 \
  --resume \
  --dry-run
```

The fixed protocol selects all 9 strict deep-history cases plus 26 update-sensitive and 25
ordinary cases, using seed 7. It fails instead of silently changing these preregistered
quotas. Dry-run makes zero LLM provider calls, although Naive Dense RAG still prepares
local Qwen embeddings and may populate its cache. With all nine generated variants,
`Observed Full Pipeline`, and one saved Mem0 method, the expected upper bound is 540
generation calls plus 660 independent-judge calls. `call_plan.json` is authoritative.

The pilot used the same exact key names and one-GPU Qwen embedding preparation as the
extension. Its earlier Kubernetes launcher has been superseded by the immutable,
archive-before-delete workflow documented above.

All nine pilot-generated methods used the same answer template, GPT-4o-mini, and the same 32K
prompt-plus-output envelope; the source-support ablation changes only its intended
grounding instruction. Full Context preserves raw-message boundaries when truncating.
When the 32K envelope is exceeded, it retains the latest complete messages and renders
the retained window in chronological order.
Naive Dense RAG uses Qwen embeddings, 384-token chunks, 64-token overlap, and top-4 dense
retrieval; chunk embeddings are cached by model, parameters, and source hash. Qwen
embedding preparation completes before remote LLM workers start.

The comparison boundaries matter:

- `Full TrajWiki`, `No Wiki Routing`, `Latest Snapshot Only`, `Budgeted Flat Raw Memory`,
  `Wiki Summaries Only`, `Lifecycle State Hidden`, `No Source-Support Constraint`,
  `Full Context`, and `Naive Dense RAG` form the shared-prompt comparison.
- `Observed Full Pipeline` is the original benchmark answer and is reported separately.
- `mem0_saved` is a historical external baseline, not a same-protocol causal ablation.
- `Lifecycle State Hidden` changes query-time rendering and filtering; it does not rebuild
  memory without ADD/REVISE/DEPRECATE.
- `Budgeted Flat Raw Memory` is the project's lexical+dense flat retriever; it is not
  GraphRAG or RAPTOR.
- `Wiki Summaries Only` exposes no raw source references, so source-support accuracy is
  unobservable.
- A no-repair or no-retry stage is judged only when the source run contains the actual
  saved stage answer. Missing stages are recorded in `unavailable_stage_rows.jsonl` and do
  not create judge calls.

The total budget is checked against the complete answer prompt using `tiktoken` with 512
output tokens and a 128-token safety margin reserved. Provider-reported prompt usage is
then reconciled in `token_budget_audit.csv`; formal runs fail if exact counting is
unavailable. Bare `local` Transformers generation and judging are serialized because the
model and Torch RNG state are shared. Remote and OpenAI-compatible providers retain the
requested concurrency.

Generation and judge jobs are written atomically per query. `--resume` accepts a job only
when its schema, experiment hash, prompt hash, variant, query ID, and complete status all
match. Failed provider attempts retain their call identity in the atomic error job and
are recovered into the call ledger before a retry. The provider-call cap therefore
includes recorded and recoverable prior attempts plus newly planned calls, while a
complete zero-call resume remains valid. Repeating a complete command produces zero new
calls. External baselines have a
separate hash/attachment manifest: adding or replacing one does not alter the core
generation hash or regenerate the nine shared-prompt methods.

Rebuild tables without provider calls, then validate completeness:

```bash
EXPERIMENT_DIR=RUN_DIR/rebuttal_experiments/answer_ablation_<config_hash>

trajwiki analyze-answer-ablation \
  "$EXPERIMENT_DIR"

trajwiki validate-run-artifacts \
  "$EXPERIMENT_DIR" --json
```

`answer_ablation_table.csv` reports both strict independent-judge accuracy (only
`CORRECT` counts as correct) and a partial-credit judge score (`CORRECT=1`,
`PARTIAL=0.5`, `INCORRECT=0`). `paired_statistics.csv` reports both definitions with
10,000 paired bootstrap draws. Prevalence-weighted stratum metrics are emitted only when
that metric is observed for every sampled query in every stratum; incomplete stage
captures remain `null`.

Do not pass the legacy Full Context or RAG result files through `--baseline-answers`.
Their prompts, models, or retrieval protocols differ from the controlled experiment, so
they are suitable only for strict ID/question/gold alignment and implementation
cross-checks. The regenerated `full_context` and `naive_dense_rag` variants are the
same-protocol results used in the main comparison. Only the already saved Mem0 answers
are attached as a clearly labeled historical external baseline.

Run the remaining 60-query offline reports with the same sampling manifest. Their outputs
are isolated under `EXPERIMENT_DIR/offline_analysis/`, so they do not overwrite full-run
282-query artifacts:

```bash
SCOPE="$EXPERIMENT_DIR/sampling_manifest.json"

trajwiki analyze-offline-ablation RUN_DIR \
  --sampling-manifest "$SCOPE"

trajwiki analyze-cost-benefit RUN_DIR \
  --sampling-manifest "$SCOPE"

trajwiki analyze-auditability RUN_DIR \
  --sampling-manifest "$SCOPE"

trajwiki analyze-failures \
  --run-path RUN_DIR \
  --sampling-manifest "$SCOPE"
```

Run the zero-call retrieval-weight sensitivity analysis:

```bash
trajwiki analyze-ranking-robustness RUN_DIR \
  --relative-perturbation 0.20 \
  --random-draws 100 \
  --seed 7 \
  --cutoff 15 \
  --page-cutoffs 5,10,15 \
  --trajectory-cutoffs 5,10,15,20,30 \
  --sampling-manifest "$SCOPE"
```

This perturbs reconstructable deterministic page-routing and trajectory-ranking
coefficients by plus or minus 20 percent and replays page/trajectory cutoffs over saved
full rankings. Lexical/mismatch residuals, RRF `k`, and the saved LLM set-rerank decision
remain fixed. Page perturbations report candidate-set reachability rather than pretending
that trajectories newly introduced by a different page set have saved query-time scores.
The trajectory-build parameter `m` cannot be replayed from the current database and must
use the paper's existing sweep.

Prepare the blinded 24-case human audit:

```bash
trajwiki prepare-audit-study RUN_DIR \
  --case-count 24 \
  --seed 7 \
  --answer-experiment "$EXPERIMENT_DIR"

trajwiki conduct-audit-study AUDIT_TASKS \
  --annotator-slot 1 \
  --annotator-id annotator-a \
  --output-path AUDIT_DIR/audit_results_1.jsonl \
  --max-exposure 2

trajwiki conduct-audit-study AUDIT_TASKS \
  --annotator-slot 2 \
  --annotator-id annotator-b \
  --output-path AUDIT_DIR/audit_results_2.jsonl \
  --max-exposure 2

trajwiki analyze-audit-study AUDIT_DIR \
  --audit-labels AUDIT_DIR/audit_labels_adjudicated.csv
```

The first exposure produces 48 counterbalanced primary judgments across two annotators;
the second exposure adds 48 crossover judgments for exploratory agreement. Give
annotators only `audit_tasks.jsonl`, never `audit_study_key.jsonl` or the adjudication
template. The researcher must adjudicate `true_source_supported`,
`true_failure_stage`, and supporting references before unblinding the condition key.
The private adjudication template includes the question, regenerated candidate answer,
gold answer, and gold references needed for that judgment, but never exposes the packet
condition.
External labels may also include `true_supporting_source_refs`,
`true_conflict_required`,
`true_conflict_handled`, `true_obsolete_required`, and `true_obsolete_handled`; without
them the report does not claim source-localization, failure-stage, conflict-detection, or
obsolete-handling accuracy.

Every context and audit artifact containing dialogue text is marked
`contains_sensitive_text=true`. Keep `rebuttal_experiments/`, normalized external
predictions, annotator data, and private keys access-controlled; define retention and
verified deletion procedures before sharing or deployment.

Inspect aggregate results:

```bash
PYTHONPATH=src python -m trajpatch report \
  --index-database-path output/trajpatch_index.sqlite \
  --dataset locomo \
  --subset multi_hop
```

Inspect one trajectory or snapshot:

```bash
PYTHONPATH=src python -m trajpatch inspect \
  --database-path output/remote/locomo/multi_hop/<run_id>/trajpatch.sqlite \
  --trajectory-id epi-conv-42-001

PYTHONPATH=src python -m trajpatch inspect \
  --database-path output/remote/locomo/multi_hop/<run_id>/trajpatch.sqlite \
  --snapshot-id epi-conv-42-001-s0001
```

Export trajectory artifacts from a database:

```bash
PYTHONPATH=src python -m trajpatch export \
  --database-path output/remote/locomo/multi_hop/<run_id>/trajpatch.sqlite \
  --output-dir output/export
```

Offline paper/diagnostic plots:

```bash
python plot/plot_hyperparameter_analysis.py \
  --run-path output/remote/locomo/multi_hop \
  --output-dir plot

python plot/plot_trajectory_drift_analysis.py \
  --run-path output/remote/locomo/multi_hop \
  --output-dir plot

python plot/plot_token_usage_analysis.py \
  --run-path output/remote/locomo/multi_hop \
  --output-dir plot

python plot/plot_runtime_usage_analysis.py \
  --run-path output/remote/locomo/multi_hop \
  --output-dir plot
```

## Development

Key directories:

```text
src/trajpatch/cli.py                  CLI entrypoint
src/trajpatch/config.py               RunConfig and option validation
src/trajpatch/pipeline/runner.py      End-to-end orchestration
src/trajpatch/pipeline/answering.py   LOCOMO answer generation and repair
src/trajpatch/memory/                 Memory extraction, trajectories, wiki, retrieval
src/trajpatch/providers/              Remote, local, OpenAI-compatible, mock providers
src/trajpatch/analysis/               Failure analysis and offline diagnostics
src/trajpatch/experiments/            Rebuttal ablations, statistics, validation, audit study
src/trajpatch/storage/                SQLite schema and repository methods
tests/unit/                           Focused unit tests
tests/integration/                    End-to-end and artifact tests
plot/                                 Offline plotting scripts
```

Run focused tests while editing:

```bash
PYTHONPATH=src pytest -q tests/unit/test_answering_prompts.py
PYTHONPATH=src pytest -q tests/unit/test_failure_attribution.py
PYTHONPATH=src pytest -q tests/unit/test_offline_ablation.py
PYTHONPATH=src pytest -q tests/unit/test_cost_benefit.py
PYTHONPATH=src pytest -q tests/unit/test_auditability.py
PYTHONPATH=src pytest -q tests/unit/test_experiment_statistics.py
PYTHONPATH=src pytest -q tests/unit/test_sample_scoped_analysis.py
PYTHONPATH=src pytest -q tests/unit/test_llm_text_parsers.py
PYTHONPATH=src pytest -q tests/unit/test_openai_compatible_provider.py
PYTHONPATH=src pytest -q tests/unit/test_vllm_server_manager.py
PYTHONPATH=src pytest -q tests/integration/test_run_index_and_report.py
PYTHONPATH=src pytest -q tests/integration/test_batch_modes.py
PYTHONPATH=src pytest -q tests/integration/test_offline_ablation_artifacts.py
PYTHONPATH=src pytest -q tests/integration/test_cost_benefit_artifacts.py
PYTHONPATH=src pytest -q tests/integration/test_auditability_artifacts.py
PYTHONPATH=src pytest -q tests/integration/test_answer_ablation_experiment.py
```

Run the full suite:

```bash
PYTHONPATH=src pytest -q
```

Optional static checks:

```bash
ruff check src tests
mypy src/trajpatch
```

Recommended reading order for new contributors:

1. `src/trajpatch/pipeline/runner.py`
2. `src/trajpatch/memory/orchestrator.py`
3. `src/trajpatch/memory/retrieval.py`
4. `src/trajpatch/memory/wiki.py`
5. `src/trajpatch/pipeline/answering.py`
6. `src/trajpatch/analysis/failure_attribution.py`
7. `src/trajpatch/analysis/offline_ablation.py`
8. `src/trajpatch/analysis/cost_benefit.py`
9. `src/trajpatch/analysis/auditability.py`
10. `src/trajpatch/experiments/answer_ablation.py`
11. `src/trajpatch/experiments/variant_contexts.py`
12. `src/trajpatch/storage/models.py`

## Design Notes

TrajWiki separates routing artifacts from answer evidence:

- Wiki pages and trajectory summaries help find the right memory region.
- Compacted snapshots provide structured answer context.
- Raw source messages provide exact evidence and temporal grounding.
- Failure analysis checks each boundary independently.

This separation makes the pipeline more diagnosable than a single dense retrieval step. It
also means memory construction, wiki routing, trajectory selection, source compaction,
answer validation, and judge behavior should be debugged as separate stages.
