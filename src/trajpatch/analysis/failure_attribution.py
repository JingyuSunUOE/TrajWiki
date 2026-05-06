"""Offline failure attribution and diffing for completed LOCOMO runs."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Iterable

from rich.console import Console
from rich.table import Table

from trajpatch.diagnostics.fallback_repair import (
    legacy_fallback_repair_diagnostics,
    summarize_fallback_repair_events,
)
from trajpatch.memory.facets import (
    build_sample_entity_lexicon,
    classify_query_shape_v1,
    extract_query_facets_v1,
    normalize_entity_key,
)
from trajpatch.memory.historical import sanitize_historical_item_terms
from trajpatch.memory.readability import count_fragment_lines
from trajpatch.memory.trajectory_summaries import is_internal_summary_keyword, sanitize_summary_keyword_values
from trajpatch.providers.metering import phase_for_task
from trajpatch.analysis.trajectory_drift import (
    build_trajectory_drift_diagnostics,
    build_trajectory_drift_rows,
    compact_query_drift_row,
    trajectory_drift_fields_for_gold_trajectories,
)
from trajpatch.utils.text import collapse_whitespace, extract_keywords


SOURCE_REF_RE = re.compile(r"\bD\d+:\d+\b")
OFFLINE_PARAMETER_CUTOFFS = (5, 10, 15, 20, 30, 50)
DIRECT_TRAJECTORY_DIAGNOSTIC_TOP_N = 50
DIRECT_TRAJECTORY_TERM_LIMIT = 12
DIRECT_TRAJECTORY_STRING_LIMIT = 80
FAILURE_REASON_ORDER = [
    "unknown_no_gold_refs",
    "memory_absent",
    "coarse_retrieval_miss",
    "fine_retrieval_miss",
    "grounding_miss",
    "answer_incomplete_after_grounding",
    "answer_overgenerated_after_grounding",
    "answer_selection_error_after_grounding",
    "unknown_other",
]
DIAGNOSTIC_FLAG_ORDER = [
    "facet_present_for_gold_ref",
    "exact_facet_missing",
    "gold_ref_present_but_claim_generalized",
    "gold_ref_present_but_no_supported_facet",
    "gold_refs_split_across_trajectories",
    "entity_linked_could_help",
    "entity_linked_added_relevant_snapshot",
    "gold_expanded_only_hit",
    "gold_entity_linked_hit",
    "gold_grounded_hit",
]
_ANSWER_ITEM_FAMILY_UMBRELLA_TERMS = {
    "activity": {"activity", "activities", "hobby", "hobbies"},
    "book": {"book", "books", "novel", "novels", "read", "reading"},
    "recipe": {"recipe", "recipes", "dish", "dishes", "cooked", "baked"},
    "instrument": {"instrument", "instruments", "music", "musical"},
    "symbol": {"symbol", "symbols"},
    "place": {"place", "places", "city", "cities", "trip", "trips", "travel"},
    "event": {"event", "events", "program", "speech", "parade", "festival"},
    "count": {"count", "number", "total"},
    "type": {"type", "types", "kind", "kinds"},
}
_DIRECT_TRAJECTORY_GENERIC_TERMS = {
    "about",
    "answer",
    "asked",
    "assistant",
    "context",
    "conversation",
    "details",
    "experience",
    "fact",
    "family",
    "feel",
    "friend",
    "good",
    "great",
    "help",
    "important",
    "information",
    "memory",
    "question",
    "said",
    "shared",
    "support",
    "talk",
    "thing",
    "things",
    "user",
}


def _dedupe_preserve(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    output: list[Any] = []
    for value in values:
        key = str(value).casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_json_artifact(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl_artifact(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _load_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        return json.loads(stripped)
    return default


def _resolve_run_dir(run_path: Path | str) -> Path:
    path = Path(run_path).expanduser().resolve()
    if path.is_file():
        raise ValueError(f"run_path must be a run directory or scope directory, got file: {path}")
    if (path / "details.json").exists() and (path / "trajpatch.sqlite").exists():
        return path
    candidates = [
        details_path.parent
        for details_path in path.rglob("details.json")
        if (details_path.parent / "trajpatch.sqlite").exists()
    ]
    if candidates:
        return max(candidates, key=lambda candidate: candidate.name)
    raise FileNotFoundError(
        f"Could not locate a completed run under {path}. Expected details.json and trajpatch.sqlite."
    )


def _resolve_incomplete_run_dir(run_path: Path | str) -> Path | None:
    path = Path(run_path).expanduser().resolve()
    if path.is_file():
        return None
    if (path / "run_failed.json").exists():
        return path
    candidates = [failure_path.parent for failure_path in path.rglob("run_failed.json")]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.name)


def load_incomplete_run_diagnostics(run_path: Path | str) -> dict[str, Any] | None:
    run_dir = _resolve_incomplete_run_dir(run_path)
    if run_dir is None:
        return None
    failure_payload = _load_json(run_dir / "run_failed.json")
    events_path = run_dir / "status" / "events.jsonl"
    recent_events: list[dict[str, Any]] = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines()[-30:]:
            if not line.strip():
                continue
            try:
                recent_events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {
        "report_type": "incomplete_run",
        "run_dir": str(run_dir),
        "run_failed": failure_payload,
        "recent_events": recent_events,
        "paths": {
            "run_failed": str(run_dir / "run_failed.json"),
            "events": str(events_path) if events_path.exists() else None,
            "database": str(run_dir / "trajpatch.sqlite") if (run_dir / "trajpatch.sqlite").exists() else None,
        },
    }


def _extract_source_refs(value: Any) -> list[str]:
    values: list[Any]
    if value is None:
        values = []
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]

    refs: list[str] = []
    seen: set[str] = set()
    for item in values:
        for match in SOURCE_REF_RE.finditer(str(item or "")):
            ref = match.group(0)
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def _gold_refs_from_query_metadata(query_metadata: dict[str, Any]) -> list[str]:
    for key in [
        "evidence_only_conversation",
        "gold_evidence_refs",
        "gold_evidence_raw",
        "gold_evidence",
    ]:
        refs = _extract_source_refs(query_metadata.get(key))
        if refs:
            return refs
    return []


def _normalize_text(value: Any) -> str:
    return collapse_whitespace(str(value or "")).casefold()


def _split_answer_items(value: Any) -> list[str]:
    text = collapse_whitespace(str(value or ""))
    if not text:
        return []
    if not any(separator in text for separator in [",", ";", " and "]):
        return [text]
    normalized = re.sub(r"\s+and\s+", ", ", text, flags=re.IGNORECASE)
    parts = [collapse_whitespace(part) for part in re.split(r"[,;]", normalized)]
    return [part for part in parts if part]


_ANSWER_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "for",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _answer_item_tokens(value: Any) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9+]+", str(value or "").casefold()))
    filtered = {token for token in tokens if token not in _ANSWER_TOKEN_STOPWORDS}
    return filtered or tokens


def _gold_answer_items(gold_answer: Any) -> list[str]:
    items = _split_answer_items(gold_answer)
    if len(items) == 1:
        return items
    return [item.strip(" \"'") for item in items if item.strip(" \"'")]


def _answer_covers_gold_item(answer_text: Any, gold_item: str) -> bool:
    normalized_answer = _normalize_text(answer_text)
    normalized_item = _normalize_text(gold_item)
    if not normalized_item:
        return False
    if normalized_item in normalized_answer:
        return True
    item_tokens = _answer_item_tokens(gold_item)
    if not item_tokens:
        return False
    answer_tokens = _answer_item_tokens(answer_text)
    return _safe_rate(len(item_tokens & answer_tokens), len(item_tokens)) >= 0.67


def _gold_all_items_covered_by_answer(gold_answer: Any, answer_text: Any) -> bool:
    items = _gold_answer_items(gold_answer)
    if not items:
        return False
    return all(_answer_covers_gold_item(answer_text, item) for item in items)


def _answer_has_extra_items(gold_answer: Any, answer_text: Any) -> bool:
    gold_items = _gold_answer_items(gold_answer)
    answer_items = _leniency_answer_items(answer_text)
    return bool(gold_items and answer_items and len(answer_items) > len(gold_items))


def _leniency_answer_items(answer_text: Any) -> list[str]:
    text = collapse_whitespace(str(answer_text or ""))
    if not text:
        return []
    numbered_parts = [
        collapse_whitespace(part).strip(" \t\n\r.,;:!?")
        for part in re.split(r"(?:^|\s)\d+[\.)]\s+", text)
        if collapse_whitespace(part).strip(" \t\n\r.,;:!?")
    ]
    if len(numbered_parts) > 1:
        return numbered_parts
    return _split_answer_items(text)


def _count_answer_is_evidence_limited_lower_bound(answer_text: Any) -> bool:
    normalized = _normalize_text(answer_text)
    if not normalized:
        return False
    evidence_limited_patterns = [
        r"\bretrieved evidence (?:confirms|supports|shows|found|contains)\b",
        r"\bin the retrieved evidence\b",
        r"\bbased on (?:the )?retrieved (?:context|evidence)\b",
        r"\bat least\b",
        r"\bconfirmed\b.*\b(?:retrieved|evidence|context)\b",
    ]
    has_evidence_limit = any(re.search(pattern, normalized) for pattern in evidence_limited_patterns)
    has_count = bool(
        re.search(
            r"\b(?:\d+|one|once|two|twice|three|four|five|six|seven|eight|nine|ten)\b",
            normalized,
        )
    )
    return has_evidence_limit and has_count


def _judge_leniency_fields(row: dict[str, Any]) -> dict[str, Any]:
    gold_answer = row.get("gold_answer")
    answer_text = row.get("answer_text")
    query = str(row.get("question") or "")
    query_shape = classify_query_shape_v1(query, {})
    gold_covered = _gold_all_items_covered_by_answer(gold_answer, answer_text)
    has_extra_items = _answer_has_extra_items(gold_answer, answer_text)
    count_lower_bound = bool(query_shape.get("count_like")) and _count_answer_is_evidence_limited_lower_bound(answer_text)
    verdict = str(row.get("judge_verdict", "")).lower()
    leniency_candidate = verdict in {"partial", "incorrect"} and (
        (gold_covered and has_extra_items) or count_lower_bound
    )
    return {
        "judge_leniency_candidate": leniency_candidate,
        "gold_all_items_covered_by_answer": gold_covered,
        "answer_has_extra_items": has_extra_items,
        "count_answer_is_evidence_limited_lower_bound": count_lower_bound,
    }


def _question_is_time_like(question: Any) -> bool:
    text = " ".join(str(question or "").casefold().split())
    return bool(
        re.search(
            r"^(?:when|what date|what month|what year|what day|which day)\b|\b(?:what date|what month|what year|what day|which day)\b",
            text,
        )
    )


def _relative_time_terms(text: Any) -> list[str]:
    normalized = " ".join(str(text or "").casefold().split())
    terms: list[str] = []
    seen: set[str] = set()
    patterns = [
        r"\btoday\b",
        r"\byesterday\b",
        r"\btomorrow\b",
        r"\b\d{1,2}\s+days?\s+ago\b",
        r"\blast week\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            term = match.group(0)
            if term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def _temporal_grounding_fields(
    *,
    question: Any,
    gold_refs: set[str],
    grounded_refs: set[str],
    retrieval_metadata: dict[str, Any],
    raw_messages_by_id: dict[str, dict[str, Any]],
    raw_message_ids_by_ref: dict[str, set[str]],
) -> dict[str, Any]:
    if not _question_is_time_like(question):
        return {
            "temporal_anchor_available_in_raw": None,
            "temporal_anchor_visible_in_prompt": None,
            "relative_time_terms_present": [],
            "temporal_anchor_missing_in_answer_context": None,
            "temporal_anchor_hint_count": retrieval_metadata.get("temporal_anchor_hint_count"),
        }
    relevant_refs = set(gold_refs) | set(grounded_refs)
    raw_dates: dict[str, str] = {}
    relative_terms: list[str] = []
    seen_terms: set[str] = set()
    for ref in sorted(relevant_refs):
        for message_id in raw_message_ids_by_ref.get(ref, set()):
            message = raw_messages_by_id.get(message_id)
            if not message:
                continue
            occurred_at = collapse_whitespace(str(message.get("occurred_at") or ""))
            if occurred_at:
                raw_dates[ref] = occurred_at
            for term in _relative_time_terms(message.get("content")):
                if term not in seen_terms:
                    seen_terms.add(term)
                    relative_terms.append(term)
    visible_anchors = dict(retrieval_metadata.get("source_message_time_anchors") or {})
    prompt_visible = bool(set(visible_anchors) & relevant_refs) or int(
        retrieval_metadata.get("temporal_anchor_hint_count") or 0
    ) > 0
    raw_available = bool(raw_dates)
    grounded_gold = bool(gold_refs and gold_refs <= grounded_refs)
    return {
        "temporal_anchor_available_in_raw": raw_available,
        "temporal_anchor_visible_in_prompt": prompt_visible,
        "relative_time_terms_present": relative_terms,
        "temporal_anchor_missing_in_answer_context": bool(raw_available and grounded_gold and not prompt_visible),
        "temporal_anchor_hint_count": retrieval_metadata.get("temporal_anchor_hint_count"),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator or 1)


def _safe_mean(values: list[int | float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_median(values: list[int | float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _judge_fields_from_row(row: dict[str, Any]) -> dict[str, Any]:
    verdict = str(row.get("judge_verdict", "")).lower()
    judge_metadata = dict(row.get("metadata", {}).get("judge_metadata", {}) or {})
    judge_mode = str(judge_metadata.get("judge_mode") or "").strip() or "legacy_unknown"
    if verdict == "judge_error" and judge_mode == "legacy_unknown":
        judge_mode = "judge_error"
    judge_execution_failed = bool(judge_metadata.get("judge_execution_failed")) or verdict == "judge_error"
    return {
        "judge_mode": judge_mode,
        "judge_structured_requested": judge_metadata.get("structured_requested"),
        "judge_structured_success": judge_metadata.get("structured_success"),
        "judge_structured_fallback_used": judge_metadata.get("structured_fallback_used"),
        "judge_structured_fallback_category": judge_metadata.get("structured_fallback_category"),
        "judge_structured_fallback_reason": judge_metadata.get("structured_fallback_reason"),
        "judge_execution_failed": judge_execution_failed,
        "judge_error_type": judge_metadata.get("judge_error_type"),
        "judge_error_message": judge_metadata.get("judge_error_message"),
        "raw_judge_verdict": judge_metadata.get("raw_judge_verdict"),
        "raw_judge_score": judge_metadata.get("raw_judge_score"),
        "judge_semantic_override_used": bool(judge_metadata.get("judge_semantic_override_used")),
        "judge_semantic_override_reason": judge_metadata.get("judge_semantic_override_reason"),
    }


def _semantic_f1_fields_from_row(row: dict[str, Any]) -> dict[str, Any]:
    semantic_metadata = dict(row.get("metadata", {}).get("semantic_metrics", {}) or {})
    exact_overlap = list(semantic_metadata.get("f1_exact_overlap_items") or [])
    soft_overlap = list(
        semantic_metadata.get("f1_soft_overlap_items")
        or semantic_metadata.get("f1_overlap_items")
        or []
    )
    unmatched_reference = list(semantic_metadata.get("f1_unmatched_reference_items") or [])
    judge_metadata = dict(row.get("metadata", {}).get("judge_metadata", {}) or {})
    raw_verdict = str(judge_metadata.get("raw_judge_verdict") or row.get("judge_verdict") or "").lower()
    return {
        "canonical_exact_overlap_items": exact_overlap,
        "canonical_soft_overlap_items": soft_overlap,
        "canonical_unmatched_reference_items": unmatched_reference,
        "canonical_soft_match_pairs": list(semantic_metadata.get("f1_soft_match_pairs") or []),
        "semantic_alias_match_pairs": [
            pair
            for pair in list(semantic_metadata.get("f1_soft_match_pairs") or [])
            if str(dict(pair).get("reason") or "") in {"alias", "phrase_alias"}
        ],
        "judge_semantic_override_used": bool(judge_metadata.get("judge_semantic_override_used")),
        "judge_semantic_override_reason": judge_metadata.get("judge_semantic_override_reason"),
        "judge_over_strict_suspected": bool(
            raw_verdict in {"partial", "incorrect"}
            and not unmatched_reference
            and float(dict(row.get("metrics") or {}).get("F1") or 0.0) >= 0.999999
        ),
    }


def _evaluation_filter_fields_from_row(row: dict[str, Any]) -> dict[str, Any]:
    evaluation_filter = dict(dict(row.get("metadata") or {}).get("evaluation_filter") or {})
    if not evaluation_filter:
        return {
            "text_only_filter_available": False,
            "text_only_eligible": None,
            "excluded_from_text_only": None,
            "visual_dependency_type": None,
            "text_only_exclusion_reason": None,
            "text_only_missing_gold_items": [],
            "text_only_gold_evidence_image_refs": [],
        }
    return {
        "text_only_filter_available": True,
        "text_only_eligible": evaluation_filter.get("text_only_eligible"),
        "excluded_from_text_only": evaluation_filter.get("excluded_from_text_only"),
        "visual_dependency_type": evaluation_filter.get("visual_dependency_type"),
        "text_only_exclusion_reason": evaluation_filter.get("exclusion_reason"),
        "text_only_missing_gold_items": list(evaluation_filter.get("gold_items_missing_from_text_input") or []),
        "text_only_gold_evidence_image_refs": list(evaluation_filter.get("gold_evidence_image_refs") or []),
    }


def _build_judge_diagnostics(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(sample_rows)
    evaluable_count = 0
    execution_failed_count = 0
    structured_requested_count = 0
    structured_success_count = 0
    text_fallback_count = 0
    text_only_count = 0
    incorrect_text_fallback_count = 0
    partial_count = 0
    judge_semantic_override_count = 0
    partial_to_correct_override_count = 0
    incorrect_to_correct_override_count = 0
    fallback_categories: Counter[str] = Counter()
    judge_acc_values: list[float] = []

    for row in sample_rows:
        verdict = str(row.get("judge_verdict", "")).lower()
        judge_fields = _judge_fields_from_row(row)
        if judge_fields["judge_semantic_override_used"]:
            judge_semantic_override_count += 1
            raw_verdict = str(judge_fields.get("raw_judge_verdict") or "").lower()
            if raw_verdict == "partial":
                partial_to_correct_override_count += 1
            elif raw_verdict == "incorrect":
                incorrect_to_correct_override_count += 1
        if verdict in {"correct", "partial", "incorrect"}:
            evaluable_count += 1
            value = dict(row.get("metrics", {}) or {}).get("judge_acc")
            if value is not None:
                judge_acc_values.append(float(value))
        if verdict == "partial":
            partial_count += 1
        if judge_fields["judge_execution_failed"]:
            execution_failed_count += 1
        if judge_fields["judge_structured_requested"] is True:
            structured_requested_count += 1
        if judge_fields["judge_structured_success"] is True:
            structured_success_count += 1
        if judge_fields["judge_mode"] == "text_fallback":
            text_fallback_count += 1
        if judge_fields["judge_mode"] == "text_only":
            text_only_count += 1
        if verdict == "incorrect" and judge_fields["judge_mode"] == "text_fallback":
            incorrect_text_fallback_count += 1
        fallback_category = judge_fields.get("judge_structured_fallback_category")
        if fallback_category:
            fallback_categories[str(fallback_category)] += 1

    incorrect_count = sum(1 for row in sample_rows if str(row.get("judge_verdict", "")).lower() == "incorrect")
    non_correct_count = sum(
        1 for row in sample_rows if str(row.get("judge_verdict", "")).lower() in {"partial", "incorrect"}
    )
    return {
        "judge_evaluable_count": evaluable_count,
        "judge_execution_failed_count": execution_failed_count,
        "judge_execution_failed_rate_over_all": _safe_rate(execution_failed_count, total),
        "partial_count": partial_count,
        "partial_rate_over_all": _safe_rate(partial_count, total),
        "partial_rate_over_non_correct": _safe_rate(partial_count, non_correct_count),
        "mean_partial_credit_judge_acc": _safe_mean(judge_acc_values),
        "structured_requested_rate_over_all": _safe_rate(structured_requested_count, total),
        "structured_success_rate_over_all": _safe_rate(structured_success_count, total),
        "text_fallback_rate_over_all": _safe_rate(text_fallback_count, total),
        "text_only_rate_over_all": _safe_rate(text_only_count, total),
        "incorrect_queries_judged_via_text_fallback_count": incorrect_text_fallback_count,
        "incorrect_queries_judged_via_text_fallback_rate_over_incorrect": _safe_rate(
            incorrect_text_fallback_count, incorrect_count
        ),
        "judge_semantic_override_count": judge_semantic_override_count,
        "judge_semantic_override_rate": _safe_rate(judge_semantic_override_count, total),
        "partial_to_correct_override_count": partial_to_correct_override_count,
        "incorrect_to_correct_override_count": incorrect_to_correct_override_count,
        "structured_fallback_category_counts": dict(fallback_categories),
    }


def _reflection_fields_from_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata", {}) or {})
    answer_metadata = dict(metadata.get("answer_metadata", {}) or {})
    def value_for(key: str, default: Any = None) -> Any:
        if key in row:
            return row.get(key)
        if key in answer_metadata:
            return answer_metadata.get(key)
        return default

    return {
        "retrieval_reflection_retry_used": bool(
            row.get("retrieval_reflection_retry_used")
            or answer_metadata.get("retrieval_reflection_retry_used")
        ),
        "retrieval_reflection_used": bool(
            row.get("retrieval_reflection_used") or answer_metadata.get("retrieval_reflection_used")
        ),
        "retrieval_reflection_stage": str(
            row.get("retrieval_reflection_stage")
            or answer_metadata.get("retrieval_reflection_stage")
            or "none"
        ),
        "reflection_rewritten_query": row.get("reflection_rewritten_query")
        or answer_metadata.get("reflection_rewritten_query")
        or "",
        "reflection_must_find_terms": list(
            row.get("reflection_must_find_terms")
            or answer_metadata.get("reflection_must_find_terms")
            or []
        ),
        "reflection_candidate_page_slugs": list(
            row.get("reflection_candidate_page_slugs")
            or answer_metadata.get("reflection_candidate_page_slugs")
            or []
        ),
        "initial_answer_abstained": value_for("initial_answer_abstained"),
        "initial_retrieval_bundle_weak": value_for("initial_retrieval_bundle_weak"),
        "reflection_reroute_event_id": value_for("reflection_reroute_event_id"),
        "post_reflection_raw_rescue_used": bool(value_for("post_reflection_raw_rescue_used", False)),
        "post_reflection_raw_rescue_reason": value_for("post_reflection_raw_rescue_reason"),
        "post_reflection_raw_rescue_event_id": value_for("post_reflection_raw_rescue_event_id"),
        "final_reflection_retrieval_event_id": value_for("final_reflection_retrieval_event_id"),
        "raw_rescue_attempted": value_for("raw_rescue_attempted"),
        "raw_rescue_trigger_reasons": list(value_for("raw_rescue_trigger_reasons", []) or []),
        "raw_rescue_skipped_reason": value_for("raw_rescue_skipped_reason"),
        "reflection_required_terms": list(value_for("reflection_required_terms", []) or []),
        "reflection_covered_terms": list(value_for("reflection_covered_terms", []) or []),
        "reflection_uncovered_terms": list(value_for("reflection_uncovered_terms", []) or []),
        "reflection_term_coverage_rate": value_for("reflection_term_coverage_rate"),
        "reflection_semantic_evidence_weak": bool(value_for("reflection_semantic_evidence_weak", False)),
        "raw_rescue_used": bool(row.get("raw_rescue_used") or answer_metadata.get("raw_rescue_used")),
        "raw_rescue_hit_count": int(
            row.get("raw_rescue_hit_count") or answer_metadata.get("raw_rescue_hit_count") or 0
        ),
        "raw_rescue_source_refs": list(
            row.get("raw_rescue_source_refs") or answer_metadata.get("raw_rescue_source_refs") or []
        ),
        "raw_rescue_compensated_memory_gap": bool(
            row.get("raw_rescue_compensated_memory_gap")
            or answer_metadata.get("raw_rescue_compensated_memory_gap")
        ),
        "reflection_answer_changed": bool(
            row.get("reflection_answer_changed") or answer_metadata.get("reflection_answer_changed")
        ),
    }


def _answer_synthesis_fields_from_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata", {}) or {})
    answer_metadata = dict(metadata.get("answer_metadata", {}) or {})
    counted_events = list(answer_metadata.get("answer_synthesis_counted_events") or [])
    excluded_events = list(answer_metadata.get("answer_synthesis_excluded_events") or [])
    return {
        "answer_synthesis_used": bool(answer_metadata.get("answer_synthesis_used")),
        "answer_synthesis_mode": answer_metadata.get("answer_synthesis_mode"),
        "answer_synthesis_can_answer": answer_metadata.get("answer_synthesis_can_answer"),
        "answer_synthesis_answer_type": answer_metadata.get("answer_synthesis_answer_type"),
        "answer_synthesis_supporting_refs": list(
            answer_metadata.get("answer_synthesis_supporting_refs") or []
        ),
        "answer_synthesis_counted_events": counted_events,
        "answer_synthesis_excluded_events": excluded_events,
        "answer_synthesis_counted_event_count": len(counted_events),
        "answer_synthesis_excluded_event_count": len(excluded_events),
        "answer_synthesis_error": answer_metadata.get("answer_synthesis_error"),
        "answer_freeform_used": bool(answer_metadata.get("answer_freeform_used")),
        "answer_freeform_rationale": answer_metadata.get("answer_freeform_rationale"),
        "answer_freeform_parse_format": answer_metadata.get("answer_freeform_parse_format"),
        "answer_type_verification_used": bool(answer_metadata.get("answer_type_verification_used")),
        "answer_type_verification_success": answer_metadata.get("answer_type_verification_success"),
        "answer_expected_type": answer_metadata.get("answer_expected_type"),
        "answer_observed_type": answer_metadata.get("answer_observed_type"),
        "answer_type_match": answer_metadata.get("answer_type_match"),
        "answer_type_issue": answer_metadata.get("answer_type_issue"),
        "answer_type_repair_instruction": answer_metadata.get("answer_type_repair_instruction"),
        "answer_type_verification_error": answer_metadata.get("answer_type_verification_error"),
        "invalid_supporting_refs": list(answer_metadata.get("invalid_supporting_refs") or []),
        "answer_synthesis_family_validation_used": bool(
            answer_metadata.get("answer_synthesis_family_validation_used")
        ),
        "answer_synthesis_invalid_family_refs": list(
            answer_metadata.get("answer_synthesis_invalid_family_refs") or []
        ),
        "answer_synthesis_question_type_mismatch": bool(
            answer_metadata.get("answer_synthesis_question_type_mismatch")
        ),
        "answer_synthesis_question_type_mismatch_reason": answer_metadata.get(
            "answer_synthesis_question_type_mismatch_reason"
        ),
        "answer_synthesis_normalized_answer_type": answer_metadata.get(
            "answer_synthesis_normalized_answer_type"
        ),
        "answer_synthesis_expected_answer_family": answer_metadata.get(
            "answer_synthesis_expected_answer_family"
        ),
        "answer_synthesis_repair_reason": answer_metadata.get("answer_synthesis_repair_reason"),
        "answer_synthesis_count_style_text_mismatch": bool(
            answer_metadata.get("answer_synthesis_count_style_text_mismatch")
        ),
        "answer_synthesis_expected_type_text_valid": answer_metadata.get(
            "answer_synthesis_expected_type_text_valid"
        ),
        "answer_synthesis_expected_type_text_rejection_reason": answer_metadata.get(
            "answer_synthesis_expected_type_text_rejection_reason"
        ),
        "answer_synthesis_type_recovery_rejected_reason": answer_metadata.get(
            "answer_synthesis_type_recovery_rejected_reason"
        ),
        "answer_synthesis_typed_retry_used": bool(
            answer_metadata.get("answer_synthesis_typed_retry_used")
        ),
        "answer_synthesis_typed_retry_expected_type": answer_metadata.get(
            "answer_synthesis_typed_retry_expected_type"
        ),
        "answer_synthesis_typed_retry_success": answer_metadata.get(
            "answer_synthesis_typed_retry_success"
        ),
        "answer_synthesis_initial_answer_type": answer_metadata.get(
            "answer_synthesis_initial_answer_type"
        ),
        "answer_synthesis_initial_final_answer_preview": answer_metadata.get(
            "answer_synthesis_initial_final_answer_preview"
        ),
        "answer_synthesis_typed_retry_error": answer_metadata.get(
            "answer_synthesis_typed_retry_error"
        ),
        "answer_synthesis_typed_retry_text_json_normalized": bool(
            answer_metadata.get("answer_synthesis_typed_retry_text_json_normalized")
        ),
        "answer_synthesis_typed_retry_text_json_missing_fields": list(
            answer_metadata.get("answer_synthesis_typed_retry_text_json_missing_fields") or []
        ),
        "answer_synthesis_typed_retry_final_policy": answer_metadata.get(
            "answer_synthesis_typed_retry_final_policy"
        ),
        "answer_synthesis_type_mismatch_recovered": bool(
            answer_metadata.get("answer_synthesis_type_mismatch_recovered")
        ),
        "answer_synthesis_type_mismatch_recovery_reason": answer_metadata.get(
            "answer_synthesis_type_mismatch_recovery_reason"
        ),
        "answer_synthesis_recovered_answer_type": answer_metadata.get(
            "answer_synthesis_recovered_answer_type"
        ),
        "answer_synthesis_recovered_from_answer_type": answer_metadata.get(
            "answer_synthesis_recovered_from_answer_type"
        ),
        "answer_synthesis_internal_abstain_reason_suppressed": bool(
            answer_metadata.get("answer_synthesis_internal_abstain_reason_suppressed")
        ),
        "answer_synthesis_safe_abstain_used": bool(
            answer_metadata.get("answer_synthesis_safe_abstain_used")
        ),
        "answer_overgeneric_item_detected": bool(answer_metadata.get("answer_overgeneric_item_detected")),
        "answer_overgeneric_items": list(answer_metadata.get("answer_overgeneric_items") or []),
        "answer_specific_replacement_candidates": list(
            answer_metadata.get("answer_specific_replacement_candidates") or []
        ),
        "answer_scope_mismatched_extra_items": list(
            answer_metadata.get("answer_scope_mismatched_extra_items") or []
        ),
        "answer_specific_item_repair_used": bool(answer_metadata.get("answer_specific_item_repair_used")),
        "answer_supported_list_items": list(answer_metadata.get("answer_supported_list_items") or []),
        "answer_supported_list_item_refs": dict(answer_metadata.get("answer_supported_list_item_refs") or {}),
        "answer_supported_required_items": list(answer_metadata.get("answer_supported_required_items") or []),
        "answer_supported_required_item_refs": dict(
            answer_metadata.get("answer_supported_required_item_refs") or {}
        ),
        "answer_list_scope_kind": answer_metadata.get("answer_list_scope_kind"),
        "answer_required_item_candidates": list(answer_metadata.get("answer_required_item_candidates") or []),
        "answer_optional_surface_values": list(answer_metadata.get("answer_optional_surface_values") or []),
        "answer_scope_rejected_items": list(answer_metadata.get("answer_scope_rejected_items") or []),
        "answer_scope_rejection_reasons": list(answer_metadata.get("answer_scope_rejection_reasons") or []),
        "answer_missing_required_items": list(answer_metadata.get("answer_missing_required_items") or []),
        "answer_repair_scope_filtered_items": list(
            answer_metadata.get("answer_repair_scope_filtered_items") or []
        ),
        "event_canonical_alias_items": list(answer_metadata.get("event_canonical_alias_items") or []),
        "answer_required_items_missing_before_repair": list(
            answer_metadata.get("answer_required_items_missing_before_repair") or []
        ),
        "answer_repair_removed_scope_mismatched_items": list(
            answer_metadata.get("answer_repair_removed_scope_mismatched_items") or []
        ),
        "answer_repair_missing_required_items_after_repair": list(
            answer_metadata.get("answer_repair_missing_required_items_after_repair") or []
        ),
        "answer_missing_supported_list_items": list(
            answer_metadata.get("answer_missing_supported_list_items") or []
        ),
        "answer_abstain_despite_supported_items": bool(
            answer_metadata.get("answer_abstain_despite_supported_items")
        ),
        "answer_list_coverage_repair_used": bool(
            answer_metadata.get("answer_list_coverage_repair_used")
        ),
        "answer_list_coverage_repair_success": bool(
            answer_metadata.get("answer_list_coverage_repair_success")
        ),
        "bridge_finalization_used": bool(answer_metadata.get("bridge_finalization_used")),
        "bridge_finalization_alias": answer_metadata.get("bridge_finalization_alias"),
        "bridge_finalization_target": answer_metadata.get("bridge_finalization_target"),
        "bridge_finalization_source_refs": list(answer_metadata.get("bridge_finalization_source_refs") or []),
        "bridge_finalization_action": answer_metadata.get("bridge_finalization_action"),
        "bridge_finalization_conflicted": bool(answer_metadata.get("bridge_finalization_conflicted")),
        "bridge_finalization_failed_reason": answer_metadata.get("bridge_finalization_failed_reason"),
        "answer_bridge_repair_used": bool(answer_metadata.get("answer_bridge_repair_used")),
        "answer_bridge_repair_success": bool(answer_metadata.get("answer_bridge_repair_success")),
        "answer_repair_dropped_supported_items": list(
            answer_metadata.get("answer_repair_dropped_supported_items") or []
        ),
        "answer_repair_post_validation_failed": bool(
            answer_metadata.get("answer_repair_post_validation_failed")
        ),
        "answer_repair_post_validation_action": answer_metadata.get("answer_repair_post_validation_action"),
        "answer_repair_preserved_initial_answer": bool(
            answer_metadata.get("answer_repair_preserved_initial_answer")
        ),
        "answer_repair_arbitration_triggered": bool(
            answer_metadata.get("answer_repair_arbitration_triggered")
        ),
        "answer_repair_arbitration_trigger_reason": answer_metadata.get(
            "answer_repair_arbitration_trigger_reason"
        ),
        "answer_repair_arbitration_used": bool(
            answer_metadata.get("answer_repair_arbitration_used")
        ),
        "answer_repair_arbitration_success": answer_metadata.get(
            "answer_repair_arbitration_success"
        ),
        "answer_repair_arbitration_decision": answer_metadata.get(
            "answer_repair_arbitration_decision"
        ),
        "answer_repair_arbitration_violation": answer_metadata.get(
            "answer_repair_arbitration_violation"
        ),
        "answer_repair_arbitration_confidence": answer_metadata.get(
            "answer_repair_arbitration_confidence"
        ),
        "answer_repair_arbitration_reason": answer_metadata.get("answer_repair_arbitration_reason"),
        "answer_repair_arbitration_action": answer_metadata.get("answer_repair_arbitration_action"),
        "answer_repair_arbitration_kept_initial": bool(
            answer_metadata.get("answer_repair_arbitration_kept_initial")
        ),
        "answer_repair_arbitration_used_safe_abstain": bool(
            answer_metadata.get("answer_repair_arbitration_used_safe_abstain")
        ),
        "answer_repair_initial_answer_preview": answer_metadata.get(
            "answer_repair_initial_answer_preview"
        ),
        "answer_repair_repaired_answer_preview": answer_metadata.get(
            "answer_repair_repaired_answer_preview"
        ),
        "source_family_validation_alias_hits": list(
            answer_metadata.get("source_family_validation_alias_hits") or []
        ),
        "source_family_validation_support_text_used": list(
            answer_metadata.get("source_family_validation_support_text_used") or []
        ),
        "count_validation_ref_acceptance_reasons": list(
            answer_metadata.get("count_validation_ref_acceptance_reasons") or []
        ),
        "count_validation_ref_rejection_reasons": list(
            answer_metadata.get("count_validation_ref_rejection_reasons") or []
        ),
        "count_validation_llm_candidate_events": list(
            answer_metadata.get("count_validation_llm_candidate_events") or []
        ),
        "count_validation_llm_trigger_reasons": list(
            answer_metadata.get("count_validation_llm_trigger_reasons") or []
        ),
        "count_validation_llm_skipped_reason": answer_metadata.get("count_validation_llm_skipped_reason"),
        "count_validation_llm_used": bool(answer_metadata.get("count_validation_llm_used")),
        "count_validation_llm_success": bool(answer_metadata.get("count_validation_llm_success")),
        "count_validation_llm_scope": answer_metadata.get("count_validation_llm_scope"),
        "count_validation_llm_confidence": answer_metadata.get("count_validation_llm_confidence"),
        "count_validation_llm_decisions": list(answer_metadata.get("count_validation_llm_decisions") or []),
        "count_validation_llm_error": answer_metadata.get("count_validation_llm_error"),
        "count_validation_llm_changed_count": int(
            answer_metadata.get("count_validation_llm_changed_count") or 0
        ),
        "count_validation_source_derived_candidate_events": list(
            answer_metadata.get("count_validation_source_derived_candidate_events") or []
        ),
        "count_validation_source_derived_candidate_count": int(
            answer_metadata.get("count_validation_source_derived_candidate_count") or 0
        ),
        "count_validation_source_derived_candidate_refs": list(
            answer_metadata.get("count_validation_source_derived_candidate_refs") or []
        ),
        "count_validation_source_derived_trigger_terms": list(
            answer_metadata.get("count_validation_source_derived_trigger_terms") or []
        ),
        "count_validation_source_derived_action_hits": dict(
            answer_metadata.get("count_validation_source_derived_action_hits") or {}
        ),
        "count_validation_source_derived_object_hits": dict(
            answer_metadata.get("count_validation_source_derived_object_hits") or {}
        ),
        "count_validation_source_derived_passive_rejected_refs": list(
            answer_metadata.get("count_validation_source_derived_passive_rejected_refs") or []
        ),
        "count_validation_source_derived_pronoun_caption_refs": list(
            answer_metadata.get("count_validation_source_derived_pronoun_caption_refs") or []
        ),
        "answer_temporal_alignment_checked": bool(answer_metadata.get("answer_temporal_alignment_checked")),
        "answer_temporal_candidate_dates": list(answer_metadata.get("answer_temporal_candidate_dates") or []),
        "answer_temporal_selected_source_ref": answer_metadata.get("answer_temporal_selected_source_ref"),
        "answer_temporal_selected_date": answer_metadata.get("answer_temporal_selected_date"),
        "answer_temporal_selected_answer_text": answer_metadata.get("answer_temporal_selected_answer_text"),
        "answer_temporal_selected_resolution_kind": answer_metadata.get(
            "answer_temporal_selected_resolution_kind"
        ),
        "answer_temporal_selected_resolution_granularity": answer_metadata.get(
            "answer_temporal_selected_resolution_granularity"
        ),
        "answer_temporal_selected_relative_term": answer_metadata.get(
            "answer_temporal_selected_relative_term"
        ),
        "answer_temporal_selected_confidence": answer_metadata.get("answer_temporal_selected_confidence"),
        "answer_temporal_candidate_score": answer_metadata.get("answer_temporal_candidate_score"),
        "answer_temporal_candidate_match_terms": list(
            answer_metadata.get("answer_temporal_candidate_match_terms") or []
        ),
        "answer_temporal_relevant_candidate_count": int(
            answer_metadata.get("answer_temporal_relevant_candidate_count") or 0
        ),
        "answer_temporal_low_confidence_candidate_count": int(
            answer_metadata.get("answer_temporal_low_confidence_candidate_count") or 0
        ),
        "answer_temporal_candidates_suppressed_count": int(
            answer_metadata.get("answer_temporal_candidates_suppressed_count") or 0
        ),
        "answer_temporal_no_query_relevant_candidate": bool(
            answer_metadata.get("answer_temporal_no_query_relevant_candidate")
        ),
        "answer_temporal_alignment_valid": answer_metadata.get("answer_temporal_alignment_valid"),
        "answer_temporal_alignment_rejection_reason": answer_metadata.get(
            "answer_temporal_alignment_rejection_reason"
        ),
        "answer_temporal_repair_used": bool(answer_metadata.get("answer_temporal_repair_used")),
        "answer_temporal_repair_success": bool(answer_metadata.get("answer_temporal_repair_success")),
        "answer_temporal_repair_action": answer_metadata.get("answer_temporal_repair_action"),
        "answer_list_coverage_skipped": answer_metadata.get("answer_list_coverage_skipped"),
        "answer_list_coverage_skip_reason": answer_metadata.get("answer_list_coverage_skip_reason"),
        "answer_list_repair_blocked_by_expected_type": bool(
            answer_metadata.get("answer_list_repair_blocked_by_expected_type")
        ),
        "answer_count_naturalized_lower_bound_reason": answer_metadata.get(
            "answer_count_naturalized_lower_bound_reason"
        ),
        "answer_count_lower_bound_reasons": list(answer_metadata.get("answer_count_lower_bound_reasons") or []),
        "answer_count_lower_bound_excluded_event_count": int(
            answer_metadata.get("answer_count_lower_bound_excluded_event_count") or 0
        ),
        "count_validation_excluded_events": list(answer_metadata.get("count_validation_excluded_events") or []),
        "count_validation_positive_event_signal": list(
            answer_metadata.get("count_validation_positive_event_signal") or []
        ),
        "count_validation_rejection_signal": list(
            answer_metadata.get("count_validation_rejection_signal") or []
        ),
        "bridge_facts_used": list(answer_metadata.get("bridge_facts_used") or []),
        "bridge_facts_missing": list(answer_metadata.get("bridge_facts_missing") or []),
        "bridge_facts_conflicted": list(answer_metadata.get("bridge_facts_conflicted") or []),
        "bridge_facts_ignored": list(answer_metadata.get("bridge_facts_ignored") or []),
    }


def _broad_entity_fields_from_retrieval_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "broad_entity_page_selected": bool(metadata.get("broad_entity_page_selected")),
        "broad_entity_page_ids": list(metadata.get("broad_entity_page_ids") or []),
        "broad_entity_profile_page_ids": list(metadata.get("broad_entity_profile_page_ids") or []),
        "broad_entity_profile_suppressed": bool(metadata.get("broad_entity_profile_suppressed")),
        "fine_grained_entity_page_ids": list(metadata.get("fine_grained_entity_page_ids") or []),
        "entity_facet_page_candidate_count": metadata.get("entity_facet_page_candidate_count"),
        "selected_page_universe_size_before_broad_cap": metadata.get(
            "selected_page_universe_size_before_broad_cap"
        ),
        "selected_page_universe_size_after_broad_cap": metadata.get(
            "selected_page_universe_size_after_broad_cap"
        ),
        "broad_entity_candidate_cap_used": bool(metadata.get("broad_entity_candidate_cap_used")),
        "broad_entity_added_trajectory_ids": list(metadata.get("broad_entity_added_trajectory_ids") or []),
        "broad_entity_profile_added_trajectory_ids": list(
            metadata.get("broad_entity_profile_added_trajectory_ids") or []
        ),
    }


def _page_granularity_fields_from_retrieval_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    selected_rows = [
        dict(row)
        for row in list(metadata.get("selected_page_rows_compact") or metadata.get("selected_page_rows") or [])
        if isinstance(row, dict)
    ]
    if selected_rows:
        trajectory_counts = [
            int(row.get("trajectory_count") or len(list(row.get("trajectory_ids") or [])) or 0)
            for row in selected_rows
        ]
        histogram = {
            str(count): frequency
            for count, frequency in sorted(Counter(trajectory_counts).items())
        }
        singleton_count = sum(1 for count in trajectory_counts if count == 1)
        medium_count = sum(1 for count in trajectory_counts if 3 <= count <= 6)
        low_quality_singleton_count = sum(
            1
            for row, count in zip(selected_rows, trajectory_counts)
            if count == 1
            and (
                str(row.get("singleton_policy") or "") == "merge_required_low_quality"
                or float(row.get("low_quality_singleton_penalty") or 0.0) < 0.0
            )
        )
        allowed_specific_singleton_count = sum(
            1
            for row, count in zip(selected_rows, trajectory_counts)
            if count == 1 and str(row.get("singleton_policy") or "") == "allowed_isolated_specific"
        )
        low_quality_penalty_applied = sum(
            1 for row in selected_rows if float(row.get("low_quality_singleton_penalty") or 0.0) < 0.0
        )
        metadata_available = True
    else:
        histogram = dict(metadata.get("selected_page_trajectory_count_histogram") or {})
        singleton_count = metadata.get("selected_singleton_page_count")
        medium_count = metadata.get("selected_medium_granularity_page_count")
        low_quality_singleton_count = metadata.get("selected_low_quality_singleton_page_count")
        allowed_specific_singleton_count = metadata.get("selected_allowed_specific_singleton_page_count")
        low_quality_penalty_applied = metadata.get("low_quality_singleton_penalty_applied")
        metadata_available = bool(
            histogram
            or singleton_count is not None
            or medium_count is not None
            or metadata.get("singleton_page_penalty_applied") is not None
            or metadata.get("low_quality_singleton_penalty_applied") is not None
            or metadata.get("medium_page_bonus_applied") is not None
        )
    return {
        "page_granularity_metadata_available": metadata_available,
        "page_granularity_metadata_source": "retrieval_metadata" if metadata_available else None,
        "page_granularity_diagnostic_mode": (
            str(metadata.get("page_granularity_diagnostic_mode") or "retrieval_metadata")
            if metadata_available
            else "missing"
        ),
        "selected_singleton_page_count": singleton_count,
        "selected_medium_granularity_page_count": medium_count,
        "selected_allowed_specific_singleton_page_count": allowed_specific_singleton_count,
        "selected_low_quality_singleton_page_count": low_quality_singleton_count,
        "selected_page_trajectory_count_histogram": histogram,
        "singleton_page_penalty_applied": metadata.get("singleton_page_penalty_applied"),
        "low_quality_singleton_penalty_applied": low_quality_penalty_applied,
        "medium_page_bonus_applied": metadata.get("medium_page_bonus_applied"),
        "selected_singleton_page_ids": list(metadata.get("selected_singleton_page_ids") or []),
        "selected_medium_page_ids": list(metadata.get("selected_medium_page_ids") or []),
    }


def _page_granularity_fields_from_page_ids(
    page_ids: list[str],
    page_to_trajectory_ids: dict[str, list[str]],
) -> dict[str, Any]:
    counts = [
        len(list(dict.fromkeys(page_to_trajectory_ids.get(page_id, []))))
        for page_id in page_ids
        if page_id in page_to_trajectory_ids
    ]
    if not counts:
        return {
            "page_granularity_metadata_available": False,
            "page_granularity_metadata_source": "missing",
            "page_granularity_diagnostic_mode": "missing",
            "selected_singleton_page_count": None,
            "selected_medium_granularity_page_count": None,
            "selected_allowed_specific_singleton_page_count": None,
            "selected_low_quality_singleton_page_count": None,
            "selected_page_trajectory_count_histogram": {},
            "singleton_page_penalty_applied": None,
            "low_quality_singleton_penalty_applied": None,
            "medium_page_bonus_applied": None,
            "selected_singleton_page_ids": [],
            "selected_medium_page_ids": [],
        }
    histogram = {str(count): frequency for count, frequency in sorted(Counter(counts).items())}
    singleton_ids = [
        page_id
        for page_id in page_ids
        if len(list(dict.fromkeys(page_to_trajectory_ids.get(page_id, [])))) == 1
    ]
    medium_ids = [
        page_id
        for page_id in page_ids
        if 3 <= len(list(dict.fromkeys(page_to_trajectory_ids.get(page_id, [])))) <= 6
    ]
    return {
        "page_granularity_metadata_available": True,
        "page_granularity_metadata_source": "sqlite_wiki_pages_fallback",
        "page_granularity_diagnostic_mode": "sqlite_wiki_pages_fallback",
        "selected_singleton_page_count": len(singleton_ids),
        "selected_medium_granularity_page_count": len(medium_ids),
        "selected_allowed_specific_singleton_page_count": None,
        "selected_low_quality_singleton_page_count": None,
        "selected_page_trajectory_count_histogram": histogram,
        "singleton_page_penalty_applied": None,
        "low_quality_singleton_penalty_applied": None,
        "medium_page_bonus_applied": None,
        "selected_singleton_page_ids": singleton_ids,
        "selected_medium_page_ids": medium_ids,
    }


def _routing_text_marker_fields_for_pages(
    page_ids: list[str],
    page_metadata_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected_metadata = [page_metadata_by_id.get(page_id, {}) for page_id in page_ids]
    marker_counts = [
        int(metadata.get("routing_text_internal_marker_count") or 0)
        for metadata in selected_metadata
        if metadata
    ]
    cleaned_values = [
        metadata.get("routing_text_cleaned")
        for metadata in selected_metadata
        if "routing_text_cleaned" in metadata
    ]
    leaking_page_ids = [
        page_id
        for page_id in page_ids
        if int(page_metadata_by_id.get(page_id, {}).get("routing_text_internal_marker_count") or 0) > 0
    ]
    return {
        "routing_text_cleaned": (
            all(bool(value) for value in cleaned_values)
            if cleaned_values
            else None
        ),
        "routing_text_internal_marker_count": sum(marker_counts) if marker_counts else None,
        "routing_text_internal_marker_page_ids": leaking_page_ids,
    }


def _wiki_fragmentation_fields_from_retrieval_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    available = metadata.get("wiki_fragmentation_diagnostics_available")
    return {
        "wiki_fragmentation_diagnostics_available": available,
        "wiki_fragmentation_metadata_source": "retrieval_metadata" if available else None,
        "wiki_non_index_page_count": metadata.get("wiki_non_index_page_count"),
        "wiki_singleton_non_index_page_count": metadata.get("wiki_singleton_non_index_page_count"),
        "wiki_singleton_non_index_page_rate": metadata.get("wiki_singleton_non_index_page_rate"),
        "wiki_mean_trajectories_per_non_index_page": metadata.get("wiki_mean_trajectories_per_non_index_page"),
        "wiki_singleton_rate_by_seed_type": dict(metadata.get("wiki_singleton_rate_by_seed_type") or {}),
        "wiki_post_plan_rescue_singleton_count": metadata.get("wiki_post_plan_rescue_singleton_count"),
        "wiki_entity_facet_singleton_count": metadata.get("wiki_entity_facet_singleton_count"),
        "wiki_allowed_specific_singleton_count": metadata.get("wiki_allowed_specific_singleton_count"),
        "wiki_low_quality_singleton_count": metadata.get("wiki_low_quality_singleton_count"),
        "wiki_low_quality_singleton_merged_count": metadata.get("wiki_low_quality_singleton_merged_count"),
        "wiki_overwide_non_index_page_count": metadata.get("wiki_overwide_non_index_page_count"),
        "wiki_overwide_page_split_count": metadata.get("wiki_overwide_page_split_count"),
        "wiki_max_non_index_trajectory_count": metadata.get("wiki_max_non_index_trajectory_count"),
    }


def _wiki_fragmentation_fields_from_page_tables(
    sample_id: str,
    sample_to_pages: dict[str, set[str]],
    page_to_trajectory_ids: dict[str, list[str]],
    page_metadata_by_id: dict[str, dict[str, Any]],
    page_types: dict[str, str],
) -> dict[str, Any]:
    page_ids = sorted(sample_to_pages.get(sample_id, set()))
    non_index_ids = [page_id for page_id in page_ids if page_types.get(page_id) != "index"]
    if not non_index_ids:
        return {
            "wiki_fragmentation_diagnostics_available": False,
            "wiki_fragmentation_metadata_source": "missing",
            "wiki_non_index_page_count": None,
            "wiki_singleton_non_index_page_count": None,
            "wiki_singleton_non_index_page_rate": None,
            "wiki_mean_trajectories_per_non_index_page": None,
            "wiki_singleton_rate_by_seed_type": {},
            "wiki_post_plan_rescue_singleton_count": None,
            "wiki_entity_facet_singleton_count": None,
            "wiki_allowed_specific_singleton_count": None,
            "wiki_low_quality_singleton_count": None,
            "wiki_low_quality_singleton_merged_count": None,
            "wiki_overwide_non_index_page_count": None,
            "wiki_overwide_page_split_count": None,
            "wiki_max_non_index_trajectory_count": None,
        }
    counts = [len(list(dict.fromkeys(page_to_trajectory_ids.get(page_id, [])))) for page_id in non_index_ids]
    seed_counts: Counter[str] = Counter()
    seed_singletons: Counter[str] = Counter()
    for page_id, count in zip(non_index_ids, counts):
        seed_type = str(page_metadata_by_id.get(page_id, {}).get("seed_type") or "unknown")
        seed_counts[seed_type] += 1
        if count == 1:
            seed_singletons[seed_type] += 1
    singleton_count = sum(1 for count in counts if count == 1)
    allowed_specific_singletons = sum(
        1
        for page_id, count in zip(non_index_ids, counts)
        if count == 1
        and page_metadata_by_id.get(page_id, {}).get("wiki_singleton_policy") == "allowed_isolated_specific"
    )
    low_quality_singletons = sum(
        1
        for page_id, count in zip(non_index_ids, counts)
        if count == 1
        and page_metadata_by_id.get(page_id, {}).get("wiki_singleton_policy") == "merge_required_low_quality"
    )
    low_quality_merged = sum(
        1
        for page_id in non_index_ids
        if page_metadata_by_id.get(page_id, {}).get("wiki_singleton_low_quality_merged") is True
    )
    overwide_non_index_pages = sum(1 for count in counts if count > 6)
    overwide_split_pages = sum(
        1
        for page_id in non_index_ids
        if page_metadata_by_id.get(page_id, {}).get("wiki_overwide_page_split") is True
    )
    return {
        "wiki_fragmentation_diagnostics_available": True,
        "wiki_fragmentation_metadata_source": "sqlite_wiki_pages_fallback",
        "wiki_non_index_page_count": len(non_index_ids),
        "wiki_singleton_non_index_page_count": singleton_count,
        "wiki_singleton_non_index_page_rate": _safe_rate(singleton_count, len(non_index_ids)),
        "wiki_mean_trajectories_per_non_index_page": _safe_mean(counts),
        "wiki_singleton_rate_by_seed_type": {
            seed_type: _safe_rate(seed_singletons[seed_type], count)
            for seed_type, count in sorted(seed_counts.items())
        },
        "wiki_post_plan_rescue_singleton_count": seed_singletons.get("post_plan_rescue", 0),
        "wiki_entity_facet_singleton_count": seed_singletons.get("entity_facet", 0),
        "wiki_allowed_specific_singleton_count": allowed_specific_singletons,
        "wiki_low_quality_singleton_count": low_quality_singletons,
        "wiki_low_quality_singleton_merged_count": low_quality_merged,
        "wiki_overwide_non_index_page_count": overwide_non_index_pages,
        "wiki_overwide_page_split_count": overwide_split_pages,
        "wiki_max_non_index_trajectory_count": max(counts) if counts else None,
    }


def _page_family_fields_from_retrieval_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    selected_rows = [
        dict(row)
        for row in list(metadata.get("selected_page_rows_compact") or metadata.get("selected_page_rows") or [])
        if isinstance(row, dict)
    ]
    if not selected_rows:
        selected_rows = [
            dict(row)
            for row in list(metadata.get("page_ranked_rows") or [])[:10]
            if isinstance(row, dict)
        ]
    scores = [float(row.get("page_family_match_score") or 0.0) for row in selected_rows]
    overlap_terms = _dedupe_preserve(
        str(term)
        for row in selected_rows
        for term in list(row.get("page_query_object_overlap_terms") or [])
        if str(term).strip()
    )
    object_terms = _dedupe_preserve(
        str(term)
        for row in selected_rows
        for term in list(row.get("page_query_object_terms") or [])
        if str(term).strip()
    )
    return {
        "page_family_routing_available": bool(selected_rows and any("page_family_match_score" in row for row in selected_rows)),
        "page_family_match_score_max": max(scores) if scores else None,
        "page_family_query_object_terms": object_terms,
        "page_family_query_object_overlap_terms": overlap_terms,
        "page_family_mismatch_penalty_count": sum(
            1 for row in selected_rows if float(row.get("page_family_mismatch_penalty") or 0.0) > 0.0
        ) if selected_rows else None,
    }


def _family_ranking_fields_from_metadata(
    *,
    retrieval_metadata: dict[str, Any],
    gold_trajectory_ids: list[str],
    top_k_trajectory_ids: list[str],
    selection_pool_trajectory_ids: list[str],
) -> dict[str, Any]:
    score_rows = [
        dict(item)
        for item in list(retrieval_metadata.get("trajectory_family_match_scores") or [])
        if isinstance(item, dict)
    ]
    score_by_id = {
        str(item.get("trajectory_id") or ""): float(item.get("score") or 0.0)
        for item in score_rows
        if str(item.get("trajectory_id") or "")
    }
    strong_ids = {
        trajectory_id for trajectory_id, score in score_by_id.items()
        if score >= 0.55
    }
    gold_set = set(gold_trajectory_ids)
    pool_set = set(selection_pool_trajectory_ids)
    topk_set = set(top_k_trajectory_ids)
    selected_family_matches = [
        dict(item)
        for item in list(retrieval_metadata.get("trajectory_selected_family_matches") or [])
        if isinstance(item, dict)
    ]
    source_event_rows = [
        dict(item)
        for item in list(retrieval_metadata.get("trajectory_source_event_match_scores") or [])
        if isinstance(item, dict)
    ]
    selected_source_event_matches = [
        dict(item)
        for item in list(retrieval_metadata.get("trajectory_selected_source_event_matches") or [])
        if isinstance(item, dict)
    ]
    source_event_score_by_id = {
        str(item.get("trajectory_id") or ""): float(item.get("score") or 0.0)
        for item in source_event_rows
        if str(item.get("trajectory_id") or "")
    }
    source_event_strong_ids = {
        trajectory_id for trajectory_id, score in source_event_score_by_id.items()
        if score >= 0.70
    }
    return {
        "trajectory_family_match_scores": score_rows,
        "trajectory_family_mismatch_penalties": list(
            retrieval_metadata.get("trajectory_family_mismatch_penalties") or []
        ),
        "trajectory_selected_family_matches": selected_family_matches,
        "trajectory_query_object_terms": sorted(
            {
                str(term)
                for item in score_rows
                for term in list(item.get("query_object_terms") or [])
                if str(term).strip()
            }
        ),
        "trajectory_family_query_overlap_terms": sorted(
            {
                str(term)
                for item in score_rows
                for term in list(item.get("query_object_overlap_terms") or [])
                if str(term).strip()
            }
        ),
        "gold_family_matched_in_selection_pool_count": len(gold_set & pool_set & strong_ids),
        "gold_family_matched_in_top_k_count": len(gold_set & topk_set & strong_ids),
        "trajectory_selected_family_match_count": len(selected_family_matches),
        "trajectory_source_event_match_scores": source_event_rows,
        "trajectory_selected_source_event_matches": selected_source_event_matches,
        "trajectory_source_event_query_profile": dict(
            retrieval_metadata.get("trajectory_source_event_query_profile") or {}
        ),
        "trajectory_source_event_match_miss_count": retrieval_metadata.get(
            "trajectory_source_event_match_miss_count"
        ),
        "gold_source_event_matched_in_selection_pool_count": len(
            gold_set & pool_set & source_event_strong_ids
        ),
        "gold_source_event_matched_in_top_k_count": len(gold_set & topk_set & source_event_strong_ids),
        "trajectory_selected_source_event_match_count": len(selected_source_event_matches),
        "trajectory_source_event_matched_terms": sorted(
            {
                str(term)
                for item in source_event_rows
                for term in list(item.get("matched_terms") or [])
                if str(term).strip()
            }
        ),
        "trajectory_source_event_matched_refs": sorted(
            {
                str(ref)
                for item in source_event_rows
                for ref in list(item.get("matched_refs") or [])
                if str(ref).strip()
            }
        ),
    }


def _llm_usage_fields_from_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata", {}) or {})
    usage = dict(metadata.get("llm_usage") or {})
    by_task = dict(metadata.get("llm_usage_by_task") or {})
    fallback_counts = dict(metadata.get("llm_fallback_counts") or {})
    repair_counts = dict(metadata.get("llm_repair_counts") or {})
    fallback_repair_summary = dict(metadata.get("fallback_repair_summary") or {})
    fallback_repair_quality_flags = dict(metadata.get("fallback_repair_quality_flags") or {})
    if not usage and not by_task:
        return {
            "llm_usage": {},
            "llm_usage_by_task": {},
            "llm_fallback_counts": {},
            "llm_repair_counts": {},
            "fallback_repair_summary": fallback_repair_summary,
            "fallback_repair_quality_flags": fallback_repair_quality_flags,
            "llm_answer_calls": None,
            "llm_judge_calls": None,
            "llm_semantic_calls": None,
            "llm_repair_calls": None,
            "llm_fallback_count": None,
            "llm_repair_count": None,
        }

    def calls_for_phase(phase: str) -> float:
        total = 0.0
        for task, task_usage in by_task.items():
            if phase_for_task(str(task)) == phase:
                total += float(dict(task_usage or {}).get("provider_call_count", 0.0) or 0.0)
        return total

    return {
        "llm_usage": usage,
        "llm_usage_by_task": by_task,
        "llm_fallback_counts": fallback_counts,
        "llm_repair_counts": repair_counts,
        "fallback_repair_summary": fallback_repair_summary,
        "fallback_repair_quality_flags": fallback_repair_quality_flags,
        "llm_answer_calls": calls_for_phase("answer"),
        "llm_judge_calls": calls_for_phase("judge"),
        "llm_semantic_calls": calls_for_phase("semantic_metrics"),
        "llm_repair_calls": calls_for_phase("repair"),
        "llm_fallback_count": sum(float(value or 0) for value in fallback_counts.values()),
        "llm_repair_count": sum(float(value or 0) for value in repair_counts.values()),
    }


def _build_answer_synthesis_diagnostics(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(sample_rows)
    fields_by_row = [_answer_synthesis_fields_from_row(row) for row in sample_rows]
    used_rows = [fields for fields in fields_by_row if fields["answer_synthesis_used"]]
    can_answer_rows = [fields for fields in fields_by_row if fields["answer_synthesis_can_answer"] is True]
    non_correct_can_answer_count = sum(
        1
        for row, fields in zip(sample_rows, fields_by_row, strict=False)
        if fields["answer_synthesis_can_answer"] is True
        and str(row.get("judge_verdict", "")).lower() in {"partial", "incorrect"}
    )
    count_rows = [
        fields
        for fields in fields_by_row
        if str(fields.get("answer_synthesis_answer_type") or "").casefold() == "count"
    ]
    excluded_reason_counts: Counter[str] = Counter()
    invalid_supporting_ref_count = 0
    invalid_family_ref_count = 0
    question_type_mismatch_count = 0
    count_validation_excluded_count = 0
    bridge_fact_used_count = 0
    typed_retry_used_count = 0
    typed_retry_success_count = 0
    freeform_used_count = 0
    type_verification_used_count = 0
    type_verification_success_count = 0
    answer_type_mismatch_count = 0
    type_mismatch_recovered_count = 0
    typed_retry_text_json_normalized_count = 0
    internal_abstain_reason_suppressed_count = 0
    expected_type_text_rejected_count = 0
    source_family_alias_hit_count = 0
    count_ref_validation_rejected_count = 0
    count_validation_llm_used_count = 0
    count_validation_llm_success_count = 0
    count_validation_llm_changed_count = 0
    source_derived_candidate_count = 0
    source_derived_candidate_accepted_count = 0
    source_derived_pronoun_caption_candidate_count = 0
    source_derived_passive_rejected_count = 0
    temporal_alignment_rejected_count = 0
    temporal_repair_used_count = 0
    temporal_low_confidence_candidate_count = 0
    temporal_no_query_relevant_candidate_count = 0
    temporal_candidate_selection_rejected_count = 0
    date_question_list_repair_blocked_count = 0
    list_scope_rejected_item_count = 0
    missing_required_item_count = 0
    overgeneric_answer_count = 0
    scope_mismatched_extra_item_count = 0
    specific_item_repair_used_count = 0
    missing_supported_list_item_count = 0
    abstain_despite_supported_items_count = 0
    list_coverage_repair_used_count = 0
    repair_dropped_supported_item_count = 0
    repair_discarded_by_post_validation_count = 0
    repair_arbitration_triggered_count = 0
    repair_arbitration_used_count = 0
    repair_arbitration_keep_initial_count = 0
    repair_arbitration_use_repair_count = 0
    repair_arbitration_safe_abstain_count = 0
    repair_arbitration_failed_count = 0
    answer_repair_removed_scope_mismatched_item_count = 0
    answer_repair_missing_required_items_after_repair_count = 0
    event_canonical_alias_item_count = 0
    list_required_item_count = 0
    list_required_item_after_repair_hit_count = 0
    bridge_finalization_used_count = 0
    bridge_finalization_conflict_count = 0
    answer_bridge_repair_used_count = 0
    preference_query_shape_count = sum(
        1 for row in sample_rows
        if str(dict(row.get("query_shape") or {}).get("item_family") or "").casefold() == "preference"
    )
    for fields in fields_by_row:
        invalid_supporting_ref_count += len(list(fields.get("invalid_supporting_refs") or []))
        invalid_family_ref_count += len(list(fields.get("answer_synthesis_invalid_family_refs") or []))
        question_type_mismatch_count += int(bool(fields.get("answer_synthesis_question_type_mismatch")))
        count_validation_excluded_count += len(list(fields.get("count_validation_excluded_events") or []))
        bridge_fact_used_count += len(list(fields.get("bridge_facts_used") or []))
        typed_retry_used_count += int(bool(fields.get("answer_synthesis_typed_retry_used")))
        typed_retry_success_count += int(bool(fields.get("answer_synthesis_typed_retry_success")))
        freeform_used_count += int(bool(fields.get("answer_freeform_used")))
        type_verification_used_count += int(bool(fields.get("answer_type_verification_used")))
        type_verification_success_count += int(bool(fields.get("answer_type_verification_success")))
        answer_type_mismatch_count += int(fields.get("answer_type_match") is False)
        type_mismatch_recovered_count += int(bool(fields.get("answer_synthesis_type_mismatch_recovered")))
        typed_retry_text_json_normalized_count += int(
            bool(fields.get("answer_synthesis_typed_retry_text_json_normalized"))
        )
        internal_abstain_reason_suppressed_count += int(
            bool(fields.get("answer_synthesis_internal_abstain_reason_suppressed"))
        )
        expected_type_text_rejected_count += int(
            fields.get("answer_synthesis_expected_type_text_valid") is False
        )
        for row in list(fields.get("source_family_validation_alias_hits") or []):
            source_family_alias_hit_count += len(list(dict(row).get("alias_hits") or []))
        count_ref_validation_rejected_count += len(
            list(fields.get("count_validation_ref_rejection_reasons") or [])
        )
        count_validation_llm_used_count += int(bool(fields.get("count_validation_llm_used")))
        count_validation_llm_success_count += int(bool(fields.get("count_validation_llm_success")))
        count_validation_llm_changed_count += int(fields.get("count_validation_llm_changed_count") or 0)
        source_derived_candidate_count += int(fields.get("count_validation_source_derived_candidate_count") or 0)
        source_derived_pronoun_caption_candidate_count += len(
            list(fields.get("count_validation_source_derived_pronoun_caption_refs") or [])
        )
        source_derived_passive_rejected_count += len(
            list(fields.get("count_validation_source_derived_passive_rejected_refs") or [])
        )
        temporal_alignment_rejected_count += int(
            fields.get("answer_temporal_alignment_valid") is False
        )
        temporal_repair_used_count += int(bool(fields.get("answer_temporal_repair_used")))
        temporal_low_confidence_candidate_count += int(
            fields.get("answer_temporal_low_confidence_candidate_count") or 0
        )
        temporal_no_query_relevant_candidate_count += int(
            bool(fields.get("answer_temporal_no_query_relevant_candidate"))
        )
        temporal_candidate_selection_rejected_count += int(
            bool(fields.get("answer_temporal_alignment_checked"))
            and fields.get("answer_temporal_selected_source_ref") is None
            and fields.get("answer_temporal_alignment_valid") is False
        )
        date_question_list_repair_blocked_count += int(
            bool(fields.get("answer_list_repair_blocked_by_expected_type"))
        )
        list_scope_rejected_item_count += len(list(fields.get("answer_scope_rejected_items") or []))
        missing_required_item_count += len(list(fields.get("answer_missing_required_items") or []))
        source_derived_event_ids = {
            str(dict(event).get("event_id") or "")
            for event in list(fields.get("count_validation_source_derived_candidate_events") or [])
            if isinstance(event, dict)
        }
        for decision in list(fields.get("count_validation_llm_decisions") or []):
            decision_dict = dict(decision)
            if (
                str(decision_dict.get("event_id") or "") in source_derived_event_ids
                and str(decision_dict.get("decision") or "").upper() == "COUNT"
            ):
                source_derived_candidate_accepted_count += 1
        overgeneric_answer_count += int(bool(fields.get("answer_overgeneric_item_detected")))
        scope_mismatched_extra_item_count += int(
            bool(list(fields.get("answer_scope_mismatched_extra_items") or []))
        )
        specific_item_repair_used_count += int(bool(fields.get("answer_specific_item_repair_used")))
        missing_supported_list_item_count += int(
            bool(list(fields.get("answer_missing_supported_list_items") or []))
        )
        abstain_despite_supported_items_count += int(
            bool(fields.get("answer_abstain_despite_supported_items"))
        )
        list_coverage_repair_used_count += int(bool(fields.get("answer_list_coverage_repair_used")))
        event_canonical_alias_item_count += len(list(fields.get("event_canonical_alias_items") or []))
        answer_repair_removed_scope_mismatched_item_count += len(
            list(fields.get("answer_repair_removed_scope_mismatched_items") or [])
        )
        answer_repair_missing_required_items_after_repair_count += len(
            list(fields.get("answer_repair_missing_required_items_after_repair") or [])
        )
        dropped_supported_items = list(fields.get("answer_repair_dropped_supported_items") or [])
        repair_dropped_supported_item_count += len(dropped_supported_items)
        repair_discarded_by_post_validation_count += int(
            bool(fields.get("answer_repair_post_validation_failed"))
        )
        repair_arbitration_triggered_count += int(
            bool(fields.get("answer_repair_arbitration_triggered"))
        )
        repair_arbitration_used_count += int(bool(fields.get("answer_repair_arbitration_used")))
        repair_arbitration_failed_count += int(
            bool(fields.get("answer_repair_arbitration_triggered"))
            and fields.get("answer_repair_arbitration_success") is False
        )
        arbitration_action = str(fields.get("answer_repair_arbitration_action") or "")
        repair_arbitration_keep_initial_count += int(arbitration_action == "keep_initial")
        repair_arbitration_use_repair_count += int(arbitration_action == "use_repair")
        repair_arbitration_safe_abstain_count += int(arbitration_action == "safe_abstain")
        required_items = list(fields.get("answer_supported_required_items") or [])
        list_required_item_count += len(required_items)
        list_required_item_after_repair_hit_count += max(0, len(required_items) - len(dropped_supported_items))
        bridge_finalization_used_count += int(bool(fields.get("bridge_finalization_used")))
        bridge_finalization_conflict_count += int(bool(fields.get("bridge_finalization_conflicted")))
        answer_bridge_repair_used_count += int(bool(fields.get("answer_bridge_repair_used")))
        for event in list(fields.get("answer_synthesis_excluded_events") or []):
            reason = str(dict(event).get("reason") or "unknown").strip() or "unknown"
            excluded_reason_counts[reason] += 1
    return {
        "used_count": len(used_rows),
        "used_rate_over_all": _safe_rate(len(used_rows), total),
        "structured_count": sum(1 for fields in fields_by_row if fields["answer_synthesis_mode"] == "structured"),
        "freeform_v2_count": sum(1 for fields in fields_by_row if fields["answer_synthesis_mode"] == "freeform_v2"),
        "freeform_used_count": freeform_used_count,
        "text_json_count": sum(1 for fields in fields_by_row if fields["answer_synthesis_mode"] == "text_json"),
        "legacy_fallback_count": sum(
            1 for fields in fields_by_row if fields["answer_synthesis_mode"] == "legacy_fallback"
        ),
        "can_answer_count": len(can_answer_rows),
        "can_answer_rate_over_all": _safe_rate(len(can_answer_rows), total),
        "can_answer_but_non_correct_count": non_correct_can_answer_count,
        "can_answer_but_non_correct_rate_over_can_answer": _safe_rate(
            non_correct_can_answer_count,
            len(can_answer_rows),
        ),
        "count_answer_type_count": len(count_rows),
        "mean_counted_events_over_count_questions": _safe_mean(
            [int(fields.get("answer_synthesis_counted_event_count") or 0) for fields in count_rows]
        ),
        "mean_excluded_events_over_count_questions": _safe_mean(
            [int(fields.get("answer_synthesis_excluded_event_count") or 0) for fields in count_rows]
        ),
        "excluded_event_reason_counts": dict(excluded_reason_counts),
        "invalid_supporting_ref_count": invalid_supporting_ref_count,
        "invalid_family_ref_count": invalid_family_ref_count,
        "question_type_mismatch_count": question_type_mismatch_count,
        "typed_retry_used_count": typed_retry_used_count,
        "typed_retry_success_count": typed_retry_success_count,
        "typed_retry_success_rate": _safe_rate(typed_retry_success_count, typed_retry_used_count),
        "answer_type_verification_used_count": type_verification_used_count,
        "answer_type_verification_success_count": type_verification_success_count,
        "answer_type_verification_success_rate": _safe_rate(
            type_verification_success_count,
            type_verification_used_count,
        ),
        "answer_type_mismatch_count": answer_type_mismatch_count,
        "type_mismatch_recovered_count": type_mismatch_recovered_count,
        "typed_retry_text_json_normalized_count": typed_retry_text_json_normalized_count,
        "internal_abstain_reason_suppressed_count": internal_abstain_reason_suppressed_count,
        "expected_type_text_rejected_count": expected_type_text_rejected_count,
        "source_family_alias_hit_count": source_family_alias_hit_count,
        "count_ref_validation_rejected_count": count_ref_validation_rejected_count,
        "count_validation_llm_used_count": count_validation_llm_used_count,
        "count_validation_llm_success_count": count_validation_llm_success_count,
        "count_validation_llm_changed_count": count_validation_llm_changed_count,
        "count_validation_llm_success_rate": _safe_rate(
            count_validation_llm_success_count,
            count_validation_llm_used_count,
        ),
        "source_derived_count_candidate_count": source_derived_candidate_count,
        "source_derived_count_candidate_accepted_count": source_derived_candidate_accepted_count,
        "source_derived_pronoun_caption_candidate_count": source_derived_pronoun_caption_candidate_count,
        "source_derived_passive_rejected_count": source_derived_passive_rejected_count,
        "temporal_alignment_rejected_count": temporal_alignment_rejected_count,
        "temporal_repair_used_count": temporal_repair_used_count,
        "temporal_low_confidence_candidate_count": temporal_low_confidence_candidate_count,
        "temporal_no_query_relevant_candidate_count": temporal_no_query_relevant_candidate_count,
        "temporal_candidate_selection_rejected_count": temporal_candidate_selection_rejected_count,
        "date_question_list_repair_blocked_count": date_question_list_repair_blocked_count,
        "list_scope_rejected_item_count": list_scope_rejected_item_count,
        "missing_required_item_count": missing_required_item_count,
        "overgeneric_answer_count": overgeneric_answer_count,
        "scope_mismatched_extra_item_count": scope_mismatched_extra_item_count,
        "specific_item_repair_used_count": specific_item_repair_used_count,
        "missing_supported_list_item_count": missing_supported_list_item_count,
        "abstain_despite_supported_items_count": abstain_despite_supported_items_count,
        "list_coverage_repair_used_count": list_coverage_repair_used_count,
        "event_canonical_alias_item_count": event_canonical_alias_item_count,
        "answer_repair_removed_scope_mismatched_item_count": answer_repair_removed_scope_mismatched_item_count,
        "answer_repair_missing_required_items_after_repair_count": answer_repair_missing_required_items_after_repair_count,
        "repair_dropped_supported_item_count": repair_dropped_supported_item_count,
        "repair_discarded_by_post_validation_count": repair_discarded_by_post_validation_count,
        "repair_arbitration_triggered_count": repair_arbitration_triggered_count,
        "repair_arbitration_used_count": repair_arbitration_used_count,
        "repair_arbitration_keep_initial_count": repair_arbitration_keep_initial_count,
        "repair_arbitration_use_repair_count": repair_arbitration_use_repair_count,
        "repair_arbitration_safe_abstain_count": repair_arbitration_safe_abstain_count,
        "repair_arbitration_failed_count": repair_arbitration_failed_count,
        "list_required_item_recall_after_repair": _safe_rate(
            list_required_item_after_repair_hit_count,
            list_required_item_count,
        ),
        "bridge_finalization_used_count": bridge_finalization_used_count,
        "bridge_finalization_conflict_count": bridge_finalization_conflict_count,
        "answer_bridge_repair_used_count": answer_bridge_repair_used_count,
        "preference_query_shape_count": preference_query_shape_count,
        "count_validation_excluded_event_count": count_validation_excluded_count,
        "bridge_fact_used_count": bridge_fact_used_count,
    }


def _build_retrieval_reflection_diagnostics(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(sample_rows)
    fields_by_index = [_reflection_fields_from_row(row) for row in sample_rows]
    retry_indexes = [
        index for index, fields in enumerate(fields_by_index) if fields["retrieval_reflection_used"]
    ]
    raw_attempt_indexes = [
        index for index in retry_indexes if fields_by_index[index]["raw_rescue_attempted"] is True
    ]
    raw_skipped_indexes = [
        index for index in retry_indexes if fields_by_index[index]["raw_rescue_attempted"] is False
    ]
    raw_indexes = [index for index, fields in enumerate(fields_by_index) if fields["raw_rescue_used"]]
    raw_hit_indexes = [
        index for index in raw_attempt_indexes if int(fields_by_index[index]["raw_rescue_hit_count"]) > 0
    ]
    semantic_weak_indexes = [
        index for index in retry_indexes if fields_by_index[index]["reflection_semantic_evidence_weak"]
    ]
    post_reflection_raw_indexes = [
        index for index in retry_indexes if fields_by_index[index]["post_reflection_raw_rescue_used"]
    ]
    answer_changed_indexes = [
        index for index in retry_indexes if fields_by_index[index]["reflection_answer_changed"]
    ]
    successful_retry_indexes = [
        index
        for index in retry_indexes
        if str(sample_rows[index].get("judge_verdict", "")).lower() in {"correct", "partial"}
    ]
    return {
        "retry_triggered_count": len(retry_indexes),
        "retry_triggered_rate_over_all": _safe_rate(len(retry_indexes), total),
        "wiki_reroute_used_count": sum(
            1
            for index in retry_indexes
            if fields_by_index[index]["retrieval_reflection_stage"] == "wiki"
        ),
        "raw_rescue_used_count": len(raw_indexes),
        "raw_rescue_used_rate_over_all": _safe_rate(len(raw_indexes), total),
        "raw_rescue_attempted_count": len(raw_attempt_indexes),
        "raw_rescue_attempted_rate_over_retry": _safe_rate(len(raw_attempt_indexes), len(retry_indexes)),
        "raw_rescue_skipped_count": len(raw_skipped_indexes),
        "raw_rescue_skipped_rate_over_retry": _safe_rate(len(raw_skipped_indexes), len(retry_indexes)),
        "raw_rescue_hit_count": len(raw_hit_indexes),
        "raw_rescue_hit_rate_over_raw_rescue": _safe_rate(len(raw_hit_indexes), len(raw_indexes)),
        "raw_rescue_hit_rate_over_attempted": _safe_rate(len(raw_hit_indexes), len(raw_attempt_indexes)),
        "semantic_weak_trigger_count": len(semantic_weak_indexes),
        "semantic_weak_trigger_rate_over_retry": _safe_rate(len(semantic_weak_indexes), len(retry_indexes)),
        "post_reflection_raw_rescue_count": len(post_reflection_raw_indexes),
        "post_reflection_raw_rescue_rate_over_retry": _safe_rate(
            len(post_reflection_raw_indexes), len(retry_indexes)
        ),
        "answer_changed_count": len(answer_changed_indexes),
        "answer_changed_rate_over_retry": _safe_rate(len(answer_changed_indexes), len(retry_indexes)),
        "retry_correct_or_partial_count": len(successful_retry_indexes),
        "retry_correct_or_partial_rate_over_retry": _safe_rate(
            len(successful_retry_indexes),
            len(retry_indexes),
        ),
    }


def _judge_diagnostics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    scalar_keys = [
        "judge_evaluable_count",
        "judge_execution_failed_count",
        "judge_execution_failed_rate_over_all",
        "partial_count",
        "partial_rate_over_all",
        "partial_rate_over_non_correct",
        "mean_partial_credit_judge_acc",
        "structured_requested_rate_over_all",
        "structured_success_rate_over_all",
        "text_fallback_rate_over_all",
        "text_only_rate_over_all",
        "incorrect_queries_judged_via_text_fallback_count",
        "incorrect_queries_judged_via_text_fallback_rate_over_incorrect",
    ]
    delta: dict[str, Any] = {}
    for key in scalar_keys:
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value is None or after_value is None:
            delta[key] = None
        else:
            delta[key] = float(after_value) - float(before_value)
    before_categories = dict(before.get("structured_fallback_category_counts", {}))
    after_categories = dict(after.get("structured_fallback_category_counts", {}))
    delta["structured_fallback_category_counts"] = {
        category: {
            "before": int(before_categories.get(category, 0)),
            "after": int(after_categories.get(category, 0)),
            "delta": int(after_categories.get(category, 0)) - int(before_categories.get(category, 0)),
        }
        for category in sorted(set(before_categories) | set(after_categories))
    }
    return delta


def _llm_call_diagnostics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    numeric_keys = [
        "provider_call_count",
        "logical_call_count",
        "batch_call_count",
        "batch_item_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "fallback_count",
        "repair_count",
        "error_count",
        "cache_hit_count",
        "cache_miss_count",
    ]

    def section_delta(before_section: dict[str, Any], after_section: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (
                None
                if before_section.get(key) is None or after_section.get(key) is None
                else float(after_section.get(key) or 0) - float(before_section.get(key) or 0)
            )
            for key in numeric_keys
        }

    before_tasks = dict(before.get("by_task") or {})
    after_tasks = dict(after.get("by_task") or {})
    return {
        "overall": section_delta(dict(before.get("overall") or {}), dict(after.get("overall") or {})),
        "by_task": {
            task: {
                "before": dict(before_tasks.get(task) or {}),
                "after": dict(after_tasks.get(task) or {}),
                "delta": section_delta(
                    dict(before_tasks.get(task) or {}),
                    dict(after_tasks.get(task) or {}),
                ),
            }
            for task in sorted(set(before_tasks) | set(after_tasks))
        },
    }


def _fallback_repair_diagnostics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    numeric_keys = [
        "event_count",
        "weighted_event_count",
        "fallback_event_count",
        "repair_event_count",
        "cache_event_count",
        "extra_provider_call_count",
        "extra_total_tokens",
        "extra_latency_ms",
        "repair_discarded_count",
        "repair_discard_rate",
        "quality_risk_event_count",
        "quality_risk_weighted_event_count",
    ]
    paper_keys = [
        "extra_llm_call_rate",
        "extra_token_overhead_rate",
        "mean_judge_score_with_fallback",
        "mean_judge_score_without_fallback",
        "incorrect_rate_with_repair",
        "incorrect_rate_without_repair",
    ]

    def section_delta(before_section: dict[str, Any], after_section: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        for key in keys:
            before_value = before_section.get(key)
            after_value = after_section.get(key)
            if before_value is None or after_value is None:
                delta[key] = None
                continue
            try:
                delta[key] = float(after_value) - float(before_value)
            except (TypeError, ValueError):
                delta[key] = None
        return delta

    before_tasks = dict(before.get("by_task") or {})
    after_tasks = dict(after.get("by_task") or {})
    return {
        "diagnostic_mode": {
            "before": before.get("diagnostic_mode"),
            "after": after.get("diagnostic_mode"),
        },
        "loaded_event_count": (
            None
            if before.get("loaded_event_count") is None or after.get("loaded_event_count") is None
            else float(after.get("loaded_event_count") or 0) - float(before.get("loaded_event_count") or 0)
        ),
        "overall": section_delta(dict(before.get("overall") or {}), dict(after.get("overall") or {}), numeric_keys),
        "paper_ready": section_delta(
            dict(before.get("paper_ready") or {}),
            dict(after.get("paper_ready") or {}),
            paper_keys,
        ),
        "by_task": {
            task: {
                "before": dict(before_tasks.get(task) or {}),
                "after": dict(after_tasks.get(task) or {}),
                "delta": section_delta(
                    dict(before_tasks.get(task) or {}),
                    dict(after_tasks.get(task) or {}),
                    numeric_keys,
                ),
            }
            for task in sorted(set(before_tasks) | set(after_tasks))
        },
    }


def _safe_percentile(values: list[int | float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil((float(percentile) / 100.0) * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def _length_stats(values: list[int | float]) -> dict[str, Any]:
    numeric_values = [float(value) for value in values if value is not None]
    return {
        "count": len(numeric_values),
        "mean": _safe_mean(numeric_values),
        "median": _safe_median(numeric_values),
        "p90": _safe_percentile(numeric_values, 90),
        "p95": _safe_percentile(numeric_values, 95),
        "max": max(numeric_values) if numeric_values else None,
    }


def _rank_in_order(ordered_ids: list[str], target_ids: set[str]) -> int | None:
    if not ordered_ids or not target_ids:
        return None
    for index, current_id in enumerate(ordered_ids, start=1):
        if current_id in target_ids:
            return index
    return None


def _min_covering_trajectory_count(
    gold_refs: set[str],
    trajectory_ids: list[str],
    trajectory_refs: dict[str, set[str]],
) -> int:
    if not gold_refs or not trajectory_ids:
        return 0
    relevant_ids = [trajectory_id for trajectory_id in trajectory_ids if trajectory_refs.get(trajectory_id, set()) & gold_refs]
    for size in range(1, len(relevant_ids) + 1):
        for subset in combinations(relevant_ids, size):
            covered_refs: set[str] = set()
            for trajectory_id in subset:
                covered_refs.update(trajectory_refs.get(trajectory_id, set()) & gold_refs)
            if gold_refs <= covered_refs:
                return size
    return len(relevant_ids)


def _resolve_top_k_trajectory_ids(retrieval_event: dict[str, Any]) -> list[str]:
    top_k = int(retrieval_event.get("top_k") or 0)
    direct_ids = [str(item) for item in list(retrieval_event.get("trajectory_ids") or []) if str(item).strip()]
    if direct_ids:
        return direct_ids[: top_k or len(direct_ids)]
    metadata = dict(retrieval_event.get("metadata") or {})
    selected_ids = [
        str(item)
        for item in list(metadata.get("coarse_top_k_selected_ids") or [])
        if str(item).strip()
    ]
    if selected_ids:
        return selected_ids[: top_k or len(selected_ids)]
    ranked_ids = [str(item) for item in list(retrieval_event.get("coarse_ranked_ids") or []) if str(item).strip()]
    if ranked_ids:
        return ranked_ids[: top_k or len(ranked_ids)]
    return []


def _resolve_top_t_page_ids(retrieval_event: dict[str, Any]) -> list[str]:
    top_t = int(retrieval_event.get("top_t_pages") or 0)
    direct_ids = [str(item) for item in list(retrieval_event.get("page_ids") or []) if str(item).strip()]
    if direct_ids:
        return direct_ids[: top_t or len(direct_ids)]
    metadata = dict(retrieval_event.get("metadata") or {})
    selected_ids = [
        str(item)
        for item in list(metadata.get("page_rerank_selected_ids") or [])
        if str(item).strip()
    ]
    if selected_ids:
        return selected_ids[: top_t or len(selected_ids)]
    ranked_ids = [
        str(item)
        for item in list(metadata.get("page_candidate_ids") or [])
        if str(item).strip()
    ]
    if ranked_ids:
        return ranked_ids[: top_t or len(ranked_ids)]
    return []


def _compute_top_k_coverage(
    *,
    gold_refs: set[str],
    gold_trajectory_ids: list[str],
    top_k_trajectory_ids: list[str],
    selection_pool_trajectory_ids: list[str] | None = None,
    trajectory_refs: dict[str, set[str]],
) -> dict[str, Any]:
    selection_pool_available = selection_pool_trajectory_ids is not None
    selection_pool_ids = list(selection_pool_trajectory_ids or [])
    gold_trajectory_ids_in_top_k = [
        trajectory_id for trajectory_id in top_k_trajectory_ids if trajectory_id in set(gold_trajectory_ids)
    ]
    missing_gold_trajectory_ids_from_top_k = [
        trajectory_id for trajectory_id in gold_trajectory_ids if trajectory_id not in set(top_k_trajectory_ids)
    ]
    if selection_pool_available:
        gold_trajectory_ids_in_selection_pool = [
            trajectory_id for trajectory_id in selection_pool_ids if trajectory_id in set(gold_trajectory_ids)
        ]
        missing_gold_trajectory_ids_from_selection_pool = [
            trajectory_id for trajectory_id in gold_trajectory_ids if trajectory_id not in set(selection_pool_ids)
        ]
    else:
        gold_trajectory_ids_in_selection_pool = []
        missing_gold_trajectory_ids_from_selection_pool = []
    gold_trajectory_count = len(gold_trajectory_ids)
    gold_trajectory_count_in_top_k = len(gold_trajectory_ids_in_top_k)
    gold_trajectory_count_in_selection_pool = (
        len(gold_trajectory_ids_in_selection_pool) if selection_pool_available else None
    )
    gold_trajectory_recall_at_k = (
        _safe_rate(gold_trajectory_count_in_top_k, gold_trajectory_count)
        if gold_trajectory_count > 0
        else None
    )
    gold_trajectory_recall_in_selection_pool = (
        _safe_rate(int(gold_trajectory_count_in_selection_pool or 0), gold_trajectory_count)
        if selection_pool_available and gold_trajectory_count > 0
        else None
    )
    selection_pool_can_cover_all_gold_trajectories = (
        bool(gold_trajectory_count) and gold_trajectory_count_in_selection_pool == gold_trajectory_count
    ) if selection_pool_available else None
    top_k_refs = set().union(*(trajectory_refs.get(trajectory_id, set()) for trajectory_id in top_k_trajectory_ids))
    top_k_gold_ref_coverage_count = len(gold_refs & top_k_refs)
    top_k_gold_ref_coverage_rate = _safe_rate(top_k_gold_ref_coverage_count, len(gold_refs)) if gold_refs else None
    top_k_can_cover_all_gold_refs = bool(gold_refs and gold_refs <= top_k_refs) if gold_refs else None
    top_k_covering_trajectory_count = (
        _min_covering_trajectory_count(gold_refs, top_k_trajectory_ids, trajectory_refs)
        if top_k_can_cover_all_gold_refs
        else None
    )
    return {
        "top_k_trajectory_ids": top_k_trajectory_ids,
        "selection_pool_trajectory_ids": selection_pool_ids,
        "trajectory_selection_pool_available": selection_pool_available,
        "gold_trajectory_ids_in_top_k": gold_trajectory_ids_in_top_k,
        "missing_gold_trajectory_ids_from_top_k": missing_gold_trajectory_ids_from_top_k,
        "gold_trajectory_ids_in_selection_pool": gold_trajectory_ids_in_selection_pool,
        "missing_gold_trajectory_ids_from_selection_pool": missing_gold_trajectory_ids_from_selection_pool,
        "gold_trajectory_count_in_top_k": gold_trajectory_count_in_top_k,
        "gold_trajectory_count_in_selection_pool": gold_trajectory_count_in_selection_pool,
        "gold_trajectory_recall_at_k": gold_trajectory_recall_at_k,
        "gold_trajectory_recall_in_selection_pool": gold_trajectory_recall_in_selection_pool,
        "selection_pool_can_cover_all_gold_trajectories": selection_pool_can_cover_all_gold_trajectories,
        "top_k_gold_ref_coverage_count": top_k_gold_ref_coverage_count,
        "top_k_gold_ref_coverage_rate": top_k_gold_ref_coverage_rate,
        "top_k_can_cover_all_gold_refs": top_k_can_cover_all_gold_refs,
        "top_k_covering_trajectory_count": top_k_covering_trajectory_count,
    }


def _direct_signal_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if value is None:
        return strings
    if isinstance(value, str):
        compact = collapse_whitespace(value)
        return [compact] if compact else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        for child in value.values():
            strings.extend(_direct_signal_strings(child))
        return strings
    if isinstance(value, (list, tuple, set)):
        for child in value:
            strings.extend(_direct_signal_strings(child))
        return strings
    return []


def _direct_query_terms(
    *,
    question: str,
    query_entities: list[str],
    query_facets: dict[str, list[str]],
    query_shape: dict[str, Any],
) -> set[str]:
    terms = {
        term
        for term in extract_keywords(question)
        if term not in _ANSWER_TOKEN_STOPWORDS and term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
    }
    for entity in query_entities:
        terms.update(
            term for term in extract_keywords(entity) if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        )
        entity_key = normalize_entity_key(entity)
        if entity_key:
            terms.add(entity_key)
    for value in list(query_facets.get("tags") or []) + list(query_facets.get("values") or []):
        terms.update(
            term
            for term in extract_keywords(str(value).replace("_", " "))
            if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        )
    item_family = str(query_shape.get("item_family") or "").strip()
    if item_family:
        terms.update(
            term
            for term in extract_keywords(item_family.replace("_", " "))
            if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        )
    return terms


def _direct_trajectory_signal_texts(
    *,
    trajectory_id: str,
    trajectory_metadata: dict[str, dict[str, Any]],
    claims_by_trajectory: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], bool]:
    profile = _direct_trajectory_signal_profile(
        trajectory_id=trajectory_id,
        trajectory_metadata=trajectory_metadata,
        claims_by_trajectory=claims_by_trajectory,
    )
    return list(profile["texts"]), bool(profile["has_metadata_signals"])


def _truncate_direct_value(value: Any, limit: int = DIRECT_TRAJECTORY_STRING_LIMIT) -> str:
    text = collapse_whitespace(str(value or ""))
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _compact_direct_terms(values: Iterable[Any], *, limit: int = DIRECT_TRAJECTORY_TERM_LIMIT) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _truncate_direct_value(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def _direct_term_matches(query_terms: set[str], values: Iterable[Any]) -> list[str]:
    matched: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = collapse_whitespace(str(value or ""))
        if not text:
            continue
        value_terms = {
            term
            for term in extract_keywords(text)
            if term not in _ANSWER_TOKEN_STOPWORDS and term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        }
        if not (query_terms & value_terms):
            continue
        compact = _truncate_direct_value(text)
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        matched.append(compact)
        if len(matched) >= DIRECT_TRAJECTORY_TERM_LIMIT:
            break
    return matched


def _direct_trajectory_signal_profile(
    *,
    trajectory_id: str,
    trajectory_metadata: dict[str, dict[str, Any]],
    claims_by_trajectory: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    metadata = dict(trajectory_metadata.get(trajectory_id) or {})
    metadata_values: list[str] = []
    profile: dict[str, list[str]] = {
        "exact_terms": [],
        "source_surface_terms": [],
        "historical_terms": [],
        "display_items": [],
        "display_counts": [],
        "display_key_facts": [],
        "facet_values": [],
        "entity_mentions": [],
        "entity_keys": [],
        "drift_cluster_keys": [],
    }

    def _add_profile_values(name: str, value: Any) -> None:
        strings = _direct_signal_strings(value)
        profile.setdefault(name, []).extend(strings)
        metadata_values.extend(strings)

    for key in [
        "label",
        "retrieval_summary_text",
        "trajectory_identity_summary_v1",
        "trajectory_recent_update_v1",
    ]:
        metadata_values.extend(_direct_signal_strings(metadata.get(key)))
    historical_values = sanitize_historical_item_terms(
        list(
            metadata.get("trajectory_historical_item_terms_v2")
            or metadata.get("trajectory_historical_item_terms_v1")
            or []
        ),
        limit=DIRECT_TRAJECTORY_TERM_LIMIT,
    )
    if historical_values:
        _add_profile_values("historical_terms", historical_values)
    for key, profile_key in [
        ("historical_item_terms", "historical_terms"),
        ("wiki_historical_item_terms", "historical_terms"),
        ("exact_terms", "exact_terms"),
        ("exact_terms_v2", "exact_terms"),
        ("source_surface_terms_v1", "source_surface_terms"),
        ("source_surface_raw_terms_v1", "source_surface_terms"),
        ("display_items", "display_items"),
        ("display_counts", "display_counts"),
        ("display_key_facts", "display_key_facts"),
        ("facet_values", "facet_values"),
        ("entity_mentions", "entity_mentions"),
        ("entity_keys", "entity_keys"),
        ("drift_cluster_keys", "drift_cluster_keys"),
    ]:
        _add_profile_values(profile_key, metadata.get(key))
    for card_key in ["trajectory_evidence_card_v1", "trajectory_historical_evidence_card_v1"]:
        card = metadata.get(card_key)
        metadata_values.extend(_direct_signal_strings(card))
        if isinstance(card, dict):
            for key, profile_key in [
                ("source_surface_terms", "source_surface_terms"),
                ("display_items", "display_items"),
                ("display_counts", "display_counts"),
                ("display_key_facts", "display_key_facts"),
                ("facet_values", "facet_values"),
                ("entity_mentions", "entity_mentions"),
                ("source_anchors", "historical_terms"),
            ]:
                profile.setdefault(profile_key, []).extend(_direct_signal_strings(card.get(key)))
            profile.setdefault("historical_terms", []).extend(
                _direct_signal_strings(
                    sanitize_historical_item_terms(list(card.get("historical_item_terms") or []), limit=24)
                )
            )
    claim_values: list[str] = []
    for claim in claims_by_trajectory.get(trajectory_id, []):
        claim_values.append(str(claim.get("text") or ""))
        exact_terms = _direct_signal_strings(claim.get("exact_terms"))
        facets = _direct_signal_strings(claim.get("facets"))
        display_signals = _direct_signal_strings(claim.get("display_signals"))
        claim_values.extend(exact_terms)
        claim_values.extend(facets)
        claim_values.extend(display_signals)
        profile["exact_terms"].extend(exact_terms)
        profile["facet_values"].extend(facets)
        profile["display_items"].extend(display_signals)
    texts = [value for value in [*metadata_values, *claim_values] if collapse_whitespace(str(value or ""))]
    has_metadata_signals = any(collapse_whitespace(value) for value in metadata_values)
    return {
        "texts": texts,
        "has_metadata_signals": has_metadata_signals,
        **{key: _compact_direct_terms(values, limit=50) for key, values in profile.items()},
    }


def _internal_metadata_term_hits(values: Iterable[Any]) -> list[str]:
    hits: list[str] = []
    for value in values:
        text = collapse_whitespace(str(value or ""))
        if not text:
            continue
        if is_internal_summary_keyword(text):
            hits.append(text)
            continue
        hits.extend(token for token in extract_keywords(text) if is_internal_summary_keyword(token))
    return list(dict.fromkeys(hits))


def _metadata_term_diagnostics(trajectory_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keyword_counts: list[int] = []
    summary_keyword_internal_hits: list[str] = []
    historical_internal_hits: list[str] = []
    historical_summary_fallback_count = 0
    keyword_policy_counts: Counter[str] = Counter()
    historical_policy_counts: Counter[str] = Counter()
    for metadata in trajectory_metadata.values():
        raw_keywords = list(
            metadata.get("retrieval_summary_keywords_v2")
            or metadata.get("retrieval_summary_keywords")
            or []
        )
        keywords = sanitize_summary_keyword_values(raw_keywords, limit=64)
        keyword_counts.append(len(keywords))
        summary_keyword_internal_hits.extend(_internal_metadata_term_hits(raw_keywords))
        keyword_policy_counts[str(metadata.get("retrieval_summary_keyword_policy") or "unknown")] += 1
        historical_terms = list(
            metadata.get("trajectory_historical_item_terms_v2")
            or metadata.get("trajectory_historical_item_terms_v1")
            or []
        )
        historical_internal_hits.extend(_internal_metadata_term_hits(historical_terms))
        card = metadata.get("trajectory_historical_evidence_card_v1")
        if isinstance(card, dict):
            historical_policy_counts[str(card.get("historical_item_terms_policy") or "unknown")] += 1
            if list(card.get("historical_item_terms_summary_fallback_v1") or []):
                historical_summary_fallback_count += 1
            historical_internal_hits.extend(_internal_metadata_term_hits(list(card.get("historical_item_terms") or [])))
        else:
            historical_policy_counts["missing_card"] += 1
    return {
        "trajectory_count": len(trajectory_metadata),
        "summary_keyword_mean_count": _safe_mean(keyword_counts),
        "summary_keyword_internal_term_hit_count": len(summary_keyword_internal_hits),
        "summary_keyword_internal_terms": sorted(set(summary_keyword_internal_hits))[:24],
        "summary_keyword_policy_counts": dict(keyword_policy_counts),
        "historical_item_internal_term_hit_count": len(historical_internal_hits),
        "historical_item_internal_terms": sorted(set(historical_internal_hits))[:24],
        "historical_item_summary_fallback_count": historical_summary_fallback_count,
        "historical_item_policy_counts": dict(historical_policy_counts),
    }


def _metadata_term_fields_for_trajectories(
    trajectory_ids: Iterable[str],
    trajectory_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    keyword_policies: list[str] = []
    historical_policies: list[str] = []
    internal_hits: list[str] = []
    summary_fallback_count = 0
    for trajectory_id in trajectory_ids:
        metadata = dict(trajectory_metadata.get(str(trajectory_id)) or {})
        if not metadata:
            continue
        keyword_policies.append(str(metadata.get("retrieval_summary_keyword_policy") or "unknown"))
        raw_keywords = list(
            metadata.get("retrieval_summary_keywords_v2")
            or metadata.get("retrieval_summary_keywords")
            or []
        )
        internal_hits.extend(_internal_metadata_term_hits(raw_keywords))
        terms = list(
            metadata.get("trajectory_historical_item_terms_v2")
            or metadata.get("trajectory_historical_item_terms_v1")
            or []
        )
        internal_hits.extend(_internal_metadata_term_hits(terms))
        card = metadata.get("trajectory_historical_evidence_card_v1")
        if isinstance(card, dict):
            historical_policies.append(str(card.get("historical_item_terms_policy") or "unknown"))
            internal_hits.extend(_internal_metadata_term_hits(list(card.get("historical_item_terms") or [])))
            if list(card.get("historical_item_terms_summary_fallback_v1") or []):
                summary_fallback_count += 1
    return {
        "metadata_terms_keyword_policy": ",".join(sorted(set(keyword_policies))) if keyword_policies else None,
        "metadata_terms_historical_policy": ",".join(sorted(set(historical_policies))) if historical_policies else None,
        "metadata_terms_internal_leak_count": len(internal_hits),
        "metadata_terms_internal_leaks": sorted(set(internal_hits))[:12],
        "metadata_terms_summary_fallback_count": summary_fallback_count,
    }


def _score_direct_trajectory(
    *,
    trajectory_id: str,
    query_terms: set[str],
    query_entities: list[str],
    query_facets: dict[str, list[str]],
    query_shape: dict[str, Any],
    trajectory_metadata: dict[str, dict[str, Any]],
    claims_by_trajectory: dict[str, list[dict[str, Any]]],
) -> tuple[float, bool]:
    profile = _score_direct_trajectory_profile(
        trajectory_id=trajectory_id,
        trajectory_metadata=trajectory_metadata,
        claims_by_trajectory=claims_by_trajectory,
        query_terms=query_terms,
        query_entities=query_entities,
        query_facets=query_facets,
        query_shape=query_shape,
    )
    return float(profile["final_score"]), bool(profile["has_metadata_signals"])


def _score_direct_trajectory_profile(
    *,
    trajectory_id: str,
    query_terms: set[str],
    query_entities: list[str],
    query_facets: dict[str, list[str]],
    query_shape: dict[str, Any],
    trajectory_metadata: dict[str, dict[str, Any]],
    claims_by_trajectory: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    signal_profile = _direct_trajectory_signal_profile(
        trajectory_id=trajectory_id,
        trajectory_metadata=trajectory_metadata,
        claims_by_trajectory=claims_by_trajectory,
    )
    texts = list(signal_profile["texts"])
    has_metadata_signals = bool(signal_profile["has_metadata_signals"])
    haystack = " ".join(texts).casefold()
    candidate_terms = {
        term
        for text in texts
        for term in extract_keywords(text)
        if term not in _ANSWER_TOKEN_STOPWORDS and term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
    }
    overlap = query_terms & candidate_terms
    lexical_score = float(len(overlap))
    if query_terms:
        lexical_score += 2.0 * (len(overlap) / len(query_terms))
    entity_overlap_score = 0.0
    matched_entity_terms: set[str] = set()
    for entity in query_entities:
        entity_key = normalize_entity_key(entity)
        entity_terms = {
            term for term in extract_keywords(entity) if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        }
        if entity_key and entity_key in candidate_terms:
            entity_overlap_score += 3.0
            matched_entity_terms.add(entity_key)
        elif entity_terms & candidate_terms:
            entity_overlap_score += 2.0
            matched_entity_terms.update(entity_terms & candidate_terms)
        elif entity and entity.casefold() in haystack:
            entity_overlap_score += 1.5
            matched_entity_terms.add(entity)
    facet_overlap_score = 0.0
    matched_facet_terms: set[str] = set()
    for facet in list(query_facets.get("tags") or []) + list(query_facets.get("values") or []):
        facet_terms = {
            term
            for term in extract_keywords(str(facet).replace("_", " "))
            if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        }
        if facet_terms and facet_terms <= candidate_terms:
            facet_overlap_score += 2.0
            matched_facet_terms.update(facet_terms)
        elif facet_terms & candidate_terms:
            facet_overlap_score += 1.0
            matched_facet_terms.update(facet_terms & candidate_terms)
    family_match_score = 0.0
    matched_family_terms: set[str] = set()
    item_family = str(query_shape.get("item_family") or "").strip()
    if item_family:
        family_terms = {
            term
            for term in extract_keywords(item_family.replace("_", " "))
            if term not in _DIRECT_TRAJECTORY_GENERIC_TERMS
        }
        if family_terms & candidate_terms:
            family_match_score += 1.5
            matched_family_terms.update(family_terms & candidate_terms)
    temporal_score = 0.0
    if bool(query_shape.get("count_like")) and re.search(r"\b(?:count|number|times?|once|twice|\d+)\b", haystack):
        temporal_score += 0.75
    if bool(query_shape.get("list_like")) and any(
        key in haystack for key in ("display_items", "items", "list", "visited", "participated", "bought", "read")
    ):
        family_match_score += 0.5
    specific_values = (
        list(signal_profile.get("exact_terms") or [])
        + list(signal_profile.get("source_surface_terms") or [])
        + list(signal_profile.get("historical_terms") or [])
        + list(signal_profile.get("display_items") or [])
        + list(signal_profile.get("display_key_facts") or [])
    )
    query_object_overlap_terms = {
        term
        for value in specific_values
        for term in extract_keywords(value)
        if term in query_terms
    }
    query_object_overlap_score = (
        1.25 * (len(query_object_overlap_terms) / len(query_terms)) if query_terms else 0.0
    )
    generic_only_penalty = 0.0
    if overlap and not query_object_overlap_terms and not matched_entity_terms and not matched_facet_terms:
        generic_only_penalty = 0.25
    final_score = (
        lexical_score
        + entity_overlap_score
        + facet_overlap_score
        + family_match_score
        + temporal_score
        + query_object_overlap_score
        - generic_only_penalty
    )
    return {
        "trajectory_id": trajectory_id,
        "final_score": final_score,
        "lexical_score": lexical_score,
        "family_match_score": family_match_score,
        "query_object_overlap_score": query_object_overlap_score,
        "entity_overlap_score": entity_overlap_score,
        "facet_overlap_score": facet_overlap_score,
        "temporal_score": temporal_score,
        "generic_only_penalty": generic_only_penalty,
        "matched_query_terms": sorted(overlap),
        "matched_entity_terms": sorted(matched_entity_terms),
        "matched_facet_terms": sorted(matched_facet_terms),
        "matched_family_terms": sorted(matched_family_terms),
        "matched_exact_terms": _direct_term_matches(query_terms, signal_profile.get("exact_terms") or []),
        "matched_source_surface_terms": _direct_term_matches(
            query_terms, signal_profile.get("source_surface_terms") or []
        ),
        "matched_historical_terms": _direct_term_matches(query_terms, signal_profile.get("historical_terms") or []),
        "entity_keys": _compact_direct_terms(
            list(signal_profile.get("entity_keys") or []) + list(signal_profile.get("entity_mentions") or [])
        ),
        "facet_values": _compact_direct_terms(signal_profile.get("facet_values") or []),
        "item_family": str(query_shape.get("item_family") or ""),
        "exact_terms": _compact_direct_terms(signal_profile.get("exact_terms") or []),
        "source_surface_terms": _compact_direct_terms(signal_profile.get("source_surface_terms") or []),
        "historical_item_terms": _compact_direct_terms(signal_profile.get("historical_terms") or []),
        "has_metadata_signals": has_metadata_signals,
        "_texts": texts,
    }


def _direct_estimated_context_tokens(texts: Iterable[Any]) -> int:
    tokenish_count = sum(len(str(text or "").split()) for text in texts)
    return max(1, int(math.ceil(tokenish_count * 1.3))) if tokenish_count else 0


def _direct_compact_score_row(
    profile: dict[str, Any],
    *,
    rank: int,
    trajectory_refs: dict[str, set[str]] | None,
    trajectory_lengths: dict[str, int] | None,
) -> dict[str, Any]:
    trajectory_id = str(profile.get("trajectory_id") or "")
    refs = sorted((trajectory_refs or {}).get(trajectory_id, set()))
    texts = list(profile.get("_texts") or [])
    return {
        "rank": rank,
        "trajectory_id": trajectory_id,
        "final_score": float(profile.get("final_score") or 0.0),
        "lexical_score": float(profile.get("lexical_score") or 0.0),
        "family_match_score": float(profile.get("family_match_score") or 0.0),
        "query_object_overlap_score": float(profile.get("query_object_overlap_score") or 0.0),
        "entity_overlap_score": float(profile.get("entity_overlap_score") or 0.0),
        "facet_overlap_score": float(profile.get("facet_overlap_score") or 0.0),
        "temporal_score": float(profile.get("temporal_score") or 0.0),
        "generic_only_penalty": float(profile.get("generic_only_penalty") or 0.0),
        "matched_query_terms": list(profile.get("matched_query_terms") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "matched_exact_terms": list(profile.get("matched_exact_terms") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "matched_source_surface_terms": list(profile.get("matched_source_surface_terms") or [])[
            :DIRECT_TRAJECTORY_TERM_LIMIT
        ],
        "matched_historical_terms": list(profile.get("matched_historical_terms") or [])[
            :DIRECT_TRAJECTORY_TERM_LIMIT
        ],
        "entity_keys": list(profile.get("entity_keys") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "facet_values": list(profile.get("facet_values") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "item_family": profile.get("item_family") or "",
        "exact_terms": list(profile.get("exact_terms") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "source_surface_terms": list(profile.get("source_surface_terms") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "historical_item_terms": list(profile.get("historical_item_terms") or [])[:DIRECT_TRAJECTORY_TERM_LIMIT],
        "snapshot_count_estimate": int((trajectory_lengths or {}).get(trajectory_id, 0) or 0),
        "source_ref_count_estimate": len(refs),
        "context_token_estimate": _direct_estimated_context_tokens(texts),
    }


def _rank_direct_trajectories(
    *,
    sample_id: str,
    question: str,
    query_entities: list[str],
    query_facets: dict[str, list[str]],
    query_shape: dict[str, Any],
    sample_to_trajectories: dict[str, set[str]],
    trajectory_metadata: dict[str, dict[str, Any]],
    claims_by_trajectory: dict[str, list[dict[str, Any]]],
    trajectory_refs: dict[str, set[str]] | None = None,
    trajectory_lengths: dict[str, int] | None = None,
) -> tuple[list[str], str, list[dict[str, Any]]]:
    candidates = sorted(str(trajectory_id) for trajectory_id in sample_to_trajectories.get(sample_id, set()))
    query_terms = _direct_query_terms(
        question=question,
        query_entities=query_entities,
        query_facets=query_facets,
        query_shape=query_shape,
    )
    scored: list[tuple[float, str, dict[str, Any]]] = []
    metadata_signal_count = 0
    for trajectory_id in candidates:
        profile = _score_direct_trajectory_profile(
            trajectory_id=trajectory_id,
            query_terms=query_terms,
            query_entities=query_entities,
            query_facets=query_facets,
            query_shape=query_shape,
            trajectory_metadata=trajectory_metadata,
            claims_by_trajectory=claims_by_trajectory,
        )
        metadata_signal_count += int(bool(profile.get("has_metadata_signals")))
        scored.append((float(profile.get("final_score") or 0.0), trajectory_id, profile))
    sorted_scored = sorted(scored, key=lambda item: (-item[0], item[1]))
    ranked = [
        trajectory_id
        for _, trajectory_id, _ in sorted_scored
    ]
    compact_rows = [
        _direct_compact_score_row(
            profile,
            rank=rank,
            trajectory_refs=trajectory_refs,
            trajectory_lengths=trajectory_lengths,
        )
        for rank, (_, _, profile) in enumerate(sorted_scored[:DIRECT_TRAJECTORY_DIAGNOSTIC_TOP_N], start=1)
    ]
    scoring_mode = "metadata_lexical_v1" if metadata_signal_count else "fallback_text"
    return ranked, scoring_mode, compact_rows


def _direct_cutoff_minimums(
    *,
    gold_refs: set[str],
    gold_trajectory_ids: list[str],
    ranked_trajectory_ids: list[str],
    trajectory_refs: dict[str, set[str]],
) -> tuple[int | None, int | None]:
    min_gold_trajectories: int | None = None
    min_gold_refs: int | None = None
    for cutoff in _cutoff_values(len(ranked_trajectory_ids)):
        coverage = _compute_top_k_coverage(
            gold_refs=gold_refs,
            gold_trajectory_ids=gold_trajectory_ids,
            top_k_trajectory_ids=ranked_trajectory_ids[:cutoff],
            trajectory_refs=trajectory_refs,
        )
        if min_gold_trajectories is None and bool(gold_trajectory_ids) and (
            int(coverage["gold_trajectory_count_in_top_k"] or 0) == len(gold_trajectory_ids)
        ):
            min_gold_trajectories = cutoff
        if min_gold_refs is None and coverage["top_k_can_cover_all_gold_refs"] is True:
            min_gold_refs = cutoff
        if min_gold_trajectories is not None and min_gold_refs is not None:
            break
    return min_gold_trajectories, min_gold_refs


def _direct_cutoff_diagnostics_from_rows(
    compact_rows: list[dict[str, Any]],
    *,
    ranked_total_count: int,
    gold_refs: set[str],
    gold_trajectory_ids: list[str],
    trajectory_refs: dict[str, set[str]],
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for cutoff in OFFLINE_PARAMETER_CUTOFFS:
        if cutoff <= len(compact_rows):
            prefix = compact_rows[:cutoff]
            prefix_ids = [str(row.get("trajectory_id") or "") for row in prefix]
            coverage = _compute_top_k_coverage(
                gold_refs=gold_refs,
                gold_trajectory_ids=gold_trajectory_ids,
                top_k_trajectory_ids=prefix_ids,
                trajectory_refs=trajectory_refs,
            )
            diagnostics[str(cutoff)] = {
                "cutoff": cutoff,
                "observed_within_saved_cutoff": True,
                "not_observed_within_saved_cutoff": False,
                "candidate_count": ranked_total_count,
                "top_k_trajectory_ids": prefix_ids,
                "estimated_snapshot_count": sum(
                    int(row.get("snapshot_count_estimate") or 0) for row in prefix
                ),
                "estimated_source_ref_count": sum(
                    int(row.get("source_ref_count_estimate") or 0) for row in prefix
                ),
                "estimated_context_token_count": sum(
                    int(row.get("context_token_estimate") or 0) for row in prefix
                ),
                "gold_trajectory_count": len(gold_trajectory_ids),
                "gold_trajectory_count_in_cutoff": coverage["gold_trajectory_count_in_top_k"],
                "gold_trajectory_recall": coverage["gold_trajectory_recall_at_k"],
                "all_gold_trajectories_in_cutoff": bool(
                    gold_trajectory_ids
                    and int(coverage["gold_trajectory_count_in_top_k"] or 0) == len(gold_trajectory_ids)
                ),
                "top_k_gold_ref_coverage_count": coverage["top_k_gold_ref_coverage_count"],
                "top_k_gold_ref_coverage_rate": coverage["top_k_gold_ref_coverage_rate"],
                "top_k_can_cover_all_gold_refs": coverage["top_k_can_cover_all_gold_refs"],
            }
        else:
            diagnostics[str(cutoff)] = {
                "cutoff": cutoff,
                "observed_within_saved_cutoff": False,
                "not_observed_within_saved_cutoff": True,
                "saved_rank_count": len(compact_rows),
                "candidate_count": ranked_total_count,
            }
    return diagnostics


def _compute_direct_trajectory_ablation_fields(
    *,
    sample_id: str,
    question: str,
    gold_refs: set[str],
    gold_trajectory_ids: list[str],
    routed_top_k_trajectory_ids: list[str],
    selected_page_trajectory_ids: list[str],
    query_entities: list[str],
    query_facets: dict[str, list[str]],
    query_shape: dict[str, Any],
    top_k: int,
    sample_to_trajectories: dict[str, set[str]],
    trajectory_metadata: dict[str, dict[str, Any]],
    claims_by_trajectory: dict[str, list[dict[str, Any]]],
    trajectory_refs: dict[str, set[str]],
    trajectory_lengths: dict[str, int] | None = None,
) -> dict[str, Any]:
    ranked_ids, scoring_mode, compact_rows = _rank_direct_trajectories(
        sample_id=sample_id,
        question=question,
        query_entities=query_entities,
        query_facets=query_facets,
        query_shape=query_shape,
        sample_to_trajectories=sample_to_trajectories,
        trajectory_metadata=trajectory_metadata,
        claims_by_trajectory=claims_by_trajectory,
        trajectory_refs=trajectory_refs,
        trajectory_lengths=trajectory_lengths,
    )
    effective_k = max(0, int(top_k or len(routed_top_k_trajectory_ids) or 0))
    direct_top_k_ids = ranked_ids[:effective_k] if effective_k else []
    direct_coverage = _compute_top_k_coverage(
        gold_refs=gold_refs,
        gold_trajectory_ids=gold_trajectory_ids,
        top_k_trajectory_ids=direct_top_k_ids,
        trajectory_refs=trajectory_refs,
    )
    routed_coverage = _compute_top_k_coverage(
        gold_refs=gold_refs,
        gold_trajectory_ids=gold_trajectory_ids,
        top_k_trajectory_ids=routed_top_k_trajectory_ids,
        trajectory_refs=trajectory_refs,
    )
    direct_recall = direct_coverage["gold_trajectory_recall_at_k"]
    routed_recall = routed_coverage["gold_trajectory_recall_at_k"]
    recall_delta = (
        float(direct_recall) - float(routed_recall)
        if direct_recall is not None and routed_recall is not None
        else None
    )
    selected_page_set = set(selected_page_trajectory_ids)
    direct_top_k_set = set(direct_top_k_ids)
    routed_top_k_set = set(routed_top_k_trajectory_ids)
    top_k_overlap_count = len(direct_top_k_set & routed_top_k_set)
    top_k_union_count = len(direct_top_k_set | routed_top_k_set)
    direct_top_k_not_in_selected_page_universe = sorted(direct_top_k_set - selected_page_set)
    routed_universe_size = len(selected_page_set)
    gold_set = set(gold_trajectory_ids)
    selected_pages_cover_gold = bool(gold_set) and gold_set <= selected_page_set
    page_bottleneck = bool(
        recall_delta is not None
        and recall_delta > 0
        and gold_set
        and not selected_pages_cover_gold
    )
    if page_bottleneck:
        bottleneck = "page_routing"
    elif recall_delta is not None and recall_delta > 0:
        bottleneck = "trajectory_ranking"
    elif direct_recall is not None and routed_recall is not None:
        bottleneck = "metadata_signal_gap" if float(direct_recall) < 1.0 else "none"
    else:
        bottleneck = "unknown"
    min_gold_traj, min_gold_refs = _direct_cutoff_minimums(
        gold_refs=gold_refs,
        gold_trajectory_ids=gold_trajectory_ids,
        ranked_trajectory_ids=ranked_ids,
        trajectory_refs=trajectory_refs,
    )
    direct_cutoff_diagnostics = _direct_cutoff_diagnostics_from_rows(
        compact_rows,
        ranked_total_count=len(ranked_ids),
        gold_refs=gold_refs,
        gold_trajectory_ids=gold_trajectory_ids,
        trajectory_refs=trajectory_refs,
    )
    direct_top_k_compact_rows = compact_rows[:effective_k] if effective_k else []
    return {
        "direct_trajectory_diagnostic_available": True,
        "direct_trajectory_scoring_mode": scoring_mode,
        "direct_trajectory_candidate_universe_size": len(ranked_ids),
        "direct_trajectory_rank_limit": DIRECT_TRAJECTORY_DIAGNOSTIC_TOP_N,
        "direct_trajectory_ranked_total_count": len(ranked_ids),
        "direct_trajectory_ranked_rows_truncated": len(ranked_ids) > len(compact_rows),
        "direct_trajectory_ranked_rows_compact_top_n": compact_rows,
        "direct_trajectory_cutoff_diagnostics": direct_cutoff_diagnostics,
        "direct_trajectory_top_k_ids": direct_top_k_ids,
        "direct_gold_trajectory_count_in_top_k": direct_coverage["gold_trajectory_count_in_top_k"],
        "direct_gold_trajectory_recall_at_k": direct_recall,
        "direct_top_k_can_cover_all_gold_refs": direct_coverage["top_k_can_cover_all_gold_refs"],
        "direct_top_k_gold_ref_coverage_rate": direct_coverage["top_k_gold_ref_coverage_rate"],
        "direct_min_k_to_cover_all_gold_trajectories": min_gold_traj,
        "direct_min_k_to_cover_all_gold_refs": min_gold_refs,
        "direct_vs_page_routed_recall_delta": recall_delta,
        "direct_page_routing_bottleneck_suspected": page_bottleneck,
        "direct_trajectory_bottleneck": bottleneck,
        "routed_vs_direct_top_k_overlap_count": top_k_overlap_count,
        "routed_vs_direct_top_k_overlap_rate": _safe_rate(top_k_overlap_count, top_k_union_count)
        if top_k_union_count
        else None,
        "direct_top_k_not_in_selected_page_universe_count": len(direct_top_k_not_in_selected_page_universe),
        "direct_top_k_not_in_selected_page_universe_ids": direct_top_k_not_in_selected_page_universe,
        "direct_candidate_universe_to_routed_universe_ratio": (
            float(len(ranked_ids)) / float(routed_universe_size) if routed_universe_size else None
        ),
        "direct_top_k_estimated_snapshot_count": sum(
            int(row.get("snapshot_count_estimate") or 0) for row in direct_top_k_compact_rows
        ),
        "direct_top_k_estimated_source_ref_count": sum(
            int(row.get("source_ref_count_estimate") or 0) for row in direct_top_k_compact_rows
        ),
        "direct_top_k_estimated_context_token_count": sum(
            int(row.get("context_token_estimate") or 0) for row in direct_top_k_compact_rows
        ),
    }


def _selection_pool_ids_from_metadata(metadata: dict[str, Any]) -> list[str] | None:
    if "trajectory_selection_pool_ids" not in metadata:
        return None
    return [
        str(item)
        for item in list(metadata.get("trajectory_selection_pool_ids") or [])
        if str(item).strip()
    ]


def _sample_wiki_coverage_fields(
    *,
    sample_id: str,
    sample_to_trajectories: dict[str, set[str]],
    sample_to_pages: dict[str, set[str]],
    page_types: dict[str, str],
    page_to_trajectory_ids: dict[str, list[str]],
    page_metadata_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample_trajectory_ids = {
        trajectory_id for trajectory_id in sample_to_trajectories.get(sample_id, set()) if str(trajectory_id).strip()
    }
    non_index_trajectory_ids = {
        trajectory_id
        for page_id in sample_to_pages.get(sample_id, set())
        if page_types.get(page_id) != "index"
        for trajectory_id in page_to_trajectory_ids.get(page_id, [])
        if str(trajectory_id).strip()
    }
    index_only_trajectory_ids = sorted(sample_trajectory_ids - non_index_trajectory_ids)
    rescue_pages = [
        page_id
        for page_id in sample_to_pages.get(sample_id, set())
        if str(page_metadata_by_id.get(page_id, {}).get("wiki_rescue_reason") or "")
        == "post_plan_index_only_trajectory"
    ]
    rescue_trajectory_ids = {
        trajectory_id
        for page_id in rescue_pages
        for trajectory_id in page_to_trajectory_ids.get(page_id, [])
        if str(trajectory_id).strip()
    }
    return {
        "non_index_trajectory_coverage_rate": _safe_rate(
            len(sample_trajectory_ids & non_index_trajectory_ids),
            len(sample_trajectory_ids),
        ),
        "index_only_trajectory_count": len(index_only_trajectory_ids),
        "index_only_trajectory_ids": index_only_trajectory_ids,
        "wiki_rescue_page_count": len(rescue_pages),
        "wiki_rescue_trajectory_count": len(rescue_trajectory_ids),
    }


def _cutoff_values(max_value: int) -> list[int]:
    return list(range(1, max(0, int(max_value or 0)) + 1))


def _first_cutoff_with_flag(cutoff_diagnostics: dict[str, dict[str, Any]], flag_name: str) -> int | None:
    for cutoff in sorted((int(value) for value in cutoff_diagnostics), key=int):
        if bool(cutoff_diagnostics[str(cutoff)].get(flag_name)):
            return cutoff
    return None


def _compute_page_cutoff_fields(
    *,
    gold_page_ids: list[str],
    gold_trajectory_ids: list[str],
    top_t_page_ids: list[str],
    page_to_trajectory_ids: dict[str, list[str]],
) -> dict[str, Any]:
    gold_page_set = set(gold_page_ids)
    gold_trajectory_set = set(gold_trajectory_ids)
    diagnostics: dict[str, dict[str, Any]] = {}
    for cutoff in _cutoff_values(len(top_t_page_ids)):
        selected_page_ids = top_t_page_ids[:cutoff]
        selected_page_set = set(selected_page_ids)
        selected_page_trajectory_ids = sorted(
            {
                trajectory_id
                for page_id in selected_page_ids
                for trajectory_id in page_to_trajectory_ids.get(page_id, [])
                if str(trajectory_id).strip()
            }
        )
        selected_page_trajectory_set = set(selected_page_trajectory_ids)
        gold_pages_in_cutoff = [page_id for page_id in selected_page_ids if page_id in gold_page_set]
        page_recall = _safe_rate(len(gold_pages_in_cutoff), len(gold_page_set)) if gold_page_set else None
        all_gold_pages = bool(gold_page_set) and gold_page_set <= selected_page_set
        selected_pages_cover_all_gold_trajectories = (
            bool(gold_trajectory_set) and gold_trajectory_set <= selected_page_trajectory_set
        )
        diagnostics[str(cutoff)] = {
            "cutoff": cutoff,
            "selected_page_ids": selected_page_ids,
            "gold_pages_in_cutoff": gold_pages_in_cutoff,
            "gold_page_count": len(gold_page_set),
            "gold_pages_in_cutoff_count": len(gold_pages_in_cutoff),
            "gold_page_recall": page_recall,
            "all_gold_pages_in_cutoff": all_gold_pages,
            "selected_page_trajectory_ids": selected_page_trajectory_ids,
            "selected_page_trajectory_count": len(selected_page_trajectory_ids),
            "gold_trajectory_count": len(gold_trajectory_set),
            "all_gold_trajectories_in_selected_pages_at_cutoff": selected_pages_cover_all_gold_trajectories,
        }
    return {
        "min_t_pages_to_cover_all_gold_pages": _first_cutoff_with_flag(
            diagnostics, "all_gold_pages_in_cutoff"
        ),
        "min_t_pages_to_cover_all_gold_trajectories_via_pages": _first_cutoff_with_flag(
            diagnostics, "all_gold_trajectories_in_selected_pages_at_cutoff"
        ),
        "page_cutoff_diagnostics": diagnostics,
    }


def _compute_trajectory_cutoff_fields(
    *,
    gold_refs: set[str],
    gold_trajectory_ids: list[str],
    top_k_trajectory_ids: list[str],
    trajectory_refs: dict[str, set[str]],
) -> dict[str, Any]:
    diagnostics: dict[str, dict[str, Any]] = {}
    gold_trajectory_count = len(gold_trajectory_ids)
    for cutoff in _cutoff_values(len(top_k_trajectory_ids)):
        cutoff_trajectory_ids = top_k_trajectory_ids[:cutoff]
        coverage = _compute_top_k_coverage(
            gold_refs=gold_refs,
            gold_trajectory_ids=gold_trajectory_ids,
            top_k_trajectory_ids=cutoff_trajectory_ids,
            trajectory_refs=trajectory_refs,
        )
        diagnostics[str(cutoff)] = {
            "cutoff": cutoff,
            "top_k_trajectory_ids": cutoff_trajectory_ids,
            "gold_trajectory_count": gold_trajectory_count,
            "gold_trajectory_count_in_cutoff": coverage["gold_trajectory_count_in_top_k"],
            "gold_trajectory_recall": coverage["gold_trajectory_recall_at_k"],
            "all_gold_trajectories_in_cutoff": bool(
                gold_trajectory_count
                and int(coverage["gold_trajectory_count_in_top_k"] or 0) == gold_trajectory_count
            ),
            "top_k_gold_ref_coverage_count": coverage["top_k_gold_ref_coverage_count"],
            "top_k_gold_ref_coverage_rate": coverage["top_k_gold_ref_coverage_rate"],
            "top_k_can_cover_all_gold_refs": coverage["top_k_can_cover_all_gold_refs"],
        }
    return {
        "min_k_to_cover_all_gold_trajectories": _first_cutoff_with_flag(
            diagnostics, "all_gold_trajectories_in_cutoff"
        ),
        "min_k_to_cover_all_gold_refs": _first_cutoff_with_flag(
            diagnostics, "top_k_can_cover_all_gold_refs"
        ),
        "trajectory_cutoff_diagnostics": diagnostics,
    }


def _offline_ranked_page_rows(
    *,
    retrieval_metadata: dict[str, Any],
    top_t_page_ids: list[str],
    page_to_trajectory_ids: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mode = "missing"
    raw_rows = list(retrieval_metadata.get("page_ranked_rows_compact_top_n") or [])
    if raw_rows:
        mode = "compact_ranked_top_n"
        rows = [dict(row) for row in raw_rows if isinstance(row, dict)]
    else:
        legacy_rows = list(retrieval_metadata.get("page_ranked_rows") or [])
        if legacy_rows:
            mode = "legacy_page_ranked_rows"
            rows = [dict(row) for row in legacy_rows if isinstance(row, dict)]
        elif top_t_page_ids:
            mode = "legacy_selected_only"
            rows = [
                {
                    "rank": index,
                    "page_id": page_id,
                    "trajectory_ids": list(page_to_trajectory_ids.get(page_id, [])),
                    "trajectory_count": len(list(page_to_trajectory_ids.get(page_id, []))),
                }
                for index, page_id in enumerate(top_t_page_ids, start=1)
            ]
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        page_id = str(row.get("page_id") or "")
        if not page_id:
            continue
        trajectory_ids = [
            str(value)
            for value in list(row.get("trajectory_ids") or page_to_trajectory_ids.get(page_id, []))
            if str(value).strip()
        ]
        normalized.append(
            {
                **row,
                "rank": int(row.get("rank") or index),
                "page_id": page_id,
                "trajectory_ids": list(dict.fromkeys(trajectory_ids)),
                "trajectory_count": int(row.get("trajectory_count") or len(trajectory_ids)),
            }
        )
    saved_limit = int(retrieval_metadata.get("diagnostic_top_n_pages") or len(normalized) or 0)
    return normalized, {
        "offline_page_diagnostic_mode": mode,
        "offline_saved_page_rank_limit": saved_limit,
        "offline_page_rank_total_count": retrieval_metadata.get("page_ranked_total_count", len(normalized)),
        "offline_page_rank_truncated": retrieval_metadata.get(
            "page_ranked_rows_truncated",
            None if mode == "missing" else False,
        ),
    }


def _offline_ranked_trajectory_rows(
    *,
    retrieval_metadata: dict[str, Any],
    top_k_trajectory_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mode = "missing"
    raw_rows = list(retrieval_metadata.get("trajectory_ranked_rows_compact_top_n") or [])
    if raw_rows:
        mode = "compact_ranked_top_n"
        rows = [dict(row) for row in raw_rows if isinstance(row, dict)]
    else:
        legacy_rows = list(retrieval_metadata.get("trajectory_ranked_rows") or [])
        if legacy_rows:
            mode = "legacy_trajectory_ranked_rows"
            rows = [dict(row) for row in legacy_rows if isinstance(row, dict)]
        elif top_k_trajectory_ids:
            mode = "legacy_selected_only"
            rows = [
                {"rank": index, "trajectory_id": trajectory_id}
                for index, trajectory_id in enumerate(top_k_trajectory_ids, start=1)
            ]
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id:
            continue
        normalized.append({**row, "rank": int(row.get("rank") or index), "trajectory_id": trajectory_id})
    saved_limit = int(retrieval_metadata.get("diagnostic_top_n_trajectories") or len(normalized) or 0)
    return normalized, {
        "offline_trajectory_diagnostic_mode": mode,
        "offline_saved_trajectory_rank_limit": saved_limit,
        "offline_trajectory_rank_total_count": retrieval_metadata.get(
            "trajectory_ranked_total_count", len(normalized)
        ),
        "offline_trajectory_rank_truncated": retrieval_metadata.get(
            "trajectory_ranked_rows_truncated",
            None if mode == "missing" else False,
        ),
    }


def _compute_offline_parameter_fields(
    *,
    retrieval_metadata: dict[str, Any],
    gold_refs: set[str],
    gold_page_ids: list[str],
    gold_trajectory_ids: list[str],
    top_t_page_ids: list[str],
    top_k_trajectory_ids: list[str],
    page_to_trajectory_ids: dict[str, list[str]],
    trajectory_refs: dict[str, set[str]],
) -> dict[str, Any]:
    page_rows, page_meta = _offline_ranked_page_rows(
        retrieval_metadata=retrieval_metadata,
        top_t_page_ids=top_t_page_ids,
        page_to_trajectory_ids=page_to_trajectory_ids,
    )
    trajectory_rows, trajectory_meta = _offline_ranked_trajectory_rows(
        retrieval_metadata=retrieval_metadata,
        top_k_trajectory_ids=top_k_trajectory_ids,
    )
    gold_page_set = set(gold_page_ids)
    gold_trajectory_set = set(gold_trajectory_ids)
    page_diagnostics: dict[str, dict[str, Any]] = {}
    trajectory_diagnostics: dict[str, dict[str, Any]] = {}
    for cutoff in OFFLINE_PARAMETER_CUTOFFS:
        if cutoff <= len(page_rows):
            prefix = page_rows[:cutoff]
            selected_page_ids = [str(row.get("page_id") or "") for row in prefix]
            selected_page_set = set(selected_page_ids)
            selected_trajectory_ids = list(
                dict.fromkeys(
                    trajectory_id
                    for row in prefix
                    for trajectory_id in list(row.get("trajectory_ids") or [])
                    if str(trajectory_id).strip()
                )
            )
            selected_trajectory_set = set(selected_trajectory_ids)
            covered_refs = set().union(
                *(trajectory_refs.get(trajectory_id, set()) for trajectory_id in selected_trajectory_ids)
            ) & gold_refs
            gold_pages_in_cutoff = [page_id for page_id in selected_page_ids if page_id in gold_page_set]
            page_diagnostics[str(cutoff)] = {
                "cutoff": cutoff,
                "observed_within_saved_cutoff": True,
                "not_observed_within_saved_cutoff": False,
                "selected_page_ids": selected_page_ids,
                "selected_page_trajectory_count": len(selected_trajectory_ids),
                "gold_page_count": len(gold_page_set),
                "gold_pages_in_cutoff_count": len(gold_pages_in_cutoff),
                "gold_page_recall": _safe_rate(len(gold_pages_in_cutoff), len(gold_page_set))
                if gold_page_set
                else None,
                "all_gold_pages_in_cutoff": bool(gold_page_set) and gold_page_set <= selected_page_set,
                "gold_trajectory_count": len(gold_trajectory_set),
                "all_gold_trajectories_in_selected_pages_at_cutoff": (
                    bool(gold_trajectory_set) and gold_trajectory_set <= selected_trajectory_set
                ),
                "page_universe_gold_ref_coverage_count": len(covered_refs),
                "page_universe_gold_ref_coverage_rate": _safe_rate(len(covered_refs), len(gold_refs))
                if gold_refs
                else None,
                "page_universe_can_cover_all_gold_refs": bool(gold_refs) and gold_refs <= covered_refs,
            }
        else:
            page_diagnostics[str(cutoff)] = {
                "cutoff": cutoff,
                "observed_within_saved_cutoff": False,
                "not_observed_within_saved_cutoff": True,
                "saved_rank_count": len(page_rows),
            }
        if cutoff <= len(trajectory_rows):
            prefix_ids = [str(row.get("trajectory_id") or "") for row in trajectory_rows[:cutoff]]
            coverage = _compute_top_k_coverage(
                gold_refs=gold_refs,
                gold_trajectory_ids=gold_trajectory_ids,
                top_k_trajectory_ids=prefix_ids,
                trajectory_refs=trajectory_refs,
            )
            trajectory_diagnostics[str(cutoff)] = {
                "cutoff": cutoff,
                "observed_within_saved_cutoff": True,
                "not_observed_within_saved_cutoff": False,
                "top_k_trajectory_ids": prefix_ids,
                "gold_trajectory_count": len(gold_trajectory_ids),
                "gold_trajectory_count_in_cutoff": coverage["gold_trajectory_count_in_top_k"],
                "gold_trajectory_recall": coverage["gold_trajectory_recall_at_k"],
                "all_gold_trajectories_in_cutoff": bool(
                    gold_trajectory_ids
                    and int(coverage["gold_trajectory_count_in_top_k"] or 0) == len(gold_trajectory_ids)
                ),
                "top_k_gold_ref_coverage_count": coverage["top_k_gold_ref_coverage_count"],
                "top_k_gold_ref_coverage_rate": coverage["top_k_gold_ref_coverage_rate"],
                "top_k_can_cover_all_gold_refs": coverage["top_k_can_cover_all_gold_refs"],
            }
        else:
            trajectory_diagnostics[str(cutoff)] = {
                "cutoff": cutoff,
                "observed_within_saved_cutoff": False,
                "not_observed_within_saved_cutoff": True,
                "saved_rank_count": len(trajectory_rows),
            }
    return {
        **page_meta,
        **trajectory_meta,
        "offline_min_t_saved_to_cover_all_gold_pages": _first_cutoff_with_flag(
            page_diagnostics, "all_gold_pages_in_cutoff"
        ),
        "offline_min_t_saved_to_cover_all_gold_trajectories": _first_cutoff_with_flag(
            page_diagnostics, "all_gold_trajectories_in_selected_pages_at_cutoff"
        ),
        "offline_min_k_saved_to_cover_all_gold_trajectories": _first_cutoff_with_flag(
            trajectory_diagnostics, "all_gold_trajectories_in_cutoff"
        ),
        "offline_min_k_saved_to_cover_all_gold_refs": _first_cutoff_with_flag(
            trajectory_diagnostics, "top_k_can_cover_all_gold_refs"
        ),
        "offline_page_cutoff_diagnostics": page_diagnostics,
        "offline_trajectory_cutoff_diagnostics": trajectory_diagnostics,
    }


def _shape_bucket_keys(row: dict[str, Any]) -> list[str]:
    shape = dict(row.get("query_shape") or {})
    buckets: list[str] = []
    for key in ["list_like", "count_like", "multi_entity", "comparison_like"]:
        if shape.get(key):
            buckets.append(key)
    item_family = str(shape.get("item_family") or "").strip()
    if item_family:
        buckets.append(f"item_family:{item_family}")
    return buckets or ["single_fact"]


def _aggregate_direct_trajectory_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed_rows = [row for row in rows if row.get("direct_gold_trajectory_recall_at_k") is not None]
    direct_recall_values = [float(row["direct_gold_trajectory_recall_at_k"]) for row in observed_rows]
    routed_recall_values = [
        float(row["gold_trajectory_recall_at_k"])
        for row in rows
        if row.get("gold_trajectory_recall_at_k") is not None
    ]
    direct_ref_coverage_values = [
        float(row["direct_top_k_gold_ref_coverage_rate"])
        for row in rows
        if row.get("direct_top_k_gold_ref_coverage_rate") is not None
    ]
    deltas = [
        float(row["direct_vs_page_routed_recall_delta"])
        for row in rows
        if row.get("direct_vs_page_routed_recall_delta") is not None
    ]
    overlap_rates = [
        float(row["routed_vs_direct_top_k_overlap_rate"])
        for row in rows
        if row.get("routed_vs_direct_top_k_overlap_rate") is not None
    ]
    direct_candidate_sizes = [
        int(row.get("direct_trajectory_candidate_universe_size") or 0)
        for row in rows
        if row.get("direct_trajectory_candidate_universe_size") is not None
    ]
    routed_universe_sizes = [
        int(row.get("selected_page_universe_size") or 0)
        for row in rows
        if row.get("selected_page_universe_size") is not None
    ]
    direct_to_routed_ratios = [
        float(row["direct_candidate_universe_to_routed_universe_ratio"])
        for row in rows
        if row.get("direct_candidate_universe_to_routed_universe_ratio") is not None
    ]
    direct_context_tokens = [
        int(row.get("direct_top_k_estimated_context_token_count") or 0)
        for row in rows
        if row.get("direct_top_k_estimated_context_token_count") is not None
    ]
    not_in_page_universe = [
        int(row.get("direct_top_k_not_in_selected_page_universe_count") or 0)
        for row in rows
        if row.get("direct_top_k_not_in_selected_page_universe_count") is not None
    ]
    gold_rows = [row for row in rows if int(row.get("gold_trajectory_count") or 0) > 0]
    all_gold_direct = sum(
        1
        for row in gold_rows
        if int(row.get("direct_gold_trajectory_count_in_top_k") or 0)
        == int(row.get("gold_trajectory_count") or 0)
    )
    direct_ref_rows = [row for row in rows if row.get("direct_top_k_can_cover_all_gold_refs") is not None]
    direct_refs_all = sum(1 for row in direct_ref_rows if row.get("direct_top_k_can_cover_all_gold_refs") is True)
    bottleneck_count = sum(1 for row in rows if row.get("direct_page_routing_bottleneck_suspected") is True)
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for bucket in _shape_bucket_keys(row):
            bucket_rows[bucket].append(row)

    def _bucket_summary(bucket_values: list[dict[str, Any]]) -> dict[str, Any]:
        bucket_direct = [
            float(row["direct_gold_trajectory_recall_at_k"])
            for row in bucket_values
            if row.get("direct_gold_trajectory_recall_at_k") is not None
        ]
        bucket_delta = [
            float(row["direct_vs_page_routed_recall_delta"])
            for row in bucket_values
            if row.get("direct_vs_page_routed_recall_delta") is not None
        ]
        bucket_gold = [row for row in bucket_values if int(row.get("gold_trajectory_count") or 0) > 0]
        return {
            "query_count": len(bucket_values),
            "gold_trajectory_query_count": len(bucket_gold),
            "mean_direct_gold_trajectory_recall_at_k": _safe_mean(bucket_direct),
            "mean_direct_vs_page_routed_recall_delta": _safe_mean(bucket_delta),
            "page_routing_bottleneck_suspected_count": sum(
                1 for row in bucket_values if row.get("direct_page_routing_bottleneck_suspected") is True
            ),
        }

    return {
        "query_count": len(rows),
        "gold_trajectory_query_count": len(gold_rows),
        "mean_current_page_routed_gold_trajectory_recall_at_k": _safe_mean(routed_recall_values),
        "mean_direct_gold_trajectory_recall_at_k": _safe_mean(direct_recall_values),
        "direct_all_gold_trajectories_in_top_k_rate": (
            _safe_rate(all_gold_direct, len(gold_rows)) if gold_rows else None
        ),
        "direct_top_k_can_cover_all_gold_refs_rate": (
            _safe_rate(direct_refs_all, len(direct_ref_rows)) if direct_ref_rows else None
        ),
        "mean_direct_top_k_gold_ref_coverage_rate": _safe_mean(direct_ref_coverage_values),
        "mean_direct_vs_page_routed_recall_delta": _safe_mean(deltas),
        "mean_direct_candidate_universe_size": _safe_mean(direct_candidate_sizes),
        "mean_routed_candidate_universe_size": _safe_mean(routed_universe_sizes),
        "mean_direct_candidate_universe_to_routed_universe_ratio": _safe_mean(direct_to_routed_ratios),
        "mean_routed_vs_direct_top_k_overlap_rate": _safe_mean(overlap_rates),
        "mean_direct_top_k_not_in_selected_page_universe_count": _safe_mean(not_in_page_universe),
        "mean_direct_top_k_estimated_context_token_count": _safe_mean(direct_context_tokens),
        "page_routing_bottleneck_suspected_count": bottleneck_count,
        "page_routing_bottleneck_suspected_rate": _safe_rate(bottleneck_count, len(rows)) if rows else None,
        "by_query_shape": {bucket: _bucket_summary(values) for bucket, values in sorted(bucket_rows.items())},
    }


def _aggregate_direct_trajectory_cutoff_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for cutoff in OFFLINE_PARAMETER_CUTOFFS:
        cutoff_rows = [
            dict(dict(row.get("direct_trajectory_cutoff_diagnostics") or {}).get(str(cutoff)) or {})
            for row in rows
            if str(cutoff) in dict(row.get("direct_trajectory_cutoff_diagnostics") or {})
        ]
        observed_rows = [row for row in cutoff_rows if row.get("observed_within_saved_cutoff") is True]
        not_observed_count = sum(
            1 for row in cutoff_rows if row.get("not_observed_within_saved_cutoff") is True
        )
        gold_trajectory_rows = [
            row for row in observed_rows if int(row.get("gold_trajectory_count") or 0) > 0
        ]
        gold_ref_rows = [
            row for row in observed_rows if row.get("top_k_can_cover_all_gold_refs") is not None
        ]
        token_counts: list[int] = []
        source_ref_counts: list[int] = []
        snapshot_counts: list[int] = []
        for cutoff_row in observed_rows:
            token_counts.append(int(cutoff_row.get("estimated_context_token_count") or 0))
            source_ref_counts.append(int(cutoff_row.get("estimated_source_ref_count") or 0))
            snapshot_counts.append(int(cutoff_row.get("estimated_snapshot_count") or 0))
        output[str(cutoff)] = {
            "cutoff": cutoff,
            "observed_query_count": len(observed_rows),
            "not_observed_query_count": not_observed_count,
            "all_gold_trajectories_rate": _safe_rate(
                sum(
                    1
                    for row in gold_trajectory_rows
                    if row.get("all_gold_trajectories_in_cutoff") is True
                ),
                len(gold_trajectory_rows),
            )
            if gold_trajectory_rows
            else None,
            "mean_gold_trajectory_recall": _safe_mean(
                [
                    float(row["gold_trajectory_recall"])
                    for row in gold_trajectory_rows
                    if row.get("gold_trajectory_recall") is not None
                ]
            ),
            "top_k_can_cover_all_gold_refs_rate": _safe_rate(
                sum(1 for row in gold_ref_rows if row.get("top_k_can_cover_all_gold_refs") is True),
                len(gold_ref_rows),
            )
            if gold_ref_rows
            else None,
            "mean_top_k_gold_ref_coverage_rate": _safe_mean(
                [
                    float(row["top_k_gold_ref_coverage_rate"])
                    for row in gold_ref_rows
                    if row.get("top_k_gold_ref_coverage_rate") is not None
                ]
            ),
            "mean_estimated_context_token_count": _safe_mean(token_counts),
            "mean_estimated_source_ref_count": _safe_mean(source_ref_counts),
            "mean_estimated_snapshot_count": _safe_mean(snapshot_counts),
        }
    return output


def _build_direct_vs_routed_retrieval_diagnostics(
    query_outcomes: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_direct = _aggregate_direct_trajectory_ablation(query_outcomes)
    failed_direct = _aggregate_direct_trajectory_ablation(failed_rows)
    all_cutoffs = _aggregate_direct_trajectory_cutoff_rows(query_outcomes)
    failed_cutoffs = _aggregate_direct_trajectory_cutoff_rows(failed_rows)
    return {
        "diagnostic_mode": "metadata_direct_retrieval_ablation",
        "rank_limit": DIRECT_TRAJECTORY_DIAGNOSTIC_TOP_N,
        "query_count": len(query_outcomes),
        "failed_query_count": len(failed_rows),
        "all_queries": all_direct,
        "failed_queries": failed_direct,
        "direct_cutoffs": {
            str(cutoff): {
                "all_queries": all_cutoffs.get(str(cutoff), {}),
                "failed_queries": failed_cutoffs.get(str(cutoff), {}),
            }
            for cutoff in OFFLINE_PARAMETER_CUTOFFS
        },
        "by_query_shape": dict(all_direct.get("by_query_shape") or {}),
    }


def _compute_trajectory_length_query_fields(
    *,
    gold_trajectory_ids: list[str],
    gold_snapshot_ids: list[str],
    configured_m: int | None,
    trajectory_lengths: dict[str, int],
    snapshot_versions: dict[str, int],
    snapshot_rank_in_trajectory: dict[str, int],
) -> dict[str, Any]:
    gold_trajectory_lengths = {
        trajectory_id: int(trajectory_lengths.get(trajectory_id, 0) or 0)
        for trajectory_id in gold_trajectory_ids
    }
    gold_snapshot_ranks = {
        snapshot_id: int(snapshot_rank_in_trajectory[snapshot_id])
        for snapshot_id in gold_snapshot_ids
        if snapshot_id in snapshot_rank_in_trajectory
    }
    gold_snapshot_versions = {
        snapshot_id: int(snapshot_versions[snapshot_id])
        for snapshot_id in gold_snapshot_ids
        if snapshot_id in snapshot_versions
    }
    configured_m_int = int(configured_m or 0)
    gold_lengths = list(gold_trajectory_lengths.values())
    at_limit_count = (
        sum(1 for value in gold_lengths if configured_m_int > 0 and int(value) >= configured_m_int)
        if gold_lengths
        else 0
    )
    return {
        "gold_trajectory_lengths": gold_trajectory_lengths,
        "gold_trajectory_length_max": max(gold_lengths) if gold_lengths else None,
        "gold_trajectory_at_m_limit_count": at_limit_count,
        "gold_trajectory_at_m_limit_rate": _safe_rate(at_limit_count, len(gold_lengths)) if gold_lengths else None,
        "gold_snapshot_ranks": gold_snapshot_ranks,
        "gold_snapshot_versions": gold_snapshot_versions,
        "max_gold_snapshot_rank_required": max(gold_snapshot_ranks.values()) if gold_snapshot_ranks else None,
        "max_gold_snapshot_version_required": max(gold_snapshot_versions.values()) if gold_snapshot_versions else None,
    }


def _aggregate_trajectory_length_query_rows(
    rows: list[dict[str, Any]],
    *,
    configured_m: int,
) -> dict[str, Any]:
    gold_lengths = [
        int(value)
        for row in rows
        for value in dict(row.get("gold_trajectory_lengths") or {}).values()
        if value is not None
    ]
    required_ranks = [
        int(row["max_gold_snapshot_rank_required"])
        for row in rows
        if row.get("max_gold_snapshot_rank_required") is not None
    ]
    required_versions = [
        int(row["max_gold_snapshot_version_required"])
        for row in rows
        if row.get("max_gold_snapshot_version_required") is not None
    ]
    at_limit_count = sum(1 for value in gold_lengths if configured_m > 0 and value >= configured_m)
    return {
        "query_count": len(rows),
        "gold_trajectory_length_stats": _length_stats(gold_lengths),
        "gold_trajectory_at_m_limit_count": at_limit_count,
        "gold_trajectory_at_m_limit_rate": _safe_rate(at_limit_count, len(gold_lengths)) if gold_lengths else None,
        "max_gold_snapshot_rank_required_stats": _length_stats(required_ranks),
        "max_gold_snapshot_version_required_stats": _length_stats(required_versions),
    }


def _build_trajectory_length_diagnostics(
    *,
    configured_m: int,
    trajectory_lengths: dict[str, int],
    query_outcomes: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_lengths = list(trajectory_lengths.values())
    all_at_limit_count = sum(1 for value in all_lengths if configured_m > 0 and int(value) >= configured_m)
    all_stats = {
        **_length_stats(all_lengths),
        "at_m_limit_count": all_at_limit_count,
        "at_m_limit_rate": _safe_rate(all_at_limit_count, len(all_lengths)) if all_lengths else None,
    }
    all_query_stats = _aggregate_trajectory_length_query_rows(query_outcomes, configured_m=configured_m)
    failed_query_stats = _aggregate_trajectory_length_query_rows(failed_rows, configured_m=configured_m)

    all_length_p95 = all_stats.get("p95")
    gold_required_rank_p95 = dict(
        all_query_stats.get("max_gold_snapshot_rank_required_stats") or {}
    ).get("p95")
    all_limit_rate = all_stats.get("at_m_limit_rate")
    gold_limit_rate = all_query_stats.get("gold_trajectory_at_m_limit_rate")
    pressure_detected = bool(
        (all_limit_rate is not None and float(all_limit_rate) >= 0.10)
        or (gold_limit_rate is not None and float(gold_limit_rate) >= 0.10)
    )
    sufficient_gold_data = gold_required_rank_p95 is not None
    overprovisioned_possible = bool(
        configured_m > 0
        and all_length_p95 is not None
        and gold_required_rank_p95 is not None
        and float(all_length_p95) <= 0.5 * configured_m
        and float(gold_required_rank_p95) <= 0.5 * configured_m
        and (all_limit_rate is None or float(all_limit_rate) <= 0.02)
    )
    candidate_lower_m = None
    if configured_m > 0 and not pressure_detected and all_length_p95 is not None:
        required = max(float(all_length_p95), float(gold_required_rank_p95 or 0.0))
        candidate_lower_m = min(configured_m, max(1, int(math.ceil(required))))
    if pressure_detected:
        interpretation = "m_pressure"
    elif overprovisioned_possible:
        interpretation = "m_overprovisioned_possible"
    elif not sufficient_gold_data:
        interpretation = "insufficient_gold_data"
    else:
        interpretation = "m_looks_reasonable"
    return {
        "configured_m": configured_m,
        "all_trajectories": all_stats,
        "all_queries_gold_trajectories": all_query_stats,
        "failed_queries_gold_trajectories": failed_query_stats,
        "all_queries_gold_snapshot_requirements": {
            "max_gold_snapshot_rank_required_stats": all_query_stats["max_gold_snapshot_rank_required_stats"],
            "max_gold_snapshot_version_required_stats": all_query_stats[
                "max_gold_snapshot_version_required_stats"
            ],
        },
        "failed_queries_gold_snapshot_requirements": {
            "max_gold_snapshot_rank_required_stats": failed_query_stats[
                "max_gold_snapshot_rank_required_stats"
            ],
            "max_gold_snapshot_version_required_stats": failed_query_stats[
                "max_gold_snapshot_version_required_stats"
            ],
        },
        "recommendations": {
            "m_pressure_detected": pressure_detected,
            "m_overprovisioned_possible": overprovisioned_possible,
            "candidate_lower_m_at_95": candidate_lower_m,
            "interpretation": interpretation,
        },
    }


def _aggregate_page_cutoff_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cutoff_keys = sorted(
        {
            int(cutoff)
            for row in rows
            for cutoff in dict(row.get("page_cutoff_diagnostics") or {})
        }
    )
    aggregated: dict[str, dict[str, Any]] = {}
    for cutoff in cutoff_keys:
        cutoff_rows = [
            dict(dict(row.get("page_cutoff_diagnostics") or {}).get(str(cutoff)) or {})
            for row in rows
            if str(cutoff) in dict(row.get("page_cutoff_diagnostics") or {})
        ]
        page_recall_values = [
            float(item["gold_page_recall"])
            for item in cutoff_rows
            if item.get("gold_page_recall") is not None
        ]
        gold_page_rows = [item for item in cutoff_rows if int(item.get("gold_page_count", 0) or 0) > 0]
        gold_trajectory_rows = [
            item for item in cutoff_rows if int(item.get("gold_trajectory_count", 0) or 0) > 0
        ]
        selected_page_trajectory_counts = [
            int(item.get("selected_page_trajectory_count", 0) or 0)
            for item in cutoff_rows
        ]
        aggregated[str(cutoff)] = {
            "cutoff": cutoff,
            "observed_query_count": len(cutoff_rows),
            "gold_page_query_count": len(gold_page_rows),
            "gold_trajectory_query_count": len(gold_trajectory_rows),
            "all_gold_pages_rate": _safe_rate(
                sum(1 for item in gold_page_rows if item.get("all_gold_pages_in_cutoff") is True),
                len(gold_page_rows),
            ),
            "mean_page_recall": _safe_mean(page_recall_values),
            "selected_pages_cover_all_gold_trajectories_rate": _safe_rate(
                sum(
                    1
                    for item in gold_trajectory_rows
                    if item.get("all_gold_trajectories_in_selected_pages_at_cutoff") is True
                ),
                len(gold_trajectory_rows),
            ),
            "mean_selected_page_trajectory_count": _safe_mean(selected_page_trajectory_counts),
        }
    return aggregated


def _aggregate_trajectory_cutoff_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cutoff_keys = sorted(
        {
            int(cutoff)
            for row in rows
            for cutoff in dict(row.get("trajectory_cutoff_diagnostics") or {})
        }
    )
    aggregated: dict[str, dict[str, Any]] = {}
    for cutoff in cutoff_keys:
        cutoff_rows = [
            dict(dict(row.get("trajectory_cutoff_diagnostics") or {}).get(str(cutoff)) or {})
            for row in rows
            if str(cutoff) in dict(row.get("trajectory_cutoff_diagnostics") or {})
        ]
        gold_trajectory_rows = [
            item for item in cutoff_rows if int(item.get("gold_trajectory_count", 0) or 0) > 0
        ]
        gold_ref_rows = [
            item for item in cutoff_rows if item.get("top_k_can_cover_all_gold_refs") is not None
        ]
        recall_values = [
            float(item["gold_trajectory_recall"])
            for item in gold_trajectory_rows
            if item.get("gold_trajectory_recall") is not None
        ]
        ref_coverage_values = [
            float(item["top_k_gold_ref_coverage_rate"])
            for item in gold_ref_rows
            if item.get("top_k_gold_ref_coverage_rate") is not None
        ]
        aggregated[str(cutoff)] = {
            "cutoff": cutoff,
            "observed_query_count": len(cutoff_rows),
            "gold_trajectory_query_count": len(gold_trajectory_rows),
            "gold_ref_query_count": len(gold_ref_rows),
            "all_gold_trajectories_rate": _safe_rate(
                sum(1 for item in gold_trajectory_rows if item.get("all_gold_trajectories_in_cutoff") is True),
                len(gold_trajectory_rows),
            ),
            "mean_gold_trajectory_recall": _safe_mean(recall_values),
            "top_k_can_cover_all_gold_refs_rate": _safe_rate(
                sum(1 for item in gold_ref_rows if item.get("top_k_can_cover_all_gold_refs") is True),
                len(gold_ref_rows),
            ),
            "mean_top_k_gold_ref_coverage_rate": _safe_mean(ref_coverage_values),
        }
    return aggregated


def _suggested_cutoff(
    cutoff_rows: dict[str, dict[str, Any]],
    metric_names: list[str],
    threshold: float,
) -> int | None:
    for cutoff in sorted((int(value) for value in cutoff_rows), key=int):
        row = dict(cutoff_rows.get(str(cutoff), {}))
        if all(row.get(metric_name) is not None and float(row.get(metric_name)) >= threshold for metric_name in metric_names):
            return cutoff
    return None


def _build_cutoff_diagnostics(
    query_outcomes: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_page = _aggregate_page_cutoff_rows(query_outcomes)
    failed_page = _aggregate_page_cutoff_rows(failed_rows)
    all_trajectory = _aggregate_trajectory_cutoff_rows(query_outcomes)
    failed_trajectory = _aggregate_trajectory_cutoff_rows(failed_rows)

    page_cutoffs = {
        cutoff: {
            "all_queries": all_page.get(cutoff, {}),
            "failed_queries": failed_page.get(cutoff, {}),
        }
        for cutoff in sorted(set(all_page) | set(failed_page), key=lambda value: int(value))
    }
    trajectory_cutoffs = {
        cutoff: {
            "all_queries": all_trajectory.get(cutoff, {}),
            "failed_queries": failed_trajectory.get(cutoff, {}),
        }
        for cutoff in sorted(set(all_trajectory) | set(failed_trajectory), key=lambda value: int(value))
    }
    return {
        "page_cutoffs": page_cutoffs,
        "trajectory_cutoffs": trajectory_cutoffs,
        "suggested_cutoffs": {
            "page_95": _suggested_cutoff(
                all_page,
                ["all_gold_pages_rate", "selected_pages_cover_all_gold_trajectories_rate"],
                0.95,
            ),
            "page_99": _suggested_cutoff(
                all_page,
                ["all_gold_pages_rate", "selected_pages_cover_all_gold_trajectories_rate"],
                0.99,
            ),
            "trajectory_95": _suggested_cutoff(
                all_trajectory,
                ["all_gold_trajectories_rate", "top_k_can_cover_all_gold_refs_rate"],
                0.95,
            ),
            "trajectory_99": _suggested_cutoff(
                all_trajectory,
                ["all_gold_trajectories_rate", "top_k_can_cover_all_gold_refs_rate"],
                0.99,
            ),
        },
    }


def _aggregate_offline_page_cutoff_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for cutoff in OFFLINE_PARAMETER_CUTOFFS:
        cutoff_rows = [
            dict(dict(row.get("offline_page_cutoff_diagnostics") or {}).get(str(cutoff)) or {})
            for row in rows
            if str(cutoff) in dict(row.get("offline_page_cutoff_diagnostics") or {})
        ]
        observed_rows = [row for row in cutoff_rows if row.get("observed_within_saved_cutoff") is True]
        gold_page_rows = [row for row in observed_rows if int(row.get("gold_page_count", 0) or 0) > 0]
        gold_trajectory_rows = [
            row for row in observed_rows if int(row.get("gold_trajectory_count", 0) or 0) > 0
        ]
        gold_ref_rows = [
            row for row in observed_rows if row.get("page_universe_gold_ref_coverage_rate") is not None
        ]
        output[str(cutoff)] = {
            "cutoff": cutoff,
            "observed_query_count": len(observed_rows),
            "not_observed_query_count": sum(
                1 for row in cutoff_rows if row.get("not_observed_within_saved_cutoff") is True
            ),
            "all_gold_pages_rate": _safe_rate(
                sum(1 for row in gold_page_rows if row.get("all_gold_pages_in_cutoff") is True),
                len(gold_page_rows),
            ),
            "mean_gold_page_recall": _safe_mean(
                [
                    float(row["gold_page_recall"])
                    for row in gold_page_rows
                    if row.get("gold_page_recall") is not None
                ]
            ),
            "selected_pages_cover_all_gold_trajectories_rate": _safe_rate(
                sum(
                    1
                    for row in gold_trajectory_rows
                    if row.get("all_gold_trajectories_in_selected_pages_at_cutoff") is True
                ),
                len(gold_trajectory_rows),
            ),
            "page_universe_can_cover_all_gold_refs_rate": _safe_rate(
                sum(1 for row in gold_ref_rows if row.get("page_universe_can_cover_all_gold_refs") is True),
                len(gold_ref_rows),
            ),
            "mean_page_universe_gold_ref_coverage_rate": _safe_mean(
                [
                    float(row["page_universe_gold_ref_coverage_rate"])
                    for row in gold_ref_rows
                    if row.get("page_universe_gold_ref_coverage_rate") is not None
                ]
            ),
            "mean_selected_page_trajectory_count": _safe_mean(
                [
                    int(row.get("selected_page_trajectory_count") or 0)
                    for row in observed_rows
                ]
            ),
        }
    return output


def _aggregate_offline_trajectory_cutoff_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for cutoff in OFFLINE_PARAMETER_CUTOFFS:
        cutoff_rows = [
            dict(dict(row.get("offline_trajectory_cutoff_diagnostics") or {}).get(str(cutoff)) or {})
            for row in rows
            if str(cutoff) in dict(row.get("offline_trajectory_cutoff_diagnostics") or {})
        ]
        observed_rows = [row for row in cutoff_rows if row.get("observed_within_saved_cutoff") is True]
        gold_trajectory_rows = [
            row for row in observed_rows if int(row.get("gold_trajectory_count", 0) or 0) > 0
        ]
        gold_ref_rows = [
            row for row in observed_rows if row.get("top_k_can_cover_all_gold_refs") is not None
        ]
        output[str(cutoff)] = {
            "cutoff": cutoff,
            "observed_query_count": len(observed_rows),
            "not_observed_query_count": sum(
                1 for row in cutoff_rows if row.get("not_observed_within_saved_cutoff") is True
            ),
            "all_gold_trajectories_rate": _safe_rate(
                sum(1 for row in gold_trajectory_rows if row.get("all_gold_trajectories_in_cutoff") is True),
                len(gold_trajectory_rows),
            ),
            "mean_gold_trajectory_recall": _safe_mean(
                [
                    float(row["gold_trajectory_recall"])
                    for row in gold_trajectory_rows
                    if row.get("gold_trajectory_recall") is not None
                ]
            ),
            "top_k_can_cover_all_gold_refs_rate": _safe_rate(
                sum(1 for row in gold_ref_rows if row.get("top_k_can_cover_all_gold_refs") is True),
                len(gold_ref_rows),
            ),
            "mean_top_k_gold_ref_coverage_rate": _safe_mean(
                [
                    float(row["top_k_gold_ref_coverage_rate"])
                    for row in gold_ref_rows
                    if row.get("top_k_gold_ref_coverage_rate") is not None
                ]
            ),
        }
    return output


def _build_offline_parameter_diagnostics(
    query_outcomes: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    page_modes = Counter(str(row.get("offline_page_diagnostic_mode") or "missing") for row in query_outcomes)
    trajectory_modes = Counter(
        str(row.get("offline_trajectory_diagnostic_mode") or "missing") for row in query_outcomes
    )
    all_page = _aggregate_offline_page_cutoff_rows(query_outcomes)
    failed_page = _aggregate_offline_page_cutoff_rows(failed_rows)
    all_trajectory = _aggregate_offline_trajectory_cutoff_rows(query_outcomes)
    failed_trajectory = _aggregate_offline_trajectory_cutoff_rows(failed_rows)
    page_cutoffs = {
        cutoff: {
            "all_queries": all_page.get(cutoff, {}),
            "failed_queries": failed_page.get(cutoff, {}),
        }
        for cutoff in (str(value) for value in OFFLINE_PARAMETER_CUTOFFS)
    }
    trajectory_cutoffs = {
        cutoff: {
            "all_queries": all_trajectory.get(cutoff, {}),
            "failed_queries": failed_trajectory.get(cutoff, {}),
        }
        for cutoff in (str(value) for value in OFFLINE_PARAMETER_CUTOFFS)
    }
    return {
        "diagnostic_mode": (
            "compact_ranked_top_n"
            if page_modes.get("compact_ranked_top_n") or trajectory_modes.get("compact_ranked_top_n")
            else "legacy_selected_only"
        ),
        "query_count": len(query_outcomes),
        "failed_query_count": len(failed_rows),
        "page_diagnostic_mode_counts": dict(page_modes),
        "trajectory_diagnostic_mode_counts": dict(trajectory_modes),
        "saved_page_rank_limit": max(
            [int(row.get("offline_saved_page_rank_limit") or 0) for row in query_outcomes] or [0]
        ),
        "saved_trajectory_rank_limit": max(
            [int(row.get("offline_saved_trajectory_rank_limit") or 0) for row in query_outcomes] or [0]
        ),
        "page_cutoffs": page_cutoffs,
        "trajectory_cutoffs": trajectory_cutoffs,
    }


def _cutoff_diagnostics_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    def _section_delta(section_name: str) -> dict[str, Any]:
        before_section = dict(before.get(section_name, {}))
        after_section = dict(after.get(section_name, {}))
        output: dict[str, Any] = {}
        for cutoff in sorted(set(before_section) | set(after_section), key=lambda value: int(value)):
            before_cutoff = dict(before_section.get(cutoff, {}))
            after_cutoff = dict(after_section.get(cutoff, {}))
            output[cutoff] = {}
            for group_name in sorted(set(before_cutoff) | set(after_cutoff)):
                before_group = dict(before_cutoff.get(group_name, {}))
                after_group = dict(after_cutoff.get(group_name, {}))
                group_delta: dict[str, float | None] = {}
                for metric_name in sorted(set(before_group) | set(after_group)):
                    before_value = before_group.get(metric_name)
                    after_value = after_group.get(metric_name)
                    if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
                        group_delta[metric_name] = float(after_value) - float(before_value)
                    else:
                        group_delta[metric_name] = None
                output[cutoff][group_name] = group_delta
        return output

    before_suggested = dict(before.get("suggested_cutoffs", {}))
    after_suggested = dict(after.get("suggested_cutoffs", {}))
    return {
        "page_cutoffs": _section_delta("page_cutoffs"),
        "trajectory_cutoffs": _section_delta("trajectory_cutoffs"),
        "suggested_cutoffs": {
            key: {
                "before": before_suggested.get(key),
                "after": after_suggested.get(key),
            }
            for key in sorted(set(before_suggested) | set(after_suggested))
        },
    }


def _trajectory_length_diagnostics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "all_trajectory_p95": ("all_trajectories", "p95"),
        "all_trajectory_at_m_limit_rate": ("all_trajectories", "at_m_limit_rate"),
        "gold_trajectory_p95": ("all_queries_gold_trajectories", "gold_trajectory_length_stats", "p95"),
        "gold_trajectory_at_m_limit_rate": (
            "all_queries_gold_trajectories",
            "gold_trajectory_at_m_limit_rate",
        ),
        "gold_required_rank_p95": (
            "all_queries_gold_snapshot_requirements",
            "max_gold_snapshot_rank_required_stats",
            "p95",
        ),
        "candidate_lower_m_at_95": ("recommendations", "candidate_lower_m_at_95"),
    }

    def _get_path(section: dict[str, Any], path: tuple[str, ...]) -> Any:
        value: Any = section
        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    output: dict[str, Any] = {}
    for name, path in paths.items():
        before_value = _get_path(before, path)
        after_value = _get_path(after, path)
        delta = (
            float(after_value) - float(before_value)
            if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float))
            else None
        )
        output[name] = {"before": before_value, "after": after_value, "delta": delta}
    output["interpretation"] = {
        "before": dict(before.get("recommendations") or {}).get("interpretation"),
        "after": dict(after.get("recommendations") or {}).get("interpretation"),
    }
    return output


def _facet_value_supports_answer(facet: dict[str, Any], gold_answer_norm: str) -> bool:
    value = _normalize_text(facet.get("value"))
    value_span = _normalize_text(facet.get("value_span"))
    return bool(
        gold_answer_norm
        and (
            gold_answer_norm in value
            or gold_answer_norm in value_span
            or value in gold_answer_norm
            or value_span in gold_answer_norm
        )
    )


def _evidence_supports_gold_answer(
    gold_answer: Any,
    gold_claim_texts: list[str],
    gold_claim_facets: list[dict[str, Any]],
) -> bool:
    answer_items = [_normalize_text(item) for item in _split_answer_items(gold_answer)]
    if not answer_items:
        return False
    evidence_texts = [_normalize_text(text) for text in gold_claim_texts]
    evidence_texts.extend(_normalize_text(facet.get("value")) for facet in gold_claim_facets)
    evidence_texts.extend(_normalize_text(facet.get("value_span")) for facet in gold_claim_facets)
    evidence_texts = [text for text in evidence_texts if text]
    if not evidence_texts:
        return False
    return all(
        any(
            answer_item in evidence_text
            or evidence_text in answer_item
            for evidence_text in evidence_texts
        )
        for answer_item in answer_items
    )


def _surface_supports_gold_answer(
    gold_answer: Any,
    surface_values: list[Any],
) -> bool:
    answer_items = [_normalize_text(item) for item in _split_answer_items(gold_answer)]
    if not answer_items:
        return False
    evidence_texts = [_normalize_text(value) for value in surface_values if _normalize_text(value)]
    if not evidence_texts:
        return False
    return all(
        any(
            answer_item in evidence_text
            or evidence_text in answer_item
            for evidence_text in evidence_texts
        )
        for answer_item in answer_items
    )


def _trajectory_historical_card_texts(
    trajectory_ids: list[str],
    trajectory_metadata: dict[str, dict[str, Any]],
) -> list[str]:
    texts: list[str] = []
    for trajectory_id in trajectory_ids:
        metadata = trajectory_metadata.get(trajectory_id, {}) or {}
        card = metadata.get("trajectory_historical_evidence_card_v1")
        if isinstance(card, dict):
            fields: list[Any] = [
                card.get("identity_summary"),
                card.get("recent_update"),
                *sanitize_historical_item_terms(list(card.get("historical_item_terms") or []), limit=24),
                *list(card.get("facet_values") or []),
                *list(card.get("display_items") or []),
                *list(card.get("display_counts") or []),
                *list(card.get("display_key_facts") or []),
            ]
            fields.extend(
                anchor.get("text")
                for anchor in list(card.get("source_anchors") or [])
                if isinstance(anchor, dict)
            )
            texts.extend(collapse_whitespace(str(value or "")) for value in fields if collapse_whitespace(str(value or "")))
            continue
        texts.extend(
            sanitize_historical_item_terms(
                list(
                    metadata.get("trajectory_historical_item_terms_v2")
                    or metadata.get("trajectory_historical_item_terms_v1")
                    or []
                ),
                limit=24,
            )
        )
        texts.extend(
            collapse_whitespace(str(value or ""))
            for field_name in (
                "trajectory_drift_cluster_keys_v1",
                "display_items",
                "display_counts",
                "display_key_facts",
            )
            for value in list(metadata.get(field_name) or [])
            if collapse_whitespace(str(value or ""))
        )
    return texts


def _compute_preservation_and_stage_fields(
    *,
    gold_answer: Any,
    gold_refs: set[str],
    gold_claim_texts: list[str],
    gold_claim_facets: list[dict[str, Any]],
    gold_trajectory_ids: list[str],
    gold_page_ids: list[str],
    top_t_page_ids: list[str],
    selected_page_trajectory_ids: list[str],
    top_k_trajectory_ids: list[str],
    grounded_refs: set[str],
    trajectory_metadata: dict[str, dict[str, Any]],
    page_markdown_by_id: dict[str, str],
    snapshot_compaction_miss: bool = False,
    source_compaction_miss: bool = False,
) -> dict[str, Any]:
    gold_summary_texts = [
        collapse_whitespace(str((trajectory_metadata.get(trajectory_id, {}) or {}).get("retrieval_summary_text") or ""))
        for trajectory_id in gold_trajectory_ids
        if collapse_whitespace(str((trajectory_metadata.get(trajectory_id, {}) or {}).get("retrieval_summary_text") or ""))
    ]
    gold_wiki_texts = [
        collapse_whitespace(str(page_markdown_by_id.get(page_id) or ""))
        for page_id in gold_page_ids
        if collapse_whitespace(str(page_markdown_by_id.get(page_id) or ""))
    ]
    gold_historical_card_texts = _trajectory_historical_card_texts(gold_trajectory_ids, trajectory_metadata)

    gold_answer_fully_present_in_claims = _evidence_supports_gold_answer(
        gold_answer,
        gold_claim_texts,
        gold_claim_facets,
    )
    gold_answer_fully_present_in_trajectory_summaries = _surface_supports_gold_answer(
        gold_answer,
        gold_summary_texts,
    )
    gold_answer_fully_present_in_wiki_pages = _surface_supports_gold_answer(
        gold_answer,
        gold_wiki_texts,
    )
    gold_item_present_in_historical_card = _surface_supports_gold_answer(
        gold_answer,
        gold_historical_card_texts,
    )
    gold_item_present_in_latest_summary = gold_answer_fully_present_in_trajectory_summaries
    gold_item_present_in_wiki_after_historical_card = bool(
        gold_item_present_in_historical_card and gold_answer_fully_present_in_wiki_pages
    )
    trajectory_drift_suspected = bool(
        gold_item_present_in_historical_card
        and not gold_item_present_in_latest_summary
        and not gold_answer_fully_present_in_wiki_pages
    )

    trajectory_storage_miss = bool(gold_refs) and (not gold_answer_fully_present_in_claims)
    summary_compression_miss = bool(
        gold_answer_fully_present_in_claims and not gold_answer_fully_present_in_trajectory_summaries
    )
    wiki_compilation_compression_miss = bool(
        gold_answer_fully_present_in_trajectory_summaries and not gold_answer_fully_present_in_wiki_pages
    )

    all_gold_pages_in_top_t = bool(gold_page_ids) and set(gold_page_ids) <= set(top_t_page_ids)
    all_gold_trajectories_in_selected_pages = bool(gold_trajectory_ids) and set(gold_trajectory_ids) <= set(
        selected_page_trajectory_ids
    )
    all_gold_trajectories_in_final_top_k_after_page_routing = bool(gold_trajectory_ids) and set(
        gold_trajectory_ids
    ) <= set(top_k_trajectory_ids)

    page_routing_failed = bool(gold_page_ids) and not all_gold_pages_in_top_t
    selected_pages_failed_to_cover_all_gold_trajectories = bool(gold_trajectory_ids) and not (
        all_gold_trajectories_in_selected_pages
    )
    trajectory_retrieval_failed_after_page_routing = bool(gold_trajectory_ids) and not (
        all_gold_trajectories_in_final_top_k_after_page_routing
    )

    grounded_gold_ref_coverage_count = len(gold_refs & grounded_refs)
    grounded_gold_ref_coverage_rate = _safe_rate(grounded_gold_ref_coverage_count, len(gold_refs)) if gold_refs else None
    all_gold_refs_grounded = bool(gold_refs) and gold_refs <= grounded_refs

    strict_retrieval_success_but_answer_wrong = bool(
        gold_answer_fully_present_in_claims
        and all_gold_pages_in_top_t
        and all_gold_trajectories_in_selected_pages
        and all_gold_trajectories_in_final_top_k_after_page_routing
        and all_gold_refs_grounded
        and not snapshot_compaction_miss
        and not source_compaction_miss
    )

    return {
        "gold_answer_fully_present_in_claims": gold_answer_fully_present_in_claims,
        "gold_answer_fully_present_in_trajectory_summaries": gold_answer_fully_present_in_trajectory_summaries,
        "gold_answer_fully_present_in_wiki_pages": gold_answer_fully_present_in_wiki_pages,
        "gold_item_present_in_historical_card": gold_item_present_in_historical_card,
        "gold_item_present_in_latest_summary": gold_item_present_in_latest_summary,
        "gold_item_present_in_wiki_after_historical_card": gold_item_present_in_wiki_after_historical_card,
        "trajectory_drift_suspected": trajectory_drift_suspected,
        "trajectory_storage_miss": trajectory_storage_miss,
        "summary_compression_miss": summary_compression_miss,
        "wiki_compilation_compression_miss": wiki_compilation_compression_miss,
        "all_gold_pages_in_top_t": all_gold_pages_in_top_t,
        "all_gold_trajectories_in_selected_pages": all_gold_trajectories_in_selected_pages,
        "all_gold_trajectories_in_final_top_k_after_page_routing": all_gold_trajectories_in_final_top_k_after_page_routing,
        "page_routing_failed": page_routing_failed,
        "selected_pages_failed_to_cover_all_gold_trajectories": selected_pages_failed_to_cover_all_gold_trajectories,
        "trajectory_retrieval_failed_after_page_routing": trajectory_retrieval_failed_after_page_routing,
        "grounded_gold_ref_coverage_count": grounded_gold_ref_coverage_count,
        "grounded_gold_ref_coverage_rate": grounded_gold_ref_coverage_rate,
        "all_gold_refs_grounded": all_gold_refs_grounded,
        "strict_retrieval_success_but_answer_wrong": strict_retrieval_success_but_answer_wrong,
    }


def _claim_preservation_metadata_fields(gold_claims: list[dict[str, Any]]) -> dict[str, Any]:
    missing_candidates: list[dict[str, Any]] = []
    weak_source_links: list[dict[str, Any]] = []
    seen_missing: set[tuple[str, str]] = set()
    repair_used = False
    repair_succeeded = False
    candidate_count = 0
    metadata_present = False
    for claim in gold_claims:
        metadata = dict(claim.get("metadata") or {})
        metadata_present = metadata_present or any(
            key in metadata
            for key in (
                "claim_preservation_candidate_count_v1",
                "claim_preservation_misses_v1",
                "claim_preservation_weak_source_links_v1",
            )
        )
        candidate_count = max(candidate_count, int(metadata.get("claim_preservation_candidate_count_v1") or 0))
        repair_used = repair_used or bool(metadata.get("claim_preservation_repair_used_v1"))
        repair_succeeded = repair_succeeded or bool(metadata.get("claim_preservation_repair_succeeded_v1"))
        for candidate in list(metadata.get("claim_preservation_misses_v1") or []):
            if not isinstance(candidate, dict):
                continue
            key = (str(candidate.get("surface") or ""), str(candidate.get("category") or ""))
            if key in seen_missing:
                continue
            seen_missing.add(key)
            missing_candidates.append(dict(candidate))
        for weak_link in list(metadata.get("claim_preservation_weak_source_links_v1") or []):
            if isinstance(weak_link, dict):
                weak_source_links.append(dict(weak_link))
    return {
        "claim_preservation_candidate_count": candidate_count,
        "claim_preservation_missing_candidates": missing_candidates,
        "claim_preservation_missing_count": len(missing_candidates),
        "claim_preservation_repair_used": repair_used,
        "claim_preservation_repair_succeeded": repair_succeeded,
        "claim_preservation_weak_source_links": weak_source_links,
        "claim_preservation_metadata_present": metadata_present,
        "raw_candidate_extraction_miss": bool(gold_claims and metadata_present and candidate_count == 0),
        "claim_preservation_audit_miss": bool(missing_candidates),
        "claim_source_link_miss": bool(weak_source_links),
    }


def _source_surface_preservation_fields(
    *,
    gold_answer: Any,
    gold_refs: set[str],
    gold_claims: list[dict[str, Any]],
    gold_trajectory_ids: list[str],
    gold_page_ids: list[str],
    trajectory_metadata: dict[str, dict[str, Any]],
    page_markdown_by_id: dict[str, str],
    raw_message_ids_by_ref: dict[str, list[str]],
    raw_messages_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw_texts = [
        collapse_whitespace(str(raw_messages_by_id.get(message_id, {}).get("content") or ""))
        for ref in gold_refs
        for message_id in raw_message_ids_by_ref.get(ref, [])
        if collapse_whitespace(str(raw_messages_by_id.get(message_id, {}).get("content") or ""))
    ]
    claim_texts = [str(claim.get("text") or "") for claim in gold_claims]
    exact_values: list[Any] = []
    source_surface_terms: list[Any] = []
    source_surface_records: list[dict[str, Any]] = []
    for claim in gold_claims:
        metadata = dict(claim.get("metadata") or {})
        exact_values.extend(list(claim.get("exact_terms") or []))
        exact_values.extend(list(metadata.get("source_surface_raw_terms_v1") or []))
        exact_values.extend(list(metadata.get("source_surface_terms_v1") or []))
        source_surface_terms.extend(list(metadata.get("source_surface_raw_terms_v1") or []))
        source_surface_terms.extend(list(metadata.get("source_surface_terms_v1") or []))
        source_surface_records.extend(
            dict(record)
            for record in list(metadata.get("source_surface_records_v1") or [])
            if isinstance(record, dict)
        )
    historical_texts = _trajectory_historical_card_texts(gold_trajectory_ids, trajectory_metadata)
    wiki_texts = [
        collapse_whitespace(str(page_markdown_by_id.get(page_id) or ""))
        for page_id in gold_page_ids
        if collapse_whitespace(str(page_markdown_by_id.get(page_id) or ""))
    ]
    miss_count = sum(
        int((trajectory_metadata.get(trajectory_id, {}) or {}).get("source_surface_preservation_miss_count_v1") or 0)
        for trajectory_id in gold_trajectory_ids
    )
    raw_present = _surface_supports_gold_answer(gold_answer, raw_texts)
    claims_present = _surface_supports_gold_answer(gold_answer, claim_texts)
    exact_present = _surface_supports_gold_answer(gold_answer, exact_values)
    card_present = _surface_supports_gold_answer(gold_answer, historical_texts)
    wiki_present = _surface_supports_gold_answer(gold_answer, wiki_texts)
    return {
        "gold_surface_present_in_raw": raw_present if raw_texts else None,
        "gold_surface_preserved_in_claims": claims_present if gold_claims else None,
        "gold_surface_preserved_in_exact_terms": exact_present if exact_values else None,
        "gold_surface_preserved_in_historical_card": card_present if historical_texts else None,
        "gold_surface_preserved_in_wiki": wiki_present if wiki_texts else None,
        "source_surface_term_miss_count": miss_count,
        "source_surface_terms": list(dict.fromkeys(str(value) for value in source_surface_terms if str(value).strip())),
        "source_surface_record_count": len(source_surface_records),
    }


def _load_run_details(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    details_path = run_dir / "details.json"
    database_path = run_dir / "trajpatch.sqlite"
    details = _load_json(details_path)
    run_meta = dict(details.get("run_meta", {}))
    if str(run_meta.get("dataset")).lower() != "locomo":
        raise ValueError(
            f"Failure attribution currently supports LOCOMO runs only, got dataset={run_meta.get('dataset')!r}."
        )
    return run_meta, list(details.get("samples", [])), database_path


def _load_run_indexes(database_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        raw_messages_by_id: dict[str, dict[str, Any]] = {}
        raw_message_ids_by_ref: dict[str, set[str]] = defaultdict(set)
        sample_raw_messages: dict[str, list[SimpleNamespace]] = defaultdict(list)
        for row in connection.execute(
            "SELECT id, sample_id, speaker_name, content, source_ref, occurred_at FROM raw_messages"
        ):
            record = {
                "id": str(row["id"]),
                "sample_id": str(row["sample_id"]),
                "speaker_name": row["speaker_name"],
                "content": str(row["content"]),
                "source_ref": row["source_ref"],
                "occurred_at": row["occurred_at"],
            }
            raw_messages_by_id[record["id"]] = record
            if record["source_ref"]:
                raw_message_ids_by_ref[str(record["source_ref"])].add(record["id"])
            sample_raw_messages[record["sample_id"]].append(SimpleNamespace(**record))

        sample_entity_lexicons = {
            sample_id: build_sample_entity_lexicon(messages)
            for sample_id, messages in sample_raw_messages.items()
        }

        trajectory_to_sample: dict[str, str] = {}
        sample_to_trajectories: dict[str, set[str]] = defaultdict(set)
        trajectory_metadata: dict[str, dict[str, Any]] = {}
        for row in connection.execute(
            """
            SELECT id, sample_id, latest_snapshot_id, metadata_json
            FROM trajectories
            """
        ):
            trajectory_id = str(row["id"])
            sample_id = str(row["sample_id"])
            metadata = dict(_load_json_field(row["metadata_json"], {}))
            latest_snapshot_id = row["latest_snapshot_id"]
            if latest_snapshot_id and "latest_snapshot_id" not in metadata:
                metadata["latest_snapshot_id"] = str(latest_snapshot_id)
            trajectory_to_sample[trajectory_id] = sample_id
            sample_to_trajectories[sample_id].add(trajectory_id)
            trajectory_metadata[trajectory_id] = metadata

        claims_by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
        claims_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in connection.execute(
            """
            SELECT snapshot_id, trajectory_id, text, source_message_ids_json, metadata_json
            FROM claims
            """
        ):
            source_message_ids = [
                str(message_id)
                for message_id in _load_json_field(row["source_message_ids_json"], [])
                if message_id is not None
            ]
            source_refs = {
                str(raw_messages_by_id[message_id]["source_ref"])
                for message_id in source_message_ids
                if message_id in raw_messages_by_id and raw_messages_by_id[message_id]["source_ref"]
            }
            metadata = dict(_load_json_field(row["metadata_json"], {}))
            claim_info = {
                "text": str(row["text"]),
                "source_message_ids": source_message_ids,
                "source_refs": source_refs,
                "facets": list(metadata.get("facets_v2") or metadata.get("facets_v1") or []),
                "exact_terms": list(metadata.get("exact_terms_v2") or metadata.get("exact_terms_v1") or []),
                "display_signals": dict(metadata.get("display_signals_v1") or {}),
                "metadata": metadata,
            }
            snapshot_id = str(row["snapshot_id"])
            trajectory_id = str(row["trajectory_id"])
            claims_by_snapshot[snapshot_id].append(claim_info)
            claims_by_trajectory[trajectory_id].append(claim_info)

        snapshot_to_trajectory: dict[str, str] = {}
        trajectory_to_snapshots: dict[str, set[str]] = defaultdict(set)
        trajectory_snapshot_rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
        snapshot_versions: dict[str, int] = {}
        snapshot_refs: dict[str, set[str]] = {}
        snapshot_metadata: dict[str, dict[str, Any]] = {}
        snapshot_claim_texts: dict[str, list[str]] = {}
        snapshot_claim_facets: dict[str, list[dict[str, Any]]] = {}
        force_recall_counts = Counter(
            {
                "forced_memory_seed_count": 0,
                "low_salience_memory_count": 0,
                "llm_no_memory_forced_count": 0,
                "zero_claim_episodic_candidate_count": 0,
                "zero_claim_episodic_persisted_count": 0,
                "zero_claim_low_salience_skipped_count": 0,
            }
        )
        for row in connection.execute("SELECT id, trajectory_id, version, links_json, metadata_json FROM episodic_snapshots"):
            snapshot_id = str(row["id"])
            trajectory_id = str(row["trajectory_id"])
            version = int(row["version"] or 0)
            metadata = dict(_load_json_field(row["metadata_json"], {}))
            snapshot_metadata[snapshot_id] = metadata
            if metadata.get("forced_episodic_seed_used_v1"):
                force_recall_counts["forced_memory_seed_count"] += 1
            if metadata.get("low_salience_memory_v1"):
                force_recall_counts["low_salience_memory_count"] += 1
            if metadata.get("llm_no_memory_overridden_v1"):
                force_recall_counts["llm_no_memory_forced_count"] += 1
            if metadata.get("zero_claim_episodic_memory_v1"):
                force_recall_counts["zero_claim_episodic_candidate_count"] += 1
            if not claims_by_snapshot.get(snapshot_id):
                force_recall_counts["zero_claim_episodic_persisted_count"] += 1
            snapshot_to_trajectory[snapshot_id] = trajectory_id
            trajectory_to_snapshots[trajectory_id].add(snapshot_id)
            trajectory_snapshot_rows[trajectory_id].append((version, snapshot_id))
            snapshot_versions[snapshot_id] = version
            message_ids = {
                str(message_id)
                for message_id in _load_json_field(row["links_json"], [])
                if message_id is not None
            }
            for claim in claims_by_snapshot.get(snapshot_id, []):
                message_ids.update(claim["source_message_ids"])
            refs = {
                str(raw_messages_by_id[message_id]["source_ref"])
                for message_id in message_ids
                if message_id in raw_messages_by_id and raw_messages_by_id[message_id]["source_ref"]
            }
            snapshot_refs[snapshot_id] = refs
            snapshot_claim_texts[snapshot_id] = [claim["text"] for claim in claims_by_snapshot.get(snapshot_id, [])]
            snapshot_claim_facets[snapshot_id] = [
                facet
                for claim in claims_by_snapshot.get(snapshot_id, [])
                for facet in claim["facets"]
            ]
        trajectory_snapshot_ids_ordered = {
            trajectory_id: [snapshot_id for _, snapshot_id in sorted(rows, key=lambda item: (item[0], item[1]))]
            for trajectory_id, rows in trajectory_snapshot_rows.items()
        }
        trajectory_lengths = {
            trajectory_id: len(snapshot_ids)
            for trajectory_id, snapshot_ids in trajectory_snapshot_ids_ordered.items()
        }
        snapshot_rank_in_trajectory = {
            snapshot_id: rank
            for snapshot_ids in trajectory_snapshot_ids_ordered.values()
            for rank, snapshot_id in enumerate(snapshot_ids, start=1)
        }
        sample_trajectory_lengths = {
            sample_id: {
                trajectory_id: trajectory_lengths.get(trajectory_id, 0)
                for trajectory_id in sorted(trajectory_ids)
            }
            for sample_id, trajectory_ids in sample_to_trajectories.items()
        }

        trajectory_refs: dict[str, set[str]] = defaultdict(set)
        trajectory_claim_texts: dict[str, list[str]] = defaultdict(list)
        trajectory_claim_facets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trajectory_id, snapshot_ids in trajectory_to_snapshots.items():
            refs: set[str] = set()
            for snapshot_id in snapshot_ids:
                refs.update(snapshot_refs.get(snapshot_id, set()))
            trajectory_refs[trajectory_id] = refs
            trajectory_claim_texts[trajectory_id] = [
                claim["text"] for claim in claims_by_trajectory.get(trajectory_id, [])
            ]
            trajectory_claim_facets[trajectory_id] = [
                facet
                for claim in claims_by_trajectory.get(trajectory_id, [])
                for facet in claim["facets"]
            ]

        page_to_trajectory_ids: dict[str, list[str]] = {}
        page_markdown_by_id: dict[str, str] = {}
        page_keywords_by_id: dict[str, list[str]] = {}
        page_metadata_by_id: dict[str, dict[str, Any]] = {}
        page_to_sample: dict[str, str] = {}
        page_types: dict[str, str] = {}
        sample_to_pages: dict[str, set[str]] = defaultdict(set)
        for row in connection.execute(
            """
            SELECT id, sample_id, page_type, trajectory_ids_json, markdown_text, keywords_json, metadata_json
            FROM wiki_pages
            """
        ):
            page_id = str(row["id"])
            sample_id = str(row["sample_id"])
            trajectory_ids = [
                str(item)
                for item in _load_json_field(row["trajectory_ids_json"], [])
                if str(item).strip()
            ]
            page_to_trajectory_ids[page_id] = trajectory_ids
            page_markdown_by_id[page_id] = str(row["markdown_text"] or "")
            page_keywords_by_id[page_id] = [
                str(item)
                for item in _load_json_field(row["keywords_json"], [])
                if str(item).strip()
            ]
            page_metadata_by_id[page_id] = dict(_load_json_field(row["metadata_json"], {}))
            page_to_sample[page_id] = sample_id
            page_types[page_id] = str(row["page_type"] or "")
            sample_to_pages[sample_id].add(page_id)

        page_refs: dict[str, set[str]] = {}
        for page_id, trajectory_ids in page_to_trajectory_ids.items():
            refs: set[str] = set()
            for trajectory_id in trajectory_ids:
                refs.update(trajectory_refs.get(trajectory_id, set()))
            page_refs[page_id] = refs

        sample_memory_refs: dict[str, set[str]] = defaultdict(set)
        for sample_id, trajectory_ids in sample_to_trajectories.items():
            refs: set[str] = set()
            for trajectory_id in trajectory_ids:
                refs.update(trajectory_refs.get(trajectory_id, set()))
            sample_memory_refs[sample_id] = refs

        retrieval_events: dict[str, dict[str, Any]] = {}
        for row in connection.execute(
            """
            SELECT id, top_t_pages, top_k, page_ids_json, trajectory_ids_json, snapshot_ids_json, expanded_snapshot_ids_json,
                   source_message_ids_json, metadata_json
            FROM retrieval_events
            """
        ):
            event_id = str(row["id"])
            metadata = dict(_load_json_field(row["metadata_json"], {}))
            page_ids = [str(item) for item in _load_json_field(row["page_ids_json"], [])]
            trajectory_ids = [str(item) for item in _load_json_field(row["trajectory_ids_json"], [])]
            snapshot_hit_ids = [str(item) for item in _load_json_field(row["snapshot_ids_json"], [])]
            expanded_snapshot_ids = [
                str(item) for item in _load_json_field(row["expanded_snapshot_ids_json"], [])
            ]
            raw_expanded_snapshot_ids = [
                str(item)
                for item in list(metadata.get("raw_expanded_snapshot_ids") or [])
                if str(item).strip()
            ]
            raw_expanded_refs = [
                str(item)
                for item in list(metadata.get("raw_expanded_refs") or [])
                if str(item).strip()
            ]
            if not raw_expanded_refs and raw_expanded_snapshot_ids:
                raw_expanded_refs = sorted(
                    set().union(*(snapshot_refs.get(snapshot_id, set()) for snapshot_id in raw_expanded_snapshot_ids))
                )
            grounded_source_refs = metadata.get("source_refs", [])
            if not grounded_source_refs:
                grounded_source_refs = [
                    raw_messages_by_id[message_id]["source_ref"]
                    for message_id in _load_json_field(row["source_message_ids_json"], [])
                    if message_id in raw_messages_by_id and raw_messages_by_id[message_id]["source_ref"]
                ]
            raw_source_message_ids = [
                str(item)
                for item in list(metadata.get("raw_source_message_ids") or [])
                if str(item).strip()
            ]
            raw_source_refs = [
                str(item)
                for item in list(metadata.get("raw_source_refs") or [])
                if str(item).strip()
            ]
            if not raw_source_refs and raw_source_message_ids:
                raw_source_refs = [
                    str(raw_messages_by_id[message_id]["source_ref"])
                    for message_id in raw_source_message_ids
                    if message_id in raw_messages_by_id and raw_messages_by_id[message_id]["source_ref"]
                ]
            entity_linked_snapshot_ids = [
                str(item) for item in list(metadata.get("entity_linked_snapshot_ids") or [])
            ]
            entity_linked_trajectory_ids = [
                str(item) for item in list(metadata.get("entity_linked_trajectory_ids") or [])
            ]
            coarse_ranked = list(metadata.get("coarse_ranked_trajectories") or [])
            retrieval_events[event_id] = {
                "top_t_pages": int(row["top_t_pages"] or 0),
                "top_k": int(row["top_k"] or 0),
                "page_ids": page_ids,
                "trajectory_ids": trajectory_ids,
                "snapshot_hit_ids": snapshot_hit_ids,
                "expanded_snapshot_ids": expanded_snapshot_ids,
                "raw_expanded_snapshot_ids": raw_expanded_snapshot_ids,
                "raw_expanded_refs": sorted(set(raw_expanded_refs)),
                "grounded_source_refs": sorted(set(str(ref) for ref in grounded_source_refs if ref)),
                "raw_source_message_ids": raw_source_message_ids,
                "raw_source_refs": sorted(set(str(ref) for ref in raw_source_refs if ref)),
                "source_compaction_dropped_ids": [
                    str(item)
                    for item in list(metadata.get("source_compaction_dropped_ids") or [])
                    if str(item).strip()
                ],
                "source_compaction_dropped_refs": [
                    str(item)
                    for item in list(metadata.get("source_compaction_dropped_refs") or [])
                    if str(item).strip()
                ],
                "candidate_trajectory_refs": sorted(
                    set().union(*(trajectory_refs.get(trajectory_id, set()) for trajectory_id in trajectory_ids))
                ),
                "fine_hit_refs": sorted(
                    set().union(*(snapshot_refs.get(snapshot_id, set()) for snapshot_id in snapshot_hit_ids))
                ),
                "expanded_refs": sorted(
                    set().union(*(snapshot_refs.get(snapshot_id, set()) for snapshot_id in expanded_snapshot_ids))
                ),
                "metadata": metadata,
                "query_entities": list(metadata.get("query_entities") or []),
                "query_facets": {
                    "tags": list((metadata.get("query_facets") or {}).get("tags") or []),
                    "values": list((metadata.get("query_facets") or {}).get("values") or []),
                },
                "coarse_ranked_trajectories": coarse_ranked,
                "coarse_ranked_ids": [
                    str(item.get("trajectory_id"))
                    for item in coarse_ranked
                    if str(item.get("trajectory_id") or "").strip()
                ],
                "entity_linked_snapshot_ids": entity_linked_snapshot_ids,
                "entity_linked_trajectory_ids": entity_linked_trajectory_ids,
                "entity_linked_refs": sorted(
                    set().union(*(snapshot_refs.get(snapshot_id, set()) for snapshot_id in entity_linked_snapshot_ids))
                ),
            }

        return {
            "sample_memory_refs": sample_memory_refs,
            "retrieval_events": retrieval_events,
            "trajectory_refs": trajectory_refs,
            "snapshot_refs": snapshot_refs,
            "trajectory_to_sample": trajectory_to_sample,
            "sample_to_trajectories": sample_to_trajectories,
            "sample_to_pages": sample_to_pages,
            "snapshot_to_trajectory": snapshot_to_trajectory,
            "trajectory_to_snapshots": trajectory_to_snapshots,
            "trajectory_snapshot_ids_ordered": trajectory_snapshot_ids_ordered,
            "trajectory_lengths": trajectory_lengths,
            "snapshot_versions": snapshot_versions,
            "snapshot_metadata": snapshot_metadata,
            "snapshot_rank_in_trajectory": snapshot_rank_in_trajectory,
            "sample_trajectory_lengths": sample_trajectory_lengths,
            "trajectory_metadata": trajectory_metadata,
            "page_to_trajectory_ids": page_to_trajectory_ids,
            "page_markdown_by_id": page_markdown_by_id,
            "page_keywords_by_id": page_keywords_by_id,
            "page_metadata_by_id": page_metadata_by_id,
            "page_to_sample": page_to_sample,
            "page_types": page_types,
            "page_refs": page_refs,
            "claims_by_snapshot": claims_by_snapshot,
            "claims_by_trajectory": claims_by_trajectory,
            "trajectory_claim_texts": trajectory_claim_texts,
            "trajectory_claim_facets": trajectory_claim_facets,
            "snapshot_claim_texts": snapshot_claim_texts,
            "snapshot_claim_facets": snapshot_claim_facets,
            "raw_messages_by_id": raw_messages_by_id,
            "raw_message_ids_by_ref": raw_message_ids_by_ref,
            "sample_entity_lexicons": sample_entity_lexicons,
            "force_recall_counts": dict(force_recall_counts),
        }
    finally:
        connection.close()


def _classify_failure(
    *,
    gold_refs: set[str],
    sample_memory_refs: set[str],
    candidate_trajectory_refs: set[str],
    fine_hit_refs: set[str],
    expanded_refs: set[str],
    grounded_refs: set[str],
) -> str:
    if not gold_refs:
        return "unknown_no_gold_refs"
    if not (gold_refs & sample_memory_refs):
        return "memory_absent"
    if not (gold_refs & candidate_trajectory_refs):
        return "coarse_retrieval_miss"
    if not (gold_refs & fine_hit_refs) and not (gold_refs & expanded_refs):
        return "fine_retrieval_miss"
    if (gold_refs & expanded_refs) and not (gold_refs & grounded_refs):
        return "grounding_miss"
    if gold_refs & grounded_refs:
        return "answer_model_error"
    return "unknown_other"


def _derive_query_diagnostics(
    *,
    sample_id: str,
    question: str,
    retrieval_event: dict[str, Any],
    sample_entity_lexicons: dict[str, dict[str, str]],
) -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    entity_lexicon = sample_entity_lexicons.get(sample_id, {})
    derived = extract_query_facets_v1(question, entity_lexicon)
    derived_shape = classify_query_shape_v1(question, entity_lexicon)
    retrieval_metadata = dict(retrieval_event.get("metadata") or {})
    query_entities = list(retrieval_event.get("query_entities") or []) or list(derived["entities"])
    query_facets = dict(retrieval_event.get("query_facets") or {})
    query_shape_metadata = dict(retrieval_event.get("query_shape") or retrieval_metadata.get("query_shape") or {})
    normalized_metadata_shape = {
        "list_like": bool(query_shape_metadata.get("list_like")),
        "multi_entity": bool(query_shape_metadata.get("multi_entity")),
        "comparison_like": bool(query_shape_metadata.get("comparison_like")),
        "count_like": bool(query_shape_metadata.get("count_like")),
        "item_family": query_shape_metadata.get("item_family"),
        "tags": sorted(str(value) for value in list(query_shape_metadata.get("tags") or [])),
    }
    normalized_current_shape = {
        "list_like": bool(derived_shape.get("list_like")),
        "multi_entity": bool(derived_shape.get("multi_entity")),
        "comparison_like": bool(derived_shape.get("comparison_like")),
        "count_like": bool(derived_shape.get("count_like")),
        "item_family": derived_shape.get("item_family"),
        "tags": sorted(str(value) for value in list(derived_shape.get("tags") or [])),
    }
    normalized_current_shape["query_shape_source"] = "derived_current_v1"
    normalized_current_shape["query_shape_metadata_mismatch"] = (
        bool(query_shape_metadata) and normalized_metadata_shape != {
            key: normalized_current_shape[key]
            for key in ("list_like", "multi_entity", "comparison_like", "count_like", "item_family", "tags")
        }
    )
    return query_entities, {
        "tags": list(query_facets.get("tags") or derived["tags"]),
        "values": list(query_facets.get("values") or derived["values"]),
    }, normalized_current_shape


def _query_shape_requires_coverage(query_shape: dict[str, Any]) -> bool:
    return bool(
        query_shape.get("list_like")
        or query_shape.get("multi_entity")
        or query_shape.get("comparison_like")
        or query_shape.get("count_like")
    )


def _match_normalized_item(left: str, right: str) -> bool:
    return bool(left and right and (left in right or right in left))


_ATOMIC_GROUNDED_ANSWER_TYPES = {"date", "time", "datetime", "place", "person", "boolean", "value"}


def _normalized_expected_answer_type(answer_metadata: dict[str, Any]) -> str:
    for key in [
        "answer_expected_type",
        "expected_answer_type",
        "answer_synthesis_typed_retry_expected_type",
        "answer_synthesis_recovered_answer_type",
    ]:
        value = answer_metadata.get(key)
        normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
        if normalized:
            if normalized in {"datetime", "date_time", "temporal", "time_date"}:
                return "date"
            return normalized
    return ""


def _looks_like_single_temporal_value(value: Any) -> bool:
    text = collapse_whitespace(str(value or "")).casefold()
    if not text:
        return False
    month = (
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    )
    if re.search(rf"\b(?:\d{{1,2}}\s+)?{month}\b,?\s+\d{{4}}\b", text):
        return True
    if re.search(rf"\b{month}\s+\d{{1,2}},?\s+\d{{4}}\b", text):
        return True
    if re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", text):
        return True
    return False


def _query_shape_is_explicit_multi_item(query_shape: dict[str, Any]) -> bool:
    return bool(
        query_shape.get("list_like")
        or query_shape.get("multi_entity")
        or query_shape.get("comparison_like")
    )


def _should_treat_grounded_answer_as_atomic(
    *,
    gold_answer: Any,
    answer_text: Any,
    query_shape: dict[str, Any],
    answer_metadata: dict[str, Any],
) -> bool:
    if _query_shape_is_explicit_multi_item(query_shape):
        return False
    expected_type = _normalized_expected_answer_type(answer_metadata)
    if expected_type in _ATOMIC_GROUNDED_ANSWER_TYPES:
        return True
    # Backward-compatible fallback for old runs that predate answer_expected_type.
    return _looks_like_single_temporal_value(gold_answer) or _looks_like_single_temporal_value(answer_text)


def _classify_grounded_answer_failure(
    *,
    gold_answer: Any,
    answer_text: Any,
    query_shape: dict[str, Any],
    answer_metadata: dict[str, Any],
) -> str:
    issue = str(answer_metadata.get("answer_postcheck_issue") or "").strip()
    if issue in {"unsupported_extra_items", "unsupported_count"}:
        return "answer_overgenerated_after_grounding"
    expected_type = _normalized_expected_answer_type(answer_metadata)
    if query_shape.get("count_like") or expected_type == "count":
        gold_numbers = set(re.findall(r"\b\d+\b", str(gold_answer or "")))
        answer_numbers = set(re.findall(r"\b\d+\b", str(answer_text or "")))
        if answer_numbers and gold_numbers and answer_numbers - gold_numbers:
            return "answer_overgenerated_after_grounding"
        return "answer_selection_error_after_grounding"
    if _should_treat_grounded_answer_as_atomic(
        gold_answer=gold_answer,
        answer_text=answer_text,
        query_shape=query_shape,
        answer_metadata=answer_metadata,
    ):
        return "answer_selection_error_after_grounding"
    gold_items = [_normalize_text(item) for item in _split_answer_items(gold_answer)]
    answer_items = [_normalize_text(item) for item in _split_answer_items(answer_text)]
    if (
        _query_shape_is_explicit_multi_item(query_shape)
        or len(gold_items) > 1
    ):
        if gold_items and answer_items:
            matched_gold = sum(
                1
                for gold_item in gold_items
                if any(_match_normalized_item(gold_item, answer_item) for answer_item in answer_items)
            )
            unsupported_answer_items = [
                answer_item
                for answer_item in answer_items
                if not any(_match_normalized_item(answer_item, gold_item) for gold_item in gold_items)
            ]
            if unsupported_answer_items:
                return "answer_overgenerated_after_grounding"
            if matched_gold < len(gold_items):
                return "answer_incomplete_after_grounding"
        return "answer_selection_error_after_grounding"
    return "answer_selection_error_after_grounding"


def _gold_item_missing_but_umbrella_present(
    *,
    gold_answer: Any,
    query_shape: dict[str, Any],
    gold_claim_texts: list[str],
    gold_summary_texts: list[str],
    gold_wiki_texts: list[str],
    gold_item_present_in_claims: bool,
) -> bool:
    if gold_item_present_in_claims:
        return False
    item_family = str(query_shape.get("item_family") or "").strip()
    if not item_family:
        return False
    umbrella_terms = _ANSWER_ITEM_FAMILY_UMBRELLA_TERMS.get(item_family, set())
    if not umbrella_terms:
        return False
    combined_text = " ".join([*gold_claim_texts, *gold_summary_texts, *gold_wiki_texts]).casefold()
    if not any(term in combined_text for term in umbrella_terms):
        return False
    gold_items = [_normalize_text(item) for item in _split_answer_items(gold_answer)]
    return bool(gold_items)


def _entity_cluster_ratio(cluster_rows: list[tuple[str, ...]]) -> float | None:
    if not cluster_rows:
        return None
    unique_clusters = {cluster for cluster in cluster_rows if cluster}
    return _safe_rate(len(cluster_rows) - len(unique_clusters), len(cluster_rows))


def analyze_locomo_run_failures(
    run_path: Path | str, *, top_examples_per_bucket: int = 5
) -> dict[str, Any]:
    run_dir = _resolve_run_dir(run_path)
    run_meta, sample_rows, database_path = _load_run_details(run_dir)
    details_path = run_dir / "details.json"
    summary_payload = _load_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
    text_only_filter_diagnostics = dict(
        dict(summary_payload.get("evaluation_filters") or {}).get("text_only") or {}
    )
    cuda_preflight_diagnostics = dict(summary_payload.get("cuda_preflight") or {})
    if not cuda_preflight_diagnostics and any(
        key in run_meta
        for key in [
            "cuda_preflight_mode",
            "cuda_preflight_risk",
            "cuda_preflight_warnings",
            "cuda_preflight_errors",
            "cuda_preflight_assignments",
        ]
    ):
        cuda_preflight_diagnostics = {
            "mode": run_meta.get("cuda_preflight_mode"),
            "risk": run_meta.get("cuda_preflight_risk"),
            "warnings": run_meta.get("cuda_preflight_warnings"),
            "errors": run_meta.get("cuda_preflight_errors"),
            "assignments": run_meta.get("cuda_preflight_assignments"),
        }
    llm_call_diagnostics = dict(summary_payload.get("llm_call_diagnostics") or {})
    fallback_repair_events = _load_jsonl(run_dir / "fallback_repair_events.jsonl")
    fallback_repair_diagnostics = (
        summarize_fallback_repair_events(fallback_repair_events, sample_rows)
        if fallback_repair_events
        else dict(summary_payload.get("fallback_repair_diagnostics") or {})
    )
    if not fallback_repair_diagnostics:
        fallback_repair_diagnostics = legacy_fallback_repair_diagnostics(
            sample_rows,
            llm_call_diagnostics,
        )

    indexes = _load_run_indexes(database_path)
    sample_memory_refs_map: dict[str, set[str]] = indexes["sample_memory_refs"]
    retrieval_events: dict[str, dict[str, Any]] = indexes["retrieval_events"]
    trajectory_refs: dict[str, set[str]] = indexes["trajectory_refs"]
    snapshot_refs: dict[str, set[str]] = indexes["snapshot_refs"]
    trajectory_lengths: dict[str, int] = indexes["trajectory_lengths"]
    snapshot_versions: dict[str, int] = indexes["snapshot_versions"]
    snapshot_metadata: dict[str, dict[str, Any]] = indexes["snapshot_metadata"]
    snapshot_rank_in_trajectory: dict[str, int] = indexes["snapshot_rank_in_trajectory"]
    trajectory_snapshot_ids_ordered: dict[str, list[str]] = indexes["trajectory_snapshot_ids_ordered"]
    trajectory_to_sample: dict[str, str] = indexes["trajectory_to_sample"]
    sample_to_trajectories: dict[str, set[str]] = indexes["sample_to_trajectories"]
    sample_to_pages: dict[str, set[str]] = indexes["sample_to_pages"]
    trajectory_metadata: dict[str, dict[str, Any]] = indexes["trajectory_metadata"]
    page_to_trajectory_ids: dict[str, list[str]] = indexes["page_to_trajectory_ids"]
    page_markdown_by_id: dict[str, str] = indexes["page_markdown_by_id"]
    page_metadata_by_id: dict[str, dict[str, Any]] = indexes["page_metadata_by_id"]
    page_types: dict[str, str] = indexes["page_types"]
    claims_by_trajectory: dict[str, list[dict[str, Any]]] = indexes["claims_by_trajectory"]
    sample_entity_lexicons: dict[str, dict[str, str]] = indexes["sample_entity_lexicons"]
    raw_messages_by_id: dict[str, dict[str, Any]] = indexes["raw_messages_by_id"]
    raw_message_ids_by_ref: dict[str, set[str]] = indexes["raw_message_ids_by_ref"]
    trajectory_drift_rows = build_trajectory_drift_rows(
        database_path=database_path,
        trajectory_to_sample=trajectory_to_sample,
        trajectory_snapshot_ids_ordered=trajectory_snapshot_ids_ordered,
        snapshot_versions=snapshot_versions,
    )
    trajectory_drift_rows_by_id = {
        str(row.get("trajectory_id")): row
        for row in trajectory_drift_rows
        if str(row.get("trajectory_id") or "").strip()
    }

    failed_rows_input = [
        row for row in sample_rows if str(row.get("judge_verdict", "")).lower() in {"partial", "incorrect"}
    ]
    configured_m = int(run_meta.get("m") or 0)
    judge_diagnostics = _build_judge_diagnostics(sample_rows)
    answer_synthesis_diagnostics = _build_answer_synthesis_diagnostics(sample_rows)
    retrieval_reflection_diagnostics = _build_retrieval_reflection_diagnostics(sample_rows)
    force_recall_counts = dict(indexes.get("force_recall_counts", {}))
    memory_summary = dict(summary_payload.get("memory") or {})
    for key in [
        "zero_claim_episodic_candidate_count",
        "zero_claim_episodic_persisted_count",
        "zero_claim_low_salience_skipped_count",
    ]:
        force_recall_counts[key] = max(
            int(force_recall_counts.get(key, 0) or 0),
            int(memory_summary.get(key, 0) or 0),
        )
    force_recall_counts["zero_claim_episodic_candidate_count"] = max(
        int(force_recall_counts.get("zero_claim_episodic_candidate_count", 0) or 0),
        int(force_recall_counts.get("zero_claim_episodic_persisted_count", 0) or 0)
        + int(force_recall_counts.get("zero_claim_low_salience_skipped_count", 0) or 0),
    )

    failed_rows: list[dict[str, Any]] = []
    query_outcomes: list[dict[str, Any]] = []
    reason_counts = Counter({reason: 0 for reason in FAILURE_REASON_ORDER})
    flag_counts = Counter({flag: 0 for flag in DIAGNOSTIC_FLAG_ORDER})
    expansion_recovered_count = 0
    grounded_hit_count = 0
    memory_hit_count = 0
    coarse_hit_count = 0
    fine_hit_count = 0
    expanded_hit_count = 0
    coarse_ranks: list[int] = []
    fine_ranks: list[int] = []
    page_recall_values: list[float] = []
    all_gold_pages_in_top_t_count = 0
    all_gold_trajectories_in_selected_pages_count = 0
    all_gold_trajectories_in_final_top_k_after_page_routing_count = 0
    memory_preservation_counts = Counter(
        {
            "gold_answer_fully_present_in_claims": 0,
            "gold_answer_fully_present_in_trajectory_summaries": 0,
            "gold_answer_fully_present_in_wiki_pages": 0,
            "gold_item_present_in_claims": 0,
            "gold_item_present_in_wiki": 0,
            "gold_item_present_in_historical_card": 0,
            "gold_item_present_in_latest_summary": 0,
            "gold_item_present_in_wiki_after_historical_card": 0,
            "trajectory_drift_suspected": 0,
            "index_only_gold_trajectory_count": 0,
            "gold_item_missing_but_umbrella_present": 0,
            "trajectory_storage_miss": 0,
            "summary_compression_miss": 0,
            "wiki_compilation_compression_miss": 0,
            "gold_surface_present_in_raw": 0,
            "gold_surface_preserved_in_claims": 0,
            "gold_surface_preserved_in_exact_terms": 0,
            "gold_surface_preserved_in_historical_card": 0,
            "gold_surface_preserved_in_wiki": 0,
        }
    )
    stage_counts = Counter(
        {
            "page_routing_failed": 0,
            "selected_pages_failed_to_cover_all_gold_trajectories": 0,
            "trajectory_retrieval_failed_after_page_routing": 0,
            "all_gold_refs_grounded": 0,
            "strict_retrieval_success_but_answer_wrong": 0,
        }
    )
    compaction_counts = Counter(
        {
            "pre_compaction_expansion_could_cover_all_gold_refs": 0,
            "snapshot_compaction_miss": 0,
            "source_compaction_miss": 0,
        }
    )
    split_case_redundant_trajectory_ratios: list[float] = []
    failed_selected_page_cluster_counts: list[float] = []
    force_recall_failed_query_count = 0
    low_salience_failed_query_count = 0
    gold_refs_in_forced_memory_count = 0
    gold_refs_in_low_salience_memory_count = 0

    failed_rows_by_query: dict[tuple[str, str], dict[str, Any]] = {}

    for row in failed_rows_input:
        sample_id = str(row["sample_id"])
        query_task_id = str(row["query_task_id"])
        question = str(row["question"])
        gold_answer = row.get("gold_answer")
        query_metadata = dict(row.get("metadata", {}).get("query_metadata", {}))
        answer_metadata = dict(row.get("metadata", {}).get("answer_metadata", {}) or {})
        judge_fields = _judge_fields_from_row(row)
        semantic_f1_fields = _semantic_f1_fields_from_row(row)
        evaluation_filter_fields = _evaluation_filter_fields_from_row(row)
        reflection_fields = _reflection_fields_from_row(row)
        answer_synthesis_fields = _answer_synthesis_fields_from_row(row)
        llm_usage_fields = _llm_usage_fields_from_row(row)
        judge_leniency_fields = _judge_leniency_fields(row)
        gold_refs = set(_gold_refs_from_query_metadata(query_metadata))
        retrieval_event = retrieval_events.get(str(row.get("retrieval_event_id")), {})
        sample_memory_refs = set(sample_memory_refs_map.get(sample_id, set()))
        candidate_trajectory_refs = set(retrieval_event.get("candidate_trajectory_refs", []))
        fine_hit_refs = set(retrieval_event.get("fine_hit_refs", []))
        expanded_refs = set(retrieval_event.get("expanded_refs", []))
        raw_expanded_refs = set(retrieval_event.get("raw_expanded_refs", []))
        grounded_refs = set(retrieval_event.get("grounded_source_refs", []))
        query_entities, query_facets, query_shape = _derive_query_diagnostics(
            sample_id=sample_id,
            question=question,
            retrieval_event=retrieval_event,
            sample_entity_lexicons=sample_entity_lexicons,
        )
        query_facet_tags = {str(tag) for tag in list(query_facets.get("tags") or []) if str(tag).strip()}

        has_memory_hit = bool(gold_refs & sample_memory_refs)
        has_coarse_hit = bool(gold_refs & candidate_trajectory_refs)
        has_fine_hit = bool(gold_refs & fine_hit_refs)
        has_expanded_hit = bool(gold_refs & expanded_refs)
        has_grounded_hit = bool(gold_refs & grounded_refs)
        temporal_grounding_fields = _temporal_grounding_fields(
            question=question,
            gold_refs=gold_refs,
            grounded_refs=grounded_refs,
            retrieval_metadata=dict(retrieval_event.get("metadata") or {}),
            raw_messages_by_id=raw_messages_by_id,
            raw_message_ids_by_ref=raw_message_ids_by_ref,
        )
        raw_rescue_gold_hit = bool(set(reflection_fields["raw_rescue_source_refs"]) & gold_refs)
        reflection_fields["raw_rescue_compensated_memory_gap"] = bool(
            raw_rescue_gold_hit and not has_memory_hit
        )

        base_reason = _classify_failure(
            gold_refs=gold_refs,
            sample_memory_refs=sample_memory_refs,
            candidate_trajectory_refs=candidate_trajectory_refs,
            fine_hit_refs=fine_hit_refs,
            expanded_refs=expanded_refs,
            grounded_refs=grounded_refs,
        )
        reason = (
            _classify_grounded_answer_failure(
                gold_answer=gold_answer,
                answer_text=row.get("answer_text"),
                query_shape=query_shape,
                answer_metadata=answer_metadata,
            )
            if base_reason == "answer_model_error"
            else base_reason
        )

        gold_trajectory_ids = sorted(
            trajectory_id
            for trajectory_id in sample_to_trajectories.get(sample_id, set())
            if trajectory_refs.get(trajectory_id, set()) & gold_refs
        )
        gold_trajectory_id_set = set(gold_trajectory_ids)
        gold_page_ids = sorted(
            page_id
            for page_id in sample_to_pages.get(sample_id, set())
            if page_types.get(page_id) != "index"
            and gold_trajectory_id_set & set(page_to_trajectory_ids.get(page_id, []))
        )
        gold_non_index_page_trajectory_ids = {
            trajectory_id
            for page_id in gold_page_ids
            for trajectory_id in page_to_trajectory_ids.get(page_id, [])
            if trajectory_id in gold_trajectory_id_set
        }
        gold_index_page_trajectory_ids = {
            trajectory_id
            for page_id in sample_to_pages.get(sample_id, set())
            if page_types.get(page_id) == "index"
            for trajectory_id in page_to_trajectory_ids.get(page_id, [])
            if trajectory_id in gold_trajectory_id_set
        }
        index_only_gold_trajectory_count = len(
            gold_index_page_trajectory_ids - gold_non_index_page_trajectory_ids
        )
        gold_snapshot_ids = sorted(
            snapshot_id
            for snapshot_id, refs in snapshot_refs.items()
            if refs & gold_refs and trajectory_metadata.get(indexes["snapshot_to_trajectory"].get(snapshot_id, ""), {}) is not None
            and indexes["trajectory_to_sample"].get(indexes["snapshot_to_trajectory"].get(snapshot_id, "")) == sample_id
        )
        forced_gold_snapshot_ids = [
            snapshot_id
            for snapshot_id in gold_snapshot_ids
            if snapshot_metadata.get(snapshot_id, {}).get("forced_episodic_seed_used_v1")
        ]
        low_salience_gold_snapshot_ids = [
            snapshot_id
            for snapshot_id in gold_snapshot_ids
            if snapshot_metadata.get(snapshot_id, {}).get("low_salience_memory_v1")
        ]
        gold_refs_in_forced_memory = sorted(
            set().union(*(snapshot_refs.get(snapshot_id, set()) for snapshot_id in forced_gold_snapshot_ids)) & gold_refs
        )
        gold_refs_in_low_salience_memory = sorted(
            set().union(*(snapshot_refs.get(snapshot_id, set()) for snapshot_id in low_salience_gold_snapshot_ids)) & gold_refs
        )
        if gold_refs_in_forced_memory:
            force_recall_failed_query_count += 1
            gold_refs_in_forced_memory_count += len(gold_refs_in_forced_memory)
        if gold_refs_in_low_salience_memory:
            low_salience_failed_query_count += 1
            gold_refs_in_low_salience_memory_count += len(gold_refs_in_low_salience_memory)

        gold_claims = [
            claim
            for trajectory_id in gold_trajectory_ids
            for claim in claims_by_trajectory.get(trajectory_id, [])
            if claim["source_refs"] & gold_refs
        ]
        gold_claim_texts = [claim["text"] for claim in gold_claims]
        gold_claim_facets = [facet for claim in gold_claims for facet in claim["facets"]]
        gold_display_values = [
            value
            for claim in gold_claims
            for values in dict(claim.get("display_signals") or {}).values()
            for value in list(values or [])
        ]
        gold_answer_present_in_display_signals = _surface_supports_gold_answer(gold_answer, gold_display_values)
        claim_text_llm_failed = any(
            bool((claim.get("metadata") or {}).get("claim_text_llm_failed_v1")) for claim in gold_claims
        )
        signal_extraction_failed = any(
            bool((claim.get("metadata") or {}).get("signal_extraction_failed_v1")) for claim in gold_claims
        )
        signal_validation_discarded = [
            item
            for claim in gold_claims
            for item in list((claim.get("metadata") or {}).get("signal_extraction_discarded_v1") or [])
        ]
        summary_item_fragment_count = sum(
            count_fragment_lines(str(trajectory_metadata.get(trajectory_id, {}).get("retrieval_summary_text") or ""))
            for trajectory_id in gold_trajectory_ids
        )
        wiki_item_fragment_count = sum(
            count_fragment_lines(page_markdown_by_id.get(page_id, ""))
            for page_id in gold_page_ids
        )
        claim_preservation_fields = _claim_preservation_metadata_fields(gold_claims)
        source_surface_fields = _source_surface_preservation_fields(
            gold_answer=gold_answer,
            gold_refs=gold_refs,
            gold_claims=gold_claims,
            gold_trajectory_ids=gold_trajectory_ids,
            gold_page_ids=gold_page_ids,
            trajectory_metadata=trajectory_metadata,
            page_markdown_by_id=page_markdown_by_id,
            raw_message_ids_by_ref=raw_message_ids_by_ref,
            raw_messages_by_id=raw_messages_by_id,
        )
        gold_facet_relations = {
            str(facet.get("relation")).strip()
            for facet in gold_claim_facets
            if str(facet.get("relation") or "").strip()
        }
        gold_summary_texts = [
            collapse_whitespace(str((trajectory_metadata.get(trajectory_id, {}) or {}).get("retrieval_summary_text") or ""))
            for trajectory_id in gold_trajectory_ids
            if collapse_whitespace(str((trajectory_metadata.get(trajectory_id, {}) or {}).get("retrieval_summary_text") or ""))
        ]
        gold_wiki_texts = [
            collapse_whitespace(str(page_markdown_by_id.get(page_id) or ""))
            for page_id in gold_page_ids
            if collapse_whitespace(str(page_markdown_by_id.get(page_id) or ""))
        ]
        gold_item_present_in_claims = bool(
            gold_answer_present_in_display_signals
            or _surface_supports_gold_answer(gold_answer, gold_claim_texts)
            or _surface_supports_gold_answer(
                gold_answer,
                [str(facet.get("value") or "") for facet in gold_claim_facets]
                + [str(facet.get("value_span") or "") for facet in gold_claim_facets],
            )
        )
        gold_item_present_in_wiki = _surface_supports_gold_answer(gold_answer, gold_wiki_texts)
        gold_item_missing_but_umbrella_present = _gold_item_missing_but_umbrella_present(
            gold_answer=gold_answer,
            query_shape=query_shape,
            gold_claim_texts=gold_claim_texts,
            gold_summary_texts=gold_summary_texts,
            gold_wiki_texts=gold_wiki_texts,
            gold_item_present_in_claims=gold_item_present_in_claims,
        )

        coarse_ranked_ids = list(retrieval_event.get("coarse_ranked_ids", []))
        top_t_page_ids = _resolve_top_t_page_ids(retrieval_event)
        routing_text_marker_fields = _routing_text_marker_fields_for_pages(
            top_t_page_ids,
            page_metadata_by_id,
        )
        gold_pages_in_top_t = [page_id for page_id in top_t_page_ids if page_id in set(gold_page_ids)]
        page_routing_recall = (
            _safe_rate(len(gold_pages_in_top_t), len(gold_page_ids))
            if gold_page_ids
            else None
        )
        if page_routing_recall is not None:
            page_recall_values.append(float(page_routing_recall))
        if gold_page_ids and len(gold_pages_in_top_t) == len(gold_page_ids):
            all_gold_pages_in_top_t_count += 1
        top_k_trajectory_ids = _resolve_top_k_trajectory_ids(retrieval_event)
        retrieval_metadata = dict(retrieval_event.get("metadata") or {})
        selection_pool_trajectory_ids = _selection_pool_ids_from_metadata(retrieval_metadata)
        selected_page_trajectory_ids = [
            str(item)
            for item in list(retrieval_metadata.get("selected_page_trajectory_ids") or [])
            if str(item).strip()
        ]
        if not selected_page_trajectory_ids:
            selected_page_trajectory_ids = sorted(
                {
                    trajectory_id
                    for page_id in top_t_page_ids
                    for trajectory_id in page_to_trajectory_ids.get(page_id, [])
                }
            )
        wiki_coverage_fields = _sample_wiki_coverage_fields(
            sample_id=sample_id,
            sample_to_trajectories=sample_to_trajectories,
            sample_to_pages=sample_to_pages,
            page_types=page_types,
            page_to_trajectory_ids=page_to_trajectory_ids,
            page_metadata_by_id=page_metadata_by_id,
        )
        page_granularity_fields = _page_granularity_fields_from_retrieval_metadata(retrieval_metadata)
        if not page_granularity_fields.get("page_granularity_metadata_available"):
            page_granularity_fields = _page_granularity_fields_from_page_ids(
                top_t_page_ids,
                page_to_trajectory_ids,
            )
        wiki_fragmentation_fields = _wiki_fragmentation_fields_from_retrieval_metadata(retrieval_metadata)
        if wiki_fragmentation_fields.get("wiki_non_index_page_count") is None:
            wiki_fragmentation_fields = _wiki_fragmentation_fields_from_page_tables(
                sample_id,
                sample_to_pages,
                page_to_trajectory_ids,
                page_metadata_by_id,
                page_types,
            )
        page_family_routing_fields = _page_family_fields_from_retrieval_metadata(retrieval_metadata)
        all_gold_trajectories_in_selected_pages = bool(
            gold_trajectory_ids and gold_trajectory_id_set <= set(selected_page_trajectory_ids)
        )
        all_gold_trajectories_in_final_top_k_after_page_routing = bool(
            gold_trajectory_ids and gold_trajectory_id_set <= set(top_k_trajectory_ids)
        )
        all_gold_trajectories_in_selected_pages_count += int(all_gold_trajectories_in_selected_pages)
        all_gold_trajectories_in_final_top_k_after_page_routing_count += int(
            all_gold_trajectories_in_final_top_k_after_page_routing
        )
        selected_trajectory_clusters = [
            tuple(
                sorted(
                    normalize_entity_key(value)
                    for value in list((trajectory_metadata.get(trajectory_id, {}) or {}).get("entity_mentions") or [])
                    if str(value).strip()
                )
            )
            for trajectory_id in top_k_trajectory_ids
        ]
        top_k_redundant_entity_cluster_ratio = _entity_cluster_ratio(selected_trajectory_clusters)
        selected_page_clusters = {
            tuple(
                sorted(
                    normalize_entity_key(value)
                    for trajectory_id in page_to_trajectory_ids.get(page_id, [])
                    for value in list((trajectory_metadata.get(trajectory_id, {}) or {}).get("entity_mentions") or [])
                    if str(value).strip()
                )
            )
            for page_id in top_t_page_ids
        }
        selected_page_cluster_count = len({cluster for cluster in selected_page_clusters if cluster})
        if top_k_redundant_entity_cluster_ratio is not None and len(gold_trajectory_ids) >= 2:
            split_case_redundant_trajectory_ratios.append(float(top_k_redundant_entity_cluster_ratio))
        failed_selected_page_cluster_counts.append(float(selected_page_cluster_count))
        pre_compaction_expansion_could_cover_all_gold_refs = bool(gold_refs) and gold_refs <= raw_expanded_refs
        snapshot_compaction_miss = bool(
            gold_refs
            and gold_refs <= raw_expanded_refs
            and not gold_refs <= expanded_refs
        )
        source_compaction_miss = bool(
            gold_refs
            and gold_refs <= expanded_refs
            and not gold_refs <= grounded_refs
        )
        gold_refs_lost_by_snapshot_compaction = sorted((raw_expanded_refs & gold_refs) - expanded_refs)
        gold_refs_lost_by_source_compaction = sorted((expanded_refs & gold_refs) - grounded_refs)
        compaction_counts["pre_compaction_expansion_could_cover_all_gold_refs"] += int(
            pre_compaction_expansion_could_cover_all_gold_refs
        )
        compaction_counts["snapshot_compaction_miss"] += int(snapshot_compaction_miss)
        compaction_counts["source_compaction_miss"] += int(source_compaction_miss)
        gold_trajectory_rank = _rank_in_order(coarse_ranked_ids, set(gold_trajectory_ids))
        gold_snapshot_rank = _rank_in_order(
            list(retrieval_event.get("snapshot_hit_ids", [])),
            set(gold_snapshot_ids),
        )
        if gold_trajectory_rank is not None:
            coarse_ranks.append(gold_trajectory_rank)
        if gold_snapshot_rank is not None:
            fine_ranks.append(gold_snapshot_rank)

        entity_linked_snapshot_ids = list(retrieval_event.get("entity_linked_snapshot_ids", []))
        entity_linked_trajectory_ids = list(retrieval_event.get("entity_linked_trajectory_ids", []))
        entity_linked_refs = set(retrieval_event.get("entity_linked_refs", []))
        gold_entity_linked_hit = bool(gold_refs & entity_linked_refs)
        gold_expanded_only_hit = (not has_fine_hit) and has_expanded_hit

        preservation_and_stage = _compute_preservation_and_stage_fields(
            gold_answer=gold_answer,
            gold_refs=gold_refs,
            gold_claim_texts=gold_claim_texts,
            gold_claim_facets=gold_claim_facets,
            gold_trajectory_ids=gold_trajectory_ids,
            gold_page_ids=gold_page_ids,
            top_t_page_ids=top_t_page_ids,
            selected_page_trajectory_ids=selected_page_trajectory_ids,
            top_k_trajectory_ids=top_k_trajectory_ids,
            grounded_refs=grounded_refs,
            trajectory_metadata=trajectory_metadata,
            page_markdown_by_id=page_markdown_by_id,
            snapshot_compaction_miss=snapshot_compaction_miss,
            source_compaction_miss=source_compaction_miss,
        )
        for key in [
            "gold_answer_fully_present_in_claims",
            "gold_answer_fully_present_in_trajectory_summaries",
            "gold_answer_fully_present_in_wiki_pages",
            "gold_item_present_in_historical_card",
            "gold_item_present_in_latest_summary",
            "gold_item_present_in_wiki_after_historical_card",
            "trajectory_drift_suspected",
            "trajectory_storage_miss",
            "summary_compression_miss",
            "wiki_compilation_compression_miss",
        ]:
            memory_preservation_counts[key] += int(bool(preservation_and_stage[key]))
        memory_preservation_counts["index_only_gold_trajectory_count"] += int(
            index_only_gold_trajectory_count > 0
        )
        memory_preservation_counts["gold_item_present_in_claims"] += int(gold_item_present_in_claims)
        memory_preservation_counts["gold_item_present_in_wiki"] += int(gold_item_present_in_wiki)
        memory_preservation_counts["gold_item_missing_but_umbrella_present"] += int(
            gold_item_missing_but_umbrella_present
        )
        for key in [
            "gold_surface_present_in_raw",
            "gold_surface_preserved_in_claims",
            "gold_surface_preserved_in_exact_terms",
            "gold_surface_preserved_in_historical_card",
            "gold_surface_preserved_in_wiki",
        ]:
            memory_preservation_counts[key] += int(bool(source_surface_fields.get(key)))
        for key in stage_counts:
            stage_counts[key] += int(bool(preservation_and_stage[key]))
        supported_by_evidence = preservation_and_stage["gold_answer_fully_present_in_claims"]
        supported_by_facet = any(
            _facet_value_supports_answer(facet, _normalize_text(gold_answer))
            for facet in gold_claim_facets
        )
        facet_present_for_gold_ref = bool(gold_claim_facets)
        exact_facet_missing = bool(
            has_memory_hit
            and query_facet_tags
            and not supported_by_facet
        )
        gold_ref_present_but_no_supported_facet = bool(
            has_memory_hit
            and query_facet_tags
            and not (gold_facet_relations & query_facet_tags)
        )
        gold_ref_present_but_claim_generalized = bool(
            has_memory_hit
            and _normalize_text(gold_answer)
            and not supported_by_evidence
        )

        gold_trajectory_count = len(gold_trajectory_ids)
        gold_refs_split_across_trajectories = gold_trajectory_count > 1
        min_gold_covering_trajectory_count = _min_covering_trajectory_count(
            gold_refs, gold_trajectory_ids, trajectory_refs
        )
        top_k_coverage = _compute_top_k_coverage(
            gold_refs=gold_refs,
            gold_trajectory_ids=gold_trajectory_ids,
            top_k_trajectory_ids=top_k_trajectory_ids,
            selection_pool_trajectory_ids=selection_pool_trajectory_ids,
            trajectory_refs=trajectory_refs,
        )
        page_cutoff_fields = _compute_page_cutoff_fields(
            gold_page_ids=gold_page_ids,
            gold_trajectory_ids=gold_trajectory_ids,
            top_t_page_ids=top_t_page_ids,
            page_to_trajectory_ids=page_to_trajectory_ids,
        )
        trajectory_cutoff_fields = _compute_trajectory_cutoff_fields(
            gold_refs=gold_refs,
            gold_trajectory_ids=gold_trajectory_ids,
            top_k_trajectory_ids=top_k_trajectory_ids,
            trajectory_refs=trajectory_refs,
        )
        offline_parameter_fields = _compute_offline_parameter_fields(
            retrieval_metadata=retrieval_metadata,
            gold_refs=gold_refs,
            gold_page_ids=gold_page_ids,
            gold_trajectory_ids=gold_trajectory_ids,
            top_t_page_ids=top_t_page_ids,
            top_k_trajectory_ids=top_k_trajectory_ids,
            page_to_trajectory_ids=page_to_trajectory_ids,
            trajectory_refs=trajectory_refs,
        )
        direct_trajectory_ablation_fields = _compute_direct_trajectory_ablation_fields(
            sample_id=sample_id,
            question=question,
            gold_refs=gold_refs,
            gold_trajectory_ids=gold_trajectory_ids,
            routed_top_k_trajectory_ids=top_k_trajectory_ids,
            selected_page_trajectory_ids=selected_page_trajectory_ids,
            query_entities=query_entities,
            query_facets=query_facets,
            query_shape=query_shape,
            top_k=int(retrieval_event.get("top_k") or len(top_k_trajectory_ids) or 0),
            sample_to_trajectories=sample_to_trajectories,
            trajectory_metadata=trajectory_metadata,
            claims_by_trajectory=claims_by_trajectory,
            trajectory_refs=trajectory_refs,
            trajectory_lengths=trajectory_lengths,
        )
        trajectory_length_fields = _compute_trajectory_length_query_fields(
            gold_trajectory_ids=gold_trajectory_ids,
            gold_snapshot_ids=gold_snapshot_ids,
            configured_m=configured_m,
            trajectory_lengths=trajectory_lengths,
            snapshot_versions=snapshot_versions,
            snapshot_rank_in_trajectory=snapshot_rank_in_trajectory,
        )
        trajectory_drift_fields = trajectory_drift_fields_for_gold_trajectories(
            gold_trajectory_ids,
            trajectory_drift_rows_by_id,
        )

        expanded_trajectory_ids = {
            indexes["snapshot_to_trajectory"][snapshot_id]
            for snapshot_id in list(retrieval_event.get("expanded_snapshot_ids", []))
            if snapshot_id in indexes["snapshot_to_trajectory"]
        }
        missing_gold_trajectories = [trajectory_id for trajectory_id in gold_trajectory_ids if trajectory_id not in expanded_trajectory_ids]
        query_entity_keys = {normalize_entity_key(entity) for entity in query_entities}
        entity_linked_could_help = False
        if gold_refs_split_across_trajectories and missing_gold_trajectories and query_entity_keys:
            for trajectory_id in missing_gold_trajectories:
                metadata = trajectory_metadata.get(trajectory_id, {})
                entity_mentions = {
                    normalize_entity_key(str(value))
                    for value in list(metadata.get("entity_mentions") or [])
                    if str(value).strip()
                }
                if entity_mentions & query_entity_keys:
                    entity_linked_could_help = True
                    break

        diagnostic_flags = [
            flag_name
            for flag_name, enabled in [
                ("facet_present_for_gold_ref", facet_present_for_gold_ref),
                ("exact_facet_missing", exact_facet_missing),
                ("gold_ref_present_but_claim_generalized", gold_ref_present_but_claim_generalized),
                ("gold_ref_present_but_no_supported_facet", gold_ref_present_but_no_supported_facet),
                ("gold_refs_split_across_trajectories", gold_refs_split_across_trajectories),
                ("entity_linked_could_help", entity_linked_could_help),
                ("entity_linked_added_relevant_snapshot", gold_entity_linked_hit),
                ("gold_expanded_only_hit", gold_expanded_only_hit),
                ("gold_entity_linked_hit", gold_entity_linked_hit),
                ("gold_grounded_hit", has_grounded_hit),
                ("raw_candidate_extraction_miss", claim_preservation_fields["raw_candidate_extraction_miss"]),
                ("claim_preservation_audit_miss", claim_preservation_fields["claim_preservation_audit_miss"]),
                ("claim_source_link_miss", claim_preservation_fields["claim_source_link_miss"]),
            ]
            if enabled
        ]
        for flag in diagnostic_flags:
            flag_counts[flag] += 1

        memory_hit_count += int(has_memory_hit)
        coarse_hit_count += int(has_coarse_hit)
        fine_hit_count += int(has_fine_hit)
        expanded_hit_count += int(has_expanded_hit)
        grounded_hit_count += int(has_grounded_hit)
        if gold_expanded_only_hit:
            expansion_recovered_count += 1

        reason_counts[reason] += 1
        failed_row = {
            "sample_id": sample_id,
            "query_task_id": query_task_id,
            "question": question,
            "gold_answer": gold_answer,
            "answer_text": row.get("answer_text"),
            "judge_verdict": str(row.get("judge_verdict", "")),
            "judge_score": row.get("judge_score"),
            "judge_rationale": row.get("judge_rationale"),
            "reason": reason,
            "gold_refs": sorted(gold_refs),
            "query_entities": query_entities,
            "query_shape": query_shape,
            "query_shape_preference_like": bool(str(query_shape.get("item_family") or "").casefold() == "preference"),
            "coverage_triggered": _query_shape_requires_coverage(query_shape),
            "query_facets": {
                "tags": sorted(query_facet_tags),
                "values": sorted(set(str(value) for value in list(query_facets.get("values") or []))),
            },
            "gold_page_ids": gold_page_ids,
            "top_t_page_ids": top_t_page_ids,
            "gold_pages_in_top_t": gold_pages_in_top_t,
            "page_routing_recall": page_routing_recall,
            "selected_page_trajectory_ids": selected_page_trajectory_ids,
            "selected_page_universe_size": len(selected_page_trajectory_ids),
            **routing_text_marker_fields,
            **_broad_entity_fields_from_retrieval_metadata(retrieval_metadata),
            **page_granularity_fields,
            **wiki_fragmentation_fields,
            **page_family_routing_fields,
            **wiki_coverage_fields,
            "index_fallback_used": bool(retrieval_metadata.get("index_fallback_used")),
            "index_fallback_reason": retrieval_metadata.get("index_fallback_reason"),
            "selected_page_universe_size_before_index_fallback": retrieval_metadata.get(
                "selected_page_universe_size_before_index_fallback"
            ),
            "selected_page_universe_size_after_index_fallback": retrieval_metadata.get(
                "selected_page_universe_size_after_index_fallback"
            ),
            "index_fallback_candidate_count": retrieval_metadata.get("index_fallback_candidate_count"),
            "index_fallback_added_trajectory_ids": list(
                retrieval_metadata.get("index_fallback_added_trajectory_ids") or []
            ),
            "all_gold_trajectories_in_selected_pages": preservation_and_stage["all_gold_trajectories_in_selected_pages"],
            "all_gold_trajectories_in_final_top_k_after_page_routing": preservation_and_stage["all_gold_trajectories_in_final_top_k_after_page_routing"],
            "candidate_trajectory_ids": list(retrieval_event.get("trajectory_ids", [])),
            "top_k_trajectory_ids": list(top_k_coverage["top_k_trajectory_ids"]),
            "trajectory_selection_pool_ids": list(top_k_coverage["selection_pool_trajectory_ids"]),
            "trajectory_selection_pool_available": top_k_coverage["trajectory_selection_pool_available"],
            "trajectory_selection_pool_size": (
                len(top_k_coverage["selection_pool_trajectory_ids"])
                if top_k_coverage["trajectory_selection_pool_available"]
                else None
            ),
            "trajectory_selection_strategy": retrieval_metadata.get("trajectory_selection_strategy"),
            **_family_ranking_fields_from_metadata(
                retrieval_metadata=retrieval_metadata,
                gold_trajectory_ids=gold_trajectory_ids,
                top_k_trajectory_ids=list(top_k_coverage["top_k_trajectory_ids"]),
                selection_pool_trajectory_ids=list(top_k_coverage["selection_pool_trajectory_ids"]),
            ),
            "speaker_grounding_suspect_count": int(
                retrieval_metadata.get("answer_context_suppressed_speaker_grounding_suspect_claim_count") or 0
            ),
            "snapshot_hit_ids": list(retrieval_event.get("snapshot_hit_ids", [])),
            "raw_expanded_snapshot_ids": list(retrieval_event.get("raw_expanded_snapshot_ids", [])),
            "raw_expanded_refs": sorted(raw_expanded_refs),
            "expanded_snapshot_ids": list(retrieval_event.get("expanded_snapshot_ids", [])),
            "entity_linked_trajectory_ids": entity_linked_trajectory_ids,
            "entity_linked_snapshot_ids": entity_linked_snapshot_ids,
            "grounded_source_refs": sorted(grounded_refs),
            "pre_compaction_expansion_could_cover_all_gold_refs": pre_compaction_expansion_could_cover_all_gold_refs,
            "snapshot_compaction_miss": snapshot_compaction_miss,
            "source_compaction_miss": source_compaction_miss,
            "gold_refs_lost_by_snapshot_compaction": gold_refs_lost_by_snapshot_compaction,
            "gold_refs_lost_by_source_compaction": gold_refs_lost_by_source_compaction,
            "has_memory_hit": has_memory_hit,
            "has_coarse_hit": has_coarse_hit,
            "has_fine_hit": has_fine_hit,
            "has_expanded_hit": has_expanded_hit,
            "has_grounded_hit": has_grounded_hit,
            "gold_trajectory_rank": gold_trajectory_rank,
            "gold_snapshot_rank": gold_snapshot_rank,
            "gold_expanded_only_hit": gold_expanded_only_hit,
            "gold_entity_linked_hit": gold_entity_linked_hit,
            "gold_grounded_hit": has_grounded_hit,
            "coarse_candidate_count": len(coarse_ranked_ids) or len(list(retrieval_event.get("trajectory_ids", []))),
            "gold_trajectory_ids": gold_trajectory_ids,
            "gold_trajectory_ids_in_top_k": list(top_k_coverage["gold_trajectory_ids_in_top_k"]),
            "missing_gold_trajectory_ids_from_top_k": list(top_k_coverage["missing_gold_trajectory_ids_from_top_k"]),
            "gold_trajectory_ids_in_selection_pool": list(top_k_coverage["gold_trajectory_ids_in_selection_pool"]),
            "missing_gold_trajectory_ids_from_selection_pool": list(
                top_k_coverage["missing_gold_trajectory_ids_from_selection_pool"]
            ),
            "gold_trajectory_count_in_top_k": top_k_coverage["gold_trajectory_count_in_top_k"],
            "gold_trajectory_count_in_selection_pool": top_k_coverage["gold_trajectory_count_in_selection_pool"],
            "gold_trajectory_recall_at_k": top_k_coverage["gold_trajectory_recall_at_k"],
            "gold_trajectory_recall_in_selection_pool": top_k_coverage["gold_trajectory_recall_in_selection_pool"],
            "selection_pool_can_cover_all_gold_trajectories": top_k_coverage[
                "selection_pool_can_cover_all_gold_trajectories"
            ],
            "top_k_gold_ref_coverage_count": top_k_coverage["top_k_gold_ref_coverage_count"],
            "top_k_gold_ref_coverage_rate": top_k_coverage["top_k_gold_ref_coverage_rate"],
            "top_k_can_cover_all_gold_refs": top_k_coverage["top_k_can_cover_all_gold_refs"],
            "top_k_covering_trajectory_count": top_k_coverage["top_k_covering_trajectory_count"],
            "gold_snapshot_ids": gold_snapshot_ids,
            "gold_forced_snapshot_ids": forced_gold_snapshot_ids,
            "gold_low_salience_snapshot_ids": low_salience_gold_snapshot_ids,
            "gold_refs_in_forced_memory": gold_refs_in_forced_memory,
            "gold_refs_in_low_salience_memory": gold_refs_in_low_salience_memory,
            "gold_ref_forced_memory_hit": bool(gold_refs_in_forced_memory),
            "gold_ref_low_salience_memory_hit": bool(gold_refs_in_low_salience_memory),
            "gold_claim_texts": gold_claim_texts,
            "gold_claim_facets": gold_claim_facets,
            "gold_answer_present_in_display_signals": gold_answer_present_in_display_signals,
            "gold_item_present_in_claims": gold_item_present_in_claims,
            "gold_item_present_in_wiki": gold_item_present_in_wiki,
            "gold_item_missing_but_umbrella_present": gold_item_missing_but_umbrella_present,
            "index_only_gold_trajectory_count": index_only_gold_trajectory_count,
            "answer_postcheck_used": bool(answer_metadata.get("answer_postcheck_used")),
            "answer_postcheck_issue": answer_metadata.get("answer_postcheck_issue"),
            **temporal_grounding_fields,
            **answer_synthesis_fields,
            **llm_usage_fields,
            **judge_leniency_fields,
            **reflection_fields,
            "claim_text_llm_failed": claim_text_llm_failed,
            "signal_extraction_failed": signal_extraction_failed,
            "signal_validation_discarded": signal_validation_discarded,
            "summary_item_fragment_count": summary_item_fragment_count,
            "wiki_item_fragment_count": wiki_item_fragment_count,
            "gold_trajectory_count": gold_trajectory_count,
            "gold_refs_split_across_trajectories": gold_refs_split_across_trajectories,
            **_metadata_term_fields_for_trajectories(gold_trajectory_ids, trajectory_metadata),
            "min_gold_covering_trajectory_count": min_gold_covering_trajectory_count,
            "entity_linked_could_help": entity_linked_could_help,
            "entity_linked_added_relevant_snapshot": gold_entity_linked_hit,
            "top_k_redundant_entity_cluster_ratio": top_k_redundant_entity_cluster_ratio,
            "selected_page_cluster_count": selected_page_cluster_count,
            "diagnostic_flags": diagnostic_flags,
            **claim_preservation_fields,
            **source_surface_fields,
            **preservation_and_stage,
            **page_cutoff_fields,
            **trajectory_cutoff_fields,
            **offline_parameter_fields,
            **direct_trajectory_ablation_fields,
            **trajectory_length_fields,
            **trajectory_drift_fields,
            **semantic_f1_fields,
            **evaluation_filter_fields,
            **judge_fields,
        }
        failed_rows.append(failed_row)
        failed_rows_by_query[(sample_id, query_task_id)] = failed_row

    for row in sample_rows:
        key = (str(row["sample_id"]), str(row["query_task_id"]))
        failed_row = failed_rows_by_query.get(key)
        sample_id = key[0]
        judge_fields = _judge_fields_from_row(row)
        semantic_f1_fields = _semantic_f1_fields_from_row(row)
        evaluation_filter_fields = _evaluation_filter_fields_from_row(row)
        reflection_fields = _reflection_fields_from_row(row)
        answer_synthesis_fields = _answer_synthesis_fields_from_row(row)
        llm_usage_fields = _llm_usage_fields_from_row(row)
        judge_leniency_fields = _judge_leniency_fields(row)
        judge_verdict = str(row.get("judge_verdict", "")).lower()
        query_metadata = dict(row.get("metadata", {}).get("query_metadata", {}))
        answer_metadata = dict(row.get("metadata", {}).get("answer_metadata", {}) or {})
        gold_refs = set(_gold_refs_from_query_metadata(query_metadata))
        retrieval_event = retrieval_events.get(str(row.get("retrieval_event_id")), {})
        reflection_fields["raw_rescue_compensated_memory_gap"] = bool(
            set(reflection_fields["raw_rescue_source_refs"]) & gold_refs
            and not (gold_refs & set(sample_memory_refs_map.get(sample_id, set())))
        )
        query_entities, query_facets, query_shape = _derive_query_diagnostics(
            sample_id=sample_id,
            question=str(row["question"]),
            retrieval_event=retrieval_event,
            sample_entity_lexicons=sample_entity_lexicons,
        )
        gold_trajectory_ids = sorted(
            trajectory_id
            for trajectory_id in sample_to_trajectories.get(sample_id, set())
            if trajectory_refs.get(trajectory_id, set()) & gold_refs
        )
        gold_trajectory_id_set = set(gold_trajectory_ids)
        gold_page_ids = sorted(
            page_id
            for page_id in sample_to_pages.get(sample_id, set())
            if page_types.get(page_id) != "index"
            and gold_trajectory_id_set & set(page_to_trajectory_ids.get(page_id, []))
        )
        gold_non_index_page_trajectory_ids = {
            trajectory_id
            for page_id in gold_page_ids
            for trajectory_id in page_to_trajectory_ids.get(page_id, [])
            if trajectory_id in gold_trajectory_id_set
        }
        gold_index_page_trajectory_ids = {
            trajectory_id
            for page_id in sample_to_pages.get(sample_id, set())
            if page_types.get(page_id) == "index"
            for trajectory_id in page_to_trajectory_ids.get(page_id, [])
            if trajectory_id in gold_trajectory_id_set
        }
        index_only_gold_trajectory_count = len(
            gold_index_page_trajectory_ids - gold_non_index_page_trajectory_ids
        )
        top_t_page_ids = _resolve_top_t_page_ids(retrieval_event)
        routing_text_marker_fields = _routing_text_marker_fields_for_pages(
            top_t_page_ids,
            page_metadata_by_id,
        )
        gold_pages_in_top_t = [page_id for page_id in top_t_page_ids if page_id in set(gold_page_ids)]
        retrieval_metadata = dict(retrieval_event.get("metadata") or {})
        temporal_grounding_fields = _temporal_grounding_fields(
            question=row.get("question"),
            gold_refs=gold_refs,
            grounded_refs=set(retrieval_event.get("grounded_source_refs", [])),
            retrieval_metadata=retrieval_metadata,
            raw_messages_by_id=raw_messages_by_id,
            raw_message_ids_by_ref=raw_message_ids_by_ref,
        )
        selection_pool_trajectory_ids = _selection_pool_ids_from_metadata(retrieval_metadata)
        selected_page_trajectory_ids = [
            str(item)
            for item in list(retrieval_metadata.get("selected_page_trajectory_ids") or [])
            if str(item).strip()
        ]
        if not selected_page_trajectory_ids:
            selected_page_trajectory_ids = sorted(
                {
                    trajectory_id
                    for page_id in top_t_page_ids
                    for trajectory_id in page_to_trajectory_ids.get(page_id, [])
                }
            )
        wiki_coverage_fields = _sample_wiki_coverage_fields(
            sample_id=sample_id,
            sample_to_trajectories=sample_to_trajectories,
            sample_to_pages=sample_to_pages,
            page_types=page_types,
            page_to_trajectory_ids=page_to_trajectory_ids,
            page_metadata_by_id=page_metadata_by_id,
        )
        page_granularity_fields = _page_granularity_fields_from_retrieval_metadata(retrieval_metadata)
        if not page_granularity_fields.get("page_granularity_metadata_available"):
            page_granularity_fields = _page_granularity_fields_from_page_ids(
                top_t_page_ids,
                page_to_trajectory_ids,
            )
        wiki_fragmentation_fields = _wiki_fragmentation_fields_from_retrieval_metadata(retrieval_metadata)
        if wiki_fragmentation_fields.get("wiki_non_index_page_count") is None:
            wiki_fragmentation_fields = _wiki_fragmentation_fields_from_page_tables(
                sample_id,
                sample_to_pages,
                page_to_trajectory_ids,
                page_metadata_by_id,
                page_types,
            )
        page_family_routing_fields = _page_family_fields_from_retrieval_metadata(retrieval_metadata)
        top_k_coverage = _compute_top_k_coverage(
            gold_refs=gold_refs,
            gold_trajectory_ids=gold_trajectory_ids,
            top_k_trajectory_ids=_resolve_top_k_trajectory_ids(retrieval_event),
            selection_pool_trajectory_ids=selection_pool_trajectory_ids,
            trajectory_refs=trajectory_refs,
        )
        top_k_trajectory_ids = list(top_k_coverage["top_k_trajectory_ids"])
        page_cutoff_fields = _compute_page_cutoff_fields(
            gold_page_ids=gold_page_ids,
            gold_trajectory_ids=gold_trajectory_ids,
            top_t_page_ids=top_t_page_ids,
            page_to_trajectory_ids=page_to_trajectory_ids,
        )
        trajectory_cutoff_fields = _compute_trajectory_cutoff_fields(
            gold_refs=gold_refs,
            gold_trajectory_ids=gold_trajectory_ids,
            top_k_trajectory_ids=list(top_k_coverage["top_k_trajectory_ids"]),
            trajectory_refs=trajectory_refs,
        )
        offline_parameter_fields = _compute_offline_parameter_fields(
            retrieval_metadata=retrieval_metadata,
            gold_refs=gold_refs,
            gold_page_ids=gold_page_ids,
            gold_trajectory_ids=gold_trajectory_ids,
            top_t_page_ids=top_t_page_ids,
            top_k_trajectory_ids=top_k_trajectory_ids,
            page_to_trajectory_ids=page_to_trajectory_ids,
            trajectory_refs=trajectory_refs,
        )
        direct_trajectory_ablation_fields = _compute_direct_trajectory_ablation_fields(
            sample_id=sample_id,
            question=str(row["question"]),
            gold_refs=gold_refs,
            gold_trajectory_ids=gold_trajectory_ids,
            routed_top_k_trajectory_ids=top_k_trajectory_ids,
            selected_page_trajectory_ids=selected_page_trajectory_ids,
            query_entities=query_entities,
            query_facets=query_facets,
            query_shape=query_shape,
            top_k=int(retrieval_event.get("top_k") or len(top_k_trajectory_ids) or 0),
            sample_to_trajectories=sample_to_trajectories,
            trajectory_metadata=trajectory_metadata,
            claims_by_trajectory=claims_by_trajectory,
            trajectory_refs=trajectory_refs,
            trajectory_lengths=trajectory_lengths,
        )
        gold_snapshot_ids = sorted(
            snapshot_id
            for snapshot_id, refs in snapshot_refs.items()
            if refs & gold_refs
            and indexes["snapshot_to_trajectory"].get(snapshot_id, "") in gold_trajectory_id_set
        )
        forced_gold_snapshot_ids = [
            snapshot_id
            for snapshot_id in gold_snapshot_ids
            if snapshot_metadata.get(snapshot_id, {}).get("forced_episodic_seed_used_v1")
        ]
        low_salience_gold_snapshot_ids = [
            snapshot_id
            for snapshot_id in gold_snapshot_ids
            if snapshot_metadata.get(snapshot_id, {}).get("low_salience_memory_v1")
        ]
        gold_refs_in_forced_memory = sorted(
            set().union(*(snapshot_refs.get(snapshot_id, set()) for snapshot_id in forced_gold_snapshot_ids)) & gold_refs
        )
        gold_refs_in_low_salience_memory = sorted(
            set().union(*(snapshot_refs.get(snapshot_id, set()) for snapshot_id in low_salience_gold_snapshot_ids)) & gold_refs
        )
        trajectory_length_fields = _compute_trajectory_length_query_fields(
            gold_trajectory_ids=gold_trajectory_ids,
            gold_snapshot_ids=gold_snapshot_ids,
            configured_m=configured_m,
            trajectory_lengths=trajectory_lengths,
            snapshot_versions=snapshot_versions,
            snapshot_rank_in_trajectory=snapshot_rank_in_trajectory,
        )
        trajectory_drift_fields = trajectory_drift_fields_for_gold_trajectories(
            gold_trajectory_ids,
            trajectory_drift_rows_by_id,
        )
        gold_claims = [
            claim
            for trajectory_id in gold_trajectory_ids
            for claim in claims_by_trajectory.get(trajectory_id, [])
            if claim["source_refs"] & gold_refs
        ]
        gold_claim_texts = [claim["text"] for claim in gold_claims]
        gold_claim_facets = [facet for claim in gold_claims for facet in claim["facets"]]
        gold_display_values = [
            value
            for claim in gold_claims
            for values in dict(claim.get("display_signals") or {}).values()
            for value in list(values or [])
        ]
        gold_summary_texts = [
            collapse_whitespace(str((trajectory_metadata.get(trajectory_id, {}) or {}).get("retrieval_summary_text") or ""))
            for trajectory_id in gold_trajectory_ids
            if collapse_whitespace(str((trajectory_metadata.get(trajectory_id, {}) or {}).get("retrieval_summary_text") or ""))
        ]
        gold_wiki_texts = [
            collapse_whitespace(str(page_markdown_by_id.get(page_id) or ""))
            for page_id in gold_page_ids
            if collapse_whitespace(str(page_markdown_by_id.get(page_id) or ""))
        ]
        gold_item_present_in_claims = bool(
            _surface_supports_gold_answer(row.get("gold_answer"), gold_display_values)
            or _surface_supports_gold_answer(row.get("gold_answer"), gold_claim_texts)
            or _surface_supports_gold_answer(
                row.get("gold_answer"),
                [str(facet.get("value") or "") for facet in gold_claim_facets]
                + [str(facet.get("value_span") or "") for facet in gold_claim_facets],
            )
        )
        gold_item_present_in_wiki = _surface_supports_gold_answer(row.get("gold_answer"), gold_wiki_texts)
        gold_item_missing_but_umbrella_present = _gold_item_missing_but_umbrella_present(
            gold_answer=row.get("gold_answer"),
            query_shape=query_shape,
            gold_claim_texts=gold_claim_texts,
            gold_summary_texts=gold_summary_texts,
            gold_wiki_texts=gold_wiki_texts,
            gold_item_present_in_claims=gold_item_present_in_claims,
        )
        selected_trajectory_clusters = [
            tuple(
                sorted(
                    normalize_entity_key(value)
                    for value in list((trajectory_metadata.get(trajectory_id, {}) or {}).get("entity_mentions") or [])
                    if str(value).strip()
                )
            )
            for trajectory_id in top_k_trajectory_ids
        ]
        top_k_redundant_entity_cluster_ratio = _entity_cluster_ratio(selected_trajectory_clusters)
        selected_page_clusters = {
            tuple(
                sorted(
                    normalize_entity_key(value)
                    for trajectory_id in page_to_trajectory_ids.get(page_id, [])
                    for value in list((trajectory_metadata.get(trajectory_id, {}) or {}).get("entity_mentions") or [])
                    if str(value).strip()
                )
            )
            for page_id in top_t_page_ids
        }
        selected_page_cluster_count = len({cluster for cluster in selected_page_clusters if cluster})
        claim_preservation_fields = _claim_preservation_metadata_fields(gold_claims)
        source_surface_fields = _source_surface_preservation_fields(
            gold_answer=row.get("gold_answer"),
            gold_refs=gold_refs,
            gold_claims=gold_claims,
            gold_trajectory_ids=gold_trajectory_ids,
            gold_page_ids=gold_page_ids,
            trajectory_metadata=trajectory_metadata,
            page_markdown_by_id=page_markdown_by_id,
            raw_message_ids_by_ref=raw_message_ids_by_ref,
            raw_messages_by_id=raw_messages_by_id,
        )
        preservation_and_stage = _compute_preservation_and_stage_fields(
            gold_answer=row.get("gold_answer"),
            gold_refs=gold_refs,
            gold_claim_texts=gold_claim_texts,
            gold_claim_facets=gold_claim_facets,
            gold_trajectory_ids=gold_trajectory_ids,
            gold_page_ids=gold_page_ids,
            top_t_page_ids=top_t_page_ids,
            selected_page_trajectory_ids=selected_page_trajectory_ids,
            top_k_trajectory_ids=top_k_trajectory_ids,
            grounded_refs=set(retrieval_event.get("grounded_source_refs", [])),
            trajectory_metadata=trajectory_metadata,
            page_markdown_by_id=page_markdown_by_id,
            snapshot_compaction_miss=bool(
                gold_refs
                and gold_refs <= set(retrieval_event.get("raw_expanded_refs", []))
                and not gold_refs <= set(retrieval_event.get("expanded_refs", []))
            ),
            source_compaction_miss=bool(
                gold_refs
                and gold_refs <= set(retrieval_event.get("expanded_refs", []))
                and not gold_refs <= set(retrieval_event.get("grounded_source_refs", []))
            ),
        )
        preservation_and_stage["strict_retrieval_success_but_answer_wrong"] = bool(
            preservation_and_stage["strict_retrieval_success_but_answer_wrong"]
            and judge_verdict in {"partial", "incorrect"}
        )
        query_outcomes.append(
            {
                "sample_id": key[0],
                "query_task_id": key[1],
                "question": str(row["question"]),
                "judge_verdict": str(row.get("judge_verdict", "")),
                "judge_score": row.get("judge_score"),
                "reason": (
                    failed_row["reason"]
                    if failed_row is not None
                    else ("judge_error" if judge_verdict == "judge_error" else "correct")
                ),
                "query_shape": query_shape,
                "query_shape_preference_like": bool(str(query_shape.get("item_family") or "").casefold() == "preference"),
                "coverage_triggered": _query_shape_requires_coverage(query_shape),
                "query_entities": query_entities,
                "query_facets": query_facets,
                "gold_trajectory_count": len(gold_trajectory_ids),
                "gold_trajectory_ids": gold_trajectory_ids,
                "top_k_trajectory_ids": top_k_trajectory_ids,
                **_metadata_term_fields_for_trajectories(gold_trajectory_ids, trajectory_metadata),
                "trajectory_selection_pool_ids": list(top_k_coverage["selection_pool_trajectory_ids"]),
                "trajectory_selection_pool_available": top_k_coverage["trajectory_selection_pool_available"],
                "trajectory_selection_pool_size": (
                    len(top_k_coverage["selection_pool_trajectory_ids"])
                    if top_k_coverage["trajectory_selection_pool_available"]
                    else None
                ),
                "trajectory_selection_strategy": retrieval_metadata.get("trajectory_selection_strategy"),
                "speaker_grounding_suspect_count": int(
                    retrieval_metadata.get("answer_context_suppressed_speaker_grounding_suspect_claim_count") or 0
                ),
                "gold_trajectory_count_in_top_k": top_k_coverage["gold_trajectory_count_in_top_k"],
                "gold_trajectory_count_in_selection_pool": top_k_coverage["gold_trajectory_count_in_selection_pool"],
                "gold_trajectory_recall_at_k": top_k_coverage["gold_trajectory_recall_at_k"],
                "gold_trajectory_recall_in_selection_pool": top_k_coverage["gold_trajectory_recall_in_selection_pool"],
                "selection_pool_can_cover_all_gold_trajectories": top_k_coverage[
                    "selection_pool_can_cover_all_gold_trajectories"
                ],
                "top_k_can_cover_all_gold_refs": top_k_coverage["top_k_can_cover_all_gold_refs"],
                "gold_forced_snapshot_ids": forced_gold_snapshot_ids,
                "gold_low_salience_snapshot_ids": low_salience_gold_snapshot_ids,
                "gold_refs_in_forced_memory": gold_refs_in_forced_memory,
                "gold_refs_in_low_salience_memory": gold_refs_in_low_salience_memory,
                "gold_ref_forced_memory_hit": bool(gold_refs_in_forced_memory),
                "gold_ref_low_salience_memory_hit": bool(gold_refs_in_low_salience_memory),
                "gold_page_ids": gold_page_ids,
                "top_t_page_ids": top_t_page_ids,
                "gold_pages_in_top_t": gold_pages_in_top_t,
                "selected_page_universe_size": len(selected_page_trajectory_ids),
                **_broad_entity_fields_from_retrieval_metadata(retrieval_metadata),
                **page_granularity_fields,
                **wiki_fragmentation_fields,
                **page_family_routing_fields,
                **wiki_coverage_fields,
                "index_fallback_used": bool(retrieval_metadata.get("index_fallback_used")),
                "index_fallback_reason": retrieval_metadata.get("index_fallback_reason"),
                "selected_page_universe_size_before_index_fallback": retrieval_metadata.get(
                    "selected_page_universe_size_before_index_fallback"
                ),
                "selected_page_universe_size_after_index_fallback": retrieval_metadata.get(
                    "selected_page_universe_size_after_index_fallback"
                ),
                "index_fallback_candidate_count": retrieval_metadata.get("index_fallback_candidate_count"),
                "index_fallback_added_trajectory_ids": list(
                    retrieval_metadata.get("index_fallback_added_trajectory_ids") or []
                ),
                "page_routing_recall": (
                    _safe_rate(len(gold_pages_in_top_t), len(gold_page_ids))
                    if gold_page_ids
                    else None
                ),
                **routing_text_marker_fields,
                **_family_ranking_fields_from_metadata(
                    retrieval_metadata=retrieval_metadata,
                    gold_trajectory_ids=gold_trajectory_ids,
                    top_k_trajectory_ids=top_k_trajectory_ids,
                    selection_pool_trajectory_ids=list(top_k_coverage["selection_pool_trajectory_ids"]),
                ),
                "raw_expanded_snapshot_ids": list(retrieval_event.get("raw_expanded_snapshot_ids", [])),
                "raw_expanded_refs": list(retrieval_event.get("raw_expanded_refs", [])),
                "pre_compaction_expansion_could_cover_all_gold_refs": bool(
                    gold_refs and gold_refs <= set(retrieval_event.get("raw_expanded_refs", []))
                ),
                "snapshot_compaction_miss": bool(
                    gold_refs
                    and gold_refs <= set(retrieval_event.get("raw_expanded_refs", []))
                    and not gold_refs <= set(retrieval_event.get("expanded_refs", []))
                ),
                "source_compaction_miss": bool(
                    gold_refs
                    and gold_refs <= set(retrieval_event.get("expanded_refs", []))
                    and not gold_refs <= set(retrieval_event.get("grounded_source_refs", []))
                ),
                "gold_refs_lost_by_snapshot_compaction": sorted(
                    (set(retrieval_event.get("raw_expanded_refs", [])) & gold_refs)
                    - set(retrieval_event.get("expanded_refs", []))
                ),
                "gold_refs_lost_by_source_compaction": sorted(
                    (set(retrieval_event.get("expanded_refs", [])) & gold_refs)
                    - set(retrieval_event.get("grounded_source_refs", []))
                ),
                "gold_item_present_in_claims": gold_item_present_in_claims,
                "gold_item_present_in_wiki": gold_item_present_in_wiki,
                "gold_item_missing_but_umbrella_present": gold_item_missing_but_umbrella_present,
                "index_only_gold_trajectory_count": index_only_gold_trajectory_count,
                "answer_postcheck_used": bool(answer_metadata.get("answer_postcheck_used")),
                "answer_postcheck_issue": answer_metadata.get("answer_postcheck_issue"),
                **temporal_grounding_fields,
                **answer_synthesis_fields,
                **llm_usage_fields,
                **judge_leniency_fields,
                **reflection_fields,
                "top_k_redundant_entity_cluster_ratio": top_k_redundant_entity_cluster_ratio,
                "selected_page_cluster_count": selected_page_cluster_count,
                **claim_preservation_fields,
                **source_surface_fields,
                **preservation_and_stage,
                **page_cutoff_fields,
                **trajectory_cutoff_fields,
                **offline_parameter_fields,
                **direct_trajectory_ablation_fields,
                **trajectory_length_fields,
                **trajectory_drift_fields,
                **semantic_f1_fields,
                **evaluation_filter_fields,
                **judge_fields,
            }
        )

    failed_total = len(failed_rows)
    total_queries = len(sample_rows)
    examples_by_reason: dict[str, list[dict[str, Any]]] = {}
    for reason in FAILURE_REASON_ORDER:
        examples_by_reason[reason] = [
            row for row in failed_rows if row["reason"] == reason
        ][:top_examples_per_bucket]

    split_rows = [row for row in failed_rows if int(row.get("gold_trajectory_count", 0) or 0) > 1]
    split_total = len(split_rows)
    split_recall_values = [
        float(row["gold_trajectory_recall_at_k"])
        for row in split_rows
        if row.get("gold_trajectory_recall_at_k") is not None
    ]
    split_gold_ref_coverage_values = [
        float(row["top_k_gold_ref_coverage_rate"])
        for row in split_rows
        if row.get("top_k_gold_ref_coverage_rate") is not None
    ]
    all_gold_in_top_k_count = sum(
        1
        for row in split_rows
        if int(row.get("gold_trajectory_count_in_top_k", 0) or 0) == int(row.get("gold_trajectory_count", 0) or 0)
        and int(row.get("gold_trajectory_count", 0) or 0) > 0
    )
    top_k_cover_all_count = sum(1 for row in split_rows if row.get("top_k_can_cover_all_gold_refs") is True)
    zero_gold_in_top_k_count = sum(1 for row in split_rows if int(row.get("gold_trajectory_count_in_top_k", 0) or 0) == 0)
    one_gold_in_top_k_count = sum(1 for row in split_rows if int(row.get("gold_trajectory_count_in_top_k", 0) or 0) == 1)
    two_or_more_gold_in_top_k_count = sum(
        1 for row in split_rows if int(row.get("gold_trajectory_count_in_top_k", 0) or 0) >= 2
    )
    failed_gold_trajectory_rows = [
        row for row in failed_rows if int(row.get("gold_trajectory_count", 0) or 0) > 0
    ]
    selection_pool_available_failed_gold_rows = [
        row for row in failed_gold_trajectory_rows if row.get("trajectory_selection_pool_available") is True
    ]
    selection_pool_available_split_rows = [
        row for row in split_rows if row.get("trajectory_selection_pool_available") is True
    ]
    selection_pool_recall_values = [
        float(row["gold_trajectory_recall_in_selection_pool"])
        for row in selection_pool_available_failed_gold_rows
        if row.get("gold_trajectory_recall_in_selection_pool") is not None
    ]
    split_selection_pool_recall_values = [
        float(row["gold_trajectory_recall_in_selection_pool"])
        for row in selection_pool_available_split_rows
        if row.get("gold_trajectory_recall_in_selection_pool") is not None
    ]
    selection_pool_can_cover_all_gold_count = sum(
        1 for row in selection_pool_available_failed_gold_rows if row.get("selection_pool_can_cover_all_gold_trajectories") is True
    )
    split_selection_pool_can_cover_all_gold_count = sum(
        1 for row in selection_pool_available_split_rows if row.get("selection_pool_can_cover_all_gold_trajectories") is True
    )
    gold_trajectory_lost_before_selection_pool_count = sum(
        1
        for row in selection_pool_available_failed_gold_rows
        if row.get("selection_pool_can_cover_all_gold_trajectories") is False
    )
    gold_trajectory_lost_during_final_top_k_count = sum(
        1
        for row in selection_pool_available_failed_gold_rows
        if row.get("selection_pool_can_cover_all_gold_trajectories") is True
        and int(row.get("gold_trajectory_count_in_top_k", 0) or 0)
        < int(row.get("gold_trajectory_count", 0) or 0)
    )
    coverage_query_shape_trigger_count = sum(
        1 for row in failed_rows if _query_shape_requires_coverage(dict(row.get("query_shape") or {}))
    )
    coverage_query_shape_trigger_count_all = sum(
        1 for row in query_outcomes if _query_shape_requires_coverage(dict(row.get("query_shape") or {}))
    )
    cutoff_diagnostics = _build_cutoff_diagnostics(query_outcomes, failed_rows)
    offline_parameter_diagnostics = _build_offline_parameter_diagnostics(query_outcomes, failed_rows)
    trajectory_length_diagnostics = _build_trajectory_length_diagnostics(
        configured_m=configured_m,
        trajectory_lengths=trajectory_lengths,
        query_outcomes=query_outcomes,
        failed_rows=failed_rows,
    )
    trajectory_drift_diagnostics = build_trajectory_drift_diagnostics(
        trajectory_drift_rows=trajectory_drift_rows,
        query_rows=query_outcomes,
        failed_rows=failed_rows,
    )
    family_available_rows = [
        row for row in query_outcomes if list(row.get("trajectory_family_match_scores") or [])
    ]
    family_gold_rows = [
        row for row in family_available_rows if int(row.get("gold_trajectory_count") or 0) > 0
    ]
    source_event_rows = [
        row for row in query_outcomes if list(row.get("trajectory_source_event_match_scores") or [])
    ]
    source_event_gold_rows = [
        row for row in source_event_rows if int(row.get("gold_trajectory_count") or 0) > 0
    ]
    source_event_diagnostics = {
        "source_event_match_used_count": len(source_event_rows),
        "source_event_selected_match_count": sum(
            int(row.get("trajectory_selected_source_event_match_count") or 0)
            for row in query_outcomes
        ),
        "source_event_match_boosted_gold_count": sum(
            1
            for row in source_event_gold_rows
            if int(row.get("gold_source_event_matched_in_top_k_count") or 0) > 0
        ),
        "source_event_match_missing_gold_count": sum(
            1
            for row in source_event_gold_rows
            if int(row.get("gold_source_event_matched_in_selection_pool_count") or 0) == 0
        ),
        "source_event_metadata_trajectory_count": sum(
            1
            for row in source_event_rows
            for item in list(row.get("trajectory_source_event_match_scores") or [])
            if isinstance(item, dict) and float(item.get("score") or 0.0) > 0.0
        ),
    }
    conv26_like_trajectory_selection_diagnostics = {
        "selected_page_universe_size_mean_over_all": _safe_mean(
            [int(row.get("selected_page_universe_size") or 0) for row in query_outcomes]
        ),
        "selected_page_universe_size_median_over_all": _safe_median(
            [int(row.get("selected_page_universe_size") or 0) for row in query_outcomes]
        ),
        "selected_page_universe_size_mean_over_failed": _safe_mean(
            [int(row.get("selected_page_universe_size") or 0) for row in failed_rows]
        ),
        "broad_entity_cap_used_count_over_all": sum(
            1 for row in query_outcomes if bool(row.get("broad_entity_candidate_cap_used"))
        ),
        "broad_entity_cap_used_count_over_failed": sum(
            1 for row in failed_rows if bool(row.get("broad_entity_candidate_cap_used"))
        ),
        "broad_entity_profile_selected_count_over_all": sum(
            1 for row in query_outcomes if list(row.get("broad_entity_profile_page_ids") or [])
        ),
        "broad_entity_profile_suppressed_count_over_all": sum(
            1 for row in query_outcomes if bool(row.get("broad_entity_profile_suppressed"))
        ),
        "fine_grained_entity_page_selected_count_over_all": sum(
            1 for row in query_outcomes if list(row.get("fine_grained_entity_page_ids") or [])
        ),
        "selected_page_universe_size_before_broad_cap_mean": _safe_mean(
            [
                int(row.get("selected_page_universe_size_before_broad_cap") or 0)
                for row in query_outcomes
                if row.get("selected_page_universe_size_before_broad_cap") is not None
            ]
        ),
        "selected_page_universe_size_after_broad_cap_mean": _safe_mean(
            [
                int(row.get("selected_page_universe_size_after_broad_cap") or 0)
                for row in query_outcomes
                if row.get("selected_page_universe_size_after_broad_cap") is not None
            ]
        ),
        "family_match_available_query_count": len(family_available_rows),
        "family_match_gold_query_count": len(family_gold_rows),
        "family_match_recall_in_selection_pool": _safe_rate(
            sum(int(row.get("gold_family_matched_in_selection_pool_count") or 0) for row in family_gold_rows),
            sum(int(row.get("gold_trajectory_count") or 0) for row in family_gold_rows),
        ),
        "family_match_recall_in_final_top_k": _safe_rate(
            sum(int(row.get("gold_family_matched_in_top_k_count") or 0) for row in family_gold_rows),
            sum(int(row.get("gold_trajectory_count") or 0) for row in family_gold_rows),
        ),
        "answer_synthesis_invalid_family_ref_count": answer_synthesis_diagnostics.get("invalid_family_ref_count"),
        "answer_synthesis_question_type_mismatch_count": answer_synthesis_diagnostics.get(
            "question_type_mismatch_count"
        ),
    }
    sample_fragmentation_rows: dict[str, dict[str, Any]] = {}
    for row in query_outcomes:
        if row.get("wiki_non_index_page_count") is None:
            continue
        sample_fragmentation_rows.setdefault(str(row.get("sample_id")), row)
    wiki_fragmentation_sources = {
        str(row.get("wiki_fragmentation_metadata_source") or "")
        for row in sample_fragmentation_rows.values()
        if str(row.get("wiki_fragmentation_metadata_source") or "").strip()
    }
    selected_page_slot_count = 0
    selected_singleton_slot_count = 0
    selected_medium_slot_count = 0
    page_granularity_available_rows = [
        row for row in query_outcomes if bool(row.get("page_granularity_metadata_available"))
    ]
    for row in query_outcomes:
        histogram = dict(row.get("selected_page_trajectory_count_histogram") or {})
        for count_text, frequency in histogram.items():
            try:
                count = int(count_text)
                value = int(frequency)
            except (TypeError, ValueError):
                continue
            selected_page_slot_count += value
            if count == 1:
                selected_singleton_slot_count += value
            if 3 <= count <= 6:
                selected_medium_slot_count += value
    page_metadata_rows = list(page_metadata_by_id.values())
    routing_marker_page_count = sum(
        1 for metadata in page_metadata_rows if int(metadata.get("routing_text_internal_marker_count") or 0) > 0
    )
    routing_text_diagnostics = {
        "page_count": len(page_metadata_rows),
        "routing_text_cleaned_page_count": sum(
            1 for metadata in page_metadata_rows if metadata.get("routing_text_cleaned") is True
        ),
        "routing_text_internal_marker_page_count": routing_marker_page_count,
        "routing_text_internal_marker_rate": _safe_rate(routing_marker_page_count, len(page_metadata_rows)),
    }
    metadata_term_diagnostics = _metadata_term_diagnostics(trajectory_metadata)

    def _sum_optional_row_int(rows: Iterable[dict[str, Any]], key: str) -> int | None:
        values: list[int] = []
        for row in rows:
            if row.get(key) is None:
                continue
            try:
                values.append(int(row.get(key)))
            except (TypeError, ValueError):
                continue
        return sum(values) if values else None

    def _max_optional_row_int(rows: Iterable[dict[str, Any]], key: str) -> int | None:
        values: list[int] = []
        for row in rows:
            if row.get(key) is None:
                continue
            try:
                values.append(int(row.get(key)))
            except (TypeError, ValueError):
                continue
        return max(values) if values else None

    wiki_fragmentation_diagnostics = {
        "diagnostic_mode": (
            "retrieval_metadata"
            if "retrieval_metadata" in wiki_fragmentation_sources
            else "sqlite_wiki_pages_fallback"
            if "sqlite_wiki_pages_fallback" in wiki_fragmentation_sources
            else "missing_retrieval_metadata"
        ),
        "sample_count": len(sample_fragmentation_rows),
        "mean_non_index_page_count": _safe_mean(
            [
                int(row.get("wiki_non_index_page_count") or 0)
                for row in sample_fragmentation_rows.values()
            ]
        ),
        "mean_singleton_non_index_page_rate": _safe_mean(
            [
                float(row.get("wiki_singleton_non_index_page_rate"))
                for row in sample_fragmentation_rows.values()
                if row.get("wiki_singleton_non_index_page_rate") is not None
            ]
        ),
        "mean_trajectories_per_non_index_page": _safe_mean(
            [
                float(row.get("wiki_mean_trajectories_per_non_index_page"))
                for row in sample_fragmentation_rows.values()
                if row.get("wiki_mean_trajectories_per_non_index_page") is not None
            ]
        ),
        "post_plan_rescue_singleton_count": sum(
            int(row.get("wiki_post_plan_rescue_singleton_count") or 0)
            for row in sample_fragmentation_rows.values()
        ),
        "entity_facet_singleton_count": sum(
            int(row.get("wiki_entity_facet_singleton_count") or 0)
            for row in sample_fragmentation_rows.values()
        ),
        "allowed_specific_singleton_count": _sum_optional_row_int(
            sample_fragmentation_rows.values(), "wiki_allowed_specific_singleton_count"
        ),
        "low_quality_singleton_count": _sum_optional_row_int(
            sample_fragmentation_rows.values(), "wiki_low_quality_singleton_count"
        ),
        "low_quality_singleton_merged_count": _sum_optional_row_int(
            sample_fragmentation_rows.values(), "wiki_low_quality_singleton_merged_count"
        ),
        "overwide_non_index_page_count": _sum_optional_row_int(
            sample_fragmentation_rows.values(), "wiki_overwide_non_index_page_count"
        ),
        "overwide_page_split_count": _sum_optional_row_int(
            sample_fragmentation_rows.values(), "wiki_overwide_page_split_count"
        ),
        "max_non_index_trajectory_count": _max_optional_row_int(
            sample_fragmentation_rows.values(), "wiki_max_non_index_trajectory_count"
        ),
        "selected_page_slot_count": selected_page_slot_count,
        "selected_singleton_page_slot_count": selected_singleton_slot_count,
        "selected_singleton_page_slot_rate": (
            _safe_rate(selected_singleton_slot_count, selected_page_slot_count)
            if selected_page_slot_count
            else None
        ),
        "selected_medium_page_slot_count": selected_medium_slot_count,
        "selected_medium_page_slot_rate": (
            _safe_rate(selected_medium_slot_count, selected_page_slot_count)
            if selected_page_slot_count
            else None
        ),
        "singleton_page_penalty_applied_count": sum(
            int(row.get("singleton_page_penalty_applied") or 0) for row in query_outcomes
            if row.get("singleton_page_penalty_applied") is not None
        ),
        "low_quality_singleton_penalty_applied_count": _sum_optional_row_int(
            query_outcomes, "low_quality_singleton_penalty_applied"
        ),
        "medium_page_bonus_applied_count": sum(
            int(row.get("medium_page_bonus_applied") or 0) for row in query_outcomes
            if row.get("medium_page_bonus_applied") is not None
        ),
    }
    direct_trajectory_ablation_diagnostics = _aggregate_direct_trajectory_ablation(query_outcomes)
    direct_vs_routed_retrieval_diagnostics = _build_direct_vs_routed_retrieval_diagnostics(
        query_outcomes,
        failed_rows,
    )

    report = {
        "run_meta": {
            **run_meta,
            "resolved_run_dir": str(run_dir),
            "details_path": str(details_path),
            "database_path": str(database_path),
        },
        "totals": {
            "total_queries": total_queries,
            "failed_queries": failed_total,
            "failure_rate": _safe_rate(failed_total, total_queries),
        },
        "reason_counts": {reason: int(reason_counts[reason]) for reason in FAILURE_REASON_ORDER},
        "reason_rates_over_failed": {
            reason: _safe_rate(reason_counts[reason], failed_total)
            for reason in FAILURE_REASON_ORDER
        },
        "retrieval_coverage": {
            "memory_hit_rate_over_failed": _safe_rate(memory_hit_count, failed_total),
            "coarse_hit_rate_over_failed": _safe_rate(coarse_hit_count, failed_total),
            "fine_hit_rate_over_failed": _safe_rate(fine_hit_count, failed_total),
            "expanded_hit_rate_over_failed": _safe_rate(expanded_hit_count, failed_total),
            "grounded_hit_rate_over_failed": _safe_rate(grounded_hit_count, failed_total),
            "expansion_recovered_rate_over_failed": _safe_rate(expansion_recovered_count, failed_total),
        },
        "rank_diagnostics": {
            "coarse_rank_observed_count": len(coarse_ranks),
            "coarse_rank_mean": _safe_mean(coarse_ranks),
            "coarse_rank_median": _safe_median(coarse_ranks),
            "coarse_top_k_hit_rate_over_failed": _safe_rate(coarse_hit_count, failed_total),
            "fine_rank_observed_count": len(fine_ranks),
            "fine_rank_mean": _safe_mean(fine_ranks),
            "fine_rank_median": _safe_median(fine_ranks),
            "fine_top_r_hit_rate_over_failed": _safe_rate(fine_hit_count, failed_total),
        },
        "facet_diagnostics": {
            "facet_present_for_gold_ref_rate": _safe_rate(flag_counts["facet_present_for_gold_ref"], failed_total),
            "exact_facet_missing_rate": _safe_rate(flag_counts["exact_facet_missing"], failed_total),
            "gold_ref_present_but_claim_generalized_rate": _safe_rate(
                flag_counts["gold_ref_present_but_claim_generalized"], failed_total
            ),
            "gold_ref_present_but_no_supported_facet_rate": _safe_rate(
                flag_counts["gold_ref_present_but_no_supported_facet"], failed_total
            ),
        },
        "fragmentation_diagnostics": {
            "split_across_trajectories_rate": _safe_rate(
                flag_counts["gold_refs_split_across_trajectories"], failed_total
            ),
            "entity_linked_could_help_rate": _safe_rate(
                flag_counts["entity_linked_could_help"], failed_total
            ),
            "entity_linked_added_relevant_snapshot_rate": _safe_rate(
                flag_counts["entity_linked_added_relevant_snapshot"], failed_total
            ),
        },
        "page_routing_diagnostics": {
            "all_gold_pages_in_top_t_rate_over_failed": _safe_rate(all_gold_pages_in_top_t_count, failed_total),
            "mean_page_routing_recall_over_failed": _safe_mean(page_recall_values),
            "median_page_routing_recall_over_failed": _safe_median(page_recall_values),
            "all_gold_trajectories_in_selected_pages_rate_over_failed": _safe_rate(
                all_gold_trajectories_in_selected_pages_count,
                failed_total,
            ),
            "all_gold_trajectories_in_final_top_k_after_page_routing_rate_over_failed": _safe_rate(
                all_gold_trajectories_in_final_top_k_after_page_routing_count,
                failed_total,
            ),
        },
        "wiki_fragmentation_diagnostics": wiki_fragmentation_diagnostics,
        "routing_text_diagnostics": routing_text_diagnostics,
        "source_event_reranking_diagnostics": source_event_diagnostics,
        "wiki_coverage_diagnostics": {
            "mean_non_index_trajectory_coverage_rate_over_all": _safe_mean(
                [
                    float(row["non_index_trajectory_coverage_rate"])
                    for row in query_outcomes
                    if row.get("non_index_trajectory_coverage_rate") is not None
                ]
            ),
            "mean_non_index_trajectory_coverage_rate_over_failed": _safe_mean(
                [
                    float(row["non_index_trajectory_coverage_rate"])
                    for row in failed_rows
                    if row.get("non_index_trajectory_coverage_rate") is not None
                ]
            ),
            "queries_with_index_only_trajectories_over_failed": sum(
                1 for row in failed_rows if int(row.get("index_only_trajectory_count") or 0) > 0
            ),
            "index_fallback_used_count_over_failed": sum(
                1 for row in failed_rows if bool(row.get("index_fallback_used"))
            ),
            "index_fallback_used_count_over_all": sum(
                1 for row in query_outcomes if bool(row.get("index_fallback_used"))
            ),
        },
        "memory_preservation_diagnostics": {
            "gold_answer_fully_present_in_claims_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_answer_fully_present_in_claims"], failed_total
            ),
            "gold_answer_fully_present_in_trajectory_summaries_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_answer_fully_present_in_trajectory_summaries"], failed_total
            ),
            "gold_answer_fully_present_in_wiki_pages_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_answer_fully_present_in_wiki_pages"], failed_total
            ),
            "gold_item_present_in_claims_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_item_present_in_claims"], failed_total
            ),
            "gold_item_present_in_wiki_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_item_present_in_wiki"], failed_total
            ),
            "gold_item_present_in_historical_card_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_item_present_in_historical_card"], failed_total
            ),
            "gold_item_present_in_latest_summary_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_item_present_in_latest_summary"], failed_total
            ),
            "gold_item_present_in_wiki_after_historical_card_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_item_present_in_wiki_after_historical_card"], failed_total
            ),
            "trajectory_drift_suspected_rate_over_failed": _safe_rate(
                memory_preservation_counts["trajectory_drift_suspected"], failed_total
            ),
            "index_only_gold_trajectory_rate_over_failed": _safe_rate(
                memory_preservation_counts["index_only_gold_trajectory_count"], failed_total
            ),
            "gold_item_missing_but_umbrella_present_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_item_missing_but_umbrella_present"], failed_total
            ),
            "trajectory_storage_miss_rate_over_failed": _safe_rate(
                memory_preservation_counts["trajectory_storage_miss"], failed_total
            ),
            "summary_compression_miss_rate_over_failed": _safe_rate(
                memory_preservation_counts["summary_compression_miss"], failed_total
            ),
            "wiki_compilation_compression_miss_rate_over_failed": _safe_rate(
                memory_preservation_counts["wiki_compilation_compression_miss"], failed_total
            ),
            "gold_surface_present_in_raw_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_surface_present_in_raw"], failed_total
            ),
            "gold_surface_preserved_in_claims_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_surface_preserved_in_claims"], failed_total
            ),
            "gold_surface_preserved_in_exact_terms_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_surface_preserved_in_exact_terms"], failed_total
            ),
            "gold_surface_preserved_in_historical_card_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_surface_preserved_in_historical_card"], failed_total
            ),
            "gold_surface_preserved_in_wiki_rate_over_failed": _safe_rate(
                memory_preservation_counts["gold_surface_preserved_in_wiki"], failed_total
            ),
        },
        "memory_force_recall_diagnostics": {
            "forced_memory_seed_count": int(force_recall_counts.get("forced_memory_seed_count", 0)),
            "low_salience_memory_count": int(force_recall_counts.get("low_salience_memory_count", 0)),
            "llm_no_memory_forced_count": int(force_recall_counts.get("llm_no_memory_forced_count", 0)),
            "zero_claim_episodic_candidate_count": int(
                force_recall_counts.get("zero_claim_episodic_candidate_count", 0)
            ),
            "zero_claim_episodic_persisted_count": int(
                force_recall_counts.get("zero_claim_episodic_persisted_count", 0)
            ),
            "zero_claim_low_salience_skipped_count": int(
                force_recall_counts.get("zero_claim_low_salience_skipped_count", 0)
            ),
            "failed_queries_with_gold_refs_in_forced_memory_count": force_recall_failed_query_count,
            "failed_queries_with_gold_refs_in_forced_memory_rate_over_failed": _safe_rate(
                force_recall_failed_query_count, failed_total
            ),
            "failed_queries_with_gold_refs_in_low_salience_memory_count": low_salience_failed_query_count,
            "failed_queries_with_gold_refs_in_low_salience_memory_rate_over_failed": _safe_rate(
                low_salience_failed_query_count, failed_total
            ),
            "gold_refs_in_forced_memory_count": gold_refs_in_forced_memory_count,
            "gold_refs_in_low_salience_memory_count": gold_refs_in_low_salience_memory_count,
        },
        "stage_diagnostics": {
            "page_routing_failed_rate_over_failed": _safe_rate(
                stage_counts["page_routing_failed"], failed_total
            ),
            "selected_pages_failed_to_cover_all_gold_trajectories_rate_over_failed": _safe_rate(
                stage_counts["selected_pages_failed_to_cover_all_gold_trajectories"], failed_total
            ),
            "trajectory_retrieval_failed_after_page_routing_rate_over_failed": _safe_rate(
                stage_counts["trajectory_retrieval_failed_after_page_routing"], failed_total
            ),
            "all_gold_refs_grounded_rate_over_failed": _safe_rate(
                stage_counts["all_gold_refs_grounded"], failed_total
            ),
            "strict_retrieval_success_but_answer_wrong_rate_over_failed": _safe_rate(
                stage_counts["strict_retrieval_success_but_answer_wrong"], failed_total
            ),
        },
        "compaction_diagnostics": {
            "pre_compaction_expansion_cover_rate_over_failed": _safe_rate(
                compaction_counts["pre_compaction_expansion_could_cover_all_gold_refs"], failed_total
            ),
            "snapshot_compaction_miss_rate_over_failed": _safe_rate(
                compaction_counts["snapshot_compaction_miss"], failed_total
            ),
            "source_compaction_miss_rate_over_failed": _safe_rate(
                compaction_counts["source_compaction_miss"], failed_total
            ),
        },
        "multi_hop_top_k_diagnostics": {
            "split_case_count": split_total,
            "split_case_rate_over_failed": _safe_rate(split_total, failed_total),
            "mean_gold_trajectory_recall_at_k_over_split_cases": _safe_mean(split_recall_values),
            "median_gold_trajectory_recall_at_k_over_split_cases": _safe_median(split_recall_values),
            "selection_pool_available_failed_gold_query_count": len(selection_pool_available_failed_gold_rows),
            "selection_pool_available_split_case_count": len(selection_pool_available_split_rows),
            "mean_gold_trajectory_recall_in_selection_pool_over_failed": _safe_mean(selection_pool_recall_values),
            "mean_gold_trajectory_recall_in_selection_pool_over_split_cases": _safe_mean(
                split_selection_pool_recall_values
            ),
            "gold_trajectory_in_selection_pool_rate": _safe_mean(selection_pool_recall_values),
            "gold_trajectory_lost_before_selection_pool_count": (
                gold_trajectory_lost_before_selection_pool_count
                if selection_pool_available_failed_gold_rows
                else None
            ),
            "gold_trajectory_lost_during_final_top_k_count": (
                gold_trajectory_lost_during_final_top_k_count
                if selection_pool_available_failed_gold_rows
                else None
            ),
            "coverage_query_shape_trigger_rate": _safe_rate(coverage_query_shape_trigger_count, failed_total),
            "coverage_query_shape_trigger_rate_over_all": _safe_rate(
                coverage_query_shape_trigger_count_all,
                total_queries,
            ),
            "selection_pool_can_cover_all_gold_trajectories_rate": _safe_rate(
                selection_pool_can_cover_all_gold_count,
                len(selection_pool_available_failed_gold_rows),
            ) if selection_pool_available_failed_gold_rows else None,
            "selection_pool_can_cover_all_gold_trajectories_rate_over_split_cases": _safe_rate(
                split_selection_pool_can_cover_all_gold_count,
                len(selection_pool_available_split_rows),
            ) if selection_pool_available_split_rows else None,
            "all_gold_trajectories_in_top_k_rate_over_split_cases": _safe_rate(all_gold_in_top_k_count, split_total),
            "top_k_can_cover_all_gold_refs_rate_over_split_cases": _safe_rate(top_k_cover_all_count, split_total),
            "mean_top_k_gold_ref_coverage_rate_over_split_cases": _safe_mean(split_gold_ref_coverage_values),
            "mean_top_k_redundant_entity_cluster_ratio_over_split_cases": _safe_mean(
                split_case_redundant_trajectory_ratios
            ),
            "zero_gold_trajectory_in_top_k_rate_over_split_cases": _safe_rate(zero_gold_in_top_k_count, split_total),
            "one_gold_trajectory_in_top_k_rate_over_split_cases": _safe_rate(one_gold_in_top_k_count, split_total),
            "two_or_more_gold_trajectories_in_top_k_rate_over_split_cases": _safe_rate(
                two_or_more_gold_in_top_k_count, split_total
            ),
        },
        "conv26_like_trajectory_selection_diagnostics": conv26_like_trajectory_selection_diagnostics,
        "direct_trajectory_ablation_diagnostics": direct_trajectory_ablation_diagnostics,
        "direct_vs_routed_retrieval_diagnostics": direct_vs_routed_retrieval_diagnostics,
        "split_case_redundancy_diagnostics": {
            "mean_top_k_redundant_entity_cluster_ratio_over_split_cases": _safe_mean(
                split_case_redundant_trajectory_ratios
            ),
            "mean_selected_page_cluster_count_over_failed": _safe_mean(failed_selected_page_cluster_counts),
        },
        "cutoff_diagnostics": cutoff_diagnostics,
        "offline_parameter_diagnostics": offline_parameter_diagnostics,
        "trajectory_length_diagnostics": trajectory_length_diagnostics,
        "trajectory_drift_diagnostics": trajectory_drift_diagnostics,
        "text_only_filter_diagnostics": text_only_filter_diagnostics,
        "cuda_preflight_diagnostics": cuda_preflight_diagnostics,
        "judge_diagnostics": judge_diagnostics,
        "answer_synthesis_diagnostics": answer_synthesis_diagnostics,
        "llm_call_diagnostics": llm_call_diagnostics,
        "fallback_repair_diagnostics": {
            **fallback_repair_diagnostics,
            "event_file": str(run_dir / "fallback_repair_events.jsonl")
            if (run_dir / "fallback_repair_events.jsonl").exists()
            else None,
            "loaded_event_count": len(fallback_repair_events),
        },
        "retrieval_reflection_diagnostics": retrieval_reflection_diagnostics,
        "metadata_term_diagnostics": metadata_term_diagnostics,
        "diagnostic_flag_counts": {flag: int(flag_counts[flag]) for flag in DIAGNOSTIC_FLAG_ORDER},
        "diagnostic_flag_rates_over_failed": {
            flag: _safe_rate(flag_counts[flag], failed_total) for flag in DIAGNOSTIC_FLAG_ORDER
        },
        "examples_by_reason": examples_by_reason,
        "failed_rows": failed_rows,
        "query_outcomes": query_outcomes,
    }
    try:
        analysis_dir = run_dir / "analysis"
        _write_json_artifact(
            analysis_dir / "direct_retrieval_ablation.json",
            {
                "schema_version": "direct_retrieval_ablation_v1",
                "diagnostics": direct_vs_routed_retrieval_diagnostics,
                "legacy_direct_trajectory_ablation_diagnostics": direct_trajectory_ablation_diagnostics,
            },
        )
        _write_jsonl_artifact(
            analysis_dir / "direct_retrieval_rows.jsonl",
            (
                {
                    "schema_version": "direct_retrieval_ablation_row_v1",
                    "sample_id": row.get("sample_id"),
                    "query_task_id": row.get("query_task_id"),
                    "question": row.get("question"),
                    "query_shape": row.get("query_shape"),
                    "gold_trajectory_ids": row.get("gold_trajectory_ids"),
                    "routed_top_k_trajectory_ids": row.get("top_k_trajectory_ids"),
                    "selected_page_trajectory_ids": row.get("selected_page_trajectory_ids"),
                    "direct_trajectory_top_k_ids": row.get("direct_trajectory_top_k_ids"),
                    "direct_trajectory_ranked_rows_compact_top_n": row.get(
                        "direct_trajectory_ranked_rows_compact_top_n"
                    ),
                    "direct_trajectory_cutoff_diagnostics": row.get("direct_trajectory_cutoff_diagnostics"),
                    "routed_vs_direct_top_k_overlap_rate": row.get("routed_vs_direct_top_k_overlap_rate"),
                    "direct_top_k_not_in_selected_page_universe_count": row.get(
                        "direct_top_k_not_in_selected_page_universe_count"
                    ),
                    "direct_top_k_estimated_context_token_count": row.get(
                        "direct_top_k_estimated_context_token_count"
                    ),
                    "direct_trajectory_bottleneck": row.get("direct_trajectory_bottleneck"),
                }
                for row in query_outcomes
            ),
        )
        _write_json_artifact(
            analysis_dir / "trajectory_drift_diagnostics.json",
            trajectory_drift_diagnostics,
        )
        _write_jsonl_artifact(
            analysis_dir / "trajectory_drift_rows.jsonl",
            trajectory_drift_rows,
        )
        _write_jsonl_artifact(
            analysis_dir / "trajectory_drift_query_rows.jsonl",
            (compact_query_drift_row(row) for row in query_outcomes),
        )
        report["direct_retrieval_ablation_artifacts"] = {
            "summary": str(analysis_dir / "direct_retrieval_ablation.json"),
            "rows": str(analysis_dir / "direct_retrieval_rows.jsonl"),
        }
        report["trajectory_drift_artifacts"] = {
            "summary": str(analysis_dir / "trajectory_drift_diagnostics.json"),
            "rows": str(analysis_dir / "trajectory_drift_rows.jsonl"),
            "query_rows": str(analysis_dir / "trajectory_drift_query_rows.jsonl"),
        }
    except OSError as exc:
        report["direct_retrieval_ablation_artifacts"] = {
            "write_error": f"{type(exc).__name__}: {collapse_whitespace(str(exc))[:300]}",
        }
        report["trajectory_drift_artifacts"] = {
            "write_error": f"{type(exc).__name__}: {collapse_whitespace(str(exc))[:300]}",
        }
    return report


def diff_locomo_failure_reports(
    before_run_path: Path | str,
    after_run_path: Path | str,
    *,
    top_examples_per_bucket: int = 5,
) -> dict[str, Any]:
    before_report = analyze_locomo_run_failures(
        before_run_path, top_examples_per_bucket=top_examples_per_bucket
    )
    after_report = analyze_locomo_run_failures(
        after_run_path, top_examples_per_bucket=top_examples_per_bucket
    )

    def _index_outcomes(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (str(row["sample_id"]), str(row["query_task_id"])): row
            for row in list(report.get("query_outcomes", []))
        }

    before_outcomes = _index_outcomes(before_report)
    after_outcomes = _index_outcomes(after_report)
    all_keys = sorted(set(before_outcomes) | set(after_outcomes))

    transition_counts: Counter[tuple[str, str]] = Counter()
    coverability_transition_counts: Counter[tuple[str, str]] = Counter()
    recall_transition_counts: Counter[tuple[str, str]] = Counter()
    page_coverability_transition_counts: Counter[tuple[str, str]] = Counter()
    strict_answer_error_transition_counts: Counter[tuple[str, str]] = Counter()
    improved_queries: list[dict[str, Any]] = []
    worsened_queries: list[dict[str, Any]] = []
    changed_failures: list[dict[str, Any]] = []
    improved_multi_hop_queries: list[dict[str, Any]] = []
    worsened_multi_hop_queries: list[dict[str, Any]] = []

    def _coverability_state(row: dict[str, Any] | None) -> str:
        if row is None:
            return "missing"
        value = row.get("top_k_can_cover_all_gold_refs")
        if value is True:
            return "coverable_in_top_k"
        if value is False:
            return "not_coverable_in_top_k"
        return "unknown"

    def _recall_state(row: dict[str, Any] | None) -> str:
        if row is None:
            return "missing"
        recall = row.get("gold_trajectory_recall_at_k")
        if recall is None:
            return "unknown"
        recall_value = float(recall)
        if recall_value <= 0.0:
            return "zero_gold_traj_recall"
        if recall_value >= 1.0:
            return "full_gold_traj_recall"
        return "partial_gold_traj_recall"

    def _page_coverability_state(row: dict[str, Any] | None) -> str:
        if row is None:
            return "missing"
        gold_page_ids = list(row.get("gold_page_ids") or [])
        if not gold_page_ids:
            return "no_gold_pages"
        top_hit_ids = list(row.get("gold_pages_in_top_t") or [])
        if len(top_hit_ids) == 0:
            return "no_gold_pages_in_top_t"
        if len(top_hit_ids) == len(gold_page_ids):
            return "all_gold_pages_in_top_t"
        return "partial_gold_pages_in_top_t"

    for key in all_keys:
        before_row = before_outcomes.get(key)
        after_row = after_outcomes.get(key)
        before_reason = before_row["reason"] if before_row is not None else "missing"
        after_reason = after_row["reason"] if after_row is not None else "missing"
        transition_counts[(before_reason, after_reason)] += 1
        transition_record = {
            "sample_id": key[0],
            "query_task_id": key[1],
            "question": (
                before_row.get("question")
                if before_row is not None
                else after_row.get("question") if after_row is not None else None
            ),
            "before_reason": before_reason,
            "after_reason": after_reason,
        }
        before_gold_trajectory_count = int((before_row or {}).get("gold_trajectory_count", 0) or 0)
        after_gold_trajectory_count = int((after_row or {}).get("gold_trajectory_count", 0) or 0)
        is_split_case = before_gold_trajectory_count > 1 or after_gold_trajectory_count > 1
        if is_split_case:
            before_coverability = _coverability_state(before_row)
            after_coverability = _coverability_state(after_row)
            before_recall_state = _recall_state(before_row)
            after_recall_state = _recall_state(after_row)
            coverability_transition_counts[(before_coverability, after_coverability)] += 1
            recall_transition_counts[(before_recall_state, after_recall_state)] += 1
        page_coverability_transition_counts[
            (_page_coverability_state(before_row), _page_coverability_state(after_row))
        ] += 1
        if is_split_case:
            multi_hop_transition_record = {
                **transition_record,
                "before_coverability": before_coverability,
                "after_coverability": after_coverability,
                "before_gold_trajectory_recall_at_k": (before_row or {}).get("gold_trajectory_recall_at_k"),
                "after_gold_trajectory_recall_at_k": (after_row or {}).get("gold_trajectory_recall_at_k"),
            }
            if before_coverability != "coverable_in_top_k" and after_coverability == "coverable_in_top_k":
                improved_multi_hop_queries.append(multi_hop_transition_record)
            elif before_coverability == "coverable_in_top_k" and after_coverability != "coverable_in_top_k":
                worsened_multi_hop_queries.append(multi_hop_transition_record)
        strict_answer_error_transition_counts[
            (
                "strict_answer_error"
                if bool((before_row or {}).get("strict_retrieval_success_but_answer_wrong"))
                else "not_strict_answer_error",
                "strict_answer_error"
                if bool((after_row or {}).get("strict_retrieval_success_but_answer_wrong"))
                else "not_strict_answer_error",
            )
        ] += 1
        if before_reason != "correct" and after_reason == "correct":
            improved_queries.append(transition_record)
        elif before_reason == "correct" and after_reason != "correct":
            worsened_queries.append(transition_record)
        elif before_reason not in {"correct", "missing"} and after_reason not in {"correct", "missing"} and before_reason != after_reason:
            changed_failures.append(transition_record)

    def _delta(section_name: str) -> dict[str, float | None]:
        before_section = dict(before_report.get(section_name, {}))
        after_section = dict(after_report.get(section_name, {}))
        keys = sorted(set(before_section) | set(after_section))
        delta: dict[str, float | None] = {}
        for key in keys:
            before_value = before_section.get(key)
            after_value = after_section.get(key)
            if before_value is None or after_value is None:
                delta[key] = None
            else:
                try:
                    delta[key] = float(after_value) - float(before_value)
                except (TypeError, ValueError):
                    delta[key] = None
        return delta

    pass_before = sum(1 for row in before_outcomes.values() if row["reason"] == "correct")
    pass_after = sum(1 for row in after_outcomes.values() if row["reason"] == "correct")
    judge_error_before = sum(1 for row in before_outcomes.values() if row["reason"] == "judge_error")
    judge_error_after = sum(1 for row in after_outcomes.values() if row["reason"] == "judge_error")
    outcome_counts_delta = {
        "correct": pass_after - pass_before,
        "judge_error": judge_error_after - judge_error_before,
        **{
            reason: int(after_report["reason_counts"].get(reason, 0)) - int(before_report["reason_counts"].get(reason, 0))
            for reason in FAILURE_REASON_ORDER
        },
    }

    diff_report = {
        "before_run_meta": before_report["run_meta"],
        "after_run_meta": after_report["run_meta"],
        "totals": {
            "before_failed_queries": before_report["totals"]["failed_queries"],
            "after_failed_queries": after_report["totals"]["failed_queries"],
            "failed_queries_delta": int(after_report["totals"]["failed_queries"]) - int(before_report["totals"]["failed_queries"]),
            "before_failure_rate": before_report["totals"]["failure_rate"],
            "after_failure_rate": after_report["totals"]["failure_rate"],
            "failure_rate_delta": float(after_report["totals"]["failure_rate"]) - float(before_report["totals"]["failure_rate"]),
        },
        "outcome_counts_delta": outcome_counts_delta,
        "reason_counts_delta": {
            reason: int(after_report["reason_counts"].get(reason, 0)) - int(before_report["reason_counts"].get(reason, 0))
            for reason in FAILURE_REASON_ORDER
        },
        "reason_transition_matrix": [
            {
                "before_reason": before_reason,
                "after_reason": after_reason,
                "count": count,
            }
            for (before_reason, after_reason), count in sorted(
                transition_counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ],
        "rank_diagnostics_delta": _delta("rank_diagnostics"),
        "facet_diagnostics_delta": _delta("facet_diagnostics"),
        "fragmentation_diagnostics_delta": _delta("fragmentation_diagnostics"),
        "wiki_fragmentation_diagnostics_delta": _delta("wiki_fragmentation_diagnostics"),
        "page_routing_diagnostics_delta": _delta("page_routing_diagnostics"),
        "memory_preservation_diagnostics_delta": _delta("memory_preservation_diagnostics"),
        "stage_diagnostics_delta": _delta("stage_diagnostics"),
        "compaction_diagnostics_delta": _delta("compaction_diagnostics"),
        "multi_hop_top_k_diagnostics_delta": _delta("multi_hop_top_k_diagnostics"),
        "direct_trajectory_ablation_diagnostics_delta": _delta("direct_trajectory_ablation_diagnostics"),
        "direct_vs_routed_retrieval_diagnostics_delta": _delta("direct_vs_routed_retrieval_diagnostics"),
        "retrieval_reflection_diagnostics_delta": _delta("retrieval_reflection_diagnostics"),
        "judge_diagnostics_delta": _judge_diagnostics_delta(
            dict(before_report.get("judge_diagnostics", {})),
            dict(after_report.get("judge_diagnostics", {})),
        ),
        "llm_call_diagnostics_delta": _llm_call_diagnostics_delta(
            dict(before_report.get("llm_call_diagnostics", {})),
            dict(after_report.get("llm_call_diagnostics", {})),
        ),
        "fallback_repair_diagnostics_delta": _fallback_repair_diagnostics_delta(
            dict(before_report.get("fallback_repair_diagnostics", {})),
            dict(after_report.get("fallback_repair_diagnostics", {})),
        ),
        "cutoff_diagnostics_delta": _cutoff_diagnostics_delta(
            dict(before_report.get("cutoff_diagnostics", {})),
            dict(after_report.get("cutoff_diagnostics", {})),
        ),
        "trajectory_length_diagnostics_delta": _trajectory_length_diagnostics_delta(
            dict(before_report.get("trajectory_length_diagnostics", {})),
            dict(after_report.get("trajectory_length_diagnostics", {})),
        ),
        "trajectory_drift_diagnostics_delta": _delta("trajectory_drift_diagnostics"),
        "coverage_transition_counts": {
            "page_routing": [
                {
                    "before": before_state,
                    "after": after_state,
                    "count": count,
                }
                for (before_state, after_state), count in sorted(
                    page_coverability_transition_counts.items(),
                    key=lambda item: (-item[1], item[0][0], item[0][1]),
                )
            ],
            "coverability": [
                {
                    "before": before_state,
                    "after": after_state,
                    "count": count,
                }
                for (before_state, after_state), count in sorted(
                    coverability_transition_counts.items(),
                    key=lambda item: (-item[1], item[0][0], item[0][1]),
                )
            ],
            "gold_trajectory_recall": [
                {
                    "before": before_state,
                    "after": after_state,
                    "count": count,
                }
                for (before_state, after_state), count in sorted(
                    recall_transition_counts.items(),
                    key=lambda item: (-item[1], item[0][0], item[0][1]),
                )
            ],
            "strict_answer_error": [
                {
                    "before": before_state,
                    "after": after_state,
                    "count": count,
                }
                for (before_state, after_state), count in sorted(
                    strict_answer_error_transition_counts.items(),
                    key=lambda item: (-item[1], item[0][0], item[0][1]),
                )
            ],
        },
        "strict_answer_error_transition_counts": [
            {
                "before": before_state,
                "after": after_state,
                "count": count,
            }
            for (before_state, after_state), count in sorted(
                strict_answer_error_transition_counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ],
        "improved_queries": improved_queries[:top_examples_per_bucket],
        "worsened_queries": worsened_queries[:top_examples_per_bucket],
        "changed_failures": changed_failures[:top_examples_per_bucket],
        "improved_multi_hop_queries": improved_multi_hop_queries[:top_examples_per_bucket],
        "worsened_multi_hop_queries": worsened_multi_hop_queries[:top_examples_per_bucket],
        "before_report": before_report,
        "after_report": after_report,
    }
    return diff_report


def _render_optional_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _render_unknown(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str) and not value.strip():
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _render_optional_number(value)


def _render_unknown_bool(value: Any) -> str:
    if value is None:
        return "unknown"
    return "yes" if bool(value) else "no"


def _should_render_temporal_grounding(example: dict[str, Any]) -> bool:
    if example.get("temporal_anchor_available_in_raw") is not None:
        return True
    if example.get("temporal_anchor_visible_in_prompt") is not None:
        return True
    if example.get("temporal_anchor_missing_in_answer_context") is not None:
        return True
    if list(example.get("relative_time_terms_present") or []):
        return True
    try:
        return int(example.get("temporal_anchor_hint_count") or 0) > 0
    except (TypeError, ValueError):
        return False


def _render_count_pair(count: Any, total: Any, *, available: bool = True) -> str:
    rendered_count = _render_unknown(count) if available else "-"
    return f"{rendered_count}/{_render_unknown(total)}"


def _render_judge_score(verdict: Any, score: Any) -> str:
    if score is None:
        verdict_key = str(verdict or "").lower()
        score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(verdict_key)
    if score is None:
        return "-"
    try:
        return f"{float(score):.1f}"
    except (TypeError, ValueError):
        return str(score)


def _has_display_text(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "none"


def _render_compact_json_list(values: list[Any], *, limit: int = 8) -> str:
    deduped: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    visible = deduped[:limit]
    rendered = json.dumps(visible, ensure_ascii=False)
    omitted = len(deduped) - len(visible)
    if omitted > 0:
        rendered += f" ... (+{omitted} more unique; {len(values)} raw)"
    elif len(values) != len(deduped):
        rendered += f" ({len(values) - len(deduped)} duplicate raw omitted)"
    return rendered


def _common_cutoff_keys(cutoff_section: dict[str, Any]) -> list[str]:
    available = sorted((int(key) for key in cutoff_section if str(key).isdigit()))
    if not available:
        return []
    common = [1, 3, 5, 10, max(available)]
    return [str(value) for value in dict.fromkeys(common) if value in set(available)]


def print_locomo_failure_report(
    report: dict[str, Any],
    console: Console | None = None,
    *,
    show_ranks: bool = False,
    show_facets: bool = False,
) -> None:
    active_console = console or Console()
    run_meta = dict(report.get("run_meta", {}))
    totals = dict(report.get("totals", {}))
    reason_counts = dict(report.get("reason_counts", {}))
    reason_rates = dict(report.get("reason_rates_over_failed", {}))
    coverage = dict(report.get("retrieval_coverage", {}))
    rank_diagnostics = dict(report.get("rank_diagnostics", {}))
    judge_diagnostics = dict(report.get("judge_diagnostics", {}))
    answer_synthesis_diagnostics = dict(report.get("answer_synthesis_diagnostics", {}))
    llm_call_diagnostics = dict(report.get("llm_call_diagnostics", {}))
    fallback_repair_diagnostics = dict(report.get("fallback_repair_diagnostics", {}))
    reflection_diagnostics = dict(report.get("retrieval_reflection_diagnostics", {}))
    facet_diagnostics = dict(report.get("facet_diagnostics", {}))
    fragmentation_diagnostics = dict(report.get("fragmentation_diagnostics", {}))
    wiki_fragmentation_diagnostics = dict(report.get("wiki_fragmentation_diagnostics", {}))
    routing_text_diagnostics = dict(report.get("routing_text_diagnostics", {}))
    source_event_diagnostics = dict(report.get("source_event_reranking_diagnostics", {}))
    page_routing_diagnostics = dict(report.get("page_routing_diagnostics", {}))
    memory_preservation_diagnostics = dict(report.get("memory_preservation_diagnostics", {}))
    memory_force_recall_diagnostics = dict(report.get("memory_force_recall_diagnostics", {}))
    stage_diagnostics = dict(report.get("stage_diagnostics", {}))
    compaction_diagnostics = dict(report.get("compaction_diagnostics", {}))
    multi_hop_diagnostics = dict(report.get("multi_hop_top_k_diagnostics", {}))
    direct_ablation_diagnostics = dict(report.get("direct_trajectory_ablation_diagnostics", {}))
    direct_vs_routed_diagnostics = dict(report.get("direct_vs_routed_retrieval_diagnostics", {}))
    conv26_like_diagnostics = dict(report.get("conv26_like_trajectory_selection_diagnostics", {}))
    split_case_redundancy_diagnostics = dict(report.get("split_case_redundancy_diagnostics", {}))
    cutoff_diagnostics = dict(report.get("cutoff_diagnostics", {}))
    page_cutoffs = dict(cutoff_diagnostics.get("page_cutoffs", {}))
    trajectory_cutoffs = dict(cutoff_diagnostics.get("trajectory_cutoffs", {}))
    offline_parameter_diagnostics = dict(report.get("offline_parameter_diagnostics", {}))
    offline_page_cutoffs = dict(offline_parameter_diagnostics.get("page_cutoffs", {}))
    offline_trajectory_cutoffs = dict(offline_parameter_diagnostics.get("trajectory_cutoffs", {}))
    trajectory_length_diagnostics = dict(report.get("trajectory_length_diagnostics", {}))
    trajectory_drift_diagnostics = dict(report.get("trajectory_drift_diagnostics", {}))
    text_only_filter_diagnostics = dict(report.get("text_only_filter_diagnostics", {}))
    cuda_preflight_diagnostics = dict(report.get("cuda_preflight_diagnostics", {}))
    metadata_term_diagnostics = dict(report.get("metadata_term_diagnostics", {}))

    active_console.print(
        f"[bold]LOCOMO Failure Attribution[/bold] "
        f"run_id={run_meta.get('run_id')} scope={run_meta.get('dataset_scope_key')} "
        f"path={run_meta.get('resolved_run_dir')}"
    )
    active_console.print(
        f"failed_queries={totals.get('failed_queries', 0)} / total_queries={totals.get('total_queries', 0)} "
        f"failure_rate={float(totals.get('failure_rate', 0.0)):.4f}"
    )

    reason_table = Table(title="Failure Reasons")
    reason_table.add_column("reason")
    reason_table.add_column("count", justify="right")
    reason_table.add_column("rate_over_failed", justify="right")
    for reason in FAILURE_REASON_ORDER:
        reason_table.add_row(
            reason,
            str(int(reason_counts.get(reason, 0))),
            f"{float(reason_rates.get(reason, 0.0)):.4f}",
        )
    active_console.print(reason_table)

    if text_only_filter_diagnostics:
        filter_table = Table(title="Text-Only Evaluation Filter")
        filter_table.add_column("metric")
        filter_table.add_column("value", justify="right")
        for key in [
            "policy",
            "total_queries",
            "included_count",
            "excluded_count",
            "ambiguous_count",
        ]:
            filter_table.add_row(key, _render_unknown(text_only_filter_diagnostics.get(key)))
        for reason, count in sorted(dict(text_only_filter_diagnostics.get("by_reason") or {}).items()):
            filter_table.add_row(f"reason.{reason}", str(int(count)))
        for visual_type, count in sorted(
            dict(text_only_filter_diagnostics.get("by_visual_dependency_type") or {}).items()
        ):
            filter_table.add_row(f"type.{visual_type}", str(int(count)))
        active_console.print(filter_table)

    if cuda_preflight_diagnostics:
        cuda_table = Table(title="CUDA Preflight")
        cuda_table.add_column("metric")
        cuda_table.add_column("value", justify="right")
        for key in [
            "mode",
            "enabled",
            "risk",
            "reserve_gb",
            "visible_device_count",
            "reserved_device_indices",
        ]:
            cuda_table.add_row(key, _render_unknown(cuda_preflight_diagnostics.get(key)))
        warnings = list(cuda_preflight_diagnostics.get("warnings") or [])
        errors = list(cuda_preflight_diagnostics.get("errors") or [])
        assignments_raw = cuda_preflight_diagnostics.get("assignments") or {}
        assignments = (
            dict(assignments_raw)
            if isinstance(assignments_raw, dict)
            else list(assignments_raw)
            if isinstance(assignments_raw, list)
            else []
        )
        reservations = list(cuda_preflight_diagnostics.get("reservations") or [])
        cuda_table.add_row("warning_count", str(len(warnings)))
        cuda_table.add_row("error_count", str(len(errors)))
        cuda_table.add_row("assignment_count", str(len(assignments)))
        cuda_table.add_row("reservation_count", str(len(reservations)))
        for index, warning in enumerate(warnings[:3], start=1):
            cuda_table.add_row(f"warning.{index}", collapse_whitespace(str(warning))[:160])
        for index, error in enumerate(errors[:3], start=1):
            cuda_table.add_row(f"error.{index}", collapse_whitespace(str(error))[:160])
        active_console.print(cuda_table)

    judge_table = Table(title="Judge Diagnostics")
    judge_table.add_column("metric")
    judge_table.add_column("value", justify="right")
    for key in [
        "judge_evaluable_count",
        "judge_execution_failed_count",
        "judge_execution_failed_rate_over_all",
        "partial_count",
        "partial_rate_over_all",
        "partial_rate_over_non_correct",
        "mean_partial_credit_judge_acc",
        "structured_requested_rate_over_all",
        "structured_success_rate_over_all",
        "text_fallback_rate_over_all",
        "text_only_rate_over_all",
        "incorrect_queries_judged_via_text_fallback_count",
        "incorrect_queries_judged_via_text_fallback_rate_over_incorrect",
        "judge_semantic_override_count",
        "judge_semantic_override_rate",
        "partial_to_correct_override_count",
        "incorrect_to_correct_override_count",
    ]:
        judge_table.add_row(key, _render_optional_number(judge_diagnostics.get(key)))
    for category, count in sorted(dict(judge_diagnostics.get("structured_fallback_category_counts", {})).items()):
        judge_table.add_row(f"structured_fallback_category.{category}", str(int(count)))
    active_console.print(judge_table)

    if answer_synthesis_diagnostics:
        synthesis_table = Table(title="Answer Synthesis Diagnostics")
        synthesis_table.add_column("metric")
        synthesis_table.add_column("value", justify="right")
        for key in [
            "used_count",
            "used_rate_over_all",
            "structured_count",
            "freeform_v2_count",
            "freeform_used_count",
            "text_json_count",
            "legacy_fallback_count",
            "can_answer_count",
            "can_answer_rate_over_all",
            "can_answer_but_non_correct_count",
            "can_answer_but_non_correct_rate_over_can_answer",
            "count_answer_type_count",
            "mean_counted_events_over_count_questions",
            "mean_excluded_events_over_count_questions",
            "invalid_supporting_ref_count",
            "invalid_family_ref_count",
            "question_type_mismatch_count",
            "typed_retry_used_count",
            "typed_retry_success_count",
            "typed_retry_success_rate",
            "answer_type_verification_used_count",
            "answer_type_verification_success_count",
            "answer_type_verification_success_rate",
            "answer_type_mismatch_count",
            "type_mismatch_recovered_count",
            "typed_retry_text_json_normalized_count",
            "internal_abstain_reason_suppressed_count",
            "expected_type_text_rejected_count",
            "source_family_alias_hit_count",
            "count_ref_validation_rejected_count",
            "count_validation_llm_used_count",
            "count_validation_llm_success_count",
            "count_validation_llm_changed_count",
            "count_validation_llm_success_rate",
            "source_derived_count_candidate_count",
            "source_derived_count_candidate_accepted_count",
            "source_derived_pronoun_caption_candidate_count",
            "source_derived_passive_rejected_count",
            "temporal_alignment_rejected_count",
            "temporal_repair_used_count",
            "temporal_low_confidence_candidate_count",
            "temporal_no_query_relevant_candidate_count",
            "temporal_candidate_selection_rejected_count",
            "date_question_list_repair_blocked_count",
            "list_scope_rejected_item_count",
            "missing_required_item_count",
            "overgeneric_answer_count",
            "scope_mismatched_extra_item_count",
            "specific_item_repair_used_count",
            "missing_supported_list_item_count",
            "abstain_despite_supported_items_count",
            "list_coverage_repair_used_count",
            "repair_dropped_supported_item_count",
            "repair_discarded_by_post_validation_count",
            "repair_arbitration_triggered_count",
            "repair_arbitration_used_count",
            "repair_arbitration_keep_initial_count",
            "repair_arbitration_use_repair_count",
            "repair_arbitration_safe_abstain_count",
            "repair_arbitration_failed_count",
            "list_required_item_recall_after_repair",
            "bridge_finalization_used_count",
            "bridge_finalization_conflict_count",
            "answer_bridge_repair_used_count",
            "preference_query_shape_count",
            "count_validation_excluded_event_count",
            "bridge_fact_used_count",
        ]:
            synthesis_table.add_row(key, _render_optional_number(answer_synthesis_diagnostics.get(key)))
        excluded_reason_items = sorted(
            dict(answer_synthesis_diagnostics.get("excluded_event_reason_counts", {})).items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
        for reason, count in excluded_reason_items[:10]:
            synthesis_table.add_row(f"excluded_event_reason.{reason}", str(int(count)))
        if len(excluded_reason_items) > 10:
            omitted_count = sum(int(count) for _, count in excluded_reason_items[10:])
            synthesis_table.add_row("excluded_event_reason.other_omitted", str(int(omitted_count)))
        active_console.print(synthesis_table)

    if conv26_like_diagnostics:
        conv26_table = Table(title="Conv-26-Like Trajectory Selection Diagnostics")
        conv26_table.add_column("metric")
        conv26_table.add_column("value", justify="right")
        for key in [
            "selected_page_universe_size_mean_over_all",
            "selected_page_universe_size_median_over_all",
            "selected_page_universe_size_mean_over_failed",
            "broad_entity_cap_used_count_over_all",
            "broad_entity_cap_used_count_over_failed",
            "family_match_available_query_count",
            "family_match_gold_query_count",
            "family_match_recall_in_selection_pool",
            "family_match_recall_in_final_top_k",
            "answer_synthesis_invalid_family_ref_count",
            "answer_synthesis_question_type_mismatch_count",
        ]:
            conv26_table.add_row(key, _render_optional_number(conv26_like_diagnostics.get(key)))
        active_console.print(conv26_table)

    if llm_call_diagnostics:
        llm_overall = dict(llm_call_diagnostics.get("overall") or {})
        llm_table = Table(title="LLM Call Diagnostics")
        llm_table.add_column("scope")
        llm_table.add_column("provider_calls", justify="right")
        llm_table.add_column("logical_calls", justify="right")
        llm_table.add_column("tokens", justify="right")
        llm_table.add_column("fallbacks", justify="right")
        llm_table.add_column("repairs", justify="right")
        llm_table.add_column("errors", justify="right")
        llm_table.add_row(
            "overall",
            _render_optional_number(llm_overall.get("provider_call_count")),
            _render_optional_number(llm_overall.get("logical_call_count")),
            _render_optional_number(llm_overall.get("total_tokens")),
            _render_optional_number(llm_overall.get("fallback_count")),
            _render_optional_number(llm_overall.get("repair_count")),
            _render_optional_number(llm_overall.get("error_count")),
        )
        for phase, phase_usage in dict(llm_call_diagnostics.get("by_phase") or {}).items():
            phase_usage = dict(phase_usage or {})
            llm_table.add_row(
                f"phase:{phase}",
                _render_optional_number(phase_usage.get("provider_call_count")),
                _render_optional_number(phase_usage.get("logical_call_count")),
                _render_optional_number(phase_usage.get("total_tokens")),
                _render_optional_number(phase_usage.get("fallback_count")),
                _render_optional_number(phase_usage.get("repair_count")),
                _render_optional_number(phase_usage.get("error_count")),
            )
        active_console.print(llm_table)

        task_table = Table(title="LLM Calls By Task (Top Token Users)")
        task_table.add_column("task")
        task_table.add_column("provider_calls", justify="right")
        task_table.add_column("logical_calls", justify="right")
        task_table.add_column("tokens", justify="right")
        task_table.add_column("fallbacks", justify="right")
        task_table.add_column("repairs", justify="right")
        for task, task_usage in list(dict(llm_call_diagnostics.get("by_task") or {}).items())[:12]:
            task_usage = dict(task_usage or {})
            task_table.add_row(
                str(task),
                _render_optional_number(task_usage.get("provider_call_count")),
                _render_optional_number(task_usage.get("logical_call_count")),
                _render_optional_number(task_usage.get("total_tokens")),
                _render_optional_number(task_usage.get("fallback_count")),
                _render_optional_number(task_usage.get("repair_count")),
            )
        active_console.print(task_table)

    if fallback_repair_diagnostics:
        fr_overall = dict(fallback_repair_diagnostics.get("overall") or {})
        fr_table = Table(title="Fallback / Repair Risk Diagnostics")
        fr_table.add_column("metric")
        fr_table.add_column("value", justify="right")
        for key in [
            "diagnostic_mode",
            "loaded_event_count",
        ]:
            value = fallback_repair_diagnostics.get(key)
            if value is not None:
                fr_table.add_row(key, str(value))
        for key in [
            "event_count",
            "fallback_event_count",
            "repair_event_count",
            "repair_discard_rate",
            "extra_provider_call_count",
            "extra_total_tokens",
            "quality_risk_event_count",
            "quality_risk_weighted_event_count",
        ]:
            fr_table.add_row(key, _render_optional_number(fr_overall.get(key)))
        paper_ready = dict(fallback_repair_diagnostics.get("paper_ready") or {})
        for key in [
            "extra_llm_call_rate",
            "extra_token_overhead_rate",
            "mean_judge_score_with_fallback",
            "mean_judge_score_without_fallback",
            "incorrect_rate_with_repair",
            "incorrect_rate_without_repair",
        ]:
            fr_table.add_row(key, _render_optional_number(paper_ready.get(key)))
        active_console.print(fr_table)

        fr_task_table = Table(title="Fallback / Repair By Task")
        fr_task_table.add_column("task")
        fr_task_table.add_column("fallbacks", justify="right")
        fr_task_table.add_column("repairs", justify="right")
        fr_task_table.add_column("discard_rate", justify="right")
        fr_task_table.add_column("extra_tokens", justify="right")
        for task, task_usage in list(dict(fallback_repair_diagnostics.get("by_task") or {}).items())[:12]:
            task_usage = dict(task_usage or {})
            fr_task_table.add_row(
                str(task),
                _render_optional_number(task_usage.get("fallback_event_count")),
                _render_optional_number(task_usage.get("repair_event_count")),
                _render_optional_number(task_usage.get("repair_discard_rate")),
                _render_optional_number(task_usage.get("extra_total_tokens")),
            )
        active_console.print(fr_task_table)

    reflection_table = Table(title="Retrieval Reflection Diagnostics")
    reflection_table.add_column("metric")
    reflection_table.add_column("value", justify="right")
    for key in [
        "retry_triggered_count",
        "retry_triggered_rate_over_all",
        "wiki_reroute_used_count",
        "raw_rescue_attempted_count",
        "raw_rescue_attempted_rate_over_retry",
        "raw_rescue_skipped_count",
        "raw_rescue_skipped_rate_over_retry",
        "raw_rescue_used_count",
        "raw_rescue_used_rate_over_all",
        "raw_rescue_hit_count",
        "raw_rescue_hit_rate_over_raw_rescue",
        "raw_rescue_hit_rate_over_attempted",
        "semantic_weak_trigger_count",
        "semantic_weak_trigger_rate_over_retry",
        "post_reflection_raw_rescue_count",
        "post_reflection_raw_rescue_rate_over_retry",
        "answer_changed_count",
        "answer_changed_rate_over_retry",
        "retry_correct_or_partial_count",
        "retry_correct_or_partial_rate_over_retry",
    ]:
        reflection_table.add_row(key, _render_optional_number(reflection_diagnostics.get(key)))
    active_console.print(reflection_table)

    coverage_table = Table(title="Retrieval Coverage Over Failed Queries")
    coverage_table.add_column("metric")
    coverage_table.add_column("rate", justify="right")
    for key in [
        "memory_hit_rate_over_failed",
        "coarse_hit_rate_over_failed",
        "fine_hit_rate_over_failed",
        "expanded_hit_rate_over_failed",
        "grounded_hit_rate_over_failed",
        "expansion_recovered_rate_over_failed",
    ]:
        coverage_table.add_row(key, f"{float(coverage.get(key, 0.0)):.4f}")
    active_console.print(coverage_table)

    if source_event_diagnostics:
        source_event_table = Table(title="Source Event Reranking Diagnostics")
        source_event_table.add_column("metric")
        source_event_table.add_column("value", justify="right")
        for key in [
            "source_event_metadata_trajectory_count",
            "source_event_match_used_count",
            "source_event_selected_match_count",
            "source_event_match_boosted_gold_count",
            "source_event_match_missing_gold_count",
        ]:
            source_event_table.add_row(key, _render_optional_number(source_event_diagnostics.get(key)))
        active_console.print(source_event_table)

    rank_table = Table(title="Rank Diagnostics Over Failed Queries")
    rank_table.add_column("metric")
    rank_table.add_column("value", justify="right")
    for key in [
        "coarse_rank_observed_count",
        "coarse_rank_mean",
        "coarse_rank_median",
        "coarse_top_k_hit_rate_over_failed",
        "fine_rank_observed_count",
        "fine_rank_mean",
        "fine_rank_median",
        "fine_top_r_hit_rate_over_failed",
    ]:
        value = rank_diagnostics.get(key)
        rendered = "-" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
        rank_table.add_row(key, rendered)
    active_console.print(rank_table)

    facet_table = Table(title="Facet / Fragmentation Diagnostics Over Failed Queries")
    facet_table.add_column("metric")
    facet_table.add_column("rate", justify="right")
    for key, source in [
        ("facet_present_for_gold_ref_rate", facet_diagnostics),
        ("exact_facet_missing_rate", facet_diagnostics),
        ("gold_ref_present_but_claim_generalized_rate", facet_diagnostics),
        ("gold_ref_present_but_no_supported_facet_rate", facet_diagnostics),
        ("split_across_trajectories_rate", fragmentation_diagnostics),
        ("entity_linked_could_help_rate", fragmentation_diagnostics),
        ("entity_linked_added_relevant_snapshot_rate", fragmentation_diagnostics),
    ]:
        facet_table.add_row(key, f"{float(source.get(key, 0.0)):.4f}")
    active_console.print(facet_table)

    if wiki_fragmentation_diagnostics:
        wiki_fragmentation_table = Table(title="Wiki Fragmentation Diagnostics")
        wiki_fragmentation_table.add_column("metric")
        wiki_fragmentation_table.add_column("value", justify="right")
        for key in [
            "diagnostic_mode",
            "sample_count",
            "mean_non_index_page_count",
            "mean_singleton_non_index_page_rate",
            "mean_trajectories_per_non_index_page",
            "post_plan_rescue_singleton_count",
            "entity_facet_singleton_count",
            "allowed_specific_singleton_count",
            "low_quality_singleton_count",
            "low_quality_singleton_merged_count",
            "overwide_non_index_page_count",
            "overwide_page_split_count",
            "max_non_index_trajectory_count",
            "selected_page_slot_count",
            "selected_singleton_page_slot_count",
            "selected_singleton_page_slot_rate",
            "selected_medium_page_slot_count",
            "selected_medium_page_slot_rate",
            "singleton_page_penalty_applied_count",
            "low_quality_singleton_penalty_applied_count",
            "medium_page_bonus_applied_count",
        ]:
            wiki_fragmentation_table.add_row(key, _render_optional_number(wiki_fragmentation_diagnostics.get(key)))
        active_console.print(wiki_fragmentation_table)

    if routing_text_diagnostics:
        routing_text_table = Table(title="Routing Text Diagnostics")
        routing_text_table.add_column("metric")
        routing_text_table.add_column("value", justify="right")
        for key in [
            "page_count",
            "routing_text_cleaned_page_count",
            "routing_text_internal_marker_page_count",
            "routing_text_internal_marker_rate",
        ]:
            routing_text_table.add_row(key, _render_optional_number(routing_text_diagnostics.get(key)))
        active_console.print(routing_text_table)

    if metadata_term_diagnostics:
        metadata_terms_table = Table(title="Metadata Term Diagnostics")
        metadata_terms_table.add_column("metric")
        metadata_terms_table.add_column("value", justify="right")
        for key in [
            "trajectory_count",
            "summary_keyword_mean_count",
            "summary_keyword_internal_term_hit_count",
            "historical_item_internal_term_hit_count",
            "historical_item_summary_fallback_count",
        ]:
            metadata_terms_table.add_row(key, _render_optional_number(metadata_term_diagnostics.get(key)))
        metadata_terms_table.add_row(
            "summary_keyword_policy_counts",
            _render_compact_json_list([metadata_term_diagnostics.get("summary_keyword_policy_counts") or {}]),
        )
        metadata_terms_table.add_row(
            "historical_item_policy_counts",
            _render_compact_json_list([metadata_term_diagnostics.get("historical_item_policy_counts") or {}]),
        )
        active_console.print(metadata_terms_table)

    preservation_table = Table(title="Memory Preservation Diagnostics Over Failed Queries")
    preservation_table.add_column("metric")
    preservation_table.add_column("rate", justify="right")
    for key in [
        "gold_answer_fully_present_in_claims_rate_over_failed",
        "gold_answer_fully_present_in_trajectory_summaries_rate_over_failed",
        "gold_answer_fully_present_in_wiki_pages_rate_over_failed",
        "gold_item_present_in_claims_rate_over_failed",
        "gold_item_present_in_wiki_rate_over_failed",
        "gold_item_missing_but_umbrella_present_rate_over_failed",
        "trajectory_storage_miss_rate_over_failed",
        "summary_compression_miss_rate_over_failed",
        "wiki_compilation_compression_miss_rate_over_failed",
    ]:
        preservation_table.add_row(key, f"{float(memory_preservation_diagnostics.get(key, 0.0)):.4f}")
    active_console.print(preservation_table)

    force_recall_table = Table(title="Memory Force-Recall Diagnostics")
    force_recall_table.add_column("metric")
    force_recall_table.add_column("value", justify="right")
    for key in [
        "forced_memory_seed_count",
        "low_salience_memory_count",
        "llm_no_memory_forced_count",
        "zero_claim_episodic_candidate_count",
        "zero_claim_episodic_persisted_count",
        "zero_claim_low_salience_skipped_count",
        "failed_queries_with_gold_refs_in_forced_memory_count",
        "failed_queries_with_gold_refs_in_forced_memory_rate_over_failed",
        "failed_queries_with_gold_refs_in_low_salience_memory_count",
        "failed_queries_with_gold_refs_in_low_salience_memory_rate_over_failed",
        "gold_refs_in_forced_memory_count",
        "gold_refs_in_low_salience_memory_count",
    ]:
        force_recall_table.add_row(key, _render_optional_number(memory_force_recall_diagnostics.get(key)))
    active_console.print(force_recall_table)

    page_table = Table(title="Page Routing Diagnostics Over Failed Queries")
    page_table.add_column("metric")
    page_table.add_column("value", justify="right")
    for key in [
        "all_gold_pages_in_top_t_rate_over_failed",
        "mean_page_routing_recall_over_failed",
        "median_page_routing_recall_over_failed",
        "all_gold_trajectories_in_selected_pages_rate_over_failed",
        "all_gold_trajectories_in_final_top_k_after_page_routing_rate_over_failed",
    ]:
        value = page_routing_diagnostics.get(key)
        rendered = "-" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
        page_table.add_row(key, rendered)
    active_console.print(page_table)

    stage_table = Table(title="Routing / Retrieval Stage Diagnostics Over Failed Queries")
    stage_table.add_column("metric")
    stage_table.add_column("rate", justify="right")
    for key in [
        "page_routing_failed_rate_over_failed",
        "selected_pages_failed_to_cover_all_gold_trajectories_rate_over_failed",
        "trajectory_retrieval_failed_after_page_routing_rate_over_failed",
        "all_gold_refs_grounded_rate_over_failed",
        "strict_retrieval_success_but_answer_wrong_rate_over_failed",
    ]:
        stage_table.add_row(key, f"{float(stage_diagnostics.get(key, 0.0)):.4f}")
    active_console.print(stage_table)

    compaction_table = Table(title="Expansion / Compaction Diagnostics Over Failed Queries")
    compaction_table.add_column("metric")
    compaction_table.add_column("rate", justify="right")
    for key in [
        "pre_compaction_expansion_cover_rate_over_failed",
        "snapshot_compaction_miss_rate_over_failed",
        "source_compaction_miss_rate_over_failed",
    ]:
        compaction_table.add_row(key, f"{float(compaction_diagnostics.get(key, 0.0)):.4f}")
    active_console.print(compaction_table)

    multi_hop_table = Table(title="Multi-Hop Top-k Coverage Over Failed Queries")
    multi_hop_table.add_column("metric")
    multi_hop_table.add_column("value", justify="right")
    for key in [
        "split_case_count",
        "split_case_rate_over_failed",
        "mean_gold_trajectory_recall_at_k_over_split_cases",
        "median_gold_trajectory_recall_at_k_over_split_cases",
        "selection_pool_available_failed_gold_query_count",
        "selection_pool_available_split_case_count",
        "mean_gold_trajectory_recall_in_selection_pool_over_failed",
        "mean_gold_trajectory_recall_in_selection_pool_over_split_cases",
        "gold_trajectory_in_selection_pool_rate",
        "gold_trajectory_lost_before_selection_pool_count",
        "gold_trajectory_lost_during_final_top_k_count",
        "coverage_query_shape_trigger_rate",
        "coverage_query_shape_trigger_rate_over_all",
        "selection_pool_can_cover_all_gold_trajectories_rate",
        "selection_pool_can_cover_all_gold_trajectories_rate_over_split_cases",
        "all_gold_trajectories_in_top_k_rate_over_split_cases",
        "top_k_can_cover_all_gold_refs_rate_over_split_cases",
        "mean_top_k_gold_ref_coverage_rate_over_split_cases",
        "mean_top_k_redundant_entity_cluster_ratio_over_split_cases",
        "zero_gold_trajectory_in_top_k_rate_over_split_cases",
        "one_gold_trajectory_in_top_k_rate_over_split_cases",
        "two_or_more_gold_trajectories_in_top_k_rate_over_split_cases",
    ]:
        value = multi_hop_diagnostics.get(key)
        rendered = "-" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
        multi_hop_table.add_row(key, rendered)
    active_console.print(multi_hop_table)

    if direct_ablation_diagnostics:
        direct_table = Table(title="Direct Trajectory Ablation")
        direct_table.add_column("metric")
        direct_table.add_column("value", justify="right")
        for key in [
            "query_count",
            "gold_trajectory_query_count",
            "mean_current_page_routed_gold_trajectory_recall_at_k",
            "mean_direct_gold_trajectory_recall_at_k",
            "direct_all_gold_trajectories_in_top_k_rate",
            "direct_top_k_can_cover_all_gold_refs_rate",
            "mean_direct_top_k_gold_ref_coverage_rate",
            "mean_direct_vs_page_routed_recall_delta",
            "page_routing_bottleneck_suspected_count",
            "page_routing_bottleneck_suspected_rate",
        ]:
            direct_table.add_row(key, _render_optional_number(direct_ablation_diagnostics.get(key)))
        for bucket, bucket_row in sorted(dict(direct_ablation_diagnostics.get("by_query_shape") or {}).items())[:12]:
            bucket_row = dict(bucket_row or {})
            direct_table.add_row(
                f"shape.{bucket}.direct_recall",
                _render_optional_number(bucket_row.get("mean_direct_gold_trajectory_recall_at_k")),
            )
            direct_table.add_row(
                f"shape.{bucket}.delta",
                _render_optional_number(bucket_row.get("mean_direct_vs_page_routed_recall_delta")),
            )
        active_console.print(direct_table)

    if direct_vs_routed_diagnostics:
        all_direct = dict(direct_vs_routed_diagnostics.get("all_queries") or {})
        direct_no_routing_table = Table(title="Direct No-Page-Routing Ablation")
        direct_no_routing_table.add_column("metric")
        direct_no_routing_table.add_column("value", justify="right")
        for key in [
            "mean_current_page_routed_gold_trajectory_recall_at_k",
            "mean_direct_gold_trajectory_recall_at_k",
            "direct_all_gold_trajectories_in_top_k_rate",
            "direct_top_k_can_cover_all_gold_refs_rate",
            "mean_routed_candidate_universe_size",
            "mean_direct_candidate_universe_size",
            "mean_direct_candidate_universe_to_routed_universe_ratio",
            "mean_routed_vs_direct_top_k_overlap_rate",
            "mean_direct_top_k_not_in_selected_page_universe_count",
            "mean_direct_top_k_estimated_context_token_count",
        ]:
            direct_no_routing_table.add_row(key, _render_optional_number(all_direct.get(key)))
        active_console.print(direct_no_routing_table)

        direct_cutoffs = dict(direct_vs_routed_diagnostics.get("direct_cutoffs") or {})
        if direct_cutoffs:
            direct_cutoff_table = Table(title="Direct No-Page-Routing Cutoffs")
            direct_cutoff_table.add_column("k")
            direct_cutoff_table.add_column("observed", justify="right")
            direct_cutoff_table.add_column("all_traj_all", justify="right")
            direct_cutoff_table.add_column("traj_recall_all", justify="right")
            direct_cutoff_table.add_column("refs_cover_all", justify="right")
            direct_cutoff_table.add_column("est_tokens", justify="right")
            direct_cutoff_table.add_column("not_saved", justify="right")
            for cutoff in _common_cutoff_keys(direct_cutoffs):
                row = dict(direct_cutoffs.get(cutoff, {}))
                all_queries = dict(row.get("all_queries", {}))
                direct_cutoff_table.add_row(
                    cutoff,
                    _render_unknown(all_queries.get("observed_query_count")),
                    _render_optional_number(all_queries.get("all_gold_trajectories_rate")),
                    _render_optional_number(all_queries.get("mean_gold_trajectory_recall")),
                    _render_optional_number(all_queries.get("top_k_can_cover_all_gold_refs_rate")),
                    _render_optional_number(all_queries.get("mean_estimated_context_token_count")),
                    _render_unknown(all_queries.get("not_observed_query_count")),
                )
            active_console.print(direct_cutoff_table)

    redundancy_table = Table(title="Split-Case Redundancy Diagnostics")
    redundancy_table.add_column("metric")
    redundancy_table.add_column("value", justify="right")
    for key in [
        "mean_top_k_redundant_entity_cluster_ratio_over_split_cases",
        "mean_selected_page_cluster_count_over_failed",
    ]:
        redundancy_table.add_row(key, _render_optional_number(split_case_redundancy_diagnostics.get(key)))
    active_console.print(redundancy_table)

    if page_cutoffs:
        page_cutoff_table = Table(title="Page Cutoff Diagnostics")
        page_cutoff_table.add_column("cutoff")
        page_cutoff_table.add_column("all_pages_all", justify="right")
        page_cutoff_table.add_column("page_recall_all", justify="right")
        page_cutoff_table.add_column("page_traj_all", justify="right")
        page_cutoff_table.add_column("all_pages_failed", justify="right")
        page_cutoff_table.add_column("page_traj_failed", justify="right")
        for cutoff in _common_cutoff_keys(page_cutoffs):
            row = dict(page_cutoffs.get(cutoff, {}))
            all_queries = dict(row.get("all_queries", {}))
            failed_queries = dict(row.get("failed_queries", {}))
            page_cutoff_table.add_row(
                cutoff,
                _render_optional_number(all_queries.get("all_gold_pages_rate")),
                _render_optional_number(all_queries.get("mean_page_recall")),
                _render_optional_number(all_queries.get("selected_pages_cover_all_gold_trajectories_rate")),
                _render_optional_number(failed_queries.get("all_gold_pages_rate")),
                _render_optional_number(failed_queries.get("selected_pages_cover_all_gold_trajectories_rate")),
            )
        active_console.print(page_cutoff_table)

    if trajectory_cutoffs:
        trajectory_cutoff_table = Table(title="Trajectory Cutoff Diagnostics")
        trajectory_cutoff_table.add_column("cutoff")
        trajectory_cutoff_table.add_column("all_traj_all", justify="right")
        trajectory_cutoff_table.add_column("traj_recall_all", justify="right")
        trajectory_cutoff_table.add_column("refs_cover_all", justify="right")
        trajectory_cutoff_table.add_column("all_traj_failed", justify="right")
        trajectory_cutoff_table.add_column("refs_cover_failed", justify="right")
        for cutoff in _common_cutoff_keys(trajectory_cutoffs):
            row = dict(trajectory_cutoffs.get(cutoff, {}))
            all_queries = dict(row.get("all_queries", {}))
            failed_queries = dict(row.get("failed_queries", {}))
            trajectory_cutoff_table.add_row(
                cutoff,
                _render_optional_number(all_queries.get("all_gold_trajectories_rate")),
                _render_optional_number(all_queries.get("mean_gold_trajectory_recall")),
                _render_optional_number(all_queries.get("top_k_can_cover_all_gold_refs_rate")),
                _render_optional_number(failed_queries.get("all_gold_trajectories_rate")),
                _render_optional_number(failed_queries.get("top_k_can_cover_all_gold_refs_rate")),
            )
        active_console.print(trajectory_cutoff_table)

    if offline_parameter_diagnostics:
        offline_summary_table = Table(title="Offline Parameter Diagnostics")
        offline_summary_table.add_column("metric")
        offline_summary_table.add_column("value", justify="right")
        for key in [
            "diagnostic_mode",
            "query_count",
            "saved_page_rank_limit",
            "saved_trajectory_rank_limit",
        ]:
            offline_summary_table.add_row(key, _render_unknown(offline_parameter_diagnostics.get(key)))
        active_console.print(offline_summary_table)
    if offline_page_cutoffs:
        offline_page_table = Table(title="Offline Page Cutoff Diagnostics")
        offline_page_table.add_column("cutoff")
        offline_page_table.add_column("observed", justify="right")
        offline_page_table.add_column("all_pages_all", justify="right")
        offline_page_table.add_column("page_recall_all", justify="right")
        offline_page_table.add_column("page_traj_all", justify="right")
        offline_page_table.add_column("ref_cover_all", justify="right")
        offline_page_table.add_column("not_saved", justify="right")
        for cutoff in _common_cutoff_keys(offline_page_cutoffs):
            row = dict(offline_page_cutoffs.get(cutoff, {}))
            all_queries = dict(row.get("all_queries", {}))
            offline_page_table.add_row(
                cutoff,
                _render_unknown(all_queries.get("observed_query_count")),
                _render_optional_number(all_queries.get("all_gold_pages_rate")),
                _render_optional_number(all_queries.get("mean_gold_page_recall")),
                _render_optional_number(all_queries.get("selected_pages_cover_all_gold_trajectories_rate")),
                _render_optional_number(all_queries.get("page_universe_can_cover_all_gold_refs_rate")),
                _render_unknown(all_queries.get("not_observed_query_count")),
            )
        active_console.print(offline_page_table)
    if offline_trajectory_cutoffs:
        offline_trajectory_table = Table(title="Offline Trajectory Cutoff Diagnostics")
        offline_trajectory_table.add_column("cutoff")
        offline_trajectory_table.add_column("observed", justify="right")
        offline_trajectory_table.add_column("all_traj_all", justify="right")
        offline_trajectory_table.add_column("traj_recall_all", justify="right")
        offline_trajectory_table.add_column("refs_cover_all", justify="right")
        offline_trajectory_table.add_column("not_saved", justify="right")
        for cutoff in _common_cutoff_keys(offline_trajectory_cutoffs):
            row = dict(offline_trajectory_cutoffs.get(cutoff, {}))
            all_queries = dict(row.get("all_queries", {}))
            offline_trajectory_table.add_row(
                cutoff,
                _render_unknown(all_queries.get("observed_query_count")),
                _render_optional_number(all_queries.get("all_gold_trajectories_rate")),
                _render_optional_number(all_queries.get("mean_gold_trajectory_recall")),
                _render_optional_number(all_queries.get("top_k_can_cover_all_gold_refs_rate")),
                _render_unknown(all_queries.get("not_observed_query_count")),
            )
        active_console.print(offline_trajectory_table)

    if trajectory_length_diagnostics:
        all_lengths = dict(trajectory_length_diagnostics.get("all_trajectories", {}))
        all_gold = dict(trajectory_length_diagnostics.get("all_queries_gold_trajectories", {}))
        gold_lengths = dict(all_gold.get("gold_trajectory_length_stats", {}))
        gold_requirements = dict(
            dict(trajectory_length_diagnostics.get("all_queries_gold_snapshot_requirements", {})).get(
                "max_gold_snapshot_rank_required_stats", {}
            )
        )
        recommendations = dict(trajectory_length_diagnostics.get("recommendations", {}))
        length_table = Table(title="Trajectory Length Diagnostics")
        length_table.add_column("metric")
        length_table.add_column("value", justify="right")
        for key, value in [
            ("configured_m", trajectory_length_diagnostics.get("configured_m")),
            ("all_length_median", all_lengths.get("median")),
            ("all_length_p90", all_lengths.get("p90")),
            ("all_length_p95", all_lengths.get("p95")),
            ("all_length_max", all_lengths.get("max")),
            ("all_at_m_limit_rate", all_lengths.get("at_m_limit_rate")),
            ("gold_length_p95", gold_lengths.get("p95")),
            ("gold_required_rank_p95", gold_requirements.get("p95")),
            ("candidate_lower_m_at_95", recommendations.get("candidate_lower_m_at_95")),
            ("interpretation", recommendations.get("interpretation")),
        ]:
            length_table.add_row(key, _render_optional_number(value))
        active_console.print(length_table)

    if trajectory_drift_diagnostics:
        drift_table = Table(title="Trajectory Semantic Drift")
        drift_table.add_column("metric")
        drift_table.add_column("value", justify="right")
        head_tail_stats = dict(trajectory_drift_diagnostics.get("head_tail_cosine_stats") or {})
        adjacent_min_stats = dict(trajectory_drift_diagnostics.get("adjacent_min_cosine_stats") or {})
        summary_tail_stats = dict(trajectory_drift_diagnostics.get("summary_tail_cosine_stats") or {})
        for key, value in [
            ("trajectory_count", trajectory_drift_diagnostics.get("trajectory_count")),
            ("embedding_available_count", trajectory_drift_diagnostics.get("embedding_available_count")),
            ("missing_embedding_count", trajectory_drift_diagnostics.get("missing_embedding_count")),
            ("singleton_trajectory_rate", trajectory_drift_diagnostics.get("singleton_trajectory_rate")),
            ("head_tail_cosine_median", head_tail_stats.get("median")),
            ("head_tail_cosine_p10", head_tail_stats.get("p10")),
            ("adjacent_min_cosine_p10", adjacent_min_stats.get("p10")),
            ("summary_tail_cosine_median", summary_tail_stats.get("median")),
            ("possible_or_high_span_rate", trajectory_drift_diagnostics.get("possible_or_high_span_rate")),
        ]:
            drift_table.add_row(key, _render_optional_number(value))
        active_console.print(drift_table)

        drift_by_length = dict(trajectory_drift_diagnostics.get("drift_by_length_bucket") or {})
        if drift_by_length:
            length_drift_table = Table(title="Trajectory Drift by Length")
            length_drift_table.add_column("length")
            length_drift_table.add_column("count", justify="right")
            length_drift_table.add_column("head_tail_median", justify="right")
            length_drift_table.add_column("head_tail_p10", justify="right")
            length_drift_table.add_column("adjacent_min_p10", justify="right")
            length_drift_table.add_column("risk_rate", justify="right")
            for bucket in ["1", "2-3", "4-6", "7-10", "11-15", "16+"]:
                row = dict(drift_by_length.get(bucket) or {})
                if not row:
                    continue
                head_tail = dict(row.get("head_tail_cosine_stats") or {})
                adjacent_min = dict(row.get("adjacent_min_cosine_stats") or {})
                length_drift_table.add_row(
                    bucket,
                    _render_optional_number(row.get("trajectory_count")),
                    _render_optional_number(head_tail.get("median")),
                    _render_optional_number(head_tail.get("p10")),
                    _render_optional_number(adjacent_min.get("p10")),
                    _render_optional_number(row.get("possible_or_high_span_rate")),
                )
            active_console.print(length_drift_table)

        drift_by_verdict = dict(trajectory_drift_diagnostics.get("drift_by_query_verdict") or {})
        if drift_by_verdict:
            verdict_drift_table = Table(title="Gold Trajectory Drift by Outcome")
            verdict_drift_table.add_column("verdict")
            verdict_drift_table.add_column("queries", justify="right")
            verdict_drift_table.add_column("gold_min_head_tail_mean", justify="right")
            verdict_drift_table.add_column("risk_rate", justify="right")
            verdict_drift_table.add_column("possible_gold", justify="right")
            verdict_drift_table.add_column("high_span_gold", justify="right")
            for verdict in ["correct", "partial", "incorrect", "judge_error", "unknown"]:
                row = dict(drift_by_verdict.get(verdict) or {})
                if not row:
                    continue
                min_stats = dict(row.get("gold_head_tail_min_stats") or {})
                verdict_drift_table.add_row(
                    verdict,
                    _render_optional_number(row.get("query_count")),
                    _render_optional_number(min_stats.get("mean")),
                    _render_optional_number(row.get("trajectory_drift_risk_rate")),
                    _render_optional_number(row.get("gold_possible_drift_count")),
                    _render_optional_number(row.get("gold_high_span_count")),
                )
            active_console.print(verdict_drift_table)

    examples_by_reason = dict(report.get("examples_by_reason", {}))
    for reason in FAILURE_REASON_ORDER:
        examples = list(examples_by_reason.get(reason, []))
        if not examples:
            continue
        active_console.print(f"[bold]{reason}[/bold]")
        for index, example in enumerate(examples, start=1):
            extra_lines: list[str] = []
            verdict = str(example.get("judge_verdict") or "unknown").upper()
            judge_mode = _render_unknown(example.get("judge_mode"))
            judge_fallback = _render_unknown_bool(example.get("judge_structured_fallback_used"))
            judge_structured = _render_unknown_bool(example.get("judge_structured_success"))
            judge_error_type = example.get("judge_error_type")
            judge_bits = [
                f"score={_render_judge_score(example.get('judge_verdict'), example.get('judge_score'))}",
                f"mode={judge_mode}",
                f"structured={judge_structured}",
                f"fallback={judge_fallback}",
            ]
            if _has_display_text(judge_error_type):
                judge_bits.append(f"error={judge_error_type}")
            extra_lines.append(f"   Judge: {verdict} ({', '.join(judge_bits)})")
            if _has_display_text(example.get("judge_rationale")):
                extra_lines.append(f"   Judge explanation: {example.get('judge_rationale')}")
            elif _has_display_text(example.get("judge_error_type")) or _has_display_text(
                example.get("judge_error_message")
            ):
                error_type = _render_unknown(example.get("judge_error_type"))
                error_message = _render_unknown(example.get("judge_error_message"))
                extra_lines.append(f"   Judge error: {error_type}: {error_message}")
            if show_ranks:
                extra_lines.append(
                    "   Ranks: "
                    f"coarse={_render_unknown(example.get('gold_trajectory_rank'))}; "
                    f"fine={_render_unknown(example.get('gold_snapshot_rank'))}; "
                    f"expanded-only={_render_unknown_bool(example.get('gold_expanded_only_hit'))}; "
                    f"entity-linked={_render_unknown_bool(example.get('gold_entity_linked_hit'))}"
                )
                extra_lines.append(
                    "   Retrieval coverage: "
                    f"top-k gold trajectories="
                    f"{_render_count_pair(example.get('gold_trajectory_count_in_top_k'), example.get('gold_trajectory_count'))}; "
                    f"all gold refs covered={_render_unknown_bool(example.get('top_k_can_cover_all_gold_refs'))}; "
                    f"gold ref coverage={_render_unknown(example.get('top_k_gold_ref_coverage_rate'))}"
                )
                selection_pool_available = bool(example.get("trajectory_selection_pool_available"))
                extra_lines.append(
                    "   Selection pool: "
                    f"gold trajectories="
                    f"{_render_count_pair(example.get('gold_trajectory_count_in_selection_pool'), example.get('gold_trajectory_count'), available=selection_pool_available)}; "
                    f"final top-k={_render_count_pair(example.get('gold_trajectory_count_in_top_k'), example.get('gold_trajectory_count'))}; "
                    f"strategy={_render_unknown(example.get('trajectory_selection_strategy') or 'unknown')}"
                )
                extra_lines.append(
                    "   Page routing: "
                    f"gold pages in top-t="
                    f"{_render_count_pair(len(example.get('gold_pages_in_top_t', []) or []), len(example.get('gold_page_ids', []) or []))}; "
                    f"recall={_render_unknown(example.get('page_routing_recall'))}; "
                    f"selected pages cover all gold trajectories="
                    f"{_render_unknown_bool(example.get('all_gold_trajectories_in_selected_pages'))}"
                )
                extra_lines.append(
                    "   Text-only filter: "
                    f"eligible={_render_unknown_bool(example.get('text_only_eligible'))}; "
                    f"excluded={_render_unknown_bool(example.get('excluded_from_text_only'))}; "
                    f"type={_render_unknown(example.get('visual_dependency_type') or '-')}; "
                    f"reason={_render_unknown(example.get('text_only_exclusion_reason') or '-')}; "
                    f"missing_gold={_render_compact_json_list(list(example.get('text_only_missing_gold_items') or []))}; "
                    f"image_refs={_render_unknown(','.join(example.get('text_only_gold_evidence_image_refs') or []) or '-')}"
                )
                extra_lines.append(
                    "   Page granularity: "
                    f"selected singletons={_render_unknown(example.get('selected_singleton_page_count'))}; "
                    f"selected medium={_render_unknown(example.get('selected_medium_granularity_page_count'))}; "
                    f"histogram={_render_unknown(example.get('selected_page_trajectory_count_histogram') or {})}; "
                    f"singleton penalties={_render_unknown(example.get('singleton_page_penalty_applied'))}; "
                    f"low-quality penalties={_render_unknown(example.get('low_quality_singleton_penalty_applied'))}; "
                    f"medium bonuses={_render_unknown(example.get('medium_page_bonus_applied'))}; "
                    f"mode={_render_unknown(example.get('page_granularity_metadata_source') or '-')}"
                )
                extra_lines.append(
                    "   Page family routing: "
                    f"available={_render_unknown_bool(example.get('page_family_routing_available'))}; "
                    f"max_score={_render_unknown(example.get('page_family_match_score_max'))}; "
                    f"object_terms={_render_unknown(','.join(list(example.get('page_family_query_object_terms') or [])[:8]) or '-')}; "
                    f"overlap={_render_unknown(','.join(list(example.get('page_family_query_object_overlap_terms') or [])[:8]) or '-')}; "
                    f"mismatch_penalties={_render_unknown(example.get('page_family_mismatch_penalty_count'))}"
                )
                extra_lines.append(
                    "   Routing text: "
                    f"cleaned={_render_unknown_bool(example.get('routing_text_cleaned'))}; "
                    f"marker_leaks={_render_unknown(example.get('routing_text_internal_marker_count'))}"
                )
                extra_lines.append(
                    "   Memory preservation: "
                    f"claims={_render_unknown_bool(example.get('gold_answer_fully_present_in_claims'))}; "
                    f"summaries={_render_unknown_bool(example.get('gold_answer_fully_present_in_trajectory_summaries'))}; "
                    f"wiki={_render_unknown_bool(example.get('gold_answer_fully_present_in_wiki_pages'))}"
                )
                extra_lines.append(
                    "   Historical evidence: "
                    f"card has gold={_render_unknown_bool(example.get('gold_item_present_in_historical_card'))}; "
                    f"latest summary has gold={_render_unknown_bool(example.get('gold_item_present_in_latest_summary'))}; "
                    f"wiki after card has gold={_render_unknown_bool(example.get('gold_item_present_in_wiki_after_historical_card'))}; "
                    f"index-only gold trajectories={_render_unknown(example.get('index_only_gold_trajectory_count') or 0)}; "
                    f"drift suspected={_render_unknown_bool(example.get('trajectory_drift_suspected'))}"
                )
                extra_lines.append(
                    "   Metadata terms: "
                    f"keywords_policy={_render_unknown(example.get('metadata_terms_keyword_policy') or '-')}; "
                    f"historical_policy={_render_unknown(example.get('metadata_terms_historical_policy') or '-')}; "
                    f"internal_leaks={_render_unknown(example.get('metadata_terms_internal_leak_count'))}; "
                    f"summary_fallback={_render_unknown(example.get('metadata_terms_summary_fallback_count'))}"
                )
                extra_lines.append(
                    "   Source surface preservation: "
                    f"raw={_render_unknown_bool(example.get('gold_surface_present_in_raw'))}; "
                    f"claims={_render_unknown_bool(example.get('gold_surface_preserved_in_claims'))}; "
                    f"exact_terms={_render_unknown_bool(example.get('gold_surface_preserved_in_exact_terms'))}; "
                    f"card={_render_unknown_bool(example.get('gold_surface_preserved_in_historical_card'))}; "
                    f"wiki={_render_unknown_bool(example.get('gold_surface_preserved_in_wiki'))}; "
                    f"misses={_render_unknown(example.get('source_surface_term_miss_count'))}"
                )
                extra_lines.append(
                    "   Wiki coverage: "
                    f"non-index coverage={_render_unknown(example.get('non_index_trajectory_coverage_rate'))}; "
                    f"index-only trajectories={_render_unknown(example.get('index_only_trajectory_count'))}; "
                    f"rescue pages={_render_unknown(example.get('wiki_rescue_page_count'))}; "
                    f"rescue trajectories={_render_unknown(example.get('wiki_rescue_trajectory_count'))}; "
                    f"selected page universe={_render_unknown(example.get('selected_page_universe_size'))}"
                )
                extra_lines.append(
                    "   Wiki fragmentation: "
                    f"singleton rate={_render_unknown(example.get('wiki_singleton_non_index_page_rate'))}; "
                    f"non-index pages={_render_unknown(example.get('wiki_non_index_page_count'))}; "
                    f"mean traj/page={_render_unknown(example.get('wiki_mean_trajectories_per_non_index_page'))}; "
                    f"rescue singletons={_render_unknown(example.get('wiki_post_plan_rescue_singleton_count'))}; "
                    f"facet singletons={_render_unknown(example.get('wiki_entity_facet_singleton_count'))}; "
                    f"mode={_render_unknown(example.get('wiki_fragmentation_metadata_source') or '-')}"
                )
                extra_lines.append(
                    "   Wiki singleton policy: "
                    f"allowed_specific={_render_unknown(example.get('wiki_allowed_specific_singleton_count'))}; "
                    f"low_quality={_render_unknown(example.get('wiki_low_quality_singleton_count'))}; "
                    f"merged={_render_unknown(example.get('wiki_low_quality_singleton_merged_count'))}"
                )
                extra_lines.append(
                    "   Overwide pages: "
                    f"split={_render_unknown(example.get('wiki_overwide_page_split_count'))}; "
                    f"overwide_before_split={_render_unknown(example.get('wiki_overwide_non_index_page_count'))}; "
                    f"max_non_index_trajectory_count={_render_unknown(example.get('wiki_max_non_index_trajectory_count'))}"
                )
                extra_lines.append(
                    "   Index fallback: "
                    f"used={_render_unknown_bool(example.get('index_fallback_used'))}; "
                    f"reason={_render_unknown(example.get('index_fallback_reason') or '-')}; "
                    f"before={_render_unknown(example.get('selected_page_universe_size_before_index_fallback'))}; "
                    f"after={_render_unknown(example.get('selected_page_universe_size_after_index_fallback'))}; "
                    f"added={len(example.get('index_fallback_added_trajectory_ids') or [])}"
                )
                extra_lines.append(
                    "   Broad entity cap: "
                    f"used={_render_unknown_bool(example.get('broad_entity_candidate_cap_used'))}; "
                    f"pages={len(example.get('broad_entity_page_ids') or [])}; "
                    f"profile_pages={len(example.get('broad_entity_profile_page_ids') or [])}; "
                    f"fine_pages={len(example.get('fine_grained_entity_page_ids') or [])}; "
                    f"profile_suppressed={_render_unknown_bool(example.get('broad_entity_profile_suppressed'))}; "
                    f"before={_render_unknown(example.get('selected_page_universe_size_before_broad_cap'))}; "
                    f"after={_render_unknown(example.get('selected_page_universe_size_after_broad_cap'))}; "
                    f"added={len(example.get('broad_entity_added_trajectory_ids') or [])}"
                )
                extra_lines.append(
                    "   Force recall: "
                    f"forced gold refs={len(example.get('gold_refs_in_forced_memory', []) or [])}; "
                    f"low-salience gold refs={len(example.get('gold_refs_in_low_salience_memory', []) or [])}; "
                    f"forced snapshots={len(example.get('gold_forced_snapshot_ids', []) or [])}"
                )
                extra_lines.append(
                    "   Stage failures: "
                    f"page route={_render_unknown_bool(example.get('page_routing_failed'))}; "
                    f"selected pages={_render_unknown_bool(example.get('selected_pages_failed_to_cover_all_gold_trajectories'))}; "
                    f"trajectory top-k={_render_unknown_bool(example.get('trajectory_retrieval_failed_after_page_routing'))}; "
                    f"all gold refs grounded={_render_unknown_bool(example.get('all_gold_refs_grounded'))}"
                )
                extra_lines.append(
                    "   Compaction: "
                    f"pre-compaction cover={_render_unknown_bool(example.get('pre_compaction_expansion_could_cover_all_gold_refs'))}; "
                    f"snapshot miss={_render_unknown_bool(example.get('snapshot_compaction_miss'))}; "
                    f"source miss={_render_unknown_bool(example.get('source_compaction_miss'))}"
                )
                extra_lines.append(
                    "   Cutoff minimums: "
                    f"pages for gold={_render_unknown(example.get('min_t_pages_to_cover_all_gold_pages'))}; "
                    f"pages for trajectories={_render_unknown(example.get('min_t_pages_to_cover_all_gold_trajectories_via_pages'))}; "
                    f"k for gold trajectories={_render_unknown(example.get('min_k_to_cover_all_gold_trajectories'))}; "
                    f"k for gold refs={_render_unknown(example.get('min_k_to_cover_all_gold_refs'))}"
                )
                extra_lines.append(
                    "   Offline cutoffs: "
                    f"min_t_saved={_render_unknown(example.get('offline_min_t_saved_to_cover_all_gold_trajectories'))}; "
                    f"min_k_saved={_render_unknown(example.get('offline_min_k_saved_to_cover_all_gold_refs'))}; "
                    f"saved_page_rank_limit={_render_unknown(example.get('offline_saved_page_rank_limit'))}; "
                    f"saved_traj_rank_limit={_render_unknown(example.get('offline_saved_trajectory_rank_limit'))}; "
                    f"page_mode={_render_unknown(example.get('offline_page_diagnostic_mode'))}; "
                    f"traj_mode={_render_unknown(example.get('offline_trajectory_diagnostic_mode'))}"
                )
                extra_lines.append(
                    "   Trajectory length: "
                    f"gold max={_render_unknown(example.get('gold_trajectory_length_max'))}; "
                    f"at m limit="
                    f"{_render_count_pair(example.get('gold_trajectory_at_m_limit_count'), len(example.get('gold_trajectory_lengths', {}) or {}))}"
                )
                extra_lines.append(
                    "   Trajectory drift: "
                    f"gold_min_head_tail={_render_unknown(example.get('gold_trajectory_head_tail_cosine_min'))}; "
                    f"gold_mean_head_tail={_render_unknown(example.get('gold_trajectory_head_tail_cosine_mean'))}; "
                    f"buckets={_render_compact_json_list([example.get('gold_trajectory_drift_buckets') or {}])}; "
                    f"low_pairs={_render_unknown(example.get('gold_trajectory_low_similarity_update_pair_count'))}; "
                    f"risk={_render_unknown_bool(example.get('trajectory_drift_risk_observed'))}"
                )
                extra_lines.append(
                    "   Reflection retry: "
                    f"used={_render_unknown_bool(example.get('retrieval_reflection_used'))}; "
                    f"stage={_render_unknown(example.get('retrieval_reflection_stage') or 'none')}; "
                    f"raw rescue={_render_unknown_bool(example.get('raw_rescue_attempted'))}; "
                    f"reason={_render_unknown(','.join(example.get('raw_rescue_trigger_reasons') or []) or example.get('raw_rescue_skipped_reason') or '-')}; "
                    f"uncovered={_render_unknown(','.join(example.get('reflection_uncovered_terms') or []) or '-')}; "
                    f"coverage={_render_unknown(example.get('reflection_term_coverage_rate'))}; "
                    f"raw hits={_render_unknown(example.get('raw_rescue_hit_count') or 0)}; "
                    f"answer changed={_render_unknown_bool(example.get('reflection_answer_changed'))}"
                )
                if _should_render_temporal_grounding(example):
                    extra_lines.append(
                        "   Temporal grounding: "
                        f"raw_dates={_render_unknown_bool(example.get('temporal_anchor_available_in_raw'))}; "
                        f"prompt_dates={_render_unknown_bool(example.get('temporal_anchor_visible_in_prompt'))}; "
                        f"relative_terms={_render_unknown(','.join(example.get('relative_time_terms_present') or []) or '-')}; "
                        f"anchor_hints={_render_unknown(example.get('temporal_anchor_hint_count'))}; "
                        f"missing_in_context={_render_unknown_bool(example.get('temporal_anchor_missing_in_answer_context'))}"
                    )
                extra_lines.append(
                    "   Query shape: "
                    + json.dumps(example.get("query_shape", {}), ensure_ascii=False)
                )
                extra_lines.append(
                    "   Coverage trigger: "
                    f"triggered={_render_unknown_bool(example.get('coverage_triggered'))}; "
                    f"selection_pool_size={_render_unknown(example.get('trajectory_selection_pool_size'))}; "
                    f"lost_before_selection={len(list(example.get('missing_gold_trajectory_ids_from_selection_pool') or []))}"
                )
                extra_lines.append(
                    "   Direct trajectory ablation: "
                    f"recall@k="
                    f"{_render_count_pair(example.get('direct_gold_trajectory_count_in_top_k'), example.get('gold_trajectory_count'))}; "
                    f"ref_coverage={_render_unknown(example.get('direct_top_k_gold_ref_coverage_rate'))}; "
                    f"delta={_render_unknown(example.get('direct_vs_page_routed_recall_delta'))}; "
                    f"bottleneck={_render_unknown(example.get('direct_trajectory_bottleneck') or 'unknown')}; "
                    f"mode={_render_unknown(example.get('direct_trajectory_scoring_mode') or '-')}"
                )
                extra_lines.append(
                    "   Direct no-routing: "
                    f"recall@k="
                    f"{_render_count_pair(example.get('direct_gold_trajectory_count_in_top_k'), example.get('gold_trajectory_count'))}; "
                    f"overlap={_render_unknown(example.get('routed_vs_direct_top_k_overlap_rate'))}; "
                    f"not_in_page_universe={_render_unknown(example.get('direct_top_k_not_in_selected_page_universe_count'))}; "
                    f"est_tokens={_render_unknown(example.get('direct_top_k_estimated_context_token_count'))}; "
                    f"saved_rank_limit={_render_unknown(example.get('direct_trajectory_rank_limit'))}; "
                    f"bottleneck={_render_unknown(example.get('direct_trajectory_bottleneck') or 'unknown')}"
                )
                extra_lines.append(
                    "   Family ranking: "
                    f"family={_render_unknown(dict(example.get('query_shape') or {}).get('item_family') or '-')}; "
                    f"query_object_terms={_render_unknown(','.join(list(example.get('trajectory_query_object_terms') or [])[:8]) or '-')}; "
                    f"overlap_terms={_render_unknown(','.join(list(example.get('trajectory_family_query_overlap_terms') or [])[:8]) or '-')}; "
                    f"gold family in pool="
                    f"{_render_count_pair(example.get('gold_family_matched_in_selection_pool_count'), example.get('gold_trajectory_count'))}; "
                    f"gold family in top-k="
                    f"{_render_count_pair(example.get('gold_family_matched_in_top_k_count'), example.get('gold_trajectory_count'))}; "
                    f"selected family matched={_render_unknown(example.get('trajectory_selected_family_match_count'))}"
                )
                extra_lines.append(
                    "   Source event match: "
                    f"terms={_render_unknown(','.join(list(example.get('trajectory_source_event_matched_terms') or [])[:8]) or '-')}; "
                    f"refs={_render_unknown(','.join(list(example.get('trajectory_source_event_matched_refs') or [])[:8]) or '-')}; "
                    f"gold event in pool="
                    f"{_render_count_pair(example.get('gold_source_event_matched_in_selection_pool_count'), example.get('gold_trajectory_count'))}; "
                    f"gold event in top-k="
                    f"{_render_count_pair(example.get('gold_source_event_matched_in_top_k_count'), example.get('gold_trajectory_count'))}; "
                    f"selected matches={_render_unknown(example.get('trajectory_selected_source_event_match_count'))}"
                )
                extra_lines.append(
                    "   Answer synthesis: "
                    f"used={_render_unknown_bool(example.get('answer_synthesis_used'))}; "
                    f"mode={_render_unknown(example.get('answer_synthesis_mode') or 'none')}; "
                    f"can_answer={_render_unknown_bool(example.get('answer_synthesis_can_answer'))}; "
                    f"type={_render_unknown(example.get('answer_synthesis_answer_type') or '-')}; "
                    f"freeform={_render_unknown_bool(example.get('answer_freeform_used'))}; "
                    f"observed_type={_render_unknown(example.get('answer_observed_type') or '-')}; "
                    f"type_match={_render_unknown(example.get('answer_type_match'))}; "
                    f"verifier={_render_unknown_bool(example.get('answer_type_verification_used'))}/"
                    f"{_render_unknown_bool(example.get('answer_type_verification_success'))}; "
                    f"counted={_render_unknown(example.get('answer_synthesis_counted_event_count') or 0)}; "
                    f"excluded={_render_unknown(example.get('answer_synthesis_excluded_event_count') or 0)}; "
                    f"typed_retry={_render_unknown_bool(example.get('answer_synthesis_typed_retry_used'))}/"
                    f"{_render_unknown_bool(example.get('answer_synthesis_typed_retry_success'))}; "
                    f"expected={_render_unknown(example.get('answer_synthesis_typed_retry_expected_type') or '-')}; "
                    f"type_recovered={_render_unknown_bool(example.get('answer_synthesis_type_mismatch_recovered'))}; "
                    f"expected_text_valid={_render_unknown(example.get('answer_synthesis_expected_type_text_valid'))}; "
                    f"retry_json_normalized={_render_unknown_bool(example.get('answer_synthesis_typed_retry_text_json_normalized'))}; "
                    f"safe_abstain={_render_unknown_bool(example.get('answer_synthesis_safe_abstain_used'))}"
                )
                excluded_events = list(example.get("answer_synthesis_excluded_events") or [])
                if excluded_events:
                    excluded_summary = "; ".join(
                        f"{dict(event).get('event_id') or '?'}:{dict(event).get('reason') or 'unknown'}"
                        for event in excluded_events[:5]
                    )
                    extra_lines.append(f"   Answer synthesis excluded events: {excluded_summary}")
                validation_excluded = list(example.get("count_validation_excluded_events") or [])
                invalid_refs = list(example.get("invalid_supporting_refs") or [])
                invalid_family_refs = list(example.get("answer_synthesis_invalid_family_refs") or [])
                alias_hits = list(example.get("source_family_validation_alias_hits") or [])
                ref_acceptance = list(example.get("count_validation_ref_acceptance_reasons") or [])
                ref_rejection = list(example.get("count_validation_ref_rejection_reasons") or [])
                source_derived_refs = list(example.get("count_validation_source_derived_candidate_refs") or [])
                bridge_used = list(example.get("bridge_facts_used") or [])
                bridge_conflicted = list(example.get("bridge_facts_conflicted") or [])
                if (
                    validation_excluded
                    or invalid_refs
                    or invalid_family_refs
                    or alias_hits
                    or ref_acceptance
                    or ref_rejection
                    or bridge_used
                    or bridge_conflicted
                    or example.get("answer_synthesis_question_type_mismatch")
                    or example.get("answer_synthesis_type_mismatch_recovered")
                    or example.get("answer_synthesis_internal_abstain_reason_suppressed")
                    or example.get("answer_synthesis_expected_type_text_valid") is False
                    or source_derived_refs
                ):
                    extra_lines.append(
                        "   Answer synthesis validation: "
                        f"invalid_refs={_render_unknown(','.join(invalid_refs) or '-')}; "
                        f"invalid_family_refs={_render_unknown(','.join(invalid_family_refs) or '-')}; "
                        f"question_type_mismatch={_render_unknown_bool(example.get('answer_synthesis_question_type_mismatch'))}; "
                        f"count_style_text_mismatch={_render_unknown_bool(example.get('answer_synthesis_count_style_text_mismatch'))}; "
                        f"normalized_type={_render_unknown(example.get('answer_synthesis_normalized_answer_type') or '-')}; "
                        f"expected_text_valid={_render_unknown(example.get('answer_synthesis_expected_type_text_valid'))}; "
                        f"expected_text_rejection={_render_unknown(example.get('answer_synthesis_expected_type_text_rejection_reason') or '-')}; "
                        f"repair_reason={_render_unknown(example.get('answer_synthesis_repair_reason') or '-')}; "
                        f"source_family_alias_hits={len(alias_hits)}; "
                        f"count_refs_accepted={len(ref_acceptance)}; "
                        f"count_refs_rejected={len(ref_rejection)}; "
                        f"count_excluded={len(validation_excluded)}; "
                        f"bridges={len(bridge_used)}; "
                        f"bridge_conflicts={len(bridge_conflicted)}; "
                        f"source_derived_count_candidates={_render_unknown(example.get('count_validation_source_derived_candidate_count') or 0)}; "
                        f"source_derived_refs={_render_unknown(','.join(source_derived_refs) or '-')}; "
                        f"pronoun_caption_refs={_render_unknown(','.join(example.get('count_validation_source_derived_pronoun_caption_refs') or []) or '-')}; "
                        f"passive_rejected={_render_unknown(','.join(example.get('count_validation_source_derived_passive_rejected_refs') or []) or '-')}; "
                        f"retry_policy={_render_unknown(example.get('answer_synthesis_typed_retry_final_policy') or '-')}"
                    )
                if example.get("count_validation_llm_used") or example.get("count_validation_llm_skipped_reason"):
                    extra_lines.append(
                        "   Count validator: "
                        f"llm={_render_unknown_bool(example.get('count_validation_llm_used'))}/"
                        f"{_render_unknown_bool(example.get('count_validation_llm_success'))}; "
                        f"scope={_render_unknown(example.get('count_validation_llm_scope') or '-')}; "
                        f"changed={_render_unknown(example.get('count_validation_llm_changed_count') or 0)}; "
                        f"confidence={_render_unknown(example.get('count_validation_llm_confidence') or '-')}; "
                        f"trigger={_render_unknown(','.join(example.get('count_validation_llm_trigger_reasons') or []) or '-')}; "
                        f"skipped={_render_unknown(example.get('count_validation_llm_skipped_reason') or '-')}"
                    )
                if example.get("answer_temporal_alignment_checked"):
                    extra_lines.append(
                        "   Temporal alignment: "
                        f"valid={_render_unknown(example.get('answer_temporal_alignment_valid'))}; "
                        f"selected_ref={_render_unknown(example.get('answer_temporal_selected_source_ref') or '-')}; "
                        f"selected_date={_render_unknown(example.get('answer_temporal_selected_date') or '-')}; "
                        f"selected_answer={_render_unknown(example.get('answer_temporal_selected_answer_text') or '-')}; "
                        f"kind={_render_unknown(example.get('answer_temporal_selected_resolution_kind') or '-')}; "
                        f"granularity={_render_unknown(example.get('answer_temporal_selected_resolution_granularity') or '-')}; "
                        f"relative={_render_unknown(example.get('answer_temporal_selected_relative_term') or '-')}; "
                        f"confidence={_render_unknown(example.get('answer_temporal_selected_confidence') or '-')}; "
                        f"score={_render_unknown(example.get('answer_temporal_candidate_score') if example.get('answer_temporal_candidate_score') is not None else '-')}; "
                        f"matched={_render_compact_json_list(list(example.get('answer_temporal_candidate_match_terms') or []))}; "
                        f"relevant={_render_unknown(example.get('answer_temporal_relevant_candidate_count') or 0)}; "
                        f"low_suppressed={_render_unknown(example.get('answer_temporal_low_confidence_candidate_count') or 0)}; "
                        f"candidates={_render_unknown(len(list(example.get('answer_temporal_candidate_dates') or [])))}; "
                        f"repair={_render_unknown_bool(example.get('answer_temporal_repair_used'))}/"
                        f"{_render_unknown_bool(example.get('answer_temporal_repair_success'))}; "
                        f"reason={_render_unknown(example.get('answer_temporal_alignment_rejection_reason') or '-')}"
                    )
                if int(example.get("speaker_grounding_suspect_count") or 0):
                    extra_lines.append(
                        "   Speaker grounding: "
                        f"suspect_claims_suppressed={int(example.get('speaker_grounding_suspect_count') or 0)}"
                    )
                extra_lines.append(
                    "   LLM usage: "
                    f"answer calls={_render_unknown(example.get('llm_answer_calls'))}; "
                    f"judge calls={_render_unknown(example.get('llm_judge_calls'))}; "
                    f"semantic calls={_render_unknown(example.get('llm_semantic_calls'))}; "
                    f"fallbacks={_render_unknown(example.get('llm_fallback_count'))}; "
                    f"memory-amortized repairs={_render_unknown(example.get('llm_repair_count'))}"
                )
                fr_summary = dict(example.get("fallback_repair_summary") or {})
                fr_flags = dict(example.get("fallback_repair_quality_flags") or {})
                if fr_summary:
                    extra_lines.append(
                        "   Fallback/repair: "
                        f"events={_render_unknown(fr_summary.get('event_count'))}; "
                        f"risk_events={_render_unknown(fr_flags.get('quality_risk_event_count'))}; "
                        f"risk_weighted={_render_unknown(fr_flags.get('quality_risk_weighted_event_count'))}; "
                        f"deterministic={_render_unknown_bool(fr_flags.get('has_deterministic_fallback'))}; "
                        f"repair_discarded={_render_unknown_bool(fr_flags.get('has_discarded_repair'))}"
                    )
                extra_lines.append(
                    "   Judge leniency: "
                    f"candidate={_render_unknown_bool(example.get('judge_leniency_candidate'))}; "
                    f"gold covered={_render_unknown_bool(example.get('gold_all_items_covered_by_answer'))}; "
                    f"extras={_render_unknown_bool(example.get('answer_has_extra_items'))}; "
                    f"count lower-bound={_render_unknown_bool(example.get('count_answer_is_evidence_limited_lower_bound'))}; "
                    f"reason={_render_unknown(','.join(example.get('answer_count_lower_bound_reasons') or []) or example.get('answer_count_naturalized_lower_bound_reason') or '-')}"
                )
                extra_lines.append(
                    "   Semantic F1: "
                    f"exact={len(example.get('canonical_exact_overlap_items') or [])}; "
                    f"soft={len(example.get('canonical_soft_overlap_items') or [])}; "
                    f"unmatched_gold={_render_compact_json_list(list(example.get('canonical_unmatched_reference_items') or []))}"
                )
                extra_lines.append(
                    "   Judge override: "
                    f"used={_render_unknown_bool(example.get('judge_semantic_override_used'))}; "
                    f"reason={_render_unknown(example.get('judge_semantic_override_reason') or '-')}; "
                    f"over_strict_suspected={_render_unknown_bool(example.get('judge_over_strict_suspected'))}"
                )
                extra_lines.append(
                    "   Answer post-check: "
                    f"used={_render_unknown_bool(example.get('answer_postcheck_used'))}; "
                    f"issue={_render_unknown(example.get('answer_postcheck_issue') or 'none')}"
                )
                extra_lines.append(
                    "   Repair arbitration: "
                    f"triggered={_render_unknown_bool(example.get('answer_repair_arbitration_triggered'))}; "
                    f"decision={_render_unknown(example.get('answer_repair_arbitration_decision') or '-')}; "
                    f"violation={_render_unknown(example.get('answer_repair_arbitration_violation') or '-')}; "
                    f"action={_render_unknown(example.get('answer_repair_arbitration_action') or '-')}"
                )
                extra_lines.append(
                    "   Answer specificity: "
                    f"preference_like={_render_unknown_bool(example.get('query_shape_preference_like'))}; "
                    f"overgeneric={_render_unknown_bool(example.get('answer_overgeneric_item_detected'))}; "
                    f"extras={_render_compact_json_list(list(example.get('answer_scope_mismatched_extra_items') or []))}; "
                    f"repair={_render_unknown_bool(example.get('answer_specific_item_repair_used'))}"
                )
                extra_lines.append(
                    "   Bridge finalization: "
                    f"alias={_render_unknown(example.get('bridge_finalization_alias') or '-')}; "
                    f"target={_render_unknown(example.get('bridge_finalization_target') or '-')}; "
                    f"action={_render_unknown(example.get('bridge_finalization_action') or '-')}; "
                    f"conflict={_render_unknown_bool(example.get('bridge_finalization_conflicted'))}; "
                    f"repair={_render_unknown_bool(example.get('answer_bridge_repair_used'))}"
                )
                extra_lines.append(
                    "   List coverage: "
                    f"supported={len(list(example.get('answer_supported_list_items') or []))}; "
                    f"missing={_render_compact_json_list(list(example.get('answer_missing_supported_list_items') or []))}; "
                    f"abstain_with_support={_render_unknown_bool(example.get('answer_abstain_despite_supported_items'))}; "
                    f"repair={_render_unknown_bool(example.get('answer_list_coverage_repair_used'))}; "
                    f"skipped={_render_unknown_bool(example.get('answer_list_coverage_skipped'))}; "
                    f"skip_reason={_render_unknown(example.get('answer_list_coverage_skip_reason') or '-')}; "
                    f"blocked_by_type={_render_unknown_bool(example.get('answer_list_repair_blocked_by_expected_type'))}"
                )
                extra_lines.append(
                    "   List scope: "
                    f"kind={_render_unknown(example.get('answer_list_scope_kind') or '-')}; "
                    f"required={_render_compact_json_list(list(example.get('answer_required_item_candidates') or []))}; "
                    f"optional={_render_compact_json_list(list(example.get('answer_optional_surface_values') or []))}; "
                    f"rejected={_render_compact_json_list(list(example.get('answer_scope_rejected_items') or []))}"
                )
                extra_lines.append(
                    "   Event canonicalization: "
                    f"canonical={_render_compact_json_list(list(example.get('event_canonical_alias_items') or []))}; "
                    f"semantic_aliases={_render_compact_json_list(list(example.get('semantic_alias_match_pairs') or []))}"
                )
                extra_lines.append(
                    "   Supported list items: "
                    f"required={_render_compact_json_list(list(example.get('answer_supported_required_items') or []))}; "
                    f"missing_before_repair={_render_compact_json_list(list(example.get('answer_required_items_missing_before_repair') or []))}; "
                    f"missing_after_repair={_render_compact_json_list(list(example.get('answer_repair_missing_required_items_after_repair') or []))}"
                )
                extra_lines.append(
                    "   Answer repair validation: "
                    f"dropped_supported={_render_compact_json_list(list(example.get('answer_repair_dropped_supported_items') or []))}; "
                    f"extras_removed={_render_compact_json_list(list(example.get('answer_repair_removed_scope_mismatched_items') or []))}; "
                    f"action={_render_unknown(example.get('answer_repair_post_validation_action') or '-')}; "
                    f"preserved_initial={_render_unknown_bool(example.get('answer_repair_preserved_initial_answer'))}"
                )
                extra_lines.append(
                    "   Gold snapshot requirement: "
                    f"max rank={_render_unknown(example.get('max_gold_snapshot_rank_required'))}; "
                    f"max version={_render_unknown(example.get('max_gold_snapshot_version_required'))}"
                )
            if show_facets:
                extra_lines.append(
                    "   Diagnostic flags: "
                    + (", ".join(example.get("diagnostic_flags", [])) or "none")
                )
                extra_lines.append(
                    "   Gold claim facets: "
                    + _render_compact_json_list(list(example.get("gold_claim_facets", []) or []))
                )
            active_console.print(
                f"{index}. sample={example['sample_id']} query={example['query_task_id']}\n"
                f"   Question: {example['question']}\n"
                f"   Gold answer: {_render_unknown(example.get('gold_answer'))}\n"
                f"   Answer: {_render_unknown(example.get('answer_text'))}"
                + ("\n" + "\n".join(extra_lines) if extra_lines else "")
            )


def print_incomplete_run_diagnostics(report: dict[str, Any], console: Console | None = None) -> None:
    active_console = console or Console()
    failure = dict(report.get("run_failed") or {})
    paths = dict(report.get("paths") or {})
    active_console.print(
        "[bold red]Incomplete Run Diagnostics[/bold red] "
        f"run_id={failure.get('run_id') or '-'} path={report.get('run_dir')}"
    )

    table = Table(title="Failure")
    table.add_column("field")
    table.add_column("value")
    for key in [
        "stage",
        "worker_id",
        "sample_id",
        "query_task_id",
        "error_type",
        "error_message",
        "database_path",
        "worker_database_root",
    ]:
        value = failure.get(key)
        table.add_row(key, "-" if value is None else str(value))
    active_console.print(table)

    shard_paths = list(failure.get("failed_shard_paths") or [])
    if shard_paths:
        shard_table = Table(title="Preserved Failed Shards")
        shard_table.add_column("path")
        for path in shard_paths[:20]:
            shard_table.add_row(str(path))
        active_console.print(shard_table)

    counts = dict(failure.get("database_table_counts") or {})
    table_counts = dict(counts.get("tables") or {})
    if table_counts:
        counts_table = Table(title="Main DB Table Counts")
        counts_table.add_column("table")
        counts_table.add_column("rows", justify="right")
        for table_name, count in sorted(table_counts.items()):
            counts_table.add_row(str(table_name), str(count))
        active_console.print(counts_table)

    events = list(report.get("recent_events") or failure.get("recent_events") or [])
    if events:
        event_table = Table(title="Recent Lifecycle Events")
        event_table.add_column("time")
        event_table.add_column("event")
        event_table.add_column("stage")
        event_table.add_column("sample")
        event_table.add_column("query")
        event_table.add_column("error")
        for event in events[-12:]:
            event_table.add_row(
                str(event.get("timestamp") or "-"),
                str(event.get("event_type") or "-"),
                str(event.get("stage") or "-"),
                str(event.get("sample_id") or "-"),
                str(event.get("query_task_id") or "-"),
                str(event.get("error_type") or event.get("error_message") or "-"),
            )
        active_console.print(event_table)

    provider_failures = list(failure.get("recent_provider_failures") or [])
    if provider_failures:
        provider_table = Table(title="Recent Provider Failures")
        provider_table.add_column("role")
        provider_table.add_column("task")
        provider_table.add_column("kind")
        provider_table.add_column("error")
        for record in provider_failures[-10:]:
            metadata = dict(record.get("metadata") or {})
            provider_table.add_row(
                str(record.get("role") or metadata.get("role") or "-"),
                str(record.get("task") or metadata.get("task") or "-"),
                str(record.get("provider_call_kind") or metadata.get("provider_call_kind") or "-"),
                str(metadata.get("error_type") or metadata.get("error_message") or "-"),
            )
        active_console.print(provider_table)

    active_console.print(
        f"[dim]Artifacts:[/dim] run_failed={paths.get('run_failed') or '-'} "
        f"events={paths.get('events') or '-'} database={paths.get('database') or '-'}"
    )


def print_locomo_failure_diff(
    diff_report: dict[str, Any], console: Console | None = None
) -> None:
    active_console = console or Console()
    before_meta = dict(diff_report.get("before_run_meta", {}))
    after_meta = dict(diff_report.get("after_run_meta", {}))
    totals = dict(diff_report.get("totals", {}))

    active_console.print(
        "[bold]LOCOMO Failure Attribution Diff[/bold] "
        f"before={before_meta.get('run_id')} after={after_meta.get('run_id')}"
    )
    active_console.print(
        f"failed_queries={totals.get('before_failed_queries')} -> {totals.get('after_failed_queries')} "
        f"delta={totals.get('failed_queries_delta')} "
        f"failure_rate={float(totals.get('before_failure_rate', 0.0)):.4f} -> "
        f"{float(totals.get('after_failure_rate', 0.0)):.4f}"
    )

    delta_table = Table(title="Outcome / Reason Delta")
    delta_table.add_column("bucket")
    delta_table.add_column("delta", justify="right")
    for key, value in dict(diff_report.get("outcome_counts_delta", {})).items():
        delta_table.add_row(key, str(int(value)))
    active_console.print(delta_table)

    transition_table = Table(title="Reason Transition Matrix")
    transition_table.add_column("before_reason")
    transition_table.add_column("after_reason")
    transition_table.add_column("count", justify="right")
    for row in list(diff_report.get("reason_transition_matrix", []))[:15]:
        transition_table.add_row(
            str(row["before_reason"]),
            str(row["after_reason"]),
            str(int(row["count"])),
        )
    active_console.print(transition_table)

    rank_delta = dict(diff_report.get("rank_diagnostics_delta", {}))
    facet_delta = dict(diff_report.get("facet_diagnostics_delta", {}))
    fragmentation_delta = dict(diff_report.get("fragmentation_diagnostics_delta", {}))
    page_routing_delta = dict(diff_report.get("page_routing_diagnostics_delta", {}))
    memory_preservation_delta = dict(diff_report.get("memory_preservation_diagnostics_delta", {}))
    stage_delta = dict(diff_report.get("stage_diagnostics_delta", {}))
    compaction_delta = dict(diff_report.get("compaction_diagnostics_delta", {}))
    multi_hop_delta = dict(diff_report.get("multi_hop_top_k_diagnostics_delta", {}))
    judge_delta = dict(diff_report.get("judge_diagnostics_delta", {}))
    llm_delta = dict(diff_report.get("llm_call_diagnostics_delta", {}))
    fallback_repair_delta = dict(diff_report.get("fallback_repair_diagnostics_delta", {}))
    cutoff_delta = dict(diff_report.get("cutoff_diagnostics_delta", {}))
    trajectory_length_delta = dict(diff_report.get("trajectory_length_diagnostics_delta", {}))
    diagnostics_table = Table(title="Diagnostics Delta")
    diagnostics_table.add_column("metric")
    diagnostics_table.add_column("delta", justify="right")
    for key, value in {
        **rank_delta,
        **facet_delta,
        **fragmentation_delta,
        **page_routing_delta,
    }.items():
        rendered = "-" if value is None else f"{float(value):+.4f}"
        diagnostics_table.add_row(key, rendered)
    active_console.print(diagnostics_table)

    preservation_delta_table = Table(title="Memory Preservation Delta")
    preservation_delta_table.add_column("metric")
    preservation_delta_table.add_column("delta", justify="right")
    for key, value in memory_preservation_delta.items():
        rendered = "-" if value is None else f"{float(value):+.4f}"
        preservation_delta_table.add_row(key, rendered)
    active_console.print(preservation_delta_table)

    stage_delta_table = Table(title="Routing / Retrieval Stage Delta")
    stage_delta_table.add_column("metric")
    stage_delta_table.add_column("delta", justify="right")
    for key, value in stage_delta.items():
        rendered = "-" if value is None else f"{float(value):+.4f}"
        stage_delta_table.add_row(key, rendered)
    active_console.print(stage_delta_table)

    compaction_delta_table = Table(title="Expansion / Compaction Delta")
    compaction_delta_table.add_column("metric")
    compaction_delta_table.add_column("delta", justify="right")
    for key, value in compaction_delta.items():
        rendered = "-" if value is None else f"{float(value):+.4f}"
        compaction_delta_table.add_row(key, rendered)
    active_console.print(compaction_delta_table)

    judge_delta_table = Table(title="Judge Diagnostics Delta")
    judge_delta_table.add_column("metric")
    judge_delta_table.add_column("delta", justify="right")
    for key in [
        "judge_evaluable_count",
        "judge_execution_failed_count",
        "judge_execution_failed_rate_over_all",
        "partial_count",
        "partial_rate_over_all",
        "partial_rate_over_non_correct",
        "mean_partial_credit_judge_acc",
        "structured_requested_rate_over_all",
        "structured_success_rate_over_all",
        "text_fallback_rate_over_all",
        "text_only_rate_over_all",
        "incorrect_queries_judged_via_text_fallback_count",
        "incorrect_queries_judged_via_text_fallback_rate_over_incorrect",
    ]:
        value = judge_delta.get(key)
        rendered = "-" if value is None else f"{float(value):+.4f}"
        judge_delta_table.add_row(key, rendered)
    for category, counts in sorted(dict(judge_delta.get("structured_fallback_category_counts", {})).items()):
        judge_delta_table.add_row(
            f"structured_fallback_category.{category}",
            f"{int(counts.get('delta', 0)):+d}",
        )
    active_console.print(judge_delta_table)

    if llm_delta:
        llm_delta_table = Table(title="LLM Call Diagnostics Delta")
        llm_delta_table.add_column("metric")
        llm_delta_table.add_column("delta", justify="right")
        for key, value in dict(llm_delta.get("overall") or {}).items():
            rendered = "-" if value is None else f"{float(value):+.4f}"
            llm_delta_table.add_row(f"overall.{key}", rendered)
        for task, task_delta in list(dict(llm_delta.get("by_task") or {}).items())[:12]:
            delta_values = dict(dict(task_delta or {}).get("delta") or {})
            llm_delta_table.add_row(
                f"task.{task}.provider_call_count",
                _render_optional_number(delta_values.get("provider_call_count")),
            )
            llm_delta_table.add_row(
                f"task.{task}.total_tokens",
                _render_optional_number(delta_values.get("total_tokens")),
            )
        active_console.print(llm_delta_table)

    if fallback_repair_delta:
        fr_delta_table = Table(title="Fallback / Repair Risk Delta")
        fr_delta_table.add_column("metric")
        fr_delta_table.add_column("delta", justify="right")
        for key, value in dict(fallback_repair_delta.get("overall") or {}).items():
            fr_delta_table.add_row(key, "-" if value is None else f"{float(value):+.4f}")
        for key, value in dict(fallback_repair_delta.get("paper_ready") or {}).items():
            fr_delta_table.add_row(f"paper_ready.{key}", "-" if value is None else f"{float(value):+.4f}")
        active_console.print(fr_delta_table)

    multi_hop_delta_table = Table(title="Multi-Hop Coverage Delta")
    multi_hop_delta_table.add_column("metric")
    multi_hop_delta_table.add_column("delta", justify="right")
    for key in [
        "all_gold_trajectories_in_top_k_rate_over_split_cases",
        "top_k_can_cover_all_gold_refs_rate_over_split_cases",
        "mean_gold_trajectory_recall_at_k_over_split_cases",
        "mean_top_k_gold_ref_coverage_rate_over_split_cases",
        "mean_top_k_redundant_entity_cluster_ratio_over_split_cases",
    ]:
        value = multi_hop_delta.get(key)
        rendered = "-" if value is None else f"{float(value):+.4f}"
        multi_hop_delta_table.add_row(key, rendered)
    active_console.print(multi_hop_delta_table)

    cutoff_delta_table = Table(title="Cutoff Diagnostics Delta")
    cutoff_delta_table.add_column("metric")
    cutoff_delta_table.add_column("delta", justify="right")
    for section_name, group_name, metric_names in [
        (
            "page_cutoffs",
            "all_queries",
            ["all_gold_pages_rate", "selected_pages_cover_all_gold_trajectories_rate"],
        ),
        (
            "trajectory_cutoffs",
            "all_queries",
            ["all_gold_trajectories_rate", "top_k_can_cover_all_gold_refs_rate"],
        ),
    ]:
        section = dict(cutoff_delta.get(section_name, {}))
        for cutoff in _common_cutoff_keys(section):
            group_delta = dict(dict(section.get(cutoff, {})).get(group_name, {}))
            for metric_name in metric_names:
                value = group_delta.get(metric_name)
                rendered = "-" if value is None else f"{float(value):+.4f}"
                prefix = "page" if section_name == "page_cutoffs" else "trajectory"
                cutoff_delta_table.add_row(f"{prefix}@{cutoff}.{metric_name}", rendered)
    for key, value in dict(cutoff_delta.get("suggested_cutoffs", {})).items():
        cutoff_delta_table.add_row(
            f"suggested.{key}",
            f"{value.get('before')} -> {value.get('after')}" if isinstance(value, dict) else "-",
        )
    active_console.print(cutoff_delta_table)

    trajectory_length_delta_table = Table(title="Trajectory Length Delta")
    trajectory_length_delta_table.add_column("metric")
    trajectory_length_delta_table.add_column("before -> after")
    trajectory_length_delta_table.add_column("delta", justify="right")
    for key in [
        "all_trajectory_p95",
        "all_trajectory_at_m_limit_rate",
        "gold_trajectory_p95",
        "gold_trajectory_at_m_limit_rate",
        "gold_required_rank_p95",
        "candidate_lower_m_at_95",
    ]:
        value = dict(trajectory_length_delta.get(key, {}))
        delta = value.get("delta")
        trajectory_length_delta_table.add_row(
            key,
            f"{value.get('before')} -> {value.get('after')}",
            "-" if delta is None else f"{float(delta):+.4f}",
        )
    interpretation = dict(trajectory_length_delta.get("interpretation", {}))
    trajectory_length_delta_table.add_row(
        "interpretation",
        f"{interpretation.get('before')} -> {interpretation.get('after')}",
        "-",
    )
    active_console.print(trajectory_length_delta_table)

    strict_transition_table = Table(title="Strict Answer Error Transitions")
    strict_transition_table.add_column("before")
    strict_transition_table.add_column("after")
    strict_transition_table.add_column("count", justify="right")
    for row in list(diff_report.get("strict_answer_error_transition_counts", [])):
        strict_transition_table.add_row(str(row["before"]), str(row["after"]), str(int(row["count"])))
    active_console.print(strict_transition_table)

    for label, rows in [
        ("improved_queries", list(diff_report.get("improved_queries", []))),
        ("worsened_queries", list(diff_report.get("worsened_queries", []))),
        ("changed_failures", list(diff_report.get("changed_failures", []))),
        ("improved_multi_hop_queries", list(diff_report.get("improved_multi_hop_queries", []))),
        ("worsened_multi_hop_queries", list(diff_report.get("worsened_multi_hop_queries", []))),
    ]:
        if not rows:
            continue
        active_console.print(f"[bold]{label}[/bold]")
        for index, row in enumerate(rows, start=1):
            active_console.print(
                f"{index}. sample={row['sample_id']} query={row['query_task_id']}\n"
                f"   question={row.get('question')}\n"
                f"   before={row['before_reason']} after={row['after_reason']}"
                + (
                    f"\n   before_coverability={row.get('before_coverability')} "
                    f"after_coverability={row.get('after_coverability')}"
                    if "before_coverability" in row or "after_coverability" in row
                    else ""
                )
            )
