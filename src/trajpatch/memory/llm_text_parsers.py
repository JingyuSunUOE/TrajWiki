"""Parsers and validators for plain-text LLM memory outputs."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable

from pydantic import ValidationError

from trajpatch.exceptions import ParserValidationError
from trajpatch.utils.text import collapse_whitespace

from .schemas import (
    ClaimSignalExactTerm,
    ClaimSignalExtractionResult,
    ClaimSignalFacet,
    ClaimOp,
    ClaimTextExtractionClaim,
    ClaimTextExtractionResult,
    ClaimTransitionDecision,
    EpisodicMemoryInput,
    JudgeVerdict,
    MemoryClaim,
    TrajectoryMatchDecision,
)

_ALLOWED_CLAIM_STATUSES = {"active", "deprecated", "contradictory", "needs-confirmation"}
_CLAIM_STATUS_ORDER = ("active", "deprecated", "contradictory", "needs-confirmation")
_STATUS_ALIAS_MAP = {
    "confirmed": "active",
    "confirm": "active",
    "needs confirmation": "needs-confirmation",
    "needs_confirmation": "needs-confirmation",
    "pending confirmation": "needs-confirmation",
    "pending_confirmation": "needs-confirmation",
    "pending-confirmation": "needs-confirmation",
    "contradiction": "contradictory",
    "conflicted": "contradictory",
}


_STRUCTURED_TOP_FIELD_RE = re.compile(r"^\s*[A-Z][A-Z0-9_ ]*\s*:")


def _strip_reasoning_blocks(text: str) -> str:
    return re.sub(r"(?is)<think\b[^>]*>.*?</think>", "", text)


def _looks_like_structured_start(line: str) -> bool:
    stripped = line.strip()
    return (stripped.startswith("[") and stripped.endswith("]")) or bool(
        _STRUCTURED_TOP_FIELD_RE.match(stripped)
    )


def _normalize_lines(text: str) -> list[str]:
    lines: list[str] = []
    in_reasoning = False
    for raw_line in _strip_reasoning_blocks(text).strip().splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        marker = line.strip().casefold()
        if marker.startswith("<think"):
            in_reasoning = True
            continue
        if in_reasoning:
            if "</think>" in marker:
                in_reasoning = False
                continue
            if _looks_like_structured_start(line):
                in_reasoning = False
            else:
                continue
        if marker == "</think>":
            continue
        lines.append(line)
    return lines


def _split_sections(text: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    top_fields: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    seen_structured_content = False
    for line in _normalize_lines(text):
        if line.strip().casefold() in {"<think>", "</think>"}:
            continue
        if not seen_structured_content and not _looks_like_structured_start(line):
            continue
        seen_structured_content = True
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            sections[current_section] = []
            continue
        if current_section is None:
            if ":" not in line:
                raise ParserValidationError(
                    f"Expected KEY: VALUE line, got: {line}",
                    code="malformed_top_field",
                    section="top",
                )
            key, value = line.split(":", 1)
            top_fields[key.strip()] = value.strip()
        else:
            sections[current_section].append(line)
    return top_fields, sections


def _parse_list_section(lines: list[str], *, section: str | None = None) -> list[str]:
    if not lines:
        raise ParserValidationError(
            "Required list section is empty.",
            code="empty_list_section",
            section=section,
        )
    values: list[str] = []
    for line in lines:
        if not line.startswith("- "):
            raise ParserValidationError(
                f"Expected bullet item, got: {line}",
                code="malformed_list_record",
                section=section,
            )
        item = line[2:].strip()
        if item.lower() != "none":
            values.append(item)
    return values


def _parse_pipe_record(line: str, *, section: str) -> dict[str, str]:
    if not line.startswith("- "):
        raise ParserValidationError(
            f"Expected list record, got: {line}",
            code="malformed_list_record",
            section=section,
        )
    record: dict[str, str] = {}
    for part in line[2:].split(" | "):
        if "=" not in part:
            raise ParserValidationError(
                f"Expected key=value segment, got: {part}",
                code="malformed_pipe_record",
                section=section,
                details={"segment": part},
            )
        key, value = part.split("=", 1)
        record[key.strip()] = value.strip()
    return record


def _parse_csv(value: str) -> list[str]:
    if not value or value.lower() == "none":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _canonicalize_status(value: str) -> str:
    normalized = value.strip().lower()
    return _STATUS_ALIAS_MAP.get(normalized, normalized)


def _parse_status_flags(value: str) -> list[str]:
    return [_canonicalize_status(item) for item in _parse_csv(value)]


def _parse_claim_status(value: str) -> str:
    status = _canonicalize_status(value)
    if status not in _ALLOWED_CLAIM_STATUSES:
        raise ParserValidationError(
            f"Invalid claim status: {value.strip()}",
            code="invalid_claim_status",
            section="CLAIMS",
            field="status",
            details={"value": value.strip()},
        )
    return status


def _assign_local_claim_id(index: int) -> str:
    return f"tmp-c{index}"


def _derive_timestamp(*, provided_timestamp: str | None, exchange_timestamp: str | None) -> str:
    if exchange_timestamp:
        return str(exchange_timestamp)
    if provided_timestamp:
        return provided_timestamp.strip()
    return datetime.utcnow().isoformat()


def _derive_links(
    *,
    provided_links: Iterable[str],
    valid_link_ids: set[str] | None,
    exchange_link_ids: list[str] | None,
    diagnostics: list[dict[str, Any]] | None,
) -> list[str]:
    exchange_links = _dedupe_preserve_order(exchange_link_ids or [])
    if exchange_links:
        if valid_link_ids is None:
            return exchange_links
        return [link for link in exchange_links if link in valid_link_ids]
    return _coerce_links(
        provided_links,
        valid_link_ids=valid_link_ids,
        exchange_link_ids=exchange_link_ids,
        section="top",
        field="LINKS",
        field_path="LINKS",
        diagnostics=diagnostics,
    )


def _derive_status_flags_from_claims(claims: list[MemoryClaim]) -> list[str]:
    present = {_canonicalize_status(claim.status) for claim in claims}
    return [status for status in _CLAIM_STATUS_ORDER if status in present]


def _require_record_key(record: dict[str, str], key: str, *, section: str) -> str:
    value = record.get(key)
    if value is None:
        raise ParserValidationError(
            f"Missing required key '{key}' in {section} record.",
            code="missing_record_key",
            section=section,
            field=key,
        )
    return value


def _coerce_links(
    raw_ids: Iterable[str],
    *,
    valid_link_ids: set[str] | None,
    exchange_link_ids: list[str] | None,
    section: str,
    field: str,
    field_path: str,
    diagnostics: list[dict[str, Any]] | None,
) -> list[str]:
    parsed_ids = _dedupe_preserve_order(raw_ids)
    if valid_link_ids is None:
        return parsed_ids

    kept_ids = [link for link in parsed_ids if link in valid_link_ids]
    dropped_ids = [link for link in parsed_ids if link not in valid_link_ids]
    if not dropped_ids:
        return kept_ids

    diagnostic = {
        "kind": "link_salvage",
        "field_path": field_path,
        "parsed_ids": parsed_ids,
        "kept_ids": list(kept_ids),
        "dropped_ids": list(dropped_ids),
        "exchange_link_fallback_used": False,
    }
    if kept_ids:
        if diagnostics is not None:
            diagnostics.append(diagnostic)
        return kept_ids

    fallback_ids = _dedupe_preserve_order(exchange_link_ids or [])
    fallback_ids = [link for link in fallback_ids if link in valid_link_ids]
    if fallback_ids:
        diagnostic["kept_ids"] = list(fallback_ids)
        diagnostic["exchange_link_fallback_used"] = True
        if diagnostics is not None:
            diagnostics.append(diagnostic)
        return fallback_ids

    raise ParserValidationError(
        f"Unknown raw message ids in links: {dropped_ids}",
        code="invalid_links",
        section=section,
        field=field,
        details=diagnostic,
    )


def _record_diagnostic(diagnostics: list[dict[str, Any]] | None, payload: dict[str, Any]) -> None:
    if diagnostics is not None:
        diagnostics.append(payload)


def _default_op_rationale() -> str:
    return "Model-supplied operation omitted rationale; default rationale inserted."


def _record_ignored_op(
    diagnostics: list[dict[str, Any]] | None,
    *,
    raw_line: str,
    code: str,
    field: str | None,
    reason: str,
    details: dict[str, Any] | None = None,
) -> None:
    payload = {
        "kind": "ops_ignored",
        "field_path": "OPS",
        "raw_line": raw_line,
        "code": code,
        "field": field,
        "reason": reason,
    }
    if details:
        payload["details"] = dict(details)
    _record_diagnostic(diagnostics, payload)


def _record_defaulted_op_field(
    diagnostics: list[dict[str, Any]] | None,
    *,
    raw_line: str,
    field: str,
    value: str,
) -> None:
    _record_diagnostic(
        diagnostics,
        {
            "kind": "ops_defaulted",
            "field_path": f"OPS.{field}",
            "raw_line": raw_line,
            "field": field,
            "value": value,
        },
    )


def parse_episodic_memory(
    text: str,
    valid_link_ids: set[str] | None = None,
    *,
    exchange_link_ids: list[str] | None = None,
    exchange_timestamp: str | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    parse_claims: bool = False,
) -> EpisodicMemoryInput | None:
    text = _strip_reasoning_blocks(text)
    if text.strip() == "NO_MEMORY":
        return None
    top, sections = _split_sections(text)
    required_top = {
        "SUMMARY_CONTENT",
        "CONTEXT",
        "KEYWORDS",
    }
    missing_top = required_top - set(top)
    if missing_top:
        raise ParserValidationError(f"Missing top-level episodic fields: {sorted(missing_top)}")
    if parse_claims and "CLAIMS" not in sections:
        raise ParserValidationError("Missing episodic [CLAIMS] section.")
    if not parse_claims and "CLAIMS" in sections:
        _record_diagnostic(
            diagnostics,
            {
                "kind": "legacy_claims_ignored",
                "field_path": "CLAIMS",
                "claim_count": len(sections.get("CLAIMS", [])),
            },
        )
    if not parse_claims and "OPS" in sections:
        _record_diagnostic(
            diagnostics,
            {
                "kind": "legacy_ops_ignored",
                "field_path": "OPS",
                "op_count": len(sections.get("OPS", [])),
            },
        )
    links = _derive_links(
        provided_links=_parse_csv(top.get("LINKS", "")),
        valid_link_ids=valid_link_ids,
        exchange_link_ids=exchange_link_ids,
        diagnostics=diagnostics,
    )
    timestamp = _derive_timestamp(
        provided_timestamp=top.get("TIMESTAMP"),
        exchange_timestamp=exchange_timestamp,
    )
    claims: list[MemoryClaim] = []
    for index, line in enumerate(sections.get("CLAIMS", []) if parse_claims else [], start=1):
        item = _parse_pipe_record(line, section="CLAIMS")
        source_ids = _coerce_links(
            _parse_csv(item.get("source_message_ids", "")),
            valid_link_ids=valid_link_ids,
            exchange_link_ids=exchange_link_ids,
            section="CLAIMS",
            field="source_message_ids",
            field_path="CLAIMS.source_message_ids",
            diagnostics=diagnostics,
        )
        claims.append(
            MemoryClaim(
                claim_id=_assign_local_claim_id(index),
                status=_parse_claim_status(_require_record_key(item, "status", section="CLAIMS")),
                source_message_ids=source_ids,
                text=_require_record_key(item, "text", section="CLAIMS"),
            )
        )
    ops: list[ClaimOp] = []
    for line in sections.get("OPS", []) if parse_claims else []:
        try:
            item = _parse_pipe_record(line, section="OPS")
            source_ids = _coerce_links(
                _parse_csv(item.get("source_message_ids", "")),
                valid_link_ids=valid_link_ids,
                exchange_link_ids=exchange_link_ids,
                section="OPS",
                field="source_message_ids",
                field_path="OPS.source_message_ids",
                diagnostics=diagnostics,
            )
            claim_text = item.get("claim_text")
            if claim_text is not None and claim_text.lower() == "none":
                claim_text = None
            new_claim_id = item.get("new_claim_id")
            if new_claim_id is not None and new_claim_id.lower() == "none":
                new_claim_id = None
            rationale = item.get("rationale")
            if rationale is None:
                rationale = _default_op_rationale()
                _record_defaulted_op_field(
                    diagnostics,
                    raw_line=line,
                    field="rationale",
                    value=rationale,
                )
            ops.append(
                ClaimOp(
                    op=_require_record_key(item, "op", section="OPS"),
                    target_claim_id=_require_record_key(item, "target_claim_id", section="OPS"),
                    new_claim_id=new_claim_id,
                    source_message_ids=source_ids,
                    rationale=rationale,
                    claim_text=claim_text,
                )
            )
        except ParserValidationError as exc:
            _record_ignored_op(
                diagnostics,
                raw_line=line,
                code=str(exc.code or "invalid_ops_record"),
                field=exc.field,
                reason=str(exc),
                details=exc.details,
            )
            continue
        except ValidationError as exc:
            _record_ignored_op(
                diagnostics,
                raw_line=line,
                code="invalid_op_type",
                field="op",
                reason=str(exc),
            )
            continue
    status_flags = _derive_status_flags_from_claims(claims)
    return EpisodicMemoryInput(
        memory_type="episodic",
        timestamp=timestamp,
        summary_content=top["SUMMARY_CONTENT"],
        context=top["CONTEXT"],
        keywords=_parse_csv(top["KEYWORDS"]),
        links=links,
        status_flags=status_flags,  # type: ignore[arg-type]
        claims=claims,
        ops=ops,
        raw_text=text,
    )


def parse_match_decision(
    text: str,
    allowed_candidate_labels: dict[str, str] | None = None,
) -> TrajectoryMatchDecision:
    top, _ = _split_sections(text)
    required = {"DECISION", "SELECTED_CANDIDATE", "RATIONALE"}
    missing = required - set(top)
    if missing:
        raise ParserValidationError(f"Missing match decision fields: {sorted(missing)}")
    decision = top["DECISION"].strip().upper()
    selected_candidate = top["SELECTED_CANDIDATE"].strip()
    if selected_candidate.lower() == "none":
        selected_candidate = None
    if decision not in {"CONTINUE", "NEW"}:
        raise ParserValidationError(
            f"Invalid trajectory match decision: {decision}",
            code="invalid_trajectory_match_decision",
            section="top",
            field="DECISION",
            details={"value": decision},
        )
    if decision == "CONTINUE":
        if not selected_candidate:
            raise ParserValidationError(
                "Trajectory match CONTINUE requires SELECTED_CANDIDATE.",
                code="missing_selected_candidate",
                section="top",
                field="SELECTED_CANDIDATE",
            )
        if (
            allowed_candidate_labels is not None
            and selected_candidate not in allowed_candidate_labels
        ):
            raise ParserValidationError(
                f"Trajectory match selected candidate {selected_candidate} is not in candidate shortlist.",
                code="invalid_selected_candidate",
                section="top",
                field="SELECTED_CANDIDATE",
                details={"value": selected_candidate},
            )
        trajectory_id = (
            allowed_candidate_labels[selected_candidate]
            if allowed_candidate_labels is not None
            else selected_candidate
        )
    else:
        if selected_candidate is not None:
            raise ParserValidationError(
                "Trajectory match NEW must set SELECTED_CANDIDATE to none.",
                code="unexpected_selected_candidate",
                section="top",
                field="SELECTED_CANDIDATE",
                details={"value": selected_candidate},
            )
        trajectory_id = None
    return TrajectoryMatchDecision(
        decision=decision,  # type: ignore[arg-type]
        trajectory_id=trajectory_id,
        rationale=top["RATIONALE"],
    )


def parse_claim_text_extraction(
    text: str,
    valid_link_ids: set[str] | None = None,
    *,
    exchange_link_ids: list[str] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> ClaimTextExtractionResult:
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("TASK="))
    top, sections = _split_sections(text)
    raw_has_claims = top.get("HAS_CLAIMS", "true").strip().casefold()
    has_claims = raw_has_claims not in {"false", "no", "0"}
    if not has_claims:
        return ClaimTextExtractionResult(has_claims=False, claims=[], reason=top.get("REASON"))
    claims: list[ClaimTextExtractionClaim] = []
    for line in sections.get("CLAIMS", []):
        item = _parse_pipe_record(line, section="CLAIMS")
        source_ids = _coerce_links(
            _parse_csv(item.get("source_message_ids", "")),
            valid_link_ids=valid_link_ids,
            exchange_link_ids=exchange_link_ids,
            section="CLAIMS",
            field="source_message_ids",
            field_path="CLAIMS.source_message_ids",
            diagnostics=diagnostics,
        )
        claims.append(
            ClaimTextExtractionClaim(
                status=_parse_claim_status(_require_record_key(item, "status", section="CLAIMS")),
                source_message_ids=source_ids,
                supporting_quote=_require_record_key(item, "supporting_quote", section="CLAIMS"),
                text=_require_record_key(item, "text", section="CLAIMS"),
            )
        )
    if not claims:
        raise ParserValidationError("Claim text extraction returned no claims.", code="missing_claim_text_claims")
    return ClaimTextExtractionResult(has_claims=True, claims=claims, reason=top.get("REASON"))


def parse_claim_signal_extraction(text: str) -> ClaimSignalExtractionResult:
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("TASK="))
    _, sections = _split_sections(text)
    exact_terms: list[ClaimSignalExactTerm] = []
    for line in sections.get("EXACT_TERMS", []):
        if line.strip().lower() in {"- none", "none"}:
            continue
        item = _parse_pipe_record(line, section="EXACT_TERMS")
        exact_terms.append(
            ClaimSignalExactTerm(
                surface=_require_record_key(item, "surface", section="EXACT_TERMS"),
                category=_require_record_key(item, "category", section="EXACT_TERMS"),
                source_claim_id=_require_record_key(item, "source_claim_id", section="EXACT_TERMS"),
                source_message_ids=_parse_csv(item.get("source_message_ids", "")),
            )
        )
    facets: list[ClaimSignalFacet] = []
    for line in sections.get("FACETS", []):
        if line.strip().lower() in {"- none", "none"}:
            continue
        item = _parse_pipe_record(line, section="FACETS")
        facets.append(
            ClaimSignalFacet(
                relation=_require_record_key(item, "relation", section="FACETS"),
                value=_require_record_key(item, "value", section="FACETS"),
                entity=item.get("entity") or None,
                value_span=item.get("value_span") or None,
                source_claim_id=_require_record_key(item, "source_claim_id", section="FACETS"),
                source_message_ids=_parse_csv(item.get("source_message_ids", "")),
            )
        )
    return ClaimSignalExtractionResult(
        exact_terms=exact_terms,
        facets=facets,
        display_items=_parse_list_section(sections.get("DISPLAY_ITEMS", ["- none"]), section="DISPLAY_ITEMS"),
        display_named_entities=_parse_list_section(
            sections.get("DISPLAY_NAMED_ENTITIES", ["- none"]),
            section="DISPLAY_NAMED_ENTITIES",
        ),
        display_counts=_parse_list_section(sections.get("DISPLAY_COUNTS", ["- none"]), section="DISPLAY_COUNTS"),
        display_key_facts=_parse_list_section(
            sections.get("DISPLAY_KEY_FACTS", ["- none"]),
            section="DISPLAY_KEY_FACTS",
        ),
    )


def parse_claim_transition_decision(
    text: str,
    allowed_candidate_labels: dict[str, str] | None = None,
) -> ClaimTransitionDecision:
    top, _ = _split_sections(text)
    required = {"DECISION", "SELECTED_CANDIDATE", "RATIONALE"}
    missing = required - set(top)
    if missing:
        raise ParserValidationError(f"Missing claim transition fields: {sorted(missing)}")
    decision = top["DECISION"].strip().upper()
    selected_candidate = top["SELECTED_CANDIDATE"].strip()
    if selected_candidate.lower() == "none":
        selected_candidate = None
    if decision not in {"REVISE", "ADD"}:
        raise ParserValidationError(
            f"Invalid claim transition decision: {decision}",
            code="invalid_claim_transition_decision",
            section="top",
            field="DECISION",
            details={"value": decision},
        )
    if decision == "REVISE":
        if not selected_candidate:
            raise ParserValidationError(
                "Claim transition REVISE requires SELECTED_CANDIDATE.",
                code="missing_selected_candidate",
                section="top",
                field="SELECTED_CANDIDATE",
            )
        if (
            allowed_candidate_labels is not None
            and selected_candidate not in allowed_candidate_labels
        ):
            raise ParserValidationError(
                f"Claim transition selected candidate {selected_candidate} is not in candidate shortlist.",
                code="invalid_selected_candidate",
                section="top",
                field="SELECTED_CANDIDATE",
                details={"value": selected_candidate},
            )
        previous_claim_id = (
            allowed_candidate_labels[selected_candidate]
            if allowed_candidate_labels is not None
            else selected_candidate
        )
    else:
        if selected_candidate is not None:
            raise ParserValidationError(
                "Claim transition ADD must set SELECTED_CANDIDATE to none.",
                code="unexpected_selected_candidate",
                section="top",
                field="SELECTED_CANDIDATE",
                details={"value": selected_candidate},
            )
        previous_claim_id = None
    return ClaimTransitionDecision(
        decision=decision,  # type: ignore[arg-type]
        previous_claim_id=previous_claim_id,
        rationale=top["RATIONALE"],
    )


def parse_judge_verdict(text: str) -> JudgeVerdict:
    text = _strip_reasoning_blocks(text)
    normalized = text.strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        fenced_lines = normalized.splitlines()
        if len(fenced_lines) >= 2:
            normalized = "\n".join(fenced_lines[1:-1]).strip()
    normalized = normalized.strip().strip("`").strip()
    normalized = collapse_whitespace(normalized).strip()
    if " " not in normalized:
        normalized = normalized.rstrip(".!,;:")
    upper = normalized.upper()
    if upper in {"CORRECT", "PARTIAL", "INCORRECT"}:
        return JudgeVerdict(verdict=upper, score=None, rationale=None)
    if upper in {"PASS", "FAIL"}:
        return JudgeVerdict(
            verdict="CORRECT" if upper == "PASS" else "INCORRECT",
            score=None,
            rationale=None,
        )
    top, _ = _split_sections(text)
    verdict_text = str(top.get("VERDICT") or "").strip().upper().rstrip(".!,;:")
    if verdict_text not in {"CORRECT", "PARTIAL", "INCORRECT", "PASS", "FAIL"}:
        raise ParserValidationError("Missing or invalid judge verdict.", code="invalid_judge_verdict")
    if verdict_text == "PARTIAL":
        mapped_verdict = "PARTIAL"
    else:
        mapped_verdict = "CORRECT" if verdict_text in {"CORRECT", "PASS"} else "INCORRECT"
    score_raw = top.get("SCORE")
    rationale_raw = top.get("RATIONALE")
    return JudgeVerdict(
        verdict=mapped_verdict,
        score=float(score_raw) if score_raw not in {None, ""} else None,
        rationale=str(rationale_raw) if rationale_raw not in {None, ""} else None,
    )
