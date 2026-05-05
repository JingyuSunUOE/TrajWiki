"""Helpers for normalizing, repairing, and recovering structured memory DSL."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from trajpatch.exceptions import ParserValidationError
from trajpatch.memory.schemas import ClaimOp, EpisodicMemoryInput, MemoryClaim
from trajpatch.types import NormalizedMessage
from trajpatch.utils.text import collapse_whitespace, extract_keywords

MEMORY_DSL_START_MARKER = "BEGIN_MEMORY_DSL"
MEMORY_DSL_END_MARKER = "END_MEMORY_DSL"

EPISODIC_TOP_FIELDS = [
    "SUMMARY_CONTENT",
    "CONTEXT",
    "KEYWORDS",
]
EPISODIC_KNOWN_TOP_FIELDS = [
    "TIMESTAMP",
    "SUMMARY_CONTENT",
    "CONTEXT",
    "KEYWORDS",
    "LINKS",
    "STATUS_FLAGS",
]
EPISODIC_REQUIRED_SECTIONS: list[str] = []
EPISODIC_OPTIONAL_SECTIONS = ["CLAIMS", "OPS"]
KNOWN_TOP_FIELDS = set(EPISODIC_KNOWN_TOP_FIELDS)
KNOWN_SECTIONS = set(EPISODIC_REQUIRED_SECTIONS + EPISODIC_OPTIONAL_SECTIONS)
CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\n(.*?)```", re.DOTALL)


@dataclass(slots=True)
class PartialMemoryDraft:
    memory_type: str
    top_fields: dict[str, str] = field(default_factory=dict)
    sections: dict[str, list[str]] = field(default_factory=dict)
    raw_output: str = ""
    normalized_text: str = ""
    explanation_text: str = ""
    missing_top_fields: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    repair_targets: list[str] = field(default_factory=list)
    validation_error: str | None = None
    validation_error_code: str | None = None
    validation_error_section: str | None = None
    validation_error_field: str | None = None
    validation_error_details: dict[str, Any] = field(default_factory=dict)
    parsed_claim_count: int = 0
    parsed_op_count: int = 0

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "memory_type": self.memory_type,
            "top_fields": dict(self.top_fields),
            "sections": {key: list(value) for key, value in self.sections.items()},
            "raw_output": self.raw_output,
            "normalized_text": self.normalized_text,
            "explanation_text": self.explanation_text,
            "missing_top_fields": list(self.missing_top_fields),
            "missing_sections": list(self.missing_sections),
            "repair_targets": list(self.repair_targets),
            "validation_error": self.validation_error,
            "validation_error_code": self.validation_error_code,
            "validation_error_section": self.validation_error_section,
            "validation_error_field": self.validation_error_field,
            "validation_error_details": dict(self.validation_error_details),
            "parsed_claim_count": self.parsed_claim_count,
            "parsed_op_count": self.parsed_op_count,
        }


def _required_fields(memory_type: str) -> tuple[list[str], list[str]]:
    return EPISODIC_TOP_FIELDS, EPISODIC_REQUIRED_SECTIONS


def _looks_like_top_field(line: str) -> bool:
    if ":" not in line:
        return False
    key = line.split(":", 1)[0].strip()
    return key in KNOWN_TOP_FIELDS


def _normalize_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.strip().splitlines() if line.strip()]


def normalize_memory_text(raw_output: str) -> tuple[str, str]:
    stripped = raw_output.strip()
    if not stripped:
        return "", ""
    if stripped == "NO_MEMORY":
        return "NO_MEMORY", ""

    marker_pattern = re.compile(
        rf"{MEMORY_DSL_START_MARKER}\s*(.*?)\s*{MEMORY_DSL_END_MARKER}",
        re.DOTALL,
    )
    marker_match = marker_pattern.search(stripped)
    if marker_match:
        candidate = marker_match.group(1).strip()
        explanation = collapse_whitespace(
            marker_pattern.sub(" ", stripped).replace(MEMORY_DSL_START_MARKER, " ").replace(MEMORY_DSL_END_MARKER, " ")
        )
        return candidate, explanation

    for match in CODE_BLOCK_RE.finditer(stripped):
        block = match.group(1).strip()
        if any(section in block for section in ("[CLAIMS]", "[OPS]")) or any(
            _looks_like_top_field(line.strip()) for line in block.splitlines() if line.strip()
        ):
            explanation = collapse_whitespace(stripped[: match.start()] + " " + stripped[match.end() :])
            return block, explanation

    lines = stripped.splitlines()
    for index, line in enumerate(lines):
        if _looks_like_top_field(line.strip()):
            return "\n".join(lines[index:]).strip(), collapse_whitespace("\n".join(lines[:index]))

    return stripped, collapse_whitespace(stripped)


def _split_sections_relaxed(text: str) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    top_fields: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    orphan_lines: list[str] = []
    current_section: str | None = None
    for line in _normalize_lines(text):
        if line in {MEMORY_DSL_START_MARKER, MEMORY_DSL_END_MARKER}:
            continue
        if line.startswith("[") and line.endswith("]") and line[1:-1] in KNOWN_SECTIONS:
            current_section = line[1:-1]
            sections.setdefault(current_section, [])
            continue
        if current_section is None:
            if _looks_like_top_field(line):
                key, value = line.split(":", 1)
                top_fields[key.strip()] = value.strip()
            else:
                orphan_lines.append(line)
        else:
            sections.setdefault(current_section, []).append(line)
    return top_fields, sections, orphan_lines


def _extract_targets_from_error(memory_type: str, validation_error: ParserValidationError | str | None) -> list[str]:
    if not validation_error:
        return []
    if isinstance(validation_error, ParserValidationError):
        if validation_error.section == "OPS":
            return []
        if validation_error.section == "CLAIMS":
            return ["CLAIMS"]
        if validation_error.section == "top" and validation_error.field:
            return [str(validation_error.field)]
    candidate_names = set(_required_fields(memory_type)[0] + _required_fields(memory_type)[1])
    return sorted(name for name in candidate_names if name in str(validation_error))


def build_partial_memory_draft(
    memory_type: str,
    raw_output: str,
    *,
    validation_error: ParserValidationError | str | None = None,
) -> PartialMemoryDraft:
    normalized_text, explanation_text = normalize_memory_text(raw_output)
    top_fields, sections, orphan_lines = _split_sections_relaxed(normalized_text)
    required_top_fields, required_sections = _required_fields(memory_type)
    missing_top_fields = [field_name for field_name in required_top_fields if not top_fields.get(field_name)]
    missing_sections = [section_name for section_name in required_sections if section_name not in sections]
    repair_targets = sorted(set(missing_top_fields + missing_sections + _extract_targets_from_error(memory_type, validation_error)))
    validation_error_message = str(validation_error) if validation_error is not None else None
    validation_error_code = validation_error.code if isinstance(validation_error, ParserValidationError) else None
    validation_error_section = validation_error.section if isinstance(validation_error, ParserValidationError) else None
    validation_error_field = validation_error.field if isinstance(validation_error, ParserValidationError) else None
    validation_error_details = (
        dict(validation_error.details)
        if isinstance(validation_error, ParserValidationError)
        else {}
    )
    if orphan_lines and not explanation_text:
        explanation_text = collapse_whitespace("\n".join(orphan_lines))
    return PartialMemoryDraft(
        memory_type=memory_type,
        top_fields=top_fields,
        sections=sections,
        raw_output=raw_output,
        normalized_text=normalized_text,
        explanation_text=explanation_text,
        missing_top_fields=missing_top_fields,
        missing_sections=missing_sections,
        repair_targets=repair_targets,
        validation_error=validation_error_message,
        validation_error_code=validation_error_code,
        validation_error_section=validation_error_section,
        validation_error_field=validation_error_field,
        validation_error_details=validation_error_details,
        parsed_claim_count=len(sections.get("CLAIMS", [])),
        parsed_op_count=len(sections.get("OPS", [])),
    )


def merge_partial_memory_drafts(base: PartialMemoryDraft | None, update: PartialMemoryDraft) -> PartialMemoryDraft:
    if base is None:
        return update

    replace_targets = set(base.repair_targets) | set(update.repair_targets)
    merged_top_fields = dict(base.top_fields)
    for field_name, value in update.top_fields.items():
        if field_name not in merged_top_fields or field_name in replace_targets or not merged_top_fields[field_name]:
            merged_top_fields[field_name] = value

    merged_sections = {key: list(value) for key, value in base.sections.items()}
    for section_name, lines in update.sections.items():
        if section_name not in merged_sections or section_name in replace_targets or not merged_sections[section_name]:
            merged_sections[section_name] = list(lines)

    merged_raw_output = base.raw_output + "\n\n---\n\n" + update.raw_output
    merged_explanation = collapse_whitespace(" ".join(part for part in [base.explanation_text, update.explanation_text] if part))
    merged_text = render_partial_memory_draft(
        update.memory_type,
        PartialMemoryDraft(
            memory_type=update.memory_type,
            top_fields=merged_top_fields,
            sections=merged_sections,
        ),
    )
    merged = build_partial_memory_draft(update.memory_type, merged_text)
    merged.raw_output = merged_raw_output
    merged.explanation_text = merged_explanation
    merged.validation_error = update.validation_error or base.validation_error
    merged.validation_error_code = update.validation_error_code or base.validation_error_code
    merged.validation_error_section = update.validation_error_section or base.validation_error_section
    merged.validation_error_field = update.validation_error_field or base.validation_error_field
    merged.validation_error_details = {
        **dict(base.validation_error_details),
        **dict(update.validation_error_details),
    }
    merged.repair_targets = sorted(set(merged.repair_targets) | replace_targets)
    return merged


def render_partial_memory_draft(memory_type: str, draft: PartialMemoryDraft) -> str:
    required_top_fields, required_sections = _required_fields(memory_type)
    top_lines: list[str] = []
    for field_name in required_top_fields:
        value = draft.top_fields.get(field_name)
        if value:
            top_lines.append(f"{field_name}: {value}")
    extra_fields = sorted(field_name for field_name in draft.top_fields if field_name not in required_top_fields)
    for field_name in extra_fields:
        top_lines.append(f"{field_name}: {draft.top_fields[field_name]}")

    section_chunks: list[str] = []
    ordered_sections = required_sections + [section for section in sorted(draft.sections) if section not in required_sections]
    for section_name in ordered_sections:
        if section_name not in draft.sections:
            continue
        chunk_lines = [f"[{section_name}]"]
        chunk_lines.extend(draft.sections[section_name])
        section_chunks.append("\n".join(chunk_lines))

    chunks = []
    if top_lines:
        chunks.append("\n".join(top_lines))
    chunks.extend(section_chunks)
    return "\n\n".join(chunk for chunk in chunks if chunk).strip()


def build_section_repair_prompt(
    *,
    template: str,
    memory_type: str,
    conversation: str,
    draft: PartialMemoryDraft,
    validation_error: str,
) -> str:
    accepted = render_partial_memory_draft(memory_type, draft) or "<none>"
    missing_items = draft.repair_targets or draft.missing_top_fields + draft.missing_sections
    missing_text = "\n".join(f"- {item}" for item in missing_items) if missing_items else "- Validate and correct the malformed part."
    return (
        template
        + "\n\nAccepted content to keep exactly as-is:\n"
        + accepted
        + "\n\nMissing or invalid items to repair only:\n"
        + missing_text
        + "\n\nOriginal conversation:\n"
        + conversation
        + "\n\nPrevious raw output and explanation text:\n"
        + draft.raw_output
        + "\n\nValidation error:\n"
        + validation_error
        + "\n\nRules:\n"
        + f"- Preserve the accepted content exactly.\n"
        + "- Only output the missing or corrected DSL content.\n"
        + f"- Put the corrected DSL inside {MEMORY_DSL_START_MARKER} and {MEMORY_DSL_END_MARKER}.\n"
        + "- Do not add any commentary outside the requested repair."
    )


def derive_exchange_timestamp(exchange_messages: list[NormalizedMessage]) -> str:
    for message in reversed(exchange_messages):
        if message.occurred_at:
            return str(message.occurred_at)
    return datetime.utcnow().isoformat()


def derive_exchange_links(exchange_messages: list[NormalizedMessage]) -> list[str]:
    return [message.raw_message_id for message in exchange_messages if message.raw_message_id]


def _fallback_basis_text(draft: PartialMemoryDraft | None, exchange_messages: list[NormalizedMessage]) -> str:
    if draft is not None:
        for candidate in (draft.explanation_text, draft.normalized_text, draft.raw_output):
            collapsed = collapse_whitespace(candidate)
            if collapsed and collapsed != "NO_MEMORY":
                return collapsed
    exchange_text = collapse_whitespace(" ".join(message.content for message in exchange_messages))
    return exchange_text or "Fallback extraction synthesized from conversation text."


def _truncate_sentence(text: str, limit: int = 180) -> str:
    collapsed = collapse_whitespace(text)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def build_fallback_episodic_memory(
    draft: PartialMemoryDraft | None,
    exchange_messages: list[NormalizedMessage],
) -> EpisodicMemoryInput:
    basis = _fallback_basis_text(draft, exchange_messages)
    links = derive_exchange_links(exchange_messages)
    summary = _truncate_sentence(basis, 140) or "Fallback episodic memory extracted from conversation."
    keywords = list(sorted(extract_keywords(basis)))[:6] or ["fallback", "memory"]
    snapshot = EpisodicMemoryInput(
        memory_type="episodic",
        timestamp=derive_exchange_timestamp(exchange_messages),
        summary_content=summary,
        context=basis,
        keywords=keywords,
        links=links,
        status_flags=[],
        claims=[],
        ops=[],
        raw_text="",
    )
    snapshot.raw_text = render_episodic_memory(snapshot)
    return snapshot
def render_episodic_memory(memory: EpisodicMemoryInput) -> str:
    lines = [
        "MEMORY_TYPE: EPISODIC",
        f"TIMESTAMP: {memory.timestamp}",
        f"SUMMARY_CONTENT: {memory.summary_content}",
        f"CONTEXT: {memory.context}",
        f"KEYWORDS: {', '.join(memory.keywords) if memory.keywords else 'fallback'}",
        f"LINKS: {', '.join(memory.links) if memory.links else 'none'}",
        f"STATUS_FLAGS: {', '.join(memory.status_flags)}",
        "",
        "[CLAIMS]",
    ]
    lines.extend(
        [
            "- claim_id="
            + claim.claim_id
            + " | status="
            + claim.status
            + " | source_message_ids="
            + (", ".join(claim.source_message_ids) if claim.source_message_ids else "none")
            + " | text="
            + claim.text
            for claim in memory.claims
        ]
    )
    if memory.ops:
        lines.extend(["", "[OPS]"])
        lines.extend(
            [
                "- op="
                + op.op
                + " | target_claim_id="
                + op.target_claim_id
                + " | new_claim_id="
                + (op.new_claim_id or "none")
                + " | source_message_ids="
                + (", ".join(op.source_message_ids) if op.source_message_ids else "none")
                + " | rationale="
                + op.rationale
                + " | claim_text="
                + (op.claim_text or "none")
                for op in memory.ops
            ]
        )
    return "\n".join(lines).strip()
