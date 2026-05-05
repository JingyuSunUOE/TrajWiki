"""Source-grounded preservation checks for episodic claim extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from trajpatch.memory.facets import (
    clean_exact_terms_v1,
    clean_facet_records_v1,
    extract_claim_facets_v1,
    extract_exact_term_candidates_v1,
    facet_value_key,
    normalize_facet_value,
)
from trajpatch.memory.schemas import MemoryClaim
from trajpatch.storage.models import RawMessageRecord
from trajpatch.types import NormalizedMessage
from trajpatch.utils.text import collapse_whitespace


@dataclass(frozen=True, slots=True)
class MustPreserveCandidate:
    surface: str
    category: str
    source_message_ids: list[str]
    relation: str | None = None
    confidence: str = "high"
    source_refs: list[str] = field(default_factory=list)
    rule: str | None = None
    raw_surface: str | None = None
    canonical: str | None = None
    event_action: str | None = None
    temporal_expression: str | None = None

    def to_metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "surface": self.surface,
            "category": self.category,
            "source_message_ids": list(self.source_message_ids),
            "confidence": self.confidence,
        }
        if self.source_refs:
            payload["source_refs"] = list(self.source_refs)
        if self.rule:
            payload["rule"] = self.rule
        if self.relation:
            payload["relation"] = self.relation
        if self.raw_surface:
            payload["raw_surface"] = collapse_whitespace(self.raw_surface)
        if self.canonical:
            payload["canonical"] = collapse_whitespace(self.canonical)
        if self.event_action:
            payload["action"] = collapse_whitespace(self.event_action)
        if self.temporal_expression:
            payload["temporal_expression"] = collapse_whitespace(self.temporal_expression)
        return payload


@dataclass(slots=True)
class ClaimCoverageResult:
    covered: bool
    missing_candidates: list[MustPreserveCandidate] = field(default_factory=list)
    weak_source_links: list[dict[str, object]] = field(default_factory=list)
    diagnostics: list[dict[str, object]] = field(default_factory=list)


_INVENTORY_TRIGGER_RE = re.compile(
    r"\b(?:activities?|hobbies|interests?|items?|things?|books?|recipes?|dishes?|instruments?|symbols?)\b"
    r"[^.!?]{0,80}?\b(?:include|includes|included|are|were|is|was|involve|involves)\b\s+([^.!?]+)",
    re.IGNORECASE,
)
_ACTIVITY_VERB_LIST_RE = re.compile(
    r"\b(?:enjoys?|likes?|loves?|does|plays?|practices?|participates?\s+in|partakes?\s+in)\b\s+([^.!?]+)",
    re.IGNORECASE,
)
_PREFERENCE_EXCITEMENT_RE = re.compile(
    r"\b(?:stoked|excited)\s+(?:about|for)\s+([^.!?]+)",
    re.IGNORECASE,
)
_PREFERENCE_VERB_RE = re.compile(
    r"\b(?:love|loved|like|liked|likes|enjoy|enjoyed|enjoys|prefer|preferred|prefers)\s+([^.!?]+)",
    re.IGNORECASE,
)
_FAVORITE_VALUE_RE = re.compile(
    r"\bfavo(?:u)?rite\s+[a-zA-Z][a-zA-Z0-9'’\-\s]{0,40}?\s+"
    r"(?:is|was|are|were|:)\s*(?:a|an|the)?\s*([a-zA-Z][a-zA-Z0-9'’\-\s]{2,70})",
    re.IGNORECASE,
)
_RESEARCH_TOPIC_RE = re.compile(
    r"\bresearch(?:ed|es|ing)?\s+(?:about\s+|on\s+|into\s+)?([a-zA-Z][a-zA-Z0-9'’\-\s]{2,80})",
    re.IGNORECASE,
)
_EVENT_PHRASE_RE = re.compile(
    r"\b("
    r"school\s+speech|school\s+talk|mentorship\s+program|mentoring\s+program|mentor\s+program|[a-zA-Z][a-zA-Z'\-]+\s+workshop|"
    r"[a-zA-Z][a-zA-Z'\-]+\s+program"
    r")\b",
    re.IGNORECASE,
)
_SCHOOL_EVENT_RE = re.compile(r"\bschool\s+event\b", re.IGNORECASE)
_SCHOOL_SPEECH_SIGNAL_RE = re.compile(
    r"\b("
    r"talk(?:ed|ing)?|giving\s+(?:my\s+|her\s+|his\s+|their\s+)?talk|"
    r"gave\s+(?:a\s+)?talk|speech|shared\s+(?:my|her|his|their)\s+(?:own\s+)?journey|"
    r"encouraged\s+students"
    r")\b",
    re.IGNORECASE,
)
_MENTORSHIP_PROGRAM_RE = re.compile(r"\b(?:mentorship|mentoring|mentor)\s+program\b", re.IGNORECASE)
_EVENT_OBJECT_ACTION_RE = re.compile(
    r"\b(?P<action>host(?:ed|ing)?|attend(?:ed|ing)?|visit(?:ed|ing)?|went|go|"
    r"chose\s+to\s+go|launch(?:ed|ing)?|start(?:ed|ing)?|join(?:ed|ing)?|"
    r"participat(?:ed|ing)|run(?:ning)?|gave|give|giving)\b"
    r"(?:\s+(?:a|an|the|my|our))?"
    r"(?:\s+(?:to|in|at|for|with|on))?\s+"
    r"(?P<object>[A-Za-z][A-Za-z0-9'’\-\s]{1,80}?\b(?:competition|contest|events?|race|"
    r"campaign|conference|workshop|program|speech|talk|roadtrip|road\s+trip|meeting|parade|"
    r"museum|exhibit|fair|convention))\b",
    re.IGNORECASE,
)
_NETWORKING_EVENT_RE = re.compile(
    r"\b(?P<action>went|go|visited|attended|chose\s+to\s+go)\s+to\s+"
    r"(?P<object>networking\s+events?)\b",
    re.IGNORECASE,
)
_ROADTRIP_RE = re.compile(
    r"\b(?P<object>road\s*trip|roadtrip)\b(?P<trailing>[^.!?]{0,100})",
    re.IGNORECASE,
)
_CAR_ACCIDENT_RE = re.compile(
    r"\b(?:(?:damaged|crashed|wrecked)\s+car|car\s+(?:accident|crash|wreck)|"
    r"flatbed|tow(?:ed|ing)?\s+truck)\b",
    re.IGNORECASE,
)
_TEMPORAL_EXPRESSION_RE = re.compile(
    r"\b("
    r"yesterday|today|tomorrow|recently|last\s+(?:week|weekend|month|year|night|summer|winter|spring|fall|autumn)|"
    r"this\s+past\s+weekend|this\s+(?:week|weekend|month|year|summer|winter|spring|fall|autumn)|"
    r"next\s+(?:week|weekend|month|year)|\d+\s+(?:days?|weeks?|months?|years?)\s+ago"
    r")\b",
    re.IGNORECASE,
)
_LIST_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b|\bor\b)\s+", re.IGNORECASE)
_COUNT_CUE_RE = re.compile(
    r"\b("
    r"\d+|one|two|three|four|five|six|seven|eight|nine|ten|once|twice"
    r")\b(?:\s+(times?|books?|recipes?|dishes?|dogs?|kids?|children|events?|activities|places|cities|wins?|medals?))?",
    re.IGNORECASE,
)
_PAINTED_OBJECT_RE = re.compile(
    r"\b(?:painted|paint|drew|drawn|sketched)\b"
    r"(?:\s+(?:a|an|the|this|that|my|our|another))?\s+"
    r"([a-zA-Z][a-zA-Z0-9'’\-\s]{2,70})",
    re.IGNORECASE,
)
_PICTURE_OF_RE = re.compile(
    r"\b(?:painting|picture|image|drawing|artwork|sketch)\s+(?:of|with)\s+"
    r"(?:a|an|the|this|that|my|our)?\s*([a-zA-Z][a-zA-Z0-9'’\-\s]{2,70})",
    re.IGNORECASE,
)
_FAVORITE_FOOD_RE = re.compile(
    r"\bfavo(?:u)?rite\s+(?:food|dish|snack|dessert|meal|recipe)\s+"
    r"(?:is|was|are|were|:)?\s*(?:a|an|the)?\s*([a-zA-Z][a-zA-Z0-9'’\-\s]{2,70})",
    re.IGNORECASE,
)
_BOOK_TITLE_RE = re.compile(
    r"\b(?:read|reading|finished|started)\s+(?:the\s+book\s+)?(?:called|titled)?\s*"
    r"([A-Z][A-Za-z0-9'’:\-\s]{2,80}|[\"“][^\"”]{2,80}[\"”])",
)
_TEST_TYPE_RE = re.compile(
    r"\b(?:take|takes|taking|took|taken|retake|retakes|retaking|retook|retaken|"
    r"completed|sat|sitting|passed|failed)\s+"
    r"((?:the\s+)?[A-Za-z][A-Za-z0-9'’\-\s]{2,80}?\btest)\b",
    re.IGNORECASE,
)
_NAMED_TEST_RE = re.compile(r"\b((?:the\s+)?(?:military\s+)?aptitude\s+test)\b", re.IGNORECASE)
_PLACE_ACTION_RE = re.compile(
    r"\b(?:went|go|traveled|travelled|moved|camped|visited|returned)\s+"
    r"(?:to|in|at|from)\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z0-9'’\-\s]{2,70})",
    re.IGNORECASE,
)
_REGISTERED_ACTIVITY_RE = re.compile(
    r"\b(?:signed\s+up|signing\s+up|registered|enrolled)\s+for\s+"
    r"(?:a|an|the)?\s*([A-Za-z][A-Za-z0-9'’\-\s]{2,70}?\b(?:class|course|workshop|lesson))\b",
    re.IGNORECASE,
)
_ACTIVITY_PURPOSE_RE = re.compile(
    r"\b("
    r"pottery(?:\s+class)?|running|reading|playing\s+(?:my\s+|the\s+)?violin|violin|"
    r"hiking|camping|swimming|painting"
    r")\b[^.!?]{0,120}\b(?:destress|de-stress|clear\s+my\s+mind|headspace|refresh(?:es)?\s+me|"
    r"therapy\s+for\s+me|express\s+myself|get\s+creative|reset|recharge)\b",
    re.IGNORECASE,
)
_INSTRUMENT_ACTION_RE = re.compile(
    r"\b(?:plays?|played|learned|learning|practices?|practiced)\s+"
    r"(?:the\s+)?([a-zA-Z][a-zA-Z0-9'’\-\s]{2,50})",
    re.IGNORECASE,
)
_SOURCE_SURFACE_TRAILING_RE = re.compile(
    r"\b(?:last|this|next)\s+(?:weekend|week|month|year|night|summer|winter|spring|fall|autumn)\b.*$"
    r"|\b(?:yesterday|today|tomorrow|recently|lately)\b.*$"
    r"|\b\d+\s+days?\s+ago\b.*$"
    r"|\s+(?:with|for|because|but|so|when|while|after|before|which|who|that|where|if|though)\b.*$",
    re.IGNORECASE,
)
_SOURCE_SURFACE_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the|this|that|my|our|another)\s+", re.IGNORECASE)
_STOP_LIST_ITEMS = {
    "a",
    "an",
    "and",
    "anything",
    "everything",
    "friday",
    "it",
    "monday",
    "nothing",
    "or",
    "saturday",
    "she",
    "something",
    "stuff",
    "sunday",
    "that",
    "the",
    "them",
    "these",
    "things",
    "this",
    "those",
    "thursday",
    "tuesday",
    "wednesday",
    "what",
    "yeah",
}
_TRAILING_CLAUSE_RE = re.compile(
    r"\b(?:because|but|so|when|while|after|before|which|who|that|where|if|though)\b.*$",
    re.IGNORECASE,
)


def raw_records_from_normalized(messages: Iterable[NormalizedMessage]) -> list[RawMessageRecord]:
    records: list[RawMessageRecord] = []
    for index, message in enumerate(messages):
        message_id = message.raw_message_id or message.source_ref or f"exchange-m{index:04d}"
        records.append(
            RawMessageRecord(
                id=str(message_id),
                sample_id="",
                dataset_name="",
                turn_index=message.turn_index,
                role=message.role,
                speaker_name=message.speaker_name,
                content=message.content,
                source_ref=message.source_ref,
                occurred_at=message.occurred_at,
                metadata_json=dict(message.metadata or {}),
            )
        )
    return records


def _candidate_key(candidate: MustPreserveCandidate) -> tuple[str, str, str | None]:
    return (candidate.surface.casefold(), candidate.category, candidate.relation)


def _dedupe_candidates(candidates: Iterable[MustPreserveCandidate]) -> list[MustPreserveCandidate]:
    merged: dict[tuple[str, str, str | None], MustPreserveCandidate] = {}
    for candidate in candidates:
        surface = collapse_whitespace(candidate.surface)
        if not surface:
            continue
        normalized = MustPreserveCandidate(
            surface=surface,
            category=candidate.category,
            source_message_ids=list(dict.fromkeys(candidate.source_message_ids)),
            relation=candidate.relation,
            confidence=candidate.confidence,
            source_refs=list(dict.fromkeys(candidate.source_refs)),
            rule=candidate.rule,
            raw_surface=collapse_whitespace(candidate.raw_surface or surface),
            canonical=candidate.canonical,
            event_action=candidate.event_action,
            temporal_expression=candidate.temporal_expression,
        )
        key = _candidate_key(normalized)
        previous = merged.get(key)
        if previous is None:
            merged[key] = normalized
            continue
        preferred_rule = previous.rule
        if previous.rule in {None, "exact_term_candidate", "facet_candidate"} and normalized.rule:
            preferred_rule = normalized.rule
        merged[key] = MustPreserveCandidate(
            surface=previous.surface,
            category=previous.category,
            source_message_ids=list(dict.fromkeys([*previous.source_message_ids, *normalized.source_message_ids])),
            relation=previous.relation,
            confidence=previous.confidence,
            source_refs=list(dict.fromkeys([*previous.source_refs, *normalized.source_refs])),
            rule=preferred_rule,
            raw_surface=previous.raw_surface or normalized.raw_surface,
            canonical=previous.canonical or normalized.canonical,
            event_action=previous.event_action or normalized.event_action,
            temporal_expression=previous.temporal_expression or normalized.temporal_expression,
        )
    return sorted(merged.values(), key=lambda item: (item.category, item.surface.casefold()))


def _clean_source_surface(
    value: str,
    *,
    max_tokens: int = 6,
    preserve_leading_article: bool = False,
) -> str | None:
    value = collapse_whitespace(value.strip(" -:,.!?\"'“”‘’"))
    value = _SOURCE_SURFACE_TRAILING_RE.sub("", value).strip(" -:,.!?\"'“”‘’")
    if not preserve_leading_article:
        value = _SOURCE_SURFACE_LEADING_ARTICLE_RE.sub("", value)
    value = _clean_list_item(value) or ""
    if not value:
        return None
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'’\-]*", value)
    if not tokens or len(tokens) > max_tokens:
        return None
    if all(token.casefold() in _STOP_LIST_ITEMS for token in tokens):
        return None
    return collapse_whitespace(value)


def _candidate_from_surface(
    *,
    surface: str | None,
    category: str,
    record: RawMessageRecord,
    rule: str,
    relation: str | None = None,
    raw_surface: str | None = None,
    canonical: str | None = None,
    event_action: str | None = None,
    temporal_expression: str | None = None,
) -> MustPreserveCandidate | None:
    if not surface:
        return None
    source_ids = [record.id] if record.id else []
    if not source_ids:
        return None
    source_refs = [str(record.source_ref or record.id)] if (record.source_ref or record.id) else []
    return MustPreserveCandidate(
        surface=surface,
        category=category,
        source_message_ids=source_ids,
        relation=relation,
        source_refs=source_refs,
        rule=rule,
        raw_surface=collapse_whitespace(raw_surface or surface),
        canonical=collapse_whitespace(canonical or "") or None,
        event_action=collapse_whitespace(event_action or "") or None,
        temporal_expression=collapse_whitespace(temporal_expression or "") or None,
    )


def _temporal_expression_from_text(text: str) -> str | None:
    match = _TEMPORAL_EXPRESSION_RE.search(text)
    if not match:
        return None
    return collapse_whitespace(match.group(1))


def _event_candidate(
    *,
    record: RawMessageRecord,
    surface: str | None,
    rule: str,
    action: str | None = None,
    temporal_expression: str | None = None,
    canonical: str | None = None,
    raw_surface: str | None = None,
) -> MustPreserveCandidate | None:
    cleaned_surface = _clean_source_surface(surface or "", max_tokens=8, preserve_leading_article=False)
    if not cleaned_surface:
        return None
    canonical_surface = collapse_whitespace(canonical or cleaned_surface)
    return _candidate_from_surface(
        surface=canonical_surface,
        category="event_object",
        record=record,
        rule=rule,
        relation="event_object",
        raw_surface=raw_surface or cleaned_surface,
        canonical=canonical_surface,
        event_action=action,
        temporal_expression=temporal_expression,
    )


def _clean_list_item(value: str) -> str | None:
    value = collapse_whitespace(value.strip(" -:,.!?\"'“”‘’"))
    value = _TRAILING_CLAUSE_RE.sub("", value).strip(" -:,.!?")
    value = re.sub(
        r"^.*\b(?:include|includes|included|are|were|is|was|involve|involves)\b\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^.*\b(?:enjoys?|likes?|loves?|does|plays?|practices?|participates?\s+in|partakes?\s+in)\b\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:and|or|also|including|such as)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+(?:and|or)$", "", value, flags=re.IGNORECASE)
    value = collapse_whitespace(value)
    if not value:
        return None
    lowered = value.casefold()
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'\-]*", lowered)
    if not tokens or len(tokens) > 5:
        return None
    if all(token in _STOP_LIST_ITEMS for token in tokens):
        return None
    if any(token in _STOP_LIST_ITEMS for token in tokens) and len(tokens) == 1:
        return None
    if len(value) < 3:
        return None
    return value


def _extract_list_items(text: str) -> list[tuple[str, str]]:
    spans: list[tuple[str, str]] = []
    spans.extend((match.group(1), text[max(match.start() - 40, 0) : match.end() + 40]) for match in _INVENTORY_TRIGGER_RE.finditer(text))
    spans.extend((match.group(1), text[max(match.start() - 40, 0) : match.end() + 40]) for match in _ACTIVITY_VERB_LIST_RE.finditer(text))
    # Also catch explicit short comma inventories even when the utterance does not contain an inventory cue.
    for sentence in re.split(r"[.!?]", text):
        if sentence.count(",") >= 2:
            spans.append((sentence, sentence))
    items: list[tuple[str, str]] = []
    for span, context in spans:
        context_lower = context.casefold()
        category = "list_item"
        if "activit" in context_lower or any(token in context_lower for token in ("enjoys", "likes", "loves", "partakes", "participates", "practices")):
            category = "activity"
        elif "book" in context_lower:
            category = "book_title"
        elif "recipe" in context_lower or "dish" in context_lower:
            category = "recipe"
        elif "instrument" in context_lower or "play" in context_lower:
            category = "instrument"
        elif "symbol" in context_lower:
            category = "symbol"
        elif any(token in context_lower for token in ("city", "cities", "place", "places", "visited", "travel", "camped", "camping")):
            category = "place"
        elif any(token in context_lower for token in ("event", "workshop", "program", "speech", "parade", "group")):
            category = "event_type"
        elif "paint" in context_lower:
            category = "painted_object"
        pieces = [_clean_list_item(piece) for piece in _LIST_SPLIT_RE.split(span)]
        cleaned = [piece for piece in pieces if piece]
        if len(cleaned) < 2:
            continue
        items.extend((cleaned_item, category) for cleaned_item in cleaned)
    return list(dict.fromkeys(items))


def _extract_research_topics(text: str) -> list[str]:
    values: list[str] = []
    for match in _RESEARCH_TOPIC_RE.finditer(text):
        raw_surface = re.split(
            r"\s+\b(?:and|but|while|because|so)\b\s+",
            match.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        surface = _clean_list_item(raw_surface)
        if not surface:
            continue
        surface = collapse_whitespace(surface.strip(" -:,.!?"))
        if surface:
            values.append(surface)
    return list(dict.fromkeys(values))


def _extract_count_candidates(text: str) -> list[str]:
    values: list[str] = []
    for match in _COUNT_CUE_RE.finditer(text):
        count = collapse_whitespace(match.group(1))
        qualifier = collapse_whitespace(match.group(2) or "")
        surface = collapse_whitespace(f"{count} {qualifier}".strip())
        if surface:
            values.append(surface)
    return list(dict.fromkeys(values))


def extract_must_preserve_candidates(
    source_messages: Iterable[RawMessageRecord | NormalizedMessage],
    *,
    entity_lexicon: dict[str, str] | None = None,
) -> list[MustPreserveCandidate]:
    entity_lexicon = entity_lexicon or {}
    records = [
        message
        if isinstance(message, RawMessageRecord)
        else raw_records_from_normalized([message])[0]
        for message in source_messages
    ]
    candidates: list[MustPreserveCandidate] = []
    collapsed_records = [
        (record, collapse_whitespace(record.content or ""))
        for record in records
    ]
    joined_text = " ".join(text for _, text in collapsed_records)
    school_speech_alias_records = [
        record
        for record, text in collapsed_records
        if text and (_SCHOOL_EVENT_RE.search(text) or _SCHOOL_SPEECH_SIGNAL_RE.search(text))
    ]
    school_speech_alias_enabled = bool(
        _SCHOOL_EVENT_RE.search(joined_text) and _SCHOOL_SPEECH_SIGNAL_RE.search(joined_text)
    )
    for record in records:
        source_ids = [record.id] if record.id else []
        if not source_ids:
            continue
        text = collapse_whitespace(record.content or "")
        if not text:
            continue
        exact_candidates = extract_exact_term_candidates_v1("", [record])
        cleaned_exact_terms = clean_exact_terms_v1(
            exact_candidates,
            speaker_entities=[record.speaker_name] if record.speaker_name else [],
        )
        source_type_by_surface = {
            collapse_whitespace(str(candidate.get("value") or "")).casefold(): str(candidate.get("source_type") or "exact_term")
            for candidate in exact_candidates
        }
        for surface in cleaned_exact_terms:
            candidates.append(
                MustPreserveCandidate(
                    surface=surface,
                    category=source_type_by_surface.get(surface.casefold(), "exact_term"),
                    source_message_ids=source_ids,
                    source_refs=[str(record.source_ref or record.id)] if (record.source_ref or record.id) else [],
                    rule="exact_term_candidate",
                )
            )
        raw_facets = extract_claim_facets_v1(text, [record], entity_lexicon)
        for facet in clean_facet_records_v1(raw_facets, speaker_entities=[record.speaker_name] if record.speaker_name else []):
            relation = str(facet.get("relation") or "").strip()
            value = collapse_whitespace(str(facet.get("value") or ""))
            if relation and value:
                candidates.append(
                    MustPreserveCandidate(
                        surface=value,
                        category=relation,
                        source_message_ids=source_ids,
                        relation=relation,
                        source_refs=[str(record.source_ref or record.id)] if (record.source_ref or record.id) else [],
                        rule="facet_candidate",
                    )
                )
        for topic in _extract_research_topics(text):
            candidates.append(
                MustPreserveCandidate(
                    surface=topic,
                    category="research_topic",
                    source_message_ids=source_ids,
                    relation="research_topic",
                    source_refs=[str(record.source_ref or record.id)] if (record.source_ref or record.id) else [],
                    rule="research_topic_pattern",
                    raw_surface=topic,
                )
            )
        for pattern, rule in (
            (_PREFERENCE_EXCITEMENT_RE, "preference_excitement_pattern"),
            (_PREFERENCE_VERB_RE, "preference_verb_pattern"),
            (_FAVORITE_VALUE_RE, "favorite_value_pattern"),
        ):
            for match in pattern.finditer(text):
                raw_surface = re.split(
                    r"\s+\b(?:and|but|while|because|so)\b\s+",
                    match.group(1),
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0]
                cleaned_surface = _clean_source_surface(raw_surface, max_tokens=7)
                candidate = _candidate_from_surface(
                    surface=cleaned_surface,
                    category="preference_item",
                    record=record,
                    rule=rule,
                    relation="preference_item",
                    raw_surface=collapse_whitespace(raw_surface.strip(" -:,.!?\"'“”‘’")),
                )
                if candidate is not None:
                    candidates.append(candidate)
        for match in _PAINTED_OBJECT_RE.finditer(text):
            candidate = _candidate_from_surface(
                surface=_clean_source_surface(match.group(1)),
                category="painted_object",
                record=record,
                rule="painted_object_action_pattern",
            )
            if candidate is not None:
                candidates.append(candidate)
        for match in _PICTURE_OF_RE.finditer(text):
            candidate = _candidate_from_surface(
                surface=_clean_source_surface(match.group(1)),
                category="painted_object",
                record=record,
                rule="picture_of_object_pattern",
            )
            if candidate is not None:
                candidates.append(candidate)
        for match in _FAVORITE_FOOD_RE.finditer(text):
            candidate = _candidate_from_surface(
                surface=_clean_source_surface(match.group(1)),
                category="food",
                record=record,
                rule="favorite_food_pattern",
            )
            if candidate is not None:
                candidates.append(candidate)
        for match in _REGISTERED_ACTIVITY_RE.finditer(text):
            raw_surface = collapse_whitespace(match.group(1).strip(" -:,.!?\"'“”‘’"))
            cleaned_surface = _clean_source_surface(raw_surface, max_tokens=6)
            candidate = _candidate_from_surface(
                surface=cleaned_surface,
                category="activity",
                record=record,
                rule="registered_activity_pattern",
                relation="activity",
                raw_surface=raw_surface,
            )
            if candidate is not None:
                candidates.append(candidate)
                base_surface = re.sub(
                    r"\s+(?:class|course|workshop|lesson)\b$",
                    "",
                    candidate.surface,
                    flags=re.IGNORECASE,
                ).strip()
                if base_surface and base_surface.casefold() != candidate.surface.casefold():
                    base_candidate = _candidate_from_surface(
                        surface=base_surface,
                        category="activity",
                        record=record,
                        rule="registered_activity_base_pattern",
                        relation="activity",
                        raw_surface=base_surface,
                    )
                    if base_candidate is not None:
                        candidates.append(base_candidate)
        for match in _ACTIVITY_PURPOSE_RE.finditer(text):
            raw_surface = collapse_whitespace(match.group(1).strip(" -:,.!?\"'“”‘’"))
            cleaned_surface = _clean_source_surface(raw_surface, max_tokens=6)
            candidate = _candidate_from_surface(
                surface=cleaned_surface,
                category="activity",
                record=record,
                rule="activity_purpose_pattern",
                relation="activity",
                raw_surface=raw_surface,
            )
            if candidate is not None:
                candidates.append(candidate)
        for match in _BOOK_TITLE_RE.finditer(text):
            raw_surface = collapse_whitespace(match.group(1).strip(" -:,.!?\"'“”‘’"))
            cleaned_surface = _clean_source_surface(raw_surface, max_tokens=8, preserve_leading_article=True)
            candidate = _candidate_from_surface(
                surface=cleaned_surface,
                category="book_title",
                record=record,
                rule="book_title_action_pattern",
                raw_surface=cleaned_surface,
            )
            if candidate is not None:
                candidates.append(candidate)
        for pattern, rule in ((_TEST_TYPE_RE, "test_type_action_pattern"), (_NAMED_TEST_RE, "named_test_pattern")):
            for match in pattern.finditer(text):
                raw_surface = collapse_whitespace(match.group(1).strip(" -:,.!?\"'“”‘’"))
                cleaned_surface = _clean_source_surface(raw_surface, max_tokens=8, preserve_leading_article=True)
                candidate = _candidate_from_surface(
                    surface=cleaned_surface,
                    category="test_type",
                    record=record,
                    rule=rule,
                    relation="test_type",
                    raw_surface=cleaned_surface,
                )
                if candidate is not None:
                    candidates.append(candidate)
        for match in _PLACE_ACTION_RE.finditer(text):
            candidate = _candidate_from_surface(
                surface=_clean_source_surface(match.group(1)),
                category="place",
                record=record,
                rule="place_action_pattern",
            )
            if candidate is not None:
                candidates.append(candidate)
        for match in _INSTRUMENT_ACTION_RE.finditer(text):
            candidate = _candidate_from_surface(
                surface=_clean_source_surface(match.group(1)),
                category="instrument",
                record=record,
                rule="instrument_action_pattern",
            )
            if candidate is not None:
                candidates.append(candidate)
        temporal_expression = _temporal_expression_from_text(text)
        for match in _NETWORKING_EVENT_RE.finditer(text):
            candidate = _event_candidate(
                record=record,
                surface=match.group("object"),
                rule="networking_event_action_pattern",
                action=match.group("action"),
                temporal_expression=temporal_expression,
                raw_surface=match.group("object"),
            )
            if candidate is not None:
                candidates.append(candidate)
        for match in _EVENT_OBJECT_ACTION_RE.finditer(text):
            candidate = _event_candidate(
                record=record,
                surface=match.group("object"),
                rule="event_object_action_pattern",
                action=match.group("action"),
                temporal_expression=temporal_expression,
                raw_surface=match.group("object"),
            )
            if candidate is not None:
                candidates.append(candidate)
        roadtrip_match = _ROADTRIP_RE.search(text)
        if roadtrip_match:
            candidate = _event_candidate(
                record=record,
                surface="roadtrip",
                rule="roadtrip_temporal_pattern",
                action="roadtrip",
                temporal_expression=_temporal_expression_from_text(roadtrip_match.group("trailing")) or temporal_expression,
                raw_surface=roadtrip_match.group("object"),
                canonical="roadtrip",
            )
            if candidate is not None:
                candidates.append(candidate)
        if _CAR_ACCIDENT_RE.search(text):
            candidate = _event_candidate(
                record=record,
                surface="car accident",
                rule="car_accident_visual_or_text_pattern",
                action="accident",
                temporal_expression=temporal_expression,
                raw_surface="car accident",
                canonical="car accident",
            )
            if candidate is not None:
                candidates.append(candidate)
        for match in _EVENT_PHRASE_RE.finditer(text):
            surface = collapse_whitespace(match.group(1))
            if surface:
                candidates.append(
                    MustPreserveCandidate(
                        surface=surface,
                        category="event_type",
                        source_message_ids=source_ids,
                        relation="event_type",
                        source_refs=[str(record.source_ref or record.id)] if (record.source_ref or record.id) else [],
                        rule="event_phrase_pattern",
                        raw_surface=surface,
                        canonical=surface,
                        temporal_expression=temporal_expression,
                    )
                )
                if _MENTORSHIP_PROGRAM_RE.fullmatch(surface) and surface.casefold() != "mentoring program":
                    candidates.append(
                        MustPreserveCandidate(
                            surface="mentoring program",
                            category="event_type",
                            source_message_ids=source_ids,
                            relation="event_type",
                            source_refs=[str(record.source_ref or record.id)] if (record.source_ref or record.id) else [],
                            rule="mentoring_program_canonical_alias",
                            raw_surface=surface,
                            canonical="mentoring program",
                            temporal_expression=temporal_expression,
                        )
                    )
        if _SCHOOL_EVENT_RE.search(text):
            candidates.append(
                MustPreserveCandidate(
                    surface="school event",
                    category="event_type",
                    source_message_ids=source_ids,
                    relation="event_type",
                    source_refs=[str(record.source_ref or record.id)] if (record.source_ref or record.id) else [],
                    rule="school_event_pattern",
                    raw_surface="school event",
                    canonical="school event",
                    temporal_expression=temporal_expression,
                )
            )
        for count_value in _extract_count_candidates(text):
            candidates.append(
                MustPreserveCandidate(
                    surface=count_value,
                    category="count",
                    source_message_ids=source_ids,
                    source_refs=[str(record.source_ref or record.id)] if (record.source_ref or record.id) else [],
                    rule="count_cue_pattern",
                )
            )
        for item, category in _extract_list_items(text):
            candidates.append(
                MustPreserveCandidate(
                    surface=item,
                    category=category,
                    source_message_ids=source_ids,
                    source_refs=[str(record.source_ref or record.id)] if (record.source_ref or record.id) else [],
                    rule="list_item_pattern",
                )
            )
    if school_speech_alias_enabled and school_speech_alias_records:
        alias_source_ids = [
            record.id for record in school_speech_alias_records if record.id
        ]
        alias_source_refs = [
            str(record.source_ref or record.id)
            for record in school_speech_alias_records
            if record.source_ref or record.id
        ]
        raw_surface = "school event / giving my talk"
        temporal_expression = _temporal_expression_from_text(joined_text)
        for surface, rule in (
            ("school speech", "school_speech_event_alias_pattern"),
            ("school talk", "school_talk_event_alias_pattern"),
        ):
            candidates.append(
                MustPreserveCandidate(
                    surface=surface,
                    category="event_type",
                    source_message_ids=list(dict.fromkeys(alias_source_ids)),
                    relation="event_type",
                    source_refs=list(dict.fromkeys(alias_source_refs)),
                    rule=rule,
                    raw_surface=raw_surface,
                    canonical=surface,
                    event_action="talk",
                    temporal_expression=temporal_expression,
                )
            )
    return _dedupe_candidates(candidates)


def _normalized(value: str) -> str:
    return collapse_whitespace(str(value or "")).casefold()


def _surface_matches(needle: str, haystack: str) -> bool:
    needle_norm = _normalized(needle)
    haystack_norm = _normalized(haystack)
    return bool(needle_norm and haystack_norm and (needle_norm in haystack_norm or haystack_norm in needle_norm))


def _claim_signal_rows(
    claims: Iterable[MemoryClaim],
    source_messages_by_id: dict[str, RawMessageRecord],
    entity_lexicon: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for claim in claims:
        source_messages = [
            source_messages_by_id[source_id]
            for source_id in claim.source_message_ids
            if source_id in source_messages_by_id
        ]
        exact_terms = clean_exact_terms_v1(
            extract_exact_term_candidates_v1(claim.text, source_messages),
            speaker_entities=[message.speaker_name for message in source_messages if message.speaker_name],
        )
        facets = clean_facet_records_v1(
            extract_claim_facets_v1(claim.text, source_messages, entity_lexicon),
            speaker_entities=[message.speaker_name for message in source_messages if message.speaker_name],
        )
        rows.append(
            {
                "claim": claim,
                "text": claim.text,
                "source_message_ids": list(claim.source_message_ids),
                "exact_terms": exact_terms,
                "facets": facets,
            }
        )
    return rows


def audit_claim_preservation(
    *,
    candidates: Iterable[MustPreserveCandidate],
    claims: Iterable[MemoryClaim],
    source_messages: Iterable[RawMessageRecord | NormalizedMessage],
    entity_lexicon: dict[str, str] | None = None,
) -> ClaimCoverageResult:
    entity_lexicon = entity_lexicon or {}
    normalized_records = [
        message
        if isinstance(message, RawMessageRecord)
        else raw_records_from_normalized([message])[0]
        for message in source_messages
    ]
    source_messages_by_id = {message.id: message for message in normalized_records}
    signal_rows = _claim_signal_rows(claims, source_messages_by_id, entity_lexicon)
    missing: list[MustPreserveCandidate] = []
    weak_source_links: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for candidate in _dedupe_candidates(candidates):
        candidate_sources = set(candidate.source_message_ids)
        covering_sources: set[str] = set()
        covered = False
        for row in signal_rows:
            claim = row["claim"]
            claim_sources = set(row["source_message_ids"])  # type: ignore[arg-type]
            claim_text_match = _surface_matches(candidate.surface, str(row["text"]))
            exact_match = any(_surface_matches(candidate.surface, term) for term in list(row["exact_terms"]))
            facet_match = False
            for facet in list(row["facets"]):
                relation = str(facet.get("relation") or "").strip()
                value = str(facet.get("value") or "").strip()
                value_span = str(facet.get("value_span") or "").strip()
                if candidate.relation and relation != candidate.relation:
                    continue
                if _surface_matches(candidate.surface, value) or _surface_matches(candidate.surface, value_span):
                    facet_match = True
                elif candidate.relation and facet_value_key(relation, value) == facet_value_key(candidate.relation, candidate.surface):
                    facet_match = True
                elif candidate.relation and normalize_facet_value(value) == normalize_facet_value(candidate.surface):
                    facet_match = True
            if claim_text_match or exact_match or facet_match:
                covered = True
                covering_sources.update(claim_sources)
                diagnostics.append(
                    {
                        "candidate": candidate.to_metadata(),
                        "claim_id": getattr(claim, "claim_id", None),
                        "covered_by": "claim_text" if claim_text_match else "exact_or_facet",
                        "claim_source_message_ids": sorted(claim_sources),
                    }
                )
        if not covered:
            missing.append(candidate)
            continue
        if candidate_sources and not candidate_sources <= covering_sources:
            weak_source_links.append(
                {
                    "candidate": candidate.to_metadata(),
                    "expected_source_message_ids": sorted(candidate_sources),
                    "covering_source_message_ids": sorted(covering_sources),
                }
            )
    return ClaimCoverageResult(
        covered=not missing and not weak_source_links,
        missing_candidates=missing,
        weak_source_links=weak_source_links,
        diagnostics=diagnostics,
    )
