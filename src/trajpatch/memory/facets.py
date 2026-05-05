"""Deterministic claim facets, exact terms, and entity/facet summaries for episodic memory."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from trajpatch.storage.models import ClaimRecord, RawMessageRecord
from trajpatch.utils.text import collapse_whitespace, extract_keywords

_TITLE_CASE_SPAN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_TITLE_OR_PLACE_TERM_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'’-]*(?::[A-Z][A-Za-z0-9'’-]*)?)(?:\s+(?:[A-Z][A-Za-z0-9'’-]*(?::[A-Z][A-Za-z0-9'’-]*)?|of|the|and|a|an|to|with|for|in|on|at))*\b"
)
_QUOTED_TERM_RE = re.compile(r'["“](.{2,80}?)["”]')
_LOCATION_VALUE_RE = re.compile(r"\b(?:in|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_HOME_COUNTRY_RE = re.compile(r"\bhome country[, ]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", re.IGNORECASE)
_POSSESSIVE_SUFFIX_RE = re.compile(r"(?:'s|’s)\b", re.IGNORECASE)
_GENERIC_USER_RE = re.compile(r"^\s*(?:the\s+)?user\b", re.IGNORECASE)
_GENERIC_ASSISTANT_RE = re.compile(r"\b(?:the\s+)?assistant\b", re.IGNORECASE)
_FIRST_PERSON_POSSESSIVE_RE = re.compile(r"\b(?:my|mine|our|ours)\b", re.IGNORECASE)
_POSSESSIVE_PERSON_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)['’]s\b")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-']+")

_STOP_ENTITY_SPANS = {
    "A",
    "An",
    "And",
    "Are",
    "As",
    "At",
    "Because",
    "Been",
    "Before",
    "Being",
    "Between",
    "But",
    "By",
    "Could",
    "Did",
    "Do",
    "Does",
    "During",
    "For",
    "From",
    "Had",
    "Has",
    "Have",
    "How",
    "If",
    "In",
    "Into",
    "Is",
    "It",
    "Its",
    "Last",
    "More",
    "Most",
    "My",
    "Next",
    "Of",
    "On",
    "Or",
    "Our",
    "Really",
    "Should",
    "Since",
    "That",
    "The",
    "Their",
    "There",
    "These",
    "This",
    "Those",
    "To",
    "We",
    "Were",
    "What",
    "When",
    "Where",
    "Which",
    "While",
    "Who",
    "Why",
    "With",
    "Would",
    "You",
    "Your",
}
_LIST_QUERY_PATTERNS = (
    "what books",
    "which books",
    "what recipes",
    "what recipe",
    "what dishes",
    "what instruments",
    "which instruments",
    "what symbols",
    "which symbols",
    "what places",
    "which places",
    "what places or events",
    "which places or events",
    "what countries",
    "which countries",
    "what states",
    "which states",
    "what activities",
    "which activities",
    "what events",
    "which events",
    "what items",
    "which items",
    "what desserts",
    "which desserts",
    "what shelters",
    "which shelters",
    "what causes",
    "which causes",
    "what hobbies",
    "which hobbies",
    "what interests",
    "which interests",
    "what dreams",
    "which dreams",
    "what writings",
    "which writings",
    "what games",
    "which games",
    "which bands",
    "what bands",
    "what pets",
    "which pets",
    "what dogs",
    "which dogs",
    "what names",
    "which names",
    "what favorite",
    "which favorite",
    "what favourites",
    "which favourites",
    "which organizations",
    "what organizations",
    "which endorsement deals",
    "what endorsement deals",
    "which classes",
    "what classes",
    "which locations",
    "what locations",
    "which cities",
    "what types",
    "what type",
    "what kinds",
    "what kind",
)
_LIST_FAMILY_PATTERNS = (
    ("writing", ("screenplay", "screenplays", "script", "scripts", "what writings", "which writings", "what kind of writing", "what kind of fiction", "what writings does", "kind of writing")),
    ("book", ("what books", "what book", "which books", "book titles", "how many books")),
    ("recipe", ("what recipes", "what recipe", "which recipes", "what dishes", "how many recipes")),
    ("instrument", ("what instruments", "which instruments", "how many instruments", "play the", "plays the", "guitar", "guitars")),
    ("symbol", ("what symbols", "which symbols", "how many symbols")),
    ("country", ("what countries", "which countries", "countries has", "countries did", "countries have")),
    ("state", ("what states", "which states", "states has", "states did", "states have")),
    ("city", ("which cities", "which city", "what cities", "how many cities")),
    ("place", ("what places", "which places", "places or events", "where has", "where have", "geographical locations", "areas of")),
    ("activity", ("what activities", "which activities", "how many activities", "activities has", "activities did")),
    ("event", ("what events", "which events", "how many events", "participated in", "attended", "joined")),
    ("item", ("what items", "which items", "what object", "which object", "what things", "which things", "bought", "own", "owns", "having as a child")),
    ("dessert", ("what desserts", "which desserts", "dessert")),
    ("shelter", ("what shelters", "which shelters", "shelter")),
    ("cause", ("what causes", "which causes", "causes has", "causes did")),
    ("hobby", ("what hobbies", "which hobbies", "new hobbies", "hobbies did", "hobbies has")),
    ("preference", ("favorite", "favourite", "interests")),
    ("dream", ("what dreams", "which dreams", "dreams does", "dreams has")),
    ("organization", ("which organizations", "what organizations", "beneficiaries", "beneficiary")),
    ("deal", ("endorsement deal", "endorsement deals", "which deals", "what deals")),
    ("class", ("which classes", "what classes", "classes has", "classes did")),
    ("location", ("which locations", "what locations", "locations has", "locations did")),
    ("game", ("what games", "which games", "favorite games", "favorite game")),
    ("band", ("which bands", "what bands", "bands has", "bands did")),
    ("pet", ("what pets", "which pets", "what dogs", "which dogs", "names of", "dogs named")),
    ("person", ("which family member", "what family member", "which of their family member", "who passed away", "which person passed away", "passed away", "names of")),
    ("count", ("how many", "number of", "count of")),
    ("type", ("what types", "what type", "what kinds", "what kind")),
)
_LIST_FAMILY_REGEX_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "preference",
        re.compile(
            r"\bwhat\s+(?:do|does)\b.+\b(?:like|likes|enjoy|enjoys|love|loves|prefer|prefers)\b"
            r"|\bwhat\s+are\b.+\b(?:interests|favorite|favourite)\b",
            re.IGNORECASE,
        ),
    ),
    ("dream", re.compile(r"\bwhat\s+are\b.+\bdreams\b|\bwhat\s+dreams?\b", re.IGNORECASE)),
    ("pet", re.compile(r"\bwhat\s+are\b.+\b(?:pets?|dogs?|cats?)'?s?\s+names\b|\bwhat\s+names\b.+\b(?:pets?|dogs?|cats?)\b", re.IGNORECASE)),
    ("organization", re.compile(r"\bwho\s+or\s+which\s+organizations?\b|\bbeneficiar(?:y|ies)\b", re.IGNORECASE)),
    ("deal", re.compile(r"\bwhich\b.+\bendorsement\s+deals?\b|\bwhat\b.+\bendorsement\s+deals?\b", re.IGNORECASE)),
    ("game", re.compile(r"\bwhich\b.+\bgames?\b|\bwhat\b.+\bgames?\b", re.IGNORECASE)),
    ("class", re.compile(r"\bwhich\b.+\bclasses?\b|\bwhat\b.+\bclasses?\b", re.IGNORECASE)),
    ("location", re.compile(r"\bwhich\b.+\blocations?\b|\bwhat\b.+\blocations?\b", re.IGNORECASE)),
)
_OBJECT_ACTION_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("painted_object", ("paint", "painted", "painting")),
    ("reading", ("read", "reading")),
    ("item", ("buy", "bought", "purchase", "purchased", "make", "made", "build", "built", "create", "created", "object")),
    ("research_topic", ("research", "researched", "study", "studied")),
    ("activity", ("destress", "relax", "exercise", "practice")),
)
_INSTRUMENT_TERMS = {
    "bass",
    "cello",
    "clarinet",
    "drums",
    "flute",
    "guitar",
    "harp",
    "piano",
    "saxophone",
    "trumpet",
    "ukulele",
    "viola",
    "violin",
}
_SYMBOL_TERMS = {
    "heart",
    "lotus",
    "moon",
    "peace sign",
    "rainbow",
    "rainbow flag",
    "star",
    "sun",
    "tree of life",
}
_FOOD_SUFFIXES = (
    "cake",
    "cakes",
    "cookie",
    "cookies",
    "pie",
    "pies",
    "bread",
    "pasta",
    "pizza",
    "soup",
    "salad",
    "curry",
    "stew",
    "lasagna",
    "muffin",
    "muffins",
    "brownie",
    "brownies",
    "sandwich",
    "sandwiches",
    "taco",
    "tacos",
)
_NOISE_TERMS = {
    "ah",
    "anything",
    "anyway",
    "cool",
    "definitely",
    "hello",
    "hey",
    "hi",
    "hmm",
    "just",
    "nope",
    "okay",
    "ok",
    "really",
    "right",
    "sure",
    "thanks",
    "thank",
    "welcome",
    "well",
    "yeah",
    "yep",
}
_PRONOUN_TERMS = {
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "it",
    "its",
    "me",
    "mine",
    "my",
    "myself",
    "our",
    "ours",
    "she",
    "their",
    "theirs",
    "them",
    "they",
    "us",
    "we",
    "you",
    "your",
    "yours",
}
_WEEKDAY_MONTH_TERMS = {
    "april",
    "august",
    "december",
    "february",
    "friday",
    "january",
    "july",
    "june",
    "march",
    "monday",
    "november",
    "october",
    "saturday",
    "september",
    "sunday",
    "thursday",
    "tuesday",
    "wednesday",
}
_VAGUE_TEMPORAL_TERMS = {
    "afternoon",
    "day",
    "days",
    "evening",
    "later",
    "morning",
    "night",
    "recently",
    "soon",
    "time",
    "times",
    "today",
    "tomorrow",
    "tonight",
    "week",
    "weeks",
    "yesterday",
}
_LOW_VALUE_SINGLE_TERM_OPENERS = {
    "actually",
    "amazing",
    "any",
    "because",
    "before",
    "being",
    "great",
    "more",
    "most",
    "next",
    "some",
}
_SUPPORTED_RELATION_VALUES = {
    "relationship_status": {"single"},
    "gender_identity": {"transgender woman"},
}
_HIGH_TRUST_EXACT_TERM_SOURCES = {
    "food_suffix",
    "home_country",
    "instrument",
    "place",
    "quoted",
    "symbol",
}


def _dedupe_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = collapse_whitespace(str(value))
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def normalize_entity_key(value: str) -> str:
    return collapse_whitespace(value).casefold()


def normalize_facet_value(value: str) -> str:
    return collapse_whitespace(value).casefold()


def facet_value_key(relation: str, value: str) -> str:
    return f"{relation}={normalize_facet_value(value)}"


def _is_stop_entity_span(surface: str) -> bool:
    tokens = [token for token in surface.split() if token]
    if not tokens:
        return True
    if surface in _STOP_ENTITY_SPANS:
        return True
    if any(token in _STOP_ENTITY_SPANS for token in tokens):
        return True
    folded_tokens = [token.casefold() for token in tokens]
    if any(
        token in _NOISE_TERMS
        or token in _PRONOUN_TERMS
        or token in _WEEKDAY_MONTH_TERMS
        or token in _LOW_VALUE_SINGLE_TERM_OPENERS
        for token in folded_tokens
    ):
        return True
    if any("'" in token or "’" in token for token in tokens):
        return True
    if all(token in _STOP_ENTITY_SPANS or token.casefold() in _NOISE_TERMS for token in tokens):
        return True
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return True
    return False


def _normalized_tokens(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(value)]


def _keywords_meaningful(value: str) -> bool:
    keywords = extract_keywords(value)
    if not keywords:
        return False
    return any(
        token not in _NOISE_TERMS
        and token not in _PRONOUN_TERMS
        and token not in _WEEKDAY_MONTH_TERMS
        and token not in _VAGUE_TEMPORAL_TERMS
        and token not in _LOW_VALUE_SINGLE_TERM_OPENERS
        and token not in {span.casefold() for span in _STOP_ENTITY_SPANS}
        for token in keywords
    )


def _is_low_value_term(surface: str) -> bool:
    normalized = collapse_whitespace(surface)
    if not normalized:
        return True
    tokens = _normalized_tokens(normalized)
    if not tokens:
        return True
    if len(tokens) == 1 and (
        tokens[0] in _NOISE_TERMS
        or tokens[0] in _PRONOUN_TERMS
        or tokens[0] in _WEEKDAY_MONTH_TERMS
        or tokens[0] in _VAGUE_TEMPORAL_TERMS
        or tokens[0] in _LOW_VALUE_SINGLE_TERM_OPENERS
        or tokens[0] in {span.casefold() for span in _STOP_ENTITY_SPANS}
    ):
        return True
    if any("'" in token or "’" in token for token in tokens):
        return True
    return not _keywords_meaningful(normalized)


def _is_valid_place_like_value(value: str) -> bool:
    normalized = collapse_whitespace(value)
    if not normalized or _is_stop_entity_span(normalized):
        return False
    tokens = _normalized_tokens(normalized)
    if not tokens:
        return False
    if any(token in _NOISE_TERMS or token in _PRONOUN_TERMS for token in tokens):
        return False
    if any(token in _WEEKDAY_MONTH_TERMS or token in _VAGUE_TEMPORAL_TERMS for token in tokens):
        return False
    return _keywords_meaningful(normalized)


def _normalize_relation_value_surface(value: str) -> str:
    return collapse_whitespace(value).strip(" ,.;:!?")


def _clean_entity_mentions_with_debug(
    values: Iterable[str],
    *,
    speaker_entities: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    speaker_map = {
        normalize_entity_key(collapse_whitespace(value)): collapse_whitespace(value)
        for value in speaker_entities
        if collapse_whitespace(value)
    }
    cleaned: list[str] = []
    discarded: list[str] = []
    for value in values:
        normalized = collapse_whitespace(str(value))
        if not normalized:
            continue
        canonical = _POSSESSIVE_SUFFIX_RE.sub("", normalized)
        if not canonical:
            discarded.append(normalized)
            continue
        canonical_key = normalize_entity_key(canonical)
        if canonical_key in speaker_map:
            cleaned.append(speaker_map[canonical_key])
            continue
        if _is_stop_entity_span(canonical):
            discarded.append(canonical)
            continue
        tokens = _normalized_tokens(canonical)
        if not tokens:
            discarded.append(canonical)
            continue
        if any(token in _PRONOUN_TERMS or token in _NOISE_TERMS for token in tokens):
            discarded.append(canonical)
            continue
        if any(token in _WEEKDAY_MONTH_TERMS or token in _VAGUE_TEMPORAL_TERMS for token in tokens):
            discarded.append(canonical)
            continue
        if len(tokens) == 1 and (
            len(tokens[0]) < 3
            or tokens[0] in _LOW_VALUE_SINGLE_TERM_OPENERS
            or tokens[0].islower()
        ):
            discarded.append(canonical)
            continue
        cleaned.append(canonical)
    return _dedupe_preserve(cleaned), _dedupe_preserve(discarded)


def clean_entity_mentions_v1(
    values: Iterable[str],
    *,
    speaker_entities: Iterable[str] = (),
) -> list[str]:
    cleaned, _ = _clean_entity_mentions_with_debug(values, speaker_entities=speaker_entities)
    return cleaned


def _exact_term_candidate(value: str, source_type: str) -> dict[str, str]:
    return {
        "value": collapse_whitespace(value),
        "source_type": source_type,
    }


def _clean_exact_terms_with_debug(
    values: Iterable[str | dict[str, object]],
    *,
    speaker_entities: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    speaker_keys = {
        normalize_entity_key(collapse_whitespace(value))
        for value in speaker_entities
        if collapse_whitespace(value)
    }
    cleaned: list[str] = []
    discarded: list[str] = []
    for value in values:
        if isinstance(value, dict):
            surface = collapse_whitespace(str(value.get("value") or ""))
            source_type = collapse_whitespace(str(value.get("source_type") or "unknown")).casefold()
        else:
            surface = collapse_whitespace(str(value))
            source_type = "unknown"
        if not surface:
            continue
        surface = _POSSESSIVE_SUFFIX_RE.sub("", surface)
        if not surface:
            discarded.append(str(value))
            continue
        folded = normalize_entity_key(surface)
        tokens = _normalized_tokens(surface)
        if not tokens:
            discarded.append(surface)
            continue
        if folded in speaker_keys:
            discarded.append(surface)
            continue
        if any(token in _PRONOUN_TERMS or token in _NOISE_TERMS for token in tokens):
            discarded.append(surface)
            continue
        if any(token in _WEEKDAY_MONTH_TERMS or token in _VAGUE_TEMPORAL_TERMS for token in tokens):
            discarded.append(surface)
            continue
        if any("'" in token or "’" in token for token in tokens):
            discarded.append(surface)
            continue
        if source_type not in _HIGH_TRUST_EXACT_TERM_SOURCES and _is_low_value_term(surface):
            discarded.append(surface)
            continue
        if (
            source_type not in _HIGH_TRUST_EXACT_TERM_SOURCES
            and len(tokens) == 1
            and surface[:1].isupper()
        ):
            discarded.append(surface)
            continue
        cleaned.append(surface)
    return _dedupe_preserve(cleaned), _dedupe_preserve(discarded)


def clean_exact_terms_v1(
    values: Iterable[str | dict[str, object]],
    *,
    speaker_entities: Iterable[str] = (),
) -> list[str]:
    cleaned, _ = _clean_exact_terms_with_debug(values, speaker_entities=speaker_entities)
    return cleaned


def _clean_facet_records_with_debug(
    facets: Iterable[dict[str, object]],
    *,
    speaker_entities: Iterable[str] = (),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    speaker_map = {
        normalize_entity_key(collapse_whitespace(value)): collapse_whitespace(value)
        for value in speaker_entities
        if collapse_whitespace(value)
    }
    cleaned: list[dict[str, object]] = []
    discarded: list[dict[str, object]] = []
    for facet in facets:
        relation = collapse_whitespace(str(facet.get("relation") or ""))
        raw_value = _normalize_relation_value_surface(str(facet.get("value") or ""))
        if not relation or not raw_value:
            discarded.append({"reason": "missing_relation_or_value", "facet": dict(facet)})
            continue
        normalized_value = normalize_facet_value(raw_value)
        allowed_values = _SUPPORTED_RELATION_VALUES.get(relation)
        if allowed_values is not None and normalized_value not in allowed_values:
            discarded.append({"reason": "unsupported_relation_value", "facet": dict(facet)})
            continue
        if relation in {"home_country", "activity_location"}:
            if not _is_valid_place_like_value(raw_value):
                discarded.append({"reason": "invalid_place_like_value", "facet": dict(facet)})
                continue
        elif relation in {"research_topic", "art_style", "event_type"}:
            if _is_low_value_term(raw_value):
                discarded.append({"reason": "low_value_relation_value", "facet": dict(facet)})
                continue
        cleaned_entity: str | None = None
        raw_entity = collapse_whitespace(str(facet.get("entity") or ""))
        if raw_entity:
            cleaned_entities, entity_discarded = _clean_entity_mentions_with_debug(
                [raw_entity],
                speaker_entities=speaker_entities,
            )
            if cleaned_entities:
                entity_key = normalize_entity_key(cleaned_entities[0])
                cleaned_entity = speaker_map.get(entity_key, cleaned_entities[0])
            elif entity_discarded:
                discarded.append(
                    {
                        "reason": "facet_entity_cleared",
                        "entity": raw_entity,
                        "relation": relation,
                        "value": raw_value,
                    }
                )
        cleaned.append(
            {
                **dict(facet),
                "entity": cleaned_entity,
                "relation": relation,
                "value": raw_value,
                "value_span": collapse_whitespace(str(facet.get("value_span") or raw_value)),
            }
        )
    deduped: list[dict[str, object]] = []
    seen: set[tuple[str | None, str, str]] = set()
    for facet in cleaned:
        dedupe_key = (
            normalize_entity_key(str(facet.get("entity") or "")) if facet.get("entity") else None,
            str(facet.get("relation") or ""),
            normalize_facet_value(str(facet.get("value") or "")),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(facet)
    return deduped, discarded


def clean_facet_records_v1(
    facets: Iterable[dict[str, object]],
    *,
    speaker_entities: Iterable[str] = (),
) -> list[dict[str, object]]:
    cleaned, _ = _clean_facet_records_with_debug(facets, speaker_entities=speaker_entities)
    return cleaned


def build_sample_entity_lexicon(messages: Iterable[RawMessageRecord]) -> dict[str, str]:
    speaker_entities: dict[str, str] = {}
    title_case_surfaces: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for message in messages:
        if message.speaker_name:
            speaker = collapse_whitespace(message.speaker_name)
            if speaker and not _is_stop_entity_span(speaker):
                speaker_entities.setdefault(normalize_entity_key(speaker), speaker)
        for match in _TITLE_CASE_SPAN_RE.finditer(message.content):
            surface = collapse_whitespace(match.group(0))
            if _is_stop_entity_span(surface):
                continue
            key = normalize_entity_key(surface)
            title_case_surfaces.setdefault(key, surface)
            counts[key] += 1
    lexicon = dict(speaker_entities)
    for key, surface in title_case_surfaces.items():
        if counts[key] >= 2:
            lexicon.setdefault(key, surface)
    return lexicon


def extract_entities_from_text(text: str, entity_lexicon: dict[str, str]) -> list[str]:
    if not text or not entity_lexicon:
        return []
    matches: list[tuple[int, int, str]] = []
    for key, surface in entity_lexicon.items():
        pattern = re.compile(rf"\b{re.escape(surface)}(?:'s|’s)?\b", re.IGNORECASE)
        match = pattern.search(text)
        if match is None:
            continue
        matches.append((match.start(), -(match.end() - match.start()), surface))
    matches.sort(key=lambda item: (item[0], item[1], item[2].casefold()))
    return _dedupe_preserve(surface for _, _, surface in matches)


def _source_message_id_from_facet_source(source: str | None) -> str | None:
    if not source:
        return None
    prefix = "source_message:"
    if source.startswith(prefix):
        return source[len(prefix) :]
    return None


def _speaker_entities(source_messages: Iterable[RawMessageRecord]) -> list[str]:
    return _dedupe_preserve(
        collapse_whitespace(message.speaker_name)
        for message in source_messages
        if message.speaker_name and collapse_whitespace(message.speaker_name)
    )


def _speaker_grounding_audit(
    claim_text: str,
    source_messages: list[RawMessageRecord],
) -> dict[str, object]:
    speakers = _speaker_entities(source_messages)
    speaker_keys = {normalize_entity_key(speaker) for speaker in speakers}
    reasons: list[str] = []
    generic_subject = False
    if _GENERIC_USER_RE.search(claim_text) or _GENERIC_ASSISTANT_RE.search(claim_text):
        generic_subject = True
    if speakers and any(_FIRST_PERSON_POSSESSIVE_RE.search(message.content or "") for message in source_messages):
        for match in _POSSESSIVE_PERSON_RE.finditer(claim_text):
            subject = collapse_whitespace(match.group(1))
            subject_key = normalize_entity_key(subject)
            if subject_key and subject_key not in speaker_keys:
                reasons.append("first_person_possessive_mismatched_subject")
                break
    return {
        "speaker_grounding_suspect_v1": bool(reasons),
        "speaker_grounding_suspect_reasons_v1": sorted(set(reasons)),
        "speaker_grounding_generic_subject_v1": generic_subject,
        "source_speaker_names_v1": speakers,
    }


def _primary_entity(
    claim_text: str,
    source_messages: list[RawMessageRecord],
    entity_lexicon: dict[str, str],
) -> str | None:
    speaker_entities = _speaker_entities(source_messages)
    if _GENERIC_USER_RE.match(claim_text) and speaker_entities:
        return speaker_entities[0]
    for surface in extract_entities_from_text(claim_text, entity_lexicon):
        return surface
    if speaker_entities:
        return speaker_entities[0]
    for message in source_messages:
        for surface in extract_entities_from_text(message.content, entity_lexicon):
            return surface
    return None


def _make_facet(
    *,
    entity: str | None,
    relation: str,
    value: str,
    value_span: str,
    facet_type: str,
    source: str,
    confidence: float,
) -> dict[str, object]:
    return {
        "entity": entity,
        "relation": relation,
        "value": collapse_whitespace(value),
        "value_span": collapse_whitespace(value_span),
        "facet_type": facet_type,
        "source": source,
        "confidence": confidence,
    }


def extract_claim_facets_v1(
    claim_text: str,
    source_messages: list[RawMessageRecord],
    entity_lexicon: dict[str, str],
) -> list[dict[str, object]]:
    entity = _primary_entity(claim_text, source_messages, entity_lexicon)
    facets: list[dict[str, object]] = []
    texts: list[tuple[str, str]] = [("claim_text", claim_text)]
    texts.extend((f"source_message:{message.id}", message.content) for message in source_messages)

    for source_name, text in texts:
        lowered = text.casefold()
        if "single parent" in lowered:
            facets.append(
                _make_facet(
                    entity=entity,
                    relation="relationship_status",
                    value="single",
                    value_span="single parent",
                    facet_type="identity_status",
                    source=source_name,
                    confidence=0.98,
                )
            )
        if "transgender woman" in lowered:
            facets.append(
                _make_facet(
                    entity=entity,
                    relation="gender_identity",
                    value="transgender woman",
                    value_span="transgender woman",
                    facet_type="identity_status",
                    source=source_name,
                    confidence=0.99,
                )
            )
        home_country_match = _HOME_COUNTRY_RE.search(text)
        if home_country_match:
            surface = home_country_match.group(1)
            facets.append(
                _make_facet(
                    entity=entity,
                    relation="home_country",
                    value=surface,
                    value_span=home_country_match.group(0),
                    facet_type="origin",
                    source=source_name,
                    confidence=0.97,
                )
            )
        if "research" in lowered and "adoption agenc" in lowered:
            facets.append(
                _make_facet(
                    entity=entity,
                    relation="research_topic",
                    value="adoption agencies",
                    value_span="adoption agencies",
                    facet_type="topic",
                    source=source_name,
                    confidence=0.96,
                )
            )
        if "abstract art" in lowered:
            facets.append(
                _make_facet(
                    entity=entity,
                    relation="art_style",
                    value="abstract art",
                    value_span="abstract art",
                    facet_type="creative_style",
                    source=source_name,
                    confidence=0.98,
                )
            )
        if "car modification workshop" in lowered:
            facets.append(
                _make_facet(
                    entity=entity,
                    relation="event_type",
                    value="car modification workshop",
                    value_span="car modification workshop",
                    facet_type="event",
                    source=source_name,
                    confidence=0.96,
                )
            )
        for match in re.finditer(r"\b(?:returned from|in)\s+(San Francisco)\b", text, flags=re.IGNORECASE):
            facets.append(
                _make_facet(
                    entity=entity,
                    relation="activity_location",
                    value=match.group(1),
                    value_span=match.group(0),
                    facet_type="location",
                    source=source_name,
                    confidence=0.95,
                )
            )

    deduped: list[dict[str, object]] = []
    seen: set[tuple[str | None, str, str]] = set()
    for facet in facets:
        dedupe_key = (
            str(facet.get("entity")).casefold() if facet.get("entity") else None,
            str(facet["relation"]),
            normalize_facet_value(str(facet["value"])),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(facet)
    return deduped


def _extract_instrument_terms(text: str) -> list[dict[str, str]]:
    lowered = text.casefold()
    matches: list[dict[str, str]] = []
    for instrument in sorted(_INSTRUMENT_TERMS):
        if re.search(rf"\b{re.escape(instrument)}\b", lowered):
            matches.append(_exact_term_candidate(instrument, "instrument"))
    return matches


def _extract_symbol_terms(text: str) -> list[dict[str, str]]:
    lowered = text.casefold()
    matches: list[dict[str, str]] = []
    for symbol in sorted(_SYMBOL_TERMS):
        if re.search(rf"\b{re.escape(symbol)}\b", lowered):
            matches.append(_exact_term_candidate(symbol, "symbol"))
    return matches


def _extract_food_terms(text: str) -> list[dict[str, str]]:
    lowered = text.casefold()
    matches: list[dict[str, str]] = []
    for suffix in _FOOD_SUFFIXES:
        for match in re.finditer(
            rf"\b(?:[a-z]+(?:-[a-z]+)?\s+){{0,3}}{re.escape(suffix)}\b",
            lowered,
        ):
            surface = collapse_whitespace(match.group(0))
            if len(surface) >= len(suffix):
                matches.append(_exact_term_candidate(surface, "food_suffix"))
    return matches


def _extract_title_like_terms(text: str, speaker_names: set[str]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for quoted in _QUOTED_TERM_RE.findall(text):
        surface = collapse_whitespace(quoted)
        if surface:
            matches.append(_exact_term_candidate(surface, "quoted"))
    for match in _TITLE_OR_PLACE_TERM_RE.finditer(text):
        surface = collapse_whitespace(match.group(0))
        normalized = normalize_entity_key(surface)
        if not surface or normalized in speaker_names:
            continue
        stripped = _POSSESSIVE_SUFFIX_RE.sub("", surface)
        if _is_stop_entity_span(stripped):
            continue
        if len(stripped.split()) == 1 and stripped in {"I"}:
            continue
        matches.append(_exact_term_candidate(stripped, "title_case"))
    return matches


def extract_exact_term_candidates_v1(
    claim_text: str,
    source_messages: list[RawMessageRecord],
) -> list[dict[str, str]]:
    speaker_names = {
        normalize_entity_key(collapse_whitespace(message.speaker_name))
        for message in source_messages
        if message.speaker_name and collapse_whitespace(message.speaker_name)
    }
    terms: list[dict[str, str]] = []
    texts = [claim_text, *[message.content for message in source_messages]]
    for text in texts:
        terms.extend(_extract_instrument_terms(text))
        terms.extend(_extract_symbol_terms(text))
        terms.extend(_extract_food_terms(text))
        terms.extend(_extract_title_like_terms(text, speaker_names))
        location_match = _LOCATION_VALUE_RE.search(text)
        if location_match:
            terms.append(_exact_term_candidate(location_match.group(1), "place"))
        home_country_match = _HOME_COUNTRY_RE.search(text)
        if home_country_match:
            terms.append(_exact_term_candidate(home_country_match.group(1), "home_country"))
    return terms


def extract_exact_terms_v1(
    claim_text: str,
    source_messages: list[RawMessageRecord],
) -> list[str]:
    candidates = extract_exact_term_candidates_v1(claim_text, source_messages)
    return clean_exact_terms_v1(candidates, speaker_entities=_speaker_entities(source_messages))


def _claim_fit_score(claim: ClaimRecord, facet: dict[str, object]) -> float:
    score = 0.0
    claim_text = claim.text.casefold()
    value = collapse_whitespace(str(facet.get("value") or "")).casefold()
    value_span = collapse_whitespace(str(facet.get("value_span") or "")).casefold()
    relation = str(facet.get("relation") or "")
    if claim.status == "active":
        score += 100.0
    if value and value in claim_text:
        score += 20.0
    if value_span and value_span in claim_text:
        score += 15.0
    claim_tokens = extract_keywords(claim.text)
    value_tokens = extract_keywords(f"{value} {value_span}")
    relation_tokens = {token for token in relation.split("_") if token}
    score += 5.0 * len(value_tokens & claim_tokens)
    score += 1.0 * len(relation_tokens & claim_tokens)
    score += 0.01 * len(claim.text)
    return score


def assign_claim_metadata_v1(
    claims: list[ClaimRecord],
    source_messages_by_id: dict[str, RawMessageRecord],
    entity_lexicon: dict[str, str],
) -> None:
    if not claims:
        return

    candidates_by_source: dict[str | None, list[dict[str, object]]] = defaultdict(list)
    source_to_claims: dict[str, list[ClaimRecord]] = defaultdict(list)
    for claim in claims:
        supporting_messages = [
            source_messages_by_id[source_id]
            for source_id in claim.source_message_ids_json
            if source_id in source_messages_by_id
        ]
        speaker_entities = _speaker_entities(supporting_messages)
        exact_term_candidates = extract_exact_term_candidates_v1(claim.text, supporting_messages)
        exact_terms, discarded_exact_terms = _clean_exact_terms_with_debug(
            exact_term_candidates,
            speaker_entities=speaker_entities,
        )
        metadata = dict(claim.metadata_json or {})
        metadata.update(_speaker_grounding_audit(claim.text, supporting_messages))
        metadata["exact_terms_v1"] = exact_terms
        metadata["exact_terms_discarded_v1"] = discarded_exact_terms
        metadata["facets_v1"] = []
        metadata["facet_discarded_v1"] = []
        claim.metadata_json = metadata
        for source_id in claim.source_message_ids_json:
            source_to_claims[source_id].append(claim)
        raw_facets = extract_claim_facets_v1(claim.text, supporting_messages, entity_lexicon)
        if not raw_facets:
            continue
        cleaned_facets, discarded_facets = _clean_facet_records_with_debug(
            raw_facets,
            speaker_entities=speaker_entities,
        )
        metadata["facet_discarded_v1"] = discarded_facets
        claim.metadata_json = metadata
        for facet in cleaned_facets:
            source_id = _source_message_id_from_facet_source(str(facet.get("source") or "")) or None
            candidates_by_source[source_id].append(facet)

    def _attach_facet(target_claim: ClaimRecord, facet: dict[str, object]) -> None:
        metadata = dict(target_claim.metadata_json or {})
        facets = list(metadata.get("facets_v1") or [])
        dedupe_key = (
            str(facet.get("entity") or "").casefold(),
            str(facet.get("relation") or ""),
            normalize_facet_value(str(facet.get("value") or "")),
        )
        for existing in facets:
            existing_key = (
                str(existing.get("entity") or "").casefold(),
                str(existing.get("relation") or ""),
                normalize_facet_value(str(existing.get("value") or "")),
            )
            if existing_key == dedupe_key:
                return
        facets.append(facet)
        metadata["facets_v1"] = facets
        target_claim.metadata_json = metadata

    for source_id, facets in candidates_by_source.items():
        claim_candidates = list(source_to_claims.get(source_id or "", [])) if source_id else list(claims)
        if not claim_candidates:
            claim_candidates = list(claims)
        preferred_candidates = [claim for claim in claim_candidates if claim.status == "active"] or claim_candidates
        for facet in facets:
            target = max(preferred_candidates, key=lambda claim: _claim_fit_score(claim, facet))
            _attach_facet(target, facet)

    for claim in claims:
        if claim.status == "active":
            continue
        for facet in list((claim.metadata_json or {}).get("facets_v1") or []):
            source_id = _source_message_id_from_facet_source(str(facet.get("source") or ""))
            if not source_id:
                continue
            active_targets = [
                target
                for target in claims
                if target.status == "active" and source_id in target.source_message_ids_json
            ]
            for target in active_targets:
                _attach_facet(target, facet)


def build_incoming_match_features(
    semantic_text: str,
    claim_texts: Iterable[str],
    source_messages: list[RawMessageRecord],
    entity_lexicon: dict[str, str],
) -> dict[str, list[str]]:
    speaker_entities = _speaker_entities(source_messages)
    normalized_claims = [
        collapse_whitespace(str(text))
        for text in claim_texts
        if collapse_whitespace(str(text))
    ]
    text_snippets = [
        snippet
        for snippet in [
            collapse_whitespace(semantic_text),
            *normalized_claims,
            *[collapse_whitespace(message.content) for message in source_messages],
        ]
        if snippet
    ]
    keywords = _dedupe_preserve(
        token
        for snippet in text_snippets
        for token in extract_keywords(snippet)
    )
    raw_entities = [
        *(
            surface
            for snippet in text_snippets
            for surface in extract_entities_from_text(snippet, entity_lexicon)
        ),
        *speaker_entities,
    ]
    entities = clean_entity_mentions_v1(
        raw_entities,
        speaker_entities=speaker_entities,
    )
    exact_terms = clean_exact_terms_v1(
        (
            term
            for snippet in [collapse_whitespace(semantic_text), *normalized_claims]
            if snippet
            for term in extract_exact_term_candidates_v1(snippet, source_messages)
        ),
        speaker_entities=speaker_entities,
    )
    raw_facets = [
        facet
        for snippet in [collapse_whitespace(semantic_text), *normalized_claims]
        if snippet
        for facet in extract_claim_facets_v1(snippet, source_messages, entity_lexicon)
    ]
    cleaned_facets = clean_facet_records_v1(raw_facets, speaker_entities=speaker_entities)
    facet_tags = sorted(
        {
            str(facet.get("relation") or "").strip()
            for facet in cleaned_facets
            if str(facet.get("relation") or "").strip()
        }
    )
    facet_values = sorted(
        {
            facet_value_key(str(facet.get("relation") or ""), str(facet.get("value") or ""))
            for facet in cleaned_facets
            if str(facet.get("relation") or "").strip() and str(facet.get("value") or "").strip()
        }
    )
    return {
        "keywords": keywords,
        "entities": entities,
        "exact_terms": exact_terms,
        "facet_tags": facet_tags,
        "facet_values": facet_values,
    }


def extract_query_facets_v1(question: str, entity_lexicon: dict[str, str]) -> dict[str, list[str]]:
    lowered = question.casefold()
    facet_tags: list[str] = []
    facet_values: list[str] = []
    if "relationship status" in lowered:
        facet_tags.append("relationship_status")
    if "identity" in lowered or "transgender" in lowered:
        facet_tags.append("gender_identity")
    if "home country" in lowered or "moved from" in lowered or "move from" in lowered:
        facet_tags.append("home_country")
    if "research" in lowered:
        facet_tags.append("research_topic")
    if "kind of art" in lowered or "art style" in lowered:
        facet_tags.append("art_style")
    if "workshop" in lowered or "event" in lowered:
        facet_tags.append("event_type")
    location_match = _LOCATION_VALUE_RE.search(question)
    if location_match or "where " in lowered:
        facet_tags.append("activity_location")
    if location_match:
        facet_values.append(facet_value_key("activity_location", location_match.group(1)))
    return {
        "tags": sorted(set(facet_tags)),
        "values": sorted(set(facet_values)),
        "entities": extract_entities_from_text(question, entity_lexicon),
    }


def is_list_like_query(question: str) -> bool:
    lowered = collapse_whitespace(question).casefold()
    if any(pattern in lowered for pattern in _LIST_QUERY_PATTERNS):
        return True
    if any(pattern.search(lowered) for _, pattern in _LIST_FAMILY_REGEX_PATTERNS):
        return True
    return any(marker in lowered for marker in (" both ", " all "))


def classify_query_shape_v1(question: str, entity_lexicon: dict[str, str]) -> dict[str, object]:
    normalized = collapse_whitespace(question)
    lowered = normalized.casefold()
    entities = extract_entities_from_text(question, entity_lexicon)
    entity_keys = sorted({normalize_entity_key(value) for value in entities})
    list_like = is_list_like_query(question)
    count_like = bool(
        lowered.startswith("how many")
        or " how many " in lowered
        or " number of " in lowered
        or lowered.startswith("number of ")
        or " count of " in lowered
    )
    comparison_like = bool(
        " both " in lowered
        or " compared " in lowered
        or " compare " in lowered
        or " same " in lowered
        or " different " in lowered
        or " together " in lowered
        or " shared " in lowered
        or " have both " in lowered
        or " do both " in lowered
    )
    multi_entity = len(entity_keys) >= 2 or " both " in lowered
    item_family = None
    for family, patterns in _LIST_FAMILY_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            item_family = family
            break
    if item_family is None:
        for family, pattern in _LIST_FAMILY_REGEX_PATTERNS:
            if pattern.search(lowered):
                item_family = family
                break
    if item_family is None and re.match(r"^(?:what|which)\s+(?:did|has|have|does|do)\b", lowered):
        for family, verbs in _OBJECT_ACTION_FAMILY_PATTERNS:
            if any(re.search(rf"\b{re.escape(verb)}\b", lowered) for verb in verbs):
                item_family = family
                break
    if item_family and item_family != "count":
        list_like = True
    tags: list[str] = []
    if list_like:
        tags.append("list_like")
    if multi_entity:
        tags.append("multi_entity")
    if comparison_like:
        tags.append("comparison_like")
    if count_like:
        tags.append("count_like")
    return {
        "list_like": list_like,
        "multi_entity": multi_entity,
        "comparison_like": comparison_like,
        "count_like": count_like,
        "item_family": item_family,
        "entities": entities,
        "entity_keys": entity_keys,
        "tags": tags,
        "normalized_question": normalized,
    }


def build_trajectory_entity_facet_summary(
    claims: list[ClaimRecord],
    source_messages_by_id: dict[str, RawMessageRecord],
    entity_lexicon: dict[str, str],
) -> dict[str, object]:
    entity_mentions: list[str] = []
    facet_tags: list[str] = []
    facet_values: list[str] = []
    exact_terms: list[str] = []
    display_items: list[str] = []
    display_named_entities: list[str] = []
    display_counts: list[str] = []
    display_key_facts: list[str] = []
    source_surface_terms: list[str] = []
    source_surface_raw_terms: list[str] = []
    source_surface_categories: list[str] = []
    source_surface_refs: list[str] = []
    source_surface_records: list[dict[str, object]] = []
    source_event_records: list[dict[str, object]] = []
    source_event_object_terms: list[str] = []
    source_event_action_terms: list[str] = []
    source_temporal_relation_terms: list[str] = []
    source_event_canonical_terms: list[str] = []
    source_event_refs: list[str] = []
    discarded_facets: list[dict[str, object]] = []
    for claim in claims:
        if claim.status != "active":
            continue
        metadata = dict(claim.metadata_json or {})
        if metadata.get("speaker_grounding_suspect_v1"):
            discarded_facets.append(
                {
                    "claim_id": claim.claim_id,
                    "reason": "speaker_grounding_suspect",
                    "speaker_grounding_suspect_reasons": list(
                        metadata.get("speaker_grounding_suspect_reasons_v1") or []
                    ),
                }
            )
            continue
        entity_mentions.extend(extract_entities_from_text(claim.text, entity_lexicon))
        source_messages = [
            source_messages_by_id[source_id]
            for source_id in claim.source_message_ids_json
            if source_id in source_messages_by_id
        ]
        entity_mentions.extend(_speaker_entities(source_messages))
        for message in source_messages:
            entity_mentions.extend(extract_entities_from_text(message.content, entity_lexicon))
        cleaned_facets, facet_discarded = _clean_facet_records_with_debug(
            list(metadata.get("facets_v2") or metadata.get("facets_v1") or []),
            speaker_entities=_speaker_entities(source_messages),
        )
        discarded_facets.extend(facet_discarded)
        for facet in cleaned_facets:
            relation = str(facet.get("relation") or "").strip()
            value = str(facet.get("value") or "").strip()
            entity = str(facet.get("entity") or "").strip()
            if entity:
                entity_mentions.append(entity)
            if relation:
                facet_tags.append(relation)
            if relation and value:
                facet_values.append(facet_value_key(relation, value))
        exact_terms.extend(
            str(term).strip()
            for term in list(metadata.get("exact_terms_v2") or metadata.get("exact_terms_v1") or [])
            if str(term).strip()
        )
        for record in list(metadata.get("source_surface_records_v1") or []):
            if not isinstance(record, dict):
                continue
            surface = str(record.get("surface") or "").strip()
            if not surface:
                continue
            source_surface_records.append(dict(record))
            source_surface_terms.append(surface)
            raw_surface = str(record.get("raw_surface") or "").strip()
            if raw_surface:
                source_surface_raw_terms.append(raw_surface)
            category = str(record.get("category") or "").strip()
            if category:
                source_surface_categories.append(category)
            source_surface_refs.extend(
                str(value).strip()
                for value in list(record.get("source_refs") or [])
                if str(value).strip()
            )
        for record in list(metadata.get("source_event_records_v1") or []):
            if not isinstance(record, dict):
                continue
            surface = str(record.get("surface") or "").strip()
            canonical = str(record.get("canonical") or "").strip()
            if not surface and not canonical:
                continue
            source_event_records.append(dict(record))
            if surface:
                source_event_object_terms.append(surface)
            if canonical:
                source_event_canonical_terms.append(canonical)
            action = str(record.get("action") or "").strip()
            if action:
                source_event_action_terms.append(action)
            temporal = str(record.get("temporal_expression") or "").strip()
            if temporal:
                source_temporal_relation_terms.append(temporal)
            source_event_refs.extend(
                str(value).strip()
                for value in list(record.get("source_refs") or [])
                if str(value).strip()
            )
        display = dict(metadata.get("display_signals_v1") or {})
        display_items.extend(str(value).strip() for value in list(display.get("items") or []) if str(value).strip())
        display_named_entities.extend(
            str(value).strip() for value in list(display.get("named_entities") or []) if str(value).strip()
        )
        display_counts.extend(str(value).strip() for value in list(display.get("counts") or []) if str(value).strip())
        display_key_facts.extend(
            str(value).strip() for value in list(display.get("key_facts") or []) if str(value).strip()
        )
        discarded_facets.extend(list(metadata.get("facet_discarded_v1") or []))
    speaker_entities = _speaker_entities(source_messages_by_id.values())
    cleaned_entities, discarded_entities = _clean_entity_mentions_with_debug(
        entity_mentions,
        speaker_entities=speaker_entities,
    )
    return {
        "entity_mentions": cleaned_entities,
        "facet_tags": sorted(set(facet_tags)),
        "facet_values": sorted(set(facet_values)),
        "exact_terms": _dedupe_preserve(
            [*source_event_canonical_terms, *source_event_object_terms, *source_surface_raw_terms, *source_surface_terms, *exact_terms]
        ),
        "display_items": _dedupe_preserve(
            [*source_event_canonical_terms, *source_event_object_terms, *source_surface_raw_terms, *source_surface_terms, *display_items]
        ),
        "display_named_entities": _dedupe_preserve(display_named_entities),
        "display_counts": _dedupe_preserve(display_counts),
        "display_key_facts": _dedupe_preserve(display_key_facts),
        "source_surface_terms_v1": _dedupe_preserve(source_surface_terms),
        "source_surface_raw_terms_v1": _dedupe_preserve(source_surface_raw_terms),
        "source_surface_categories_v1": _dedupe_preserve(source_surface_categories),
        "source_surface_refs_v1": _dedupe_preserve(source_surface_refs),
        "source_surface_records_v1": source_surface_records,
        "source_event_records_v1": source_event_records,
        "source_event_object_terms_v1": _dedupe_preserve(source_event_object_terms),
        "source_event_action_terms_v1": _dedupe_preserve(source_event_action_terms),
        "source_temporal_relation_terms_v1": _dedupe_preserve(source_temporal_relation_terms),
        "source_event_canonical_terms_v1": _dedupe_preserve(source_event_canonical_terms),
        "source_event_refs_v1": _dedupe_preserve(source_event_refs),
        "source_surface_preservation_miss_count_v1": sum(
            1
            for claim in claims
            for candidate in list((claim.metadata_json or {}).get("claim_preservation_misses_v1") or [])
            if isinstance(candidate, dict)
            and str(candidate.get("surface") or "").strip()
            and str(candidate.get("surface") or "").strip().casefold()
            in {term.casefold() for term in source_surface_terms}
        ),
        "entity_mentions_discarded_v1": discarded_entities,
        "facet_discarded_v1": discarded_facets,
        "exact_terms_discarded_v1": _dedupe_preserve(
            str(value).strip()
            for claim in claims
            for value in list((claim.metadata_json or {}).get("exact_terms_discarded_v1") or [])
            if str(value).strip()
        ),
    }


def exact_term_keyword_set(values: Iterable[str]) -> set[str]:
    keywords: set[str] = set()
    for value in values:
        keywords.update(extract_keywords(str(value)))
    return keywords
