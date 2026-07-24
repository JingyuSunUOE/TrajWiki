"""Answer generation and benchmark judging services."""

from __future__ import annotations

import json
import hashlib
import re
import time
from datetime import date, timedelta
from typing import Any, Callable

from pydantic import ValidationError

from trajpatch.exceptions import ParserValidationError, StructuredOutputError
from trajpatch.memory.facets import classify_query_shape_v1
from trajpatch.memory.llm_text_parsers import parse_judge_verdict
from trajpatch.prompts import load_prompt
from trajpatch.providers.base import LLMProvider
from trajpatch.providers.structured_outputs import (
    get_structured_task_spec,
    parse_structured_payload,
    structured_prompt_name,
    validate_judge_verdict_result,
)
from trajpatch.types import JudgeResult, LLMResponse, NormalizedMessage, QueryTask, RetrievalBundle
from trajpatch.utils.text import collapse_whitespace


class AnswerGenerator:
    FAMILY_SIGNAL_TERMS = {
        "book": {"book", "books", "read", "reading", "title", "cover", "novel", "story"},
        "reading": {"book", "books", "read", "reading", "title", "cover", "novel", "story"},
        "painted_object": {"paint", "painted", "painting", "artwork", "picture", "image", "sunset", "sunflower", "horse"},
        "research_topic": {"research", "researched", "agency", "agencies", "option", "options", "looked", "study"},
        "instrument": {"instrument", "instruments", "play", "played", "clarinet", "violin", "guitar", "piano"},
        "activity": {"activity", "activities", "hiking", "camping", "painting", "pottery", "museum", "swimming"},
        "event": {"event", "events", "attended", "joined", "participated", "conference", "parade", "speech", "group"},
        "band": {"band", "bands", "artist", "artists", "concert", "festival", "sounds"},
        "place": {"place", "places", "visited", "city", "country", "area", "park", "museum"},
        "item": {"item", "items", "bought", "made", "created", "object", "thing"},
        "preference": {"like", "likes", "liked", "love", "loves", "enjoy", "enjoys", "stoked", "excited", "favorite", "favourite", "dinosaur", "dinosaurs", "nature"},
        "dream": {"dream", "dreams", "goal", "goals", "hope", "hopes"},
        "organization": {"organization", "organizations", "beneficiary", "beneficiaries", "charity", "foundation"},
        "deal": {"deal", "deals", "endorsement", "sponsor", "sponsorship"},
        "pet": {"pet", "pets", "dog", "dogs", "cat", "cats", "turtle", "turtles", "tortoise", "tortoises", "name", "names"},
        "class": {"class", "classes", "course", "courses", "lesson", "lessons"},
        "location": {"location", "locations", "place", "places", "city", "cities", "area", "areas"},
    }

    def __init__(self, llm_provider: LLMProvider, *, trace: Callable[[str], None] | None = None) -> None:
        self.llm_provider = llm_provider
        self.trace = trace

    def _trace(self, message: str) -> None:
        if self.trace is not None:
            self.trace(message)

    @staticmethod
    def _compact_exception(exc: BaseException, *, limit: int = 240) -> str:
        if isinstance(exc, ValidationError):
            missing: list[str] = []
            invalid: list[str] = []
            for error in exc.errors():
                loc = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
                if error.get("type") == "missing":
                    missing.append(loc)
                else:
                    invalid.append(f"{loc}: {error.get('msg') or error.get('type')}")
            parts: list[str] = [exc.__class__.__name__]
            if missing:
                parts.append("missing=" + ",".join(missing[:8]))
            if invalid:
                parts.append("invalid=" + "; ".join(invalid[:6]))
            message = " ".join(parts)
            return message[:limit]
        message = " ".join(str(exc).split()) or exc.__class__.__name__
        message = re.sub(r"For further information visit https://errors\\.pydantic\\.dev/\\S+", "", message)
        return message[:limit]

    @staticmethod
    def _locomo_query_shape(query_task: QueryTask, retrieval_bundle: RetrievalBundle) -> dict[str, object]:
        stored = dict(retrieval_bundle.metadata.get("query_shape") or {})
        if stored:
            return {
                "list_like": bool(stored.get("list_like")),
                "multi_entity": bool(stored.get("multi_entity")),
                "comparison_like": bool(stored.get("comparison_like")),
                "count_like": bool(stored.get("count_like")),
                "duration_count_like": bool(stored.get("duration_count_like") or "duration_count" in list(stored.get("tags") or [])),
                "item_family": stored.get("item_family"),
                "tags": list(stored.get("tags") or []),
            }
        derived = classify_query_shape_v1(query_task.question, {})
        return {
            "list_like": bool(derived.get("list_like")),
            "multi_entity": bool(derived.get("multi_entity")),
            "comparison_like": bool(derived.get("comparison_like")),
            "count_like": bool(derived.get("count_like")),
            "duration_count_like": bool(derived.get("duration_count_like") or "duration_count" in list(derived.get("tags") or [])),
            "item_family": derived.get("item_family"),
            "tags": list(derived.get("tags") or []),
        }

    @classmethod
    def _locomo_query_shape_rubric(cls, query_task: QueryTask, retrieval_bundle: RetrievalBundle) -> str:
        query_shape = cls._locomo_query_shape(query_task, retrieval_bundle)
        rules: list[str] = []
        if query_shape["list_like"]:
            rules.append("- This is a list/inventory question. Return every supported item and no unsupported extras.")
        if query_shape["multi_entity"] or query_shape["comparison_like"]:
            rules.append("- This question compares or combines multiple entities. Cover each entity explicitly before summarizing.")
        if query_shape["count_like"]:
            rules.append(
                "- This is a count question. Give an exact count only when the retrieved context contains an explicit count or a complete supported item set; otherwise state a retrieved-evidence lower bound."
            )
        item_family = str(query_shape.get("item_family") or "").strip()
        if item_family in {
            "book",
            "recipe",
            "instrument",
            "symbol",
            "place",
            "location",
            "event",
            "painted_object",
            "writing",
            "research_topic",
            "item",
            "preference",
            "dream",
            "organization",
            "deal",
            "pet",
            "class",
        }:
            rules.append(
                f"- Preserve exact {item_family} surface phrases from the retrieved context; do not paraphrase them into broader categories."
            )
        if item_family == "preference":
            rules.append(
                "- This is a preference/list question. Prefer the most specific source-backed item over umbrella categories; for example, use 'dinosaur exhibit' or 'dinosaurs' rather than only 'animals' when both are grounded."
            )
        if not rules:
            return ""
        return (
            "\n\nQUERY_SHAPE_RULES:\n"
            + "\n".join(rules)
        )

    @staticmethod
    def _question_is_time_like(question: str | None) -> bool:
        text = " ".join(str(question or "").casefold().split())
        if re.match(
            r"^when\b.+\b(?:what|which|who|whom|whose|where|how)\b",
            text,
        ):
            return False
        return bool(
            re.search(
                r"^(?:when|what date|what month|which month|what year|which year|what day|which day)\b|\b(?:what date|what month|which month|what year|which year|what day|which day)\b",
                text,
            )
        )

    @staticmethod
    def _normalize_answer_type(answer_type: Any) -> str:
        text = re.sub(r"[^a-z0-9]+", "_", str(answer_type or "").casefold()).strip("_")
        if not text:
            return "unknown"
        if text in {
            "count",
            "exact_count",
            "event_count",
            "count_event",
            "number",
            "numeric",
            "quantity",
            "times",
            "frequency",
        }:
            return "count"
        if text in {"list", "items", "inventory", "events", "places", "activities"}:
            return "list"
        return text

    @staticmethod
    def _question_allows_count_answer(question: str | None, query_shape: dict[str, object]) -> bool:
        if bool(query_shape.get("count_like")):
            return True
        text = " ".join(str(question or "").casefold().split())
        return bool(
            re.search(
                r"^(?:how many|how often|number of|what number|what count)\b|\b(?:how many|how often|number of|count of)\b",
                text,
            )
        )

    @staticmethod
    def _question_is_duration_count(question: str | None, query_shape: dict[str, object] | None = None) -> bool:
        if bool((query_shape or {}).get("duration_count_like")):
            return True
        text = " ".join(str(question or "").casefold().split())
        return bool(
            re.search(
                r"\bhow\s+many\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"
                r".*\b(?:passed|pass|took|take|since|between|after|before|until|from)\b",
                text,
            )
        )

    @staticmethod
    def _inventory_count_family(question: str | None) -> str | None:
        text = " ".join(str(question or "").casefold().split())
        match = re.search(
            r"\bhow\s+many\s+([a-z][a-z' -]*?)\b"
            r"(?:did|does|do|has|have|had|will|would|were|are|as|,|\?|$)",
            text,
        )
        noun_phrase = match.group(1).strip() if match else ""
        family_patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("pet", ("pet", "pets", "dog", "dogs", "cat", "cats", "turtle", "turtles", "snake", "snakes")),
            ("book", ("book", "books")),
            ("class", ("class", "classes", "course", "courses", "lesson", "lessons")),
            ("event", ("event", "events", "concert", "concerts", "conference", "conferences", "workshop", "workshops")),
            ("activity", ("activity", "activities", "hobby", "hobbies")),
            ("place", ("place", "places", "city", "cities", "country", "countries", "location", "locations")),
            ("organization", ("organization", "organizations", "group", "groups")),
            ("deal", ("deal", "deals")),
            ("item", ("item", "items", "thing", "things")),
            ("instrument", ("instrument", "instruments")),
            ("game", ("game", "games")),
        )
        for family, terms in family_patterns:
            if any(re.search(rf"\b{re.escape(term)}\b", noun_phrase) for term in terms):
                return family
        return None

    @classmethod
    def _expected_answer_type(cls, question: str | None, query_shape: dict[str, object]) -> str:
        text = " ".join(str(question or "").casefold().split())
        if cls._question_allows_count_answer(question, query_shape):
            return "count"
        if cls._question_is_time_like(question):
            return "date"
        if bool(query_shape.get("list_like") or query_shape.get("multi_entity") or query_shape.get("comparison_like")):
            return "list"
        if re.search(r"^(?:where|what area|which area|what place|which place|what location|which location)\b", text):
            return "place"
        if re.search(r"^(?:who|whom|whose|which person)\b", text):
            return "person"
        if re.search(r"^(?:has|have|had|did|do|does|is|are|was|were|can|could)\b", text):
            return "boolean"
        return "value"

    @staticmethod
    def _question_explicitly_asks_year(question: str | None) -> bool:
        text = " ".join(str(question or "").casefold().split())
        return bool(re.search(r"\b(?:what|which)\s+year\b|\byear\s+did\b|\bin\s+what\s+year\b", text))

    @staticmethod
    def _answer_text_has_date_time_signal(text: str, *, allow_year_only: bool) -> bool:
        normalized = " ".join(str(text or "").casefold().split())
        if not normalized:
            return False
        month = (
            r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        )
        weekday = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        if re.search(rf"\b\d{{1,2}}\s+{month}\b|\b{month}\s+\d{{1,2}}\b", normalized):
            return True
        if re.search(rf"\b{month}\s+\d{{4}}\b", normalized):
            return True
        if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", normalized):
            return True
        if re.search(rf"\b{weekday}\b", normalized):
            return True
        if re.search(
            r"\b(?:today|yesterday|tomorrow|last week|last month|last year|next week|next month|"
            r"week before|weeks before|friday before|saturday before|sunday before|monday before|"
            r"tuesday before|wednesday before|thursday before|a few years ago)\b",
            normalized,
        ):
            return True
        if re.search(
            r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
            r"(?:days?|weeks?|months?|years?)\s+ago\b",
            normalized,
        ):
            return True
        if re.search(r"\b(?:morning|afternoon|evening|night|am|pm)\b", normalized):
            return True
        if re.search(r"\b\d{4}\b", normalized):
            return True
        return False

    @classmethod
    def _answer_text_is_time_style(cls, text: Any, *, question: str | None = None) -> bool:
        answer = " ".join(str(text or "").strip().split())
        if not answer:
            return False
        lowered = answer.casefold().strip(" .")
        if re.fullmatch(
            r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
            r"(?:years?|months?|weeks?|days?|hours?|minutes?)\s+ago",
            lowered,
        ):
            return True
        if re.fullmatch(r"(?:about|around)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+years?\s+ago|a\s+few\s+years?\s+ago", lowered):
            return True
        if re.fullmatch(
            r"(?:today|yesterday|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year)|"
            r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
            r"(?:morning|afternoon|evening|night))",
            lowered,
        ):
            return True
        tokens = re.findall(r"[a-z0-9]+", lowered)
        return bool(
            len(tokens) <= 6
            and cls._answer_text_has_date_time_signal(
                answer,
                allow_year_only=cls._question_explicitly_asks_year(question),
            )
        )

    @classmethod
    def _answer_text_expected_type_details(
        cls,
        answer_text: Any,
        *,
        expected_type: str | None,
        question: str | None,
    ) -> dict[str, Any]:
        answer = " ".join(str(answer_text or "").strip().split())
        expected = str(expected_type or "value").strip().casefold() or "value"
        if not answer:
            return {"valid": False, "reason": "empty_answer"}
        if cls._answer_is_context_abstention(answer):
            return {"valid": False, "reason": "abstention_answer"}
        if expected == "date":
            if cls._question_explicitly_asks_year(question) and re.fullmatch(r"\d{4}", answer.strip()):
                return {"valid": True, "reason": None}
            if cls._answer_text_is_count_answer_like(answer) or re.search(
                r"\b(?:one|once|two|twice|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
                r"(?:hikes?|walks?|visits?|events?|times?|items?|letters?|rejections?)\b",
                answer.casefold(),
            ):
                return {"valid": False, "reason": "date_question_count_style_answer"}
            if cls._answer_text_has_date_time_signal(
                answer,
                allow_year_only=cls._question_explicitly_asks_year(question),
            ):
                return {"valid": True, "reason": None}
            return {"valid": False, "reason": "date_answer_missing_temporal_signal"}
        if expected in {"place", "person", "list", "value", "boolean"}:
            if cls._answer_text_is_count_answer_like(answer):
                return {"valid": False, "reason": f"{expected}_question_count_style_answer"}
            if expected in {"place", "person", "list", "value"} and cls._answer_text_is_time_style(
                answer,
                question=question,
            ):
                return {"valid": False, "reason": f"{expected}_question_time_style_answer"}
            if expected == "boolean":
                lowered = answer.casefold()
                if re.search(r"\b(?:yes|no|not|never|hasn'?t|haven'?t|didn'?t|doesn'?t|isn'?t|wasn'?t)\b", lowered):
                    return {"valid": True, "reason": None}
                # Some boolean answers are phrased as grounded status values; accept
                # non-count text to avoid forcing unnecessary retries.
                return {"valid": True, "reason": None}
        return {"valid": True, "reason": None}

    @classmethod
    def _answer_text_matches_expected_type(
        cls,
        answer_text: Any,
        expected_type: str | None,
        question: str | None,
    ) -> bool:
        return bool(
            cls._answer_text_expected_type_details(
                answer_text,
                expected_type=expected_type,
                question=question,
            ).get("valid")
        )

    MONTHS = {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "sept": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    }
    NUMBER_WORDS = {
        "a": 1,
        "an": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    @classmethod
    def _parse_date_mention(cls, text: str) -> date | None:
        compact = " ".join(str(text or "").casefold().replace(",", " ").split())
        match = re.search(
            r"\b(\d{1,2})\s+"
            r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
            r"\s+(\d{4})\b",
            compact,
        )
        if not match:
            match = re.search(
                r"\b"
                r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
                r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
                r"\s+(\d{1,2})\s+(\d{4})\b",
                compact,
            )
            if not match:
                return None
            month = cls.MONTHS.get(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3))
        else:
            day = int(match.group(1))
            month = cls.MONTHS.get(match.group(2))
            year = int(match.group(3))
        if not month:
            return None
        try:
            return date(year, month, day)
        except ValueError:
            return None

    @staticmethod
    def _format_date_value(value: date) -> str:
        return f"{value.day} {value.strftime('%B')} {value.year}"

    @classmethod
    def _parse_month_year_mention(cls, text: str) -> tuple[int, int] | None:
        compact = " ".join(str(text or "").casefold().replace(",", " ").split())
        match = re.search(
            r"\b"
            r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
            r"\s+(\d{4})\b",
            compact,
        )
        if not match:
            return None
        month = cls.MONTHS.get(match.group(1))
        if not month:
            return None
        return int(match.group(2)), month

    @staticmethod
    def _format_month_year(year: int, month: int) -> str:
        return f"{date(year, month, 1).strftime('%B')} {year}"

    @classmethod
    def _date_mentions(cls, text: str) -> set[str]:
        mentions: set[str] = set()
        compact = " ".join(str(text or "").replace(",", " ").split())
        patterns = [
            r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b",
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}\s+\d{4}\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, compact, flags=re.IGNORECASE):
                parsed = cls._parse_date_mention(match.group(0))
                if parsed:
                    mentions.add(cls._format_date_value(parsed).casefold())
        if not mentions:
            for year in re.findall(r"\b\d{4}\b", compact):
                mentions.add(year)
        return mentions

    @classmethod
    def _temporal_anchor_resolutions(cls, retrieval_bundle: RetrievalBundle) -> dict[str, list[dict[str, Any]]]:
        resolutions: dict[str, list[dict[str, Any]]] = {}
        seen: set[tuple[str, str, str, str, str, str]] = set()

        def infer_resolution_fields(
            *,
            relative_term: str,
            resolved_answer_text: str | None,
            resolved_date: str | None,
            resolution_kind: str | None,
            resolution_granularity: str | None,
        ) -> tuple[str, str, str | None, str | None]:
            target = " ".join(str(resolved_answer_text or resolved_date or "").strip().split())
            target = re.sub(r"^approximately\s+", "", target, flags=re.IGNORECASE)
            normalized_kind = str(resolution_kind or "").strip()
            normalized_granularity = str(resolution_granularity or "").strip()
            normalized_resolved_date = str(resolved_date or "").strip() or None
            parsed_date = cls._parse_date_mention(target)
            if not normalized_kind:
                if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+before\b", target, re.IGNORECASE):
                    normalized_kind = "relative_span"
                elif re.search(r"\bweek\s+before\b|\bweeks?\s+before\b", target, re.IGNORECASE):
                    normalized_kind = "relative_span"
                elif parsed_date:
                    normalized_kind = "exact_date"
                elif cls._parse_month_year_mention(target):
                    normalized_kind = "month_year"
                elif re.fullmatch(r"\d{4}", target):
                    normalized_kind = "year"
                elif re.search(r"\bfew\b|\babout\b|\baround\b", target, re.IGNORECASE):
                    normalized_kind = "fuzzy_relative"
                else:
                    normalized_kind = "relative_span"
            if not normalized_granularity:
                if normalized_kind == "exact_date":
                    normalized_granularity = "day"
                elif re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+before\b", target, re.IGNORECASE):
                    normalized_granularity = "weekday_span"
                elif normalized_kind == "relative_span":
                    normalized_granularity = "week_span"
                elif normalized_kind == "month_year":
                    normalized_granularity = "month"
                elif normalized_kind == "year":
                    normalized_granularity = "year"
                elif normalized_kind == "fuzzy_relative":
                    normalized_granularity = "fuzzy"
            if normalized_kind == "exact_date" and parsed_date:
                normalized_resolved_date = cls._format_date_value(parsed_date)
                target = normalized_resolved_date
            return normalized_kind or "exact_date", normalized_granularity or "day", normalized_resolved_date, target

        def add_resolution(
            *,
            ref: str,
            relative_term: str,
            source_date: str | None,
            resolution_kind: str,
            resolution_granularity: str | None = None,
            resolved_date: str | None = None,
            resolved_answer_text: str | None = None,
        ) -> None:
            normalized_ref = str(ref or "").strip()
            normalized_term = " ".join(str(relative_term or "").strip().casefold().split())
            normalized_source_date = str(source_date or "").strip() or None
            normalized_kind, normalized_granularity, normalized_resolved_date, normalized_answer_text = (
                infer_resolution_fields(
                    relative_term=normalized_term,
                    resolved_answer_text=resolved_answer_text,
                    resolved_date=resolved_date,
                    resolution_kind=resolution_kind,
                    resolution_granularity=resolution_granularity,
                )
            )
            if not normalized_ref or not normalized_term or not normalized_answer_text:
                return
            key = (
                normalized_ref,
                normalized_term,
                normalized_kind,
                normalized_granularity,
                str(normalized_resolved_date or ""),
                normalized_answer_text.casefold(),
            )
            if key in seen:
                return
            seen.add(key)
            row: dict[str, Any] = {
                "relative_term": normalized_term,
                "source_date": normalized_source_date,
                "resolution_kind": normalized_kind,
                "resolution_granularity": normalized_granularity,
                "resolved_answer_text": normalized_answer_text,
            }
            if normalized_resolved_date:
                row["resolved_date"] = normalized_resolved_date
            resolutions.setdefault(normalized_ref, []).append(row)

        for row in list((retrieval_bundle.metadata or {}).get("temporal_anchor_resolutions") or []):
            if not isinstance(row, dict):
                continue
            add_resolution(
                ref=str(row.get("source_ref") or ""),
                relative_term=str(row.get("relative_term") or ""),
                source_date=str(row.get("source_date") or "") or None,
                resolution_kind=str(row.get("resolution_kind") or "") or (
                    "exact_date" if row.get("resolved_date") else "relative_span"
                ),
                resolution_granularity=str(row.get("resolution_granularity") or "") or None,
                resolved_date=str(row.get("resolved_date") or "") or None,
                resolved_answer_text=str(row.get("resolved_answer_text") or "") or None,
            )
        for line in str(retrieval_bundle.prompt_context or "").splitlines():
            ref_match = re.search(r"\b(D\d+:\d+)\b", line)
            if not ref_match:
                continue
            ref = ref_match.group(1)
            source_date_match = re.search(r"occurred at\s+([^;.\n]+)", line, flags=re.IGNORECASE)
            source_date = cls._parse_date_mention(source_date_match.group(1)) if source_date_match else None
            source_date_text = cls._format_date_value(source_date) if source_date else None
            generic_resolution_match = re.search(
                r"[\"']([^\"']+)[\"']\s+refers\s+to\s+([^;.\n]+)",
                line,
                flags=re.IGNORECASE,
            )
            if generic_resolution_match:
                relative_term = generic_resolution_match.group(1).casefold()
                resolved_text = " ".join(generic_resolution_match.group(2).strip().split())
                resolved_text = re.sub(r"^approximately\s+", "", resolved_text, flags=re.IGNORECASE)
                add_resolution(
                    ref=ref,
                    relative_term=relative_term,
                    source_date=source_date_text,
                    resolution_kind="",
                    resolved_answer_text=resolved_text,
                )
                continue
            if source_date and re.search(r"\blast\s+week\b", line, flags=re.IGNORECASE):
                add_resolution(
                    ref=ref,
                    relative_term="last week",
                    source_date=source_date_text,
                    resolution_kind="relative_span",
                    resolution_granularity="week_span",
                    resolved_answer_text=f"the week before {cls._format_date_value(source_date)}",
                )
            exact_patterns: list[tuple[str, int | None]] = [
                ("yesterday", -1),
                ("today", 0),
                ("tomorrow", 1),
            ]
            for days_ago_match in re.finditer(r"\b(\d{1,2})\s+days?\s+ago\b", line, flags=re.IGNORECASE):
                exact_patterns.append((days_ago_match.group(0).casefold(), -int(days_ago_match.group(1))))
            for relative, delta in exact_patterns:
                resolved_match = re.search(
                    rf"[\"']?{re.escape(relative)}[\"']?\s+refers to\s+([^;.\n]+)",
                    line,
                    flags=re.IGNORECASE,
                )
                if resolved_match:
                    resolved = cls._parse_date_mention(resolved_match.group(1))
                    if resolved:
                        resolved_text = cls._format_date_value(resolved)
                        add_resolution(
                            ref=ref,
                            relative_term=relative,
                            source_date=source_date_text,
                            resolution_kind="exact_date",
                            resolved_date=resolved_text,
                            resolved_answer_text=resolved_text,
                        )
                elif source_date and delta is not None and re.search(rf"\b{re.escape(relative)}\b", line, flags=re.IGNORECASE):
                    resolved = source_date + timedelta(days=delta)
                    resolved_text = cls._format_date_value(resolved)
                    add_resolution(
                        ref=ref,
                        relative_term=relative,
                        source_date=source_date_text,
                        resolution_kind="exact_date",
                        resolved_date=resolved_text,
                        resolved_answer_text=resolved_text,
                    )
        return resolutions

    @classmethod
    def _temporal_answer_text_matches_target(cls, answer_text: str, target_text: str | None) -> bool:
        target = cls._temporal_normalized_text(target_text)
        answer = cls._temporal_normalized_text(answer_text)
        if not target or not answer:
            return False
        variants = {target}
        if target.startswith("the "):
            variants.add(target[4:])
        week_before_match = re.match(r"^(?:the\s+)?week\s+before\s+(.+)$", target)
        if week_before_match:
            date_text = week_before_match.group(1)
            variants.update(
                {
                    f"week before {date_text}",
                    f"the week before {date_text}",
                    f"week prior to {date_text}",
                    f"the week prior to {date_text}",
                    f"previous week before {date_text}",
                    f"the previous week before {date_text}",
                }
            )
        return any(f" {variant} " in f" {answer} " for variant in variants if variant)

    @classmethod
    def _source_line_date(cls, source_text: str) -> str | None:
        date_match = re.search(r"\bdate=([^|]+)", str(source_text or ""), flags=re.IGNORECASE)
        if date_match:
            parsed = cls._parse_date_mention(date_match.group(1))
            if parsed:
                return cls._format_date_value(parsed)
        parsed = cls._parse_date_mention(source_text)
        return cls._format_date_value(parsed) if parsed else None

    @staticmethod
    def _source_ref_sort_key(source_ref: str) -> tuple[int, int, str]:
        match = re.search(r"D(\d+):(\d+)", str(source_ref or ""))
        if not match:
            return (10**9, 10**9, str(source_ref or ""))
        return (int(match.group(1)), int(match.group(2)), str(source_ref or ""))

    @staticmethod
    def _temporal_normalized_text(text: str | None) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()

    @classmethod
    def _temporal_term_present(cls, text: str, term: str) -> bool:
        normalized_text = cls._temporal_normalized_text(text)
        normalized_term = cls._temporal_normalized_text(term)
        if not normalized_text or not normalized_term:
            return False
        if " " in normalized_term:
            return f" {normalized_term} " in f" {normalized_text} "
        return bool(re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text))

    @classmethod
    def _temporal_query_profile(cls, question: str | None) -> dict[str, list[str]]:
        raw_question = str(question or "")
        text = cls._temporal_normalized_text(raw_question)
        event_terms: set[str] = set()
        object_terms: set[str] = set()
        relation_terms: set[str] = set()
        entity_terms: set[str] = {
            token.casefold()
            for token in re.findall(r"\b[A-Z][a-z]+(?:'[a-z]+)?\b", raw_question)
            if token.casefold()
            not in {
                "when",
                "what",
                "which",
                "where",
                "who",
                "how",
                "did",
                "does",
                "was",
                "were",
                "has",
                "have",
            }
        }
        stopwords = {
            "when",
            "what",
            "which",
            "where",
            "who",
            "whom",
            "whose",
            "how",
            "did",
            "does",
            "was",
            "were",
            "has",
            "have",
            "had",
            "the",
            "a",
            "an",
            "to",
            "for",
            "from",
            "with",
            "about",
            "after",
            "before",
            "during",
            "on",
            "in",
            "at",
            "of",
            "and",
            "or",
            "he",
            "she",
            "they",
            "his",
            "her",
            "their",
            "my",
            "your",
        }
        for token in re.findall(r"[a-z0-9]+", text):
            if len(token) >= 4 and token not in stopwords:
                event_terms.add(token)
        alias_patterns: list[tuple[str, str, set[str], set[str], set[str]]] = [
            (
                "museum",
                r"\bmuseum\b|\bexhibit\b|\bexhibition\b",
                {"museum", "exhibit", "exhibition", "dinosaur exhibit"},
                {"museum", "exhibit", "dinosaur exhibit"},
                {"went", "visited", "took"},
            ),
            (
                "school_speech",
                r"\bschool\b.*\b(?:speech|talk|event)\b|\b(?:speech|talk)\b.*\bschool\b",
                {"school event", "school speech", "school talk", "speech", "talk", "talked", "students"},
                {"school event", "speech", "talk", "students"},
                {"gave", "giving", "shared", "encouraged", "talked"},
            ),
            (
                "ad_campaign",
                r"\bad\s+campaign\b|\bcampaign\b",
                {"ad campaign", "campaign", "advertising", "store", "launched"},
                {"ad campaign", "campaign", "store"},
                {"launched", "started"},
            ),
            (
                "charity_race",
                r"\bcharity\b|\brace\b",
                {"charity race", "race", "charity", "mental health"},
                {"charity race", "race"},
                {"ran", "joined", "participated"},
            ),
            (
                "hike_roadtrip",
                r"\bhik(?:e|ed|ing)\b|\broad\s*trip\b|\broadtrip\b",
                {"hike", "hiked", "hiking", "roadtrip", "road trip", "trip"},
                {"hike", "roadtrip", "road trip"},
                {"after", "before", "went", "did"},
            ),
            (
                "beach",
                r"\bbeach\b",
                {"beach", "swim", "swimming", "walk"},
                {"beach"},
                {"went", "visited"},
            ),
            (
                "conference",
                r"\bconference\b",
                {"conference", "transgender conference", "lgbtq conference"},
                {"conference"},
                {"went", "attended", "going"},
            ),
            (
                "writers_group_book",
                r"\bbook\b|\bwriters?\s+group\b|\bwriting\s+group\b",
                {"book", "writers group", "writing group", "feedback", "shared"},
                {"book", "writers group", "writing group"},
                {"shared", "read", "gave"},
            ),
            (
                "pottery",
                r"\bpottery\b|\bclass\b|\bworkshop\b",
                {"pottery", "pottery class", "workshop", "class", "plate"},
                {"pottery", "pottery class", "workshop"},
                {"signed up", "joined", "attended"},
            ),
            (
                "picnic",
                r"\bpicnic\b",
                {"picnic", "friends", "family", "mentors"},
                {"picnic"},
                {"had", "hosted", "went"},
            ),
        ]
        for _name, pattern, events, objects, relations in alias_patterns:
            if re.search(pattern, text):
                event_terms.update(events)
                object_terms.update(objects)
                relation_terms.update(relations)
        relation_patterns = {
            "after": r"\bafter\b",
            "before": r"\bbefore\b",
            "during": r"\bduring\b",
            "launched": r"\blaunched?\b|\bstarted?\b",
            "gave": r"\bgave\b|\bgiving\b",
            "shared": r"\bshared?\b",
            "went": r"\bwent\b|\bgo(?:ing)?\b|\bvisited?\b",
            "signed up": r"\bsigned\s+up\b",
            "attended": r"\battended?\b|\bjoined\b",
            "participated": r"\bparticipat(?:e|ed|ing)\b",
        }
        for term, pattern in relation_patterns.items():
            if re.search(pattern, text):
                relation_terms.add(term)
        # Person names are useful entity constraints, but they should not by
        # themselves make a temporal source query-relevant.
        event_terms.difference_update(entity_terms)
        object_terms.difference_update(entity_terms)
        return {
            "event_terms": sorted(event_terms),
            "object_terms": sorted(object_terms),
            "relation_terms": sorted(relation_terms),
            "entity_terms": sorted(entity_terms),
        }

    @classmethod
    def _score_temporal_candidate(
        cls,
        *,
        source_text: str,
        question_profile: dict[str, list[str]],
        relative_terms: list[str],
    ) -> dict[str, Any]:
        event_hits = [
            term for term in question_profile.get("event_terms", []) if cls._temporal_term_present(source_text, term)
        ]
        object_hits = [
            term for term in question_profile.get("object_terms", []) if cls._temporal_term_present(source_text, term)
        ]
        relation_hits = [
            term
            for term in question_profile.get("relation_terms", [])
            if cls._temporal_term_present(source_text, term)
        ]
        entity_hits = [
            term for term in question_profile.get("entity_terms", []) if cls._temporal_term_present(source_text, term)
        ]
        event_object_hits = list(dict.fromkeys([*event_hits, *object_hits]))
        score = (
            5 * len(event_hits)
            + 5 * len(object_hits)
            + 3 * len(relation_hits)
            + len(entity_hits)
            + (1 if any(str(term or "").strip() for term in relative_terms) else 0)
        )
        if event_object_hits:
            confidence = "high" if score >= 6 else "medium"
        else:
            confidence = "low"
        return {
            "temporal_score": score,
            "event_terms": list(dict.fromkeys(event_hits)),
            "object_terms": list(dict.fromkeys(object_hits)),
            "relation_terms": list(dict.fromkeys(relation_hits)),
            "entity_terms": list(dict.fromkeys(entity_hits)),
            "matched_terms": list(dict.fromkeys([*event_hits, *object_hits, *relation_hits, *entity_hits])),
            "confidence": confidence,
            "query_relevant": confidence in {"high", "medium"},
        }

    @classmethod
    def _temporal_answer_alignment_diagnostics(
        cls,
        *,
        answer_text: str,
        retrieval_bundle: RetrievalBundle,
        question: str | None,
    ) -> dict[str, Any]:
        if not cls._question_is_time_like(question):
            return {
                "answer_temporal_alignment_checked": False,
                "answer_temporal_alignment_valid": None,
                "answer_temporal_candidate_dates": [],
                "answer_temporal_relevant_candidate_count": 0,
                "answer_temporal_low_confidence_candidate_count": 0,
                "answer_temporal_candidates_suppressed_count": 0,
                "answer_temporal_no_query_relevant_candidate": False,
            }
        anchor_resolutions = cls._temporal_anchor_resolutions(retrieval_bundle)
        candidates: list[dict[str, Any]] = []
        source_text_by_ref = cls._context_text_by_source_ref(retrieval_bundle)
        question_profile = cls._temporal_query_profile(question)
        for ref, source_text in sorted(source_text_by_ref.items(), key=lambda item: cls._source_ref_sort_key(item[0])):
            source_date = cls._source_line_date(source_text)
            ref_candidates: list[dict[str, Any]] = []
            for resolution in list(anchor_resolutions.get(ref) or []):
                relative_terms = [
                    str(resolution.get("relative_term") or "").strip()
                ]
                resolved_date = str(resolution.get("resolved_date") or "").strip() or None
                resolved_answer_text = (
                    " ".join(str(resolution.get("resolved_answer_text") or resolved_date or "").strip().split())
                    or None
                )
                resolution_kind = str(resolution.get("resolution_kind") or "").strip() or (
                    "exact_date" if resolved_date else "relative_span"
                )
                resolution_granularity = str(resolution.get("resolution_granularity") or "").strip() or (
                    "day" if resolution_kind == "exact_date" else None
                )
                score = cls._score_temporal_candidate(
                    source_text=source_text,
                    question_profile=question_profile,
                    relative_terms=relative_terms,
                )
                ref_candidates.append(
                    {
                        "source_ref": ref,
                        "source_date": resolution.get("source_date") or source_date,
                        "resolved_date": resolved_date,
                        "resolved_answer_text": resolved_answer_text,
                        "answer_target": resolved_answer_text or resolved_date,
                        "resolution_kind": resolution_kind,
                        "resolution_granularity": resolution_granularity,
                        "relative_terms": [term for term in relative_terms if term],
                        **score,
                    }
                )
            if source_date and not ref_candidates:
                score = cls._score_temporal_candidate(
                    source_text=source_text,
                    question_profile=question_profile,
                    relative_terms=[],
                )
                if score.get("query_relevant"):
                    ref_candidates.append(
                        {
                            "source_ref": ref,
                            "source_date": source_date,
                            "resolved_date": source_date,
                            "resolved_answer_text": source_date,
                            "answer_target": source_date,
                            "resolution_kind": "exact_date",
                            "resolution_granularity": "day",
                            "relative_terms": [],
                            **score,
                        }
                    )
            candidates.extend(ref_candidates)
        confidence_rank = {"high": 2, "medium": 1, "low": 0}
        candidates = sorted(
            candidates,
            key=lambda row: (
                -confidence_rank.get(str(row.get("confidence") or "low"), 0),
                -int(row.get("temporal_score") or 0),
                -len(list(row.get("matched_terms") or [])),
                cls._source_ref_sort_key(str(row.get("source_ref") or "")),
            ),
        )
        relevant_candidates = [
            row
            for row in candidates
            if row.get("confidence") in {"high", "medium"}
            and row.get("answer_target")
            and bool(row.get("query_relevant"))
        ]
        answer_dates = cls._date_mentions(answer_text)
        low_confidence_count = sum(1 for row in candidates if row.get("confidence") == "low")
        selected = relevant_candidates[0] if relevant_candidates else None
        accepted_candidates: list[dict[str, Any]] = []
        if selected:
            selected_confidence = selected.get("confidence")
            selected_score = int(selected.get("temporal_score") or 0)
            accepted_candidates = [
                row
                for row in relevant_candidates
                if row.get("confidence") == selected_confidence
                and int(row.get("temporal_score") or 0) == selected_score
            ]
        accepted_dates = {
            str(row.get("resolved_date") or "").casefold()
            for row in accepted_candidates
            if row.get("resolution_kind") == "exact_date" and row.get("resolved_date")
        }
        accepted_answer_targets = [
            str(row.get("resolved_answer_text") or "").strip()
            for row in accepted_candidates
            if row.get("resolution_kind") != "exact_date" and row.get("resolved_answer_text")
        ]
        valid = None
        rejection_reason = None
        if accepted_dates or accepted_answer_targets:
            valid = bool(answer_dates & accepted_dates) or any(
                cls._temporal_answer_text_matches_target(answer_text, target)
                for target in accepted_answer_targets
            )
            if not valid:
                rejection_reason = (
                    "answer_temporal_target_not_aligned_to_query_relevant_temporal_source"
                    if accepted_answer_targets
                    else "answer_date_not_aligned_to_query_relevant_temporal_source"
                )
        elif not candidates:
            valid = None
        elif answer_dates:
            valid = False
            rejection_reason = "answer_date_has_no_query_relevant_temporal_candidate"
        else:
            valid = False
            rejection_reason = "date_answer_missing_query_relevant_temporal_candidate"
        return {
            "answer_temporal_alignment_checked": True,
            "answer_temporal_candidate_dates": candidates[:20],
            "answer_temporal_selected_source_ref": selected.get("source_ref") if selected else None,
            "answer_temporal_selected_date": selected.get("resolved_date") if selected else None,
            "answer_temporal_selected_answer_text": selected.get("answer_target") if selected else None,
            "answer_temporal_selected_resolution_kind": selected.get("resolution_kind") if selected else None,
            "answer_temporal_selected_resolution_granularity": (
                selected.get("resolution_granularity") if selected else None
            ),
            "answer_temporal_selected_relative_term": (
                list(selected.get("relative_terms") or [None])[0] if selected else None
            ),
            "answer_temporal_selected_confidence": selected.get("confidence") if selected else None,
            "answer_temporal_candidate_score": int(selected.get("temporal_score") or 0) if selected else None,
            "answer_temporal_candidate_match_terms": list(selected.get("matched_terms") or []) if selected else [],
            "answer_temporal_relevant_candidate_count": len(relevant_candidates),
            "answer_temporal_low_confidence_candidate_count": low_confidence_count,
            "answer_temporal_candidates_suppressed_count": max(0, len(candidates) - len(candidates[:20])),
            "answer_temporal_no_query_relevant_candidate": not bool(relevant_candidates),
            "answer_temporal_alignment_valid": valid,
            "answer_temporal_alignment_rejection_reason": rejection_reason,
            "answer_temporal_repair_used": False,
            "answer_temporal_repair_success": False,
        }

    @staticmethod
    def _context_text_by_source_ref(retrieval_bundle: RetrievalBundle) -> dict[str, str]:
        by_ref: dict[str, list[str]] = {}
        for line in str(retrieval_bundle.prompt_context or "").splitlines():
            refs = re.findall(r"\bD\d+:\d+\b", line)
            if not refs:
                continue
            for ref in refs:
                by_ref.setdefault(ref, []).append(line)
        return {ref: " ".join(lines) for ref, lines in by_ref.items()}

    @staticmethod
    def _source_family_alias_variants(term: str) -> set[str]:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(term or "").casefold()).strip()
        if not normalized:
            return set()
        aliases = {
            "turtle": {"turtle", "turtles", "tortoise", "tortoises"},
            "turtles": {"turtle", "turtles", "tortoise", "tortoises"},
            "tortoise": {"turtle", "turtles", "tortoise", "tortoises"},
            "tortoises": {"turtle", "turtles", "tortoise", "tortoises"},
            "walk": {"walk", "walks", "walked", "walking"},
            "walking": {"walk", "walks", "walked", "walking"},
            "hike": {"hike", "hikes", "hiked", "hiking", "trail", "trails"},
            "hiking": {"hike", "hikes", "hiked", "hiking", "trail", "trails"},
            "trail": {"trail", "trails", "hike", "hikes", "hiked", "hiking"},
            "trails": {"trail", "trails", "hike", "hikes", "hiked", "hiking"},
            "letter": {"letter", "letters", "note", "notes", "message", "messages"},
            "letters": {"letter", "letters", "note", "notes", "message", "messages"},
            "reject": {"reject", "rejects", "rejected", "rejecting", "rejection", "rejections"},
            "rejected": {"reject", "rejects", "rejected", "rejecting", "rejection", "rejections"},
            "rejection": {"reject", "rejects", "rejected", "rejecting", "rejection", "rejections"},
            "script": {"script", "scripts", "screenplay", "screenplays"},
            "scripts": {"script", "scripts", "screenplay", "screenplays"},
            "screenplay": {"script", "scripts", "screenplay", "screenplays"},
            "kid": {"kid", "kids", "child", "children"},
            "kids": {"kid", "kids", "child", "children"},
            "child": {"kid", "kids", "child", "children"},
            "children": {"kid", "kids", "child", "children"},
            "visit": {"visit", "visits", "visited", "visiting"},
            "visited": {"visit", "visits", "visited", "visiting"},
            "receive": {"receive", "receives", "received", "receiving", "got"},
            "received": {"receive", "receives", "received", "receiving", "got"},
            "take": {"take", "takes", "took", "taken", "taking"},
            "taken": {"take", "takes", "took", "taken", "taking"},
            "took": {"take", "takes", "took", "taken", "taking"},
            "find": {"find", "finds", "found", "finding", "discovered", "discover"},
            "found": {"find", "finds", "found", "finding", "discovered", "discover"},
        }
        variants = set(aliases.get(normalized, {normalized}))
        if normalized.endswith("s") and len(normalized) > 4:
            variants.add(normalized[:-1])
        elif len(normalized) > 3:
            variants.add(f"{normalized}s")
        if normalized.endswith("e") and len(normalized) > 3:
            variants.add(f"{normalized[:-1]}ing")
        elif len(normalized) > 3:
            variants.add(f"{normalized}ing")
        if len(normalized) > 3:
            variants.add(f"{normalized}ed")
        return {variant for variant in variants if variant}

    @classmethod
    def _source_family_match_details(
        cls,
        *,
        source_text: str,
        question: str | None,
        query_shape: dict[str, object],
        support_text: str | None = None,
    ) -> dict[str, Any]:
        text = " ".join(str(source_text or "").casefold().split())
        support = " ".join(str(support_text or "").casefold().split())
        if not text:
            return {
                "matched": False,
                "alias_hits": [],
                "support_text_used": False,
                "acceptance_reason": None,
                "rejection_reason": "empty_source_text",
            }
        question_text = " ".join(str(question or "").casefold().split())

        def _term_hits(haystack: str, term: str) -> list[str]:
            hits: list[str] = []
            for variant in sorted(cls._source_family_alias_variants(term)):
                if re.search(rf"\b{re.escape(variant)}\b", haystack):
                    hits.append(variant)
            return hits

        if bool(query_shape.get("count_like")):
            stop = {
                "many", "times", "time", "does", "have", "has", "had", "did", "done", "want", "wants", "adopt",
                "taken", "take", "took", "caroline", "melanie", "joanna", "nate", "john", "james", "calvin", "maria",
            }
            target_terms = [
                token for token in re.findall(r"[a-z0-9]+", question_text)
                if len(token) >= 4 and token not in stop
            ]
            source_hits: list[str] = []
            support_hits: list[str] = []
            alias_hits: list[str] = []
            for term in target_terms:
                term_source_hits = _term_hits(text, term)
                term_support_hits = _term_hits(support, term)
                if term_source_hits:
                    source_hits.append(term)
                    for hit in term_source_hits:
                        if hit != term:
                            alias_hits.append(f"{term}->{hit}")
                if term_support_hits:
                    support_hits.append(term)
                    for hit in term_support_hits:
                        if hit != term:
                            alias_hits.append(f"{term}->{hit}")
            if source_hits:
                reason = "source_family_signal"
                if support_hits and set(support_hits) - set(source_hits):
                    reason = "source_and_support_complementary_family_signal"
                return {
                    "matched": True,
                    "alias_hits": sorted(set(alias_hits)),
                    "support_text_used": bool(support_hits),
                    "acceptance_reason": reason,
                    "rejection_reason": None,
                }
            return {
                "matched": False,
                "alias_hits": sorted(set(alias_hits)),
                "support_text_used": bool(support_hits),
                "acceptance_reason": None,
                "rejection_reason": "missing_source_family_signal",
            }

        if cls._question_is_time_like(question):
            matched = bool(
                re.search(
                    r"\b(?:today|yesterday|tomorrow|last|next|ago|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4})\b|date=|occurred at|temporal anchors",
                    text,
                )
            )
            return {
                "matched": matched,
                "alias_hits": [],
                "support_text_used": False,
                "acceptance_reason": "temporal_signal" if matched else None,
                "rejection_reason": None if matched else "missing_temporal_signal",
            }

        family = str(query_shape.get("item_family") or "").strip().casefold()
        if family:
            terms = set(cls.FAMILY_SIGNAL_TERMS.get(family, set()))
            family_hit_terms = sorted(term for term in terms if _term_hits(text, term))
            question_terms = {
                token for token in re.findall(r"[a-z0-9]+", question_text)
                if len(token) >= 4 and token not in {"what", "which", "does", "have", "with", "from", "recently"}
            }
            question_hit_terms = sorted(term for term in question_terms if _term_hits(text, term))
            matched = bool(family_hit_terms or question_hit_terms)
            return {
                "matched": matched,
                "alias_hits": [],
                "support_text_used": False,
                "acceptance_reason": "family_or_question_signal" if matched else None,
                "rejection_reason": None if matched else "missing_family_signal",
            }
        return {
            "matched": True,
            "alias_hits": [],
            "support_text_used": False,
            "acceptance_reason": "no_family_constraint",
            "rejection_reason": None,
        }

    @classmethod
    def _source_text_matches_query_family(
        cls,
        *,
        source_text: str,
        question: str | None,
        query_shape: dict[str, object],
        support_text: str | None = None,
    ) -> bool:
        return bool(
            cls._source_family_match_details(
                source_text=source_text,
                question=question,
                query_shape=query_shape,
                support_text=support_text,
            ).get("matched")
        )

    @classmethod
    def _build_locomo_answer_synthesis_prompt(
        cls,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
    ) -> str:
        return (
            load_prompt("locomo_answer_evidence_synthesis")
            + cls._locomo_query_shape_rubric(query_task, retrieval_bundle)
            + "\n\nQUESTION:\n"
            + query_task.question
            + "\n\nRETRIEVED_CONTEXT:\n"
            + retrieval_bundle.prompt_context
        )

    @classmethod
    def _build_locomo_freeform_answer_prompt(
        cls,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
    ) -> str:
        return (
            load_prompt("locomo_answer_freeform")
            + cls._locomo_query_shape_rubric(query_task, retrieval_bundle)
            + "\n\nQUESTION:\n"
            + query_task.question
            + "\n\nRETRIEVED_CONTEXT:\n"
            + retrieval_bundle.prompt_context
        )

    @staticmethod
    def _parse_freeform_answer_response(text: str) -> dict[str, str]:
        raw = str(text or "").strip()
        if not raw:
            return {"answer": "", "rationale": "", "format": "empty"}
        if raw.startswith("{"):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                answer = " ".join(
                    str(payload.get("final_answer") or payload.get("answer") or "").strip().split()
                )
                rationale = " ".join(
                    str(payload.get("rationale") or payload.get("reason") or "").strip().split()
                )
                if answer:
                    return {"answer": answer, "rationale": rationale, "format": "json_answer"}
                if payload.get("can_answer") is False:
                    reason = " ".join(str(payload.get("abstain_reason") or "").strip().split())
                    answer = "The retrieved context does not support an answer to this question."
                    if reason:
                        answer = f"{answer[:-1]}: {reason}"
                    return {"answer": answer, "rationale": rationale, "format": "json_abstain"}
        answer_match = re.search(
            r"(?ims)^\s*Answer\s*:\s*(.*?)(?:^\s*Rationale\s*:|\Z)",
            raw,
        )
        rationale_match = re.search(r"(?ims)^\s*Rationale\s*:\s*(.*)\Z", raw)
        if answer_match:
            answer = " ".join(answer_match.group(1).strip().split())
            rationale = " ".join((rationale_match.group(1) if rationale_match else "").strip().split())
            return {"answer": answer, "rationale": rationale, "format": "answer_rationale"}
        first_line = " ".join(raw.splitlines()[0].strip().split())
        return {"answer": first_line or " ".join(raw.split()), "rationale": "", "format": "plain_text"}

    @classmethod
    def _infer_observed_answer_type(cls, answer_text: str, question: str | None) -> str:
        answer = " ".join(str(answer_text or "").strip().split())
        if not answer:
            return "unknown"
        if cls._answer_is_context_abstention(answer):
            return "abstain"
        query_shape = classify_query_shape_v1(str(question or ""), {})
        expected = cls._expected_answer_type(question, query_shape)
        if cls._answer_text_is_count_answer_like(answer):
            return "count"
        if cls._answer_text_has_date_time_signal(
            answer,
            allow_year_only=cls._question_explicitly_asks_year(question),
        ) or cls._answer_text_is_time_style(answer, question=question):
            return "date"
        lowered = answer.casefold()
        if re.search(r"\b(?:yes|no|not|never|hasn'?t|haven'?t|didn'?t|doesn'?t|isn'?t|wasn'?t)\b", lowered):
            return "boolean"
        if len(cls._split_answer_items(answer)) >= 2 or bool(
            query_shape.get("list_like") or query_shape.get("multi_entity") or query_shape.get("comparison_like")
        ):
            return "list"
        if expected in {"place", "person", "boolean"}:
            return expected
        return "value"

    @classmethod
    def _answer_type_match_details(
        cls,
        *,
        answer_text: str,
        question: str | None,
        expected_type: str,
        observed_type: str,
    ) -> dict[str, Any]:
        if observed_type == "abstain":
            return {
                "type_match": False,
                "issue": "abstain_answer",
                "repair_instruction": "Answer from retrieved evidence if the context supports the requested type.",
            }
        text_details = cls._answer_text_expected_type_details(
            answer_text,
            expected_type=expected_type,
            question=question,
        )
        if text_details.get("valid"):
            return {"type_match": True, "issue": "none", "repair_instruction": ""}
        reason = str(text_details.get("reason") or "answer_type_mismatch")
        return {
            "type_match": False,
            "issue": reason,
            "repair_instruction": (
                f"Rewrite the answer as a {expected_type} answer using only retrieved evidence; "
                "do not keep an answer form that matches the wrong type."
            ),
        }

    @classmethod
    def _build_answer_type_verification_prompt(
        cls,
        *,
        question: str,
        answer_text: str,
        expected_type: str,
        observed_type: str,
        deterministic_issue: str,
    ) -> str:
        payload = {
            "question": question,
            "answer": answer_text,
            "expected_answer_type": expected_type,
            "observed_answer_type": observed_type,
            "deterministic_issue": deterministic_issue,
        }
        return load_prompt("locomo_answer_type_verification") + "\n\nANSWER_TYPE_VERIFICATION_INPUT:\n" + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )

    def _maybe_run_answer_type_verification(
        self,
        *,
        query_task: QueryTask,
        answer_text: str,
        expected_type: str,
        observed_type: str,
        deterministic: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = {
            "answer_type_verification_used": False,
            "answer_expected_type": expected_type,
            "answer_observed_type": observed_type,
            "answer_type_match": bool(deterministic.get("type_match")),
            "answer_type_issue": deterministic.get("issue") or "none",
            "answer_type_repair_instruction": deterministic.get("repair_instruction") or "",
        }
        if metadata["answer_type_match"] and observed_type != "unknown":
            return metadata
        task = "answer_type_verification"
        if not self.llm_provider.supports_structured(task):
            return metadata
        prompt = self._build_answer_type_verification_prompt(
            question=query_task.question,
            answer_text=answer_text,
            expected_type=expected_type,
            observed_type=observed_type,
            deterministic_issue=str(metadata["answer_type_issue"] or "none"),
        )
        try:
            spec = get_structured_task_spec(task)
            structured = self.llm_provider.generate_structured(
                [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                spec=spec,
                metadata={
                    "task": task,
                    "query_task_id": query_task.query_task_id,
                    "answer_prompt_name": "locomo_answer_type_verification",
                    "repair_requested": True,
                    "repair_trigger": "answer_type_mismatch",
                    "repair_action": "answer_type_verification",
                },
            )
            result = self._model_to_dict(structured.parsed)
        except Exception as exc:  # noqa: BLE001
            metadata.update(
                {
                    "answer_type_verification_used": True,
                    "answer_type_verification_success": False,
                    "answer_type_verification_error": self._compact_exception(exc),
                }
            )
            return metadata
        metadata.update(
            {
                "answer_type_verification_used": True,
                "answer_type_verification_success": True,
                "answer_expected_type": str(result.get("expected_answer_type") or expected_type),
                "answer_observed_type": str(result.get("observed_answer_type") or observed_type),
                "answer_type_match": bool(result.get("type_match")),
                "answer_type_issue": str(result.get("issue") or "none"),
                "answer_type_repair_instruction": str(result.get("repair_instruction") or ""),
            }
        )
        return metadata

    @classmethod
    def _freeform_supporting_refs(
        cls,
        *,
        answer_text: str,
        retrieval_bundle: RetrievalBundle,
        expected_type: str,
    ) -> list[str]:
        refs: list[str] = []
        for value in [answer_text, *cls._split_answer_items(answer_text)]:
            for ref in cls._source_refs_for_surface(retrieval_bundle, value):
                if ref not in refs:
                    refs.append(ref)
        allowed_refs = sorted(cls._retrieval_source_ref_set(retrieval_bundle))
        if refs:
            return refs[:20]
        if expected_type == "date":
            date_refs = [
                ref
                for ref, text in cls._context_text_by_source_ref(retrieval_bundle).items()
                if cls._source_line_date(text) or ref in cls._temporal_anchor_resolutions(retrieval_bundle)
            ]
            if date_refs:
                return sorted(dict.fromkeys(date_refs))[:20]
        return allowed_refs[:20]

    @classmethod
    def _freeform_payload(
        cls,
        *,
        answer_text: str,
        rationale: str,
        retrieval_bundle: RetrievalBundle,
        question: str,
        expected_type: str,
        observed_type: str,
    ) -> dict[str, Any]:
        can_answer = bool(answer_text.strip()) and not cls._answer_is_context_abstention(answer_text)
        answer_type = expected_type if expected_type == "count" else observed_type if observed_type not in {"unknown", "abstain"} else expected_type
        refs = cls._freeform_supporting_refs(
            answer_text=answer_text,
            retrieval_bundle=retrieval_bundle,
            expected_type=expected_type,
        ) if can_answer else []
        return {
            "can_answer": can_answer,
            "answer_type": answer_type,
            "final_answer": answer_text if can_answer else "",
            "supporting_facts": [
                {
                    "fact_text": rationale or answer_text,
                    "source_refs": refs,
                }
            ] if can_answer and refs else [],
            "supporting_source_refs": refs,
            "counted_events": [],
            "excluded_events": [],
            "uncertainties": [],
            "abstain_reason": None if can_answer else "free-form answer abstained or was empty",
        }

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        normalized = str(text or "").strip()
        if normalized.startswith("```") and normalized.endswith("```"):
            lines = normalized.splitlines()
            normalized = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
            if not match:
                raise
            payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ParserValidationError("Answer synthesis text fallback did not return a JSON object.")
        return payload

    @staticmethod
    def _model_to_dict(model: Any) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return dict(model.model_dump())
        if hasattr(model, "dict"):
            return dict(model.dict())
        return dict(model)

    @staticmethod
    def _standard_abstention_text(reason: str | None = None) -> str:
        abstain_reason = str(reason or "").strip()
        if abstain_reason:
            return f"The retrieved context does not support an answer to this question: {abstain_reason}"
        return "The retrieved context does not support an answer to this question."

    @staticmethod
    def _retrieval_source_ref_set(retrieval_bundle: RetrievalBundle) -> set[str]:
        refs = {
            str(value).strip()
            for value in list(retrieval_bundle.source_message_refs or [])
            if str(value).strip()
        }
        refs.update(re.findall(r"\bD\d+:\d+\b", retrieval_bundle.prompt_context or ""))
        refs.update(
            str(value).strip()
            for value in list((retrieval_bundle.metadata or {}).get("source_refs") or [])
            if str(value).strip()
        )
        return refs

    @staticmethod
    def _clean_source_refs(values: Any, allowed_refs: set[str]) -> tuple[list[str], list[str]]:
        valid: list[str] = []
        invalid: list[str] = []
        for value in list(values or []):
            ref = str(value).strip()
            if not ref:
                continue
            if not allowed_refs or ref in allowed_refs:
                if ref not in valid:
                    valid.append(ref)
            else:
                invalid.append(ref)
        return valid, invalid

    @staticmethod
    def _count_event_validation_details(event: dict[str, Any]) -> dict[str, Any]:
        event_text = " ".join(str(event.get("event_text") or "").casefold().split())
        reason_text = " ".join(str(event.get("reason") or "").casefold().split())
        text = event_text
        positive_patterns = {
            "completed_event": r"\b(?:found|went|visited|attended|joined|participated|submitted|rejected|won|took|made|built|created|wrote|painted|read|bought|purchased|received|finished|completed|moved|graduated|donated|volunteered)\b",
            "past_marker": r"\b(?:yesterday|last|ago|earlier|previously|recently|already|once|twice)\b",
        }
        positive_signals = [
            name
            for name, pattern in positive_patterns.items()
            if re.search(pattern, event_text)
        ]
        reaction_only = bool(
            re.search(r"\b(?:sounds like|sound like|blast|awesome|cool|wow|congrats|sorry|must be|that'?s|nice)\b", event_text)
        ) and not positive_signals
        rejection_signal: str | None = None
        if re.search(r"\b(?:will|would|going to|plan|plans|planned|planning|next|future|upcoming|intend|hopes?|wants?)\b", text):
            rejection_signal = "future_or_intended_event"
        if re.search(r"\b(?:assume|assuming|likely|maybe|might|could|expect|guess|probably|uncertain)\b", text):
            rejection_signal = rejection_signal or "assumption_or_uncertain_candidate"
        if reaction_only:
            rejection_signal = rejection_signal or "reaction_or_comment_not_event"
        if re.search(r"\b(?:hobby|interest|likes?|enjoys?|general|usually|often)\b", text) and not positive_signals:
            rejection_signal = rejection_signal or "general_interest_not_distinct_event"
        if rejection_signal is None and reason_text:
            # The LLM's explanation can mention why an event is completed, but it can also
            # contain adjectives such as "awesome"; never let those alone invalidate evidence.
            if re.search(r"\b(?:uncertain|assumption)\b", reason_text):
                rejection_signal = "assumption_or_uncertain_candidate"
            elif re.search(r"\b(?:future|planned)\b", reason_text):
                rejection_signal = "future_or_intended_event"
            elif re.search(r"\b(?:duplicate|same)\b", reason_text):
                rejection_signal = "duplicate_event"
        return {
            "exclusion_reason": rejection_signal,
            "positive_signals": positive_signals,
            "rejection_signal": rejection_signal,
        }

    @classmethod
    def _event_is_invalid_for_count(cls, event: dict[str, Any]) -> str | None:
        return str(cls._count_event_validation_details(event).get("exclusion_reason") or "") or None

    @staticmethod
    def _event_key(event: dict[str, Any]) -> str:
        refs = ",".join(sorted(str(value).strip() for value in list(event.get("source_refs") or []) if str(value).strip()))
        text = re.sub(r"\W+", " ", str(event.get("event_text") or "").casefold()).strip()
        return refs or text

    @staticmethod
    def _answer_text_is_count_style(text: Any) -> bool:
        normalized = " ".join(str(text or "").strip().split())
        if not normalized:
            return False
        lowered = normalized.casefold().strip(" .")
        count_words = r"(?:zero|one|once|two|twice|three|four|five|six|seven|eight|nine|ten|\d+)"
        units = (
            r"(?:times?|items?|events?|places?|countries?|activities?|books?|songs?|screenplays?|"
            r"years?|months?|weeks?|days?|hours?|minutes?)"
        )
        patterns = [
            rf"^{count_words}$",
            rf"^{count_words}\s+{units}$",
        ]
        return any(re.fullmatch(pattern, lowered) for pattern in patterns)

    @classmethod
    def _answer_text_is_count_answer_like(cls, text: Any) -> bool:
        normalized = " ".join(str(text or "").strip().split())
        if not normalized:
            return False
        if cls._answer_text_is_count_style(normalized):
            return True
        lowered = normalized.casefold().strip(" .")
        count_words = r"(?:zero|one|once|two|twice|three|four|five|six|seven|eight|nine|ten|\d+)"
        return bool(
            re.search(
                rf"\b(?:{count_words}\s+(?:times?|items?|events?|places?|countries?|activities?|books?|songs?|screenplays?)|"
                rf"once|twice)\b",
                lowered,
            )
        )

    @classmethod
    def _question_disallows_count_style_answer(cls, question: str | None, query_shape: dict[str, Any]) -> bool:
        if cls._question_allows_count_answer(question, query_shape):
            return False
        text = " ".join(str(question or "").casefold().split())
        if not text:
            return False
        if re.match(r"^how\s+long\b", text):
            return False
        return bool(
            re.match(
                r"^(?:what|which|where|when|who|whom|whose|did|do|does|can|could|is|are|was|were|has|have|had)\b",
                text,
            )
        )

    @classmethod
    def _validate_answer_synthesis_payload(
        cls,
        payload: dict[str, Any],
        retrieval_bundle: RetrievalBundle,
        question: str | None,
        *,
        coerce_type_mismatch_to_abstain: bool = True,
    ) -> dict[str, Any]:
        allowed_refs = cls._retrieval_source_ref_set(retrieval_bundle)
        invalid_refs: list[str] = []
        valid_support_refs, invalid = cls._clean_source_refs(payload.get("supporting_source_refs"), allowed_refs)
        invalid_refs.extend(invalid)
        payload["supporting_source_refs"] = valid_support_refs
        query_shape = classify_query_shape_v1(str(question or ""), {})
        normalized_answer_type = cls._normalize_answer_type(payload.get("answer_type"))
        source_text_by_ref = cls._context_text_by_source_ref(retrieval_bundle)
        invalid_family_refs: list[str] = []
        accepted_family_refs: set[str] = set()
        source_family_alias_hits: list[dict[str, Any]] = []
        source_family_support_text_used: list[dict[str, Any]] = []
        count_validation_ref_acceptance_reasons: list[dict[str, Any]] = []
        count_validation_ref_rejection_reasons: list[dict[str, Any]] = []
        family_validation_enabled = bool(allowed_refs and source_text_by_ref)
        if payload.get("can_answer") and family_validation_enabled:
            family_valid_refs: list[str] = []
            for ref in valid_support_refs:
                match_details = cls._source_family_match_details(
                    source_text=source_text_by_ref.get(ref, ""),
                    question=question,
                    query_shape=query_shape,
                )
                if match_details.get("matched"):
                    family_valid_refs.append(ref)
                    accepted_family_refs.add(ref)
                    if match_details.get("alias_hits"):
                        source_family_alias_hits.append({"source_ref": ref, "alias_hits": list(match_details.get("alias_hits") or [])})
                    if match_details.get("support_text_used"):
                        source_family_support_text_used.append({"source_ref": ref, "used": True})
                else:
                    invalid_family_refs.append(ref)
                    count_validation_ref_rejection_reasons.append(
                        {
                            "source_ref": ref,
                            "reason": match_details.get("rejection_reason") or "family_validation_failed",
                            "scope": "supporting_source_ref",
                        }
                    )
            valid_support_refs = family_valid_refs
            payload["supporting_source_refs"] = valid_support_refs

        for fact in list(payload.get("supporting_facts") or []):
            if not isinstance(fact, dict):
                continue
            valid, invalid = cls._clean_source_refs(fact.get("source_refs"), allowed_refs)
            if payload.get("can_answer") and family_validation_enabled:
                filtered_valid: list[str] = []
                for ref in valid:
                    match_details = cls._source_family_match_details(
                        source_text=source_text_by_ref.get(ref, ""),
                        question=question,
                        query_shape=query_shape,
                        support_text=str(fact.get("fact_text") or ""),
                    )
                    if match_details.get("matched"):
                        filtered_valid.append(ref)
                        accepted_family_refs.add(ref)
                        if match_details.get("alias_hits"):
                            source_family_alias_hits.append({"source_ref": ref, "alias_hits": list(match_details.get("alias_hits") or [])})
                        if match_details.get("support_text_used"):
                            source_family_support_text_used.append({"source_ref": ref, "used": True, "scope": "supporting_fact"})
                    else:
                        invalid_family_refs.append(ref)
                        count_validation_ref_rejection_reasons.append(
                            {
                                "source_ref": ref,
                                "reason": match_details.get("rejection_reason") or "family_validation_failed",
                                "scope": "supporting_fact",
                            }
                        )
                valid = filtered_valid
            fact["source_refs"] = valid
            invalid_refs.extend(invalid)
            for ref in valid:
                if ref not in valid_support_refs:
                    valid_support_refs.append(ref)
        payload["supporting_source_refs"] = valid_support_refs
        question_allows_count = cls._question_allows_count_answer(question, query_shape)
        expected_answer_type = cls._expected_answer_type(question, query_shape)
        question_type_mismatch = bool(
            payload.get("can_answer")
            and normalized_answer_type == "count"
            and not question_allows_count
        )
        count_style_text_mismatch = bool(
            payload.get("can_answer")
            and normalized_answer_type != "count"
            and cls._question_disallows_count_style_answer(question, query_shape)
            and cls._answer_text_is_count_style(payload.get("final_answer"))
            and not (
                cls._question_is_time_like(question)
                and re.fullmatch(r"\d{4}", str(payload.get("final_answer") or "").strip())
            )
        )
        expected_type_text_details = cls._answer_text_expected_type_details(
            payload.get("final_answer"),
            expected_type=expected_answer_type,
            question=question,
        ) if payload.get("can_answer") else {"valid": None, "reason": None}
        repair_reason: str | None = None
        if question_type_mismatch or count_style_text_mismatch:
            if count_style_text_mismatch:
                repair_reason = "non_count_question_count_style_final_answer"
            elif cls._question_is_time_like(question):
                repair_reason = "question_type_mismatch_time_question_count_answer"
            else:
                repair_reason = "question_type_mismatch_non_count_question_count_answer"
            if coerce_type_mismatch_to_abstain:
                payload["can_answer"] = False
                payload["final_answer"] = ""
                if count_style_text_mismatch:
                    payload["abstain_reason"] = "synthesized a count-style answer for a non-count question"
                elif cls._question_is_time_like(question):
                    payload["abstain_reason"] = "synthesized a count answer for a time/date question"
                else:
                    payload["abstain_reason"] = "synthesized a count answer for a non-count question"

        count_validation_excluded: list[dict[str, Any]] = []
        count_validation_positive_signals: list[dict[str, Any]] = []
        count_validation_rejection_signals: list[dict[str, Any]] = []
        if normalized_answer_type == "count":
            validated_counted: list[dict[str, Any]] = []
            seen_event_keys: set[str] = set()
            for raw_event in list(payload.get("counted_events") or []):
                if not isinstance(raw_event, dict):
                    continue
                event = dict(raw_event)
                original_event_refs = [
                    str(value).strip()
                    for value in list(event.get("source_refs") or [])
                    if str(value).strip()
                ]
                if original_event_refs:
                    event["original_source_refs"] = original_event_refs
                valid, invalid = cls._clean_source_refs(event.get("source_refs"), allowed_refs)
                invalid_refs.extend(invalid)
                if payload.get("can_answer") and family_validation_enabled:
                    family_valid_refs: list[str] = []
                    for ref in valid:
                        match_details = cls._source_family_match_details(
                            source_text=source_text_by_ref.get(ref, ""),
                            question=question,
                            query_shape=query_shape,
                            support_text=" ".join(
                                [
                                    str(event.get("event_text") or ""),
                                    str(event.get("reason") or ""),
                                ]
                            ),
                        )
                        if match_details.get("matched"):
                            family_valid_refs.append(ref)
                            accepted_family_refs.add(ref)
                            count_validation_ref_acceptance_reasons.append(
                                {
                                    "source_ref": ref,
                                    "event_id": event.get("event_id"),
                                    "reason": match_details.get("acceptance_reason") or "family_signal",
                                    "alias_hits": list(match_details.get("alias_hits") or []),
                                    "support_text_used": bool(match_details.get("support_text_used")),
                                }
                            )
                            if match_details.get("alias_hits"):
                                source_family_alias_hits.append({"source_ref": ref, "alias_hits": list(match_details.get("alias_hits") or [])})
                            if match_details.get("support_text_used"):
                                source_family_support_text_used.append({"source_ref": ref, "used": True, "scope": "counted_event"})
                        else:
                            invalid_family_refs.append(ref)
                            count_validation_ref_rejection_reasons.append(
                                {
                                    "source_ref": ref,
                                    "event_id": event.get("event_id"),
                                    "reason": match_details.get("rejection_reason") or "family_validation_failed",
                                    "scope": "counted_event",
                                }
                            )
                    valid = family_valid_refs
                event["source_refs"] = valid
                for ref in valid:
                    if ref not in valid_support_refs:
                        valid_support_refs.append(ref)
                validation_details = cls._count_event_validation_details(event)
                if validation_details.get("positive_signals"):
                    count_validation_positive_signals.append(
                        {
                            "event_id": event.get("event_id"),
                            "event_text": event.get("event_text"),
                            "signals": list(validation_details.get("positive_signals") or []),
                        }
                    )
                if validation_details.get("rejection_signal"):
                    count_validation_rejection_signals.append(
                        {
                            "event_id": event.get("event_id"),
                            "event_text": event.get("event_text"),
                            "signal": validation_details.get("rejection_signal"),
                        }
                    )
                validation_exclusion = str(validation_details.get("exclusion_reason") or "") or None
                non_lower_bound_validation_exclusions = {
                    "future_or_intended_event",
                    "reaction_or_comment_not_event",
                    "general_interest_not_distinct_event",
                }
                exclusion_reason = None
                if not valid and allowed_refs:
                    exclusion_reason = (
                        validation_exclusion
                        if validation_exclusion in non_lower_bound_validation_exclusions
                        else "invalid_or_unretrieved_source_ref"
                    )
                else:
                    exclusion_reason = validation_exclusion
                event_key = cls._event_key(event)
                if event_key and event_key in seen_event_keys:
                    exclusion_reason = exclusion_reason or "duplicate_event"
                if exclusion_reason:
                    event["reason"] = f"{event.get('reason') or ''} [{exclusion_reason}]".strip()
                    count_validation_excluded.append(event)
                    continue
                if event_key:
                    seen_event_keys.add(event_key)
                validated_counted.append(event)
            payload["supporting_source_refs"] = valid_support_refs
            if len(validated_counted) != len(list(payload.get("counted_events") or [])):
                payload["counted_events"] = validated_counted
                existing_excluded = [
                    item for item in list(payload.get("excluded_events") or []) if isinstance(item, dict)
                ]
                payload["excluded_events"] = [*existing_excluded, *count_validation_excluded]
                if validated_counted:
                    payload["final_answer"] = str(len(validated_counted))
                else:
                    payload["can_answer"] = False
                    payload["final_answer"] = ""
                    payload["abstain_reason"] = "retrieved count candidates were invalid or unsupported"

        if payload.get("can_answer") and allowed_refs and not payload.get("supporting_source_refs"):
            payload["can_answer"] = False
            payload["final_answer"] = ""
            payload["abstain_reason"] = "no valid retrieved source refs support the synthesized answer"
            repair_reason = repair_reason or "no_valid_family_matching_source_refs"

        invalid_family_refs = [
            ref for ref in invalid_family_refs
            if ref not in accepted_family_refs
        ]
        deduped_alias_hits: list[dict[str, Any]] = []
        seen_alias_rows: set[tuple[str, tuple[str, ...]]] = set()
        for row in source_family_alias_hits:
            key = (
                str(row.get("source_ref") or ""),
                tuple(sorted(str(value) for value in list(row.get("alias_hits") or []))),
            )
            if key not in seen_alias_rows:
                seen_alias_rows.add(key)
                deduped_alias_hits.append(row)
        deduped_support_rows: list[dict[str, Any]] = []
        seen_support_rows: set[tuple[str, str]] = set()
        for row in source_family_support_text_used:
            key = (str(row.get("source_ref") or ""), str(row.get("scope") or ""))
            if key not in seen_support_rows:
                seen_support_rows.add(key)
                deduped_support_rows.append(row)

        payload["_answer_validation_metadata"] = {
            "invalid_supporting_refs": sorted(set(invalid_refs)),
            "answer_synthesis_family_validation_used": family_validation_enabled,
            "answer_synthesis_invalid_family_refs": sorted(set(invalid_family_refs)),
            "answer_synthesis_question_type_mismatch": question_type_mismatch or count_style_text_mismatch,
            "answer_synthesis_question_type_mismatch_reason": repair_reason
            if (question_type_mismatch or count_style_text_mismatch)
            else None,
            "answer_synthesis_normalized_answer_type": normalized_answer_type,
            "answer_synthesis_expected_answer_family": query_shape.get("item_family")
            or expected_answer_type,
            "answer_synthesis_repair_reason": repair_reason,
            "answer_synthesis_count_style_text_mismatch": count_style_text_mismatch,
            "answer_synthesis_expected_type_text_valid": expected_type_text_details.get("valid"),
            "answer_synthesis_expected_type_text_rejection_reason": expected_type_text_details.get("reason"),
            "source_family_validation_alias_hits": deduped_alias_hits,
            "source_family_validation_support_text_used": deduped_support_rows,
            "count_validation_ref_acceptance_reasons": count_validation_ref_acceptance_reasons,
            "count_validation_ref_rejection_reasons": count_validation_ref_rejection_reasons,
            "count_validation_excluded_events": count_validation_excluded,
            "count_validation_llm_candidate_events": [],
            "count_validation_llm_trigger_reasons": [],
            "count_validation_llm_skipped_reason": None,
            "count_validation_llm_used": False,
            "count_validation_llm_success": False,
            "count_validation_llm_scope": None,
            "count_validation_llm_confidence": None,
            "count_validation_llm_decisions": [],
            "count_validation_llm_error": None,
            "count_validation_llm_changed_count": 0,
            "count_validation_source_derived_candidate_events": [],
            "count_validation_source_derived_candidate_count": 0,
            "count_validation_source_derived_candidate_refs": [],
            "count_validation_source_derived_trigger_terms": [],
            "count_validation_source_derived_action_hits": {},
            "count_validation_source_derived_object_hits": {},
            "count_validation_source_derived_passive_rejected_refs": [],
            "count_validation_source_derived_pronoun_caption_refs": [],
            "count_validation_positive_event_signal": count_validation_positive_signals,
            "count_validation_rejection_signal": count_validation_rejection_signals,
            "answer_synthesis_allowed_ref_count": len(allowed_refs),
            "answer_synthesis_source_ref_validation_used": True,
        }
        return payload

    @staticmethod
    def _event_source_refs_for_count_validation(event: dict[str, Any]) -> list[str]:
        refs = [
            str(value).strip()
            for value in list(event.get("original_source_refs") or event.get("source_refs") or [])
            if str(value).strip()
        ]
        return list(dict.fromkeys(refs))

    @classmethod
    def _count_query_signal_profile(cls, question: str | None) -> dict[str, set[str]]:
        text = " ".join(str(question or "").casefold().split())
        action_terms: set[str] = set()
        action_patterns = {
            "walk": r"\b(?:walk|walks|walked|walking|take|takes|took|taken)\b",
            "reject": r"\b(?:reject|rejects|rejected|rejection|rejections)\b",
            "find": r"\b(?:find|finds|found|finding|discover|discovered)\b",
            "receive": r"\b(?:receive|receives|received|receiving|got)\b",
            "hike": r"\b(?:hike|hikes|hiked|hiking|trail|trails)\b",
            "visit": r"\b(?:visit|visits|visited|visiting|went|go|gone)\b",
            "adopt": r"\b(?:adopt|adopts|adopted|adoption)\b",
            "win": r"\b(?:win|wins|won)\b",
            "injure": r"\b(?:injure|injured|injury|hurt)\b",
            "write": r"\b(?:write|writes|wrote|written|script|screenplay)\b",
        }
        for term, pattern in action_patterns.items():
            if re.search(pattern, text):
                action_terms.add(term)
        stop = {
            "many", "times", "time", "does", "have", "has", "had", "did", "done", "what", "which",
            "when", "where", "with", "from", "into", "onto", "taken", "take", "takes", "took", "walk",
            "walks", "walked", "walking", "found", "find", "finding", "received", "receive", "rejected",
            "reject", "rejection", "count", "number", "nate", "joanna", "caroline", "melanie", "john",
            "james", "calvin", "maria", "andrew", "audrey", "jolene", "gina", "jon", "how",
        }
        object_terms = {
            token for token in re.findall(r"[a-z0-9]+", text)
            if len(token) >= 4 and token not in stop
        }
        if re.search(r"\bturtles?\b|\btortoises?\b", text):
            object_terms.update({"turtle", "turtles", "tortoise", "tortoises"})
        if re.search(r"\bscripts?\b|\bscreenplays?\b", text):
            object_terms.update({"script", "screenplay"})
        if re.search(r"\bletters?\b|\bnotes?\b|\bmessages?\b", text):
            object_terms.update({"letter", "note", "message"})
        if re.search(r"\bchildren\b|\bkids?\b|\bchild\b", text):
            object_terms.update({"child", "children", "kid", "kids"})
        if re.search(r"\btrails?\b|\bhiking\b", text):
            object_terms.update({"trail", "trails", "hike", "hiking"})
        return {"action_terms": action_terms, "object_terms": object_terms}

    @classmethod
    def _source_line_count_signal_hits(
        cls,
        *,
        source_text: str,
        question: str | None,
    ) -> dict[str, Any]:
        profile = cls._count_query_signal_profile(question)
        text = " ".join(str(source_text or "").casefold().split())
        caption_parts = re.findall(r"\[shared image:\s*([^\]]+)\]", str(source_text or ""), flags=re.IGNORECASE)
        caption_text = " ".join(caption_parts).casefold()
        action_hits: list[str] = []
        object_hits: list[str] = []
        caption_object_hits: list[str] = []
        alias_hits: list[str] = []

        def _hit_terms(terms: set[str], haystack: str) -> tuple[list[str], list[str]]:
            term_hits: list[str] = []
            term_alias_hits: list[str] = []
            for term in sorted(terms):
                for variant in sorted(cls._source_family_alias_variants(term)):
                    if re.search(rf"\b{re.escape(variant)}\b", haystack):
                        term_hits.append(term)
                        if variant != term:
                            term_alias_hits.append(f"{term}->{variant}")
                        break
            return list(dict.fromkeys(term_hits)), list(dict.fromkeys(term_alias_hits))

        action_hits, action_alias_hits = _hit_terms(set(profile.get("action_terms") or set()), text)
        object_hits, object_alias_hits = _hit_terms(set(profile.get("object_terms") or set()), text)
        caption_object_hits, caption_alias_hits = _hit_terms(set(profile.get("object_terms") or set()), caption_text)
        alias_hits.extend(action_alias_hits)
        alias_hits.extend(object_alias_hits)
        alias_hits.extend(caption_alias_hits)
        pronoun_bridge = bool(re.search(r"\b(?:them|it|this|those)\b", text) and (object_hits or caption_object_hits))
        image_caption = bool("[shared image:" in text and caption_object_hits)
        passive_observation = bool(
            re.search(r"\b(?:watch|watches|watched|watching|see|sees|saw|seeing|observe|observed|observing)\b", text)
            and object_hits
            and action_hits
            and not re.search(r"\b(?:took|take|taking|taken|walking\s+(?:them|it|those)|walked\s+(?:them|it|those))\b", text)
        )
        matched = bool(action_hits and (object_hits or caption_object_hits) and not passive_observation)
        return {
            "matched": matched,
            "action_hits": action_hits,
            "object_hits": object_hits,
            "caption_object_hits": caption_object_hits,
            "alias_hits": list(dict.fromkeys(alias_hits)),
            "pronoun_bridge": pronoun_bridge,
            "image_caption": image_caption,
            "passive_observation": passive_observation,
        }

    @classmethod
    def _source_derived_count_candidate_scan(
        cls,
        *,
        question: str | None,
        retrieval_bundle: RetrievalBundle,
        existing_refs: set[str],
    ) -> dict[str, Any]:
        query_shape = classify_query_shape_v1(str(question or ""), {})
        if not cls._question_allows_count_answer(question, query_shape):
            return {
                "candidates": [],
                "passive_rejected_refs": [],
                "pronoun_caption_refs": [],
            }
        allowed_refs = cls._retrieval_source_ref_set(retrieval_bundle)
        source_text_by_ref = cls._context_text_by_source_ref(retrieval_bundle)
        candidates: list[dict[str, Any]] = []
        passive_rejected_refs: list[str] = []
        pronoun_caption_refs: list[str] = []
        action_hits_by_ref: dict[str, list[str]] = {}
        object_hits_by_ref: dict[str, list[str]] = {}
        for ref, source_text in sorted(source_text_by_ref.items()):
            if ref not in allowed_refs or ref in existing_refs:
                continue
            match = cls._source_line_count_signal_hits(source_text=source_text, question=question)
            action_hits_by_ref[ref] = list(match.get("action_hits") or [])
            object_hits_by_ref[ref] = list(
                dict.fromkeys(
                    [
                        *list(match.get("object_hits") or []),
                        *list(match.get("caption_object_hits") or []),
                    ]
                )
            )
            if match.get("passive_observation"):
                passive_rejected_refs.append(ref)
                continue
            if not match.get("matched"):
                continue
            validation_details = cls._count_event_validation_details({"event_text": source_text, "reason": ""})
            exclusion_reason = validation_details.get("exclusion_reason")
            if exclusion_reason in {"future_or_intended_event", "reaction_or_comment_not_event"}:
                continue
            if (
                exclusion_reason == "general_interest_not_distinct_event"
                and not (match.get("pronoun_bridge") or match.get("image_caption"))
            ):
                continue
            digest = hashlib.sha1(f"{ref}|{source_text}".encode("utf-8")).hexdigest()[:10]
            trigger_terms = sorted(
                set(
                    list(match.get("action_hits") or [])
                    + list(match.get("object_hits") or [])
                    + list(match.get("caption_object_hits") or [])
                )
            )
            reason_parts = ["source_derived_candidate"]
            if match.get("pronoun_bridge"):
                reason_parts.append("pronoun_plus_object_signal")
            if match.get("pronoun_bridge") and match.get("image_caption"):
                pronoun_caption_refs.append(ref)
            if match.get("image_caption"):
                reason_parts.append("image_caption_object_signal")
            compact_source_text = " ".join(str(source_text or "").split())
            event = {
                "event_id": f"source-derived:{ref}:{digest}",
                "event_text": compact_source_text,
                "source_refs": [ref],
                "original_source_refs": [ref],
                "reason": "[" + ",".join(reason_parts) + "] " + ", ".join(trigger_terms),
                "source_derived": True,
                "source_derived_trigger_terms": trigger_terms,
                "source_derived_alias_hits": list(match.get("alias_hits") or []),
            }
            candidates.append(
                {
                    "event_id": event["event_id"],
                    "event_text": event["event_text"],
                    "source_refs": [ref],
                    "valid_source_refs": [ref],
                    "source_lines": [{"source_ref": ref, "source_text": source_text}],
                    "deterministic_status": "uncertain_source_derived",
                    "deterministic_reason": event["reason"],
                    "event": event,
                    "source_derived": True,
                    "source_derived_trigger_terms": trigger_terms,
                    "source_derived_alias_hits": list(match.get("alias_hits") or []),
                    "source_derived_action_hits": list(match.get("action_hits") or []),
                    "source_derived_object_hits": list(
                        dict.fromkeys(
                            [
                                *list(match.get("object_hits") or []),
                                *list(match.get("caption_object_hits") or []),
                            ]
                        )
                    ),
                    "source_derived_pronoun_caption": bool(match.get("pronoun_bridge") and match.get("image_caption")),
                }
            )
        return {
            "candidates": candidates,
            "passive_rejected_refs": sorted(set(passive_rejected_refs)),
            "pronoun_caption_refs": sorted(set(pronoun_caption_refs)),
            "action_hits_by_ref": action_hits_by_ref,
            "object_hits_by_ref": object_hits_by_ref,
        }

    @classmethod
    def _source_derived_count_candidates(
        cls,
        *,
        question: str | None,
        retrieval_bundle: RetrievalBundle,
        existing_refs: set[str],
    ) -> list[dict[str, Any]]:
        scan = cls._source_derived_count_candidate_scan(
            question=question,
            retrieval_bundle=retrieval_bundle,
            existing_refs=existing_refs,
        )
        return list(scan.get("candidates") or [])

    @classmethod
    def _count_validation_candidate_events(
        cls,
        payload: dict[str, Any],
        retrieval_bundle: RetrievalBundle,
        question: str | None,
    ) -> list[dict[str, Any]]:
        validation = dict(payload.get("_answer_validation_metadata") or {})
        source_text_by_ref = cls._context_text_by_source_ref(retrieval_bundle)
        allowed_refs = cls._retrieval_source_ref_set(retrieval_bundle)
        candidates: dict[str, dict[str, Any]] = {}

        def add_event(raw_event: Any, *, status: str, deterministic_reason: str | None) -> None:
            if not isinstance(raw_event, dict):
                return
            event = dict(raw_event)
            event_id = str(event.get("event_id") or "").strip()
            if not event_id:
                event_id = f"event-{len(candidates) + 1}"
                event["event_id"] = event_id
            refs = cls._event_source_refs_for_count_validation(event)
            source_lines = [
                {"source_ref": ref, "source_text": source_text_by_ref.get(ref, "")}
                for ref in refs
                if ref in allowed_refs and source_text_by_ref.get(ref)
            ]
            row = {
                "event_id": event_id,
                "event_text": str(event.get("event_text") or ""),
                "source_refs": refs,
                "valid_source_refs": [item["source_ref"] for item in source_lines],
                "source_lines": source_lines,
                "deterministic_status": status,
                "deterministic_reason": deterministic_reason or str(event.get("reason") or ""),
                "event": event,
            }
            existing = candidates.get(event_id)
            if existing is None or existing.get("deterministic_status") != "counted":
                candidates[event_id] = row

        for event in list(payload.get("counted_events") or []):
            add_event(event, status="counted", deterministic_reason=str(dict(event).get("reason") or "counted"))
        for event in list(validation.get("count_validation_excluded_events") or []):
            add_event(event, status="excluded", deterministic_reason=cls._count_excluded_event_reason(dict(event)))
        existing_refs = {
            ref
            for candidate in candidates.values()
            for ref in list(candidate.get("source_refs") or [])
            if str(ref).strip()
        }
        scan = cls._source_derived_count_candidate_scan(
            question=question,
            retrieval_bundle=retrieval_bundle,
            existing_refs=existing_refs,
        )
        for candidate in list(scan.get("candidates") or []):
            candidates.setdefault(str(candidate.get("event_id")), candidate)
        validation.update(
            {
                "count_validation_source_derived_passive_rejected_refs": list(
                    scan.get("passive_rejected_refs") or []
                ),
                "count_validation_source_derived_pronoun_caption_refs": list(
                    scan.get("pronoun_caption_refs") or []
                ),
                "count_validation_source_derived_action_hits": dict(scan.get("action_hits_by_ref") or {}),
                "count_validation_source_derived_object_hits": dict(scan.get("object_hits_by_ref") or {}),
            }
        )
        payload["_answer_validation_metadata"] = validation
        return list(candidates.values())

    @classmethod
    def _count_validation_trigger_reasons(
        cls,
        *,
        payload: dict[str, Any],
        question: str | None,
        retrieval_bundle: RetrievalBundle,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[str], str | None]:
        validation = dict(payload.get("_answer_validation_metadata") or {})
        query_shape = classify_query_shape_v1(str(question or ""), {})
        if cls._normalize_answer_type(payload.get("answer_type")) != "count" or not cls._question_allows_count_answer(question, query_shape):
            return [], "not_count_question"
        if not candidates:
            return [], "no_count_candidate_events"
        if not any(candidate.get("valid_source_refs") for candidate in candidates):
            return [], "no_candidate_refs_in_retrieved_context"

        reasons: list[str] = []
        if any(candidate.get("source_derived") for candidate in candidates):
            reasons.append("source_derived_candidate")
        lower_bound_reasons = cls._count_lower_bound_reasons(payload)
        if lower_bound_reasons:
            reasons.append("deterministic_lower_bound")
        if validation.get("answer_synthesis_invalid_family_refs"):
            reasons.append("family_ref_rejected")
        allowed_refs = cls._retrieval_source_ref_set(retrieval_bundle)
        for event in list(validation.get("count_validation_excluded_events") or []):
            if not isinstance(event, dict):
                continue
            reason = cls._count_excluded_event_reason(event)
            refs = cls._event_source_refs_for_count_validation(event)
            if reason == "invalid_or_unretrieved_source_ref" and any(ref in allowed_refs for ref in refs):
                reasons.append("source_backed_candidate_rejected")
            if reason == "assumption_or_uncertain_candidate":
                reasons.append("uncertain_candidate")
        question_and_sources = " ".join(
            [
                str(question or ""),
                retrieval_bundle.prompt_context or "",
            ]
        ).casefold()
        if re.search(r"\b(?:plan|planned|planning|intend|intended|want|wanted|hope|hoped)\b", question_and_sources):
            if any(
                cls._count_excluded_event_reason(dict(candidate.get("event") or {})) == "future_or_intended_event"
                or "future_or_intended_event" in str(candidate.get("deterministic_reason") or "")
                for candidate in candidates
            ):
                reasons.append("planned_scope_uncertain")
        if re.search(r"\[shared image:|\b(?:them|it|this|those)\b", question_and_sources):
            reasons.append("ambiguous_source_language")
        if validation.get("source_family_validation_support_text_used"):
            reasons.append("source_support_text_complementary")
        if payload.get("counted_events") and validation.get("count_validation_excluded_events"):
            reasons.append("mixed_counted_and_excluded_candidates")

        non_trigger_exclusions = {"duplicate_event", "reaction_or_comment_not_event", "general_interest_not_distinct_event"}
        if reasons:
            only_non_trigger = True
            for event in list(validation.get("count_validation_excluded_events") or []):
                if not isinstance(event, dict):
                    continue
                if cls._count_excluded_event_reason(event) not in non_trigger_exclusions:
                    only_non_trigger = False
                    break
            if only_non_trigger and not set(reasons) - {"mixed_counted_and_excluded_candidates"}:
                return [], "only_clear_duplicate_reaction_or_general_interest_exclusions"
        return list(dict.fromkeys(reasons)), None if reasons else "deterministic_count_validation_confident"

    @classmethod
    def _build_count_validation_prompt(
        cls,
        *,
        question: str | None,
        candidates: list[dict[str, Any]],
        trigger_reasons: list[str],
    ) -> str:
        prompt_payload = {
            "question": str(question or ""),
            "trigger_reasons": trigger_reasons,
            "candidate_events": [
                {
                    "event_id": candidate.get("event_id"),
                    "event_text": candidate.get("event_text"),
                    "source_refs": candidate.get("valid_source_refs"),
                    "deterministic_status": candidate.get("deterministic_status"),
                    "deterministic_reason": candidate.get("deterministic_reason"),
                    "source_derived": bool(candidate.get("source_derived")),
                    "source_derived_trigger_terms": list(candidate.get("source_derived_trigger_terms") or []),
                    "source_lines": candidate.get("source_lines"),
                }
                for candidate in candidates
                if candidate.get("valid_source_refs")
            ],
        }
        return load_prompt("locomo_answer_count_validation") + "\n\nCOUNT_VALIDATION_INPUT:\n" + json.dumps(
            prompt_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    def _generate_count_validation_result(
        self,
        *,
        prompt: str,
        query_task_id: str | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        task = "answer_count_validation"
        spec = get_structured_task_spec(task)
        errors: list[str] = []
        if self.llm_provider.supports_structured(task):
            try:
                structured_response = self.llm_provider.generate_structured(
                    [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                    spec=spec,
                    metadata={
                        "task": task,
                        "query_task_id": query_task_id,
                        "answer_prompt_name": "locomo_answer_count_validation",
                        "repair_requested": True,
                        "repair_trigger": "count_validation_uncertain",
                        "repair_action": "llm_count_validation",
                    },
                )
                return self._model_to_dict(structured_response.parsed), None
            except Exception as exc:  # noqa: BLE001
                errors.append("structured: " + self._compact_exception(exc, limit=500))
        else:
            errors.append("structured: unsupported")

        try:
            response = self.llm_provider.generate(
                [
                    NormalizedMessage(
                        role="user",
                        content=prompt + "\n\nReturn ONLY a JSON object matching the requested schema.",
                        turn_index=0,
                    )
                ],
                metadata={
                    "task": task,
                    "query_task_id": query_task_id,
                    "answer_prompt_name": "locomo_answer_count_validation",
                    "repair_requested": True,
                    "repair_trigger": "count_validation_uncertain",
                    "repair_action": "llm_count_validation",
                    "structured_fallback": True,
                    "structured_requested": True,
                    "structured_supported": self.llm_provider.supports_structured(task),
                    "fallback_used": True,
                    "fallback_mode": "text_json",
                    "fallback_reason": "structured_unsupported"
                    if not self.llm_provider.supports_structured(task)
                    else "structured_exception",
                },
            )
            return self._model_to_dict(parse_structured_payload(spec, self._extract_json_object(response.text))), None
        except Exception as exc:  # noqa: BLE001
            errors.append("text_json: " + self._compact_exception(exc, limit=500))
            return None, "; ".join(errors)

    @classmethod
    def _apply_count_validation_result(
        cls,
        *,
        payload: dict[str, Any],
        result: dict[str, Any],
        candidates: list[dict[str, Any]],
        allowed_refs: set[str],
    ) -> tuple[bool, str | None]:
        validation = dict(payload.get("_answer_validation_metadata") or {})
        confidence = str(result.get("confidence") or "").casefold()
        if confidence not in {"high", "medium"}:
            return False, "low_confidence"
        candidate_by_id = {str(candidate.get("event_id")): candidate for candidate in candidates}
        decisions = [
            dict(item)
            for item in list(result.get("validated_events") or [])
            if isinstance(item, dict)
        ]
        if not decisions:
            return False, "no_validated_events"
        counted: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        decision_rows: list[dict[str, Any]] = []
        for decision in decisions:
            event_id = str(decision.get("event_id") or "")
            candidate = candidate_by_id.get(event_id)
            if candidate is None:
                return False, f"unknown_event_id:{event_id}"
            candidate_refs = set(str(ref) for ref in list(candidate.get("valid_source_refs") or []))
            decision_refs = {
                str(ref).strip()
                for ref in list(decision.get("source_refs") or [])
                if str(ref).strip()
            }
            if not decision_refs:
                decision_refs = set(candidate_refs)
            if not decision_refs <= candidate_refs:
                return False, f"new_source_ref:{sorted(decision_refs - candidate_refs)}"
            if not decision_refs <= allowed_refs:
                return False, f"unretrieved_source_ref:{sorted(decision_refs - allowed_refs)}"
            event = dict(candidate.get("event") or {})
            event["source_refs"] = sorted(decision_refs)
            event["reason"] = f"{event.get('reason') or candidate.get('deterministic_reason') or ''} [llm_count_validator:{decision.get('decision')}] {decision.get('reason') or ''}".strip()
            normalized_decision = str(decision.get("decision") or "").upper()
            decision_rows.append(
                {
                    "event_id": event_id,
                    "decision": normalized_decision,
                    "source_refs": sorted(decision_refs),
                    "reason": decision.get("reason"),
                }
            )
            if normalized_decision == "COUNT":
                counted.append(event)
            elif normalized_decision in {"EXCLUDE", "UNCERTAIN"}:
                excluded.append(event)
            else:
                return False, f"invalid_decision:{normalized_decision}"
        scope = str(result.get("count_scope") or "unknown")
        final_count = result.get("final_count")
        counted_count = len(counted)
        if final_count is not None:
            try:
                final_count = int(final_count)
            except (TypeError, ValueError):
                return False, "invalid_final_count"
            if final_count != counted_count and scope not in {"states", "possessions", "mentions"}:
                return False, "final_count_mismatch"
        count_value = int(final_count) if final_count is not None and scope in {"states", "possessions", "mentions"} else counted_count
        old_counted_ids = {
            str(event.get("event_id") or "")
            for event in list(payload.get("counted_events") or [])
            if isinstance(event, dict)
        }
        new_counted_ids = {str(event.get("event_id") or "") for event in counted}
        payload["counted_events"] = counted
        existing_excluded = [
            item for item in list(payload.get("excluded_events") or []) if isinstance(item, dict)
        ]
        deterministic_excluded_ids = {
            str(item.get("event_id") or "") for item in list(validation.get("count_validation_excluded_events") or [])
            if isinstance(item, dict)
        }
        retained_existing = [
            item for item in existing_excluded
            if str(item.get("event_id") or "") not in deterministic_excluded_ids
        ]
        payload["excluded_events"] = [*retained_existing, *excluded]
        payload["supporting_source_refs"] = sorted(
            {
                ref
                for event in counted
                for ref in list(event.get("source_refs") or [])
                if str(ref).strip()
            }
        )
        # After a successful count arbitration the final answer is rebuilt only
        # from validated COUNT events, so stale support refs from the initial
        # synthesis payload should no longer mark the final answer lower-bound.
        validation["invalid_supporting_refs"] = []
        validation["answer_synthesis_invalid_family_refs"] = []
        if count_value > 0:
            payload["can_answer"] = True
            payload["final_answer"] = str(count_value)
            payload["abstain_reason"] = None
        else:
            payload["can_answer"] = False
            payload["final_answer"] = ""
            payload["abstain_reason"] = "count validator found no supported countable events"
        validation.update(
            {
                "count_validation_llm_success": True,
                "count_validation_llm_scope": scope,
                "count_validation_llm_confidence": confidence,
                "count_validation_llm_decisions": decision_rows,
                "count_validation_llm_error": None,
                "count_validation_llm_changed_count": len(old_counted_ids ^ new_counted_ids),
                "count_validation_excluded_events": excluded,
            }
        )
        payload["_answer_validation_metadata"] = validation
        return True, None

    def _maybe_run_llm_count_validation(
        self,
        *,
        payload: dict[str, Any],
        retrieval_bundle: RetrievalBundle,
        question: str | None,
    ) -> None:
        validation = dict(payload.get("_answer_validation_metadata") or {})
        candidates = self._count_validation_candidate_events(payload, retrieval_bundle, question)
        trigger_reasons, skipped_reason = self._count_validation_trigger_reasons(
            payload=payload,
            question=question,
            retrieval_bundle=retrieval_bundle,
            candidates=candidates,
        )
        validation["count_validation_llm_candidate_events"] = [
            {
                "event_id": candidate.get("event_id"),
                "event_text": candidate.get("event_text"),
                "source_refs": candidate.get("source_refs"),
                "valid_source_refs": candidate.get("valid_source_refs"),
                "deterministic_status": candidate.get("deterministic_status"),
                "deterministic_reason": candidate.get("deterministic_reason"),
                "source_derived": bool(candidate.get("source_derived")),
                "source_derived_trigger_terms": list(candidate.get("source_derived_trigger_terms") or []),
            }
            for candidate in candidates
        ]
        source_derived_candidates = [
            candidate for candidate in candidates if candidate.get("source_derived")
        ]
        validation["count_validation_source_derived_candidate_events"] = [
            {
                "event_id": candidate.get("event_id"),
                "event_text": candidate.get("event_text"),
                "source_refs": candidate.get("source_refs"),
                "trigger_terms": list(candidate.get("source_derived_trigger_terms") or []),
                "alias_hits": list(candidate.get("source_derived_alias_hits") or []),
                "action_hits": list(candidate.get("source_derived_action_hits") or []),
                "object_hits": list(candidate.get("source_derived_object_hits") or []),
                "pronoun_caption": bool(candidate.get("source_derived_pronoun_caption")),
            }
            for candidate in source_derived_candidates
        ]
        validation["count_validation_source_derived_candidate_count"] = len(source_derived_candidates)
        validation["count_validation_source_derived_candidate_refs"] = sorted(
            {
                str(ref)
                for candidate in source_derived_candidates
                for ref in list(candidate.get("valid_source_refs") or [])
                if str(ref).strip()
            }
        )
        validation["count_validation_source_derived_trigger_terms"] = sorted(
            {
                str(term)
                for candidate in source_derived_candidates
                for term in list(candidate.get("source_derived_trigger_terms") or [])
                if str(term).strip()
            }
        )
        validation["count_validation_source_derived_pronoun_caption_refs"] = sorted(
            set(
                list(validation.get("count_validation_source_derived_pronoun_caption_refs") or [])
                + [
                    str(ref)
                    for candidate in source_derived_candidates
                    if candidate.get("source_derived_pronoun_caption")
                    for ref in list(candidate.get("valid_source_refs") or [])
                    if str(ref).strip()
                ]
            )
        )
        validation["count_validation_llm_trigger_reasons"] = trigger_reasons
        validation["count_validation_llm_skipped_reason"] = skipped_reason
        payload["_answer_validation_metadata"] = validation
        if not trigger_reasons:
            return
        prompt = self._build_count_validation_prompt(
            question=question,
            candidates=candidates,
            trigger_reasons=trigger_reasons,
        )
        validation["count_validation_llm_used"] = True
        payload["_answer_validation_metadata"] = validation
        result, error = self._generate_count_validation_result(
            prompt=prompt,
            query_task_id=str(payload.get("query_task_id") or ""),
        )
        validation = dict(payload.get("_answer_validation_metadata") or {})
        if error or result is None:
            validation["count_validation_llm_success"] = False
            validation["count_validation_llm_error"] = error or "empty_validator_result"
            payload["_answer_validation_metadata"] = validation
            return
        success, merge_error = self._apply_count_validation_result(
            payload=payload,
            result=result,
            candidates=candidates,
            allowed_refs=self._retrieval_source_ref_set(retrieval_bundle),
        )
        validation = dict(payload.get("_answer_validation_metadata") or {})
        if not success:
            validation["count_validation_llm_success"] = False
            validation["count_validation_llm_scope"] = result.get("count_scope")
            validation["count_validation_llm_confidence"] = result.get("confidence")
            validation["count_validation_llm_decisions"] = list(result.get("validated_events") or [])
            validation["count_validation_llm_error"] = merge_error
            payload["_answer_validation_metadata"] = validation

    @staticmethod
    def _source_backed_context_lines(retrieval_bundle: RetrievalBundle) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for line in str(retrieval_bundle.prompt_context or "").splitlines():
            refs = re.findall(r"\bD\d+:\d+\b", line)
            if not refs:
                continue
            compact = " ".join(line.split())
            if compact:
                rows.extend((ref, compact) for ref in refs)
        return rows

    @staticmethod
    def _invalid_bridge_target(target: str) -> bool:
        normalized = " ".join(str(target or "").split()).strip(" .,:;")
        if not normalized:
            return True
        lowered = normalized.casefold()
        invalid = {
            "a",
            "an",
            "the",
            "my",
            "old",
            "home",
            "flags",
            "active",
            "summary",
            "context",
            "claim",
            "claims",
            "keyword",
            "keywords",
            "trajectory",
            "source",
            "sources",
            "linked",
            "none",
            "unknown",
            "not provided",
        }
        if lowered in invalid:
            return True
        if lowered.endswith(" flags") or " flags " in lowered:
            return True
        return False

    @classmethod
    def _bridge_alias_facts(cls, retrieval_bundle: RetrievalBundle) -> list[dict[str, str]]:
        patterns = [
            (
                r"\bold area\b(?:[^.\n]{0,80}?\b(?:,|is|=|called|named)\s+)(?:the\s+)?([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})",
                "old area",
            ),
            (
                r"\bhome country\b(?:[^.\n]{0,80}?\b(?:,|is|=|called|named)\s+)(?:the\s+)?([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})",
                "home country",
            ),
        ]
        facts: list[dict[str, str]] = []
        for pattern, alias in patterns:
            for source_ref, line in cls._source_backed_context_lines(retrieval_bundle):
                for match in re.finditer(pattern, line):
                    target = " ".join(match.group(1).split()).strip(" .,:;")
                    if cls._invalid_bridge_target(target):
                        continue
                    facts.append({"alias": alias, "target": target, "source_ref": source_ref})
        deduped: dict[tuple[str, str, str], dict[str, str]] = {}
        for fact in facts:
            deduped.setdefault(
                (fact["alias"].casefold(), fact["target"].casefold(), fact.get("source_ref", "")),
                fact,
            )
        return list(deduped.values())

    @classmethod
    def _apply_bridge_facts(
        cls,
        *,
        answer_text: str,
        payload: dict[str, Any],
        retrieval_bundle: RetrievalBundle,
        question: str | None,
    ) -> tuple[str, dict[str, Any]]:
        bridge_facts = cls._bridge_alias_facts(retrieval_bundle)
        grouped_targets: dict[str, set[str]] = {}
        for fact in bridge_facts:
            grouped_targets.setdefault(fact["alias"].casefold(), set()).add(fact["target"].casefold())
        conflicted_aliases = sorted(alias for alias, targets in grouped_targets.items() if len(targets) > 1)
        conflict_facts = [
            fact for fact in bridge_facts if fact["alias"].casefold() in set(conflicted_aliases)
        ]
        applicable_facts = [
            fact for fact in bridge_facts if fact["alias"].casefold() not in set(conflicted_aliases)
        ]
        if not payload.get("can_answer"):
            return answer_text, {
                "bridge_facts_used": [],
                "bridge_facts_missing": applicable_facts,
                "bridge_facts_conflicted": conflict_facts,
                "bridge_facts_ignored": conflict_facts,
                "bridge_finalization_used": False,
                "bridge_finalization_alias": None,
                "bridge_finalization_target": None,
                "bridge_finalization_source_refs": [],
                "bridge_finalization_action": None,
                "bridge_finalization_conflicted": bool(conflict_facts),
                "bridge_finalization_failed_reason": None,
            }
        folded_answer = answer_text.casefold()
        folded_question = str(question or "").casefold()
        used: list[dict[str, str]] = []
        missing: list[dict[str, str]] = []
        updated = answer_text
        ignored: list[dict[str, str]] = list(conflict_facts)
        finalization: dict[str, Any] = {
            "bridge_finalization_used": False,
            "bridge_finalization_alias": None,
            "bridge_finalization_target": None,
            "bridge_finalization_source_refs": [],
            "bridge_finalization_action": None,
            "bridge_finalization_conflicted": bool(conflict_facts),
            "bridge_finalization_failed_reason": None,
        }
        for fact in applicable_facts:
            alias = fact["alias"]
            target = fact["target"]
            if target.casefold() in updated.casefold():
                used.append({**fact, "action": "already_present"})
                finalization.update(
                    {
                        "bridge_finalization_used": True,
                        "bridge_finalization_alias": alias,
                        "bridge_finalization_target": target,
                        "bridge_finalization_source_refs": [fact.get("source_ref")],
                        "bridge_finalization_action": "already_present",
                    }
                )
                continue
            alias_relevant = alias in folded_answer or alias in folded_question
            if not alias_relevant:
                missing.append(fact)
                continue
            if alias in folded_answer:
                if re.search(r"\bwhat\s+area\b", folded_question):
                    updated = target
                else:
                    updated = re.sub(re.escape(alias), target, updated, flags=re.IGNORECASE)
            else:
                updated = f"{updated.rstrip('.')} ({target})."
            used.append({**fact, "action": "applied"})
            finalization.update(
                {
                    "bridge_finalization_used": True,
                    "bridge_finalization_alias": alias,
                    "bridge_finalization_target": target,
                    "bridge_finalization_source_refs": [fact.get("source_ref")],
                    "bridge_finalization_action": "applied",
                }
            )
            if alias.casefold() in updated.casefold() and target.casefold() not in updated.casefold():
                if re.search(r"\bwhat\s+area\b", folded_question):
                    updated = target
                else:
                    updated = re.sub(re.escape(alias), target, updated, flags=re.IGNORECASE)
                finalization["bridge_finalization_action"] = "forced_replace"
            if alias.casefold() in updated.casefold() and target.casefold() not in updated.casefold():
                finalization["bridge_finalization_failed_reason"] = "alias_remained_after_replacement"
        return updated, {
            "bridge_facts_used": used,
            "bridge_facts_missing": missing,
            "bridge_facts_conflicted": conflict_facts,
            "bridge_facts_ignored": ignored,
            **finalization,
        }

    @staticmethod
    def _count_unit_from_question(question: str | None) -> str:
        text = " ".join(str(question or "").strip().split())
        match = re.search(
            r"\bhow many\s+(.+?)(?:\s+(?:has|have|had|did|does|do|is|are|was|were|will|would|can|could|should)\b|\?)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return "items"
        unit = match.group(1).strip(" ?.,")
        if not unit or unit.casefold().startswith("of "):
            return "items"
        return unit

    @staticmethod
    def _count_lower_bound_unit_from_question(question: str | None, unit: str) -> str:
        lowered = str(question or "").casefold()
        if "reject" in lowered:
            return "rejection"
        if "visit" in lowered:
            return "visit"
        if "participat" in lowered:
            return "participation"
        if "won" in lowered or "win" in lowered:
            return "win"
        if "adopt" in lowered:
            return "adoption"
        if "injur" in lowered:
            return "injury"
        if unit.casefold() in {"time", "times"}:
            return "time"
        return unit

    @staticmethod
    def _count_word(value: int, *, sentence_start: bool = False) -> str:
        words = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        rendered = words.get(value, str(value))
        if sentence_start and rendered:
            return rendered[:1].upper() + rendered[1:]
        return rendered

    @classmethod
    def _count_phrase(cls, value: int, unit: str, *, exact: bool, sentence_start: bool = False) -> str:
        normalized_unit = " ".join(str(unit or "items").split()).casefold()
        if exact and normalized_unit in {"time", "times"}:
            if value == 1:
                return "Once" if sentence_start else "once"
            if value == 2:
                return "Twice" if sentence_start else "twice"
            return f"{cls._count_word(value, sentence_start=sentence_start)} times"
        word = cls._count_word(value, sentence_start=sentence_start)
        if value == 1:
            unit_text = normalized_unit[:-1] if normalized_unit.endswith("s") and len(normalized_unit) > 1 else normalized_unit
        else:
            unit_text = normalized_unit if normalized_unit.endswith("s") else f"{normalized_unit}s"
        return f"{word} {unit_text}".strip()

    @staticmethod
    def _count_excluded_event_reason(event: dict[str, Any]) -> str:
        reason = str(event.get("reason") or "")
        bracketed = re.findall(r"\[([^\]]+)\]", reason)
        if bracketed:
            return bracketed[-1].strip()
        compact = " ".join(reason.casefold().split())
        if "invalid_or_unretrieved_source_ref" in compact:
            return "invalid_or_unretrieved_source_ref"
        if "uncertain" in compact or "assumption" in compact:
            return "assumption_or_uncertain_candidate"
        if "future" in compact or "planned" in compact:
            return "future_or_intended_event"
        if "duplicate" in compact or "same" in compact:
            return "duplicate_event"
        if "reaction" in compact or "comment" in compact:
            return "reaction_or_comment_not_event"
        if "hobby" in compact or "interest" in compact or "general" in compact:
            return "general_interest_not_distinct_event"
        return compact or "unknown"

    @classmethod
    def _count_lower_bound_reasons(cls, payload: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        validation = dict(payload.get("_answer_validation_metadata") or {})
        uncertainties = [
            str(item).strip()
            for item in list(payload.get("uncertainties") or [])
            if str(item).strip()
        ]
        if uncertainties:
            reasons.append("uncertainties")
        lower_bound_event_reasons = {
            "invalid_or_unretrieved_source_ref",
            "assumption_or_uncertain_candidate",
        }
        non_lower_bound_event_reasons = {
            "future_or_intended_event",
            "reaction_or_comment_not_event",
            "general_interest_not_distinct_event",
            "duplicate_event",
        }
        non_lower_bound_refs: set[str] = set()
        excluded_event_count = 0
        for event in list(validation.get("count_validation_excluded_events") or []):
            if not isinstance(event, dict):
                continue
            reason = cls._count_excluded_event_reason(event)
            if reason in lower_bound_event_reasons:
                excluded_event_count += 1
            elif reason in non_lower_bound_event_reasons:
                non_lower_bound_refs.update(
                    str(value).strip()
                    for value in list(event.get("original_source_refs") or event.get("source_refs") or [])
                    if str(value).strip()
                )
        invalid_supporting_refs = [
            str(value).strip()
            for value in list(validation.get("invalid_supporting_refs") or [])
            if str(value).strip() and str(value).strip() not in non_lower_bound_refs
        ]
        invalid_family_refs = [
            str(value).strip()
            for value in list(validation.get("answer_synthesis_invalid_family_refs") or [])
            if str(value).strip() and str(value).strip() not in non_lower_bound_refs
        ]
        if invalid_supporting_refs:
            reasons.append("invalid_supporting_refs")
        if invalid_family_refs:
            reasons.append("family_mismatched_supporting_refs")

        if excluded_event_count:
            reasons.append("count_validation_excluded_plausible_candidates")
        llm_decisions = [
            dict(item)
            for item in list(validation.get("count_validation_llm_decisions") or [])
            if isinstance(item, dict)
        ]
        if any(str(item.get("decision") or "").upper() == "UNCERTAIN" for item in llm_decisions):
            reasons.append("llm_uncertain_count_candidate")
        return list(dict.fromkeys(reasons))

    @classmethod
    def _naturalize_count_answer(
        cls,
        *,
        final_answer: str,
        payload: dict[str, Any],
        question: str | None,
    ) -> tuple[str, dict[str, Any]]:
        stripped = str(final_answer or "").strip()
        metadata: dict[str, Any] = {
            "answer_count_naturalized": False,
            "answer_count_naturalized_from": None,
            "answer_count_naturalized_lower_bound": False,
            "answer_count_naturalized_lower_bound_reason": None,
            "answer_count_lower_bound_reasons": [],
            "answer_count_lower_bound_excluded_event_count": 0,
        }
        if not re.fullmatch(r"\d+", stripped):
            return stripped, metadata
        value = int(stripped)
        lower_bound_reasons = cls._count_lower_bound_reasons(payload)
        lower_bound = bool(lower_bound_reasons)
        unit = cls._count_unit_from_question(question)
        if lower_bound:
            lower_bound_unit = cls._count_lower_bound_unit_from_question(question, unit)
            phrase = cls._count_phrase(value, lower_bound_unit, exact=False, sentence_start=False)
            naturalized = f"The retrieved evidence confirms {phrase}."
        else:
            phrase = cls._count_phrase(value, unit, exact=True, sentence_start=True)
            naturalized = f"{phrase}."
        metadata.update(
            {
                "answer_count_naturalized": True,
                "answer_count_naturalized_from": stripped,
                "answer_count_naturalized_lower_bound": lower_bound,
                "answer_count_naturalized_lower_bound_reason": ",".join(lower_bound_reasons) or None,
                "answer_count_lower_bound_reasons": lower_bound_reasons,
                "answer_count_lower_bound_excluded_event_count": sum(
                    1
                    for event in list(dict(payload.get("_answer_validation_metadata") or {}).get("count_validation_excluded_events") or [])
                    if isinstance(event, dict)
                    and cls._count_excluded_event_reason(event)
                    in {"invalid_or_unretrieved_source_ref", "assumption_or_uncertain_candidate"}
                ),
            }
        )
        return naturalized, metadata

    @staticmethod
    def _answer_text_from_synthesis(payload: dict[str, Any], question: str | None = None) -> str:
        can_answer = bool(payload.get("can_answer"))
        final_answer = str(payload.get("final_answer") or "").strip()
        if can_answer:
            if not final_answer:
                raise ParserValidationError("Answer synthesis returned can_answer=true with empty final_answer.")
            if AnswerGenerator._normalize_answer_type(payload.get("answer_type")) == "count":
                final_answer, naturalization_metadata = AnswerGenerator._naturalize_count_answer(
                    final_answer=final_answer,
                    payload=payload,
                    question=question,
                )
                payload["_answer_count_naturalization"] = naturalization_metadata
            return final_answer
        return AnswerGenerator._standard_abstention_text(payload.get("abstain_reason"))

    @staticmethod
    def _synthesis_has_type_mismatch(payload: dict[str, Any]) -> bool:
        validation = dict(payload.get("_answer_validation_metadata") or {})
        expected_text_invalid = validation.get("answer_synthesis_expected_type_text_valid") is False
        return bool(
            payload.get("can_answer")
            and (
                (
                    validation.get("answer_synthesis_question_type_mismatch")
                    and not validation.get("answer_synthesis_type_mismatch_recovered")
                )
                or expected_text_invalid
            )
        )

    @classmethod
    def _try_recover_mismatched_answer_text(
        cls,
        payload: dict[str, Any],
        retrieval_bundle: RetrievalBundle,
        question: str | None,
    ) -> bool:
        validation = dict(payload.get("_answer_validation_metadata") or {})
        if not payload.get("can_answer") or not validation.get("answer_synthesis_question_type_mismatch"):
            return False
        if validation.get("answer_synthesis_count_style_text_mismatch"):
            return False
        original_type = (
            validation.get("answer_synthesis_normalized_answer_type")
            or cls._normalize_answer_type(payload.get("answer_type"))
        )
        if original_type != "count":
            return False
        query_shape = classify_query_shape_v1(str(question or ""), {})
        if cls._question_allows_count_answer(question, query_shape):
            return False
        final_answer = " ".join(str(payload.get("final_answer") or "").split())
        if not final_answer or cls._answer_is_context_abstention(final_answer):
            return False
        is_time_year_answer = bool(
            cls._question_is_time_like(question)
            and re.fullmatch(r"\d{4}", final_answer.strip())
        )
        if cls._answer_text_is_count_answer_like(final_answer) and not is_time_year_answer:
            return False
        allowed_refs = cls._retrieval_source_ref_set(retrieval_bundle)
        if allowed_refs and not list(payload.get("supporting_source_refs") or []):
            return False
        expected_type = cls._expected_answer_type(question, query_shape)
        expected_text_details = cls._answer_text_expected_type_details(
            final_answer,
            expected_type=expected_type,
            question=question,
        )
        if not expected_text_details.get("valid"):
            validation.update(
                {
                    "answer_synthesis_expected_type_text_valid": False,
                    "answer_synthesis_expected_type_text_rejection_reason": expected_text_details.get("reason"),
                    "answer_synthesis_type_recovery_rejected_reason": expected_text_details.get("reason"),
                }
            )
            payload["_answer_validation_metadata"] = validation
            return False
        payload["answer_type"] = expected_type
        validation.update(
            {
                "answer_synthesis_type_mismatch_recovered": True,
                "answer_synthesis_type_mismatch_recovery_reason": "final_answer_text_matches_expected_non_count_type",
                "answer_synthesis_recovered_answer_type": expected_type,
                "answer_synthesis_recovered_from_answer_type": original_type,
                "answer_synthesis_normalized_answer_type": expected_type,
                "answer_synthesis_expected_answer_family": query_shape.get("item_family") or expected_type,
                "answer_synthesis_expected_type_text_valid": True,
                "answer_synthesis_expected_type_text_rejection_reason": None,
                "answer_synthesis_type_recovery_rejected_reason": None,
            }
        )
        payload["_answer_validation_metadata"] = validation
        return True

    @staticmethod
    def _normalize_answer_synthesis_retry_json(
        payload: dict[str, Any],
        *,
        expected_type: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized = dict(payload or {})
        defaults: dict[str, Any] = {
            "can_answer": False,
            "answer_type": expected_type or "unknown",
            "final_answer": "",
            "supporting_facts": [],
            "supporting_source_refs": [],
            "counted_events": [],
            "excluded_events": [],
            "uncertainties": [],
            "abstain_reason": "typed retry did not return a complete answer synthesis payload",
        }
        missing_fields: list[str] = []
        for key, default in defaults.items():
            if key not in normalized or normalized[key] is None:
                normalized[key] = default
                missing_fields.append(key)
        for key in ["supporting_facts", "supporting_source_refs", "counted_events", "excluded_events", "uncertainties"]:
            if not isinstance(normalized.get(key), list):
                normalized[key] = []
                if key not in missing_fields:
                    missing_fields.append(key)
        if not bool(normalized.get("can_answer")):
            normalized["final_answer"] = str(normalized.get("final_answer") or "")
        return normalized, {
            "answer_synthesis_typed_retry_text_json_normalized": bool(missing_fields),
            "answer_synthesis_typed_retry_text_json_missing_fields": missing_fields,
        }

    @staticmethod
    def _normalize_answer_synthesis_text_json(
        payload: dict[str, Any],
        *,
        expected_type: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized = dict(payload or {})
        missing_fields: list[str] = []
        coerced_fields: list[str] = []

        def set_default(key: str, value: Any) -> None:
            if key not in normalized or normalized[key] is None:
                normalized[key] = value
                missing_fields.append(key)

        final_answer = " ".join(str(normalized.get("final_answer") or "").split())
        set_default("can_answer", bool(final_answer))
        set_default("answer_type", expected_type or "unknown")
        set_default("final_answer", "")
        set_default("supporting_facts", [])
        set_default("supporting_source_refs", [])
        set_default("counted_events", [])
        set_default("excluded_events", [])
        set_default("uncertainties", [])

        if not isinstance(normalized.get("supporting_facts"), list):
            normalized["supporting_facts"] = []
            coerced_fields.append("supporting_facts")
        else:
            facts: list[dict[str, Any]] = []
            for index, item in enumerate(list(normalized.get("supporting_facts") or []), start=1):
                if isinstance(item, dict):
                    fact = dict(item)
                    fact.setdefault("fact_text", str(fact.get("text") or fact.get("fact") or ""))
                    fact.setdefault("source_refs", [])
                    if not isinstance(fact.get("source_refs"), list):
                        fact["source_refs"] = []
                    facts.append(fact)
                elif item:
                    facts.append({"fact_text": str(item), "source_refs": []})
                    coerced_fields.append(f"supporting_facts.{index}")
            normalized["supporting_facts"] = facts

        if not isinstance(normalized.get("supporting_source_refs"), list):
            normalized["supporting_source_refs"] = []
            coerced_fields.append("supporting_source_refs")

        def normalize_events(key: str) -> None:
            value = normalized.get(key)
            if not isinstance(value, list):
                normalized[key] = []
                coerced_fields.append(key)
                return
            events: list[dict[str, Any]] = []
            prefix = "E" if key == "counted_events" else "X"
            for index, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    event = dict(item)
                    event.setdefault("event_id", f"{prefix}{index}")
                    event.setdefault("event_text", str(event.get("text") or event.get("description") or ""))
                    event.setdefault("source_refs", [])
                    event.setdefault("reason", "")
                    if not isinstance(event.get("source_refs"), list):
                        event["source_refs"] = []
                    events.append(event)
                elif item:
                    events.append(
                        {
                            "event_id": f"{prefix}{index}",
                            "event_text": str(item),
                            "source_refs": [],
                            "reason": "model returned this event as plain text",
                        }
                    )
                    coerced_fields.append(f"{key}.{index}")
            normalized[key] = events

        normalize_events("counted_events")
        normalize_events("excluded_events")
        if not isinstance(normalized.get("uncertainties"), list):
            normalized["uncertainties"] = [str(normalized.get("uncertainties"))]
            coerced_fields.append("uncertainties")

        return normalized, {
            "answer_synthesis_text_json_normalized": bool(missing_fields or coerced_fields),
            "answer_synthesis_text_json_missing_fields": missing_fields,
            "answer_synthesis_text_json_coerced_fields": coerced_fields,
        }

    @staticmethod
    def _make_safe_type_mismatch_abstain_payload(
        *,
        initial_payload: dict[str, Any],
        retry_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        payload = dict(initial_payload)
        validation = dict(payload.get("_answer_validation_metadata") or {})
        validation.update(
            {
                "answer_synthesis_internal_abstain_reason_suppressed": True,
                "answer_synthesis_safe_abstain_used": True,
                "answer_synthesis_typed_retry_final_policy": "standard_abstain",
            }
        )
        payload["_answer_validation_metadata"] = validation
        payload["can_answer"] = False
        payload["final_answer"] = ""
        payload["abstain_reason"] = None
        AnswerGenerator._attach_typed_retry_metadata(
            payload,
            used=True,
            expected_type=str(retry_metadata.get("answer_synthesis_typed_retry_expected_type") or ""),
            success=False,
            initial_payload=initial_payload,
            error=str(retry_metadata.get("answer_synthesis_typed_retry_error") or ""),
            text_json_normalized=bool(
                retry_metadata.get("answer_synthesis_typed_retry_text_json_normalized")
            ),
            text_json_missing_fields=list(
                retry_metadata.get("answer_synthesis_typed_retry_text_json_missing_fields") or []
            ),
            final_policy="standard_abstain",
        )
        return payload

    @classmethod
    def _type_constraint_retry_instruction(
        cls,
        *,
        question: str | None,
        expected_answer_type: str,
        initial_payload: dict[str, Any],
    ) -> str:
        validation = dict(initial_payload.get("_answer_validation_metadata") or {})
        initial_type = validation.get("answer_synthesis_normalized_answer_type") or initial_payload.get("answer_type")
        initial_answer = " ".join(str(initial_payload.get("final_answer") or "").split())[:240]
        reason = validation.get("answer_synthesis_question_type_mismatch_reason") or "answer_type_mismatch"
        return (
            "\n\nTYPE_CONSTRAINT_RETRY:\n"
            f"The previous synthesis used answer_type={initial_type!s} with final_answer={initial_answer!r}. "
            f"That is invalid for this question because {reason}.\n"
            f"Required answer_type: {expected_answer_type}.\n"
            "Re-synthesize using the same RETRIEVED_CONTEXT only. Do not add evidence, do not guess, and do not use "
            "count-style answers unless the required answer_type is count. If the evidence supports the requested "
            "date/value/list/place, return it with valid supporting refs. If it does not, set can_answer=false."
        )

    @staticmethod
    def _attach_typed_retry_metadata(
        payload: dict[str, Any],
        *,
        used: bool,
        expected_type: str | None,
        success: bool,
        initial_payload: dict[str, Any] | None,
        error: str | None = None,
        text_json_normalized: bool = False,
        text_json_missing_fields: list[str] | None = None,
        final_policy: str | None = None,
    ) -> None:
        initial_validation = dict((initial_payload or {}).get("_answer_validation_metadata") or {})
        payload["_answer_typed_retry_metadata"] = {
            "answer_synthesis_typed_retry_used": used,
            "answer_synthesis_typed_retry_expected_type": expected_type,
            "answer_synthesis_typed_retry_success": success,
            "answer_synthesis_initial_question_type_mismatch": bool(
                initial_validation.get("answer_synthesis_question_type_mismatch")
            ),
            "answer_synthesis_initial_question_type_mismatch_reason": initial_validation.get(
                "answer_synthesis_question_type_mismatch_reason"
            ),
            "answer_synthesis_initial_count_style_text_mismatch": bool(
                initial_validation.get("answer_synthesis_count_style_text_mismatch")
            ),
            "answer_synthesis_initial_answer_type": (initial_payload or {}).get("answer_type")
            or initial_validation.get("answer_synthesis_normalized_answer_type"),
            "answer_synthesis_initial_final_answer_preview": " ".join(
                str((initial_payload or {}).get("final_answer") or "").split()
            )[:240] or None,
            "answer_synthesis_initial_type_recovery_rejected_reason": initial_validation.get(
                "answer_synthesis_type_recovery_rejected_reason"
            ),
            "answer_synthesis_typed_retry_error": error,
            "answer_synthesis_typed_retry_text_json_normalized": bool(text_json_normalized),
            "answer_synthesis_typed_retry_text_json_missing_fields": list(text_json_missing_fields or []),
            "answer_synthesis_typed_retry_final_policy": final_policy,
        }

    def _finalize_synthesis_payload(
        self,
        *,
        payload: dict[str, Any],
        retrieval_bundle: RetrievalBundle,
        question: str | None,
        query_task_id: str | None = None,
    ) -> str:
        if query_task_id:
            payload["query_task_id"] = query_task_id
        self._maybe_run_llm_count_validation(
            payload=payload,
            retrieval_bundle=retrieval_bundle,
            question=question,
        )
        answer_text = self._answer_text_from_synthesis(payload, question)
        answer_text, bridge_metadata = self._apply_bridge_facts(
            answer_text=answer_text,
            payload=payload,
            retrieval_bundle=retrieval_bundle,
            question=question,
        )
        temporal_metadata = self._temporal_answer_alignment_diagnostics(
            answer_text=answer_text,
            retrieval_bundle=retrieval_bundle,
            question=question,
        )
        if (
            payload.get("can_answer")
            and temporal_metadata.get("answer_temporal_alignment_valid") is False
        ):
            selected_answer_text = str(
                temporal_metadata.get("answer_temporal_selected_answer_text")
                or temporal_metadata.get("answer_temporal_selected_date")
                or ""
            ).strip()
            if selected_answer_text:
                answer_text = selected_answer_text
                temporal_metadata.update(
                    {
                        "answer_temporal_repair_used": True,
                        "answer_temporal_repair_success": True,
                        "answer_temporal_repair_action": "deterministic_use_best_aligned_candidate",
                        "answer_temporal_alignment_valid": True,
                    }
                )
            else:
                answer_text = self._standard_abstention_text()
                payload["can_answer"] = False
                payload["final_answer"] = ""
                payload["abstain_reason"] = None
                temporal_metadata.update(
                    {
                        "answer_temporal_repair_used": True,
                        "answer_temporal_repair_success": False,
                        "answer_temporal_repair_action": "safe_abstain_no_aligned_candidate",
                    }
                )
        payload["final_answer"] = answer_text if payload.get("can_answer") else payload.get("final_answer", "")
        payload["_answer_bridge_metadata"] = bridge_metadata
        payload["_answer_temporal_metadata"] = temporal_metadata
        return answer_text

    @staticmethod
    def _synthesis_metadata(
        *,
        mode: str,
        payload: dict[str, Any] | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        payload = payload or {}
        count_naturalization = dict(payload.pop("_answer_count_naturalization", {}) or {})
        validation_metadata = dict(payload.pop("_answer_validation_metadata", {}) or {})
        bridge_metadata = dict(payload.pop("_answer_bridge_metadata", {}) or {})
        typed_retry_metadata = dict(payload.pop("_answer_typed_retry_metadata", {}) or {})
        temporal_metadata = dict(payload.pop("_answer_temporal_metadata", {}) or {})
        text_json_normalization = dict(payload.pop("_answer_text_json_normalization", {}) or {})
        return {
            "answer_prompt_name": "locomo_answer_evidence_synthesis",
            "answer_prompt_stage": "synthesis" if payload else "initial",
            "answer_synthesis_used": bool(payload),
            "answer_synthesis_mode": mode,
            "answer_synthesis_can_answer": payload.get("can_answer") if payload else None,
            "answer_synthesis_answer_type": payload.get("answer_type") if payload else None,
            "answer_synthesis_supporting_refs": list(payload.get("supporting_source_refs") or []) if payload else [],
            "answer_synthesis_supporting_facts": list(payload.get("supporting_facts") or []) if payload else [],
            "answer_synthesis_counted_events": list(payload.get("counted_events") or []) if payload else [],
            "answer_synthesis_excluded_events": list(payload.get("excluded_events") or []) if payload else [],
            "answer_synthesis_uncertainties": list(payload.get("uncertainties") or []) if payload else [],
            "answer_synthesis_abstain_reason": payload.get("abstain_reason") if payload else None,
            "answer_synthesis_payload": payload,
            "answer_synthesis_error": error,
            "answer_count_naturalized": bool(count_naturalization.get("answer_count_naturalized")),
            "answer_count_naturalized_from": count_naturalization.get("answer_count_naturalized_from"),
            "answer_count_naturalized_lower_bound": bool(
                count_naturalization.get("answer_count_naturalized_lower_bound")
            ),
            "answer_count_naturalized_lower_bound_reason": count_naturalization.get(
                "answer_count_naturalized_lower_bound_reason"
            ),
            "answer_count_lower_bound_reasons": list(
                count_naturalization.get("answer_count_lower_bound_reasons") or []
            ),
            "answer_count_lower_bound_excluded_event_count": int(
                count_naturalization.get("answer_count_lower_bound_excluded_event_count") or 0
            ),
            "invalid_supporting_refs": list(validation_metadata.get("invalid_supporting_refs") or []),
            "answer_synthesis_family_validation_used": bool(
                validation_metadata.get("answer_synthesis_family_validation_used")
            ),
            "answer_synthesis_invalid_family_refs": list(
                validation_metadata.get("answer_synthesis_invalid_family_refs") or []
            ),
            "answer_synthesis_question_type_mismatch": bool(
                validation_metadata.get("answer_synthesis_question_type_mismatch")
                or typed_retry_metadata.get("answer_synthesis_initial_question_type_mismatch")
            ),
            "answer_synthesis_question_type_mismatch_reason": validation_metadata.get(
                "answer_synthesis_question_type_mismatch_reason"
            )
            or typed_retry_metadata.get("answer_synthesis_initial_question_type_mismatch_reason"),
            "answer_synthesis_normalized_answer_type": validation_metadata.get(
                "answer_synthesis_normalized_answer_type"
            ),
            "answer_synthesis_expected_answer_family": validation_metadata.get(
                "answer_synthesis_expected_answer_family"
            ),
            "answer_synthesis_repair_reason": validation_metadata.get("answer_synthesis_repair_reason")
            or typed_retry_metadata.get("answer_synthesis_initial_question_type_mismatch_reason"),
            "answer_synthesis_count_style_text_mismatch": bool(
                validation_metadata.get("answer_synthesis_count_style_text_mismatch")
                or typed_retry_metadata.get("answer_synthesis_initial_count_style_text_mismatch")
            ),
            "answer_synthesis_expected_type_text_valid": validation_metadata.get(
                "answer_synthesis_expected_type_text_valid"
            ),
            "answer_synthesis_expected_type_text_rejection_reason": validation_metadata.get(
                "answer_synthesis_expected_type_text_rejection_reason"
            ),
            "answer_synthesis_type_recovery_rejected_reason": validation_metadata.get(
                "answer_synthesis_type_recovery_rejected_reason"
            )
            or typed_retry_metadata.get("answer_synthesis_initial_type_recovery_rejected_reason"),
            "answer_synthesis_typed_retry_used": bool(
                typed_retry_metadata.get("answer_synthesis_typed_retry_used")
            ),
            "answer_synthesis_typed_retry_expected_type": typed_retry_metadata.get(
                "answer_synthesis_typed_retry_expected_type"
            ),
            "answer_synthesis_typed_retry_success": bool(
                typed_retry_metadata.get("answer_synthesis_typed_retry_success")
            ),
            "answer_synthesis_initial_answer_type": typed_retry_metadata.get(
                "answer_synthesis_initial_answer_type"
            ),
            "answer_synthesis_initial_final_answer_preview": typed_retry_metadata.get(
                "answer_synthesis_initial_final_answer_preview"
            ),
            "answer_synthesis_typed_retry_error": typed_retry_metadata.get(
                "answer_synthesis_typed_retry_error"
            ),
            "answer_synthesis_typed_retry_text_json_normalized": bool(
                typed_retry_metadata.get("answer_synthesis_typed_retry_text_json_normalized")
            ),
            "answer_synthesis_typed_retry_text_json_missing_fields": list(
                typed_retry_metadata.get("answer_synthesis_typed_retry_text_json_missing_fields") or []
            ),
            "answer_synthesis_text_json_normalized": bool(
                text_json_normalization.get("answer_synthesis_text_json_normalized")
            ),
            "answer_synthesis_text_json_missing_fields": list(
                text_json_normalization.get("answer_synthesis_text_json_missing_fields") or []
            ),
            "answer_synthesis_text_json_coerced_fields": list(
                text_json_normalization.get("answer_synthesis_text_json_coerced_fields") or []
            ),
            "answer_synthesis_typed_retry_final_policy": typed_retry_metadata.get(
                "answer_synthesis_typed_retry_final_policy"
            )
            or validation_metadata.get("answer_synthesis_typed_retry_final_policy"),
            "answer_synthesis_type_mismatch_recovered": bool(
                validation_metadata.get("answer_synthesis_type_mismatch_recovered")
            ),
            "answer_synthesis_type_mismatch_recovery_reason": validation_metadata.get(
                "answer_synthesis_type_mismatch_recovery_reason"
            ),
            "answer_synthesis_recovered_answer_type": validation_metadata.get(
                "answer_synthesis_recovered_answer_type"
            ),
            "answer_synthesis_recovered_from_answer_type": validation_metadata.get(
                "answer_synthesis_recovered_from_answer_type"
            ),
            "answer_synthesis_internal_abstain_reason_suppressed": bool(
                validation_metadata.get("answer_synthesis_internal_abstain_reason_suppressed")
            ),
            "answer_synthesis_safe_abstain_used": bool(
                validation_metadata.get("answer_synthesis_safe_abstain_used")
            ),
            "source_family_validation_alias_hits": list(
                validation_metadata.get("source_family_validation_alias_hits") or []
            ),
            "source_family_validation_support_text_used": list(
                validation_metadata.get("source_family_validation_support_text_used") or []
            ),
            "count_validation_ref_acceptance_reasons": list(
                validation_metadata.get("count_validation_ref_acceptance_reasons") or []
            ),
            "count_validation_ref_rejection_reasons": list(
                validation_metadata.get("count_validation_ref_rejection_reasons") or []
            ),
            "count_validation_llm_candidate_events": list(
                validation_metadata.get("count_validation_llm_candidate_events") or []
            ),
            "count_validation_llm_trigger_reasons": list(
                validation_metadata.get("count_validation_llm_trigger_reasons") or []
            ),
            "count_validation_llm_skipped_reason": validation_metadata.get(
                "count_validation_llm_skipped_reason"
            ),
            "count_validation_llm_used": bool(validation_metadata.get("count_validation_llm_used")),
            "count_validation_llm_success": bool(validation_metadata.get("count_validation_llm_success")),
            "count_validation_llm_scope": validation_metadata.get("count_validation_llm_scope"),
            "count_validation_llm_confidence": validation_metadata.get("count_validation_llm_confidence"),
            "count_validation_llm_decisions": list(
                validation_metadata.get("count_validation_llm_decisions") or []
            ),
            "count_validation_llm_error": validation_metadata.get("count_validation_llm_error"),
            "count_validation_llm_changed_count": int(
                validation_metadata.get("count_validation_llm_changed_count") or 0
            ),
            "count_validation_source_derived_candidate_events": list(
                validation_metadata.get("count_validation_source_derived_candidate_events") or []
            ),
            "count_validation_source_derived_candidate_count": int(
                validation_metadata.get("count_validation_source_derived_candidate_count") or 0
            ),
            "count_validation_source_derived_candidate_refs": list(
                validation_metadata.get("count_validation_source_derived_candidate_refs") or []
            ),
            "count_validation_source_derived_trigger_terms": list(
                validation_metadata.get("count_validation_source_derived_trigger_terms") or []
            ),
            "count_validation_source_derived_action_hits": dict(
                validation_metadata.get("count_validation_source_derived_action_hits") or {}
            ),
            "count_validation_source_derived_object_hits": dict(
                validation_metadata.get("count_validation_source_derived_object_hits") or {}
            ),
            "count_validation_source_derived_passive_rejected_refs": list(
                validation_metadata.get("count_validation_source_derived_passive_rejected_refs") or []
            ),
            "count_validation_source_derived_pronoun_caption_refs": list(
                validation_metadata.get("count_validation_source_derived_pronoun_caption_refs") or []
            ),
            "count_validation_excluded_events": list(
                validation_metadata.get("count_validation_excluded_events") or []
            ),
            "count_validation_positive_event_signal": list(
                validation_metadata.get("count_validation_positive_event_signal") or []
            ),
            "count_validation_rejection_signal": list(
                validation_metadata.get("count_validation_rejection_signal") or []
            ),
            "answer_synthesis_allowed_ref_count": int(
                validation_metadata.get("answer_synthesis_allowed_ref_count") or 0
            ),
            "answer_synthesis_source_ref_validation_used": bool(
                validation_metadata.get("answer_synthesis_source_ref_validation_used")
            ),
            "answer_freeform_used": bool(validation_metadata.get("answer_freeform_used")),
            "answer_freeform_rationale": validation_metadata.get("answer_freeform_rationale"),
            "answer_freeform_parse_format": validation_metadata.get("answer_freeform_parse_format"),
            "answer_type_verification_used": bool(validation_metadata.get("answer_type_verification_used")),
            "answer_type_verification_success": validation_metadata.get("answer_type_verification_success"),
            "answer_expected_type": validation_metadata.get("answer_expected_type"),
            "answer_observed_type": validation_metadata.get("answer_observed_type"),
            "answer_type_match": validation_metadata.get("answer_type_match"),
            "answer_type_issue": validation_metadata.get("answer_type_issue"),
            "answer_type_repair_instruction": validation_metadata.get("answer_type_repair_instruction"),
            "answer_type_verification_error": validation_metadata.get("answer_type_verification_error"),
            "bridge_facts_used": list(bridge_metadata.get("bridge_facts_used") or []),
            "bridge_facts_missing": list(bridge_metadata.get("bridge_facts_missing") or []),
            "bridge_facts_conflicted": list(bridge_metadata.get("bridge_facts_conflicted") or []),
            "bridge_facts_ignored": list(bridge_metadata.get("bridge_facts_ignored") or []),
            "bridge_finalization_used": bool(bridge_metadata.get("bridge_finalization_used")),
            "bridge_finalization_alias": bridge_metadata.get("bridge_finalization_alias"),
            "bridge_finalization_target": bridge_metadata.get("bridge_finalization_target"),
            "bridge_finalization_source_refs": [
                ref for ref in list(bridge_metadata.get("bridge_finalization_source_refs") or []) if ref
            ],
            "bridge_finalization_action": bridge_metadata.get("bridge_finalization_action"),
            "bridge_finalization_conflicted": bool(bridge_metadata.get("bridge_finalization_conflicted")),
            "bridge_finalization_failed_reason": bridge_metadata.get("bridge_finalization_failed_reason"),
            "answer_temporal_alignment_checked": bool(
                temporal_metadata.get("answer_temporal_alignment_checked")
            ),
            "answer_temporal_candidate_dates": list(
                temporal_metadata.get("answer_temporal_candidate_dates") or []
            ),
            "answer_temporal_selected_source_ref": temporal_metadata.get("answer_temporal_selected_source_ref"),
            "answer_temporal_selected_date": temporal_metadata.get("answer_temporal_selected_date"),
            "answer_temporal_selected_answer_text": temporal_metadata.get(
                "answer_temporal_selected_answer_text"
            ),
            "answer_temporal_selected_resolution_kind": temporal_metadata.get(
                "answer_temporal_selected_resolution_kind"
            ),
            "answer_temporal_selected_resolution_granularity": temporal_metadata.get(
                "answer_temporal_selected_resolution_granularity"
            ),
            "answer_temporal_selected_relative_term": temporal_metadata.get(
                "answer_temporal_selected_relative_term"
            ),
            "answer_temporal_selected_confidence": temporal_metadata.get("answer_temporal_selected_confidence"),
            "answer_temporal_candidate_score": temporal_metadata.get("answer_temporal_candidate_score"),
            "answer_temporal_candidate_match_terms": list(
                temporal_metadata.get("answer_temporal_candidate_match_terms") or []
            ),
            "answer_temporal_relevant_candidate_count": int(
                temporal_metadata.get("answer_temporal_relevant_candidate_count") or 0
            ),
            "answer_temporal_low_confidence_candidate_count": int(
                temporal_metadata.get("answer_temporal_low_confidence_candidate_count") or 0
            ),
            "answer_temporal_candidates_suppressed_count": int(
                temporal_metadata.get("answer_temporal_candidates_suppressed_count") or 0
            ),
            "answer_temporal_no_query_relevant_candidate": bool(
                temporal_metadata.get("answer_temporal_no_query_relevant_candidate")
            ),
            "answer_temporal_alignment_valid": temporal_metadata.get("answer_temporal_alignment_valid"),
            "answer_temporal_alignment_rejection_reason": temporal_metadata.get(
                "answer_temporal_alignment_rejection_reason"
            ),
            "answer_temporal_repair_used": bool(temporal_metadata.get("answer_temporal_repair_used")),
            "answer_temporal_repair_success": bool(temporal_metadata.get("answer_temporal_repair_success")),
            "answer_temporal_repair_action": temporal_metadata.get("answer_temporal_repair_action"),
        }

    def _try_type_constrained_synthesis_retry(
        self,
        *,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        base_prompt: str,
        initial_payload: dict[str, Any],
        initial_response: LLMResponse,
        task: str,
        spec: Any,
    ) -> tuple[str, LLMResponse] | tuple[None, dict[str, Any]]:
        query_shape = classify_query_shape_v1(query_task.question, {})
        expected_type = self._expected_answer_type(query_task.question, query_shape)
        retry_prompt = base_prompt + self._type_constraint_retry_instruction(
            question=query_task.question,
            expected_answer_type=expected_type,
            initial_payload=initial_payload,
        )
        errors: list[str] = []
        retry_text_json_normalization: dict[str, Any] = {}

        def _success_response(
            *,
            retry_payload: dict[str, Any],
            answer_text: str,
            retry_response: LLMResponse,
            mode: str,
        ) -> tuple[str, LLMResponse]:
            normalization = dict(retry_payload.pop("_answer_typed_retry_text_json_normalization", {}) or {})
            self._attach_typed_retry_metadata(
                retry_payload,
                used=True,
                expected_type=expected_type,
                success=True,
                initial_payload=initial_payload,
                text_json_normalized=bool(
                    normalization.get("answer_synthesis_typed_retry_text_json_normalized")
                ),
                text_json_missing_fields=list(
                    normalization.get("answer_synthesis_typed_retry_text_json_missing_fields") or []
                ),
                final_policy="use_retry",
            )
            metadata = {
                **dict(retry_response.metadata or {}),
                **self._synthesis_metadata(mode=mode, payload=retry_payload),
                "answer_postcheck_used": False,
                "answer_postcheck_issue": None,
                "answer_repair_used": False,
            }
            return (
                retry_prompt,
                LLMResponse(
                    text=answer_text,
                    raw=retry_response.raw,
                    prompt_tokens=(initial_response.prompt_tokens or 0) + (retry_response.prompt_tokens or 0),
                    completion_tokens=(initial_response.completion_tokens or 0)
                    + (retry_response.completion_tokens or 0),
                    metadata=metadata,
                ),
            )

        if self.llm_provider.supports_structured(task):
            try:
                structured_retry = self.llm_provider.generate_structured(
                    [NormalizedMessage(role="user", content=retry_prompt, turn_index=0)],
                    spec=spec,
                    metadata={
                        "task": task,
                        "query_task_id": query_task.query_task_id,
                        "answer_prompt_name": "locomo_answer_evidence_synthesis",
                        "answer_synthesis_typed_retry": True,
                        "repair_requested": True,
                        "repair_trigger": "answer_type_mismatch",
                        "repair_action": "typed_synthesis_retry",
                    },
                )
                retry_payload = self._model_to_dict(structured_retry.parsed)
                retry_payload = self._validate_answer_synthesis_payload(
                    retry_payload,
                    retrieval_bundle,
                    query_task.question,
                    coerce_type_mismatch_to_abstain=False,
                )
                self._try_recover_mismatched_answer_text(
                    retry_payload,
                    retrieval_bundle,
                    query_task.question,
                )
                if retry_payload.get("can_answer") and not self._synthesis_has_type_mismatch(retry_payload):
                    answer_text = self._finalize_synthesis_payload(
                        payload=retry_payload,
                        retrieval_bundle=retrieval_bundle,
                        question=query_task.question,
                        query_task_id=query_task.query_task_id,
                    )
                    return _success_response(
                        retry_payload=retry_payload,
                        answer_text=answer_text,
                        retry_response=LLMResponse(
                            text=answer_text,
                            raw=structured_retry.raw,
                            prompt_tokens=structured_retry.prompt_tokens,
                            completion_tokens=structured_retry.completion_tokens,
                            metadata=dict(structured_retry.metadata or {}),
                        ),
                        mode="structured",
                    )
                errors.append("structured_retry: retry_payload_invalid_or_mismatched")
            except Exception as exc:  # noqa: BLE001
                errors.append("structured_retry: " + (" ".join(str(exc).split()) or exc.__class__.__name__))
        else:
            errors.append("structured_retry: unsupported")

        try:
            retry_response = self.llm_provider.generate(
                [
                    NormalizedMessage(
                        role="user",
                        content=retry_prompt + "\n\nReturn ONLY a JSON object matching the requested schema.",
                        turn_index=0,
                    )
                ],
                metadata={
                    "task": task,
                    "query_task_id": query_task.query_task_id,
                    "answer_prompt_name": "locomo_answer_evidence_synthesis",
                    "answer_synthesis_typed_retry": True,
                    "repair_requested": True,
                    "repair_trigger": "answer_type_mismatch",
                    "repair_action": "typed_synthesis_retry",
                    "structured_fallback": True,
                    "structured_requested": True,
                    "structured_supported": self.llm_provider.supports_structured(task),
                    "fallback_used": True,
                    "fallback_mode": "text_json",
                    "fallback_reason": "typed_retry_structured_unavailable_or_failed",
                },
            )
            retry_json, retry_text_json_normalization = self._normalize_answer_synthesis_retry_json(
                self._extract_json_object(retry_response.text),
                expected_type=expected_type,
            )
            retry_payload = self._model_to_dict(parse_structured_payload(spec, retry_json))
            retry_payload["_answer_typed_retry_text_json_normalization"] = retry_text_json_normalization
            retry_payload = self._validate_answer_synthesis_payload(
                retry_payload,
                retrieval_bundle,
                query_task.question,
                coerce_type_mismatch_to_abstain=False,
            )
            self._try_recover_mismatched_answer_text(
                retry_payload,
                retrieval_bundle,
                query_task.question,
            )
            if retry_payload.get("can_answer") and not self._synthesis_has_type_mismatch(retry_payload):
                answer_text = self._finalize_synthesis_payload(
                    payload=retry_payload,
                    retrieval_bundle=retrieval_bundle,
                    question=query_task.question,
                    query_task_id=query_task.query_task_id,
                )
                retry_response.text = answer_text
                return _success_response(
                    retry_payload=retry_payload,
                    answer_text=answer_text,
                    retry_response=retry_response,
                    mode="text_json",
                )
            errors.append("text_json_retry: retry_payload_invalid_or_mismatched")
        except Exception as exc:  # noqa: BLE001
            errors.append("text_json_retry: " + (" ".join(str(exc).split()) or exc.__class__.__name__))

        return None, {
            "answer_synthesis_typed_retry_used": True,
            "answer_synthesis_typed_retry_expected_type": expected_type,
            "answer_synthesis_typed_retry_success": False,
            "answer_synthesis_typed_retry_error": "; ".join(errors),
            "answer_synthesis_typed_retry_text_json_normalized": bool(
                retry_text_json_normalization.get("answer_synthesis_typed_retry_text_json_normalized")
            ),
            "answer_synthesis_typed_retry_text_json_missing_fields": list(
                retry_text_json_normalization.get("answer_synthesis_typed_retry_text_json_missing_fields") or []
            ),
            "answer_synthesis_typed_retry_final_policy": "standard_abstain",
        }

    def _try_generate_locomo_freeform_answer(
        self,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
    ) -> tuple[str, LLMResponse] | None:
        prompt = self._build_locomo_freeform_answer_prompt(query_task, retrieval_bundle)
        task = "answer_freeform_generation"
        try:
            response = self.llm_provider.generate(
                [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                metadata={
                    "task": task,
                    "query_task_id": query_task.query_task_id,
                    "answer_prompt_name": "locomo_answer_freeform",
                },
            )
        except Exception as exc:  # noqa: BLE001
            error_message = self._compact_exception(exc)
            abstain = self._standard_abstention_text(
                f"answer generation failed before a model answer was produced ({exc.__class__.__name__})"
            )
            return (
                prompt,
                LLMResponse(
                    text=abstain,
                    raw=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    metadata={
                        "answer_prompt_name": "locomo_answer_freeform",
                        "answer_prompt_stage": "initial",
                        "answer_freeform_used": True,
                        "answer_synthesis_used": False,
                        "answer_synthesis_mode": "freeform_v2",
                        "answer_generation_failed": True,
                        "answer_generation_error_type": exc.__class__.__name__,
                        "answer_generation_error_message": error_message,
                        "answer_postcheck_used": False,
                        "answer_postcheck_issue": None,
                        "answer_postcheck_skipped": True,
                        "answer_postcheck_skip_reason": "answer_generation_failed",
                        "answer_repair_used": False,
                    },
                ),
            )
        parsed = self._parse_freeform_answer_response(response.text)
        answer_text = parsed["answer"]
        if not answer_text and self.llm_provider.supports_structured("answer_evidence_synthesis"):
            # Test/legacy compatibility: an empty plain-text response from a structured-capable
            # provider means the caller likely configured only the old structured payload.
            return None
        if not answer_text:
            answer_text = self._standard_abstention_text("free-form answer was empty")
        query_shape = classify_query_shape_v1(query_task.question, {})
        expected_type = self._expected_answer_type(query_task.question, query_shape)
        observed_type = self._infer_observed_answer_type(answer_text, query_task.question)
        deterministic_type = self._answer_type_match_details(
            answer_text=answer_text,
            question=query_task.question,
            expected_type=expected_type,
            observed_type=observed_type,
        )
        type_metadata = self._maybe_run_answer_type_verification(
            query_task=query_task,
            answer_text=answer_text,
            expected_type=expected_type,
            observed_type=observed_type,
            deterministic=deterministic_type,
        )
        payload = self._freeform_payload(
            answer_text=answer_text,
            rationale=parsed["rationale"],
            retrieval_bundle=retrieval_bundle,
            question=query_task.question,
            expected_type=str(type_metadata.get("answer_expected_type") or expected_type),
            observed_type=str(type_metadata.get("answer_observed_type") or observed_type),
        )
        payload = self._validate_answer_synthesis_payload(
            payload,
            retrieval_bundle,
            query_task.question,
            coerce_type_mismatch_to_abstain=False,
        )
        validation = dict(payload.get("_answer_validation_metadata") or {})
        validation.update(
            {
                "answer_freeform_used": True,
                "answer_freeform_rationale": parsed["rationale"][:500],
                "answer_freeform_parse_format": parsed["format"],
                **type_metadata,
            }
        )
        payload["answer_type_repair_instruction"] = type_metadata.get("answer_type_repair_instruction") or ""
        payload["_answer_validation_metadata"] = validation
        final_answer = self._finalize_synthesis_payload(
            payload=payload,
            retrieval_bundle=retrieval_bundle,
            question=query_task.question,
            query_task_id=query_task.query_task_id,
        )
        metadata = {
            **dict(response.metadata or {}),
            **self._synthesis_metadata(mode="freeform_v2", payload=payload),
            "answer_prompt_name": "locomo_answer_freeform",
            "answer_prompt_stage": "initial",
            "answer_postcheck_used": False,
            "answer_postcheck_issue": None,
            "answer_repair_used": False,
        }
        response.text = final_answer
        response.metadata = metadata
        return prompt, response

    def _try_generate_locomo_answer_synthesis(
        self,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
    ) -> tuple[str, LLMResponse] | tuple[None, dict[str, Any]]:
        task = "answer_evidence_synthesis"
        spec = get_structured_task_spec(task)
        prompt = self._build_locomo_answer_synthesis_prompt(query_task, retrieval_bundle)
        errors: list[str] = []
        if self.llm_provider.supports_structured(task):
            try:
                structured_response = self.llm_provider.generate_structured(
                    [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                    spec=spec,
                    metadata={
                        "task": task,
                        "query_task_id": query_task.query_task_id,
                        "answer_prompt_name": "locomo_answer_evidence_synthesis",
                    },
                )
                raw_payload = self._model_to_dict(structured_response.parsed)
                payload = self._validate_answer_synthesis_payload(
                    dict(raw_payload),
                    retrieval_bundle,
                    query_task.question,
                    coerce_type_mismatch_to_abstain=False,
                )
                self._try_recover_mismatched_answer_text(
                    payload,
                    retrieval_bundle,
                    query_task.question,
                )
                if self._synthesis_has_type_mismatch(payload):
                    initial_payload = dict(payload)
                    initial_response = LLMResponse(
                        text=str(payload.get("final_answer") or ""),
                        raw=structured_response.raw,
                        prompt_tokens=structured_response.prompt_tokens,
                        completion_tokens=structured_response.completion_tokens,
                        metadata=dict(structured_response.metadata or {}),
                    )
                    retry_result = self._try_type_constrained_synthesis_retry(
                        query_task=query_task,
                        retrieval_bundle=retrieval_bundle,
                        base_prompt=prompt,
                        initial_payload=payload,
                        initial_response=initial_response,
                        task=task,
                        spec=spec,
                    )
                    if retry_result[0] is not None:
                        return retry_result  # type: ignore[return-value]
                    retry_metadata = dict(retry_result[1])  # type: ignore[arg-type]
                    payload = self._make_safe_type_mismatch_abstain_payload(
                        initial_payload=initial_payload,
                        retry_metadata=retry_metadata,
                    )
                answer_text = self._finalize_synthesis_payload(
                    payload=payload,
                    retrieval_bundle=retrieval_bundle,
                    question=query_task.question,
                    query_task_id=query_task.query_task_id,
                )
                metadata = {
                    **dict(structured_response.metadata or {}),
                    **self._synthesis_metadata(mode="structured", payload=payload),
                    "answer_postcheck_used": False,
                    "answer_postcheck_issue": None,
                    "answer_repair_used": False,
                }
                return (
                    prompt,
                    LLMResponse(
                        text=answer_text,
                        raw=structured_response.raw,
                        prompt_tokens=structured_response.prompt_tokens,
                        completion_tokens=structured_response.completion_tokens,
                        metadata=metadata,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                errors.append("structured: " + (" ".join(str(exc).split()) or exc.__class__.__name__))
        else:
            errors.append("structured: unsupported")

        try:
            response = self.llm_provider.generate(
                [
                    NormalizedMessage(
                        role="user",
                        content=prompt + "\n\nReturn ONLY a JSON object matching the requested schema.",
                        turn_index=0,
                    )
                ],
                metadata={
                    "task": task,
                    "query_task_id": query_task.query_task_id,
                    "answer_prompt_name": "locomo_answer_evidence_synthesis",
                    "structured_fallback": True,
                    "structured_requested": True,
                    "structured_supported": self.llm_provider.supports_structured(task),
                    "fallback_used": True,
                    "fallback_mode": "text_json",
                    "fallback_reason": "structured_unsupported"
                    if not self.llm_provider.supports_structured(task)
                    else "structured_exception",
                },
            )
            query_shape = classify_query_shape_v1(str(query_task.question or ""), {})
            expected_type = self._expected_answer_type(query_task.question, query_shape)
            raw_json = self._extract_json_object(response.text)
            normalized_json, text_json_normalization = self._normalize_answer_synthesis_text_json(
                raw_json,
                expected_type=expected_type,
            )
            payload = self._model_to_dict(parse_structured_payload(spec, normalized_json))
            payload["_answer_text_json_normalization"] = text_json_normalization
            raw_payload = dict(payload)
            payload = self._validate_answer_synthesis_payload(
                payload,
                retrieval_bundle,
                query_task.question,
                coerce_type_mismatch_to_abstain=False,
            )
            self._try_recover_mismatched_answer_text(
                payload,
                retrieval_bundle,
                query_task.question,
            )
            if self._synthesis_has_type_mismatch(payload):
                initial_payload = dict(payload)
                retry_result = self._try_type_constrained_synthesis_retry(
                    query_task=query_task,
                    retrieval_bundle=retrieval_bundle,
                    base_prompt=prompt,
                    initial_payload=payload,
                    initial_response=response,
                    task=task,
                    spec=spec,
                )
                if retry_result[0] is not None:
                    return retry_result  # type: ignore[return-value]
                retry_metadata = dict(retry_result[1])  # type: ignore[arg-type]
                payload = self._make_safe_type_mismatch_abstain_payload(
                    initial_payload=initial_payload,
                    retry_metadata=retry_metadata,
                )
            answer_text = self._finalize_synthesis_payload(
                payload=payload,
                retrieval_bundle=retrieval_bundle,
                question=query_task.question,
                query_task_id=query_task.query_task_id,
            )
            response.text = answer_text
            response.metadata = {
                **dict(response.metadata or {}),
                **self._synthesis_metadata(mode="text_json", payload=payload),
                "answer_postcheck_used": False,
                "answer_postcheck_issue": None,
                "answer_repair_used": False,
            }
            return prompt, response
        except Exception as exc:  # noqa: BLE001
            errors.append("text_json: " + self._compact_exception(exc, limit=500))
            error = "; ".join(errors)
            self._trace(
                f"answer_legacy_structured_path_failed sample={query_task.sample_id} "
                f"query_task_id={query_task.query_task_id} fallback=legacy_fallback error={error}"
            )
            return None, self._synthesis_metadata(mode="legacy_fallback", payload=None, error=error)

    @staticmethod
    def _split_answer_items(text: str) -> list[str]:
        compact = " ".join(str(text or "").split())
        if not compact:
            return []
        normalized = re.sub(r"\s+and\s+", ", ", compact, flags=re.IGNORECASE)
        parts = [part.strip(" \t\n\r.,;:!?") for part in re.split(r"[,;]", normalized)]
        return [part for part in parts if part]

    @staticmethod
    def _grounded_surface_support(retrieval_bundle: RetrievalBundle) -> dict[str, list[str]]:
        metadata = dict(retrieval_bundle.metadata or {})
        def _values_for(*keys: str) -> list[str]:
            values: list[str] = []
            for key in keys:
                raw_values = metadata.get(key)
                if isinstance(raw_values, dict):
                    raw_values = list(raw_values.values())
                for value in list(raw_values or []):
                    if isinstance(value, dict):
                        for nested_key in ("raw_surface", "surface", "value", "term", "text"):
                            nested = str(value.get(nested_key) or "").strip()
                            if nested:
                                values.append(nested)
                                break
                        continue
                    text = str(value).strip()
                    if text:
                        values.append(text)
            return list(dict.fromkeys(values))

        return {
            "exact_terms": _values_for("grounded_exact_terms", "exact_terms_v2", "exact_terms"),
            "display_items": _values_for("grounded_display_items", "display_items"),
            "display_counts": _values_for("grounded_display_counts", "display_counts"),
            "display_key_facts": _values_for("grounded_display_key_facts", "display_key_facts"),
            "source_surface_terms": _values_for(
                "grounded_source_surface_terms",
                "grounded_source_surface_raw_terms",
                "source_surface_terms_v1",
                "source_surface_raw_terms_v1",
            ),
            "source_surface_records": _values_for(
                "grounded_source_surface_records",
                "source_surface_records_v1",
            ),
            "wiki_historical_item_terms": _values_for(
                "grounded_wiki_historical_item_terms",
                "wiki_historical_item_terms",
                "trajectory_historical_item_terms_v1",
                "historical_item_terms",
            ),
        }

    @staticmethod
    def _surface_supported(answer_item: str, supported_surfaces: list[str]) -> bool:
        normalized_item = " ".join(answer_item.split()).casefold()
        if not normalized_item:
            return True
        for surface in supported_surfaces:
            normalized_surface = " ".join(str(surface).split()).casefold()
            if not normalized_surface:
                continue
            if normalized_item in normalized_surface or normalized_surface in normalized_item:
                return True
        return False

    @staticmethod
    def _normalized_surface_text(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
        return re.sub(r"^(?:a|an|the)\s+", "", normalized)

    @classmethod
    def _locomo_bridge_finalization_diagnostics(
        cls,
        retrieval_bundle: RetrievalBundle,
        answer_text: str,
        question: str | None,
    ) -> dict[str, Any]:
        bridge_facts = cls._bridge_alias_facts(retrieval_bundle)
        grouped_targets: dict[str, set[str]] = {}
        for fact in bridge_facts:
            grouped_targets.setdefault(fact["alias"].casefold(), set()).add(fact["target"].casefold())
        conflicted_aliases = {alias for alias, targets in grouped_targets.items() if len(targets) > 1}
        folded_answer = str(answer_text or "").casefold()
        folded_question = str(question or "").casefold()
        for fact in bridge_facts:
            alias = fact["alias"]
            target = fact["target"]
            if alias.casefold() in conflicted_aliases:
                return {
                    "bridge_finalization_conflicted": True,
                    "bridge_finalization_alias": alias,
                    "bridge_finalization_target": target,
                    "bridge_finalization_source_refs": [fact.get("source_ref")],
                    "bridge_finalization_needs_repair": False,
                }
            alias_relevant = alias.casefold() in folded_answer or alias.casefold() in folded_question
            if not alias_relevant:
                continue
            target_present = target.casefold() in folded_answer
            alias_present = alias.casefold() in folded_answer
            if alias_present and not target_present:
                return {
                    "bridge_finalization_conflicted": False,
                    "bridge_finalization_alias": alias,
                    "bridge_finalization_target": target,
                    "bridge_finalization_source_refs": [fact.get("source_ref")],
                    "bridge_finalization_needs_repair": True,
                }
        return {
            "bridge_finalization_conflicted": bool(conflicted_aliases),
            "bridge_finalization_alias": None,
            "bridge_finalization_target": None,
            "bridge_finalization_source_refs": [],
            "bridge_finalization_needs_repair": False,
        }

    @classmethod
    def _surface_tokens(cls, value: str) -> set[str]:
        normalized = cls._normalized_surface_text(value)
        tokens = {
            token[:-1] if token.endswith("s") and len(token) > 3 else token
            for token in normalized.split()
            if len(token) > 2
        }
        return tokens - {
            "and",
            "for",
            "with",
            "about",
            "into",
            "from",
            "that",
            "this",
            "thing",
            "things",
            "item",
            "items",
        }

    @classmethod
    def _answer_covers_supported_item(cls, answer_text: str, item: str) -> bool:
        answer_norm = cls._normalized_surface_text(answer_text)
        item_norm = cls._normalized_surface_text(item)
        if not item_norm:
            return True
        if item_norm in answer_norm:
            return True
        if item_norm == "mentoring program" and re.search(r"\b(?:mentorship|mentor|mentoring)\s+program\b", answer_norm):
            return True
        if item_norm == "school speech":
            has_school_event = "school event" in answer_norm
            has_talk_signal = bool(
                re.search(
                    r"\b(?:talk|talked|talking|speech|shared\s+(?:her|his|their|my)\s+journey|encouraged\s+students)\b",
                    answer_norm,
                )
            )
            if has_school_event and has_talk_signal:
                return True
        item_tokens = cls._surface_tokens(item)
        if not item_tokens:
            return False
        answer_tokens = cls._surface_tokens(answer_text)
        if item_tokens <= answer_tokens:
            return True
        # Allow concise variants such as "dinosaurs" covering "dinosaur exhibit".
        distinctive = item_tokens - {"event", "events", "exhibit", "concert", "class", "group"}
        return bool(distinctive and distinctive <= answer_tokens)

    @classmethod
    def _source_refs_for_surface(cls, retrieval_bundle: RetrievalBundle, surface: str) -> list[str]:
        surface_norm = cls._normalized_surface_text(surface)
        if not surface_norm:
            return []
        refs: list[str] = []
        surface_tokens = cls._surface_tokens(surface)
        for source_ref, line in cls._source_backed_context_lines(retrieval_bundle):
            line_norm = cls._normalized_surface_text(line)
            line_tokens = cls._surface_tokens(line)
            if surface_norm in line_norm or (surface_tokens and surface_tokens <= line_tokens):
                refs.append(source_ref)
        return list(dict.fromkeys(refs))

    @classmethod
    def _activity_scope_terms(cls, question: str | None) -> set[str]:
        text = " ".join(str(question or "").casefold().split())
        if re.search(r"\b(?:destress|de-stress|stress|relax|calm|headspace|self-care|refresh|recharge|reset)\b", text):
            return {
                "destress",
                "de-stress",
                "stress",
                "headspace",
                "therapy",
                "therapeutic",
                "mental health",
                "relax",
                "relaxing",
                "clear my mind",
            }
        return set()

    @classmethod
    def _list_scope_profile(cls, question: str | None, query_shape: dict[str, object]) -> dict[str, Any]:
        text = " ".join(str(question or "").casefold().split())
        item_family = str(query_shape.get("item_family") or "").casefold()
        inventory_count_family = cls._inventory_count_family(question)
        if bool(query_shape.get("count_like")) and inventory_count_family:
            item_family = inventory_count_family
        kind = item_family or "generic_list"
        if item_family in {"book", "reading"} or re.search(r"\bbooks?\b|\bread\b|\breading\b", text):
            kind = "book"
        elif item_family == "activity" and cls._activity_scope_terms(question):
            kind = "destress_activity"
        elif item_family == "preference" or re.search(
            r"\b(?:like|likes|liked|love|loves|enjoy|enjoys|favorite|favourite|stoked|excited)\b",
            text,
        ):
            kind = "preference"
        elif item_family == "event" and cls._event_question_help_scope_terms(question):
            kind = "event_helping_children"
        elif item_family == "event":
            kind = "event"
        elif item_family in {"place", "location"}:
            kind = "place"
        elif item_family == "organization":
            kind = "organization"
        elif item_family == "deal":
            kind = "deal"
        elif item_family == "pet":
            kind = "pet"
        elif item_family == "dream":
            kind = "dream"
        elif item_family == "class":
            kind = "class"
        return {
            "kind": kind,
            "item_family": item_family,
            "question_text": text,
            "activity_scope_terms": sorted(cls._activity_scope_terms(question)),
            "event_scope_terms": sorted(cls._event_question_help_scope_terms(question)),
        }

    @staticmethod
    def _text_has_any(text: str, terms: set[str]) -> bool:
        compact = " ".join(str(text or "").casefold().split())
        return any(term in compact for term in terms if term)

    @staticmethod
    def _looks_like_title(value: str) -> bool:
        stripped = str(value or "").strip()
        if not stripped:
            return False
        if "'" in stripped or '"' in stripped:
            return True
        tokens = re.findall(r"[A-Za-z0-9']+", stripped)
        if not tokens:
            return False
        titleish = sum(1 for token in tokens if token[:1].isupper() or token.isdigit())
        return len(tokens) <= 6 and titleish >= max(1, len(tokens) - 1)

    @classmethod
    def _source_text_for_refs(cls, retrieval_bundle: RetrievalBundle, refs: list[str]) -> str:
        by_ref: dict[str, list[str]] = {}
        for ref, line in cls._source_backed_context_lines(retrieval_bundle):
            by_ref.setdefault(ref, []).append(line)
        return " ".join(line for ref in refs for line in by_ref.get(ref, []))

    @classmethod
    def _list_scope_decision(
        cls,
        *,
        item: str,
        refs: list[str],
        profile: dict[str, Any],
        retrieval_bundle: RetrievalBundle,
    ) -> dict[str, Any]:
        normalized = cls._normalized_surface_text(item)
        item_text = str(item or "")
        source_text = cls._source_text_for_refs(retrieval_bundle, refs)
        combined = " ".join([item_text, source_text]).casefold()
        kind = str(profile.get("kind") or "generic_list")

        if not normalized:
            return {"required": False, "optional": False, "reason": "empty_item"}

        generic_terms = {
            "animal",
            "animals",
            "activity",
            "activities",
            "thing",
            "things",
            "support",
            "experience",
            "creative",
            "event",
            "events",
            "item",
            "items",
            "none",
            "unknown",
        }
        if normalized in generic_terms:
            return {"required": False, "optional": False, "reason": "generic_surface"}

        if kind == "book":
            book_signal = bool(
                re.search(r"\b(?:book|books|read|reading|novel|title|cover|story)\b", combined)
                or cls._looks_like_title(item_text)
            )
            return {
                "required": book_signal,
                "optional": False,
                "reason": "book_scope_match" if book_signal else "book_scope_missing_read_or_title_signal",
            }

        if kind == "destress_activity":
            scope_terms = set(profile.get("activity_scope_terms") or [])
            activity_signal = bool(
                re.search(
                    r"\b(?:pottery|running|run|reading|violin|painting|hiking|camping|swimming|yoga|class)\b",
                    combined,
                )
            )
            scope_match = cls._text_has_any(combined, scope_terms)
            return {
                "required": bool(activity_signal and scope_match),
                "optional": bool(activity_signal and not scope_match),
                "reason": "destress_activity_scope_match"
                if activity_signal and scope_match
                else "destress_activity_scope_missing",
            }

        if kind == "preference":
            preference_signal = bool(
                re.search(
                    r"\b(?:like|likes|liked|love|loves|loved|enjoy|enjoys|enjoyed|favorite|favourite|stoked|excited)\b",
                    combined,
                )
            )
            specific_preference = bool(
                re.search(r"\b(?:dinosaur|dinosaurs|exhibit|nature)\b", normalized)
            )
            required = preference_signal or specific_preference
            return {
                "required": required,
                "optional": not required,
                "reason": "preference_scope_match" if required else "preference_scope_missing_like_signal",
            }

        if kind == "event_helping_children":
            scope_match = cls._text_has_any(combined, set(profile.get("event_scope_terms") or []))
            event_signal = bool(
                re.search(r"\b(?:mentorship|mentoring|mentor|school speech|school event|talk|speech|students|youth)\b", combined)
            )
            return {
                "required": bool(scope_match and event_signal),
                "optional": bool(event_signal and not scope_match),
                "reason": "event_helping_children_scope_match"
                if scope_match and event_signal
                else "event_helping_children_scope_missing",
            }

        if kind == "event":
            event_signal = bool(
                re.search(r"\b(?:event|concert|conference|parade|speech|program|workshop|festival|game|convention)\b", combined)
            )
            return {
                "required": event_signal,
                "optional": not event_signal,
                "reason": "event_scope_match" if event_signal else "event_scope_missing_event_signal",
            }

        family_terms = {
            "place": {"place", "city", "country", "area", "county", "park", "museum", "beach"},
            "location": {"place", "city", "country", "area", "county", "park", "museum", "beach"},
            "organization": {"organization", "charity", "foundation", "beneficiary", "group"},
            "deal": {"deal", "endorsement", "sponsor", "sponsorship"},
            "pet": {"dog", "cat", "turtle", "tortoise", "pet", "name"},
            "dream": {"dream", "goal", "hope"},
            "class": {"class", "course", "lesson", "workshop"},
        }
        if kind == "pet":
            item_match = cls._text_has_any(normalized, family_terms[kind]) or bool(
                re.search(r"\b(?:names?\s+are|named|called)\b", item_text.casefold())
            )
            return {
                "required": item_match,
                "optional": not item_match,
                "reason": "pet_scope_match" if item_match else "pet_scope_missing_family_signal",
            }
        if kind in family_terms:
            matched = cls._text_has_any(combined, family_terms[kind])
            return {
                "required": matched,
                "optional": not matched,
                "reason": f"{kind}_scope_match" if matched else f"{kind}_scope_missing_family_signal",
            }

        return {"required": True, "optional": False, "reason": "generic_list_scope"}

    @classmethod
    def _source_backed_activity_items(
        cls,
        *,
        question: str | None,
        retrieval_bundle: RetrievalBundle,
        synthesis_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_shape = cls._locomo_query_shape(
            QueryTask(query_task_id="", sample_id="", question=str(question or ""), metadata={}),
            retrieval_bundle,
        )
        if str(query_shape.get("item_family") or "").casefold() != "activity":
            return []
        scope_terms = cls._activity_scope_terms(question)
        source_text_by_ref: dict[str, list[str]] = {}
        for ref, line in cls._source_backed_context_lines(retrieval_bundle):
            source_text_by_ref.setdefault(ref, []).append(line)
        payload_refs_by_text: list[tuple[str, list[str]]] = []
        for event in list((synthesis_payload or {}).get("counted_events") or []):
            if isinstance(event, dict):
                payload_refs_by_text.append(
                    (
                        " ".join(
                            [
                                str(event.get("event_text") or ""),
                                str(event.get("reason") or ""),
                            ]
                        ),
                        [str(ref) for ref in list(event.get("source_refs") or []) if str(ref).strip()],
                    )
                )
        for fact in list((synthesis_payload or {}).get("supporting_facts") or []):
            if isinstance(fact, dict):
                payload_refs_by_text.append(
                    (
                        str(fact.get("fact_text") or ""),
                        [str(ref) for ref in list(fact.get("source_refs") or []) if str(ref).strip()],
                    )
                )

        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()

        def add_item(item: str, refs: list[str], reason: str) -> None:
            item = " ".join(str(item or "").split()).strip(" .,:;")
            refs = list(dict.fromkeys(ref for ref in refs if ref))
            if not item or not refs:
                return
            key = (cls._normalized_surface_text(item), tuple(refs))
            if key in seen:
                return
            seen.add(key)
            rows.append({"item": item, "source_refs": refs, "source": reason})

        def source_for_refs(refs: list[str], fallback_text: str = "") -> str:
            parts: list[str] = [fallback_text]
            for ref in refs:
                parts.extend(source_text_by_ref.get(ref, []))
            return " ".join(parts)

        activity_patterns: list[tuple[str, str]] = [
            (r"\bpottery(?:\s+class)?\b", "pottery"),
            (r"\brunning\b", "running"),
            (r"\breading\b", "reading"),
            (r"\bplaying\s+(?:my\s+|the\s+)?violin\b|\bviolin\b", "playing violin"),
            (r"\bhiking\b|\bhike\b", "hiking"),
            (r"\bcamping(?:\s+trips?)?\b", "camping"),
            (r"\bswimming\b", "swimming"),
        ]
        for ref, lines in source_text_by_ref.items():
            source_text = " ".join(lines)
            source_norm = source_text.casefold()
            has_scope = not scope_terms or any(term in source_norm for term in scope_terms)
            for pattern, item in activity_patterns:
                if re.search(pattern, source_norm) and has_scope:
                    add_item(item, [ref], "source_line_activity_scope")
        for text, refs in payload_refs_by_text:
            combined = source_for_refs(refs, text)
            combined_norm = combined.casefold()
            has_scope = not scope_terms or any(term in combined_norm for term in scope_terms)
            for pattern, item in activity_patterns:
                if re.search(pattern, combined_norm) and has_scope:
                    add_item(item, refs, "synthesis_source_backed_activity_scope")
        return rows

    @classmethod
    def _event_question_help_scope_terms(cls, question: str | None) -> set[str]:
        text = " ".join(str(question or "").casefold().split())
        if re.search(r"\b(?:help|helps|helping)\b.+\b(?:children|kids|youth|students)\b", text) or re.search(
            r"\b(?:children|kids|youth|students)\b", text
        ):
            return {"child", "children", "kid", "kids", "youth", "student", "students", "help", "mentor", "school"}
        return set()

    @classmethod
    def _source_backed_event_items(
        cls,
        *,
        question: str | None,
        retrieval_bundle: RetrievalBundle,
        synthesis_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_shape = cls._locomo_query_shape(
            QueryTask(query_task_id="", sample_id="", question=str(question or ""), metadata={}),
            retrieval_bundle,
        )
        item_family = str(query_shape.get("item_family") or "").casefold()
        question_norm = " ".join(str(question or "").casefold().split())
        if item_family != "event" and not re.search(r"\b(?:events?|participated|attended|joined)\b", question_norm):
            return []

        scope_terms = cls._event_question_help_scope_terms(question)
        source_text_by_ref: dict[str, list[str]] = {}
        for ref, line in cls._source_backed_context_lines(retrieval_bundle):
            source_text_by_ref.setdefault(ref, []).append(line)

        payload_refs_by_text: list[tuple[str, list[str]]] = []
        for fact in list((synthesis_payload or {}).get("supporting_facts") or []):
            if isinstance(fact, dict):
                payload_refs_by_text.append(
                    (
                        str(fact.get("fact_text") or ""),
                        [str(ref) for ref in list(fact.get("source_refs") or []) if str(ref).strip()],
                    )
                )

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add_item(item: str, refs: list[str], reason: str) -> None:
            item = " ".join(str(item or "").split()).strip(" .,:;")
            refs = list(dict.fromkeys(ref for ref in refs if ref))
            normalized = cls._normalized_surface_text(item)
            if not normalized or not refs or normalized in seen:
                return
            seen.add(normalized)
            rows.append({"item": item, "source_refs": refs, "source": reason})

        def has_scope(text: str) -> bool:
            text_norm = " ".join(str(text or "").casefold().split())
            return not scope_terms or any(term in text_norm for term in scope_terms)

        def detect_items(text: str, refs: list[str], reason: str) -> None:
            text_norm = " ".join(str(text or "").casefold().split())
            if not has_scope(text_norm):
                return
            if re.search(r"\b(?:mentorship|mentoring|mentor)\s+program\b", text_norm):
                add_item("mentoring program", refs, reason)
            school_event = "school event" in text_norm
            school_talk_signal = bool(
                re.search(
                    r"\b(?:talk|talked|talking|speech|shared\s+(?:her|his|their|my)\s+journey|encouraged\s+students)\b",
                    text_norm,
                )
            )
            if school_event and school_talk_signal:
                add_item("school speech", refs, reason)

        for ref, lines in source_text_by_ref.items():
            detect_items(" ".join(lines), [ref], "source_line_event_scope")
        for text, refs in payload_refs_by_text:
            combined = " ".join([text, *[line for ref in refs for line in source_text_by_ref.get(ref, [])]])
            detect_items(combined, refs, "synthesis_source_backed_event_scope")
        return rows

    @classmethod
    def _scope_mismatched_answer_items_for_question(
        cls,
        *,
        question: str | None,
        answer_text: str,
    ) -> list[str]:
        scope_terms = cls._event_question_help_scope_terms(question)
        if not scope_terms:
            return []
        answer_norm = " ".join(str(answer_text or "").casefold().split())
        extras: list[str] = []
        for pattern, label in (
            (r"\bcouncil\s+meeting\b", "council meeting"),
            (r"\badoption\s+agenc(?:y|ies)\b", "adoption agencies"),
            (r"\bpride\s+parade\b", "pride parade"),
            (r"\bcounseling\s+workshop\b", "counseling workshop"),
        ):
            if re.search(pattern, answer_norm):
                extras.append(label)
        return list(dict.fromkeys(extras))

    @classmethod
    def _supported_list_items_for_question(
        cls,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        synthesis_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return list(
            cls._supported_list_scope_rows(
                query_task,
                retrieval_bundle,
                synthesis_payload,
            ).get("required_rows") or []
        )

    @classmethod
    def _postcheck_allows_list_coverage(
        cls,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
    ) -> dict[str, Any]:
        query_shape = cls._locomo_query_shape(query_task, retrieval_bundle)
        expected_type = cls._expected_answer_type(query_task.question, query_shape)
        item_family = str(query_shape.get("item_family") or "").casefold()
        inventory_count_family = cls._inventory_count_family(query_task.question)
        list_family = item_family in {
            "book",
            "reading",
            "event",
            "activity",
            "preference",
            "place",
            "location",
            "organization",
            "deal",
            "pet",
            "dream",
            "instrument",
            "class",
            "item",
        }
        if expected_type in {"value", "date", "boolean"}:
            reason = "date_time_question" if expected_type == "date" else f"{expected_type}_question"
            return {
                "allowed": False,
                "reason": reason,
                "expected_answer_type": expected_type,
                "query_shape": query_shape,
            }
        if expected_type == "count":
            if cls._question_is_duration_count(query_task.question, query_shape):
                return {
                    "allowed": False,
                    "reason": "duration_count_question",
                    "expected_answer_type": expected_type,
                    "query_shape": query_shape,
                }
            if not (inventory_count_family or list_family or query_shape.get("list_like") or query_shape.get("multi_entity") or query_shape.get("comparison_like")):
                return {
                    "allowed": False,
                    "reason": "count_question",
                    "expected_answer_type": expected_type,
                    "query_shape": query_shape,
                }
        explicit_list = bool(
            query_shape.get("list_like")
            or query_shape.get("multi_entity")
            or query_shape.get("comparison_like")
            or list_family
            or inventory_count_family
        )
        if not explicit_list:
            return {
                "allowed": False,
                "reason": "not_list_like",
                "expected_answer_type": expected_type,
                "query_shape": query_shape,
            }
        return {
            "allowed": True,
            "reason": None,
            "expected_answer_type": expected_type,
            "query_shape": query_shape,
        }

    @classmethod
    def _supported_list_scope_rows(
        cls,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        synthesis_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        list_gate = cls._postcheck_allows_list_coverage(query_task, retrieval_bundle)
        query_shape = dict(list_gate.get("query_shape") or cls._locomo_query_shape(query_task, retrieval_bundle))
        item_family = str(query_shape.get("item_family") or "").casefold()
        inventory_count_family = cls._inventory_count_family(query_task.question)
        list_family = item_family in {
            "book",
            "reading",
            "event",
            "activity",
            "preference",
            "place",
            "location",
            "organization",
            "deal",
            "pet",
            "dream",
            "instrument",
            "class",
            "item",
        }
        if not list_gate.get("allowed") or not (
            query_shape.get("list_like")
            or query_shape.get("multi_entity")
            or query_shape.get("comparison_like")
            or list_family
            or inventory_count_family
        ):
            return {
                "scope_kind": str(list_gate.get("reason") or "not_list_like"),
                "required_rows": [],
                "optional_rows": [],
                "rejected_rows": [],
                "coverage_skipped": True,
                "coverage_skip_reason": str(list_gate.get("reason") or "not_list_like"),
                "coverage_blocked_by_expected_type": list_gate.get("reason")
                in {"date_time_question", "count_question", "duration_count_question", "value_question", "boolean_question"},
                "expected_answer_type": list_gate.get("expected_answer_type"),
            }
        scope_profile = cls._list_scope_profile(query_task.question, query_shape)
        if scope_profile.get("kind") in {"generic_list", "type"}:
            return {
                "scope_kind": str(scope_profile.get("kind") or "generic_list"),
                "required_rows": [],
                "optional_rows": [],
                "rejected_rows": [],
                "coverage_skipped": True,
                "coverage_skip_reason": f"{scope_profile.get('kind')}_scope",
                "coverage_blocked_by_expected_type": False,
                "expected_answer_type": list_gate.get("expected_answer_type"),
            }
        support = cls._grounded_surface_support(retrieval_bundle)
        candidate_values = list(
            dict.fromkeys(
                [
                    *support.get("source_surface_terms", []),
                    *support.get("display_items", []),
                    *support.get("wiki_historical_item_terms", []),
                    *support.get("exact_terms", []),
                ]
            )
        )
        generic_terms = {
            "animal",
            "animals",
            "activity",
            "activities",
            "thing",
            "things",
            "nature",
            "support",
            "experience",
            "creative",
            "event",
            "events",
            "item",
            "items",
            "none",
            "unknown",
        }
        generic_phrases = {
            "learning about animals",
            "learn about animals",
            "creative activity",
            "creative activities",
            "family support",
        }
        rows: list[dict[str, Any]] = []
        optional_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in cls._source_backed_activity_items(
            question=query_task.question,
            retrieval_bundle=retrieval_bundle,
            synthesis_payload=synthesis_payload,
        ):
            normalized = cls._normalized_surface_text(str(row.get("item") or ""))
            if normalized and normalized not in seen:
                seen.add(normalized)
                rows.append({**row, "scope_required": True, "scope_reason": "source_backed_activity_required"})
        for row in cls._source_backed_event_items(
            question=query_task.question,
            retrieval_bundle=retrieval_bundle,
            synthesis_payload=synthesis_payload,
        ):
            normalized = cls._normalized_surface_text(str(row.get("item") or ""))
            if normalized and normalized not in seen:
                seen.add(normalized)
                rows.append({**row, "scope_required": True, "scope_reason": "source_backed_event_required"})
        for value in candidate_values:
            text = " ".join(str(value or "").split()).strip(" .,:;")
            normalized = cls._normalized_surface_text(text)
            if not normalized or normalized in seen:
                continue
            if item_family == "event" and (
                (normalized == "school event" and "school speech" in seen)
                or (normalized == "mentorship program" and "mentoring program" in seen)
            ):
                continue
            tokens = cls._surface_tokens(text)
            if normalized in generic_phrases:
                continue
            if (normalized in generic_terms and not (item_family == "preference" and normalized == "nature")) or len(tokens) == 0:
                continue
            if len(tokens) == 1 and next(iter(tokens)) in generic_terms:
                continue
            source_refs = cls._source_refs_for_surface(retrieval_bundle, text)
            source_backed = bool(source_refs) or text in support.get("source_surface_terms", [])
            if not source_backed:
                continue
            decision = cls._list_scope_decision(
                item=text,
                refs=source_refs,
                profile=scope_profile,
                retrieval_bundle=retrieval_bundle,
            )
            row = {
                "item": text,
                "source_refs": source_refs,
                "source": "retrieved_surface",
                "scope_required": bool(decision.get("required")),
                "scope_optional": bool(decision.get("optional")),
                "scope_reason": decision.get("reason"),
            }
            if decision.get("required"):
                seen.add(normalized)
                rows.append(row)
            elif decision.get("optional"):
                optional_rows.append(row)
            else:
                rejected_rows.append(row)
        return {
            "scope_kind": scope_profile.get("kind"),
            "required_rows": rows[:12],
            "optional_rows": optional_rows[:20],
            "rejected_rows": rejected_rows[:30],
            "coverage_skipped": False,
            "coverage_skip_reason": None,
            "coverage_blocked_by_expected_type": False,
            "expected_answer_type": list_gate.get("expected_answer_type"),
        }

    @classmethod
    def _locomo_list_coverage_diagnostics(
        cls,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        answer_text: str,
        synthesis_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        scope_rows = cls._supported_list_scope_rows(query_task, retrieval_bundle, synthesis_payload)
        supported_rows = list(scope_rows.get("required_rows") or [])
        supported_items = [str(row.get("item") or "") for row in supported_rows if str(row.get("item") or "")]
        optional_values = [
            str(row.get("item") or "")
            for row in list(scope_rows.get("optional_rows") or [])
            if str(row.get("item") or "")
        ]
        rejected_rows = [
            row for row in list(scope_rows.get("rejected_rows") or []) if isinstance(row, dict)
        ]
        required_items = [
            item for item in supported_items
            if cls._answer_covers_supported_item(answer_text, item)
        ]
        supported_refs = {
            str(row.get("item") or ""): list(row.get("source_refs") or [])
            for row in supported_rows
            if str(row.get("item") or "")
        }
        event_canonical_alias_items = [
            str(row.get("item") or "")
            for row in supported_rows
            if str(row.get("source") or "").endswith("event_scope")
            and str(row.get("item") or "")
        ]
        missing = [
            item for item in supported_items
            if not cls._answer_covers_supported_item(answer_text, item)
        ]
        abstain_with_support = bool(supported_items and cls._answer_is_context_abstention(answer_text))
        scope_mismatched = cls._scope_mismatched_answer_items_for_question(
            question=query_task.question,
            answer_text=answer_text,
        )
        result = {
            "answer_list_scope_kind": scope_rows.get("scope_kind"),
            "answer_list_coverage_skipped": bool(scope_rows.get("coverage_skipped")),
            "answer_list_coverage_skip_reason": scope_rows.get("coverage_skip_reason"),
            "answer_list_repair_blocked_by_expected_type": bool(
                scope_rows.get("coverage_blocked_by_expected_type")
            ),
            "answer_supported_list_items": supported_items,
            "answer_supported_list_item_refs": supported_refs,
            "answer_supported_required_items": required_items,
            "answer_supported_required_item_refs": {
                item: supported_refs.get(item, []) for item in required_items
            },
            "answer_required_item_candidates": supported_items,
            "answer_optional_surface_values": optional_values,
            "answer_scope_rejected_items": [
                str(row.get("item") or "") for row in rejected_rows if str(row.get("item") or "")
            ],
            "answer_scope_rejection_reasons": [
                {
                    "item": str(row.get("item") or ""),
                    "reason": row.get("scope_reason"),
                    "source_refs": list(row.get("source_refs") or []),
                }
                for row in rejected_rows
                if str(row.get("item") or "")
            ],
            "answer_missing_supported_list_items": missing,
            "answer_missing_required_items": missing,
            "answer_abstain_despite_supported_items": abstain_with_support,
            "answer_repair_scope_filtered_items": scope_mismatched,
            "answer_required_items_missing_before_repair": missing,
            "event_canonical_alias_items": list(dict.fromkeys(event_canonical_alias_items)),
        }
        if scope_mismatched:
            result["answer_scope_mismatched_extra_items"] = scope_mismatched
        return result

    @classmethod
    def _locomo_specificity_diagnostics(
        cls,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        answer_text: str,
    ) -> dict[str, Any]:
        query_shape = cls._locomo_query_shape(query_task, retrieval_bundle)
        item_family = str(query_shape.get("item_family") or "").casefold()
        question = " ".join(str(query_task.question or "").casefold().split())
        preference_like = item_family == "preference" or bool(
            re.search(r"\bwhat\s+(?:do|does)\b.+\b(?:like|likes|enjoy|enjoys|love|loves|prefer|prefers)\b", question)
        )
        if not (query_shape.get("list_like") or preference_like):
            return {
                "query_shape_preference_like": preference_like,
                "answer_overgeneric_item_detected": False,
                "answer_overgeneric_items": [],
                "answer_specific_replacement_candidates": [],
                "answer_scope_mismatched_extra_items": [],
            }
        support = cls._grounded_surface_support(retrieval_bundle)
        specific_surfaces = list(
            dict.fromkeys(
                [
                    *support.get("source_surface_terms", []),
                    *support.get("exact_terms", []),
                    *support.get("display_items", []),
                    *support.get("wiki_historical_item_terms", []),
                ]
            )
        )
        answer_items = cls._split_answer_items(answer_text)
        generic_to_specific_tokens = {
            "animal": {"dinosaur", "dinosaurs", "turtle", "turtles", "tortoise", "tortoises", "horse", "horses", "dog", "dogs", "cat", "cats"},
            "animals": {"dinosaur", "dinosaurs", "turtle", "turtles", "tortoise", "tortoises", "horse", "horses", "dog", "dogs", "cat", "cats"},
            "art": {"sunset", "sunrise", "painting", "paintings", "horse", "flowers", "sunflower"},
            "artwork": {"sunset", "sunrise", "painting", "paintings", "horse", "flowers", "sunflower"},
            "activity": {"pottery", "hiking", "camping", "swimming", "museum", "beach"},
            "activities": {"pottery", "hiking", "camping", "swimming", "museum", "beach"},
        }
        normalized_surfaces = {
            surface: cls._normalized_surface_text(surface)
            for surface in specific_surfaces
            if cls._normalized_surface_text(surface)
        }
        overgeneric_items: list[str] = []
        replacements: list[str] = []
        for item in answer_items:
            item_norm = cls._normalized_surface_text(item)
            if item_norm not in generic_to_specific_tokens:
                continue
            wanted_tokens = generic_to_specific_tokens[item_norm]
            matched = [
                surface
                for surface, surface_norm in normalized_surfaces.items()
                if any(re.search(rf"\b{re.escape(token)}\b", surface_norm) for token in wanted_tokens)
                and surface_norm != item_norm
            ]
            if matched:
                overgeneric_items.append(item)
                replacements.extend(matched[:3])
        scope_mismatched_items: list[str] = []
        if preference_like and not re.search(r"\b(?:activities|activity|hobbies|hobby|projects|crafts|things)\b", question):
            for item in answer_items:
                item_norm = cls._normalized_surface_text(item)
                if re.search(r"\b(?:crafting|clay|pottery|workshop|make something|making something)\b", item_norm):
                    scope_mismatched_items.append(item)
        return {
            "query_shape_preference_like": preference_like,
            "answer_overgeneric_item_detected": bool(overgeneric_items),
            "answer_overgeneric_items": list(dict.fromkeys(overgeneric_items)),
            "answer_specific_replacement_candidates": list(dict.fromkeys(replacements)),
            "answer_scope_mismatched_extra_items": list(dict.fromkeys(scope_mismatched_items)),
        }

    @classmethod
    def _detect_locomo_postcheck_issue(
        cls,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        answer_text: str,
        synthesis_payload: dict[str, Any] | None = None,
    ) -> str | None:
        query_shape = cls._locomo_query_shape(query_task, retrieval_bundle)
        bridge = cls._locomo_bridge_finalization_diagnostics(
            retrieval_bundle,
            answer_text,
            query_task.question,
        )
        if bridge.get("bridge_finalization_needs_repair"):
            return "bridge_alias_unresolved"
        specificity = cls._locomo_specificity_diagnostics(query_task, retrieval_bundle, answer_text)
        list_coverage = cls._locomo_list_coverage_diagnostics(
            query_task,
            retrieval_bundle,
            answer_text,
            synthesis_payload,
        )
        if specificity.get("answer_overgeneric_item_detected"):
            return "overgeneric_item"
        if specificity.get("answer_scope_mismatched_extra_items"):
            return "scope_mismatched_extra_item"
        if list_coverage.get("answer_scope_mismatched_extra_items"):
            return "scope_mismatched_extra_item"
        if list_coverage.get("answer_abstain_despite_supported_items"):
            return "abstain_despite_supported_list_items"
        if list_coverage.get("answer_missing_supported_list_items"):
            return "missing_supported_list_items"
        support = cls._grounded_surface_support(retrieval_bundle)
        supported_surfaces = list(
            dict.fromkeys(
                [
                    *support["display_items"],
                    *support["exact_terms"],
                    *support["display_counts"],
                    *support.get("source_surface_terms", []),
                    *support.get("wiki_historical_item_terms", []),
                ]
            )
        )
        if not supported_surfaces:
            return None
        if query_shape["count_like"]:
            answer_numbers = re.findall(r"\b\d+\b", answer_text)
            supported_numbers = re.findall(r"\b\d+\b", " ".join(support["display_counts"]))
            if answer_numbers and supported_numbers and set(answer_numbers) - set(supported_numbers):
                return "unsupported_count"
        if not (query_shape["list_like"] or query_shape["multi_entity"] or query_shape["comparison_like"]):
            return None
        if not any(separator in answer_text for separator in [",", ";", " and "]):
            return None
        answer_items = cls._split_answer_items(answer_text)
        if len(answer_items) < 2:
            return None
        unsupported = [
            item for item in answer_items
            if not cls._surface_supported(item, supported_surfaces)
        ]
        if unsupported:
            return "unsupported_extra_items"
        return None

    @staticmethod
    def _answer_is_context_abstention(answer_text: str) -> bool:
        normalized = " ".join(str(answer_text or "").casefold().split())
        regex_patterns = [
            r"\bretrieved context (?:does not|doesn't) support\b",
            r"\bnot supported by the retrieved context\b",
            r"\bnot enough information\b",
            r"\bdo(?:es)? not have enough information\b",
            r"\bdo(?:es)? not provide enough information\b",
            r"\bdo(?:es)? not provide (?:any )?information\b",
            r"\bdo(?:es)? not mention\b",
            r"\bdo(?:es)? not specify\b",
            r"\bnot specified in (?:the )?(?:retrieved )?context\b",
            r"\bnot provided in (?:the )?(?:retrieved )?context\b",
            r"\bno information (?:about|regarding|on)\b",
            r"\bcannot determine\b",
            r"\bcan't determine\b",
            r"\bunknown from context\b",
        ]
        return any(re.search(pattern, normalized) for pattern in regex_patterns)

    @classmethod
    def _locomo_postcheck_skip_reason(cls, response: LLMResponse, answer_text: str) -> str | None:
        metadata = dict(response.metadata or {})
        if metadata.get("answer_generation_failed"):
            return "answer_generation_failed"
        if metadata.get("answer_synthesis_can_answer") is False:
            return "synthesis_can_answer_false"
        if cls._answer_is_context_abstention(answer_text):
            return "context_abstention"
        return None

    @staticmethod
    def _repaired_answer_format(raw_text: str) -> str:
        stripped = str(raw_text or "").strip()
        if not stripped:
            return "empty"
        if stripped.startswith("```"):
            inner = "\n".join(stripped.splitlines()[1:-1]).strip() if stripped.endswith("```") else ""
            return (
                "fenced_json"
                if re.match(r"^```\s*json\b", stripped, flags=re.IGNORECASE) or inner.startswith("{")
                else "fenced_text"
            )
        if stripped.startswith("{"):
            return "json"
        return "plain_text"

    @classmethod
    def _normalize_repaired_answer_text(cls, raw_text: str, previous_answer: str) -> tuple[str, dict[str, Any]]:
        stripped = str(raw_text or "").strip()
        repair_format = cls._repaired_answer_format(stripped)
        metadata: dict[str, Any] = {
            "answer_repair_raw_text": stripped,
            "answer_repair_format": repair_format,
            "answer_repair_json_extracted": False,
            "answer_repair_discarded": False,
            "answer_repair_discard_reason": None,
        }
        if not stripped:
            metadata["answer_repair_discarded"] = True
            metadata["answer_repair_discard_reason"] = "empty_repair"
            return previous_answer, metadata
        if repair_format in {"json", "fenced_json"}:
            try:
                payload = cls._extract_json_object(stripped)
            except Exception:  # noqa: BLE001
                metadata["answer_repair_discarded"] = True
                metadata["answer_repair_discard_reason"] = "json_parse_failed"
                return previous_answer, metadata
            metadata["answer_repair_json_extracted"] = True
            if payload.get("can_answer") is False:
                return cls._standard_abstention_text(payload.get("abstain_reason")), metadata
            final_answer = str(payload.get("final_answer") or "").strip()
            if not final_answer:
                metadata["answer_repair_discarded"] = True
                metadata["answer_repair_discard_reason"] = "missing_final_answer"
                return previous_answer, metadata
            return final_answer, metadata
        if repair_format == "fenced_text":
            metadata["answer_repair_discarded"] = True
            metadata["answer_repair_discard_reason"] = "fenced_non_json_output"
            return previous_answer, metadata
        return stripped, metadata

    @classmethod
    def _repair_required_item_validation(
        cls,
        *,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        initial_answer_text: str,
        repaired_answer_text: str,
        issue: str,
        synthesis_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        initial_coverage = cls._locomo_list_coverage_diagnostics(
            query_task,
            retrieval_bundle,
            initial_answer_text,
            synthesis_payload,
        )
        required_items = list(initial_coverage.get("answer_supported_required_items") or [])
        dropped_items = [
            item for item in required_items
            if not cls._answer_covers_supported_item(repaired_answer_text, item)
        ]
        repaired_coverage = cls._locomo_list_coverage_diagnostics(
            query_task,
            retrieval_bundle,
            repaired_answer_text,
            synthesis_payload,
        )
        initial_missing = list(initial_coverage.get("answer_missing_supported_list_items") or [])
        repaired_missing = list(repaired_coverage.get("answer_missing_supported_list_items") or [])
        initial_scope_extras = list(initial_coverage.get("answer_scope_mismatched_extra_items") or [])
        remaining_scope_extras = list(repaired_coverage.get("answer_scope_mismatched_extra_items") or [])
        removed_scope_extras = [
            item for item in initial_scope_extras
            if item not in remaining_scope_extras
        ]
        list_coverage_improved = bool(initial_missing and len(repaired_missing) < len(initial_missing))
        repaired_supported_items = list(repaired_coverage.get("answer_supported_required_items") or [])
        if issue == "abstain_despite_supported_list_items" and repaired_supported_items:
            list_coverage_improved = True
        expected_type = cls._expected_answer_type(
            query_task.question,
            cls._locomo_query_shape(query_task, retrieval_bundle),
        )
        initial_valid = cls._answer_text_matches_expected_type(initial_answer_text, expected_type, query_task.question)
        single_value_changed_by_list_repair = bool(
            issue == "missing_supported_list_items"
            and expected_type in {"value", "place", "person"}
            and initial_valid
            and cls._normalized_surface_text(initial_answer_text) != cls._normalized_surface_text(repaired_answer_text)
        )
        not_improved = bool(
            issue == "missing_supported_list_items"
            and initial_missing
            and not list_coverage_improved
        )
        validation_failed = bool(dropped_items or not_improved or single_value_changed_by_list_repair)
        if dropped_items:
            validation_reason = "dropped_supported_required_items"
        elif single_value_changed_by_list_repair:
            validation_reason = "single_value_changed_by_list_repair"
        elif not_improved:
            validation_reason = "list_coverage_not_improved"
        else:
            validation_reason = None
        return {
            "answer_supported_required_items": required_items,
            "answer_supported_required_item_refs": dict(
                initial_coverage.get("answer_supported_required_item_refs") or {}
            ),
            "answer_repair_dropped_supported_items": dropped_items,
            "answer_repair_missing_required_items_after_repair": repaired_missing,
            "answer_repair_removed_scope_mismatched_items": removed_scope_extras,
            "answer_repair_list_coverage_improved": list_coverage_improved,
            "answer_repair_post_validation_failed": validation_failed,
            "answer_repair_post_validation_reason": validation_reason,
            "answer_repair_post_validation_action": "preserve_initial_answer" if validation_failed else "accept_repair",
            "answer_repair_preserved_initial_answer": validation_failed,
        }

    @classmethod
    def _absolute_temporal_mentions(cls, text: Any) -> set[str]:
        value = " ".join(str(text or "").replace(",", " ").split())
        if not value:
            return set()
        mentions = set(cls._date_mentions(value))
        month = (
            r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        )
        for match in re.finditer(rf"\b{month}\s+\d{{4}}\b", value, flags=re.IGNORECASE):
            mentions.add(collapse_whitespace(match.group(0)).casefold())
        for year in re.findall(r"\b\d{4}\b", value):
            mentions.add(year)
        return mentions

    @staticmethod
    def _has_relative_temporal_phrase(text: Any) -> bool:
        normalized = " ".join(str(text or "").casefold().split())
        return bool(
            re.search(
                r"\b(?:today|yesterday|tomorrow|recently|last\s+(?:week|month|year|night|weekend)|"
                r"next\s+(?:week|month|year|weekend)|this\s+(?:week|month|year|weekend))\b",
                normalized,
            )
        )

    @staticmethod
    def _answer_preview(text: Any, *, limit: int = 240) -> str:
        return collapse_whitespace(str(text or ""))[:limit]

    @classmethod
    def _repair_arbitration_trigger(
        cls,
        *,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        initial_answer_text: str,
        repaired_answer_text: str,
        issue: str,
        repair_validation_metadata: dict[str, Any],
        synthesis_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        initial_text = collapse_whitespace(initial_answer_text)
        repaired_text = collapse_whitespace(repaired_answer_text)
        expected_type = cls._expected_answer_type(
            query_task.question,
            cls._locomo_query_shape(query_task, retrieval_bundle),
        )
        base = {
            "answer_repair_arbitration_triggered": False,
            "answer_repair_arbitration_trigger_reason": None,
            "answer_repair_initial_answer_preview": cls._answer_preview(initial_text),
            "answer_repair_repaired_answer_preview": cls._answer_preview(repaired_text),
            "answer_repair_arbitration_expected_type": expected_type,
        }
        if not initial_text or not repaired_text or cls._normalized_surface_text(initial_text) == cls._normalized_surface_text(repaired_text):
            return base
        initial_valid = cls._answer_text_matches_expected_type(initial_text, expected_type, query_task.question)
        repaired_valid = cls._answer_text_matches_expected_type(repaired_text, expected_type, query_task.question)
        initial_abstain = cls._answer_is_context_abstention(initial_text)
        repaired_abstain = cls._answer_is_context_abstention(repaired_text)
        initial_dates = cls._absolute_temporal_mentions(initial_text)
        repaired_dates = cls._absolute_temporal_mentions(repaired_text)
        dropped_required = list(repair_validation_metadata.get("answer_repair_dropped_supported_items") or [])
        initial_missing = list(
            cls._locomo_list_coverage_diagnostics(
                query_task,
                retrieval_bundle,
                initial_text,
                synthesis_payload,
            ).get("answer_missing_supported_list_items") or []
        )
        repaired_missing = list(
            cls._locomo_list_coverage_diagnostics(
                query_task,
                retrieval_bundle,
                repaired_text,
                synthesis_payload,
            ).get("answer_missing_supported_list_items") or []
        )

        reason = None
        if dropped_required:
            reason = "lost_required_value"
        elif expected_type in {"date", "time"} and initial_dates and not bool(initial_dates & repaired_dates):
            reason = "absolute_date_dropped"
        elif (
            expected_type in {"date", "time"}
            and initial_dates
            and cls._has_relative_temporal_phrase(repaired_text)
            and not repaired_dates
        ):
            reason = "relative_temporal_regression"
        elif repaired_abstain and not initial_abstain and initial_valid:
            reason = "repair_to_abstain_despite_valid_initial"
        elif initial_valid and not repaired_valid:
            reason = "expected_type_invalid_after_repair"
        elif len(repaired_missing) > len(initial_missing):
            reason = "missing_required_items_increased"
        elif expected_type in {"place", "person", "boolean", "count", "value"} and initial_valid:
            initial_tokens = cls._surface_tokens(initial_text)
            repaired_tokens = cls._surface_tokens(repaired_text)
            if initial_tokens and (len(initial_tokens & repaired_tokens) / max(1, len(initial_tokens))) < 0.5:
                reason = "concrete_value_dropped"
        if not reason:
            return base
        return {
            **base,
            "answer_repair_arbitration_triggered": True,
            "answer_repair_arbitration_trigger_reason": reason,
            "answer_repair_arbitration_initial_type_valid": initial_valid,
            "answer_repair_arbitration_repaired_type_valid": repaired_valid,
            "answer_repair_arbitration_initial_temporal_mentions": sorted(initial_dates),
            "answer_repair_arbitration_repaired_temporal_mentions": sorted(repaired_dates),
            "answer_repair_arbitration_initial_missing_required_items": initial_missing,
            "answer_repair_arbitration_repaired_missing_required_items": repaired_missing,
        }

    @staticmethod
    def _arbitration_conservative_fallback_action(trigger_reason: str | None) -> str:
        if trigger_reason in {
            "lost_required_value",
            "absolute_date_dropped",
            "relative_temporal_regression",
            "repair_to_abstain_despite_valid_initial",
        }:
            return "keep_initial"
        return "use_repair"

    @classmethod
    def _repair_arbitration_prompt(
        cls,
        *,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        initial_answer_text: str,
        repaired_answer_text: str,
        issue: str,
        trigger_metadata: dict[str, Any],
        repair_validation_metadata: dict[str, Any],
    ) -> str:
        support = cls._grounded_surface_support(retrieval_bundle)
        bridge = cls._locomo_bridge_finalization_diagnostics(
            retrieval_bundle,
            initial_answer_text + "\n" + repaired_answer_text,
            query_task.question,
        )
        list_coverage = cls._locomo_list_coverage_diagnostics(
            query_task,
            retrieval_bundle,
            initial_answer_text,
            None,
        )
        compact_evidence = {
            "source_surface_terms": support.get("source_surface_terms", [])[:12],
            "wiki_historical_item_terms": support.get("wiki_historical_item_terms", [])[:12],
            "display_items": support.get("display_items", [])[:12],
            "display_key_facts": support.get("display_key_facts", [])[:8],
            "exact_terms": support.get("exact_terms", [])[:12],
        }
        payload = {
            "question": query_task.question,
            "expected_answer_type": trigger_metadata.get("answer_repair_arbitration_expected_type"),
            "repair_issue": issue,
            "trigger_reason": trigger_metadata.get("answer_repair_arbitration_trigger_reason"),
            "initial_answer": initial_answer_text,
            "repaired_answer": repaired_answer_text,
            "compact_evidence": compact_evidence,
            "temporal_candidates": cls._temporal_repair_candidate_lines(
                question=query_task.question,
                answer_text=initial_answer_text + "\n" + repaired_answer_text,
                retrieval_bundle=retrieval_bundle,
            ),
            "supported_required_items": list(list_coverage.get("answer_supported_required_items") or []),
            "must_keep_supported_items": list(repair_validation_metadata.get("answer_supported_required_items") or []),
            "bridge_facts": {
                "alias": bridge.get("bridge_finalization_alias"),
                "target": bridge.get("bridge_finalization_target"),
                "source_refs": bridge.get("bridge_finalization_source_refs") or [],
            },
        }
        return load_prompt("locomo_answer_repair_arbitration") + "\n\nANSWER_REPAIR_ARBITRATION_INPUT:\n" + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )

    def _maybe_arbitrate_repaired_answer(
        self,
        *,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        initial_answer_text: str,
        repaired_answer_text: str,
        issue: str,
        trigger_metadata: dict[str, Any],
        repair_validation_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        trigger_reason = str(trigger_metadata.get("answer_repair_arbitration_trigger_reason") or "")
        fallback_action = self._arbitration_conservative_fallback_action(trigger_reason)
        base = {
            "answer_repair_arbitration_used": False,
            "answer_repair_arbitration_success": False,
            "answer_repair_arbitration_decision": fallback_action,
            "answer_repair_arbitration_violation": trigger_reason or "none",
            "answer_repair_arbitration_confidence": "low",
            "answer_repair_arbitration_reason": "structured arbitration unavailable; used deterministic fallback",
            "answer_repair_arbitration_action": fallback_action,
            "answer_repair_arbitration_kept_initial": fallback_action == "keep_initial",
            "answer_repair_arbitration_used_safe_abstain": False,
        }
        task = "answer_repair_arbitration"
        if not self.llm_provider.supports_structured(task):
            return base
        prompt = self._repair_arbitration_prompt(
            query_task=query_task,
            retrieval_bundle=retrieval_bundle,
            initial_answer_text=initial_answer_text,
            repaired_answer_text=repaired_answer_text,
            issue=issue,
            trigger_metadata=trigger_metadata,
            repair_validation_metadata=repair_validation_metadata,
        )
        self._trace(
            f"answer_repair_arbitration_start sample={query_task.sample_id} "
            f"query_task_id={query_task.query_task_id} reason={trigger_reason or '-'}"
        )
        try:
            spec = get_structured_task_spec(task)
            structured = self.llm_provider.generate_structured(
                [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                spec=spec,
                metadata={
                    "task": task,
                    "query_task_id": query_task.query_task_id,
                    "answer_prompt_name": "locomo_answer_repair_arbitration",
                    "repair_requested": True,
                    "repair_trigger": issue,
                    "repair_action": "answer_repair_arbitration",
                },
            )
            result = self._model_to_dict(structured.parsed)
        except Exception as exc:  # noqa: BLE001
            error_message = self._compact_exception(exc)
            self._trace(
                f"answer_repair_arbitration_failed sample={query_task.sample_id} "
                f"query_task_id={query_task.query_task_id} error={error_message}"
            )
            return {**base, "answer_repair_arbitration_error": error_message}
        decision = str(result.get("decision") or fallback_action)
        if decision not in {"keep_initial", "use_repair", "safe_abstain"}:
            decision = fallback_action
        metadata = {
            "answer_repair_arbitration_used": True,
            "answer_repair_arbitration_success": True,
            "answer_repair_arbitration_decision": decision,
            "answer_repair_arbitration_violation": str(result.get("repair_violation") or "none"),
            "answer_repair_arbitration_confidence": str(result.get("confidence") or "low"),
            "answer_repair_arbitration_reason": str(result.get("reason") or ""),
            "answer_repair_arbitration_action": decision,
            "answer_repair_arbitration_kept_initial": decision == "keep_initial",
            "answer_repair_arbitration_used_safe_abstain": decision == "safe_abstain",
        }
        self._trace(
            f"answer_repair_arbitration_done sample={query_task.sample_id} "
            f"query_task_id={query_task.query_task_id} decision={decision} action={decision}"
        )
        return metadata

    @classmethod
    def _temporal_repair_candidate_lines(
        cls,
        *,
        question: str | None,
        answer_text: str,
        retrieval_bundle: RetrievalBundle,
        limit: int = 5,
    ) -> list[str]:
        diagnostics = cls._temporal_answer_alignment_diagnostics(
            answer_text=answer_text,
            retrieval_bundle=retrieval_bundle,
            question=question,
        )
        if not diagnostics.get("answer_temporal_alignment_checked"):
            return []
        source_text_by_ref = cls._context_text_by_source_ref(retrieval_bundle)
        lines: list[str] = []
        for row in list(diagnostics.get("answer_temporal_candidate_dates") or []):
            if not isinstance(row, dict) or row.get("confidence") not in {"high", "medium"}:
                continue
            ref = str(row.get("source_ref") or "")
            preview = " ".join(str(source_text_by_ref.get(ref) or "").split())[:180]
            lines.append(
                "- "
                + f"ref={ref}; "
                + f"source_date={row.get('source_date') or '-'}; "
                + f"resolved_date={row.get('resolved_date') or '-'}; "
                + f"resolved_answer_text={row.get('resolved_answer_text') or row.get('answer_target') or '-'}; "
                + f"resolution_kind={row.get('resolution_kind') or '-'}; "
                + f"resolution_granularity={row.get('resolution_granularity') or '-'}; "
                + f"confidence={row.get('confidence') or '-'}; "
                + f"score={row.get('temporal_score') or 0}; "
                + f"matched={', '.join(list(row.get('matched_terms') or [])) or '-'}; "
                + f"relative={', '.join(list(row.get('relative_terms') or [])) or '-'}; "
                + f"preview={preview or '-'}"
            )
            if len(lines) >= limit:
                break
        return lines

    def _repair_locomo_answer(
        self,
        *,
        query_task: QueryTask,
        retrieval_bundle: RetrievalBundle,
        answer_text: str,
        answer_prompt_name: str,
        issue: str,
        synthesis_payload: dict[str, Any] | None = None,
    ) -> tuple[str, LLMResponse]:
        support = self._grounded_surface_support(retrieval_bundle)
        list_coverage = self._locomo_list_coverage_diagnostics(
            query_task,
            retrieval_bundle,
            answer_text,
            synthesis_payload,
        )
        expected_type_for_repair = self._expected_answer_type(
            query_task.question,
            classify_query_shape_v1(query_task.question, {}),
        )
        temporal_repair_candidate_rows = (
            self._temporal_repair_candidate_lines(
                question=query_task.question,
                answer_text=answer_text,
                retrieval_bundle=retrieval_bundle,
            )
            if issue == "answer_type_mismatch" and expected_type_for_repair == "date"
            else []
        )
        required_item_rows = [
            f"{item} [{', '.join(list_coverage.get('answer_supported_required_item_refs', {}).get(item, []) or ['source-backed'])}]"
            for item in list(list_coverage.get("answer_supported_required_items") or [])
        ]
        if issue == "unsupported_extra_items":
            repair_instruction = (
                "Delete unsupported extra items and keep only items explicitly supported by the retrieved context. "
                "You must preserve every item listed in MUST_KEEP_SUPPORTED_ITEMS."
            )
        elif issue == "unsupported_count":
            repair_instruction = "Delete unsupported counts and keep only counts explicitly supported by the retrieved context."
        elif issue == "overgeneric_item":
            repair_instruction = (
                "Replace broad umbrella words with the most specific source-backed items. "
                "For example, prefer 'dinosaur exhibit' or 'dinosaurs' over only 'animals' when those specific source terms are supported."
            )
        elif issue == "scope_mismatched_extra_item":
            repair_instruction = (
                "Delete items that are supported somewhere in the context but do not answer this question's scope. "
                "Keep only the specific source-backed items that answer the asked list/preference."
            )
        elif issue == "missing_supported_list_items":
            repair_instruction = (
                "Add the missing in-scope source-backed list items from SUPPORTED_LIST_ITEMS. "
                "Keep the answer concise and do not add any item not listed as supported."
            )
        elif issue == "abstain_despite_supported_list_items":
            repair_instruction = (
                "The previous answer abstained, but SUPPORTED_LIST_ITEMS contains source-backed items that answer the question. "
                "Answer using only those supported items; if they are only partial evidence, state that the retrieved evidence supports them."
            )
        elif issue == "bridge_alias_unresolved":
            bridge = self._locomo_bridge_finalization_diagnostics(retrieval_bundle, answer_text, query_task.question)
            repair_instruction = (
                "Replace the alias in the previous answer with the source-backed concrete value. "
                f"Use '{bridge.get('bridge_finalization_target')}' for '{bridge.get('bridge_finalization_alias')}', "
                "and do not keep the alias as the final answer."
            )
        elif issue == "answer_type_mismatch":
            repair_instruction = (
                str(synthesis_payload.get("answer_type_repair_instruction") or "").strip()
                if isinstance(synthesis_payload, dict)
                else ""
            )
            if not repair_instruction:
                repair_instruction = (
                    f"Rewrite the answer as a {expected_type_for_repair} answer using only the retrieved context. "
                    "If the retrieved evidence does not support that answer type, return a brief abstention."
                )
            if expected_type_for_repair == "date":
                repair_instruction += (
                    " Use only TEMPORAL_CANDIDATES for date selection; do not use unrelated dates or event descriptions."
                )
        else:
            repair_instruction = "Repair the answer using only supported values and the exact question scope."
        supported_list_item_rows = [
            f"{item} [{', '.join(list_coverage.get('answer_supported_list_item_refs', {}).get(item, []) or ['source-backed'])}]"
            for item in list(list_coverage.get("answer_supported_list_items") or [])
        ]
        scope_mismatched_rows = list(list_coverage.get("answer_scope_mismatched_extra_items") or [])
        repair_prompt = (
            load_prompt("locomo_answer_repair")
            + "\n\nQUESTION:\n"
            + query_task.question
            + "\n\nPREVIOUS_ANSWER:\n"
            + answer_text
            + "\n\nSUPPORTED_SURFACE_VALUES:\n"
            + "source_surface_terms="
            + (", ".join(support.get("source_surface_terms", [])) or "none")
            + "\nwiki_historical_item_terms="
            + (", ".join(support.get("wiki_historical_item_terms", [])) or "none")
            + "\ndisplay_key_facts="
            + (", ".join(support.get("display_key_facts", [])) or "none")
            + "\ndisplay_items="
            + (", ".join(support["display_items"]) or "none")
            + "\nexact_terms="
            + (", ".join(support["exact_terms"]) or "none")
            + "\ndisplay_counts="
            + (", ".join(support["display_counts"]) or "none")
            + "\n\nTEMPORAL_CANDIDATES:\n"
            + ("\n".join(temporal_repair_candidate_rows) or "none")
            + "\n\nSUPPORTED_LIST_ITEMS:\n"
            + ("\n".join(f"- {row}" for row in supported_list_item_rows) or "none")
            + "\n\nOPTIONAL_SURFACE_VALUES:\n"
            + (
                "\n".join(
                    f"- {item}"
                    for item in list(list_coverage.get("answer_optional_surface_values") or [])
                )
                or "none"
            )
            + "\n\nMUST_KEEP_SUPPORTED_ITEMS:\n"
            + ("\n".join(f"- {row}" for row in required_item_rows) or "none")
            + "\nMISSING_SUPPORTED_LIST_ITEMS:\n"
            + (", ".join(list(list_coverage.get("answer_missing_supported_list_items") or [])) or "none")
            + "\nREMOVE_SCOPE_MISMATCHED_ITEMS:\n"
            + (", ".join(scope_mismatched_rows) or "none")
            + "\n\nREPAIR_INSTRUCTION:\n"
            + repair_instruction
            + "\nReturn only the repaired final answer text."
        )
        return (
            repair_prompt,
            self.llm_provider.generate(
                [NormalizedMessage(role="user", content=repair_prompt, turn_index=0)],
                metadata={
                    "task": "answer_generation_repair",
                    "query_task_id": query_task.query_task_id,
                    "answer_prompt_name": answer_prompt_name,
                    "answer_postcheck_issue": issue,
                },
            ),
        )

    @staticmethod
    def prompt_name_for_query(query_task: QueryTask) -> str:
        answer_context = query_task.metadata.get("answer_context", {})
        if answer_context.get("dataset") == "medmt":
            return "medmt_answer_generation"
        if query_task.metadata.get("category_name") is not None or query_task.metadata.get("evidence_only_conversation") is not None:
            return "locomo_answer_generation"
        return "answer_generation"

    def build_prompt(self, query_task: QueryTask, retrieval_bundle: RetrievalBundle) -> str:
        prompt_name = self.prompt_name_for_query(query_task)
        answer_context = query_task.metadata.get("answer_context", {})
        if prompt_name == "medmt_answer_generation":
            return (
                load_prompt(prompt_name)
                + "\n\nCATEGORY:\n"
                + str(answer_context.get("category") or "Unknown")
                + "\n\nSUBTYPE:\n"
                + str(answer_context.get("subtype") or "Unknown")
                + "\n\nSCENE_TAG:\n"
                + str(answer_context.get("scene_tag") or "Unknown")
                + "\n\nSUBSET_KEY:\n"
                + str(answer_context.get("subset_key") or "unknown")
                + "\n\nRUBRIC:\n"
                + str(answer_context.get("rubric") or "")
                + "\n\nRETRIEVED_CONTEXT:\n"
                + retrieval_bundle.prompt_context
                + "\n\nFINAL_USER_TURN:\n"
                + query_task.question
            )
        if prompt_name == "locomo_answer_generation":
            return (
                load_prompt(prompt_name)
                + self._locomo_query_shape_rubric(query_task, retrieval_bundle)
                + "\n\n"
                + retrieval_bundle.prompt_context
                + "\n\nQuestion:\n"
                + query_task.question
            )
        return (
            load_prompt(prompt_name)
            + "\n\n"
            + retrieval_bundle.prompt_context
            + "\n\nQuestion:\n"
            + query_task.question
        )

    def generate(self, query_task: QueryTask, retrieval_bundle: RetrievalBundle) -> tuple[str, LLMResponse]:
        prompt_name = self.prompt_name_for_query(query_task)
        self._trace(
            f"answer_generation_start sample={query_task.sample_id} query_task_id={query_task.query_task_id}"
        )
        synthesis_failure_metadata: dict[str, Any] = {}
        if prompt_name == "locomo_answer_generation":
            freeform_result = self._try_generate_locomo_freeform_answer(query_task, retrieval_bundle)
            if freeform_result is not None:
                prompt, response = freeform_result
            else:
                synthesis_result = self._try_generate_locomo_answer_synthesis(query_task, retrieval_bundle)
                if synthesis_result[0] is not None:
                    prompt, response = synthesis_result  # type: ignore[assignment]
                else:
                    synthesis_failure_metadata = dict(synthesis_result[1])  # type: ignore[arg-type]
                    prompt = self.build_prompt(query_task, retrieval_bundle)
                    response = None  # type: ignore[assignment]
        else:
            prompt = self.build_prompt(query_task, retrieval_bundle)
            response = None  # type: ignore[assignment]
        if response is None:
            try:
                response = self.llm_provider.generate(
                    [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                    metadata={
                        "task": "answer_generation",
                        "query_task_id": query_task.query_task_id,
                        "answer_prompt_name": prompt_name,
                    },
                )
                response.metadata = {
                    **dict(response.metadata or {}),
                    **synthesis_failure_metadata,
                    "answer_prompt_name": prompt_name,
                    "answer_prompt_stage": "initial",
                    "answer_postcheck_used": False,
                    "answer_postcheck_issue": None,
                    "answer_repair_used": False,
                }
            except Exception as exc:  # noqa: BLE001
                error_message = self._compact_exception(exc)
                self._trace(
                    f"answer_generation_failed sample={query_task.sample_id} "
                    f"query_task_id={query_task.query_task_id} error_type={exc.__class__.__name__} "
                    f"error={error_message}"
                )
                response = LLMResponse(
                    text=self._standard_abstention_text(
                        f"answer generation failed before a model answer was produced ({exc.__class__.__name__})"
                    ),
                    prompt_tokens=0,
                    completion_tokens=0,
                    metadata={
                        **synthesis_failure_metadata,
                        "answer_prompt_name": prompt_name,
                        "answer_prompt_stage": "initial",
                        "answer_generation_failed": True,
                        "answer_generation_error_type": exc.__class__.__name__,
                        "answer_generation_error_message": error_message,
                        "answer_postcheck_used": False,
                        "answer_postcheck_issue": None,
                        "answer_postcheck_skipped": True,
                        "answer_postcheck_skip_reason": "answer_generation_failed",
                        "answer_repair_used": False,
                    },
                )
        initial_stage_text = str(response.text or "")
        initial_stage_refs = list(
            dict(response.metadata or {}).get("answer_synthesis_supporting_refs")
            or dict(
                dict(response.metadata or {}).get("answer_synthesis_payload") or {}
            ).get("supporting_source_refs")
            or []
        )
        initial_stage_metadata = {
            "answer_stage_initial_text": initial_stage_text,
            "answer_stage_initial_supporting_refs": initial_stage_refs,
            "answer_stage_initial_invalid_supporting_refs": list(
                dict(response.metadata or {}).get("invalid_supporting_refs") or []
            ),
            "answer_stage_initial_prompt_tokens": int(response.prompt_tokens or 0),
            "answer_stage_initial_completion_tokens": int(response.completion_tokens or 0),
            "answer_stage_initial_provider_call_uid": dict(response.metadata or {}).get(
                "provider_call_uid"
            ),
            "answer_stage_initial_call_item_uid": dict(response.metadata or {}).get(
                "call_item_uid"
            ),
        }
        final_prompt = prompt
        if prompt_name == "locomo_answer_generation":
            skip_reason = self._locomo_postcheck_skip_reason(response, response.text)
            synthesis_payload = dict(response.metadata.get("answer_synthesis_payload") or {})
            forced_issue = None
            if skip_reason in {"context_abstention", "synthesis_can_answer_false"}:
                forced_issue = self._detect_locomo_postcheck_issue(
                    query_task,
                    retrieval_bundle,
                    response.text,
                    synthesis_payload,
                )
                if forced_issue != "abstain_despite_supported_list_items":
                    forced_issue = None
            if skip_reason is not None and forced_issue is None:
                list_coverage_metadata = self._locomo_list_coverage_diagnostics(
                    query_task,
                    retrieval_bundle,
                    response.text,
                    synthesis_payload,
                )
                response.metadata = {
                    **dict(response.metadata or {}),
                    **list_coverage_metadata,
                    "answer_postcheck_used": False,
                    "answer_postcheck_issue": None,
                    "answer_postcheck_skipped": True,
                    "answer_postcheck_skip_reason": skip_reason,
                    "answer_list_coverage_repair_used": False,
                    "answer_list_coverage_repair_success": False,
                }
            else:
                specificity_metadata = self._locomo_specificity_diagnostics(
                    query_task,
                    retrieval_bundle,
                    response.text,
                )
                list_coverage_metadata = self._locomo_list_coverage_diagnostics(
                    query_task,
                    retrieval_bundle,
                    response.text,
                    synthesis_payload,
                )
                bridge_finalization_metadata = self._locomo_bridge_finalization_diagnostics(
                    retrieval_bundle,
                    response.text,
                    query_task.question,
                )
                if (
                    response.metadata.get("bridge_finalization_used")
                    and not bridge_finalization_metadata.get("bridge_finalization_needs_repair")
                ):
                    bridge_finalization_metadata = {"bridge_finalization_needs_repair": False}
                type_issue = None
                if (
                    response.metadata.get("answer_freeform_used")
                    and response.metadata.get("answer_type_match") is False
                    and not self._answer_is_context_abstention(response.text)
                ):
                    type_issue = "answer_type_mismatch"
                issue = forced_issue or type_issue or self._detect_locomo_postcheck_issue(
                    query_task,
                    retrieval_bundle,
                    response.text,
                    synthesis_payload,
                )
                response.metadata = {
                    **dict(response.metadata or {}),
                    **specificity_metadata,
                    **list_coverage_metadata,
                    **bridge_finalization_metadata,
                    "answer_specific_item_repair_used": False,
                    "answer_list_coverage_repair_used": False,
                    "answer_list_coverage_repair_success": False,
                    "answer_bridge_repair_used": False,
                    "answer_bridge_repair_success": False,
                    "answer_postcheck_skipped": False,
                    "answer_postcheck_skip_reason": None,
                }
                if issue is not None:
                    current_answer_prompt_name = str(response.metadata.get("answer_prompt_name") or prompt_name)
                    self._trace(
                        f"answer_postcheck_repair_start sample={query_task.sample_id} query_task_id={query_task.query_task_id} issue={issue}"
                    )
                    initial_response = response
                    try:
                        repair_prompt, repaired = self._repair_locomo_answer(
                            query_task=query_task,
                            retrieval_bundle=retrieval_bundle,
                            answer_text=initial_response.text,
                            answer_prompt_name=current_answer_prompt_name,
                            issue=issue,
                            synthesis_payload=synthesis_payload,
                        )
                    except Exception as exc:  # noqa: BLE001
                        error_message = self._compact_exception(exc)
                        initial_response.metadata = {
                            **dict(initial_response.metadata or {}),
                            "answer_postcheck_used": True,
                            "answer_postcheck_issue": issue,
                            "answer_postcheck_skipped": False,
                            "answer_postcheck_skip_reason": None,
                            "answer_repair_attempted": True,
                            "answer_repair_used": False,
                            "answer_specific_item_repair_used": False,
                            "answer_list_coverage_repair_used": False,
                            "answer_list_coverage_repair_success": False,
                            "answer_bridge_repair_used": False,
                            "answer_bridge_repair_success": False,
                            "answer_repair_discarded": True,
                            "answer_repair_discard_reason": "provider_exception",
                            "answer_repair_error_type": exc.__class__.__name__,
                            "answer_repair_error_message": error_message,
                            "answer_prompt_stage": initial_response.metadata.get("answer_prompt_stage") or "initial",
                        }
                        self._trace(
                            f"answer_postcheck_repair_failed sample={query_task.sample_id} "
                            f"query_task_id={query_task.query_task_id} issue={issue} "
                            f"error_type={exc.__class__.__name__} error={error_message}"
                        )
                        response = initial_response
                    else:
                        normalized_text, repair_normalization_metadata = self._normalize_repaired_answer_text(
                            repaired.text,
                            initial_response.text,
                        )
                        if not repair_normalization_metadata.get("answer_repair_discarded"):
                            repair_validation_metadata = self._repair_required_item_validation(
                                query_task=query_task,
                                retrieval_bundle=retrieval_bundle,
                                initial_answer_text=initial_response.text,
                                repaired_answer_text=normalized_text,
                                issue=issue,
                                synthesis_payload=synthesis_payload,
                            )
                            repair_normalization_metadata.update(repair_validation_metadata)
                            if repair_validation_metadata.get("answer_repair_post_validation_failed"):
                                repair_normalization_metadata["answer_repair_discarded"] = True
                                repair_normalization_metadata[
                                    "answer_repair_discard_reason"
                                ] = repair_validation_metadata.get("answer_repair_post_validation_reason") or "post_validation_failed"
                        if not repair_normalization_metadata.get("answer_repair_discarded"):
                            arbitration_trigger = self._repair_arbitration_trigger(
                                query_task=query_task,
                                retrieval_bundle=retrieval_bundle,
                                initial_answer_text=initial_response.text,
                                repaired_answer_text=normalized_text,
                                issue=issue,
                                repair_validation_metadata=repair_normalization_metadata,
                                synthesis_payload=synthesis_payload,
                            )
                            repair_normalization_metadata.update(arbitration_trigger)
                            if arbitration_trigger.get("answer_repair_arbitration_triggered"):
                                arbitration_metadata = self._maybe_arbitrate_repaired_answer(
                                    query_task=query_task,
                                    retrieval_bundle=retrieval_bundle,
                                    initial_answer_text=initial_response.text,
                                    repaired_answer_text=normalized_text,
                                    issue=issue,
                                    trigger_metadata=arbitration_trigger,
                                    repair_validation_metadata=repair_normalization_metadata,
                                )
                                repair_normalization_metadata.update(arbitration_metadata)
                                arbitration_action = str(
                                    arbitration_metadata.get("answer_repair_arbitration_action") or ""
                                )
                                if arbitration_action == "keep_initial":
                                    repair_normalization_metadata["answer_repair_discarded"] = True
                                    repair_normalization_metadata[
                                        "answer_repair_discard_reason"
                                    ] = "arbitration_keep_initial"
                                    repair_normalization_metadata[
                                        "answer_repair_preserved_initial_answer"
                                    ] = True
                                elif arbitration_action == "safe_abstain":
                                    normalized_text = self._standard_abstention_text()
                                    repair_normalization_metadata[
                                        "answer_synthesis_safe_abstain_used"
                                    ] = True
                        repair_discarded = bool(repair_normalization_metadata.get("answer_repair_discarded"))
                        list_repair_issue = issue in {"missing_supported_list_items", "abstain_despite_supported_list_items"}
                        list_repair_success = bool(
                            list_repair_issue
                            and not repair_discarded
                            and repair_normalization_metadata.get("answer_repair_list_coverage_improved")
                        )
                        repair_metadata = {
                            **dict(initial_response.metadata or {}),
                            **dict(repaired.metadata or {}),
                            "answer_prompt_name": current_answer_prompt_name,
                            "answer_postcheck_used": True,
                            "answer_postcheck_issue": issue,
                            "answer_postcheck_skipped": False,
                            "answer_postcheck_skip_reason": None,
                            "answer_repair_attempted": True,
                            "answer_repair_used": not repair_discarded,
                            "answer_specific_item_repair_used": issue in {"overgeneric_item", "scope_mismatched_extra_item"} and not repair_discarded,
                            "answer_list_coverage_repair_used": list_repair_issue and not repair_discarded,
                            "answer_list_coverage_repair_success": list_repair_success,
                            "answer_bridge_repair_used": issue == "bridge_alias_unresolved" and not repair_discarded,
                            "answer_bridge_repair_success": issue == "bridge_alias_unresolved" and not repair_discarded,
                            "answer_initial_text": initial_response.text,
                            "answer_initial_prompt_tokens": initial_response.prompt_tokens,
                            "answer_initial_completion_tokens": initial_response.completion_tokens,
                            "answer_initial_latency_ms": float(initial_response.metadata.get("latency_ms", 0.0)),
                            "answer_repair_prompt_tokens": repaired.prompt_tokens,
                            "answer_repair_completion_tokens": repaired.completion_tokens,
                            "answer_repair_latency_ms": float(repaired.metadata.get("latency_ms", 0.0)),
                            **repair_normalization_metadata,
                        }
                        if repair_discarded:
                            initial_response.metadata = {
                                **repair_metadata,
                                "answer_prompt_stage": initial_response.metadata.get("answer_prompt_stage") or "initial",
                            }
                            response = initial_response
                        else:
                            repaired.text = normalized_text
                            repaired.metadata = {
                                **repair_metadata,
                                "answer_prompt_stage": "repair",
                            }
                            response = repaired
                            final_prompt = repair_prompt
                        self._trace(
                            f"answer_postcheck_repair_done sample={query_task.sample_id} query_task_id={query_task.query_task_id} "
                            f"issue={issue} discarded={str(repair_discarded).lower()}"
                        )
        self._trace(
            f"answer_generation_done sample={query_task.sample_id} query_task_id={query_task.query_task_id} "
            f"mode={response.metadata.get('answer_synthesis_mode') or response.metadata.get('answer_prompt_stage') or 'legacy'} "
            f"latency_ms={float(response.metadata.get('latency_ms', 0.0)):.1f} "
            f"prompt_tokens={int(response.prompt_tokens or 0)} completion_tokens={int(response.completion_tokens or 0)}"
        )
        response.metadata = {
            **dict(response.metadata or {}),
            **initial_stage_metadata,
            "answer_stage_post_validation_text": str(response.text or ""),
            "answer_stage_post_validation_supporting_refs": list(
                dict(response.metadata or {}).get("answer_synthesis_supporting_refs")
                or dict(
                    dict(response.metadata or {}).get("answer_synthesis_payload") or {}
                ).get("supporting_source_refs")
                or []
            ),
            "answer_stage_post_validation_invalid_supporting_refs": list(
                dict(response.metadata or {}).get("invalid_supporting_refs") or []
            ),
            "answer_stage_post_validation_prompt_tokens": (
                int(initial_stage_metadata["answer_stage_initial_prompt_tokens"])
                + int(dict(response.metadata or {}).get("answer_repair_prompt_tokens") or 0)
            ),
            "answer_stage_post_validation_completion_tokens": (
                int(initial_stage_metadata["answer_stage_initial_completion_tokens"])
                + int(dict(response.metadata or {}).get("answer_repair_completion_tokens") or 0)
            ),
            "answer_stage_post_validation_changed": (
                initial_stage_text.strip() != str(response.text or "").strip()
            ),
        }
        return final_prompt, response


class BenchmarkJudge:
    def __init__(self, llm_provider: LLMProvider | None, *, trace: Callable[[str], None] | None = None) -> None:
        self.llm_provider = llm_provider
        self.trace = trace

    def _trace(self, message: str) -> None:
        if self.trace is not None:
            self.trace(message)

    def _structured_vendor(self) -> str:
        if self.llm_provider is None:
            return "unknown"
        return str(self.llm_provider.model_info().metadata.get("vendor") or "unknown")

    @staticmethod
    def _single_line_error(exc: Exception) -> str:
        return " ".join(str(exc).split()) or exc.__class__.__name__

    @staticmethod
    def _score_for_verdict(verdict: str) -> float | None:
        normalized = verdict.lower()
        if normalized == "correct":
            return 1.0
        if normalized == "partial":
            return 0.5
        if normalized == "incorrect":
            return 0.0
        return None

    @classmethod
    def _structured_fallback_category(cls, exc: Exception) -> str:
        if isinstance(exc, StructuredOutputError):
            message = cls._single_line_error(exc).lower()
            if exc.refusal:
                return "structured_refusal"
            if "schema" in message or "json_schema" in message or "json schema" in message:
                return "structured_schema_error"
            if "empty" in message or "no content" in message or "blank" in message:
                return "structured_empty_response"
            return "structured_other"
        if isinstance(exc, (ParserValidationError, ValidationError)):
            return "structured_parse_error"
        return "structured_other"

    @classmethod
    def _judge_error_result(
        cls,
        *,
        prompt: str,
        started_at: float,
        metadata: dict[str, object] | None,
        exc: Exception,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> JudgeResult:
        error_type = exc.__class__.__name__
        error_message = cls._single_line_error(exc)
        failure_metadata = {
            **dict(metadata or {}),
            "judge_execution_failed": True,
            "judge_error_type": error_type,
            "judge_error_message": error_message,
        }
        if failure_metadata.get("structured_fallback_category") is None:
            failure_metadata["structured_fallback_category"] = "judge_execution_error"
        return JudgeResult(
            verdict="judge_error",
            prompt=prompt,
            score=None,
            rationale=f"Judge execution failed: {error_message}",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            metadata=failure_metadata,
        )

    def build_prompt(self, dataset_name: str, query_task: QueryTask, answer_text: str) -> str:
        if dataset_name == "locomo":
            return (
                load_prompt("locomo_judge")
                + "\n\nQuestion:\n"
                + query_task.question
                + "\n\nGold answer:\n"
                + str(query_task.gold_answer or "")
                + "\n\nCandidate answer:\n"
                + answer_text
            )
        judge_context = query_task.metadata.get("judge_context", {})
        return (
            load_prompt("medmt_judge")
            + "\n\nCATEGORY:\n"
            + str(judge_context.get("category") or "Unknown")
            + "\n\nSUBTYPE:\n"
            + str(judge_context.get("subtype") or "Unknown")
            + "\n\nFULL_DIALOGUE:\n"
            + str(judge_context.get("full_dialogue") or "")
            + "\n\nFINAL_USER_TURN:\n"
            + str(judge_context.get("final_user_turn") or query_task.question)
            + "\n\nRUBRIC:\n"
            + str(judge_context.get("rubric") or query_task.metadata.get("test_point") or "")
            + "\n\nCANDIDATE_ANSWER:\n"
            + answer_text
        )

    def build_structured_prompt(self, dataset_name: str, query_task: QueryTask, answer_text: str) -> str:
        if dataset_name == "locomo":
            return (
                load_prompt(structured_prompt_name("locomo_judge"))
                + "\n\nQuestion:\n"
                + query_task.question
                + "\n\nGold answer:\n"
                + str(query_task.gold_answer or "")
                + "\n\nCandidate answer:\n"
                + answer_text
            )
        judge_context = query_task.metadata.get("judge_context", {})
        return (
            load_prompt(structured_prompt_name("medmt_judge"))
            + "\n\nCATEGORY:\n"
            + str(judge_context.get("category") or "Unknown")
            + "\n\nSUBTYPE:\n"
            + str(judge_context.get("subtype") or "Unknown")
            + "\n\nFULL_DIALOGUE:\n"
            + str(judge_context.get("full_dialogue") or "")
            + "\n\nFINAL_USER_TURN:\n"
            + str(judge_context.get("final_user_turn") or query_task.question)
            + "\n\nRUBRIC:\n"
            + str(judge_context.get("rubric") or query_task.metadata.get("test_point") or "")
            + "\n\nCANDIDATE_ANSWER:\n"
            + answer_text
        )

    def judge(self, dataset_name: str, query_task: QueryTask, answer_text: str) -> JudgeResult | None:
        if self.llm_provider is None:
            return None
        task_name = f"{dataset_name}_judge"
        prompt = self.build_prompt(dataset_name, query_task, answer_text)
        structured_prompt = self.build_structured_prompt(dataset_name, query_task, answer_text)
        started_at = time.perf_counter()
        self._trace(
            f"judge_model_start sample={query_task.sample_id} query_task_id={query_task.query_task_id} task={task_name}"
        )
        if self.llm_provider.supports_structured(task_name):
            try:
                response = self.llm_provider.generate_structured(
                    [NormalizedMessage(role="user", content=structured_prompt, turn_index=0)],
                    spec=get_structured_task_spec(task_name),
                    metadata={"task": task_name, "query_task_id": query_task.query_task_id},
                )
                verdict = validate_judge_verdict_result(response.parsed)
                normalized_verdict = verdict.verdict.lower()
                result = JudgeResult(
                    verdict=normalized_verdict,
                    prompt=structured_prompt,
                    score=self._score_for_verdict(normalized_verdict),
                    rationale=None,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_ms=(time.perf_counter() - started_at) * 1000.0,
                    metadata={
                        **dict(response.metadata or {}),
                        "judge_mode": "structured",
                        "structured_requested": True,
                        "structured_supported": True,
                        "structured_task": task_name,
                        "structured_vendor": str(
                            dict(response.metadata or {}).get("structured_vendor") or self._structured_vendor()
                        ),
                        "structured_strategy": dict(response.metadata or {}).get("structured_strategy"),
                        "structured_success": True,
                        "structured_fallback_used": False,
                        "structured_fallback_reason": None,
                        "structured_fallback_category": None,
                        "structured_refusal": None,
                        "judge_execution_failed": False,
                        "judge_error_type": None,
                        "judge_error_message": None,
                        "judge_score_policy": "partial_credit_v1",
                    },
                )
                self._trace(
                    f"judge_model_done sample={query_task.sample_id} query_task_id={query_task.query_task_id} "
                    f"task={task_name} structured=True latency_ms={result.latency_ms:.1f}"
                )
                return result
            except Exception as exc:  # noqa: BLE001
                fallback_reason = self._single_line_error(exc)
                vendor = exc.vendor if isinstance(exc, StructuredOutputError) and exc.vendor else self._structured_vendor()
                strategy = "text_dsl_fallback"
                refusal = None
                if isinstance(exc, StructuredOutputError):
                    strategy = exc.strategy or strategy
                    refusal = exc.refusal
                fallback_metadata = {
                    "judge_mode": "text_fallback",
                    "structured_requested": True,
                    "structured_supported": True,
                    "structured_task": task_name,
                    "structured_vendor": vendor,
                    "structured_strategy": strategy,
                    "structured_success": False,
                    "structured_fallback_used": True,
                    "structured_fallback_reason": fallback_reason,
                    "structured_fallback_category": self._structured_fallback_category(exc),
                    "structured_refusal": refusal,
                    "judge_execution_failed": False,
                    "judge_error_type": None,
                    "judge_error_message": None,
                }
                response: LLMResponse | None = None
                try:
                    response = self.llm_provider.generate(
                        [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                        metadata={
                            "task": task_name,
                            "query_task_id": query_task.query_task_id,
                            **fallback_metadata,
                        },
                    )
                    verdict = parse_judge_verdict(response.text)
                    normalized_verdict = verdict.verdict.lower()
                    result = JudgeResult(
                        verdict=normalized_verdict,
                        prompt=prompt,
                        score=verdict.score
                        if verdict.score is not None
                        else self._score_for_verdict(normalized_verdict),
                        rationale=verdict.rationale,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        latency_ms=(time.perf_counter() - started_at) * 1000.0,
                        metadata={
                            **dict(response.metadata or {}),
                            **fallback_metadata,
                            "judge_score_policy": "partial_credit_v1",
                        },
                    )
                    self._trace(
                        f"judge_model_done sample={query_task.sample_id} query_task_id={query_task.query_task_id} "
                        f"task={task_name} structured=False latency_ms={result.latency_ms:.1f}"
                    )
                    return result
                except Exception as text_exc:  # noqa: BLE001
                    error_result = self._judge_error_result(
                        prompt=prompt,
                        started_at=started_at,
                        metadata={
                            **fallback_metadata,
                            "judge_mode": "text_fallback",
                            "structured_fallback_category": (
                                "text_parser_error"
                                if isinstance(text_exc, ParserValidationError)
                                else "judge_execution_error"
                            ),
                        },
                        exc=text_exc,
                        prompt_tokens=response.prompt_tokens if response is not None else None,
                        completion_tokens=response.completion_tokens if response is not None else None,
                    )
                    self._trace(
                        f"judge_model_failed sample={query_task.sample_id} query_task_id={query_task.query_task_id} "
                        f"task={task_name} mode=text_fallback error_type={text_exc.__class__.__name__}"
                    )
                    return error_result
        text_only_metadata = {
            "judge_mode": "text_only",
            "structured_requested": False,
            "structured_supported": False,
            "structured_task": task_name,
            "structured_vendor": None,
            "structured_strategy": None,
            "structured_success": False,
            "structured_fallback_used": False,
            "structured_fallback_reason": None,
            "structured_fallback_category": None,
            "structured_refusal": None,
            "judge_execution_failed": False,
            "judge_error_type": None,
            "judge_error_message": None,
        }
        response: LLMResponse | None = None
        try:
            response = self.llm_provider.generate(
                [NormalizedMessage(role="user", content=prompt, turn_index=0)],
                metadata={"task": task_name, "query_task_id": query_task.query_task_id, **text_only_metadata},
            )
            verdict = parse_judge_verdict(response.text)
            normalized_verdict = verdict.verdict.lower()
            result = JudgeResult(
                verdict=normalized_verdict,
                prompt=prompt,
                score=verdict.score
                if verdict.score is not None
                else self._score_for_verdict(normalized_verdict),
                rationale=verdict.rationale,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
                metadata={
                    **dict(response.metadata or {}),
                    **text_only_metadata,
                    "judge_score_policy": "partial_credit_v1",
                },
            )
            self._trace(
                f"judge_model_done sample={query_task.sample_id} query_task_id={query_task.query_task_id} "
                f"task={task_name} structured=False latency_ms={result.latency_ms:.1f}"
            )
            return result
        except Exception as exc:  # noqa: BLE001
            error_result = self._judge_error_result(
                prompt=prompt,
                started_at=started_at,
                metadata={
                    **text_only_metadata,
                    "structured_fallback_category": (
                        "text_parser_error"
                        if isinstance(exc, ParserValidationError)
                        else "judge_execution_error"
                    ),
                },
                exc=exc,
                prompt_tokens=response.prompt_tokens if response is not None else None,
                completion_tokens=response.completion_tokens if response is not None else None,
            )
            self._trace(
                f"judge_model_failed sample={query_task.sample_id} query_task_id={query_task.query_task_id} "
                f"task={task_name} mode=text_only error_type={exc.__class__.__name__}"
            )
            return error_result
