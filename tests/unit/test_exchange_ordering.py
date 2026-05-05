from __future__ import annotations

from trajpatch.pipeline.runner import PipelineRunner
from trajpatch.types import NormalizedMessage


def _message(role: str, content: str, turn_index: int, *, session_date: str | None = None) -> NormalizedMessage:
    return NormalizedMessage(
        role=role,  # type: ignore[arg-type]
        content=content,
        turn_index=turn_index,
        occurred_at=session_date,
        metadata={"session_date": session_date} if session_date else {},
        raw_message_id=f"sample-m{turn_index:04d}",
    )


def test_ordered_exchanges_keeps_user_assistant_pair() -> None:
    messages = [
        _message("user", "I smoke 5 cigarettes a day.", 1),
        _message("assistant", "Thanks, noted.", 2),
    ]

    exchanges = PipelineRunner._ordered_exchanges(messages)

    assert [[message.role for message in exchange] for exchange in exchanges] == [["user", "assistant"]]


def test_ordered_exchanges_keeps_session_start_assistant_singleton() -> None:
    messages = [_message("assistant", "My old area was hit by a flood.", 1)]

    exchanges = PipelineRunner._ordered_exchanges(messages)

    assert [[message.role for message in exchange] for exchange in exchanges] == [["assistant"]]
    assert PipelineRunner._assistant_only_exchange_count(exchanges) == 1


def test_ordered_exchanges_keeps_assistant_start_before_user_pair() -> None:
    messages = [
        _message("assistant", "My old area was hit by a flood.", 1),
        _message("user", "Where was that?", 2),
        _message("assistant", "West County.", 3),
    ]

    exchanges = PipelineRunner._ordered_exchanges(messages)

    assert [[message.role for message in exchange] for exchange in exchanges] == [
        ["assistant"],
        ["user", "assistant"],
    ]
    assert PipelineRunner._assistant_only_exchange_count(exchanges) == 1


def test_ordered_exchanges_keeps_consecutive_assistant_singletons() -> None:
    messages = [
        _message("assistant", "First update.", 1),
        _message("assistant", "Second update.", 2),
    ]

    exchanges = PipelineRunner._ordered_exchanges(messages)

    assert [[message.role for message in exchange] for exchange in exchanges] == [["assistant"], ["assistant"]]
    assert PipelineRunner._assistant_only_exchange_count(exchanges) == 2


def test_ordered_exchanges_does_not_pair_across_session_boundary() -> None:
    messages = [
        _message("user", "No worries, keep going.", 1, session_date="day one"),
        _message("assistant", "I went to Rome last week.", 2, session_date="day two"),
    ]

    exchanges = PipelineRunner._ordered_exchanges(messages)

    assert [[message.role for message in exchange] for exchange in exchanges] == [["assistant"]]
    assert exchanges[0][0].content == "I went to Rome last week."
    assert PipelineRunner._assistant_only_exchange_count(exchanges) == 1
