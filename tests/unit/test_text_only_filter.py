from __future__ import annotations

from trajpatch.analysis.text_only_filter import audit_text_only_visibility


def _row(
    *,
    question: str,
    gold_answer: str,
    gold_refs: list[str],
    reference_items: list[str] | None = None,
) -> dict:
    return {
        "sample_id": "conv-1",
        "query_task_id": "conv-1_qa_0",
        "question": question,
        "gold_answer": gold_answer,
        "metadata": {
            "query_metadata": {"gold_evidence_refs": gold_refs},
            "semantic_metrics": {"f1_reference_items": reference_items or []},
        },
    }


def test_book_title_only_on_cover_is_excluded_as_ocr() -> None:
    result = audit_text_only_visibility(
        _row(
            question="What books has Melanie read?",
            gold_answer='"Nothing is Impossible", "Charlotte\'s Web"',
            gold_refs=["D7:8", "D6:10"],
            reference_items=["Nothing is Impossible", "Charlotte's Web"],
        ),
        {
            "D7:8": (
                "This book I read last year reminds me to always pursue my dreams. "
                "[shared image: a photography of a book cover with a gold coin on it]"
            ),
            "D6:10": "She also likes to read! We started Charlotte's Web together.",
        },
    )

    assert result["excluded_from_text_only"] is True
    assert result["visual_dependency_type"] == "ocr_text_on_image"
    assert result["gold_items_missing_from_text_input"] == ["Nothing is Impossible"]


def test_text_present_answer_remains_eligible() -> None:
    result = audit_text_only_visibility(
        _row(
            question="What book did they start?",
            gold_answer="Charlotte's Web",
            gold_refs=["D6:10"],
            reference_items=["Charlotte's Web"],
        ),
        {"D6:10": "She also likes to read! We started Charlotte's Web together."},
    )

    assert result["excluded_from_text_only"] is False
    assert result["text_only_eligible"] is True


def test_existing_image_caption_counts_as_text_input() -> None:
    result = audit_text_only_visibility(
        _row(
            question="How many tortoises were in the image?",
            gold_answer="two tortoises",
            gold_refs=["D5:4"],
            reference_items=["two tortoises"],
        ),
        {"D5:4": "I was walking them. [shared image: a photo of two tortoises on a path]"},
    )

    assert result["excluded_from_text_only"] is False
    assert result["gold_items_missing_from_text_input"] == []


def test_temporal_answer_without_image_is_not_visual_excluded() -> None:
    result = audit_text_only_visibility(
        _row(
            question="When did Melanie go on a hike?",
            gold_answer="19 October 2023",
            gold_refs=["D18:17"],
            reference_items=["19 October 2023"],
        ),
        {"D18:17": "date=20 October 2023 | Melanie: I went hiking yesterday."},
    )

    assert result["excluded_from_text_only"] is False
    assert result["visual_dependency_type"] is None


def test_ambiguous_image_evidence_defaults_to_included() -> None:
    result = audit_text_only_visibility(
        _row(
            question="What did Melanie mention?",
            gold_answer="a keepsake",
            gold_refs=["D1:1"],
            reference_items=["keepsake"],
        ),
        {"D1:1": "This meant a lot to me. [shared image: a personal photo]"},
    )

    assert result["excluded_from_text_only"] is False
    assert result["visual_dependency_type"] == "ambiguous_needs_review"
    assert result["exclusion_reason"] == "ambiguous_needs_review"
