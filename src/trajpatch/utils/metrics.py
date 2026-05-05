"""Evaluation helpers for benchmark lexical metrics."""

from __future__ import annotations

import re
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def bleu1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    return max(0.0, min(1.0, overlap / len(pred_tokens)))


def evidence_recall(retrieved_refs: list[str], gold_evidence: list[str]) -> float:
    if not gold_evidence:
        return 1.0
    retrieved = set(retrieved_refs)
    gold = set(gold_evidence)
    return len(retrieved & gold) / len(gold)
