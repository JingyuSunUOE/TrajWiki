"""Pydantic models used by the memory subsystem."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, validator


ClaimStatus = Literal["active", "deprecated", "contradictory", "needs-confirmation"]
ClaimOpType = Literal["ADD", "REVISE", "DEPRECATE"]


class MemoryClaim(BaseModel):
    claim_id: str
    status: ClaimStatus
    source_message_ids: list[str]
    text: str


class ClaimTextExtractionClaim(BaseModel):
    status: ClaimStatus
    source_message_ids: list[str]
    supporting_quote: str
    text: str


class ClaimTextExtractionResult(BaseModel):
    has_claims: bool
    claims: list[ClaimTextExtractionClaim] = Field(default_factory=list)
    reason: str | None = None


class ClaimSignalExactTerm(BaseModel):
    surface: str
    category: str
    source_claim_id: str
    source_message_ids: list[str] = Field(default_factory=list)


class ClaimSignalFacet(BaseModel):
    relation: str
    value: str
    entity: str | None = None
    value_span: str | None = None
    source_claim_id: str
    source_message_ids: list[str] = Field(default_factory=list)


class ClaimSignalExtractionResult(BaseModel):
    exact_terms: list[ClaimSignalExactTerm] = Field(default_factory=list)
    facets: list[ClaimSignalFacet] = Field(default_factory=list)
    display_items: list[str] = Field(default_factory=list)
    display_named_entities: list[str] = Field(default_factory=list)
    display_counts: list[str] = Field(default_factory=list)
    display_key_facts: list[str] = Field(default_factory=list)


class ClaimOp(BaseModel):
    op: ClaimOpType
    target_claim_id: str
    new_claim_id: str | None = None
    source_message_ids: list[str]
    rationale: str
    claim_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodicMemoryInput(BaseModel):
    memory_type: Literal["episodic"]
    timestamp: str
    summary_content: str
    context: str
    keywords: list[str]
    links: list[str]
    status_flags: list[ClaimStatus]
    claims: list[MemoryClaim]
    ops: list[ClaimOp] = Field(default_factory=list)
    raw_text: str

    @property
    def semantic_text(self) -> str:
        return "\n".join(
            [
                self.summary_content,
                self.context,
                "Keywords: " + ", ".join(self.keywords),
            ]
        )


class TrajectoryMatchDecision(BaseModel):
    decision: Literal["CONTINUE", "NEW"]
    trajectory_id: str | None
    rationale: str


class ClaimTransitionDecision(BaseModel):
    decision: Literal["REVISE", "ADD"]
    previous_claim_id: str | None
    rationale: str


class JudgeVerdict(BaseModel):
    verdict: Literal["CORRECT", "PARTIAL", "INCORRECT"]
    score: float | None = None
    rationale: str | None = None

    @validator("score")
    def score_bounds(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return max(0.0, min(1.0, value))
