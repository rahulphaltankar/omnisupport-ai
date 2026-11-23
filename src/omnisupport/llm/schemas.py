"""Pydantic output contracts for every LLM task. Anything an LLM returns must validate against one of these."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

IssueCategory = Literal["delivery_late", "damaged_item", "refund_status", "wrong_item", "return_request", "payment_issue", "account_access",
                        "product_question", "warranty", "lost_parcel", "other"]
Priority = Literal["low", "medium", "high", "urgent"]
ISSUE_CATEGORIES = list(IssueCategory.__args__)  # type: ignore[attr-defined]
PRIORITIES = list(Priority.__args__)  # type: ignore[attr-defined]


class _Strict(BaseModel):
    model_config = {"extra": "ignore", "str_strip_whitespace": True}


class TicketExtraction(_Strict):
    """Extraction of a support record from a short message (kept compatible with the supplied starter schema)."""
    issue_category: IssueCategory
    priority: Priority
    order_id: Optional[str] = None
    customer_intent: str = Field(min_length=3, max_length=300)
    needs_human_review: bool
    missing_information: list[str] = Field(default_factory=list)

    @field_validator("order_id")
    @classmethod
    def _order_id_format(cls, v):
        if v in (None, "", "null", "None", "none"):
            return None
        if not __import__("re").fullmatch(r"ORD\d{8}", v.upper()):
            raise ValueError("order_id must look like ORD followed by 8 digits")
        return v.upper()


class ClassifySummary(_Strict):
    issue_category: IssueCategory
    summary: str = Field(min_length=3, max_length=500)
    uncertainties: list[str] = Field(default_factory=list)
    recommended_next_step: str = Field(min_length=3, max_length=300)


class AccountRisk(_Strict):
    issue_category: IssueCategory
    risk_flags: list[str] = Field(default_factory=list)
    needs_human_review: bool


class PolicySummary(_Strict):
    customer_facing_summary: str = Field(min_length=3, max_length=600)
    claims_supported_by_input: bool


class SafeResponse(_Strict):
    safe_response: str = Field(min_length=3, max_length=600)
    prohibited_request_detected: bool


class ActionProposal(_Strict):
    proposed_action: str = Field(min_length=3, max_length=300)
    needs_human_approval: bool
    reason: str = Field(min_length=3, max_length=400)


class EscalationSummary(_Strict):
    summary: str = Field(min_length=3, max_length=500)
    safety_risk: bool
    priority: Priority
    needs_human_review: bool


class ProductAnswer(_Strict):
    answer: str = Field(min_length=3, max_length=600)
    uncertainty: str = Field(min_length=3, max_length=300)
    recommended_source: str = Field(min_length=3, max_length=200)


SCHEMAS: dict[str, type[BaseModel]] = {
    "extraction": TicketExtraction, "classify_summarise": ClassifySummary, "account_risk": AccountRisk, "policy_summary": PolicySummary,
    "secret_guard": SafeResponse, "action_proposal": ActionProposal, "escalation_summary": EscalationSummary, "product_answer": ProductAnswer,
}
