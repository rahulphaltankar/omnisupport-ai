"""Field-level leakage register, enforced in code.

Prediction moment (T0): **the instant a ticket is created, before any human triage**.
Only information the customer/order system already holds at that moment may be used.
Every raw field has exactly one role; ``assert_features_valid`` makes violations fail loudly.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FEATURE_OK = "feature_ok"
TARGET = "target"
POST_OUTCOME = "excluded_post_outcome"
TRIAGE = "excluded_assigned_at_triage"
SENSITIVE = "evaluation_only_sensitive"
IDENTIFIER = "excluded_identifier"
AMBIGUOUS = "excluded_ambiguous_timing"


@dataclass(frozen=True)
class FieldRule:
    role: str
    available_at_t0: bool
    rationale: str


REGISTER: dict[str, FieldRule] = {
    "ticket_id": FieldRule(IDENTIFIER, True, "Row identifier; carries no behavioural signal and would let a model memorise rows."),
    "customer_id": FieldRule(IDENTIFIER, True, "Used only as a CV/split grouping key to stop the same customer straddling folds."),
    "order_id": FieldRule(IDENTIFIER, True, "Identifier; sequential with ticket order so it can proxy time/ordering."),
    "image_id": FieldRule(IDENTIFIER, True, "Only 480 rows link to an image; used by the vision task, not by tabular models."),
    "timestamp": FieldRule(FEATURE_OK, True, "Ticket creation time: used for calendar features and for the temporal train/val/test split."),
    "support_channel": FieldRule(FEATURE_OK, True, "Known at creation."),
    "customer_region": FieldRule(FEATURE_OK, True, "Customer profile attribute; 'Unknown' retained as a category."),
    "customer_tenure_months": FieldRule(FEATURE_OK, True, "Account age at creation."),
    "product_category": FieldRule(FEATURE_OK, True, "Attribute of the order the ticket refers to."),
    "order_value": FieldRule(FEATURE_OK, True, "Order value in GBP known from the order system."),
    "delivery_delay_days": FieldRule(FEATURE_OK, True, "Delay observed by the carrier/order system when the ticket is opened (assumption A3)."),
    "previous_ticket_count": FieldRule(FEATURE_OK, True, "Count of PRIOR tickets by definition, so excludes the current one."),
    "issue_title": FieldRule(FEATURE_OK, True, "Customer-entered title (30 distinct strings); written before triage."),
    "issue_description": FieldRule(FEATURE_OK, True, "Customer-entered message."),
    "attachment_available": FieldRule(FEATURE_OK, True, "Attachment uploaded by the customer at creation (assumption A4)."),
    "issue_category": FieldRule(TRIAGE, False, "Operational label assigned by support staff after reading the ticket. Excluded from T0 tabular models "
                                              "(the text itself is the legitimate input); it is the target of the NLP classifier."),
    "priority": FieldRule(TRIAGE, False, "Assigned during triage and strongly outcome-linked; excluded to avoid circular prediction."),
    "customer_sentiment": FieldRule(POST_OUTCOME, False, "Documented as 'derived from post-contact review process'; escalation rate 43% vs 20% by sentiment."),
    "refund_requested": FieldRule(AMBIGUOUS, False, "Timing of recording unclear and it is determined by issue type; excluded conservatively (no gain measured)."),
    "refund_amount": FieldRule(POST_OUTCOME, False, "Known only after the case is handled."),
    "csat_score": FieldRule(POST_OUTCOME, False, "Collected after closure (knowledge_base/29_csat_followup.md); 56,852 missing."),
    "resolution_status_after_7d": FieldRule(POST_OUTCOME, False, "Status observed seven days later."),
    "final_agent_disposition": FieldRule(POST_OUTCOME, False, "'specialist_escalation' is identical to escalated=1 (perfect leak)."),
    "escalated": FieldRule(TARGET, False, "Classification target: case routed to a specialist queue."),
    "resolution_time_hours": FieldRule(TARGET, False, "Regression target: hours to operational resolution."),
    "customer_age_band": FieldRule(SENSITIVE, True, "Age band: kept out of models (data minimisation / fairness); used only for subgroup evaluation."),
    "preferred_language": FieldRule(SENSITIVE, True, "Language: kept out of models; used only for subgroup evaluation."),
    "accessibility_support_flag": FieldRule(SENSITIVE, True, "Accessibility need: kept out of models; used for subgroup evaluation (ablation in ml_classification)."),
}

ALLOWED_RAW_FIELDS = sorted(k for k, v in REGISTER.items() if v.role == FEATURE_OK)
FORBIDDEN_RAW_FIELDS = sorted(k for k, v in REGISTER.items() if v.role in {POST_OUTCOME, TRIAGE, TARGET, AMBIGUOUS})
SENSITIVE_FIELDS = sorted(k for k, v in REGISTER.items() if v.role == SENSITIVE)


class LeakageError(ValueError):
    pass


def assert_raw_columns_allowed(columns) -> None:
    """Fail if a *forbidden raw field* is passed to a feature builder."""
    bad = [c for c in columns if c in FORBIDDEN_RAW_FIELDS]
    if bad:
        raise LeakageError(f"Forbidden (post-outcome/triage/target) fields supplied to features: {bad}")


def assert_features_valid(feature_names, allow_sensitive: bool = False) -> None:
    """Final guard on a model's input columns (engineered names are checked by source field)."""
    from omnisupport.data.features import FEATURE_SOURCES  # local import: avoid cycle

    problems = []
    for f in feature_names:
        if f in FORBIDDEN_RAW_FIELDS:
            problems.append(f"{f}: forbidden raw field ({REGISTER[f].role})")
            continue
        srcs = FEATURE_SOURCES.get(f)
        if srcs is None:
            problems.append(f"{f}: not a registered feature")
            continue
        for s in srcs:
            role = REGISTER[s].role
            if role == SENSITIVE and not allow_sensitive:
                problems.append(f"{f}: derives from sensitive field {s}")
            elif role not in (FEATURE_OK, SENSITIVE):
                problems.append(f"{f}: derives from {s} ({role})")
    if problems:
        raise LeakageError("; ".join(problems))


def register_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"field": k, "role": v.role, "available_at_prediction_time_T0": v.available_at_t0, "rationale": v.rationale}
         for k, v in REGISTER.items()]
    )
