"""Local, read-only/advisory tools over data/agent_store CSVs. Strict pydantic argument schemas; controlled errors; no external side effects."""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnisupport.config import AGENT_STORE, FRONTLINE_REFUND_LIMIT_GBP, PAYMENTS_RISK_REFUND_LIMIT_GBP

RETURN_WINDOW_DAYS = 30            # knowledge_base/01_returns_standard.md
MINOR_DAMAGE_PARTIAL_PCT = 0.20    # ASSUMPTION: policy gives no percentage (08_partial_refunds.md); 20% is a documented placeholder, always customer-confirmed
CONDITION_ALIASES = {"damaged": "damaged_item", "damage": "damaged_item", "wrong": "wrong_item", "minor": "minor_damage", "cosmetic": "minor_damage"}
CONDITIONS = ("damaged_item", "wrong_item", "minor_damage", "not_as_described", "change_of_mind", "unknown")
FULL_REFUND_CONDITIONS = {"damaged_item", "wrong_item"}


class ToolError(RuntimeError):
    """Raised for transient/unexpected tool failures (not for a controlled not-found)."""


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class OrderArgs(_Args):
    order_id: str

    @field_validator("order_id")
    @classmethod
    def _fmt(cls, v):
        v = v.upper()
        if not re.fullmatch(r"ORD\d{8}", v):
            raise ValueError("order_id must match ORD + 8 digits")
        return v


class CustomerArgs(_Args):
    customer_id: str = Field(pattern=r"^CUST\d{7}$")


class RefundArgs(OrderArgs):
    condition: str
    requested_amount: float | None = Field(default=None, ge=0)

    @field_validator("condition")
    @classmethod
    def _cond(cls, v):
        v = CONDITION_ALIASES.get(v, v)
        if v not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        return v


class EligibilityArgs(OrderArgs):
    condition: str = "unknown"

    @field_validator("condition")
    @classmethod
    def _cond(cls, v):
        return CONDITION_ALIASES.get(v, v)


class PolicyArgs(_Args):
    query: str = Field(min_length=3, max_length=400)


@lru_cache(maxsize=4)
def _table(name: str, key: str, store: str = str(AGENT_STORE)) -> dict:
    with open(Path(store) / name, newline="", encoding="utf-8") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def lookup_order(order_id: str, store: str = str(AGENT_STORE)) -> dict:
    a = OrderArgs(order_id=order_id)
    r = _table("orders.csv", "order_id", store).get(a.order_id)
    if r is None:
        return {"status": "not_found", "found": False, "order_id": a.order_id, "message": "No such order in the local store."}
    return {"status": "ok", "found": True, "order_id": r["order_id"], "customer_id": r["customer_id"], "product_category": r["product_category"], "order_value": float(r["order_value"]),
            "dispatch_state": r["dispatch_state"], "delivery_status": r["delivery_status"], "days_since_delivery": int(r["days_since_delivery"]), "order_timestamp": r["order_timestamp"]}


def lookup_customer(customer_id: str, store: str = str(AGENT_STORE)) -> dict:
    a = CustomerArgs(customer_id=customer_id)
    r = _table("customers.csv", "customer_id", store).get(a.customer_id)
    if r is None:
        return {"status": "not_found", "customer_id": a.customer_id}
    # age band / language deliberately not returned: not needed for the task (data minimisation)
    return {"status": "ok", "customer_id": r["customer_id"], "customer_region": r["customer_region"], "customer_tenure_months": int(r["customer_tenure_months"]),
            "accessibility_support_flag": int(r["accessibility_support_flag"])}


def check_return_eligibility(order_id: str, condition: str = "unknown", store: str = str(AGENT_STORE)) -> dict:
    a = EligibilityArgs(order_id=order_id, condition=condition)
    o = lookup_order(a.order_id, store)
    if o["status"] != "ok":
        return {"status": "not_found", "decision": "manual_review", "reason": "order not found", "order_id": a.order_id}
    d = o["days_since_delivery"]
    if o["delivery_status"] != "delivered":
        return {"status": "ok", "order_id": a.order_id, "decision": "manual_review", "reason": f"delivery status is '{o['delivery_status']}', not delivered", "days_since_delivery": d}
    if o["dispatch_state"] != "dispatched":
        return {"status": "ok", "order_id": a.order_id, "decision": "manual_review", "reason": "order not yet dispatched; cancellation (not return) may apply", "days_since_delivery": d}
    if a.condition in FULL_REFUND_CONDITIONS and d <= RETURN_WINDOW_DAYS * 2:
        return {"status": "ok", "order_id": a.order_id, "decision": "eligible", "reason": f"{a.condition.replace('_', ' ')} reported {d} days after delivery; no-cost return/remedy applies", "days_since_delivery": d}
    if d <= RETURN_WINDOW_DAYS:
        return {"status": "ok", "order_id": a.order_id, "decision": "eligible", "reason": f"within the {RETURN_WINDOW_DAYS}-day window ({d} days since delivery)", "days_since_delivery": d}
    return {"status": "ok", "order_id": a.order_id, "decision": "ineligible", "reason": f"{d} days since delivery exceeds the {RETURN_WINDOW_DAYS}-day window; warranty or manual review may apply", "days_since_delivery": d}


def calculate_refund(order_id: str, condition: str, requested_amount: float | None = None, store: str = str(AGENT_STORE)) -> dict:
    """Advisory proposal only. Refund never exceeds order value; thresholds from knowledge_base/09_refund_limits.md."""
    a = RefundArgs(order_id=order_id, condition=condition, requested_amount=requested_amount)
    o = lookup_order(a.order_id, store)
    if o["status"] != "ok":
        return {"status": "not_found", "order_id": a.order_id}
    v = o["order_value"]
    if a.condition in FULL_REFUND_CONDITIONS:
        amt, basis = v, "100% of order value (damaged/wrong item)"
    elif a.condition == "minor_damage":
        amt, basis = round(v * MINOR_DAMAGE_PARTIAL_PCT, 2), f"{int(MINOR_DAMAGE_PARTIAL_PCT * 100)}% partial refund (assumed placeholder rate; customer must agree)"
    else:
        amt, basis = 0.0, "no automatic refund for this condition; manual review"
    notes = []
    if a.requested_amount is not None and a.requested_amount > v:
        notes.append(f"requested GBP {a.requested_amount:g} exceeds order value GBP {v:g}; proposal capped at order value")
    reqd = a.requested_amount if (a.requested_amount is not None) else amt
    biggest = max(amt, reqd)
    over = biggest > FRONTLINE_REFUND_LIMIT_GBP
    route = "payments_or_risk_team" if biggest > PAYMENTS_RISK_REFUND_LIMIT_GBP else ("senior_agent" if over else "frontline")
    return {"status": "ok", "order_id": a.order_id, "order_value": v, "condition": a.condition, "proposed_refund": round(min(amt, v), 2), "basis": basis,
            "requires_approval": bool(over), "approval_route": route, "customer_confirmation_needed": a.condition == "minor_damage",
            "notes": notes, "advisory_only": True}


_RAG = None


def search_policy(query: str) -> dict:
    global _RAG
    a = PolicyArgs(query=query)
    if _RAG is None:
        from omnisupport.rag.pipeline import PolicyRAG
        _RAG = PolicyRAG()
    ans = _RAG.answer(a.query)
    return {"status": "ok" if not ans.abstained else "no_evidence", "answer": ans.answer, "citations": [{"doc_id": c["doc_id"], "version": c["version"]} for c in ans.citations],
            "abstained": ans.abstained, "conflict": ans.conflict}


TOOLS = {"lookup_order": lookup_order, "lookup_customer": lookup_customer, "check_return_eligibility": check_return_eligibility, "calculate_refund": calculate_refund, "search_policy": search_policy}
# All registered tools are read-only/advisory; state-changing actions exist only as ApprovalGate proposals.
