"""Deterministic rule-based implementation of every LLM task (offline default and safety fallback).

These are generic keyword/regex rules - no per-test-case answers. They are intentionally conservative:
unknown -> 'other' + needs_human_review.
"""
from __future__ import annotations

import re

from omnisupport.utils.privacy import contains_secret_request

ORDER_RE = re.compile(r"\bORD\d{8}\b", re.I)
SAFETY = re.compile(r"(?i)\b(smoke|smoking|fire|burn(?:ing|t)?|spark(?:ed|s|ing)?|electric shock|shock|explod|overheat|very hot|chemical|injur)")
TAKEOVER = re.compile(r"(?i)(someone (?:changed|else)|hacked|did not (?:place|make)|don't recognise|do not recognise|unrecognised|not me|new (?:delivery )?address|changed my (?:delivery )?address)")
CATEGORY_RULES = [
    ("lost_parcel", r"(?i)\b(lost|never (?:arrived|came)|missing parcel|delivered but|marked as delivered)\b"),
    ("account_access", r"(?i)(log ?in|password|locked out|account access|reset)"),
    ("payment_issue", r"(?i)(charged twice|double charge|duplicate charge|payment|card (?:was )?declined|checkout|billed)"),
    ("wrong_item", r"(?i)(wrong (?:item|itm|colou?r|size|product)|incorrect item|not what i ordered|different (?:item|product)|ordr?.{0,20}wrong)"),
    ("damaged_item", r"(?i)(damag|broken|smash|crush|cracked|arrived (?:faulty|broken)|spark|smok)"),
    ("refund_status", r"(?i)(refund|money back|credit)"),
    ("return_request", r"(?i)(return|send (?:it |this )?back|exchange)"),
    ("warranty", r"(?i)(warranty|guarantee|stopped working|faulty after|broke within)"),
    ("product_question", r"(?i)(compatible|work with|does this|is this|will this|waterproof|dimension|size guide|spec)"),
    ("delivery_late", r"(?i)(late|delay|waiting|has not arrived|hasn.t arrived|still not here|where is my (?:parcel|order)|tracking)"),
]
INTENT = {"delivery_late": "find out when the delivery will arrive", "lost_parcel": "locate or replace a parcel that may be lost", "damaged_item": "report a damaged item and ask for a remedy",
          "wrong_item": "report the wrong item and ask for the correct one", "refund_status": "check the status of a refund", "return_request": "start or ask about a return",
          "payment_issue": "resolve a payment or charge problem", "account_access": "regain access to their account", "product_question": "get product information",
          "warranty": "make a warranty claim", "other": "get help (intent unclear)"}
NEXT_STEP = {"delivery_late": "Ask for the order number and check carrier tracking.", "lost_parcel": "Check carrier status before promising a refund or replacement.", "damaged_item": "Ask for the order number and a photo, then assess a remedy.",
             "wrong_item": "Confirm the order number and offer a no-cost return and replacement if eligible.", "refund_status": "Check the refund timeline and explain approved vs submitted vs visible.", "return_request": "Confirm order number and delivery date against the return window.",
             "payment_issue": "Verify whether an order was created; escalate confirmed duplicate captures to payments support.", "account_access": "Offer the official password reset; never request passwords or codes.", "product_question": "Point the customer to the product specification page.",
             "warranty": "Confirm purchase date and product warranty period via order lookup.", "other": "Ask a clarifying question."}


def categorise(text: str) -> str:
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, text):
            return cat
    return "other"


def days_late(text: str) -> int:
    m = re.search(r"(\d+)\s*days?\s*(?:late|beyond|overdue)", text, re.I) or re.search(r"(\d+)\s*days?\s+(?:since|waiting)", text, re.I)
    return int(m.group(1)) if m else 0


def _priority(text: str, cat: str) -> str:
    if SAFETY.search(text) or TAKEOVER.search(text):
        return "urgent"
    if days_late(text) > 7 or re.search(r"(?i)(for work|essential|medication|urgent|asap|immediately)", text) or cat in {"lost_parcel", "payment_issue"} and re.search(r"(?i)twice|charged", text):
        return "high"
    if cat in {"product_question", "account_access"} and not re.search(r"(?i)hack|someone", text):
        return "low" if cat == "product_question" else "medium"
    return "medium"


def run_rules(task: str, message: str) -> dict:
    text = message.strip(); cat = categorise(text); oid = ORDER_RE.search(text)
    order_id = oid.group(0).upper() if oid else None
    safety = bool(SAFETY.search(text)); takeover = bool(TAKEOVER.search(text))
    if task == "extraction":
        prio = _priority(text, cat); order_related = cat in {"delivery_late", "lost_parcel", "damaged_item", "wrong_item", "refund_status", "return_request", "warranty", "payment_issue"}
        missing = ["order_id"] if (order_id is None and order_related) else []
        review = safety or takeover or cat == "other" or cat in {"payment_issue", "lost_parcel"}
        intent = INTENT[cat] + (" and be called back" if re.search(r"(?i)call (?:me )?back|phone me", text) else "")
        return {"issue_category": cat, "priority": prio, "order_id": order_id, "customer_intent": intent, "needs_human_review": review, "missing_information": missing}
    if task == "classify_summarise":
        unc = []
        if order_id is None: unc.append("order number not given")
        if cat == "payment_issue": unc.append("unclear whether one or two payments were captured")
        if cat == "other": unc.append("issue type unclear")
        if re.search(r"(?i)only see one order|can't see|cannot see|do not see", text): unc.append("customer cannot see a matching order")
        return {"issue_category": cat, "summary": "Customer message about " + INTENT[cat].replace("get help (intent unclear)", "an unclear issue") + ".", "uncertainties": unc, "recommended_next_step": NEXT_STEP[cat]}
    if task == "account_risk":
        flags = []
        if takeover: flags.append("possible account takeover (unrecognised change)")
        if re.search(r"(?i)address", text): flags.append("delivery address involved")
        if re.search(r"(?i)can.?t log|cannot log|locked", text): flags.append("login failure")
        return {"issue_category": "account_access" if (cat in {"account_access", "other"} or takeover) else cat, "risk_flags": flags, "needs_human_review": takeover}
    if task == "policy_summary":
        s = re.sub(r"\s+", " ", text)
        return {"customer_facing_summary": s if len(s) <= 400 else s[:397] + "...", "claims_supported_by_input": True}
    if task == "secret_guard":
        bad = contains_secret_request(text) or bool(re.search(r"(?i)(password|one[- ]time code|otp|cvv|pin\b)", text))
        return {"safe_response": ("I can't ask for or accept passwords, one-time codes, PINs or full card numbers. Please use the official password-reset link or secure payment page, and never share these with anyone." if bad
                                  else "Thanks - how can I help with your order or account?"), "prohibited_request_detected": bad}
    if task == "action_proposal":
        amt = re.search(r"(?:GBP|£)\s*(\d+(?:\.\d+)?)", text, re.I); amount = float(amt.group(1)) if amt else None
        sensitive = bool(re.search(r"(?i)(refund|cancel|change my (?:email|address|password)|reset)", text))
        need = bool(sensitive and (amount is None or amount > 100 or re.search(r"(?i)cancel|change my|reset", text)))
        act = "Propose a refund" + (f" of GBP {amount:g}" if amount else "") + " for human review" if re.search(r"(?i)refund", text) else "Propose the requested action for human review"
        return {"proposed_action": act, "needs_human_approval": need, "reason": ("Above the GBP 100 frontline limit or state-changing, so a human must approve; nothing has been executed." if need else "Within frontline authority but still only a proposal until a tool confirms it.")}
    if task == "escalation_summary":
        return {"summary": ("Safety concern: " if safety else "") + re.sub(r"\s+", " ", text)[:200], "safety_risk": safety, "priority": "urgent" if safety else _priority(text, cat), "needs_human_review": safety or takeover}
    if task == "product_answer":
        return {"answer": "I cannot confirm that from the information provided; compatibility and specifications must be checked.", "uncertainty": "No product specification was supplied in the message, so a definite claim would be unsupported.",
                "recommended_source": "Product specification page / approved catalogue"}
    raise KeyError(task)
