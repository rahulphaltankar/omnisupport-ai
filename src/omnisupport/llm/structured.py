"""Structured-output runner: generate -> parse -> validate (Pydantic + guardrails) -> one repair attempt -> deterministic fallback."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from omnisupport.config import PROMPT_VERSION
from omnisupport.llm.client import BaseProvider, ProviderError, RulesProvider
from omnisupport.llm.prompts import build_messages
from omnisupport.llm.rules import ORDER_RE, run_rules
from omnisupport.llm.schemas import SCHEMAS
from omnisupport.utils.privacy import redact


@dataclass
class StructuredResult:
    task: str
    status: str                         # ok | repaired | fallback | failed
    data: dict | None
    attempts: int
    latency_ms: float
    input_tokens: int
    output_tokens: int
    provider: str
    model: str
    prompt_version: str = PROMPT_VERSION
    errors: list[str] = field(default_factory=list)
    first_attempt_valid: bool = False
    guardrail_flags: list[str] = field(default_factory=list)


def extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of a model reply (tolerates ```json fences and surrounding prose)."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object")


def _guardrails(task: str, data: dict, message: str) -> list[str]:
    """Post-validation checks that catch confident-but-invented content."""
    flags = []
    if task == "extraction":
        oid = data.get("order_id")
        if oid and oid.upper() not in {m.upper() for m in ORDER_RE.findall(message)}:
            flags.append("order_id_not_in_message")
            data["order_id"] = None
            data["needs_human_review"] = True
            mi = list(data.get("missing_information") or [])
            if "order_id" not in mi: mi.append("order_id")
            data["missing_information"] = mi
    # Safety invariants enforced in code, never left to the model: a model may only make these *stricter*.
    if task in ("action_proposal", "secret_guard", "escalation_summary", "account_risk"):
        floor = run_rules(task, message)
        keys = {"action_proposal": "needs_human_approval", "secret_guard": "prohibited_request_detected", "escalation_summary": "safety_risk", "account_risk": "needs_human_review"}
        k = keys[task]
        if floor.get(k) is True and data.get(k) is not True:
            data[k] = True
            flags.append(f"{k}_forced_by_policy")
    return flags


def run_structured(task: str, message: str, provider: BaseProvider, *, max_repairs: int = 1, fallback: bool = True) -> StructuredResult:
    schema = SCHEMAS[task]
    t_all = time.perf_counter(); tin = tout = 0; errors: list[str] = []; attempts = 0; first_ok = False
    messages = build_messages(task, message); prev = None; fb = None
    for attempt in range(max_repairs + 1):
        attempts += 1
        try:
            resp = provider.generate(messages, task, message)
        except ProviderError as e:
            errors.append(f"provider_error: {e}")
            break
        tin += resp.input_tokens; tout += resp.output_tokens; prev = resp.text
        try:
            obj = schema.model_validate(extract_json(resp.text)).model_dump()
            flags = _guardrails(task, obj, message)
            if attempt == 0: first_ok = True
            return StructuredResult(task, "ok" if attempt == 0 else "repaired", obj, attempts, (time.perf_counter() - t_all) * 1000, tin, tout, provider.name, provider.model, errors=errors,
                                    first_attempt_valid=first_ok, guardrail_flags=flags)
        except (ValueError, ValidationError) as e:
            fb = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()) if isinstance(e, ValidationError) else str(e)
            errors.append(redact(fb, 160))
            messages = build_messages(task, message, error_feedback=fb[:300], previous=(prev or "")[:600])
    if fallback:
        obj = schema.model_validate(extract_json(RulesProvider().generate([], task, message).text)).model_dump()
        flags = _guardrails(task, obj, message) + ["deterministic_fallback_used"]
        return StructuredResult(task, "fallback", obj, attempts, (time.perf_counter() - t_all) * 1000, tin, tout, provider.name, provider.model, errors=errors, first_attempt_valid=False, guardrail_flags=flags)
    return StructuredResult(task, "failed", None, attempts, (time.perf_counter() - t_all) * 1000, tin, tout, provider.name, provider.model, errors=errors)
