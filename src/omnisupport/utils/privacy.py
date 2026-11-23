"""Privacy helpers: redaction of free text and pseudonymisation of identifiers for logs."""
from __future__ import annotations

import hashlib
import re

_PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3,5}\)?[ -]?)\d{3,4}[ -]?\d{3,4}(?!\w)")),
    ("SECRET", re.compile(r"(?i)\b(password|passcode|pin|otp|one[- ]time code|api[_ -]?key|secret|token)\b\s*(?:is|:|=)?\s*\S+")),
    ("API_KEY", re.compile(r"\b(?:sk|hf|gh[pousr])[-_][A-Za-z0-9_\-]{16,}\b")),
]


def redact(text: str | None, max_len: int = 200) -> str:
    """Mask emails, card numbers, phones and secrets, then truncate for log safety."""
    if not text:
        return ""
    out = str(text)
    for label, pat in _PATTERNS:
        out = pat.sub(f"[{label}]", out)
    return out if len(out) <= max_len else out[:max_len] + "..."


def pseudonymise(value: str | None, salt: str = "omnisupport") -> str | None:
    """Stable, non-reversible token for synthetic identifiers in logs."""
    if value is None:
        return None
    return "h_" + hashlib.sha256((salt + str(value)).encode()).hexdigest()[:10]


def contains_secret_request(text: str) -> bool:
    """True when the user text asks the assistant to collect or handle credentials."""
    t = text.lower()
    asks = any(k in t for k in ("ask me for my", "ask for my", "collect my", "send you my", "here is my", "my password is", "my pin is"))
    secret = any(k in t for k in ("password", "one-time code", "one time code", "otp", "card number", "cvv", "pin ", "passcode"))
    return secret and (asks or "code" in t and "ask" in t)
