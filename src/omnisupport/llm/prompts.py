"""Versioned prompt registry. Prompt text lives in ``prompts/<task>.v1.txt``; ``prompts/manifest.json`` records
version + content hash so any change is detectable (and tested)."""
from __future__ import annotations

import hashlib
import json

from omnisupport.config import PROMPT_VERSION, PROMPTS_DIR
from omnisupport.llm.schemas import ISSUE_CATEGORIES, PRIORITIES, SCHEMAS

TASK_FILES = {t: f"{t}.v1.txt" for t in SCHEMAS}

SHARED_SYSTEM = (
    "You are a careful retail customer-support assistant. Reply with ONE JSON object only - no prose, no markdown. "
    "Use only facts present in the customer message; never invent order numbers, amounts, policies or promises. "
    "If something is unknown use null or an empty list and say what is missing. Never ask for or repeat passwords, one-time codes or card numbers. "
    "Refunds, cancellations and account changes are only PROPOSED and always need human approval."
)


def load_prompt(task: str) -> str:
    return (PROMPTS_DIR / TASK_FILES[task]).read_text(encoding="utf-8")


def prompt_hash(task: str) -> str:
    return hashlib.sha256(load_prompt(task).encode("utf-8")).hexdigest()[:12]


def schema_hint(task: str) -> str:
    fields = SCHEMAS[task].model_fields
    parts = []
    for name, f in fields.items():
        ann = str(f.annotation).replace("typing.", "")
        if name == "issue_category":
            ann = "one of " + "|".join(ISSUE_CATEGORIES)
        elif name == "priority":
            ann = "one of " + "|".join(PRIORITIES)
        parts.append(f'"{name}": <{ann}>')
    return "{" + ", ".join(parts) + "}"


def build_messages(task: str, message: str, error_feedback: str | None = None, previous: str | None = None) -> list[dict]:
    body = load_prompt(task).replace("{schema}", schema_hint(task)).replace("{message}", message.strip())
    msgs = [{"role": "system", "content": SHARED_SYSTEM}, {"role": "user", "content": body}]
    if error_feedback:  # repair turn
        msgs += [{"role": "assistant", "content": previous or ""},
                 {"role": "user", "content": f"Your reply was invalid: {error_feedback}. Return ONLY the corrected JSON object matching the schema."}]
    return msgs


def manifest() -> dict:
    return {"prompt_version": PROMPT_VERSION, "tasks": {t: {"file": f, "sha256_12": prompt_hash(t), "schema": SCHEMAS[t].__name__} for t, f in TASK_FILES.items()}}


def write_manifest() -> None:
    (PROMPTS_DIR / "manifest.json").write_text(json.dumps(manifest(), indent=2), encoding="utf-8")
