import json

from omnisupport.config import EVAL_DIR, PROMPTS_DIR
from omnisupport.llm.prompts import build_messages, manifest
from omnisupport.llm.rules import run_rules

CASES = json.loads((EVAL_DIR / "prompt_test_cases.json").read_text(encoding="utf-8"))
TMAP = json.loads((PROMPTS_DIR / "case_task_map.json").read_text())


def test_prompt_manifest_matches_files():
    """Any prompt edit without a version bump + manifest refresh fails here (prompt regression guard)."""
    assert json.loads((PROMPTS_DIR / "manifest.json").read_text()) == manifest()


def test_every_case_has_required_fields_via_rules():
    for c in CASES:
        out = run_rules(TMAP[c["id"]], c["input"])
        want = c["required_fields"] if c["id"] != "P09" else ["issue_category", "order_id"]
        assert all(f in out for f in want), c["id"]


def test_behavioural_assertions():
    by = {c["id"]: c for c in CASES}
    assert run_rules("extraction", by["P01"]["input"])["order_id"] == "ORD00001234"
    p5 = run_rules("extraction", by["P05"]["input"])
    assert p5["order_id"] is None and "order_id" in p5["missing_information"]
    assert run_rules("secret_guard", by["P06"]["input"])["prohibited_request_detected"] is True
    assert run_rules("action_proposal", by["P07"]["input"])["needs_human_approval"] is True
    assert run_rules("escalation_summary", by["P08"]["input"])["safety_risk"] is True
    assert run_rules("extraction", by["P09"]["input"])["order_id"] == "ORD00077777"
    assert "cannot" in run_rules("product_answer", by["P10"]["input"])["answer"].lower()
    assert run_rules("account_risk", by["P03"]["input"])["needs_human_review"] is True


def test_prompts_include_schema_and_untrusted_message_once():
    m = build_messages("extraction", "hello ORD00000001")
    assert "{schema}" not in m[1]["content"] and "{message}" not in m[1]["content"] and m[1]["content"].count("hello ORD00000001") == 1
    assert "never invent" in m[0]["content"].lower()
