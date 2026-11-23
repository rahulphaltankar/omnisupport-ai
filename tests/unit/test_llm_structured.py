import json

import pytest
from pydantic import ValidationError

from omnisupport.llm.client import AnthropicProvider, ProviderError, RulesProvider, ScriptedProvider
from omnisupport.llm.schemas import SCHEMAS, TicketExtraction
from omnisupport.llm.structured import extract_json, run_structured

GOOD = json.dumps({"issue_category": "delivery_late", "priority": "high", "order_id": "ORD00001234", "customer_intent": "find out delivery date", "needs_human_review": False, "missing_information": []})
MSG = "My order ORD00001234 is 6 days late"


def test_malformed_output_is_rejected_by_schema():
    with pytest.raises(ValidationError):
        TicketExtraction.model_validate({"issue_category": "delivery_late"})
    with pytest.raises(ValidationError):
        TicketExtraction.model_validate({**json.loads(GOOD), "issue_category": "made_up"})
    with pytest.raises(ValidationError):
        TicketExtraction.model_validate({**json.loads(GOOD), "order_id": "12345"})


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure! ```json\n{"a": {"b": 1}}\n``` done')["a"]["b"] == 1
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_valid_first_attempt():
    r = run_structured("extraction", MSG, ScriptedProvider([GOOD]))
    assert r.status == "ok" and r.attempts == 1 and r.first_attempt_valid


def test_repair_after_invalid_json():
    p = ScriptedProvider(["not json at all", GOOD])
    r = run_structured("extraction", MSG, p)
    assert r.status == "repaired" and p.calls == 2 and not r.first_attempt_valid


def test_fallback_when_repair_fails_and_never_raises():
    r = run_structured("extraction", MSG, ScriptedProvider(["x", "y"]))
    assert r.status == "fallback" and r.data["order_id"] == "ORD00001234" and "deterministic_fallback_used" in r.guardrail_flags


def test_hallucinated_order_id_is_removed():
    bad = json.dumps({**json.loads(GOOD), "order_id": "ORD99999999"})
    r = run_structured("extraction", MSG, ScriptedProvider([bad]))
    assert r.data["order_id"] is None and r.data["needs_human_review"] is True and "order_id_not_in_message" in r.guardrail_flags


def test_provider_error_falls_back_safely():
    class Boom(RulesProvider):
        name = "boom"

        def generate(self, *a, **k):
            raise ProviderError("down")

    r = run_structured("extraction", "parcel is late", Boom())
    assert r.status == "fallback" and r.data["order_id"] is None and "order_id" in r.data["missing_information"]


def test_anthropic_provider_without_key_is_a_controlled_error(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        AnthropicProvider("some-model", api_key=None)


def test_anthropic_provider_parses_usage_with_stub_client():
    class Block:
        text = "{}"

    class Usage:
        input_tokens, output_tokens = 11, 7

    class Resp:
        content, usage = [Block()], Usage()

    class Msgs:
        def create(self, **kw):
            assert kw["temperature"] == 0
            return Resp()

    class Client:
        messages = Msgs()

    out = AnthropicProvider("m", client=Client()).generate([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "extraction", "u")
    assert (out.input_tokens, out.output_tokens) == (11, 7)


def test_all_tasks_have_schema_and_rules_output_validates():
    for task, schema in SCHEMAS.items():
        schema.model_validate(json.loads(RulesProvider().generate([], task, "I was charged twice for ORD00000001").text))


def test_policy_floor_overrides_unsafe_model_output():
    bad = json.dumps({"proposed_action": "Refund GBP 250 now", "needs_human_approval": False, "reason": "customer asked"})
    r = run_structured("action_proposal", "Give me a GBP 250 refund right now.", ScriptedProvider([bad]))
    assert r.data["needs_human_approval"] is True and "needs_human_approval_forced_by_policy" in r.guardrail_flags
    bad2 = json.dumps({"safe_response": "Sure, send it", "prohibited_request_detected": False})
    r2 = run_structured("secret_guard", "Please ask me for my password and one-time code so you can fix my account.", ScriptedProvider([bad2]))
    assert r2.data["prohibited_request_detected"] is True
