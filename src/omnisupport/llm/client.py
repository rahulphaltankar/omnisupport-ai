"""LLM provider interface: rules (offline) | local_hf (transformers) | anthropic. Keys come from the environment only."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from omnisupport.config import LLMSettings
from omnisupport.llm.rules import run_rules


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    streaming: bool = False
    meta: dict = field(default_factory=dict)


class ProviderError(RuntimeError):
    pass


class BaseProvider:
    name = "base"
    model = ""

    def generate(self, messages: list[dict], task: str, raw_message: str) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


class RulesProvider(BaseProvider):
    """Deterministic provider: returns JSON produced by generic rules; zero tokens, ~zero cost."""
    name, model = "rules", "rules-v1"

    def generate(self, messages, task, raw_message):
        t0 = time.perf_counter()
        return LLMResponse(json.dumps(run_rules(task, raw_message)), 0, 0, (time.perf_counter() - t0) * 1000, self.model)


class LocalHFProvider(BaseProvider):
    """Real local inference with a Hugging Face instruct model (greedy decoding, CPU by default)."""
    name = "local_hf"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens: int = 200):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.model = torch, model_name
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()

    def generate(self, messages, task, raw_message):
        torch = self.torch
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.lm.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
        dt = (time.perf_counter() - t0) * 1000
        gen = out[0][enc["input_ids"].shape[1]:]
        return LLMResponse(self.tok.decode(gen, skip_special_tokens=True), int(enc["input_ids"].shape[1]), int(len(gen)), dt, self.model, False, {"context_tokens": int(enc["input_ids"].shape[1])})


class AnthropicProvider(BaseProvider):
    """Anthropic Messages API. Requires LLM_API_KEY/ANTHROPIC_API_KEY. Not exercised live in this repository's evidence
    (no key was available); covered by a stub-client unit test of request/usage handling."""
    name = "anthropic"

    def __init__(self, model_name: str, api_key: str | None = None, client=None, timeout_s: float = 60.0, max_tokens: int = 300):
        self.model, self.max_tokens, self.timeout_s = model_name, max_tokens, timeout_s
        if client is not None:
            self.client = client
        else:
            key = api_key or LLMSettings().api_key
            if not key:
                raise ProviderError("no API key configured (set LLM_API_KEY in the environment)")
            import anthropic

            self.client = anthropic.Anthropic(api_key=key, timeout=timeout_s)

    def generate(self, messages, task, raw_message):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        t0 = time.perf_counter()
        try:
            r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system, messages=chat, temperature=0)
        except Exception as e:  # network/auth/rate-limit: surface as a controlled error without leaking details/keys
            raise ProviderError(f"anthropic request failed: {type(e).__name__}") from None
        dt = (time.perf_counter() - t0) * 1000
        text = "".join(getattr(b, "text", "") for b in r.content)
        u = getattr(r, "usage", None)
        return LLMResponse(text, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0), dt, self.model)


class OpenAICompatProvider(BaseProvider):
    """Any OpenAI-compatible /chat/completions endpoint (Mistral, Groq, OpenRouter, DeepSeek, Gemini's compat API, ...).
    Configure with LLM_BASE_URL, LLM_API_KEY and MODEL_NAME in the environment. Uses only the standard library.
    Not exercised live in this repository's evidence (no key was available); covered by a stubbed-transport unit test."""
    name = "openai_compat"

    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None, transport=None, timeout_s: float = 60.0, max_tokens: int = 300):
        self.model, self.timeout_s, self.max_tokens = model_name, timeout_s, max_tokens
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self._key = (api_key or LLMSettings().api_key or "").strip().strip('"').strip("'") or None
        self._transport = transport
        if not self.base_url or not model_name:
            raise ProviderError("set LLM_BASE_URL and MODEL_NAME for the openai_compat provider")
        if transport is None and not self._key:
            raise ProviderError("no API key configured (set LLM_API_KEY in the environment)")

    def _post(self, payload: dict) -> dict:
        if self._transport is not None:
            return self._transport(payload)
        import json as _json
        import urllib.request

        req = urllib.request.Request(self.base_url + "/chat/completions", data=_json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:  # nosec - URL comes from operator config
            return _json.loads(r.read().decode())

    def generate(self, messages, task, raw_message):
        t0 = time.perf_counter()
        try:
            r = self._post({"model": self.model, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens})
            text = r["choices"][0]["message"]["content"] or ""
        except Exception as e:  # never leak the key; the HTTP status and the provider's short error text are safe and useful
            code = getattr(e, "code", None)
            detail = ""
            if hasattr(e, "read"):
                try:
                    detail = e.read().decode("utf-8", "replace")[:160]
                except Exception:
                    detail = ""
            self.last_error = f"HTTP {code} {detail}".strip() if code else type(e).__name__
            raise ProviderError(f"openai_compat request failed: {self.last_error}") from None
        u = r.get("usage") or {}
        return LLMResponse(text, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), (time.perf_counter() - t0) * 1000, self.model)


class ScriptedProvider(BaseProvider):
    """Test double: returns queued strings (malformed JSON, refusals, ...) to exercise validation/repair paths."""
    name, model = "scripted", "scripted"

    def __init__(self, outputs: list[str]):
        self.outputs, self.calls = list(outputs), 0

    def generate(self, messages, task, raw_message):
        self.calls += 1
        text = self.outputs.pop(0) if self.outputs else ""
        return LLMResponse(text, 50, 20, 1.0, self.model)


def get_provider(settings: LLMSettings | None = None) -> BaseProvider:
    s = settings or LLMSettings()
    if s.provider == "anthropic":
        return AnthropicProvider(s.model_name, timeout_s=s.timeout_s)
    if s.provider == "openai_compat":
        return OpenAICompatProvider(s.model_name, timeout_s=s.timeout_s)
    if s.provider == "local_hf":
        return LocalHFProvider(s.model_name or "Qwen/Qwen2.5-0.5B-Instruct", s.max_new_tokens)
    return RulesProvider()


def estimate_cost_usd(resp: LLMResponse, provider: str) -> float:
    """Local models: compute cost = wall seconds x assumed $/h (env COMPUTE_USD_PER_HOUR, default 0.34 ~ 8 vCPU general-purpose VM; ASSUMPTION).
    API models: tokens x env LLM_PRICE_IN_PER_MTOK / LLM_PRICE_OUT_PER_MTOK (no default - unknown prices are never invented)."""
    if provider == "local_hf":
        return resp.latency_ms / 1000 / 3600 * float(os.getenv("COMPUTE_USD_PER_HOUR", "0.34"))
    if provider in ("anthropic", "openai_compat"):
        pin, pout = os.getenv("LLM_PRICE_IN_PER_MTOK"), os.getenv("LLM_PRICE_OUT_PER_MTOK")
        if pin and pout:
            return resp.input_tokens / 1e6 * float(pin) + resp.output_tokens / 1e6 * float(pout)
        return float("nan")
    return 0.0
