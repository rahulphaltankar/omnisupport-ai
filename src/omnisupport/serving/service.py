"""Service layer shared by the FastAPI app and the Gradio UI. Every component degrades gracefully and never raises raw internals."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from omnisupport.config import AGENT_VERSION, DATA_DIR, INDEX_VERSION, MODEL_VERSION, MODELS, PROMPT_VERSION, ROOT, LLMSettings
from omnisupport.utils.privacy import redact

IMAGES_ROOT = (DATA_DIR / "images").resolve()
ALLOWED_SUFFIX = {".png", ".jpg", ".jpeg"}
_lock = threading.Lock()
_cache: dict = {}


class ServiceUnavailable(RuntimeError):
    pass


class BadInput(ValueError):
    pass


def versions() -> dict:
    return {"project": "1.0.0", "model_version": MODEL_VERSION, "prompt_version": PROMPT_VERSION, "index_version": INDEX_VERSION, "agent_version": AGENT_VERSION,
            "llm": LLMSettings().public_dict()}


def _get(name, factory):
    with _lock:
        if name not in _cache:
            _cache[name] = factory()
        return _cache[name]


# ---------------------------------------------------------------- escalation classifier
def _escalation_bundle():
    import joblib

    meta = json.loads((MODELS / "serving" / "escalation_metadata.json").read_text(encoding="utf-8"))
    return joblib.load(MODELS / "serving" / "escalation_hgb.joblib"), meta


def predict_ticket(payload: dict) -> dict:
    """T0 escalation-risk advice (probability + threshold decision). Advisory: a human decides."""
    import pandas as pd

    from omnisupport.data.features import build_features

    try:
        model, meta = _get("escalation", _escalation_bundle)
    except Exception as e:  # missing file or pickle/sklearn version mismatch -> controlled degradation
        raise ServiceUnavailable(f"escalation model unavailable ({type(e).__name__}); see docs/monitoring_and_operations.md") from None
    row = {"timestamp": payload.get("timestamp") or "2025-12-01T12:00:00", "support_channel": payload["support_channel"], "customer_region": payload["customer_region"],
           "customer_tenure_months": payload["customer_tenure_months"], "product_category": payload["product_category"], "order_value": payload["order_value"],
           "delivery_delay_days": payload["delivery_delay_days"], "previous_ticket_count": payload["previous_ticket_count"], "issue_title": payload["issue_title"],
           "issue_description": payload.get("issue_description"), "attachment_available": payload.get("attachment_available", 0)}
    X = build_features(pd.DataFrame([row]))[meta["features"]]
    p = float(model.predict_proba(X)[0, 1])
    thr = float(meta["decision_threshold"])
    return {"escalation_probability": round(p, 4), "decision_threshold": thr, "flag_for_priority_review": bool(p >= thr), "model_version": meta.get("model_version", MODEL_VERSION),
            "note": "Advisory routing support; a human agent decides. Threshold chosen on validation with FN cost 3x FP.", "base_rate_test": meta.get("test_metrics", {}).get("base_rate")}


# ---------------------------------------------------------------- LLM extraction
def extract_ticket(text: str, task: str = "extraction") -> dict:
    from omnisupport.llm.client import get_provider
    from omnisupport.llm.structured import run_structured

    prov = _get("provider", lambda: _safe_provider())
    r = run_structured(task, text, prov)
    return {"status": r.status, "data": r.data, "attempts": r.attempts, "provider": r.provider, "model": r.model, "prompt_version": r.prompt_version, "latency_ms": round(r.latency_ms, 1),
            "guardrail_flags": r.guardrail_flags, "errors": r.errors[:3]}


def _safe_provider():
    from omnisupport.llm.client import RulesProvider, get_provider

    try:
        return get_provider()
    except Exception:  # missing key / model -> deterministic fallback, never a crash
        return RulesProvider()


# ---------------------------------------------------------------- RAG
def rag_answer(question: str) -> dict:
    from omnisupport.rag.pipeline import PolicyRAG

    rag = _get("rag", lambda: PolicyRAG())
    a = rag.answer(question)
    return {"status": "no_answer" if a.abstained else "answered", "answer": a.answer, "citations": a.citations, "abstained": a.abstained, "conflict": a.conflict, "conflict_detail": a.conflict_detail,
            "needs_human_review": a.needs_human_review, "top_score": round(a.top_score, 3), "index_version": a.index_version, "retrieval_backend": rag.backend_name, "latency_ms": round(a.latency_ms, 1)}


# ---------------------------------------------------------------- agent
def agent_run(request: str) -> dict:
    from omnisupport.agent.core import SupportAgent

    ag = _get("agent", lambda: SupportAgent())
    r = ag.run(request)
    return {"status": r.status, "reply": r.reply, "tools_called": r.tools_called, "approval_required": r.approval_required, "approval": r.approval, "missing_info": r.missing_info,
            "trace": r.trace, "tool_failures": r.tool_failures, "agent_version": r.agent_version, "run_id": r.run_id}


def agent_approve(action_id: str, approver: str, approve: bool, note: str = "") -> dict:
    from omnisupport.agent.core import SupportAgent

    ag = _get("agent", lambda: SupportAgent())
    return ag.gate.decide(action_id, approver, approve, note)


# ---------------------------------------------------------------- vision
def resolve_image(image_path: str) -> Path:
    """Only files inside data/images with an image suffix are readable (blocks path traversal / arbitrary file reads)."""
    p = Path(image_path)
    p = (ROOT / p if not p.is_absolute() else p).resolve()
    if IMAGES_ROOT not in p.parents:
        raise BadInput("image_path must be inside data/images")
    if p.suffix.lower() not in ALLOWED_SUFFIX:
        raise BadInput("unsupported image type")
    if not p.is_file():
        raise FileNotFoundError("image not found")
    return p


def _vision_model():
    import torch

    from omnisupport.vision import core as V

    vm = MODELS / "vision"
    meta = json.loads((vm / "metadata.json").read_text(encoding="utf-8"))
    net = V.BasicCNN(len(V.CLASSES))
    sd = torch.load(vm / "basic_cnn.pt", map_location="cpu")
    net.load_state_dict({k.replace("net.", "", 1) if k.startswith("net.") else k: v for k, v in sd.items() if k not in ("mean", "std")}, strict=False)
    return net.eval(), meta


def vision_predict(image_path: str) -> dict:
    p = resolve_image(image_path)           # validate the path first: invalid input must be a 4xx even when torch is not installed
    try:
        import torch

        from omnisupport.vision import core as V
        net, meta = _get("vision", _vision_model)
    except Exception as e:
        raise ServiceUnavailable(f"vision model unavailable ({type(e).__name__})") from None
    x = V.load_tensor([p.relative_to(ROOT)], root=ROOT)
    with torch.no_grad():
        prob = torch.softmax(net(x), 1)[0].numpy()
    i = int(np.argmax(prob))
    conf = float(prob[i])
    return {"predicted_class": V.CLASSES[i], "confidence": round(conf, 4), "probabilities": {c: round(float(q), 4) for c, q in zip(V.CLASSES, prob)}, "low_confidence": conf < 0.6,
            "model": "basic_cnn (served for latency; transfer model compared in reports/vision_report.md)", "model_version": meta.get("model_version", MODEL_VERSION),
            "note": "Advisory triage of a synthetic return image; low confidence should go to a human."}


def safe_detail(e: Exception) -> str:
    return redact(str(e), 160)
