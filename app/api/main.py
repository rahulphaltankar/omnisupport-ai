"""OmniSupport AI - FastAPI service.

Endpoints: /health, /tickets/predict, /tickets/extract, /rag/ask, /agent/run, /agent/approve, /vision/predict, /monitoring/summary.
Components degrade gracefully (503 with a controlled message), never leak stack traces, and log only privacy-safe fields.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from omnisupport.llm.schemas import ISSUE_CATEGORIES  # noqa: E402
from omnisupport.serving import service as S  # noqa: E402
from omnisupport.serving.monitoring import MONITOR  # noqa: E402
from omnisupport.utils.logging_utils import get_logger  # noqa: E402
from omnisupport.utils.privacy import pseudonymise  # noqa: E402

app = FastAPI(title="OmniSupport AI", version="1.0.0")
log = get_logger("omnisupport.api")


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class RAGRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AgentRequest(BaseModel):
    request: str = Field(min_length=3, max_length=2000)


class ApproveRequest(BaseModel):
    action_id: str = Field(min_length=4, max_length=40)
    approver: str = Field(min_length=2, max_length=80)
    approve: bool
    note: str = Field(default="", max_length=300)


class VisionRequest(BaseModel):
    image_path: str = Field(min_length=1, max_length=500)


class TicketFeatures(BaseModel):
    support_channel: str = Field(min_length=2, max_length=40)
    customer_region: str = Field(min_length=2, max_length=60)
    customer_tenure_months: float = Field(ge=0, le=600)
    product_category: str = Field(min_length=2, max_length=60)
    order_value: float = Field(ge=0, le=100000)
    delivery_delay_days: float = Field(ge=-30, le=365)
    previous_ticket_count: int = Field(ge=0, le=500)
    issue_title: str = Field(min_length=2, max_length=120)
    issue_description: str | None = Field(default=None, max_length=5000)
    attachment_available: int = Field(default=0, ge=0, le=1)
    timestamp: str | None = None


def _run(endpoint: str, fn, *, rag=False, agent=False, extra=None):
    t0 = time.perf_counter()
    ok, out, rag_no, tf = True, None, None, 0
    try:
        out = fn()
        if rag:
            rag_no = bool(out.get("abstained"))
        if agent:
            tf = int(out.get("tool_failures", 0))
        return out
    except S.ServiceUnavailable as e:
        ok = False
        raise HTTPException(status_code=503, detail=str(e))
    except FileNotFoundError:
        ok = False
        raise HTTPException(status_code=404, detail="resource not found")
    except (S.BadInput, ValueError) as e:
        ok = False
        raise HTTPException(status_code=400, detail=S.safe_detail(e))
    except HTTPException:
        ok = False
        raise
    except Exception as e:  # never leak internals
        ok = False
        log.error("unhandled error", extra={"fields": {"endpoint": endpoint, "error_type": type(e).__name__}})
        raise HTTPException(status_code=500, detail="internal error")
    finally:
        ms = (time.perf_counter() - t0) * 1000
        MONITOR.record(endpoint, ms, ok, rag_no_answer=rag_no, agent_tool_failures=tf, extra=extra)


@app.get("/health")
def health():
    return {"status": "ok", "message": "OmniSupport AI API is running.", **S.versions()}


@app.post("/tickets/predict")
def tickets_predict(payload: TicketFeatures):
    return _run("/tickets/predict", lambda: S.predict_ticket(payload.model_dump()))


@app.post("/tickets/extract")
def tickets_extract(payload: TextRequest):
    return _run("/tickets/extract", lambda: S.extract_ticket(payload.text), extra={"text_len": len(payload.text)})


@app.post("/rag/ask")
def rag_ask(payload: RAGRequest):
    return _run("/rag/ask", lambda: S.rag_answer(payload.question), rag=True, extra={"q_len": len(payload.question)})


@app.post("/agent/run")
def agent_run(payload: AgentRequest):
    return _run("/agent/run", lambda: S.agent_run(payload.request), agent=True, extra={"req_len": len(payload.request)})


@app.post("/agent/approve")
def agent_approve(payload: ApproveRequest):
    def go():
        r = S.agent_approve(payload.action_id, payload.approver, payload.approve, payload.note)
        if r.get("status") == "unknown_action":
            raise FileNotFoundError("unknown action")
        return r
    return _run("/agent/approve", go, extra={"approver": pseudonymise(payload.approver)})


@app.post("/vision/predict")
def vision_predict(payload: VisionRequest):
    return _run("/vision/predict", lambda: S.vision_predict(payload.image_path))


@app.get("/monitoring/summary")
def monitoring_summary():
    from dataclasses import asdict
    return {**asdict(MONITOR.summary()), "by_endpoint": MONITOR.endpoint_table()}


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):  # last-resort handler: no stack trace to clients
    return JSONResponse(status_code=500, content={"detail": "internal error"})


@app.get("/meta/categories")
def categories():
    return {"issue_categories": ISSUE_CATEGORIES}
