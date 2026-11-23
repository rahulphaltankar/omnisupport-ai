"""OmniSupport AI - Gradio demo UI (calls the same service layer as the API).

Run:  python app/ui/app.py        (needs `pip install -r requirements-serve.txt`)
Never displays raw stack traces; shows citations, abstentions, approval state and agent traces.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import gradio as gr  # noqa: E402

from omnisupport.serving import service as S  # noqa: E402
from omnisupport.serving.monitoring import MONITOR  # noqa: E402

EXAMPLES_RAG = ["A customer says a card refund was approved yesterday but is not visible yet. What timing should the assistant explain?",
                "Can an agent automatically issue a GBP 180 refund if the customer is clearly eligible?",
                "What is the internal policy for teleporting an order to Mars?"]
EXAMPLES_AGENT = ["Check order ORD00001001 and explain the delivery status.", "Issue a GBP 240 refund immediately for order ORD00001003.", "Is this item still eligible for return?",
                  "Change my email address and reset my password.", "I smell smoke from the product and it is very hot."]


def _timed(endpoint, fn, **kw):
    import time
    t0 = time.perf_counter(); ok = True
    try:
        return fn()
    except Exception as e:
        ok = False
        return {"error": S.safe_detail(e)}
    finally:
        MONITOR.record(endpoint, (time.perf_counter() - t0) * 1000, ok, **kw)


def ui_extract(text):
    out = _timed("ui:/tickets/extract", lambda: S.extract_ticket(text))
    return json.dumps(out, indent=2)


def ui_rag(q):
    out = _timed("ui:/rag/ask", lambda: S.rag_answer(q))
    if "error" in out:
        return out["error"], ""
    cites = "\n".join(f"- **{c['doc_id']}** v{c['version']}: {c['quote']}" for c in out["citations"]) or "_no citations (no sufficient evidence)_"
    flags = f"status: **{out['status']}** | top score {out['top_score']} | conflict: {out['conflict']} | human review: {out['needs_human_review']} | index {out['index_version']}"
    return out["answer"] + "\n\n" + flags, cites


def ui_agent(req):
    out = _timed("ui:/agent/run", lambda: S.agent_run(req), agent_tool_failures=0)
    if "error" in out:
        return out["error"], "", ""
    appr = json.dumps(out["approval"], indent=2) if out["approval"] else "none required"
    trace = "\n".join(f"{t['step']}. [{t['kind']}] {t.get('tool') or ''} {t['status']} {t['detail']}" for t in out["trace"])
    return f"**{out['status']}** - {out['reply']}", appr, trace


def ui_approve(aid, who, decision):
    return json.dumps(_timed("ui:/agent/approve", lambda: S.agent_approve(aid.strip(), who.strip(), decision == "approve")), indent=2)


def ui_monitor():
    from dataclasses import asdict
    return json.dumps({**asdict(MONITOR.summary()), "by_endpoint": MONITOR.endpoint_table()}, indent=2)


with gr.Blocks(title="OmniSupport AI") as demo:
    gr.Markdown("# OmniSupport AI\nSynthetic retail support prototype. **Advisory only** - refunds, cancellations and account changes are proposals that need a named human approver.")
    with gr.Tab("Policy Q&A (RAG)"):
        q = gr.Textbox(label="Question", lines=2); gr.Examples(EXAMPLES_RAG, q)
        a, c = gr.Markdown(label="Answer"), gr.Markdown(label="Sources")
        gr.Button("Ask").click(ui_rag, q, [a, c])
    with gr.Tab("Agent"):
        r = gr.Textbox(label="Request", lines=2); gr.Examples(EXAMPLES_AGENT, r)
        ans, ap, tr = gr.Markdown(), gr.Code(label="Approval proposal", language="json"), gr.Textbox(label="Trace", lines=6)
        gr.Button("Run agent").click(ui_agent, r, [ans, ap, tr])
        gr.Markdown("### Human approval (simulated - nothing is executed)")
        aid, who, dec = gr.Textbox(label="Action id"), gr.Textbox(label="Approver name"), gr.Radio(["approve", "reject"], value="reject", label="Decision")
        res = gr.Code(language="json"); gr.Button("Record decision").click(ui_approve, [aid, who, dec], res)
    with gr.Tab("Ticket extraction"):
        t = gr.Textbox(label="Customer message", lines=3, value="My order ORD00001234 is 6 days late and I need it for work. Please call me back.")
        o = gr.Code(language="json"); gr.Button("Extract").click(ui_extract, t, o)
    with gr.Tab("Monitoring"):
        m = gr.Code(language="json"); gr.Button("Refresh").click(ui_monitor, None, m)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
