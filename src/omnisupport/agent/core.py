"""Deterministic, auditable support agent. A rules planner chooses tools; every call is validated, traced and error-handled.
State-changing actions (refund, cancel, account change) are only ever *proposed*; the ApprovalGate holds them until a human decides."""
from __future__ import annotations

import csv
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from pydantic import ValidationError

from omnisupport.agent import tools as T
from omnisupport.config import AGENT_STORE, AGENT_VERSION, FRONTLINE_REFUND_LIMIT_GBP, LOG_DIR
from omnisupport.llm.rules import ORDER_RE, SAFETY
from omnisupport.utils.privacy import redact

INJ = re.compile(r"(?i)(ignore (?:all |any |the )?(?:previous|prior|above|your) (?:instructions|rules|policy)|bypass|override (?:the )?(?:approval|policy|limit)|without (?:approval|asking))")
ACCOUNT_CHANGE = re.compile(r"(?i)(change|update|reset).{0,30}(email|password|address|phone)")
CANCEL = re.compile(r"(?i)\bcancel")
REFUND = re.compile(r"(?i)(refund|money back|reimburse)")
AMOUNT = re.compile(r"(?i)(?:GBP|£)\s*(\d+(?:\.\d+)?)")
ELIG = re.compile(r"(?i)(eligible|eligibility|can i return|still.*return|return window)")
STATUS = re.compile(r"(?i)(status|where is|delivery|track|arrive|check (?:order|my order))")
POLICY = re.compile(r"(?i)(policy|what does the policy|rules? (?:say|on))")


@dataclass
class TraceStep:
    step: int
    kind: str                    # plan | tool | guard | approval | final
    tool: str | None = None
    args: dict | None = None
    status: str = "ok"
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class AgentResult:
    request: str
    status: str                  # answered | needs_info | awaiting_approval | escalated | error
    reply: str
    tools_called: list[str]
    trace: list[dict]
    approval_required: bool = False
    approval: dict | None = None
    missing_info: list[str] = field(default_factory=list)
    agent_version: str = AGENT_VERSION
    tool_failures: int = 0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class ApprovalGate:
    """Holds proposed actions. Nothing executes: 'approved' means a human authorised a downstream process, which this prototype does not perform."""

    def __init__(self, ledger_path=None):
        self.pending: dict[str, dict] = {}
        self.ledger_path = ledger_path or (LOG_DIR / "approvals.jsonl")

    def propose(self, action: str, params: dict, reason: str, route: str) -> dict:
        aid = "ACT-" + uuid.uuid4().hex[:8]
        rec = {"action_id": aid, "action": action, "params": params, "reason": reason, "route": route, "status": "pending_human_approval",
               "created": datetime.now(timezone.utc).isoformat(), "executed": False}
        self.pending[aid] = rec
        self._log(rec)
        return rec

    def decide(self, action_id: str, approver: str, approve: bool, note: str = "") -> dict:
        rec = self.pending.get(action_id)
        if rec is None:
            return {"status": "unknown_action"}
        if rec["status"] != "pending_human_approval":
            return {"status": "already_decided", "action": rec}
        if not approver or approver.strip().lower() in {"agent", "system", "auto", "llm", "assistant"}:
            return {"status": "rejected", "reason": "a named human approver is required"}
        rec.update(status="approved_for_downstream" if approve else "rejected_by_human", approver=approver, note=redact(note, 120),
                   decided=datetime.now(timezone.utc).isoformat(), executed=False)
        self._log(rec)
        return rec

    def _log(self, rec):
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass


class SupportAgent:
    def __init__(self, tools: dict | None = None, gate: ApprovalGate | None = None, max_retries: int = 1):
        self.tools = tools or T.TOOLS
        self.gate = gate or ApprovalGate()
        self.max_retries = max_retries

    def _call(self, name: str, trace: list, counters: dict, **kw):
        step = TraceStep(len(trace) + 1, "tool", name, dict(kw))
        t0 = time.perf_counter()
        if name not in self.tools:
            step.status, step.detail = "blocked", "tool not registered"
            trace.append(step)
            return {"status": "error", "message": "unknown tool"}
        res = {"status": "tool_error"}
        for attempt in range(self.max_retries + 1):
            try:
                res = self.tools[name](**kw)
                step.status, step.detail = res.get("status", "ok"), f"attempt {attempt + 1}"
                break
            except (ValidationError, ValueError) as e:       # bad arguments: do not retry
                res = {"status": "invalid_arguments", "message": redact(str(e).split("\n")[0], 120)}
                step.status, step.detail = "invalid_arguments", res["message"]
                break
            except Exception as e:                           # transient tool failure: retry once, then degrade
                counters["tool_failures"] += 1
                res = {"status": "tool_error", "message": type(e).__name__}
                step.status, step.detail = "tool_error", f"{type(e).__name__} (attempt {attempt + 1})"
        step.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        trace.append(step)
        counters["called"].append(name)
        return res

    @staticmethod
    def _note(trace, kind, detail, status="ok"):
        trace.append(TraceStep(len(trace) + 1, kind, None, None, status, detail))

    def run(self, request: str) -> AgentResult:
        trace: list[TraceStep] = []
        c = {"called": [], "tool_failures": 0}
        req = request.strip()
        oid_m = ORDER_RE.search(req)
        oid = oid_m.group(0).upper() if oid_m else None
        amt_m = AMOUNT.search(req)
        requested = float(amt_m.group(1)) if amt_m else None
        if INJ.search(req):
            self._note(trace, "guard", "instruction-override attempt detected; policy and approval rules unchanged", "flagged")

        def done(status, reply, **kw):
            self._note(trace, "final", status)
            return AgentResult(req[:300], status, reply, c["called"], [asdict(t) for t in trace], tool_failures=c["tool_failures"], **kw)

        if SAFETY.search(req):
            self._note(trace, "plan", "safety keywords -> retrieve safety policy and escalate")
            pol = self._call("search_policy", trace, c, query="product safety incident smoke fire escalation")
            act = self.gate.propose("escalate_safety_incident", {"summary": redact(req, 120)}, "safety incident must be escalated immediately", "human_agent")
            self._note(trace, "approval", f"escalation proposed {act['action_id']}")
            cite = ", ".join(f"{x['doc_id']} v{x['version']}" for x in pol.get("citations", [])) or "no policy evidence available"
            return done("escalated", "Safety first: please stop using the product and keep away from it, and follow approved safety guidance. I have escalated this to a human agent immediately; "
                                     f"I will not give diagnostic instructions. (Policy evidence: {cite}.)", approval_required=True, approval=act)

        if ACCOUNT_CHANGE.search(req):
            self._note(trace, "plan", "account change requested -> human verification route, no tool call")
            act = self.gate.propose("account_change_request", {"fields": "email/password (values not captured)"}, "identity verification and human handling required", "human_agent")
            self._note(trace, "approval", f"proposed {act['action_id']}")
            return done("awaiting_approval", "I can't change your email address or reset your password myself. A human agent must verify your identity first - I have logged a request. "
                                              "Please never share your password or one-time codes with anyone, including me.", approval_required=True, approval=act)

        wants_refund, wants_cancel = bool(REFUND.search(req)), bool(CANCEL.search(req))
        order_flow = wants_refund or wants_cancel or bool(ELIG.search(req)) or bool(STATUS.search(req)) or oid is not None
        if POLICY.search(req) and not (wants_refund or wants_cancel) and oid is None:
            order_flow = False

        if order_flow and oid is None:
            self._note(trace, "plan", "order-specific request without an order id -> ask, call no order tool")
            return done("needs_info", "I need your order number (format ORD followed by 8 digits) before I can check this. Please send it and I'll continue.", missing_info=["order_id"])

        if not order_flow:
            self._note(trace, "plan", "policy question -> search_policy")
            pol = self._call("search_policy", trace, c, query=req)
            if pol.get("status") in ("tool_error", "invalid_arguments", "error"):
                return done("error", "Policy search is temporarily unavailable. I have not guessed an answer; a human agent can help.")
            if pol.get("abstained"):
                return done("answered", "I couldn't find enough policy evidence to answer this reliably, so I won't guess. A human agent can help.")
            return done("answered", pol["answer"])

        self._note(trace, "plan", f"order flow for {oid}")
        order = self._call("lookup_order", trace, c, order_id=oid)
        if order.get("status") == "tool_error":
            return done("error", "The order system is temporarily unavailable. I have not made any assumption about your order; please try again shortly or ask a human agent.")
        if order.get("status") != "ok":
            return done("needs_info", f"I couldn't find an order matching {oid}. Could you double-check the number?", missing_info=["valid_order_id"])
        facts = (f"Order {oid}: {order['product_category']}, GBP {order['order_value']:.2f}, dispatch '{order['dispatch_state']}', "
                 f"delivery '{order['delivery_status']}', {order['days_since_delivery']} days since delivery.")

        if wants_cancel:
            elig = self._call("check_return_eligibility", trace, c, order_id=oid, condition="unknown")
            act = self.gate.propose("cancel_order", {"order_id": oid}, f"order is '{order['dispatch_state']}'; cancelling a dispatched order needs a human decision", "human_agent")
            self._note(trace, "approval", f"proposed {act['action_id']}")
            msg = f"{facts} This order has already been dispatched, so I cannot cancel it - it has NOT been cancelled. " if order["dispatch_state"] == "dispatched" else f"{facts} "
            return done("awaiting_approval", msg + f"Return eligibility: {elig.get('decision', 'unknown')} ({elig.get('reason', 'n/a')}). I've routed this to a human agent for approval.",
                        approval_required=True, approval=act)

        if wants_refund:
            if re.search(r"(?i)wrong", req):
                cond = "wrong_item"
            elif re.search(r"(?i)minor|cosmetic|scratch|packaging", req) and not re.search(r"(?i)smash|broken|crushed", req):
                cond = "minor_damage"
            elif re.search(r"(?i)damag|broken|smash|faulty", req):
                cond = "damaged_item"
            else:
                cond = self._condition_from_store(oid) or "unknown"
            ref = self._call("calculate_refund", trace, c, order_id=oid, condition=cond, requested_amount=requested)
            if ref.get("status") != "ok":
                return done("error", "I couldn't compute a refund proposal for that order, so I'm not proposing one; a human agent can review it.")
            if ref["requires_approval"]:
                act = self.gate.propose("issue_refund", {"order_id": oid, "amount_gbp": ref["proposed_refund"], "requested_gbp": requested}, f"{ref['basis']}; route {ref['approval_route']}", ref["approval_route"])
                self._note(trace, "approval", f"proposed {act['action_id']}")
                extra = (" " + "; ".join(ref["notes"]) + ".") if ref["notes"] else ""
                conf = " The customer must also agree to the partial amount." if ref["customer_confirmation_needed"] else ""
                return done("awaiting_approval", f"{facts} Proposed refund: GBP {ref['proposed_refund']:.2f} ({ref['basis']}).{extra} This exceeds frontline authority (GBP {FRONTLINE_REFUND_LIMIT_GBP:g}) "
                                                  f"so NO refund has been issued; it awaits {ref['approval_route']} approval.{conf}", approval_required=True, approval=act)
            return done("answered", f"{facts} Proposed refund: GBP {ref['proposed_refund']:.2f} ({ref['basis']}). This is a proposal within frontline authority; nothing has been paid by this assistant."
                        + (" The customer must agree to this partial amount before it is offered as final." if ref["customer_confirmation_needed"] else ""))

        if ELIG.search(req):
            elig = self._call("check_return_eligibility", trace, c, order_id=oid, condition="unknown")
            return done("answered", f"{facts} Return eligibility: {elig.get('decision', 'manual_review')} - {elig.get('reason', 'n/a')}.")
        return done("answered", facts + " I have not inferred anything beyond the order record.")

    @staticmethod
    def _condition_from_store(oid):
        try:
            with open(AGENT_STORE / "returns.csv", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["order_id"] == oid:
                        return r["item_condition"] if r["item_condition"] in T.CONDITIONS else None
        except OSError:
            return None
        return None
