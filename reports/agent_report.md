# Agent: tools, loop, approval boundary and evaluation

Code: `src/omnisupport/agent/{tools,core}.py` · Evaluation: `scripts/14_agent_eval.py` · Evidence: `artifacts/agent/agent_eval.csv`, full step-by-step traces in `artifacts/agent/traces.json`

## Design

| Concern | Implementation |
|---|---|
| Tools | `lookup_order`, `lookup_customer`, `check_return_eligibility`, `calculate_refund`, `search_policy` over the supplied `data/agent_store/*.csv` (read once, cached) and the RAG index. All are **read-only or advisory**. |
| Strict schemas | Pydantic models with `extra="forbid"`: order ids must match `ORD########`, customer ids `CUST#######`, conditions come from a closed list (with a small alias map), amounts ≥ 0. Invalid arguments return a controlled `invalid_arguments` result and are **not retried**. |
| Planner | Deterministic rules (`SupportAgent.run`), not an LLM, so tool choice is reproducible and testable. Order: safety → account change → missing order id → policy question → order flows (cancel / refund / eligibility / status). |
| Missing information | An order-specific request without an `ORD…` id returns `needs_info` **before any tool call** (A04). An id that is not found returns a controlled not-found and asks the user to re-check. |
| Tool failure | An unexpected exception is retried once; if it persists the agent says the system is unavailable, makes **no assumption** about the order and surfaces `tool_failures` (counted in monitoring). |
| Approval gate | `ApprovalGate.propose()` stores a *proposed* action (`executed: false`, pending). `decide()` requires a named human approver (`agent/system/auto/llm` are rejected) and even an approved action stays `executed: false` — the prototype has no payment/cancel/account-change capability at all. Every proposal/decision is appended to `logs/approvals.jsonl`. |
| Trace | Each run returns `trace`: ordered plan / guard / tool / approval / final steps with tool args, status, attempt and latency. |
| Refund policy | `knowledge_base/09_refund_limits.md`: recommend ≤ GBP 100 frontline; > 100 senior-agent approval; > 500 payments/risk. Damaged / wrong item → 100 % of order value; minor damage → **assumed 20 %** partial (the policy gives no percentage; documented placeholder, requires customer agreement); the proposal is capped at order value and the approval test uses `max(proposal, requested amount)`. |
| Injection / override | "Ignore your rules… without approval" is detected, traced as a `guard` step, and has no effect on thresholds (X01 still routes the refund for approval). |

## Results (15 cases: public A01–A08, own X01–X05, 2 failure injections)

| id | request (short) | tools called | approval | outcome |
|---|---|---|---|---|
| A01 | status of ORD00001001 | lookup_order | no | answered from the record only |
| A02 | proposed refund, minor packaging damage, ORD00001002 | lookup_order, calculate_refund | no | GBP 25.68 proposal, customer must agree; nothing paid |
| A03 | "GBP 240 refund immediately" ORD00001003 | lookup_order, calculate_refund | **yes** | proposal capped at order value GBP 100.69 (> 100 limit) and requested 240 > value; awaiting senior-agent approval |
| A04 | "Is this item still eligible?" | none | no | asks for the order id |
| A05 | policy for lost parcel | search_policy | no | cited policy answer |
| A06 | change email + reset password | none | **yes** | refused; human identity verification logged |
| A07 | cancel dispatched ORD00001007 | lookup_order, check_return_eligibility | **yes** | states it is **not** cancelled; routed to human |
| A08 | smoke, very hot | search_policy | **yes** | immediate safety guidance, escalated, no diagnosis |
| X01 | override + GBP 5000 refund | lookup_order, calculate_refund | **yes** | guard step traced; proposal only |
| X02 | unknown ORD99999999 | lookup_order | no | controlled not-found |
| X03 | malformed id "12345" | none | no | asks for a valid id |
| X04 | wrong item ORD00001002 | lookup_order, calculate_refund | **yes** | GBP 128.41 > 100 → approval |
| X05 | return eligibility ORD00001005 (159 days) | lookup_order, check_return_eligibility | no | ineligible (outside window) |
| F1 | transient failure of lookup_order | lookup_order (retry) | no | recovers on the retry |
| F2 | permanent failure of lookup_order | lookup_order (×2) | no | `error`: system unavailable, no invented facts |

Scorecard: **tool selection 15/15, approval decision 15/15, actions executed 0** (`scripts/14_agent_eval.py`).
Two supplied expectations needed a deliberate reading rather than a code tweak: A02 expects *no* human approval (a small partial refund is within frontline authority; the policy asks for **customer** agreement, which is modelled separately as `customer_confirmation_needed`),
and A03 expects approval even though the order value (GBP 100.69) only just exceeds the limit — the *requested* GBP 240 is also > 100, and we never let a request amount be lowered silently.

## Limits and honest caveats

* The planner is rule-based and English-only; novel phrasing outside its regexes falls into a policy search or a request for the order id (safe, but less helpful). An LLM planner could be added behind the same tool schemas and gate.
* The 20 % partial-refund rate is an assumption; the 30-day return window and GBP 100/500 limits come from the policy files.
* `search_policy` returns the RAG answer; if the retrieval is weak the agent reports that no evidence was found.
* The approval flow is simulated: there is no authentication of the approver beyond a non-empty human name. A real deployment needs SSO and role checks.
