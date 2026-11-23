"""Agent entry point: adapter over src/omnisupport/agent/core.py (deterministic planner + approval gate + trace)."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from omnisupport.agent.core import SupportAgent  # noqa: E402

_AGENT = None


def run_agent(user_request: str) -> dict:
    global _AGENT
    _AGENT = _AGENT or SupportAgent()
    r = _AGENT.run(user_request)
    tool_status = [t["status"] for t in r.trace if t["kind"] == "tool"]
    if "tool_error" in tool_status:
        status = "tool_error"
    elif "not_found" in tool_status:
        status = "not_found"
    elif r.status == "needs_info":
        status = "needs_information"
    else:
        status = r.status
    return {"status": status, "reply": r.reply, "tools_called": r.tools_called, "approval_required": r.approval_required, "approval": r.approval, "trace": r.trace, "missing_info": r.missing_info}
