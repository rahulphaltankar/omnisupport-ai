"""Structured LLM output: Pydantic contract + validated call. Implementation in src/omnisupport/llm/ (providers: rules | local_hf | anthropic)."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from omnisupport.llm.schemas import TicketExtraction  # noqa: E402,F401
from omnisupport.llm.structured import run_structured  # noqa: E402


def call_llm_for_structured_output(message: str) -> TicketExtraction:
    """Validate -> one repair attempt -> deterministic fallback. Provider chosen by LLM_PROVIDER env (default: offline rules)."""
    from omnisupport.serving.service import _safe_provider

    res = run_structured("extraction", message, _safe_provider())
    return TicketExtraction.model_validate(res.data)
