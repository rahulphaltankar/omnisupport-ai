"""Agent tools: thin adapters over the tested implementation in src/omnisupport/agent/tools.py (local CSV store, read-only, no external actions)."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from omnisupport.agent import tools as _t  # noqa: E402

AGENT_STORE = _t.AGENT_STORE
ORDERS_PATH, CUSTOMERS_PATH, RETURNS_PATH = AGENT_STORE / "orders.csv", AGENT_STORE / "customers.csv", AGENT_STORE / "returns.csv"


def lookup_order(order_id: str) -> dict:
    """Return one synthetic order or a controlled not-found result."""
    try:
        return _t.lookup_order(order_id)
    except ValueError as e:
        return {"status": "invalid_arguments", "found": False, "message": str(e).splitlines()[0]}


def lookup_customer(customer_id: str) -> dict:
    try:
        return _t.lookup_customer(customer_id)
    except ValueError as e:
        return {"status": "invalid_arguments", "found": False, "message": str(e).splitlines()[0]}


def calculate_refund(order_id: str, condition: str) -> dict:
    """Advisory refund proposal incl. whether human approval is required."""
    try:
        return _t.calculate_refund(order_id, condition)
    except ValueError as e:
        return {"status": "invalid_arguments", "message": str(e).splitlines()[0]}


def search_policy(query: str) -> dict:
    return _t.search_policy(query)


def check_return_eligibility(order_id: str, condition: str) -> dict:
    try:
        return _t.check_return_eligibility(order_id, condition)
    except ValueError as e:
        return {"status": "invalid_arguments", "message": str(e).splitlines()[0]}
