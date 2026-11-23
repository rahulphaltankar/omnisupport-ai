import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from omnisupport.serving.monitoring import MONITOR

client = TestClient(app, raise_server_exceptions=False)
TICKET = {"support_channel": "email", "customer_region": "London", "customer_tenure_months": 12, "product_category": "electronics", "order_value": 120.0, "delivery_delay_days": 5,
          "previous_ticket_count": 2, "issue_title": "Late delivery", "issue_description": "Parcel is 5 days late and I need it urgently"}


def test_health_reports_versions():
    b = client.get("/health").json()
    assert b["status"] == "ok" and b["model_version"] and b["prompt_version"] and b["index_version"]
    assert "api_key" not in b["llm"] and isinstance(b["llm"]["api_key_configured"], bool)  # status only, never the key


def test_extract_returns_validated_structure():
    r = client.post("/tickets/extract", json={"text": "My order ORD00001234 is 6 days late and I need it for work."})
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["order_id"] == "ORD00001234" and d["issue_category"] == "delivery_late"


def test_extract_never_invents_order_id():
    d = client.post("/tickets/extract", json={"text": "My parcel is late again and I am frustrated."}).json()["data"]
    assert d["order_id"] is None and "order_id" in d["missing_information"]


def test_rag_unsupported_question_is_no_answer():
    b = client.post("/rag/ask", json={"question": "What is the policy for teleporting an order to Mars?"}).json()
    assert b["abstained"] is True and b["status"] == "no_answer" and b["citations"] == []


def test_rag_supported_question_cites_sources():
    b = client.post("/rag/ask", json={"question": "Who must approve refunds above GBP 100?"}).json()
    assert b["abstained"] is False and b["citations"] and b["index_version"]


def test_agent_approval_flow_never_executes():
    r = client.post("/agent/run", json={"request": "Issue a GBP 240 refund immediately for order ORD00001003."}).json()
    assert r["approval_required"] and r["approval"]["executed"] is False
    aid = r["approval"]["action_id"]
    bad = client.post("/agent/approve", json={"action_id": aid, "approver": "agent", "approve": True}).json()
    assert bad["status"] == "rejected"
    ok = client.post("/agent/approve", json={"action_id": aid, "approver": "senior.agent", "approve": True}).json()
    assert ok["status"] == "approved_for_downstream" and ok["executed"] is False
    assert client.post("/agent/approve", json={"action_id": "ACT-doesnotexist", "approver": "x.y", "approve": True}).status_code == 404


def test_agent_missing_order_id_asks_for_it():
    r = client.post("/agent/run", json={"request": "Is this item still eligible for return?"}).json()
    assert r["status"] == "needs_info" and r["tools_called"] == []


@pytest.mark.parametrize("path", ["../../README.md", "C:/Windows/win.ini", "data/images/../../pyproject.toml", "/etc/passwd"])
def test_vision_blocks_path_traversal(path):
    assert client.post("/vision/predict", json={"image_path": path}).status_code in {400, 404}


def test_vision_missing_image_is_404():
    assert client.post("/vision/predict", json={"image_path": "data/images/does_not_exist.jpg"}).status_code == 404


def test_predict_validates_inputs():
    assert client.post("/tickets/predict", json={**TICKET, "order_value": -5}).status_code == 422
    assert client.post("/tickets/predict", json={"order_value": 5}).status_code == 422


def test_predict_returns_probability_or_controlled_503():
    r = client.post("/tickets/predict", json=TICKET)
    assert r.status_code in {200, 503}
    if r.status_code == 200:
        b = r.json()
        assert 0 <= b["escalation_probability"] <= 1 and isinstance(b["flag_for_priority_review"], bool)
    else:
        assert "unavailable" in r.json()["detail"]


def test_monitoring_summary_counts_requests():
    before = MONITOR.summary().request_count
    client.get("/health")  # health is not counted
    client.post("/rag/ask", json={"question": "What is the capital of Australia?"})
    s = client.get("/monitoring/summary").json()
    assert s["request_count"] == before + 1 and "rag_no_answer_rate" in s and s["by_endpoint"]
