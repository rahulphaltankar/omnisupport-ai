import csv

from omnisupport.serving.monitoring import SUMMARY_FIELDS, Monitor


def test_summary_counts_and_export(tmp_path):
    m = Monitor(tmp_path / "req.jsonl")
    m.record("/rag/ask", 10, True, rag_no_answer=True)
    m.record("/rag/ask", 30, True, rag_no_answer=False)
    m.record("/agent/run", 20, False, agent_tool_failures=2, extra={"text": "mail a@b.co", "api_key": "k"})
    s = m.summary()
    assert (s.request_count, s.failure_count, s.agent_tool_failure_count) == (3, 1, 2)
    assert s.average_latency_ms == 20.0 and s.rag_no_answer_rate == 0.5
    p = m.export(tmp_path / "s.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert list(rows[0].keys()) == SUMMARY_FIELDS and rows[0]["request_count"] == "3"
    log = (tmp_path / "req.jsonl").read_text()
    assert "a@b.co" not in log and "api_key" not in log
