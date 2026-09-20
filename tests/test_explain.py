from __future__ import annotations

import pytest
import requests

from partcheck.explain import ExplainError, explain
from partcheck.models import Finding, PartInfo, Report, Severity


def _report_with_finding() -> Report:
    part = PartInfo(
        path="part.stl", units="mm", watertight=True, face_count=12, vertex_count=8,
        bounds_mm=[[0, 0, 0], [1, 1, 1]],
    )
    finding = Finding(
        check="thin_wall", severity=Severity.ERROR, face_ids=[0], value=0.4,
        threshold=1.0, message="thin",
    )
    return Report(part=part, findings=[finding], summary={"thin_wall": 1})


def _empty_report() -> Report:
    part = PartInfo(
        path="part.stl", units="mm", watertight=True, face_count=12, vertex_count=8,
        bounds_mm=[[0, 0, 0], [1, 1, 1]],
    )
    return Report(part=part, findings=[], summary={})


def test_explain_skips_api_call_when_no_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = explain(_empty_report())
    assert "No manufacturability issues" in result


def test_explain_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Don't let a real .env file in the repo root repopulate the var we just cleared.
    monkeypatch.setattr("partcheck.explain.load_dotenv", lambda *a, **k: None)
    with pytest.raises(ExplainError, match="GROQ_API_KEY"):
        explain(_report_with_finding())


def test_explain_returns_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "This wall is too thin because..."}}]}

    def fake_post(url, headers, json, timeout):
        assert headers["Authorization"] == "Bearer fake-key"
        assert json["model"]
        assert "thin_wall" in json["messages"][1]["content"]
        return FakeResponse()

    monkeypatch.setattr("partcheck.explain.requests.post", fake_post)
    result = explain(_report_with_finding())
    assert result == "This wall is too thin because..."


def test_explain_wraps_request_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr("partcheck.explain.requests.post", fake_post)
    with pytest.raises(ExplainError, match="Groq API request failed"):
        explain(_report_with_finding())


def test_explain_rejects_malformed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"unexpected": "shape"}

    monkeypatch.setattr("partcheck.explain.requests.post", lambda *a, **k: FakeResponse())
    with pytest.raises(ExplainError, match="Unexpected Groq API response"):
        explain(_report_with_finding())
