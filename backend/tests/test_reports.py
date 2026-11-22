import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemini_reports import GeminiReportService, Report


def test_report_schema_accepts_valid_portuguese_payload():
    report = Report.model_validate({
        "summary": "Resumo técnico.",
        "main_factors": ["Temperatura do processo."],
        "limitations": ["Score não calibrado."],
        "recommended_actions": ["Solicitar avaliação técnica."],
        "conclusion": "Manter inspeção humana.",
    })
    assert report.summary == "Resumo técnico."


def test_report_service_without_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_REPORTS_ENABLED", "true")
    service = GeminiReportService()

    try:
        service.generate_report({"schema_version": "1.0"})
        raise AssertionError("Era esperado GeminiUnavailableError")
    except Exception as exc:
        assert exc.__class__.__name__ == "GeminiUnavailableError"


def test_report_service_accepts_mocked_valid_response(monkeypatch):
    class Models:
        def generate_content(self, **kwargs):
            class Response:
                text = json.dumps({
                    "summary": "Resumo mockado.",
                    "main_factors": ["Torque."],
                    "limitations": ["Score não calibrado."],
                    "recommended_actions": ["Avaliação técnica."],
                    "conclusion": "Concluir após inspeção.",
                })
            return Response()

    class Client:
        models = Models()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_REPORTS_ENABLED", "true")
    service = GeminiReportService()
    service._client = Client()
    report = service.generate_report({"schema_version": "1.0"})

    assert report.summary == "Resumo mockado."
