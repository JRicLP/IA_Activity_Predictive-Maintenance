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


def test_report_service_retries_transient_failure(monkeypatch):
    attempts = {"count": 0}

    class Models:
        def generate_content(self, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                class TransientError(Exception):
                    response = type("Response", (), {"status_code": 503})()
                raise TransientError()
            class Response:
                text = json.dumps({
                    "summary": "Resumo apos retry.",
                    "main_factors": ["Torque."],
                    "limitations": ["Score nao calibrado."],
                    "recommended_actions": ["Avaliacao tecnica."],
                    "conclusion": "Concluir apos inspeção.",
                })
            return Response()

    class Client:
        models = Models()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_REPORTS_ENABLED", "true")
    monkeypatch.setenv("GEMINI_RETRY_DELAY_SECONDS", "0")
    service = GeminiReportService()
    service._client = Client()
    report = service.generate_report({"schema_version": "1.0"})

    assert attempts["count"] == 2
    assert report.summary == "Resumo apos retry."


def test_report_service_returns_rate_limit_without_retry(monkeypatch):
    attempts = {"count": 0}

    class Models:
        def generate_content(self, **kwargs):
            attempts["count"] += 1
            class RateError(Exception):
                response = type("Response", (), {"status_code": 429})()
            raise RateError()

    class Client:
        models = Models()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_REPORTS_ENABLED", "true")
    service = GeminiReportService()
    service._client = Client()

    try:
        service.generate_report({"schema_version": "1.0"})
        raise AssertionError("Era esperado GeminiRateLimitError")
    except Exception as exc:
        assert exc.__class__.__name__ == "GeminiRateLimitError"
    assert attempts["count"] == 1


def test_report_service_does_not_retry_unavailable_model(monkeypatch):
    attempts = {"count": 0}

    class Models:
        def generate_content(self, **kwargs):
            attempts["count"] += 1
            class ModelError(Exception):
                code = 404
            raise ModelError()

    class Client:
        models = Models()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_REPORTS_ENABLED", "true")
    service = GeminiReportService()
    service._client = Client()

    try:
        service.generate_report({"schema_version": "1.0"})
        raise AssertionError("Era esperado GeminiUnavailableError")
    except Exception as exc:
        assert exc.__class__.__name__ == "GeminiUnavailableError"
    assert attempts["count"] == 1
