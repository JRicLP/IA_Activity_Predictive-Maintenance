import json
import os
import threading
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1)
    main_factors: list[str] = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)
    recommended_actions: list[str] = Field(min_length=1)
    conclusion: str = Field(min_length=1)


class GeminiUnavailableError(RuntimeError):
    pass


class GeminiRateLimitError(RuntimeError):
    pass


class GeminiTimeoutError(RuntimeError):
    pass


class GeminiProviderError(RuntimeError):
    pass


class GeminiReportService:
    def __init__(self):
        self.key = os.getenv("GEMINI_API_KEY")
        self.enabled = os.getenv("GEMINI_REPORTS_ENABLED", "true").lower() == "true"
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        if self.model in {"gemini-2.5-flash", "models/gemini-2.5-flash"}:
            self.model = "gemini-3.8-flash"
        fallback = os.getenv(
            "GEMINI_FALLBACK_MODELS",
            "gemini-3.1-flash-lite",
        )
        self.fallback_models = [
            item.strip()
            for item in fallback.split(",")
            if item.strip()
        ]
        self.candidate_models = list(dict.fromkeys(
            [self.model, *self.fallback_models]
        ))
        self.timeout = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
        self.cooldown_seconds = 10.0
        self.max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("GEMINI_RETRY_DELAY_SECONDS", "1"))
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._client = (
            genai.Client(
                api_key=self.key,
                http_options=types.HttpOptions(
                    timeout=int(self.timeout * 1000),
                    retryOptions=types.HttpRetryOptions(
                        attempts=1,
                        httpStatusCodes=[408, 500, 502, 503, 504],
                    ),
                ),
            )
            if self.key
            else None
        )

    @staticmethod
    def report_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "main_factors": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "recommended_actions": {"type": "array", "items": {"type": "string"}},
                "conclusion": {"type": "string"},
            },
            "required": [
                "summary",
                "main_factors",
                "limitations",
                "recommended_actions",
                "conclusion",
            ],
        }

    def generate_report(self, evidence: dict) -> Report:
        if not self.enabled:
            raise GeminiUnavailableError("Relatorios Gemini desativados.")
        if not self.key or self.key == "COLOQUE_SUA_CHAVE_AQUI":
            raise GeminiUnavailableError("GEMINI_API_KEY nao configurada.")

        with self._lock:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.cooldown_seconds:
                raise GeminiRateLimitError("Aguarde antes de gerar outro relatório.")

        system_instruction = (
            "Você é um assistente de interpretação técnica de modelos de manutenção preditiva. "
            "Use exclusivamente o JSON de evidências fornecido. Não invente medições, métricas, "
            "percentuais, causas, falhas específicas ou inspeções realizadas. Diferencie score do "
            "modelo de probabilidade calibrada. Contribuições SHAP descrevem influência na predição, "
            "não causalidade. Cite fatores com sinal correto e unidade quando houver. Se houver "
            "variáveis fora do intervalo de treino, destaque incerteza. Não prescreva intervenções "
            "perigosas; recomende avaliação por equipe técnica. Responda em português do Brasil "
            "no esquema estruturado solicitado."
        )
        prompt = (
            system_instruction
            + "\n\nJSON DE EVIDÊNCIAS:\n"
            + json.dumps(evidence, ensure_ascii=False, default=str)
        )

        last_error = None
        attempts = max(1, self.max_retries)

        for attempt in range(attempts):
            try:
                model = self.candidate_models[
                    min(attempt, len(self.candidate_models) - 1)
                ]
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=self.report_schema(),
                        temperature=0.2,
                    ),
                )
                text = getattr(response, "text", None)
                if not text:
                    raise GeminiProviderError("Gemini retornou uma resposta vazia.")
                try:
                    report = Report.model_validate(json.loads(text))
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise GeminiProviderError(
                        "A resposta do Gemini falhou na validacao."
                    ) from exc

                with self._lock:
                    self._last_request = time.monotonic()
                return report

            except GeminiProviderError as exc:
                last_error = exc
            except (TimeoutError, httpx.TimeoutException) as exc:
                last_error = GeminiTimeoutError("Tempo limite ao consultar Gemini.")
            except Exception as exc:
                response = getattr(exc, "response", None)
                status = (
                    getattr(response, "status_code", None)
                    or getattr(exc, "status_code", None)
                    or getattr(exc, "code", None)
                )
                if status == 429:
                    raise GeminiRateLimitError("Limite gratuito do Gemini atingido.")
                if status == 404:
                    raise GeminiUnavailableError(
                        "O modelo Gemini configurado nao esta disponivel para esta conta."
                    )
                if status in (408, 504):
                    last_error = GeminiTimeoutError("Tempo limite ao consultar Gemini.")
                else:
                    last_error = GeminiProviderError("Falha temporaria no provedor Gemini.")

            if attempt < attempts - 1:
                time.sleep(self.retry_delay * (2 ** attempt))

        if isinstance(last_error, GeminiTimeoutError):
            raise last_error
        raise last_error or GeminiProviderError("Falha temporaria no provedor Gemini.")
