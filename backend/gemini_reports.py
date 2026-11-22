import json
import os
import threading
import time
from pathlib import Path
from typing import Literal

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
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
        self.cooldown_seconds = 10.0
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._client = (
            genai.Client(
                api_key=self.key,
                http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
            )
            if self.key
            else None
        )

    def generate_report(self, evidence: dict) -> Report:
        if not self.enabled:
            raise GeminiUnavailableError("Relatorios Gemini desativados.")
        if not self.key or self.key == "COLOQUE_SUA_CHAVE_AQUI":
            raise GeminiUnavailableError("GEMINI_API_KEY nao configurada.")

        with self._lock:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.cooldown_seconds:
                raise GeminiRateLimitError("Aguarde antes de gerar outro relatório.")
            self._last_request = time.monotonic()

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

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Report,
                    temperature=0.2,
                ),
            )
        except TimeoutError:
            raise GeminiTimeoutError("Tempo limite ao consultar Gemini.")
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                raise GeminiRateLimitError("Limite gratuito do Gemini atingido.")
            if status in (408, 504):
                raise GeminiTimeoutError("Tempo limite ao consultar Gemini.")
            raise GeminiProviderError("Falha temporaria no provedor Gemini.") from exc

        text = getattr(response, "text", None)
        if not text:
            raise GeminiProviderError("Gemini retornou uma resposta vazia.")
        try:
            data = json.loads(text)
            report = Report.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise GeminiProviderError("A resposta do Gemini falhou na validação.") from exc
        return report
