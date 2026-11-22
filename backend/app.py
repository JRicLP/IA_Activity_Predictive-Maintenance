import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from evidence import build_evidence
from explain import ExplainService
from gemini_reports import (
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiReportService,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from predict import Predictor

BASE_DIR = Path(__file__).resolve().parent
METRICAS_PATH = BASE_DIR / "models" / "metricas.json"
STATIC_DIR = BASE_DIR.parent / "front" / "static"


class Measurements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["L", "M", "H"]
    air_temperature_k: Annotated[float, Field(strict=True, gt=0, le=10000)]
    process_temperature_k: Annotated[float, Field(strict=True, gt=0, le=10000)]
    rotational_speed_rpm: Annotated[float, Field(strict=True, gt=0, le=1_000_000)]
    torque_nm: Annotated[float, Field(strict=True, ge=0, le=1_000_000)]
    tool_wear_min: Annotated[float, Field(strict=True, ge=0, le=1_000_000)]


class Prediction(BaseModel):
    prediction: Literal[0, 1]
    model_score: float
    inspection_recommended: bool
    message: str
    threshold: float
    model_version: str
    out_of_training_range: list[str]
    # Features calculadas a partir das medições (o front mostra esses indicadores)
    temp_diff: float
    power_w: float
    wear_x_torque: float


class Explanation(BaseModel):
    model_version: str
    model_score: float
    threshold: float
    positive_class: int
    base_value: float | None
    decomposition: float | None
    decomposition_tolerance: float | None
    contributions: list[dict]
    out_of_training_range: list[str]
    explanation_method: str
    scale: str
    global_importance: list[dict]


class ExplainResponse(BaseModel):
    prediction: dict
    explanation: Explanation


class ReportResponse(BaseModel):
    report: dict
    model_version: str
    provider: Literal["gemini"] = "gemini"
    evidence: dict
    generated_at: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()
    app.state.explain = ExplainService(app.state.predictor)
    app.state.reports = GeminiReportService()
    # metricas.json é gerado pelo train_export.py (métricas só da validação)
    app.state.metricas = (
        json.loads(METRICAS_PATH.read_text(encoding="utf-8")) if METRICAS_PATH.exists() else None
    )
    yield


app = FastAPI(title="Sentinela · API de inspeção", version="1.1.0", lifespan=lifespan)
origins = os.getenv(
    "SENTINELA_ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.predictor.metadata["model_version"]}


@app.post("/predict", response_model=Prediction)
def predict(data: Measurements, request: Request):
    return request.app.state.predictor.predict(data.model_dump())


@app.post("/explain", response_model=ExplainResponse)
def explain(data: Measurements, request: Request):
    prediction = request.app.state.predictor.predict(data.model_dump())
    local = request.app.state.explain.explain_local(data.model_dump(), prediction)
    global_result = request.app.state.explain.explain_global()
    local["global_importance"] = global_result["features"]
    try:
        explained = Explanation.model_validate(local)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Explicador indisponivel.") from exc
    return {"prediction": prediction, "explanation": explained}


@app.post("/reports/generate", response_model=ReportResponse)
def generate_report(data: Measurements, request: Request):
    predictor = request.app.state.predictor
    explain_service = request.app.state.explain
    readings = data.model_dump()
    prediction = predictor.predict(readings)
    local = explain_service.explain_local(readings, prediction)
    global_result = explain_service.explain_global()
    local["global_importance"] = global_result["features"]
    evidence = build_evidence(
        readings,
        prediction,
        local,
        request.app.state.metricas,
    )
    try:
        report = request.app.state.reports.generate_report(evidence.model_dump())
    except GeminiUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GeminiRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except GeminiTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except GeminiProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "report": report.model_dump(),
        "model_version": prediction["model_version"],
        "provider": "gemini",
        "evidence": evidence.model_dump(),
        "generated_at": evidence.generated_at,
    }


@app.get("/metricas")
def metricas(request: Request):
    if request.app.state.metricas is None:
        raise HTTPException(
            status_code=503,
            detail="models/metricas.json não encontrado. Rode train_export.py para gerá-lo.",
        )
    return request.app.state.metricas


# O front é servido pela própria API (mesma origem, sem problema de CORS).
# Este mount precisa ser o ÚLTIMO: as rotas acima têm prioridade.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="front")
