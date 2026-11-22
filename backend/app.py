import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()
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
