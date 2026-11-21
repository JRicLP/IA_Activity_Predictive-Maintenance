import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from predict import Predictor


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()
    yield


app = FastAPI(title="Sentinela · API de inspeção", version="1.0.0", lifespan=lifespan)
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
