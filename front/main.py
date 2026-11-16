import json
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from features import montar_features

app = FastAPI(title="Sentinela")
modelo = joblib.load("modelo_rf.joblib")
with open("metricas.json", encoding="utf-8") as f:
    METRICAS = json.load(f)

class Entrada(BaseModel):
    type: Literal["L", "M", "H"]
    air_temp: float = Field(gt=250, lt=350)
    process_temp: float = Field(gt=250, lt=350)
    rpm: float = Field(gt=0, lt=10000)
    torque: float = Field(ge=0, lt=500)
    tool_wear: float = Field(ge=0, lt=1000)

@app.post("/api/predict")
def predict(e: Entrada):
    X = montar_features(pd.DataFrame([e.model_dump()]))
    score = float(modelo.predict_proba(X)[0, 1])
    lin = X.iloc[0]
    return {
        "score": score,
        "alerta": score > METRICAS["limiar"],
        "limiar": METRICAS["limiar"],
        "temp_diff": float(lin["temp_diff"]),
        "power_w": float(lin["power_w"]),
        "wear_x_torque": float(lin["wear_x_torque"]),
    }

@app.get("/api/metricas")
def metricas():
    return METRICAS

app.mount("/", StaticFiles(directory="static", html=True), name="static")
