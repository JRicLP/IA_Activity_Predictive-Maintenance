import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import FEATURES, build_features

MODELS_DIR = Path(__file__).resolve().parent / "models"


class Predictor:
    def __init__(self):
        self.metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
        path = MODELS_DIR / "random_forest.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata["model_sha256"]:
            raise RuntimeError("O arquivo do modelo não corresponde aos metadados.")
        for package, expected in self.metadata["versions"].items():
            if version(package) != expected:
                raise RuntimeError(f"Instale {package}=={expected} para carregar o modelo.")
        self.model = joblib.load(path)
        if tuple(self.model.feature_names_in_) != FEATURES or list(self.model.classes_) != [0, 1]:
            raise RuntimeError("Entradas ou classes do modelo incompatíveis com a API.")

    def predict(self, readings: dict) -> dict:
        features = build_features(pd.DataFrame([readings]))
        score = float(self.model.predict_proba(features)[0, 1])
        threshold = self.metadata["threshold"]
        alert = score > threshold
        outside = [
            name for name, (low, high) in self.metadata["training_ranges"].items()
            if not low <= readings[name] <= high
        ]
        return {
            "prediction": int(alert),
            "model_score": score,
            "inspection_recommended": alert,
            "message": ("Condição associada à falha. Recomendada inspeção." if alert else
                        "Nenhuma condição de falha detectada para as medições informadas."),
            "threshold": threshold,
            "model_version": self.metadata["model_version"],
            "out_of_training_range": outside,
        }
