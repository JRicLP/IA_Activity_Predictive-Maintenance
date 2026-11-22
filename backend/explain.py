import threading
from pathlib import Path

import numpy as np
import pandas as pd
from shapash import SmartExplainer
from sklearn.model_selection import train_test_split

from preprocessing import CSV_FIELDS, build_features


DATASET_PATH = Path(__file__).resolve().parent.parent / "ai4i2020.csv"


class ExplainService:
    def __init__(self, predictor, reference_size: int = 300):
        self.predictor = predictor
        self.reference_size = reference_size
        self._global_result = None
        self._lock = threading.Lock()

    def _compile(self, features: pd.DataFrame):
        y_pred = pd.Series(
            self.predictor.model.predict(features),
            index=features.index,
        )
        proba = pd.Series(
            self.predictor.model.predict_proba(features)[:, 1],
            index=features.index,
        )
        explainer = SmartExplainer(
            model=self.predictor.model,
            features_dict={feature: feature for feature in features.columns},
            label_dict={0: "Sem falha", 1: "Falha"},
        )
        explainer.compile(
            x=features,
            y_pred=y_pred,
            proba_values=proba,
        )
        return explainer

    def _training_reference(self) -> pd.DataFrame:
        frame = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
        target = frame["Machine failure"]
        train_ids, remainder = train_test_split(
            target.index,
            test_size=0.40,
            stratify=target,
            random_state=42,
        )
        valid_ids, test_ids = train_test_split(
            remainder,
            test_size=0.50,
            stratify=target.loc[remainder],
            random_state=42,
        )
        del valid_ids, test_ids
        readings = (
            frame.loc[train_ids, list(CSV_FIELDS)]
            .rename(columns=CSV_FIELDS)
            .reset_index(drop=True)
        )
        return build_features(readings).sample(
            n=min(self.reference_size, len(readings)),
            random_state=42,
        )

    def explain_global(self) -> dict:
        if self._global_result is not None:
            return self._global_result
        with self._lock:
            if self._global_result is not None:
                return self._global_result
            features = self._training_reference()
            explainer = self._compile(features)
            contributions = explainer.explainer.contributions[1]
            importance = contributions.abs().mean(axis=0)
            result = [
                {
                    "feature": feature,
                    "importance": float(importance[feature]),
                }
                for feature in importance.sort_values(ascending=False).index
            ]
            self._global_result = {
                "model_version": self.predictor.metadata["model_version"],
                "positive_class": 1,
                "method": "shapash",
                "reference": "6000 registros de treino; amostra deterministica de 300",
                "features": result,
            }
            return self._global_result

    def explain_local(self, readings: dict, prediction: dict) -> dict:
        features = build_features(pd.DataFrame([readings]))
        with self._lock:
            explainer = self._compile(features)
        contribution_frame = explainer.explainer.contributions[1]
        contribution_row = contribution_frame.iloc[0]

        base_value = None
        backend_explainer = getattr(explainer.explainer, "backend", None)
        underlying = getattr(backend_explainer, "explainer", None)
        expected = getattr(underlying, "expected_value", None)
        if expected is not None:
            values = np.asarray(expected).reshape(-1)
            if len(values) > 1:
                base_value = float(values[1])
            elif len(values) == 1:
                base_value = float(values[0])

        contributions = []
        for feature in contribution_row.index:
            value = float(features.iloc[0][feature])
            contribution = float(contribution_row[feature])
            if not np.isfinite(value) or not np.isfinite(contribution):
                raise RuntimeError("Contribuicao Shapash invalida.")
            contributions.append(
                {
                    "feature": feature,
                    "value": value,
                    "contribution": contribution,
                }
            )
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)

        decomposed = None
        if base_value is not None:
            decomposed = base_value + sum(item["contribution"] for item in contributions)
            tolerance = 0.05
        else:
            tolerance = None

        return {
            "model_version": prediction["model_version"],
            "model_score": prediction["model_score"],
            "threshold": prediction["threshold"],
            "positive_class": 1,
            "base_value": base_value,
            "decomposition": decomposed,
            "decomposition_tolerance": tolerance,
            "contributions": contributions,
            "out_of_training_range": prediction["out_of_training_range"],
            "explanation_method": "shapash",
            "scale": "class_1_score",
        }
