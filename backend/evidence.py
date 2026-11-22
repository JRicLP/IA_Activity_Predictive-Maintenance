from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


METHOD_NOTES = [
    "Dataset AI4I 2020 sintético.",
    "O score não é uma probabilidade calibrada de quebra futura.",
    "Contribuições SHAP explicam influência na predição, não causalidade.",
    "O resultado não substitui avaliação humana.",
]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    model_version: str
    generated_at: str
    reading: dict[str, Any]
    derived: dict[str, float]
    model_score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    prediction: int
    inspection_recommended: bool
    alert_rule: str = "model_score > threshold"
    out_of_training_range: list[str]
    local_contributions: list[dict[str, Any]]
    global_importance: list[dict[str, Any]]
    validation_metrics: dict[str, Any]
    omitted_evidence: list[str] = Field(default_factory=list)
    methodological_notes: list[str] = Field(default_factory=lambda: list(METHOD_NOTES))


def build_evidence(
    readings: dict,
    prediction: dict,
    explanation: dict,
    metrics: dict | None,
    top_k: int = 8,
) -> Evidence:
    units = {
        "type": "L/M/H",
        "air_temperature_k": "K",
        "process_temperature_k": "K",
        "rotational_speed_rpm": "rpm",
        "torque_nm": "Nm",
        "tool_wear_min": "min",
    }
    reading = {
        name: {"value": readings[name], "unit": units[name]}
        for name in units
    }
    contributions = explanation["contributions"]
    if len(contributions) > top_k:
        omitted = [
            item["feature"]
            for item in contributions[top_k:]
        ]
    else:
        omitted = []

    validation = {}
    if metrics:
        validation = {
            "split": "validacao",
            "pr_auc": metrics.get("pr_auc"),
            "f1": metrics.get("f1"),
            "recall": metrics.get("recall"),
            "precisao": metrics.get("precisao"),
            "matriz_confusao": metrics.get("matriz_confusao"),
        }

    return Evidence(
        model_version=prediction["model_version"],
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        reading=reading,
        derived={
            "temp_diff": prediction["temp_diff"],
            "power_w": prediction["power_w"],
            "wear_x_torque": prediction["wear_x_torque"],
        },
        model_score=prediction["model_score"],
        threshold=prediction["threshold"],
        prediction=prediction["prediction"],
        inspection_recommended=prediction["inspection_recommended"],
        out_of_training_range=prediction["out_of_training_range"],
        local_contributions=contributions[:top_k],
        global_importance=explanation.get("global_importance", [])[:top_k],
        validation_metrics=validation,
        omitted_evidence=omitted,
    )
