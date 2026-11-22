import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explain import ExplainService
from predict import Predictor


READINGS = {
    "type": "L",
    "air_temperature_k": 301.9,
    "process_temperature_k": 310.4,
    "rotational_speed_rpm": 1286,
    "torque_nm": 62.8,
    "tool_wear_min": 22,
}


def test_local_explanation_matches_prediction_and_is_finite():
    predictor = Predictor()
    prediction = predictor.predict(READINGS)
    service = ExplainService(predictor)
    explanation = service.explain_local(READINGS, prediction)

    assert explanation["positive_class"] == 1
    assert explanation["model_score"] == prediction["model_score"]
    assert len(explanation["contributions"]) == 11
    assert all(
        math.isfinite(item["value"]) and math.isfinite(item["contribution"])
        for item in explanation["contributions"]
    )
    if explanation["base_value"] is not None:
        expected = explanation["base_value"] + sum(
            item["contribution"] for item in explanation["contributions"]
        )
        assert abs(expected - explanation["model_score"]) <= 0.05


def test_global_explanation_is_sorted_and_has_positive_importance():
    predictor = Predictor()
    service = ExplainService(predictor, reference_size=100)
    global_result = service.explain_global()

    importance = [item["importance"] for item in global_result["features"]]
    assert len(importance) == 11
    assert importance == sorted(importance, reverse=True)
    assert all(math.isfinite(value) and value >= 0 for value in importance)


def test_100_predictions_have_finite_decompositions():
    predictor = Predictor()
    service = ExplainService(predictor)
    readings = []
    for index in range(100):
        readings.append({
            "type": ("L", "M", "H")[index % 3],
            "air_temperature_k": 295.3 + (index % 10) * 0.9,
            "process_temperature_k": 305.7 + (index % 10) * 0.8,
            "rotational_speed_rpm": 1168 + index * 15,
            "torque_nm": 3.8 + (index % 20) * 3.5,
            "tool_wear_min": index % 254,
        })

    for item in readings:
        prediction = predictor.predict(item)
        explanation = service.explain_local(item, prediction)
        assert all(math.isfinite(row["contribution"]) for row in explanation["contributions"])
