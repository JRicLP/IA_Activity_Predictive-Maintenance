import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app import app


READINGS = {
    "type": "L",
    "air_temperature_k": 301.9,
    "process_temperature_k": 310.4,
    "rotational_speed_rpm": 1286,
    "torque_nm": 62.8,
    "tool_wear_min": 22,
}


def test_health_predict_and_explain_preserve_contract():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200

        prediction = client.post("/predict", json=READINGS)
        assert prediction.status_code == 200

        explanation = client.post("/explain", json=READINGS)
        assert explanation.status_code == 200
        body = explanation.json()
        assert body["explanation"]["positive_class"] == 1
        assert len(body["explanation"]["contributions"]) == 11
