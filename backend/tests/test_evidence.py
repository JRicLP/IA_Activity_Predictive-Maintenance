import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence import build_evidence
from explain import ExplainService
from predict import Predictor


def test_evidence_is_structured_and_limits_top_features():
    predictor = Predictor()
    readings = {
        "type": "L",
        "air_temperature_k": 301.9,
        "process_temperature_k": 310.4,
        "rotational_speed_rpm": 1286,
        "torque_nm": 62.8,
        "tool_wear_min": 22,
    }
    prediction = predictor.predict(readings)
    explanation = ExplainService(predictor).explain_local(readings, prediction)
    explanation["global_importance"] = [
        {"feature": "temp_diff", "importance": 0.2}
    ]
    evidence = build_evidence(readings, prediction, explanation, {}, top_k=3)

    assert evidence.schema_version == "1.0"
    assert len(evidence.local_contributions) == 3
    assert len(evidence.omitted_evidence) == 8
    assert evidence.generated_at.endswith("+00:00")
