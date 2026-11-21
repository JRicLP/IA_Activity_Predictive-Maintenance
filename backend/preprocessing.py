import numpy as np
import pandas as pd


CSV_FIELDS = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}
NUMERIC_FIELDS = tuple(CSV_FIELDS.values())[1:]
FEATURES = (*NUMERIC_FIELDS, "temp_diff", "power_w", "wear_x_torque",
            "Type_H", "Type_L", "Type_M")


def build_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Convert the six raw inputs to the ordered notebook features."""
    result = readings.loc[:, list(NUMERIC_FIELDS)].astype(float)
    result["temp_diff"] = result["process_temperature_k"] - result["air_temperature_k"]
    result["power_w"] = result["torque_nm"] * result["rotational_speed_rpm"] * (2 * np.pi / 60)
    result["wear_x_torque"] = result["tool_wear_min"] * result["torque_nm"]
    for category in ("H", "L", "M"):
        result[f"Type_{category}"] = readings["type"].eq(category).astype(int)
    return result.loc[:, list(FEATURES)]
