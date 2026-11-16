import numpy as np
import pandas as pd

# Mesma ordem de colunas do notebook do grupo (rodada 2, "Com Type"):
# 5 medições, 3 features físicas e o one-hot de Type (H, L, M).
FEATURES = ["air_temp", "process_temp", "rpm", "torque", "tool_wear",
            "temp_diff", "power_w", "wear_x_torque", "Type_H", "Type_L", "Type_M"]

def montar_features(d: pd.DataFrame) -> pd.DataFrame:
    """Usada no treino E na API, para os dois calcularem as features do mesmo jeito."""
    X = d.copy()
    X["temp_diff"] = X["process_temp"] - X["air_temp"]
    X["power_w"] = X["torque"] * X["rpm"] * 2 * np.pi / 60
    X["wear_x_torque"] = X["tool_wear"] * X["torque"]
    for t in ("H", "L", "M"):
        X[f"Type_{t}"] = (X["type"] == t).astype(float)
    return X[FEATURES]
