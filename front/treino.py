"""Reproduz o modelo do grupo (Random Forest sem pesos, com Type e 3 features físicas).
Divisão 60/20/20, random_state=42. O CONJUNTO DE TESTE NÃO É USADO AQUI:
todas as métricas, exemplos e gráficos vêm da VALIDAÇÃO."""
import json, joblib
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             confusion_matrix, f1_score, recall_score, precision_score)
from features import montar_features, FEATURES

csv_path = Path(__file__).resolve().parent / "ai4i2020.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig").rename(columns={
    "Type": "type", "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp", "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque", "Tool wear [min]": "tool_wear", "Machine failure": "y"})
COLS = ["type", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]
X, y = df[COLS], df["y"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
del X_test, y_test   # reservado para a avaliação final; nada abaixo usa o teste

modelo = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight=None,
                                random_state=42, n_jobs=-1)
modelo.fit(montar_features(X_train), y_train)

Xv = montar_features(X_valid)
score = modelo.predict_proba(Xv)[:, 1]
pred = modelo.predict(Xv)                      # limiar padrão (0,5), sem ajuste
p, r, _ = precision_recall_curve(y_valid, score)
am = X_valid[["rpm", "torque"]].assign(y=y_valid)
am = pd.concat([am[am.y == 1], am[am.y == 0].sample(300, random_state=42)])

metricas = {
    "limiar": 0.5,
    "pr_auc": float(average_precision_score(y_valid, score)),
    "f1": float(f1_score(y_valid, pred)),
    "recall": float(recall_score(y_valid, pred)),
    "precisao": float(precision_score(y_valid, pred)),
    "matriz_confusao": confusion_matrix(y_valid, pred).tolist(),
    "curva_pr": {"precisao": p[::5].tolist(), "recall": r[::5].tolist()},
    "importancias": dict(zip(FEATURES, map(float, modelo.feature_importances_))),
    "hist_score": {
        "sem_falha": np.histogram(score[y_valid == 0], bins=20, range=(0, 1))[0].tolist(),
        "falha": np.histogram(score[y_valid == 1], bins=20, range=(0, 1))[0].tolist()},
    "exemplos": {
        "sem_alerta": X_valid[(y_valid == 0) & (score < 0.05)].iloc[0].to_dict(),
        "com_alerta": X_valid[(y_valid == 1) & (pred == 1)].iloc[0].to_dict()},
    "amostra": am.round(1).values.tolist(),
}
joblib.dump(modelo, "modelo_rf.joblib")
json.dump(metricas, open("metricas.json", "w"), indent=2, default=float)

print("VALIDAÇÃO:", {k: round(v, 3) for k, v in metricas.items() if k in ("pr_auc", "f1", "recall", "precisao")})
print("Matriz de confusão [[TN, FP], [FN, TP]]:", metricas["matriz_confusao"])
print("Notebook do grupo (referência): recall 0,794 | precisão 0,982 | F1 0,878 | AP 0,873 | 54/68 falhas, 14 perdidas, 1 falso alarme")
