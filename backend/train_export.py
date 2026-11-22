"""Treina a Random Forest do grupo e gera TUDO que a API e o front usam.

Saídas (pasta models/):
  - random_forest.joblib  o modelo
  - metadata.json         configuração, versões das libs e hashes (a API confere ao subir)
  - metricas.json         métricas, gráficos e exemplos da aba "Validação do modelo"

Regra do grupo: o conjunto de TESTE não é usado aqui. Métricas e exemplos
vêm só da VALIDAÇÃO.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import train_test_split

from preprocessing import CSV_FIELDS, FEATURES, NUMERIC_FIELDS, build_features

MODELS_DIR = Path(__file__).resolve().parent / "models"
THRESHOLD = 0.5


def original_split(target):
    train, remainder = train_test_split(
        target.index, test_size=0.4, stratify=target, random_state=42
    )
    valid, test = train_test_split(
        remainder, test_size=0.5, stratify=target.loc[remainder], random_state=42
    )
    return train, valid, test


def metricas_validacao(model, readings_valid, y_valid):
    """Tudo calculado só na validação (2.000 linhas)."""
    score = model.predict_proba(build_features(readings_valid))[:, 1]
    pred = (score > THRESHOLD).astype(int)  # mesma regra de alerta da API
    yv = y_valid.to_numpy()
    prec, rec, _ = precision_recall_curve(yv, score)

    # amostra para o gráfico rotação × torque: todas as falhas + 300 sem falha
    amostra = readings_valid[["rotational_speed_rpm", "torque_nm"]].assign(y=yv)
    amostra = pd.concat([amostra[amostra.y == 1],
                         amostra[amostra.y == 0].sample(300, random_state=42)])

    def exemplo(mascara):
        linha = readings_valid[mascara].iloc[0]
        return {"type": str(linha["type"]), **{c: float(linha[c]) for c in NUMERIC_FIELDS}}

    return {
        "limiar": THRESHOLD,
        "pr_auc": float(average_precision_score(yv, score)),
        "f1": float(f1_score(yv, pred)),
        "recall": float(recall_score(yv, pred)),
        "precisao": float(precision_score(yv, pred)),
        "matriz_confusao": confusion_matrix(yv, pred).tolist(),
        "curva_pr": {"precisao": prec[::5].tolist(), "recall": rec[::5].tolist()},
        "importancias": dict(zip(FEATURES, map(float, model.feature_importances_))),
        "hist_score": {
            "sem_falha": np.histogram(score[yv == 0], bins=20, range=(0, 1))[0].tolist(),
            "falha": np.histogram(score[yv == 1], bins=20, range=(0, 1))[0].tolist()},
        # nomes dos campos iguais aos do POST /predict
        "exemplos": {
            "sem_alerta": exemplo((yv == 0) & (score < 0.05)),
            "com_alerta": exemplo((yv == 1) & (pred == 1))},
        "amostra": amostra.round(1).values.tolist(),
    }


def export(dataset: Path):
    frame = pd.read_csv(dataset, encoding="utf-8-sig")
    target = frame["Machine failure"]
    train_ids, valid_ids, test_ids = original_split(target)
    assert train_ids.intersection(valid_ids).empty and train_ids.intersection(test_ids).empty
    readings = frame.loc[train_ids, list(CSV_FIELDS)].rename(columns=CSV_FIELDS)
    assert len(readings) == 6000 and int(target.loc[train_ids].sum()) == 203

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight=None, random_state=42, n_jobs=4
    ).fit(build_features(readings), target.loc[train_ids])

    # Métricas da validação (o teste, test_ids, não é lido em nenhum ponto abaixo)
    readings_valid = frame.loc[valid_ids, list(CSV_FIELDS)].rename(columns=CSV_FIELDS)
    metricas = metricas_validacao(model, readings_valid, target.loc[valid_ids])

    model.n_jobs = 1
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "random_forest.joblib"
    joblib.dump(model, model_path, compress=3)
    metadata = {
        "model_version": "ai4i-rf-v1",
        "model": "RandomForestClassifier",
        "threshold": THRESHOLD,
        "alert_rule": "model_score > threshold",
        "target": "Machine failure",
        "positive_class": 1,
        "features": list(FEATURES),
        "training_rows": len(readings),
        "training_failures": int(target.loc[train_ids].sum()),
        "training": {"random_state": 42, "n_estimators": 200, "max_depth": 10,
                     "class_weight": None, "resampling": None, "split": "60/20/20 original"},
        "training_ranges": {name: [float(readings[name].min()), float(readings[name].max())]
                            for name in NUMERIC_FIELDS},
        "versions": {name: version(name) for name in
                     ("scikit-learn", "numpy", "pandas", "joblib", "scipy", "threadpoolctl")},
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (MODELS_DIR / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Modelo salvo em {model_path}")
    print("VALIDAÇÃO:", {k: round(metricas[k], 3) for k in ("pr_auc", "f1", "recall", "precisao")})
    print("Matriz de confusão [[TN, FP], [FN, TP]]:", metricas["matriz_confusao"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    export(parser.parse_args().dataset)
