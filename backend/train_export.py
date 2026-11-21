"""Recreate the notebook reference model using the original training split."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


def export(dataset: Path):
    frame = pd.read_csv(dataset)
    target = frame["Machine failure"]
    train_ids, valid_ids, test_ids = original_split(target)
    assert train_ids.intersection(valid_ids).empty and train_ids.intersection(test_ids).empty
    readings = frame.loc[train_ids, list(CSV_FIELDS)].rename(columns=CSV_FIELDS)
    assert len(readings) == 6000 and int(target.loc[train_ids].sum()) == 203

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight=None, random_state=42, n_jobs=4
    ).fit(build_features(readings), target.loc[train_ids])
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
    print(f"Modelo salvo em {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    export(parser.parse_args().dataset)
