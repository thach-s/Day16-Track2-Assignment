#!/usr/bin/env python3
"""CPU LightGBM benchmark for the Lab 16 credit-card fraud dataset."""

import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path.home() / "ml-benchmark" / "creditcard.csv"
RESULT_PATH = Path.home() / "ml-benchmark" / "benchmark_result.json"
RANDOM_STATE = 42


def milliseconds(seconds: float) -> float:
    return round(seconds * 1_000, 4)


def main() -> None:
    load_started = time.perf_counter()
    data = pd.read_csv(DATA_PATH)
    load_seconds = time.perf_counter() - load_started

    features = data.drop(columns=["Class"])
    target = data["Class"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1_000,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    training_started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    training_seconds = time.perf_counter() - training_started

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)

    # Warm up before measuring latency so library initialization is excluded.
    model.predict_proba(x_test.iloc[:1])
    latency_runs = []
    one_row = x_test.iloc[:1]
    for _ in range(200):
        started = time.perf_counter()
        model.predict_proba(one_row)
        latency_runs.append(time.perf_counter() - started)

    batch = x_test.iloc[:1_000]
    started = time.perf_counter()
    model.predict_proba(batch)
    batch_seconds = time.perf_counter() - started

    result = {
        "environment": {
            "instance_type": "c7i-flex.large",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "lightgbm_version": lgb.__version__,
            "scikit_learn_version": sklearn.__version__,
            "rows": int(len(data)),
            "features": int(features.shape[1]),
        },
        "load_data_seconds": round(load_seconds, 4),
        "training_seconds": round(training_seconds, 4),
        "best_iteration": int(model.best_iteration_),
        "auc_roc": round(roc_auc_score(y_test, probabilities), 6),
        "accuracy": round(accuracy_score(y_test, predictions), 6),
        "f1_score": round(f1_score(y_test, predictions), 6),
        "precision": round(precision_score(y_test, predictions), 6),
        "recall": round(recall_score(y_test, predictions), 6),
        "inference_latency_1_row_ms_median": milliseconds(float(np.median(latency_runs))),
        "inference_1000_rows_seconds": round(batch_seconds, 6),
        "inference_throughput_rows_per_second": round(len(batch) / batch_seconds, 2),
    }

    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved results to {RESULT_PATH}")


if __name__ == "__main__":
    main()
