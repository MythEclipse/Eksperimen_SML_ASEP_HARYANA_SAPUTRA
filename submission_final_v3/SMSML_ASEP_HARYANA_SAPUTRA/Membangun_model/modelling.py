"""
modelling.py — Basic MLflow Autolog Modelling
Poker Hand Classification — ASEP HARYANA SAPUTRA

Kriteria 2 Phase 1: Basic modelling with mlflow.autolog()
Loads preprocessed data, trains RandomForestClassifier, logs everything automatically.

Usage:
    python modelling.py
    python modelling.py --preprocessed-dir ../preprocessing/pokerhand_preprocessing
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "Nothing", 1: "One Pair", 2: "Two Pairs", 3: "Three of a Kind",
    4: "Straight", 5: "Flush", 6: "Full House", 7: "Four of a Kind",
    8: "Straight Flush", 9: "Royal Flush",
}


def load_preprocessed(preprocessed_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load X_train, X_test, y_train, y_test from preprocessed CSV files."""
    d = Path(preprocessed_dir)
    required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for fname in required:
        if not (d / fname).exists():
            raise FileNotFoundError(
                f"Missing {fname} in {d}. Run automate_ASEP_HARYANA_SAPUTRA.py first."
            )

    X_train = pd.read_csv(d / "X_train.csv").values
    X_test  = pd.read_csv(d / "X_test.csv").values
    y_train = pd.read_csv(d / "y_train.csv")["CLASS"].values
    y_test  = pd.read_csv(d / "y_test.csv")["CLASS"].values

    log.info("Loaded: X_train=%s, X_test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def run_experiment(
    model,
    model_name: str,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
) -> None:
    """Train model under an active MLflow run with autolog enabled."""
    log.info("Training %s …", model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    log.info("  Accuracy : %.4f", acc)
    log.info("  F1 (weighted): %.4f", f1)

    # Classification report logged as extra artifact
    report = classification_report(
        y_test, y_pred,
        target_names=[CLASS_NAMES[i] for i in range(10)]
    )
    report_path = f"/tmp/{model_name}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path, artifact_path="reports")
    log.info("  Classification report saved as artifact.")

    # Confusion matrix heatmap
    _save_confusion_matrix_artifact(
        confusion_matrix(y_test, y_pred),
        model_name=model_name,
    )


def _save_confusion_matrix_artifact(cm: np.ndarray, model_name: str) -> None:
    """Save confusion matrix as PNG and log as MLflow artifact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[CLASS_NAMES[i] for i in range(10)],
        yticklabels=[CLASS_NAMES[i] for i in range(10)],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    path = f"/tmp/{model_name}_confusion_matrix.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="plots")
    log.info("  Confusion matrix saved as artifact.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLflow autolog modelling — Poker Hand")
    parser.add_argument(
        "--preprocessed-dir",
        default=os.path.join(os.path.dirname(__file__), "pokerhand_preprocessing"),
        help="Directory with preprocessed CSVs",
    )
    parser.add_argument(
        "--experiment-name", default="poker-hand-classification",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--tracking-uri", default="mlruns",
        help="MLflow tracking URI (local dir or remote)",
    )
    args = parser.parse_args()

    # MLflow setup
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    X_train, X_test, y_train, y_test = load_preprocessed(args.preprocessed_dir)

    # -----------------------------------------------------------------------
    # Run 1: RandomForestClassifier with autolog
    # -----------------------------------------------------------------------
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    with mlflow.start_run(run_name="RandomForest_autolog") as run:
        log.info("MLflow run started: %s", run.info.run_id)
        mlflow.set_tag("author", "ASEP HARYANA SAPUTRA")
        mlflow.set_tag("dataset", "Poker Hand UCI")
        mlflow.set_tag("criterion", "Kriteria2-Basic")

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        run_experiment(rf, "RandomForest", X_train, X_test, y_train, y_test)

    log.info("Run 1 (RandomForest) complete.")

    # -----------------------------------------------------------------------
    # Run 2: GradientBoosting (lighter, for comparison)
    # -----------------------------------------------------------------------
    with mlflow.start_run(run_name="GradientBoosting_autolog") as run:
        log.info("MLflow run started: %s", run.info.run_id)
        mlflow.set_tag("author", "ASEP HARYANA SAPUTRA")
        mlflow.set_tag("dataset", "Poker Hand UCI")
        mlflow.set_tag("criterion", "Kriteria2-Basic")

        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        run_experiment(gb, "GradientBoosting", X_train, X_test, y_train, y_test)

    log.info("Run 2 (GradientBoosting) complete.")
    log.info("All experiments logged. Run: mlflow ui --port 5000 --backend-store-uri %s", args.tracking_uri)


if __name__ == "__main__":
    main()
