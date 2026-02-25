"""
modelling.py — MLProject entrypoint (Kriteria 3)
Poker Hand Classification — ASEP HARYANA SAPUTRA

This is the modelling.py adapted for mlflow run . — paths are relative to this file.
Reads preprocessed data bundled in the MLProject directory.

Usage (direct):
    python modelling.py --n-estimators 200 --max-depth 15
Usage (via MLflow):
    mlflow run . -P n_estimators=200 -P max_depth=15
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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

HERE = Path(__file__).parent


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = pd.read_csv(data_dir / "X_train.csv").values
    X_test  = pd.read_csv(data_dir / "X_test.csv").values
    y_train = pd.read_csv(data_dir / "y_train.csv")["CLASS"].values
    y_test  = pd.read_csv(data_dir / "y_test.csv")["CLASS"].values
    log.info("Data loaded — X_train=%s X_test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators",    type=int,   default=200)
    parser.add_argument("--max-depth",       type=int,   default=15)
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--data-dir", default=str(HERE / "pokerhand_preprocessing"))
    args = parser.parse_args()

    with mlflow.start_run():
        # Log hyperparams
        mlflow.log_param("n_estimators",      args.n_estimators)
        mlflow.log_param("max_depth",         args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.set_tag("author",    "ASEP HARYANA SAPUTRA")
        mlflow.set_tag("dataset",   "Poker Hand UCI")
        mlflow.set_tag("criterion", "Kriteria3-MLProject")

        X_train, X_test, y_train, y_test = load_data(Path(args.data_dir))

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc       = accuracy_score(y_test, y_pred)
        f1_w      = f1_score(y_test, y_pred, average="weighted")
        f1_macro  = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_metric("accuracy",           acc)
        mlflow.log_metric("f1_weighted",        f1_w)
        mlflow.log_metric("f1_macro",           f1_macro)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted",    recall)

        log.info("Metrics — acc=%.4f f1_w=%.4f f1_macro=%.4f", acc, f1_w, f1_macro)

        # Confusion matrix artifact
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[CLASS_NAMES[i] for i in range(10)],
                    yticklabels=[CLASS_NAMES[i] for i in range(10)], ax=ax)
        ax.set_title("Confusion Matrix — RandomForest (MLProject)")
        plt.tight_layout()
        cm_path = "/tmp/confusion_matrix.png"
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=[CLASS_NAMES[i] for i in range(10)], digits=4
        )
        report_path = "/tmp/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="reports")
        print(report)

        # Log model
        mlflow.sklearn.log_model(model, "model")
        log.info("Model logged. Run complete.")


if __name__ == "__main__":
    main()
