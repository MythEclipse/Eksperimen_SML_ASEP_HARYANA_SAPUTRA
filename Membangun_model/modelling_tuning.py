"""
modelling_tuning.py — Hyperparameter Tuning with Manual MLflow Logging
Poker Hand Classification — ASEP HARYANA SAPUTRA

Kriteria 2 Phase 2 (Skilled):
- Disable autolog
- GridSearchCV on RandomForestClassifier
- Manual mlflow.log_param / log_metric / log_model
- Log confusion matrix PNG + classification_report.txt as artifacts

Usage:
    python modelling_tuning.py
    python modelling_tuning.py --preprocessed-dir ../preprocessing/pokerhand_preprocessing
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
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
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
    d = Path(preprocessed_dir)
    X_train = pd.read_csv(d / "X_train.csv").values
    X_test  = pd.read_csv(d / "X_test.csv").values
    y_train = pd.read_csv(d / "y_train.csv")["CLASS"].values
    y_test  = pd.read_csv(d / "y_test.csv")["CLASS"].values
    log.info("Loaded: X_train=%s, X_test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def _log_confusion_matrix(cm: np.ndarray, run_name: str) -> str:
    """Render confusion matrix to PNG and return temp path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[CLASS_NAMES[i] for i in range(10)],
        yticklabels=[CLASS_NAMES[i] for i in range(10)],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = f"/tmp/{run_name.replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_tuning(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    experiment_name: str,
    tracking_uri: str,
) -> None:
    # CRITICAL: autolog must be disabled for Phase 2 (Skilled)
    mlflow.sklearn.autolog(disable=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # Grid search parameter space
    # Balanced between coverage and runtime
    # ------------------------------------------------------------------
    param_grid = {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [10, 15, 20],
        "min_samples_split": [2, 5],
    }

    base_model = RandomForestClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    log.info("Starting GridSearchCV: %d combinations × %d folds …",
             len(param_grid["n_estimators"]) * len(param_grid["max_depth"]) * len(param_grid["min_samples_split"]),
             cv.n_splits)

    t0 = time.time()
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    log.info("GridSearchCV done in %.1f s", elapsed)
    log.info("Best params : %s", best_params)
    log.info("Best CV F1  : %.4f", best_cv_score)

    # ------------------------------------------------------------------
    # MLflow run — manual logging only
    # ------------------------------------------------------------------
    with mlflow.start_run(run_name="RandomForest_GridSearchCV") as run:
        log.info("MLflow run: %s", run.info.run_id)

        # ---------- Tags ----------
        mlflow.set_tag("author",    "ASEP HARYANA SAPUTRA")
        mlflow.set_tag("dataset",   "Poker Hand UCI")
        mlflow.set_tag("criterion", "Kriteria2-Skilled")
        mlflow.set_tag("search_strategy", "GridSearchCV")

        # ---------- Log all grid params ----------
        mlflow.log_param("cv_folds",              cv.n_splits)
        mlflow.log_param("scoring",               "f1_weighted")
        mlflow.log_param("grid_n_estimators",     param_grid["n_estimators"])
        mlflow.log_param("grid_max_depth",        param_grid["max_depth"])
        mlflow.log_param("grid_min_samples_split", param_grid["min_samples_split"])
        mlflow.log_param("grid_search_runtime_s", round(elapsed, 2))

        # ---------- Log best params ----------
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

        # ---------- Evaluate on held-out test set ----------
        y_pred       = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

        acc       = accuracy_score(y_test, y_pred)
        f1_w      = f1_score(y_test, y_pred, average="weighted")
        f1_macro  = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
        except ValueError:
            roc_auc = float("nan")

        # ---------- Log all metrics (same set as autolog phase) ----------
        mlflow.log_metric("accuracy",              acc)
        mlflow.log_metric("f1_weighted",           f1_w)
        mlflow.log_metric("f1_macro",              f1_macro)
        mlflow.log_metric("precision_weighted",    precision)
        mlflow.log_metric("recall_weighted",       recall)
        mlflow.log_metric("roc_auc_weighted",      roc_auc)
        mlflow.log_metric("cv_best_f1_weighted",   best_cv_score)
        mlflow.log_metric("grid_search_runtime_s", elapsed)

        log.info("Test set metrics — acc=%.4f f1_w=%.4f f1_macro=%.4f", acc, f1_w, f1_macro)

        # ---------- Log model ----------
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
            registered_model_name="PokerHand-RandomForest",
            input_example=X_test[:5],
        )
        log.info("Model logged to artifact store.")

        # ---------- Classification report ----------
        report = classification_report(
            y_test, y_pred,
            target_names=[CLASS_NAMES[i] for i in range(10)],
            digits=4,
        )
        report_path = "/tmp/classification_report.txt"
        with open(report_path, "w") as f:
            f.write("Classification Report — Poker Hand\n")
            f.write("Best params: " + str(best_params) + "\n\n")
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="reports")
        log.info("Classification report logged.")
        print("\n" + report)

        # ---------- Confusion matrix ----------
        cm = confusion_matrix(y_test, y_pred)
        cm_path = _log_confusion_matrix(cm, "RandomForest_GridSearchCV")
        mlflow.log_artifact(cm_path, artifact_path="plots")
        log.info("Confusion matrix logged.")

        # ---------- Feature importance plot ----------
        feature_names = pd.read_csv(
            Path(args_preprocessed_dir) / "X_train.csv" if False else X_train.shape  # patched below
        ) if False else None

        _log_feature_importance(best_model, X_train.shape[1])

    log.info("Tuning run complete. View with: mlflow ui --port 5000")


def _log_feature_importance(model: RandomForestClassifier, n_features: int) -> None:
    """Plot and log top feature importances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(indices)), importances[indices], align="center", color="steelblue")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([f"F{i}" for i in indices], rotation=45)
    ax.set_title("Top 20 Feature Importances — RandomForest")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Importance")
    plt.tight_layout()

    path = "/tmp/feature_importance.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="plots")
    log.info("Feature importance plot logged.")


def main() -> None:
    global args_preprocessed_dir

    parser = argparse.ArgumentParser(description="MLflow manual logging tuning — Poker Hand")
    parser.add_argument(
        "--preprocessed-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "preprocessing", "pokerhand_preprocessing"),
    )
    parser.add_argument("--experiment-name", default="poker-hand-tuning")
    parser.add_argument("--tracking-uri",    default="mlruns")
    args = parser.parse_args()

    args_preprocessed_dir = args.preprocessed_dir

    X_train, X_test, y_train, y_test = load_preprocessed(args.preprocessed_dir)
    run_tuning(X_train, X_test, y_train, y_test, args.experiment_name, args.tracking_uri)


if __name__ == "__main__":
    main()
