"""
inference.py — Model Inference & Serving Test
Poker Hand Classification — ASEP HARYANA SAPUTRA

Kriteria 4 Phase 1:
Tests the served model endpoint. Supports both:
  (a) Direct MLflow model loading (local)
  (b) HTTP REST endpoint (mlflow models serve)

Usage:
  # Direct load:
  python inference.py --mode local --model-uri runs:/<RUN_ID>/model

  # REST endpoint (mlflow models serve must be running):
  python inference.py --mode rest --endpoint http://localhost:5001/invocations
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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

# Curated sample hands for meaningful inference tests
SAMPLE_HANDS = [
    # S1 C1 S2 C2 S3 C3 S4 C4 S5 C5  → Expected
    [1, 1, 1, 13, 1, 12, 1, 11, 1, 10],   # Royal Flush (Hearts)
    [2, 5, 2, 6,  2, 7,  2, 8,  2, 9 ],   # Straight Flush (Spades)
    [1, 7, 2, 7,  3, 7,  4, 7,  1, 2 ],   # Four of a Kind
    [1, 8, 2, 8,  3, 8,  1, 3,  2, 3 ],   # Full House
    [3, 2, 3, 5,  3, 7,  3, 9,  3, 11],   # Flush (Diamonds)
    [1, 5, 2, 6,  3, 7,  4, 8,  1, 9 ],   # Straight
    [1, 4, 2, 4,  3, 4,  4, 9,  1, 2 ],   # Three of a Kind
    [1, 6, 2, 6,  3, 9,  4, 9,  1, 3 ],   # Two Pairs
    [1, 3, 2, 3,  3, 10, 4, 7,  1, 5 ],   # One Pair
    [1, 2, 2, 5,  3, 7,  4, 9,  1, 11],   # Nothing
]

COLUMN_NAMES = ["S1","C1","S2","C2","S3","C3","S4","C4","S5","C5"]


def load_model_local(model_uri: str):
    """Load model directly from MLflow artifact store."""
    import mlflow.sklearn
    log.info("Loading model from: %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    log.info("Model loaded: %s", type(model).__name__)
    return model


def infer_local(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Run inference with a loaded sklearn model."""
    t0 = time.perf_counter()
    preds = model.predict(X)
    try:
        proba = model.predict_proba(X)
    except AttributeError:
        proba = None
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("Inference latency: %.2f ms for %d samples", latency_ms, len(X))
    return preds, proba


def infer_rest(endpoint: str, X: np.ndarray) -> list[int]:
    """
    Send inference request to mlflow models serve HTTP endpoint.
    Format: {"dataframe_split": {"columns": [...], "data": [...]}}
    """
    import urllib.request

    payload = json.dumps({
        "dataframe_split": {
            "columns": COLUMN_NAMES,
            "data": X.tolist(),
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url=endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        latency_ms = (time.perf_counter() - t0) * 1000
        body = json.loads(resp.read().decode("utf-8"))

    log.info("REST latency: %.2f ms", latency_ms)
    return body.get("predictions", body)


def print_results(X: np.ndarray, preds: np.ndarray, proba: np.ndarray | None = None) -> None:
    print("\n" + "="*70)
    print(f"{'Hand':<6} {'Predicted Class':<25} {'Confidence':>10}  {'Input'}")
    print("="*70)
    for i, (row, pred) in enumerate(zip(X, preds)):
        conf = f"{proba[i, pred]:.2%}" if proba is not None else "N/A"
        hand_str = " ".join(f"S{(j//2)+1}={row[j*2]:.0f}/C{(j//2)+1}={row[j*2+1]:.0f}" for j in range(5))
        print(f"  {i+1:<4} {CLASS_NAMES.get(int(pred), '?'):<25} {conf:>10}  [{hand_str}]")
    print("="*70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference test for Poker Hand model")
    parser.add_argument("--mode",       choices=["local", "rest"], default="local")
    parser.add_argument("--model-uri",  default=None, help="MLflow model URI (local mode)")
    parser.add_argument("--endpoint",   default="http://localhost:5001/invocations",
                        help="REST endpoint URL (rest mode)")
    parser.add_argument("--preprocessed-dir",
                        default=os.path.join(os.path.dirname(__file__), "..", "preprocessing", "pokerhand_preprocessing") if False else None,
                        help="Directory with preprocessed CSVs for bulk test")
    args = parser.parse_args()

    X = np.array(SAMPLE_HANDS, dtype=float)

    log.info("="*60)
    log.info("Poker Hand Inference Test — %d sample hands", len(X))
    log.info("="*60)

    if args.mode == "local":
        if args.model_uri is None:
            # Try to find latest run automatically
            import mlflow
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments(filter_string="name = 'poker-hand-classification'")
            if experiments:
                runs = client.search_runs(
                    experiment_ids=[experiments[0].experiment_id],
                    order_by=["start_time DESC"], max_results=1
                )
                if runs:
                    run_id = runs[0].info.run_id
                    args.model_uri = f"runs:/{run_id}/best_model"
                    log.info("Auto-detected model URI: %s", args.model_uri)
            if args.model_uri is None:
                log.error("No model URI provided and no MLflow experiment found. "
                          "Run modelling.py first, then pass --model-uri.")
                sys.exit(1)

        model = load_model_local(args.model_uri)
        preds, proba = infer_local(model, X)

    elif args.mode == "rest":
        log.info("Sending %d samples to %s", len(X), args.endpoint)
        preds = np.array(infer_rest(args.endpoint, X))
        proba = None

    print_results(X, preds, proba)


import os  # noqa: E402 — needed for os.path.join in default arg

if __name__ == "__main__":
    main()
