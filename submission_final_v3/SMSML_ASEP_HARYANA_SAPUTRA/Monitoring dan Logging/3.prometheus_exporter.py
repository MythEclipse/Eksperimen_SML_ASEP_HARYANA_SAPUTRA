"""
prometheus_exporter.py — Model Monitoring Prometheus Exporter
Poker Hand Classification — ASEP HARYANA SAPUTRA

Kriteria 4 Phase 2 (Skilled):
Exposes ≥5 custom Prometheus metrics for model serving monitoring.

Metrics exposed on http://localhost:8000/metrics:
  1.  model_request_count_total            (Counter)
  2.  model_response_latency_seconds       (Histogram)
  3.  model_prediction_class_total         (Counter, label: class_label)
  4.  model_input_feature_drift_gauge      (Gauge) — mean absolute deviation vs baseline
  5.  model_errors_total                   (Counter, label: error_type)
  6.  model_confidence_score_histogram     (Histogram) — max softmax probability
  7.  model_active_connections_gauge       (Gauge)
  8.  model_batch_size_histogram           (Histogram)
  9.  model_data_drift_score_gauge         (Gauge) — aggregate PSI score
  10. model_uptime_seconds_total           (Counter)

Usage:
    python prometheus_exporter.py
    python prometheus_exporter.py --model-uri models:/PokerHand-RandomForest/latest
    python prometheus_exporter.py --port 8000 --inference-host localhost --inference-port 5001
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
import threading
from pathlib import Path
from typing import Any

import numpy as np
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
    REGISTRY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric definitions (10 metrics — Advanced level)
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "model_request_count_total",
    "Total number of prediction requests received by the model",
    ["endpoint", "status"],
)

RESPONSE_LATENCY = Histogram(
    "model_response_latency_seconds",
    "Prediction response latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_CLASS = Counter(
    "model_prediction_class_total",
    "Number of predictions per predicted poker hand class",
    ["class_id", "class_label"],
)

FEATURE_DRIFT = Gauge(
    "model_input_feature_drift_gauge",
    "Mean absolute deviation of input features from training baseline (feature index)",
    ["feature_name"],
)

MODEL_ERRORS = Counter(
    "model_errors_total",
    "Total model serving errors",
    ["error_type"],
)

CONFIDENCE_SCORE = Histogram(
    "model_confidence_score_histogram",
    "Distribution of model confidence scores (max softmax probability)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

ACTIVE_CONNECTIONS = Gauge(
    "model_active_connections_gauge",
    "Number of currently active inference connections",
)

BATCH_SIZE = Histogram(
    "model_batch_size_histogram",
    "Distribution of inference request batch sizes",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
)

DATA_DRIFT_SCORE = Gauge(
    "model_data_drift_score_gauge",
    "Aggregate Population Stability Index (PSI) score for input drift detection",
)

UPTIME_SECONDS = Counter(
    "model_uptime_seconds_total",
    "Total seconds the model server has been running",
)

MODEL_INFO = Info(
    "model_metadata",
    "Model metadata: name, version, dataset",
)

# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "Nothing", 1: "One_Pair", 2: "Two_Pairs", 3: "Three_of_a_Kind",
    4: "Straight", 5: "Flush", 6: "Full_House", 7: "Four_of_a_Kind",
    8: "Straight_Flush", 9: "Royal_Flush",
}

# Training baseline distribution for drift detection (from EDA)
# Each value = mean of that feature in training set
BASELINE_MEANS = {
    "S1": 2.5, "S2": 2.5, "S3": 2.5, "S4": 2.5, "S5": 2.5,
    "C1": 7.0, "C2": 7.0, "C3": 7.0, "C4": 7.0, "C5": 7.0,
}


class ModelServer:
    """
    Simulates a model serving layer. In production, replace simulate_inference()
    with actual calls to the MLflow served endpoint or loaded model.
    """

    def __init__(self, model_uri: str | None = None) -> None:
        self.model = None
        self.model_uri = model_uri
        self._request_window: list[float] = []

        # Register model metadata
        MODEL_INFO.info({
            "name":    "PokerHand-RandomForest",
            "version": "1.0",
            "dataset": "Poker Hand UCI",
            "author":  "ASEP_HARYANA_SAPUTRA",
        })

        if model_uri:
            self._load_model(model_uri)

    def _load_model(self, uri: str) -> None:
        try:
            import mlflow.sklearn
            self.model = mlflow.sklearn.load_model(uri)
            log.info("Model loaded from: %s", uri)
        except Exception as exc:
            log.warning("Could not load model from %s: %s. Running in simulation mode.", uri, exc)
            MODEL_ERRORS.labels(error_type="model_load_failure").inc()

    def simulate_inference(self, batch_size: int = 1) -> dict[str, Any]:
        """
        Simulate a prediction request. Replace with real model.predict() call
        when serving behind mlflow models serve or FastAPI.
        """
        t_start = time.perf_counter()

        try:
            ACTIVE_CONNECTIONS.inc()
            REQUEST_COUNT.labels(endpoint="/invocations", status="received").inc()
            BATCH_SIZE.observe(batch_size)

            # Simulate input data
            suit_values = np.random.randint(1, 5,  size=(batch_size, 5)).astype(float)
            rank_values = np.random.randint(1, 14, size=(batch_size, 5)).astype(float)
            X = np.hstack([suit_values, rank_values])

            if self.model is not None:
                proba = self.model.predict_proba(X)
                preds = np.argmax(proba, axis=1)
            else:
                # Realistic class-weighted simulation (mirrors training distribution)
                weights = [50, 42, 5, 2, 0.4, 0.2, 0.14, 0.024, 0.02, 0.02]
                total = sum(weights)
                probs_dist = [w / total for w in weights]
                preds = np.random.choice(10, size=batch_size, p=probs_dist)
                proba = np.zeros((batch_size, 10))
                for i, p in enumerate(preds):
                    proba[i, p] = 0.6 + random.random() * 0.35
                    # distribute rest
                    remaining = 1.0 - proba[i, p]
                    other_idx = [j for j in range(10) if j != p]
                    rnd = np.random.dirichlet(np.ones(9)) * remaining
                    for j, idx in enumerate(other_idx):
                        proba[i, idx] = rnd[j]

            # Update prediction class counters
            for pred in preds:
                PREDICTION_CLASS.labels(
                    class_id=str(pred),
                    class_label=CLASS_NAMES[pred],
                ).inc()

            # Update confidence scores
            for row in proba:
                CONFIDENCE_SCORE.observe(float(np.max(row)))

            # Compute feature drift vs baseline
            col_names = [f"S{i+1}" for i in range(5)] + [f"C{i+1}" for i in range(5)]
            for i, col in enumerate(col_names):
                drift = abs(X[:, i].mean() - BASELINE_MEANS[col])
                FEATURE_DRIFT.labels(feature_name=col).set(drift)

            # Aggregate PSI placeholder (simplified)
            psi_score = float(np.random.uniform(0.01, 0.15))
            DATA_DRIFT_SCORE.set(psi_score)

            elapsed = time.perf_counter() - t_start
            RESPONSE_LATENCY.observe(elapsed)
            REQUEST_COUNT.labels(endpoint="/invocations", status="success").inc()

            return {"predictions": preds.tolist(), "latency_s": elapsed}

        except Exception as exc:
            MODEL_ERRORS.labels(error_type=type(exc).__name__).inc()
            REQUEST_COUNT.labels(endpoint="/invocations", status="error").inc()
            log.error("Inference error: %s", exc)
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()


def _uptime_ticker(stop_event: threading.Event) -> None:
    """Background thread: increment uptime counter every second."""
    while not stop_event.is_set():
        UPTIME_SECONDS.inc()
        time.sleep(1.0)


def _load_simulator(server: ModelServer, stop_event: threading.Event, rps: float = 5.0) -> None:
    """Background thread: drive synthetic load to populate metrics."""
    interval = 1.0 / rps
    while not stop_event.is_set():
        batch = random.choice([1, 1, 1, 2, 4, 8])
        try:
            result = server.simulate_inference(batch_size=batch)
        except Exception:
            pass
        time.sleep(interval + random.uniform(-0.02, 0.02))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prometheus exporter — Poker Hand model monitoring")
    parser.add_argument("--port",        type=int, default=8000, help="Metrics server port")
    parser.add_argument("--model-uri",   default=None,           help="MLflow model URI")
    parser.add_argument("--rps",         type=float, default=5.0, help="Simulated requests/sec")
    args = parser.parse_args()

    log.info("="*60)
    log.info("Poker Hand Model — Prometheus Exporter")
    log.info("  Metrics port : %d", args.port)
    log.info("  Model URI    : %s", args.model_uri or "simulation mode")
    log.info("  RPS (sim)    : %.1f", args.rps)
    log.info("="*60)

    # Start metrics server
    start_http_server(args.port)
    log.info("Metrics exposed at http://localhost:%d/metrics", args.port)

    server = ModelServer(model_uri=args.model_uri)
    stop_event = threading.Event()

    # Start background threads
    t_uptime = threading.Thread(target=_uptime_ticker, args=(stop_event,), daemon=True)
    t_load   = threading.Thread(target=_load_simulator, args=(server, stop_event, args.rps), daemon=True)
    t_uptime.start()
    t_load.start()

    log.info("Exporter running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(10)
            log.info("Exporter healthy — serving metrics on :%d", args.port)
    except KeyboardInterrupt:
        log.info("Shutting down exporter …")
        stop_event.set()


if __name__ == "__main__":
    main()
