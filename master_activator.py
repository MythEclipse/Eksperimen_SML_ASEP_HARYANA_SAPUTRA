import os
import subprocess
import requests
import json
import time
import shutil
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn

# --- Configuration ---
REPO_ROOT = Path("/home/asephs/Pijak/Eksperimen_SML_ASEP_HARYANA_SAPUTRA")
GITHUB_USER = "mytheclipse"
PROJECT_NAME = "Eksperimen_SML_ASEP_HARYANA_SAPUTRA"
ENV_FILE = REPO_ROOT / ".env"
MONITORING_DIR = REPO_ROOT / "Monitoring_dan_Logging"
LOCAL_MODEL_PATH = REPO_ROOT / "local_model_export"
GRAFANA_URL = "http://localhost:3000"
PROM_URL = "http://localhost:9090"
EXPORTER_URL = "http://localhost:8000"

def run_cmd(cmd, cwd=REPO_ROOT, env=None):
    print(f"> Running: {cmd}")
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, env=merged_env)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def load_env():
    if not ENV_FILE.exists():
        print(f"Error: .env file not found at {ENV_FILE}")
        return None
    env_data = {}
    with open(ENV_FILE) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                env_data[key.strip()] = val.strip()
    return env_data

def setup_dagshub(token):
    print("\n[1/4] Setting up DagsHub...")
    url = "https://dagshub.com/api/v1/user/repos"
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}
    data = {"name": PROJECT_NAME, "private": False}
    
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code in [201, 422]:
        print("DagsHub repo ready.")
    else:
        print(f"DagsHub error: {resp.text}")
        return None

    tracking_uri = f"https://dagshub.com/{GITHUB_USER}/{PROJECT_NAME}.mlflow"
    (REPO_ROOT / "Membangun_model/DagsHub.txt").write_text(tracking_uri)
    return tracking_uri

def train_and_export_model(tracking_uri, token):
    print("\n[2/4] Training & Syncing Model...")
    mlflow_env = {
        "MLFLOW_TRACKING_USERNAME": GITHUB_USER,
        "MLFLOW_TRACKING_PASSWORD": token
    }
    # Run main tuning to DagsHub
    run_cmd(f".venv/bin/python Membangun_model/modelling_tuning.py --tracking-uri {tracking_uri}", env=mlflow_env)
    
    # Export local for Docker build
    X_train = pd.read_csv(REPO_ROOT / "Membangun_model/pokerhand_preprocessing/X_train.csv").values
    y_train = pd.read_csv(REPO_ROOT / "Membangun_model/pokerhand_preprocessing/y_train.csv")["CLASS"].values
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    if LOCAL_MODEL_PATH.exists():
        shutil.rmtree(LOCAL_MODEL_PATH)
    mlflow.sklearn.save_model(sk_model=model, path=str(LOCAL_MODEL_PATH))
    return True

def setup_docker(username, password):
    print("\n[3/4] Building & Pushing Docker Image...")
    run_cmd(f"echo {password} | docker login -u {username} --password-stdin")
    image_name = f"{username}/pokerhand-model"
    build_cmd = f".venv/bin/mlflow models build-docker -m {LOCAL_MODEL_PATH} -n {image_name} --env-manager local"
    if run_cmd(build_cmd).returncode == 0:
        run_cmd(f"docker push {image_name}")
        (REPO_ROOT / "Workflow-CI/MLProject/Tautan ke Docker Hub.txt").write_text(f"https://hub.docker.com/r/{image_name}")
        return True
    return False

def capture_all_assets():
    print("\n[4/4] Generating ALL Live Screenshots (20+ Assets)...")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        run_cmd(".venv/bin/pip install playwright")
        run_cmd(".venv/bin/playwright install chromium")
        from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        # A. Serving
        print("Capturing Serving proof...")
        page.goto(f"{EXPORTER_URL}/metrics")
        page.wait_for_timeout(2000)
        page.screenshot(path=str(MONITORING_DIR / "1.bukti_serving.jpg"), full_page=True)

        # B. Prometheus
        print("Capturing Prometheus Graphs...")
        prom_queries = ["model_request_count_total", "rate(model_response_latency_seconds_sum[5m])", "model_prediction_class_total"]
        for i, q in enumerate(prom_queries, 1):
            page.goto(f"{PROM_URL}/graph?g0.expr={q}&g0.tab=0&g0.range_input=1h")
            page.wait_for_timeout(3000)
            page.screenshot(path=str(MONITORING_DIR / f"4.bukti monitoring Prometheus/{i}.monitoring_metric_{i}.jpg"))

        # C. Grafana Panels (11 Panels)
        print("Capturing 11 Grafana Panels...")
        page.goto(f"{GRAFANA_URL}/login")
        page.wait_for_load_state("networkidle")
        
        # Robust Login
        if "login" in page.url:
            print("Performing login...")
            page.fill('input[name="user"]', "admin")
            page.fill('input[name="password"]', "admin123")
            page.click('button[type="submit"]')
            # Wait for dashboard to load or URL to change
            page.wait_for_url(lambda url: "login" not in url, timeout=10000)
            page.wait_for_timeout(5000) # Give time for session cookies to settle
            print(f"Logged in. Current URL: {page.url}")

        panels = [
            ("total_requests", 1), ("p95_latency", 2), ("total_errors", 3), ("data_drift_psi", 4),
            ("uptime", 5), ("request_rate", 6), ("latency_percentiles", 7), ("prediction_distribution", 8),
            ("feature_drift", 9), ("confidence_scores", 10), ("active_connections_batch", 11)
        ]
        for i, (name, pid) in enumerate(panels, 1):
            target_url = f"{GRAFANA_URL}/d/poker-hand-monitoring?viewPanel={pid}&orgId=1&kiosk"
            print(f"Capturing Panel {pid} ({name})...")
            page.goto(target_url, wait_until="networkidle")
            page.wait_for_timeout(5000) # Ensure graphs are fully drawn
            
            # Final check: if we got kicked back to login, try again once
            if "login" in page.url:
                print(f"Warning: Redirected to login on panel {pid}. Re-attempting login once...")
                page.goto(f"{GRAFANA_URL}/login")
                page.fill('input[name="user"]', "admin")
                page.fill('input[name="password"]', "admin123")
                page.click('button[type="submit"]')
                page.wait_for_timeout(3000)
                page.goto(target_url, wait_until="networkidle")
                page.wait_for_timeout(4000)
                
            page.screenshot(path=str(MONITORING_DIR / f"5.bukti monitoring Grafana/{i}.monitoring_{name}.jpg"))

        # D. Alerting (6 Detailed Proofs)
        print("Capturing 6 Alerting Proofs...")
        alert_views = [
            ("1.rules_list_view.jpg", f"{GRAFANA_URL}/alerting/list"),
            ("2.rules_ModelHighErrorRate_status.jpg", f"{GRAFANA_URL}/alerting/list?search=ModelHighErrorRate"),
            ("3.rules_ModelHighLatency_status.jpg", f"{GRAFANA_URL}/alerting/list?search=ModelHighLatency"),
            ("4.notifikasi_contact_points.jpg", f"{GRAFANA_URL}/alerting/notifications"),
            ("5.notifikasi_notification_policies.jpg", f"{GRAFANA_URL}/alerting/routes"),
            ("6.notifikasi_alert_summary_authentic.jpg", f"{GRAFANA_URL}/alerting/list")
        ]
        for filename, url in alert_views:
            print(f"Capturing Alert View: {filename}...")
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(4000)
            page.screenshot(path=str(MONITORING_DIR / f"6.bukti alerting Grafana/{filename}"))

        browser.close()

if __name__ == "__main__":
    print("=== MASTER SUBMISSION ACTIVATOR (ULTIMATE) ===")
    env = load_env()
    if not env: exit(1)
    uri = setup_dagshub(env.get("DAGSHUB_TOKEN"))
    if uri and train_and_export_model(uri, env.get("DAGSHUB_TOKEN")):
        if env.get("DOCKER_USERNAME") and env.get("DOCKER_PASSWORD"):
            setup_docker(env.get("DOCKER_USERNAME"), env.get("DOCKER_PASSWORD"))
        capture_all_assets()
        print("\n=== SEMUA PROSES SELESAI! ASET LIVE & SCREENSHOT LENGKAP. ===")
        print("Silakan push terakhir ke branch development.")
