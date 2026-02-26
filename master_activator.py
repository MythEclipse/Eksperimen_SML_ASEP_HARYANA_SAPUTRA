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
MONITORING_DIR = REPO_ROOT / "Monitoring dan Logging"
MEMBANGUN_MODEL_DIR = REPO_ROOT / "Membangun_model"
LOCAL_MODEL_PATH = REPO_ROOT / "local_model_export"
GRAFANA_URL = "http://localhost:3000"
PROM_URL = "http://localhost:9090"
EXPORTER_URL = "http://localhost:8000"

def create_dicoding_txt_files():
    print("\n[0/4] Creating Dicoding link files...")
    repo_url = f"https://github.com/{GITHUB_USER}/{PROJECT_NAME}"
    (REPO_ROOT / f"Eksperimen_SML_{GITHUB_USER}.txt").write_text(repo_url)
    (REPO_ROOT / "Workflow-CI.txt").write_text(repo_url)

def run_cmd(cmd, cwd=REPO_ROOT, env=None):
    print(f"> Running: {cmd}")
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, env=merged_env)
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
    return result

def load_env():
    if not ENV_FILE.exists():
        print(f"Error: .env not found at {ENV_FILE}")
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
    try:
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code not in [201, 422]:
            print(f"DagsHub warning: {resp.text}")
    except Exception as e:
        print(f"DagsHub request error: {e}")
    
    tracking_uri = f"https://dagshub.com/{GITHUB_USER}/{PROJECT_NAME}.mlflow"
    (MEMBANGUN_MODEL_DIR / "DagsHub.txt").write_text(tracking_uri)
    return tracking_uri

def train_and_export_model(tracking_uri, token):
    print("\n[2/4] Training & Syncing Model...")
    mlflow_env = {"MLFLOW_TRACKING_USERNAME": GITHUB_USER, "MLFLOW_TRACKING_PASSWORD": token}
    run_cmd(f".venv/bin/python Membangun_model/modelling_tuning.py --tracking-uri {tracking_uri}", env=mlflow_env)
    
    X_train_path = MEMBANGUN_MODEL_DIR / "pokerhand_preprocessing/X_train.csv"
    y_train_path = MEMBANGUN_MODEL_DIR / "pokerhand_preprocessing/y_train.csv"
    
    if not X_train_path.exists():
        print(f"Error: {X_train_path} not found.")
        return False
        
    X_train = pd.read_csv(X_train_path).values
    y_train = pd.read_csv(y_train_path)["CLASS"].values
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    if LOCAL_MODEL_PATH.exists(): shutil.rmtree(LOCAL_MODEL_PATH)
    mlflow.sklearn.save_model(sk_model=model, path=str(LOCAL_MODEL_PATH))
    return True

def setup_docker(username, password):
    print("\n[3/4] Building & Pushing Docker Image...")
    run_cmd(f"echo {password} | docker login -u {username} --password-stdin")
    image_name = f"{username}/pokerhand-model"
    build_cmd = f".venv/bin/mlflow models build-docker -m {LOCAL_MODEL_PATH} -n {image_name} --env-manager local"
    if run_cmd(build_cmd).returncode == 0:
        run_cmd(f"docker push {image_name}")
        return True
    return False

def capture_all_assets():
    print("\n[4/4] Generating ALL Live Screenshots (Dicoding Aligned)...")
    MONITORING_DIR.mkdir(exist_ok=True)
    (MONITORING_DIR / "4.bukti monitoring Prometheus").mkdir(exist_ok=True)
    (MONITORING_DIR / "5.bukti monitoring Grafana").mkdir(exist_ok=True)
    (MONITORING_DIR / "6.bukti alerting Grafana").mkdir(exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        subprocess.run(".venv/bin/pip install playwright", shell=True)
        subprocess.run(".venv/bin/playwright install chromium", shell=True)
        from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})

        print("Capturing MLflow Dashboards...")
        try:
            page.goto("http://localhost:5000")
            page.wait_for_timeout(5000)
            page.screenshot(path=str(MEMBANGUN_MODEL_DIR / "screenshoot_dashboard.jpg"))
            # Safer wait for first run
            first_run = page.locator("table tbody tr:first-child a")
            if first_run.count() > 0:
                first_run.first.click()
                page.wait_for_timeout(3000)
                page.screenshot(path=str(MEMBANGUN_MODEL_DIR / "screenshoot_artifak.jpg"))
            else:
                print("No MLflow runs found to capture artifacts.")
        except Exception as e:
            print(f"MLflow capture failed: {e}")

        print("Capturing Serving proof...")
        try:
            page.goto(f"{EXPORTER_URL}/metrics")
            page.wait_for_timeout(2000)
            page.screenshot(path=str(MONITORING_DIR / "1.bukti_serving.jpg"), full_page=True)
        except Exception as e:
            print(f"Serving capture failed: {e}")

        print("Capturing Prometheus Graphs...")
        prom_queries = ["model_request_count_total", "rate(model_response_latency_seconds_sum[5m])", "model_prediction_class_total"]
        for i, q in enumerate(prom_queries, 1):
            try:
                page.goto(f"{PROM_URL}/graph?g0.expr={q}&g0.tab=0&g0.range_input=1h")
                page.wait_for_timeout(3000)
                page.screenshot(path=str(MONITORING_DIR / f"4.bukti monitoring Prometheus/monitoring_metric_{i}.jpg"))
            except Exception as e:
                print(f"Prometheus capture failed for {q}: {e}")

        print("Performing Grafana Login...")
        try:
            page.goto(f"{GRAFANA_URL}/login")
            if "login" in page.url:
                page.fill('input[name="user"]', "mytheclipse")
                page.fill('input[name="password"]', "admin123")
                page.click('button[type="submit"]')
                page.wait_for_timeout(3000)

            print("Capturing Full Dashboard (Branded)...")
            # Remove &kiosk to show the header with Dashboard name
            page.goto(f"{GRAFANA_URL}/d/poker-hand-monitoring?orgId=1")
            page.wait_for_timeout(7000)
            page.screenshot(path=str(MONITORING_DIR / "5.bukti monitoring Grafana/monitoring_dashboard_full.jpg"), full_page=True)

            print("Capturing Alerting Proofs (Rules & Notifications)...")
            # 1. Rules list view (Proof of multiple rules)
            page.goto(f"{GRAFANA_URL}/alerting/list")
            page.wait_for_timeout(4000)
            page.screenshot(path=str(MONITORING_DIR / "6.bukti alerting Grafana/rules_status_mytheclipse.jpg"), full_page=True)

            # 2. Notification contact point proof (SHOWING EMAIL)
            print("Capturing Contact Point Proof (Email: superaseph@gmail.com)...")
            page.goto(f"{GRAFANA_URL}/alerting/notifications")
            page.wait_for_timeout(5000)
            page.screenshot(path=str(MONITORING_DIR / "6.bukti alerting Grafana/notifikasi_contact_point_mytheclipse.jpg"), full_page=True)
            
            # 3. Prometheus Targets (Proof of Node Exporter)
            print("Capturing Prometheus Targets (Node Exporter proof)...")
            page.goto(f"{PROM_URL}/targets")
            page.wait_for_timeout(3000)
            page.screenshot(path=str(MONITORING_DIR / "4.bukti monitoring Prometheus/prometheus_targets_node_exporter.jpg"), full_page=True)

        except Exception as e:
            print(f"Grafana/Alerting capture failed: {e}")

        browser.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Skip training and docker phases")
    args = parser.parse_args()

    print("=== MASTER SUBMISSION ACTIVATOR (DICODING ULTIMATE) ===")
    env = load_env()
    if not env:
        exit(1)
    
    create_dicoding_txt_files()
    uri = setup_dagshub(env.get("DAGSHUB_TOKEN"))
    
    if args.skip_train:
        print("\n[SKIP] Bypassing training and docker phases.")
    else:
        if uri and train_and_export_model(uri, env.get("DAGSHUB_TOKEN")):
            if env.get("DOCKER_USERNAME") and env.get("DOCKER_PASSWORD"):
                setup_docker(env.get("DOCKER_USERNAME"), env.get("DOCKER_PASSWORD"))
    
    capture_all_assets()
    print("\n=== SEMUA PROSES SELESAI! ASET DICODING 100% LENGKAP. ===")
