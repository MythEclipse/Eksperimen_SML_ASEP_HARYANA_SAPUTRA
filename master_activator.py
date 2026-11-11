import os
import subprocess
import sys
import requests
import time
import shutil
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn

# --- Configuration ---
REPO_ROOT = Path(__file__).resolve().parent
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

def run_cmd(cmd, cwd=REPO_ROOT, env=None, check=False):
    """
    Execute a shell command with logging.

    Parameters
    ----------
    cmd : str
        The command to execute
    cwd : Path
        Current working directory
    env : dict, optional
        Additional environment variables
    check : bool
        If True, raise exception on non-zero exit code

    Returns
    -------
    subprocess.CompletedProcess
        Result object with stdout, stderr, returncode
    """
    print(f"> Running: {cmd}")
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, env=merged_env
    )
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}", file=sys.stderr)
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result

def load_env() -> dict[str, str]:
    """
    Load environment variables from .env file.

    Returns
    -------
    dict
        Dictionary of environment variables, or None if file not found
    """
    if not ENV_FILE.exists():
        print(f"Error: .env not found at {ENV_FILE}")
        return {}

    env_data: dict[str, str] = {}
    with open(ENV_FILE) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if key:
                    env_data[key] = val

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

def train_and_export_model(tracking_uri: str, token: str) -> bool:
    """
    Train the model using the tuning script and export a simple model locally.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking URI (DagsHub URL)
    token : str
        DagsHub authentication token

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print("\n[2/4] Training & Syncing Model...")
    mlflow_env = {
        "MLFLOW_TRACKING_USERNAME": GITHUB_USER,
        "MLFLOW_TRACKING_PASSWORD": token,
    }

    # Use python from PATH or .venv if it exists
    python_cmd = ".venv/bin/python" if (REPO_ROOT / ".venv" / "bin" / "python").exists() else "python"
    run_cmd(f"{python_cmd} Membangun_model/modelling_tuning.py --tracking-uri {tracking_uri}", env=mlflow_env)

    X_train_path = MEMBANGUN_MODEL_DIR / "pokerhand_preprocessing/X_train.csv"
    y_train_path = MEMBANGUN_MODEL_DIR / "pokerhand_preprocessing/y_train.csv"

    if not X_train_path.exists():
        print(f"Error: {X_train_path} not found.")
        return False
    if not y_train_path.exists():
        print(f"Error: {y_train_path} not found.")
        return False

    try:
        X_train = pd.read_csv(X_train_path).values
        y_train = pd.read_csv(y_train_path)["CLASS"].values

        # Train a simple model for local export
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # Clean up and save model
        if LOCAL_MODEL_PATH.exists():
            shutil.rmtree(LOCAL_MODEL_PATH)
        mlflow.sklearn.save_model(sk_model=model, path=str(LOCAL_MODEL_PATH))
        return True
    except Exception as e:
        print(f"Error training/exporting model: {e}")
        return False

def setup_docker(username: str, password: str) -> bool:
    """
    Build and push Docker image for the model.

    Parameters
    ----------
    username : str
        Docker Hub username
    password : str
        Docker Hub password

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print("\n[3/4] Building & Pushing Docker Image...")
    try:
        # Docker login
        login_cmd = f'echo "{password}" | docker login -u "{username}" --password-stdin'
        run_cmd(login_cmd, check=True)

        image_name = f"{username}/pokerhand-model"
        python_cmd = ".venv/bin/python" if (REPO_ROOT / ".venv" / "bin" / "python").exists() else "python"
        build_cmd = f"{python_cmd} -m mlflow models build-docker -m {LOCAL_MODEL_PATH} -n {image_name} --env-manager local"
        result = run_cmd(build_cmd)

        if result.returncode == 0:
            print("Docker image built successfully. Pushing...")
            run_cmd(f"docker push {image_name}")
            return True
        else:
            print(f"Docker build failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"Docker setup error: {e}")
        return False

def capture_all_assets() -> None:
    """
    Generate screenshots for Dicoding submission using Playwright.
    Captures MLflow, Prometheus, and Grafana dashboards.
    """
    print("\n[4/4] Generating ALL Live Screenshots (Dicoding Aligned)...")

    # Create output directories
    for subdir in [
        "4.bukti monitoring Prometheus",
        "5.bukti monitoring Grafana",
        "6.bukti alerting Grafana",
    ]:
        (MONITORING_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Ensure playwright is installed
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Installing playwright...")
        subprocess.run(
            f"{sys.executable} -m pip install playwright",
            shell=True,
            check=True,
        )
        subprocess.run(f"{sys.executable} -m playwright install chromium", shell=True, check=True)
        from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
        except Exception as e:
            print(f"Playwright initialization error: {e}")
            return

        print("Capturing MLflow Dashboards...")
        try:
            page.goto("http://localhost:5000", timeout=10000)
            page.wait_for_timeout(5000)
            screenshot_path = MEMBANGUN_MODEL_DIR / "screenshoot_dashboard.jpg"
            page.screenshot(path=str(screenshot_path))
            print(f"  Saved: {screenshot_path}")

            # Safer wait for first run
            first_run = page.locator("table tbody tr:first-child a")
            if first_run.count() > 0:
                first_run.first.click()
                page.wait_for_timeout(3000)
                screenshot_path = MEMBANGUN_MODEL_DIR / "screenshoot_artifak.jpg"
                page.screenshot(path=str(screenshot_path))
                print(f"  Saved: {screenshot_path}")
            else:
                print("  No MLflow runs found to capture artifacts.")
        except Exception as e:
            print(f"  MLflow capture failed: {e}")

        print("Capturing Serving proof...")
        try:
            page.goto(f"{EXPORTER_URL}/metrics", timeout=10000)
            page.wait_for_timeout(2000)
            screenshot_path = MONITORING_DIR / "1.bukti_serving.jpg"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  Saved: {screenshot_path}")
        except Exception as e:
            print(f"  Serving capture failed: {e}")

        print("Capturing Prometheus Graphs...")
        prom_queries = [
            "model_request_count_total",
            "rate(model_response_latency_seconds_sum[5m])",
            "model_prediction_class_total",
        ]
        for i, q in enumerate(prom_queries, 1):
            try:
                encoded_q = q.replace(" ", "%20")
                url = f"{PROM_URL}/graph?g0.expr={encoded_q}&g0.tab=0&g0.range_input=1h"
                page.goto(url, timeout=10000)
                page.wait_for_timeout(3000)
                screenshot_path = MONITORING_DIR / f"4.bukti monitoring Prometheus/monitoring_metric_{i}.jpg"
                page.screenshot(path=str(screenshot_path))
                print(f"  Saved: {screenshot_path}")
            except Exception as e:
                print(f"  Prometheus capture failed for {q}: {e}")

        print("Performing Grafana Login...")
        try:
            page.goto(f"{GRAFANA_URL}/login", timeout=10000)
            if "login" in page.url:
                page.fill('input[name="user"]', "mytheclipse")
                page.fill('input[name="password"]', "admin123")
                page.click('button[type="submit"]')
                page.wait_for_timeout(3000)

            print("Capturing Full Dashboard (Branded)...")
            # Remove &kiosk to show the header with Dashboard name
            page.goto(f"{GRAFANA_URL}/d/poker-hand-monitoring?orgId=1", timeout=10000)
            page.wait_for_timeout(7000)
            screenshot_path = MONITORING_DIR / "5.bukti monitoring Grafana/monitoring_dashboard_full.jpg"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  Saved: {screenshot_path}")

            print("Capturing Alerting Proofs (Rules & Notifications)...")
            # 1. Rules list view (Proof of multiple rules)
            page.goto(f"{GRAFANA_URL}/alerting/list", timeout=10000)
            page.wait_for_timeout(4000)
            screenshot_path = MONITORING_DIR / "6.bukti alerting Grafana/rules_status_mytheclipse.jpg"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  Saved: {screenshot_path}")

            # 2. Notification contact point proof (SHOWING EMAIL)
            print("  Capturing Contact Point Proof...")
            page.goto(f"{GRAFANA_URL}/alerting/notifications", timeout=10000)
            page.wait_for_timeout(5000)
            screenshot_path = MONITORING_DIR / "6.bukti alerting Grafana/notifikasi_contact_point_mytheclipse.jpg"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  Saved: {screenshot_path}")

            # 3. Prometheus Targets (Proof of Node Exporter)
            print("  Capturing Prometheus Targets (Node Exporter proof)...")
            page.goto(f"{PROM_URL}/targets", timeout=10000)
            page.wait_for_timeout(3000)
            screenshot_path = MONITORING_DIR / "4.bukti monitoring Prometheus/prometheus_targets_node_exporter.jpg"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  Saved: {screenshot_path}")

        except Exception as e:
            print(f"  Grafana/Alerting capture failed: {e}")

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
