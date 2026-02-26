import os
import subprocess
import requests
import json
import time
from pathlib import Path

# --- Configuration ---
REPO_ROOT = Path("/home/asephs/Pijak/Eksperimen_SML_ASEP_HARYANA_SAPUTRA")
GITHUB_USER = "mytheclipse"
PROJECT_NAME = "Eksperimen_SML_ASEP_HARYANA_SAPUTRA"
ENV_FILE = REPO_ROOT / ".env"

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
        print("Please copy .env.template to .env and fill in your credentials.")
        return None
    
    env_data = {}
    with open(ENV_FILE) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                env_data[key.strip()] = val.strip()
    return env_data

def get_latest_run_id(tracking_uri):
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        # Search for runs in the tuning experiment
        runs = mlflow.search_runs(experiment_names=["poker-hand-tuning"], order_by=["start_time DESC"], max_results=1)
        if not runs.empty:
            return runs.iloc[0].run_id
    except Exception as e:
        print(f"Failed to get latest run ID via API: {e}")
    return None

def setup_dagshub(token):
    print("\n--- Setting up DagsHub ---")
    url = f"https://dagshub.com/api/v1/user/repos"
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}
    data = {"name": PROJECT_NAME, "private": False}
    
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 201:
        print(f"Successfully created DagsHub repo: {PROJECT_NAME}")
    elif resp.status_code == 422:
        print(f"DagsHub repo {PROJECT_NAME} already exists.")
    else:
        print(f"Failed to create DagsHub repo: {resp.text}")
        return False

    tracking_uri = f"https://dagshub.com/{GITHUB_USER}/{PROJECT_NAME}.mlflow"
    (REPO_ROOT / "Membangun_model/DagsHub.txt").write_text(tracking_uri)
    print(f"Tracking URI set to: {tracking_uri}")
    return tracking_uri

def setup_docker(username, password, tracking_uri, token):
    print("\n--- Setting up Docker Hub ---")
    login_cmd = f"echo {password} | docker login -u {username} --password-stdin"
    if run_cmd(login_cmd).returncode == 0:
        print("Docker login successful.")
    else:
        print("Docker login failed.")
        return False

    # Get latest run from DagsHub
    print("Fetching latest run from DagsHub...")
    os.environ["MLFLOW_TRACKING_USERNAME"] = GITHUB_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    run_id = get_latest_run_id(tracking_uri)
    
    if not run_id:
        print("Error: Could not find a valid MLflow run ID on DagsHub.")
        return False

    # The artifact path in modelling_tuning.py is "best_model"
    model_uri = f"runs:/{run_id}/best_model"
    image_name = f"{username}/pokerhand-model"
    
    print(f"Building and pushing image for run {run_id}: {image_name}")
    # Pass DagsHub tracking to build-docker so it can download the model
    env = {
        "MLFLOW_TRACKING_URI": tracking_uri,
        "MLFLOW_TRACKING_USERNAME": GITHUB_USER,
        "MLFLOW_TRACKING_PASSWORD": token
    }
    
    # We use --env-manager local to avoid conda issues if possible
    build_cmd = f".venv/bin/mlflow models build-docker -m {model_uri} -n {image_name} --env-manager local"
    if run_cmd(build_cmd, env=env).returncode == 0:
        run_cmd(f"docker push {image_name}")
        (REPO_ROOT / "Workflow-CI/MLProject/Tautan ke Docker Hub.txt").write_text(f"https://hub.docker.com/r/{image_name}")
        print("Docker push successful.")
        return True
    else:
        print("Docker build failed. Check if 'best_model' artifact exists in run.")
        return False

if __name__ == "__main__":
    print("=== Dicoding SML Submission Activator (v3 - Env Based) ===")
    
    env = load_env()
    if not env:
        exit(1)
        
    dagshub_token = env.get("DAGSHUB_TOKEN")
    docker_user   = env.get("DOCKER_USERNAME")
    docker_pass   = env.get("DOCKER_PASSWORD")

    if not dagshub_token:
        print("Error: DAGSHUB_TOKEN missing in .env")
        exit(1)

    tracking_uri = setup_dagshub(dagshub_token)
    if tracking_uri:
        print("\nMenjalankan training untuk sinkronisasi DagsHub...")
        mlflow_env = {
            "MLFLOW_TRACKING_USERNAME": GITHUB_USER,
            "MLFLOW_TRACKING_PASSWORD": dagshub_token
        }
        # Run tuning and point tracking to DagsHub
        run_cmd(f".venv/bin/python Membangun_model/modelling_tuning.py --tracking-uri {tracking_uri}", env=mlflow_env)
        
        if docker_user and docker_pass:
            setup_docker(docker_user, docker_pass, tracking_uri, dagshub_token)
        else:
            print("\nSkip Docker Hub setup (credentials missing in .env)")
            
        print("\nMeregenerasi screenshot dengan data LIVE...")
        run_cmd(".venv/bin/python generate_submission_assets.py")
        
        print("\n=== SEMUA PROSES SELESAI === ")
        print("Gunakan 'git add . && git commit' lalu push ke branch development.")
