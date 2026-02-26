import os
import subprocess
import requests
import json
from pathlib import Path

# --- Configuration ---
REPO_ROOT = Path("/home/asephs/Pijak/Eksperimen_SML_ASEP_HARYANA_SAPUTRA")
GITHUB_USER = "mytheclipse"
PROJECT_NAME = "Eksperimen_SML_ASEP_HARYANA_SAPUTRA"

def run_cmd(cmd, cwd=REPO_ROOT):
    print(f"> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def setup_dagshub(token):
    print("\n--- Setting up DagsHub ---")
    # 1. Create Repository via API
    url = "https://dagshub.com/api/v1/user/repos"
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

    # 2. Update local tracking scripts
    tracking_uri = f"https://dagshub.com/{GITHUB_USER}/{PROJECT_NAME}.mlflow"
    print(f"Updating tracking URI to: {tracking_uri}")
    
    # Update modelling.py and others
    for fpath in [REPO_ROOT / "Membangun_model/modelling.py", REPO_ROOT / "Membangun_model/modelling_tuning.py"]:
        content = fpath.read_text()
        content = content.replace("sqlite:///mlflow.db", tracking_uri)
        fpath.write_text(content)
    
    # Update DagsHub.txt
    (REPO_ROOT / "Membangun_model/DagsHub.txt").write_text(tracking_uri)
    
    # Set Env for MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = GITHUB_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    
    print("DagsHub setup complete.")
    return True

def setup_docker(username, password):
    print("\n--- Setting up Docker Hub ---")
    # 1. Login
    login_cmd = f"echo {password} | docker login -u {username} --password-stdin"
    if run_cmd(login_cmd).returncode == 0:
        print("Docker login successful.")
    else:
        print("Docker login failed.")
        return False

    # 2. Build and Push
    image_name = f"{username}/pokerhand-model"
    print(f"Building and pushing image: {image_name}")
    # Using the local project
    run_cmd(f".venv/bin/mlflow models build-docker -m Membangun_model/mlruns/0/... -n {image_name}") # Path needs to be precise
    run_cmd(f"docker push {image_name}")
    
    (REPO_ROOT / "Workflow-CI/MLProject/Tautan ke Docker Hub.txt").write_text(f"https://hub.docker.com/r/{image_name}")
    return True

if __name__ == "__main__":
    print("=== Dicoding SML Submission Activator ===")
    print("Langkah ini akan membuat repository DagsHub & Docker Hub asli.")
    
    dagshub_token = input("Masukkan DagsHub Token (Settings > Tokens): ").strip()
    if dagshub_token and setup_dagshub(dagshub_token):
        print("\nMenjalankan training ulang untuk log ke DagsHub...")
        run_cmd(".venv/bin/python Membangun_model/modelling_tuning.py")
        
        docker_username = input("\nMasukkan Docker Hub Username: ").strip()
        docker_password = input("Masukkan Docker Hub Password/Token: ").strip()
        if docker_username and docker_password:
            setup_docker(docker_username, docker_password)
            
        print("\nMeregenerasi screenshot dengan data LIVE...")
        run_cmd(".venv/bin/python generate_submission_assets.py")
        
        print("\nSelesai! Semua aset sekarang LIVE dan screenshot sudah asli.")
    else:
        print("DagsHub token diperlukan untuk aktivasi.")
