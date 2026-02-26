import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

REPO_ROOT = Path("/home/asephs/Pijak/Eksperimen_SML_ASEP_HARYANA_SAPUTRA")
MONITORING_DIR = REPO_ROOT / "Monitoring dan Logging"
MEMBANGUN_MODEL_DIR = REPO_ROOT / "Membangun_model"
GRAFANA_URL = "http://localhost:3000"
PROM_URL = "http://localhost:9090"
EXPORTER_URL = "http://localhost:8000"

def final_capture():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})

        # Grafana Login
        print("Logging into Grafana...")
        page.goto(f"{GRAFANA_URL}/login")
        if "login" in page.url:
            page.fill('input[name="user"]', "mytheclipse")
            page.fill('input[name="password"]', "admin123")
            page.click('button[type="submit"]')
            page.wait_for_timeout(3000)

        # 1. Full Dashboard (Branded)
        print("Capturing Full Dashboard...")
        page.goto(f"{GRAFANA_URL}/d/poker-hand-monitoring?orgId=1")
        page.wait_for_timeout(7000)
        page.screenshot(path=str(MONITORING_DIR / "5.bukti monitoring Grafana/monitoring_dashboard_full.jpg"), full_page=True)

        # 2. Rule Status
        print("Capturing Rule Status...")
        page.goto(f"{GRAFANA_URL}/alerting/list")
        page.wait_for_timeout(4000)
        page.screenshot(path=str(MONITORING_DIR / "6.bukti alerting Grafana/rules_status_mytheclipse.jpg"), full_page=True)

        # 3. Notification Contact Point (Proof of superaseph@gmail.com)
        print("Capturing Contact Point (superaseph@gmail.com proof)...")
        page.goto(f"{GRAFANA_URL}/alerting/notifications")
        page.wait_for_timeout(5000)
        # Search or just screenshot the list
        page.screenshot(path=str(MONITORING_DIR / "6.bukti alerting Grafana/notifikasi_contact_point_mytheclipse.jpg"), full_page=True)

        # 4. Prometheus Targets (Node Exporter proof)
        print("Capturing Prometheus Targets (Node Exporter port 9100 proof)...")
        page.goto(f"{PROM_URL}/targets")
        page.wait_for_timeout(3000)
        page.screenshot(path=str(MONITORING_DIR / "4.bukti monitoring Prometheus/prometheus_targets_node_exporter.jpg"), full_page=True)

        # 5. Serving proof
        print("Capturing Serving Proof...")
        page.goto(f"{EXPORTER_URL}/metrics")
        page.wait_for_timeout(2000)
        page.screenshot(path=str(MONITORING_DIR / "1.bukti_serving.jpg"), full_page=True)

        browser.close()

if __name__ == "__main__":
    final_capture()
    print("Final capture complete!")
