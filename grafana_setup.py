import requests
import json
import time

GRAFANA_URL = "http://localhost:3000"
AUTH = ('admin', 'W@m2PZJPrZ4gf2m')

def setup_grafana():
    print("Mulai setup otomatis Grafana...")
    
    # 1. Add Prometheus Data Source
    ds_payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://localhost:9090",
        "access": "proxy",
        "isDefault": True
    }
    
    res = requests.post(f"{GRAFANA_URL}/api/datasources", json=ds_payload, auth=AUTH)
    if res.status_code in [200, 409]: # 409 means it already exists
        print("[OK] Data Source Prometheus berhasil ditambahkan.")
    else:
        print(f"[ERROR] Gagal menambah data source: {res.text}")
        
    # 2. Create Dashboard
    panels = []
    metrics = [
        ("CPU Usage", "cpu_usage_percent", "timeseries"),
        ("Memory Usage %", "memory_usage_percent", "timeseries"),
        ("Memory Usage Bytes", "memory_usage_bytes", "timeseries"),
        ("Disk Usage %", "disk_usage_percent", "timeseries"),
        ("Network Sent", "network_bytes_sent_total", "timeseries"),
        ("Network Recv", "network_bytes_recv_total", "timeseries"),
        ("Active Predictions", "active_predictions_in_progress", "stat"),
        ("Total Requests", "prediction_request_total", "stat"),
        ("Total Errors", "prediction_error_total", "stat"),
        ("Model Uptime", "model_uptime_seconds", "stat"),
        ("Prediction Latency Sum", "prediction_latency_seconds_sum", "timeseries"),
        ("Prediction Latency Count", "prediction_latency_seconds_count", "timeseries"),
        ("Prediction Value Sum", "prediction_value_sum", "timeseries"),
        ("Prediction Value Count", "prediction_value_count", "timeseries"),
        ("Process CPU", "process_cpu_seconds_total", "timeseries")
    ]
    
    for i, (title, expr, ptype) in enumerate(metrics):
        panels.append({
            "title": title,
            "type": ptype,
            "gridPos": {"h": 8, "w": 8, "x": (i % 3) * 8, "y": (i // 3) * 8},
            "targets": [{"expr": expr, "refId": "A"}]
        })
        
    dashboard_payload = {
        "dashboard": {
            "title": "Grafana-M. Faqih Ibrahim Al-fathi",
            "panels": panels,
            "schemaVersion": 16,
            "tags": ["sml", "tugas-akhir"]
        },
        "overwrite": True
    }
    
    res = requests.post(f"{GRAFANA_URL}/api/dashboards/db", json=dashboard_payload, auth=AUTH)
    if res.status_code == 200:
        print("[OK] Dashboard berhasil dibuat.")
    else:
        print(f"[ERROR] Gagal membuat dashboard: {res.text}")

    print("Setup otomatis selesai!")

if __name__ == "__main__":
    setup_grafana()
