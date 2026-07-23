import requests
import json

GRAFANA_URL = "http://localhost:3000"
AUTH = ('admin', 'W@m2PZJPrZ4gf2m')

def setup_alerts():
    print("Mulai setup otomatis Alerting...")
    
    # 1. Get Prometheus Datasource UID
    res = requests.get(f"{GRAFANA_URL}/api/datasources/name/Prometheus", auth=AUTH)
    if res.status_code != 200:
        print(f"[ERROR] Gagal mendapatkan datasource: {res.text}")
        return
    ds_uid = res.json()["uid"]
    
    # 2. Create Folder for Alerts
    folder_payload = {"title": "SML Alerts"}
    res = requests.post(f"{GRAFANA_URL}/api/folders", json=folder_payload, auth=AUTH)
    if res.status_code in [200, 409]:
        folder_uid = res.json().get("uid") if res.status_code == 200 else requests.get(f"{GRAFANA_URL}/api/folders", auth=AUTH).json()[0]["uid"]
    else:
        print(f"[ERROR] Gagal membuat folder: {res.text}")
        return
        
    # 3. Create Rules
    rules = [
        ("High Error Rate", "prediction_error_total", "5"),
        ("High CPU Usage", "cpu_usage_percent", "80"),
        ("High Memory Usage", "memory_usage_percent", "70")
    ]
    
    for title, metric, threshold in rules:
        payload = {
            "title": title,
            "condition": "C",
            "data": [
                {
                    "refId": "A",
                    "datasourceUid": ds_uid,
                    "relativeTimeRange": {"from": 300, "to": 0},
                    "model": {
                        "expr": metric,
                        "refId": "A"
                    }
                },
                {
                    "refId": "C",
                    "datasourceUid": "-100", 
                    "relativeTimeRange": {"from": 300, "to": 0},
                    "model": {
                        "type": "math",
                        "expression": f"$A > {threshold}",
                        "refId": "C"
                    }
                }
            ],
            "folderUID": folder_uid,
            "ruleGroup": "Model Alerts",
            "noDataState": "NoData",
            "execErrState": "Error",
            "for": "1m"
        }
        
        res = requests.post(f"{GRAFANA_URL}/api/v1/provisioning/alert-rules", json=payload, auth=AUTH)
        if res.status_code == 201:
            print(f"[OK] Alert '{title}' berhasil dibuat.")
        else:
            print(f"[ERROR] Gagal membuat alert '{title}': {res.text}")

if __name__ == "__main__":
    setup_alerts()
