"""
Prometheus Exporter & Model API - California Housing Model
Nama: M. Faqih Ibrahim Al-fathi

FastAPI application yang bertindak sebagai endpoint prediksi (/predict)
dan sekaligus mengekspos metrik ke Prometheus (/metrics).
"""

import time
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)
import joblib
import pandas as pd
import os

# ============================================================
# Inisialisasi FastAPI
# ============================================================
app = FastAPI(title="California Housing Model API", version="1.0")

# ============================================================
# Definisi Metriks Prometheus
# ============================================================
prediction_request_total = Counter('prediction_request_total', 'Total request prediksi')
prediction_error_total = Counter('prediction_error_total', 'Total error prediksi')
prediction_latency_seconds = Histogram('prediction_latency_seconds', 'Latensi prediksi (s)')
prediction_value = Histogram('prediction_value', 'Distribusi nilai prediksi')
cpu_usage_percent = Gauge('cpu_usage_percent', 'CPU usage %')
memory_usage_bytes = Gauge('memory_usage_bytes', 'Memory usage bytes')
memory_usage_percent = Gauge('memory_usage_percent', 'Memory usage %')
disk_usage_percent = Gauge('disk_usage_percent', 'Disk usage %')
request_size_bytes = Summary('request_size_bytes', 'Request size bytes')
active_predictions = Gauge('active_predictions_in_progress', 'Active predictions')
model_uptime_seconds = Gauge('model_uptime_seconds', 'Model uptime (s)')
network_bytes_sent = Gauge('network_bytes_sent_total', 'Bytes sent')
network_bytes_recv = Gauge('network_bytes_recv_total', 'Bytes recv')

model_info = Info('model', 'Model info')
model_info.info({'name': 'California Housing', 'version': '1.0'})

start_time = time.time()
model = None
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Membangun_model', 'mlruns')

def load_latest_model():
    global model
    try:
        for root, dirs, files in os.walk(MODEL_DIR):
            if 'model.pkl' in files:
                model_path = os.path.join(root, 'model.pkl')
                model = joblib.load(model_path)
                print(f"[INFO] Loaded model from {model_path}")
                return
        print("[WARNING] Could not find model.pkl")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

load_latest_model()

class Features(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    RoomsPerHousehold: float
    BedroomRatio: float
    PopPerHousehold: float

class PredictRequest(BaseModel):
    data: List[Features]

def update_system_metrics():
    cpu_usage_percent.set(psutil.cpu_percent())
    mem = psutil.virtual_memory()
    memory_usage_bytes.set(mem.used)
    memory_usage_percent.set(mem.percent)
    disk = psutil.disk_usage('/')
    disk_usage_percent.set(disk.percent)
    net = psutil.net_io_counters()
    network_bytes_sent.set(net.bytes_sent)
    network_bytes_recv.set(net.bytes_recv)
    model_uptime_seconds.set(time.time() - start_time)

@app.post("/predict")
async def predict(request: Request, body: PredictRequest):
    prediction_request_total.inc()
    active_predictions.inc()
    start_req = time.time()
    body_bytes = await request.body()
    request_size_bytes.observe(len(body_bytes))
    
    try:
        df = pd.DataFrame([item.dict() for item in body.data])
        preds = model.predict(df) if model is not None else [2.5] * len(df)
        for p in preds:
            prediction_value.observe(float(p))
            
        latency = time.time() - start_req
        prediction_latency_seconds.observe(latency)
        active_predictions.dec()
        return {"predictions": preds.tolist()}
    
    except Exception as e:
        prediction_error_total.inc()
        latency = time.time() - start_req
        prediction_latency_seconds.observe(latency)
        active_predictions.dec()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    update_system_metrics()
    return PlainTextResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting FastAPI on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
