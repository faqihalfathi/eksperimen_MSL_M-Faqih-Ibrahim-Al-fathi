"""
Prometheus Exporter - California Housing Model Monitoring
Nama: M. Faqih Ibrahim Al-fathi

Custom Prometheus exporter yang memonitor model ML yang di-serve.
Mengekspos 10+ metriks untuk monitoring Prometheus dan Grafana.
"""

import time
import random
import requests
import psutil
import threading
from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    REGISTRY,
    CollectorRegistry
)

# ============================================================
# Konfigurasi
# ============================================================
MODEL_SERVE_URL = "http://127.0.0.1:5001/invocations"  # MLflow model serve URL
EXPORTER_PORT = 8000  # Port untuk Prometheus scraper
SCRAPE_INTERVAL = 15  # Interval pengambilan metrik sistem (detik)

# ============================================================
# Definisi Metriks (10+ metriks)
# ============================================================

# 1. Total prediction request count
prediction_request_total = Counter(
    'prediction_request_total',
    'Total jumlah request prediksi yang diterima'
)

# 2. Prediction request errors
prediction_error_total = Counter(
    'prediction_error_total',
    'Total jumlah error pada request prediksi'
)

# 3. Prediction latency (histogram)
prediction_latency_seconds = Histogram(
    'prediction_latency_seconds',
    'Latensi prediksi dalam detik',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# 4. Prediction value distribution
prediction_value = Histogram(
    'prediction_value',
    'Distribusi nilai prediksi (harga rumah)',
    buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
)

# 5. CPU usage
cpu_usage_percent = Gauge(
    'cpu_usage_percent',
    'Persentase penggunaan CPU'
)

# 6. Memory usage
memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Penggunaan memory dalam bytes'
)

# 7. Memory usage percentage
memory_usage_percent = Gauge(
    'memory_usage_percent',
    'Persentase penggunaan memory'
)

# 8. Disk usage
disk_usage_percent = Gauge(
    'disk_usage_percent',
    'Persentase penggunaan disk'
)

# 9. Request size (summary)
request_size_bytes = Summary(
    'request_size_bytes',
    'Ukuran request dalam bytes'
)

# 10. Active predictions in progress
active_predictions = Gauge(
    'active_predictions_in_progress',
    'Jumlah prediksi yang sedang berjalan'
)

# 11. Model uptime
model_uptime_seconds = Gauge(
    'model_uptime_seconds',
    'Waktu uptime model dalam detik'
)

# 12. Successful predictions total
prediction_success_total = Counter(
    'prediction_success_total',
    'Total prediksi yang berhasil'
)

# 13. Response time percentiles (summary)
response_time_summary = Summary(
    'response_time_seconds',
    'Summary waktu respons dalam detik'
)

# 14. Network bytes sent
network_bytes_sent = Gauge(
    'network_bytes_sent_total',
    'Total bytes yang dikirim melalui network'
)

# 15. Network bytes received
network_bytes_recv = Gauge(
    'network_bytes_recv_total',
    'Total bytes yang diterima melalui network'
)

# Model info
model_info = Info(
    'model',
    'Informasi tentang model yang di-serve'
)

# ============================================================
# Global
# ============================================================
start_time = time.time()


def update_system_metrics():
    """Update metriks sistem (CPU, memory, disk, network) secara periodik."""
    while True:
        try:
            # CPU
            cpu_usage_percent.set(psutil.cpu_percent(interval=1))
            
            # Memory
            mem = psutil.virtual_memory()
            memory_usage_bytes.set(mem.used)
            memory_usage_percent.set(mem.percent)
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_usage_percent.set(disk.percent)
            
            # Network
            net = psutil.net_io_counters()
            network_bytes_sent.set(net.bytes_sent)
            network_bytes_recv.set(net.bytes_recv)
            
            # Uptime
            model_uptime_seconds.set(time.time() - start_time)
            
        except Exception as e:
            print(f"[ERROR] Gagal update system metrics: {e}")
        
        time.sleep(SCRAPE_INTERVAL)


def make_prediction(input_data=None):
    """
    Melakukan request prediksi ke model yang di-serve.
    Merekam metrik terkait.
    
    Args:
        input_data: Data input untuk prediksi. Jika None, gunakan sampel default.
    """
    if input_data is None:
        # Sampel data default (11 fitur setelah feature engineering + scaling)
        input_data = {
            "dataframe_split": {
                "columns": [
                    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude",
                    "RoomsPerHousehold", "BedroomRatio", "PopPerHousehold"
                ],
                "data": [[
                    0.5, -0.3, 0.1, -0.2, 0.4, -0.1, 0.8, -0.7,
                    0.2, -0.15, 0.3
                ]]
            }
        }
    
    prediction_request_total.inc()
    active_predictions.inc()
    
    request_str = str(input_data)
    request_size_bytes.observe(len(request_str.encode('utf-8')))
    
    start = time.time()
    
    try:
        response = requests.post(
            MODEL_SERVE_URL,
            json=input_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        latency = time.time() - start
        prediction_latency_seconds.observe(latency)
        response_time_summary.observe(latency)
        
        if response.status_code == 200:
            prediction_success_total.inc()
            result = response.json()
            
            # Log prediction value
            if isinstance(result, dict) and 'predictions' in result:
                for pred in result['predictions']:
                    prediction_value.observe(float(pred))
            elif isinstance(result, list):
                for pred in result:
                    prediction_value.observe(float(pred))
            
            print(f"[INFO] Prediksi berhasil - Latency: {latency:.4f}s - Result: {result}")
            return result
        else:
            prediction_error_total.inc()
            print(f"[ERROR] Prediksi gagal - Status: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        prediction_error_total.inc()
        latency = time.time() - start
        prediction_latency_seconds.observe(latency)
        response_time_summary.observe(latency)
        print(f"[ERROR] Tidak dapat terhubung ke model server di {MODEL_SERVE_URL}")
        return None
    except Exception as e:
        prediction_error_total.inc()
        latency = time.time() - start
        prediction_latency_seconds.observe(latency)
        response_time_summary.observe(latency)
        print(f"[ERROR] Error saat prediksi: {e}")
        return None
    finally:
        active_predictions.dec()


def simulate_traffic(interval=5, num_requests=None):
    """
    Simulasi traffic prediksi untuk demo monitoring.
    
    Args:
        interval: Interval antar request (detik)
        num_requests: Jumlah request. None = unlimited.
    """
    count = 0
    print(f"\n[INFO] Mulai simulasi traffic (interval: {interval}s)...")
    
    while num_requests is None or count < num_requests:
        # Generate random input
        input_data = {
            "dataframe_split": {
                "columns": [
                    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude",
                    "RoomsPerHousehold", "BedroomRatio", "PopPerHousehold"
                ],
                "data": [[
                    random.uniform(-2, 3),
                    random.uniform(-2, 2),
                    random.uniform(-1, 2),
                    random.uniform(-1, 1),
                    random.uniform(-1, 3),
                    random.uniform(-1, 1),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(-1, 2),
                    random.uniform(-1, 1),
                    random.uniform(-1, 2)
                ]]
            }
        }
        
        make_prediction(input_data)
        count += 1
        time.sleep(interval)


def main():
    """Menjalankan Prometheus exporter dan simulasi traffic."""
    print("=" * 60)
    print("PROMETHEUS EXPORTER - California Housing Model")
    print("Nama: M. Faqih Ibrahim Al-fathi")
    print("=" * 60)
    
    # Set model info
    model_info.info({
        'name': 'California Housing Predictor',
        'version': '1.0',
        'author': 'M. Faqih Ibrahim Al-fathi',
        'framework': 'scikit-learn',
        'type': 'RandomForestRegressor'
    })
    
    # Start Prometheus HTTP server
    print(f"\n[INFO] Starting Prometheus exporter on port {EXPORTER_PORT}...")
    start_http_server(EXPORTER_PORT)
    print(f"[INFO] Prometheus metrics tersedia di: http://localhost:{EXPORTER_PORT}/metrics")
    
    # Start system metrics thread
    system_thread = threading.Thread(target=update_system_metrics, daemon=True)
    system_thread.start()
    print("[INFO] System metrics monitoring dimulai")
    
    # Start traffic simulation
    print(f"\n[INFO] Model serve URL: {MODEL_SERVE_URL}")
    print("[INFO] Pastikan model sudah di-serve sebelum menjalankan simulasi")
    print("[INFO] Jalankan: mlflow models serve -m <model_path> -p 5001 --no-conda")
    
    try:
        simulate_traffic(interval=10)
    except KeyboardInterrupt:
        print("\n[INFO] Exporter dihentikan.")


if __name__ == "__main__":
    main()
