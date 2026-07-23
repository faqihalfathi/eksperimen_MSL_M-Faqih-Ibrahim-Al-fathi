"""
Inference Script - California Housing Model
Nama: M. Faqih Ibrahim Al-fathi

Script untuk melakukan inference menggunakan model yang sudah di-serve.
Mendukung single prediction dan batch prediction.
"""

import requests
import json
import pandas as pd
import numpy as np
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Konfigurasi
# ============================================================
MODEL_SERVE_URL = "http://127.0.0.1:5001/invocations"

# Nama fitur setelah preprocessing + feature engineering
FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
    "RoomsPerHousehold", "BedroomRatio", "PopPerHousehold"
]


def predict_single(features, url=MODEL_SERVE_URL):
    """
    Melakukan prediksi untuk satu sampel.
    
    Args:
        features (list): List 11 nilai fitur (sudah di-scale).
        url (str): URL model server.
    
    Returns:
        float: Prediksi harga rumah median.
    """
    payload = {
        "dataframe_split": {
            "columns": FEATURE_NAMES,
            "data": [features]
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    latency = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        prediction = result['predictions'][0] if isinstance(result, dict) else result[0]
        print(f"[OK] Prediksi: {prediction:.4f} (latency: {latency:.4f}s)")
        return prediction
    else:
        print(f"[ERROR] Status: {response.status_code}")
        print(f"[ERROR] Response: {response.text}")
        return None


def predict_batch(data, url=MODEL_SERVE_URL):
    """
    Melakukan prediksi untuk batch data.
    
    Args:
        data (list of list): Batch data input.
        url (str): URL model server.
    
    Returns:
        list: List prediksi.
    """
    payload = {
        "dataframe_split": {
            "columns": FEATURE_NAMES,
            "data": data
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    latency = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        predictions = result['predictions'] if isinstance(result, dict) else result
        print(f"[OK] Batch prediksi: {len(predictions)} sampel (latency: {latency:.4f}s)")
        return predictions
    else:
        print(f"[ERROR] Status: {response.status_code}")
        print(f"[ERROR] Response: {response.text}")
        return None


def health_check(url=MODEL_SERVE_URL):
    """Cek apakah model server berjalan."""
    try:
        # Kirim request kecil untuk test
        payload = {
            "dataframe_split": {
                "columns": FEATURE_NAMES,
                "data": [[0.0] * len(FEATURE_NAMES)]
            }
        }
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if response.status_code == 200:
            print(f"[OK] Model server aktif di {url}")
            return True
        else:
            print(f"[WARNING] Server merespons dengan status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Tidak dapat terhubung ke {url}")
        print("[INFO] Pastikan model sudah di-serve:")
        print("       mlflow models serve -m <model_path> -p 5001 --no-conda")
        return False


def demo_inference():
    """Demo inference dengan beberapa sampel."""
    print("=" * 60)
    print("INFERENCE - California Housing Model")
    print("Nama: M. Faqih Ibrahim Al-fathi")
    print("=" * 60)
    
    # Health check
    print("\n--- Health Check ---")
    if not health_check():
        print("\n[INFO] Model server tidak tersedia. Gunakan mode offline.")
        print("[INFO] Contoh cara serve model:")
        print("  mlflow models serve -m mlruns/0/<RUN_ID>/artifacts/model -p 5001 --no-conda")
        return
    
    # Single prediction
    print("\n--- Single Prediction ---")
    sample_features = [
        1.5,    # MedInc (scaled)
        -0.3,   # HouseAge (scaled)
        0.1,    # AveRooms (scaled)
        -0.2,   # AveBedrms (scaled)
        0.4,    # Population (scaled)
        -0.1,   # AveOccup (scaled)
        0.8,    # Latitude (scaled)
        -0.7,   # Longitude (scaled)
        0.2,    # RoomsPerHousehold (scaled)
        -0.15,  # BedroomRatio (scaled)
        0.3     # PopPerHousehold (scaled)
    ]
    
    print(f"Input features: {sample_features}")
    predict_single(sample_features)
    
    # Batch prediction
    print("\n--- Batch Prediction ---")
    batch_data = [
        [1.5, -0.3, 0.1, -0.2, 0.4, -0.1, 0.8, -0.7, 0.2, -0.15, 0.3],
        [-0.5, 0.8, -0.3, 0.1, -0.2, 0.3, -0.5, 0.6, -0.1, 0.2, -0.3],
        [2.0, -1.0, 0.5, -0.5, 1.0, -0.3, 1.5, -1.2, 0.4, -0.3, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 1.5, -0.8, 0.6, -0.5, 0.8, -1.0, 1.0, -0.4, 0.5, -0.6],
    ]
    
    predictions = predict_batch(batch_data)
    if predictions:
        print("\nHasil batch prediction:")
        for i, pred in enumerate(predictions):
            print(f"  Sampel {i+1}: {pred:.4f}")
    
    print("\n[INFO] Inference selesai!")


if __name__ == "__main__":
    demo_inference()
