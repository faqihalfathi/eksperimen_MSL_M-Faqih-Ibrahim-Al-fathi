import requests
import pandas as pd
import time
import os
import random

PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Membangun_model', 'california_housing_preprocessing')

def send_traffic():
    test_path = os.path.join(PREPROCESSED_DIR, 'test.csv')
    try:
        df = pd.read_csv(test_path)
        print(f"[INFO] Loaded test data: {df.shape}")
        
        # Ambil sampel acak 50 baris untuk di-hit pelan-pelan
        samples = df.sample(50)
        
        print("[INFO] Memulai pengiriman request ke API (http://localhost:8000/predict)...")
        for i, (_, row) in enumerate(samples.iterrows()):
            features = row.drop('MedHouseVal').to_dict()
            
            payload = {"data": [features]}
            
            try:
                res = requests.post("http://localhost:8000/predict", json=payload, timeout=2)
                if res.status_code == 200:
                    print(f"[{i+1}/50] OK - Prediksi: {res.json()['predictions'][0]:.4f}")
                else:
                    print(f"[{i+1}/50] ERROR - {res.status_code}")
            except Exception as e:
                print(f"[{i+1}/50] FAILED - {e}")
                
            time.sleep(random.uniform(0.5, 2.0))
            
        print("[INFO] Selesai mengirim traffic!")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    send_traffic()
