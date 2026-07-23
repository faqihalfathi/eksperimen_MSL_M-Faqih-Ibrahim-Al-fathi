"""
Script untuk generate California Housing raw dataset ke CSV.
Jalankan sekali untuk membuat file dataset.
"""
from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def generate_raw_dataset():
    """Download dan simpan California Housing dataset sebagai CSV."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame  # sudah termasuk target column 'MedHouseVal'
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "california_housing_raw")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "california_housing.csv")
    df.to_csv(output_path, index=False)
    print(f"[INFO] Raw dataset berhasil disimpan ke: {output_path}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return output_path

if __name__ == "__main__":
    generate_raw_dataset()
