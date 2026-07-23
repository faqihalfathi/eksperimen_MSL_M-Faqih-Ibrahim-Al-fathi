"""
Automate Preprocessing - California Housing Dataset
Nama: M. Faqih Ibrahim Al-fathi

Script ini mengotomatisasi seluruh tahapan preprocessing
yang telah dilakukan pada notebook eksperimen.
Mengembalikan data yang siap dilatih (train dan test set).
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Konfigurasi
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'california_housing_preprocessing')
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'california_housing_raw')


def load_data():
    """
    Load California Housing dataset.
    Mengecek apakah raw CSV sudah ada, jika tidak, download dari sklearn.
    
    Returns:
        pd.DataFrame: DataFrame dengan semua fitur dan target.
    """
    raw_path = os.path.join(RAW_DIR, 'california_housing.csv')
    
    if os.path.exists(raw_path):
        print("[INFO] Loading data dari CSV lokal...")
        df = pd.read_csv(raw_path)
    else:
        print("[INFO] Downloading data dari sklearn...")
        california = fetch_california_housing(as_frame=True)
        df = california.frame
        # Simpan raw data
        os.makedirs(RAW_DIR, exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"[INFO] Raw data disimpan ke {raw_path}")
    
    print(f"[INFO] Data loaded: {df.shape}")
    return df


def handle_duplicates(df):
    """
    Menghapus baris duplikat dari dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame tanpa duplikat.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    print(f"[INFO] Duplikat dihapus: {removed} baris (dari {before} menjadi {after})")
    return df


def handle_outliers(df, target_col='MedHouseVal'):
    """
    Handling outliers menggunakan IQR capping (winsorization).
    Hanya diterapkan pada fitur (bukan target).
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Nama kolom target.
    
    Returns:
        pd.DataFrame: DataFrame dengan outlier yang sudah di-clip.
    """
    df_processed = df.copy()
    feature_cols = [col for col in df.columns if col != target_col]
    
    for col in feature_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = len(df_processed[
            (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        ])
        
        df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        
        if outlier_count > 0:
            print(f"[INFO] {col}: {outlier_count} outliers di-clip")
    
    return df_processed


def feature_engineering(df):
    """
    Menambahkan fitur baru yang relevan.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame dengan fitur tambahan.
    """
    df_processed = df.copy()
    
    # Rooms per household age
    df_processed['RoomsPerHousehold'] = (
        df_processed['AveRooms'] / df_processed['HouseAge'].clip(lower=1)
    )
    
    # Bedroom ratio (bedrooms per total rooms)
    df_processed['BedroomRatio'] = (
        df_processed['AveBedrms'] / df_processed['AveRooms'].clip(lower=0.1)
    )
    
    # Population per household
    df_processed['PopPerHousehold'] = (
        df_processed['Population'] / df_processed['AveOccup'].clip(lower=1)
    )
    
    print("[INFO] Fitur baru ditambahkan: RoomsPerHousehold, BedroomRatio, PopPerHousehold")
    print(f"[INFO] Shape setelah feature engineering: {df_processed.shape}")
    
    return df_processed


def split_data(df, target_col='MedHouseVal'):
    """
    Memisahkan dataset menjadi train dan test set.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Nama kolom target.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"[INFO] Train set: {X_train.shape}")
    print(f"[INFO] Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Melakukan feature scaling menggunakan StandardScaler.
    Scaler di-fit pada train set dan diterapkan pada keduanya.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("[INFO] Feature scaling selesai (StandardScaler)")
    return X_train_scaled, X_test_scaled, scaler


def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler):
    """
    Menyimpan dataset preprocessed ke file CSV dan scaler ke joblib.
    
    Args:
        X_train, X_test: Scaled feature DataFrames.
        y_train, y_test: Target Series.
        scaler: Fitted StandardScaler.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Gabungkan fitur dan target
    train_data = X_train.copy()
    train_data['MedHouseVal'] = y_train.values
    
    test_data = X_test.copy()
    test_data['MedHouseVal'] = y_test.values
    
    # Full preprocessed data
    full_data = pd.concat([train_data, test_data], axis=0)
    
    # Simpan CSV
    train_data.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    full_data.to_csv(os.path.join(OUTPUT_DIR, 'california_housing_preprocessed.csv'), index=False)
    
    # Simpan scaler
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    
    print(f"[INFO] Data train disimpan: {train_data.shape}")
    print(f"[INFO] Data test disimpan: {test_data.shape}")
    print(f"[INFO] Full preprocessed disimpan: {full_data.shape}")
    print(f"[INFO] Scaler disimpan: scaler.joblib")
    print(f"[INFO] Semua file disimpan di: {OUTPUT_DIR}")


def preprocess_data():
    """
    Fungsi utama yang menjalankan seluruh pipeline preprocessing.
    Mengembalikan data yang siap dilatih.
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
    """
    print("=" * 60)
    print("AUTOMATE PREPROCESSING - California Housing Dataset")
    print("Nama: M. Faqih Ibrahim Al-fathi")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n--- Step 1: Data Loading ---")
    df = load_data()
    
    # Step 2: Handle duplicates
    print("\n--- Step 2: Handle Duplicates ---")
    df = handle_duplicates(df)
    
    # Step 3: Handle outliers
    print("\n--- Step 3: Handle Outliers ---")
    df = handle_outliers(df)
    
    # Step 4: Feature engineering
    print("\n--- Step 4: Feature Engineering ---")
    df = feature_engineering(df)
    
    # Step 5: Train-test split
    print("\n--- Step 5: Train-Test Split ---")
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Step 6: Feature scaling
    print("\n--- Step 6: Feature Scaling ---")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 7: Save preprocessed data
    print("\n--- Step 7: Simpan Data Preprocessed ---")
    save_preprocessed_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI! Data siap dilatih.")
    print("=" * 60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
