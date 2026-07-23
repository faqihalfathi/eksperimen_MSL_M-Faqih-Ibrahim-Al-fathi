"""
Modelling - California Housing Dataset (Basic: Autolog)
Nama: M. Faqih Ibrahim Al-fathi

Melatih model RandomForestRegressor menggunakan MLflow autolog.
MLflow Tracking UI disimpan secara lokal.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Konfigurasi
# ============================================================
PREPROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'california_housing_preprocessing')
EXPERIMENT_NAME = "California_Housing_Basic"


def load_preprocessed_data():
    """Load dataset yang sudah dipreprocessing."""
    train_path = os.path.join(PREPROCESSED_DIR, 'train.csv')
    test_path = os.path.join(PREPROCESSED_DIR, 'test.csv')
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    X_train = train_data.drop('MedHouseVal', axis=1)
    y_train = train_data['MedHouseVal']
    X_test = test_data.drop('MedHouseVal', axis=1)
    y_test = test_data['MedHouseVal']
    
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model():
    """Melatih model RandomForestRegressor dengan MLflow autolog."""
    print("=" * 60)
    print("MODELLING - California Housing (Basic: Autolog)")
    print("Nama: M. Faqih Ibrahim Al-fathi")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Enable autolog
    mlflow.sklearn.autolog(log_models=True)
    
    with mlflow.start_run(run_name="RandomForest_Autolog"):
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        print("\n[INFO] Training model...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n=== Hasil Evaluasi ===")
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE:  {test_rmse:.4f}")
        print(f"Test MAE:   {test_mae:.4f}")
        print(f"Test R²:    {test_r2:.4f}")
        
        print(f"\n[INFO] Run ID: {mlflow.active_run().info.run_id}")
        print(f"[INFO] Experiment: {EXPERIMENT_NAME}")
        print("[INFO] Model berhasil dilatih dan di-log ke MLflow!")
        print("[INFO] Jalankan 'mlflow ui' untuk melihat dashboard.")


if __name__ == "__main__":
    train_model()
