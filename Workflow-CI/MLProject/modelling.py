"""
Modelling untuk MLflow Project - California Housing Dataset
Nama: M. Faqih Ibrahim Al-fathi

Script ini dijalankan sebagai entry point dari MLflow Project.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import json
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments dari MLflow Project."""
    parser = argparse.ArgumentParser(description='Train California Housing Model')
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--random_state', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MLflow Project - California Housing Modelling")
    print("Nama: M. Faqih Ibrahim Al-fathi")
    print("=" * 60)
    
    # Load preprocessed data
    data_dir = os.path.join(os.path.dirname(__file__), 'california_housing_preprocessing')
    train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    X_train = train_data.drop('MedHouseVal', axis=1)
    y_train = train_data['MedHouseVal']
    X_test = test_data.drop('MedHouseVal', axis=1)
    y_test = test_data['MedHouseVal']
    
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    
    # MLflow manual logging
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("model_type", "RandomForestRegressor")
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=-1
        )
        
        print("[INFO] Training model...")
        model.fit(X_train, y_train)
        
        # Predict dan evaluate
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2", train_r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Artefak 1: Feature Importance
        fig, ax = plt.subplots(figsize=(10, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        ax.barh(range(len(X_train.columns)), importances[indices], color='steelblue')
        ax.set_yticks(range(len(X_train.columns)))
        ax.set_yticklabels([X_train.columns[i] for i in indices])
        ax.set_title('Feature Importance')
        ax.invert_yaxis()
        plt.tight_layout()
        fi_path = "feature_importance.png"
        plt.savefig(fi_path, dpi=150)
        plt.close()
        mlflow.log_artifact(fi_path)
        
        # Artefak 2: Performance Summary
        summary = {
            "model": "RandomForestRegressor",
            "params": vars(args),
            "metrics": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
        }
        summary_path = "performance_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)
        
        # Tags
        mlflow.set_tag("author", "M. Faqih Ibrahim Al-fathi")
        mlflow.set_tag("dataset", "California Housing")
        
        print(f"\n=== Hasil ===")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
