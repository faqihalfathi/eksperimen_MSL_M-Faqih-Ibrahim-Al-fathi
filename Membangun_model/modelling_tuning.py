"""
Modelling Tuning - California Housing Dataset (Advanced: Manual Logging + DagsHub)
Nama: M. Faqih Ibrahim Al-fathi

Melatih model dengan hyperparameter tuning menggunakan GridSearchCV.
Menggunakan manual logging ke DagsHub (MLflow remote tracking).
Menyertakan minimal 2 artefak tambahan di luar autolog.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, median_absolute_error
)
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Konfigurasi
# ============================================================
PREPROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'california_housing_preprocessing')
EXPERIMENT_NAME = "California_Housing_Tuning"
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')

# ============================================================
# DagsHub Configuration
# Uncomment dan isi sesuai akun DagsHub Anda
# ============================================================
import dagshub
dagshub.init(
    repo_owner='faqihalfathi',
    repo_name='eksperimen_MSL_M-Faqih-Ibrahim-Al-fathi',
    mlflow=True
)
MLFLOW_TRACKING_URI = 'https://dagshub.com/faqihalfathi/eksperimen_MSL_M-Faqih-Ibrahim-Al-fathi.mlflow'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


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


def create_feature_importance_plot(model, feature_names, save_path):
    """
    Artefak 1: Feature importance plot.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_names)))
    
    bars = ax.barh(
        range(len(feature_names)),
        importances[indices],
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance - Best Model', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Tambahkan nilai di bar
    for bar, val in zip(bars, importances[indices]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Feature importance plot disimpan: {save_path}")


def create_residual_plot(y_true, y_pred, save_path):
    """
    Artefak 2: Residual plot.
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Residual vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color='#2196F3')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residuals vs Predicted Values', fontsize=13, fontweight='bold')
    
    # Distribusi Residual
    axes[1].hist(residuals, bins=50, color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Residual Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Residual plot disimpan: {save_path}")


def create_actual_vs_predicted_plot(y_true, y_pred, save_path):
    """
    Artefak 3: Actual vs Predicted scatter plot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='#9C27B0')
    
    # Garis diagonal (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Actual vs Predicted plot disimpan: {save_path}")


def create_model_comparison_plot(results, save_path):
    """
    Artefak 4: Model comparison plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = list(results.keys())
    metrics = {
        'RMSE': [results[m]['rmse'] for m in models],
        'MAE': [results[m]['mae'] for m in models],
        'R²': [results[m]['r2'] for m in models],
    }
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        bars = ax.bar(models, values, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Model comparison plot disimpan: {save_path}")


def train_with_tuning():
    """
    Melatih model dengan hyperparameter tuning dan manual logging.
    """
    print("=" * 60)
    print("MODELLING TUNING - California Housing (Advanced)")
    print("Nama: M. Faqih Ibrahim Al-fathi")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Buat direktori artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # ============================================================
    # Model 1: RandomForestRegressor dengan GridSearchCV
    # ============================================================
    print("\n--- Model 1: RandomForest dengan Hyperparameter Tuning ---")
    
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=2),
        rf_param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=2
    )
    
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    rf_pred = rf_best.predict(X_test)
    
    # ============================================================
    # Model 2: GradientBoostingRegressor dengan GridSearchCV
    # ============================================================
    print("\n--- Model 2: GradientBoosting dengan Hyperparameter Tuning ---")
    
    gb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    }
    
    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=2
    )
    
    gb_grid.fit(X_train, y_train)
    gb_best = gb_grid.best_estimator_
    gb_pred = gb_best.predict(X_test)
    
    # ============================================================
    # Evaluasi dan perbandingan
    # ============================================================
    results = {}
    
    for name, model, y_pred in [
        ('RandomForest', rf_best, rf_pred),
        ('GradientBoosting', gb_best, gb_pred)
    ]:
        results[name] = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'explained_variance': explained_variance_score(y_test, y_pred),
            'max_error': max_error(y_test, y_pred),
            'median_ae': median_absolute_error(y_test, y_pred),
        }
    
    # Pilih model terbaik berdasarkan RMSE
    best_model_name = min(results, key=lambda x: results[x]['rmse'])
    best_model = rf_best if best_model_name == 'RandomForest' else gb_best
    best_pred = rf_pred if best_model_name == 'RandomForest' else gb_pred
    best_params = rf_grid.best_params_ if best_model_name == 'RandomForest' else gb_grid.best_params_
    best_metrics = results[best_model_name]
    
    print(f"\n[INFO] Best Model: {best_model_name}")
    
    # ============================================================
    # Manual Logging ke MLflow
    # ============================================================
    print("\n--- Manual Logging ke MLflow ---")
    
    with mlflow.start_run(run_name=f"Best_{best_model_name}_Tuned"):
        # Log parameters (manual)
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("model_type", best_model_name)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("random_state", 42)
        
        # Log metrics (manual — sama dengan autolog + tambahan)
        mlflow.log_metric("mse", best_metrics['mse'])
        mlflow.log_metric("rmse", best_metrics['rmse'])
        mlflow.log_metric("mae", best_metrics['mae'])
        mlflow.log_metric("r2_score", best_metrics['r2'])
        mlflow.log_metric("explained_variance", best_metrics['explained_variance'])
        mlflow.log_metric("max_error", best_metrics['max_error'])
        mlflow.log_metric("median_absolute_error", best_metrics['median_ae'])
        
        # Log training metrics
        y_train_pred = best_model.predict(X_train)
        mlflow.log_metric("train_rmse", np.sqrt(mean_squared_error(y_train, y_train_pred)))
        mlflow.log_metric("train_r2", r2_score(y_train, y_train_pred))
        mlflow.log_metric("train_mae", mean_absolute_error(y_train, y_train_pred))
        
        # Log model (manual)
        mlflow.sklearn.log_model(best_model, "model")
        
        # ============================================================
        # Artefak Tambahan (minimal 2, kita buat 4)
        # ============================================================
        
        # Artefak 1: Feature Importance Plot
        fi_path = os.path.join(ARTIFACTS_DIR, 'feature_importance.png')
        create_feature_importance_plot(best_model, list(X_train.columns), fi_path)
        mlflow.log_artifact(fi_path, "plots")
        
        # Artefak 2: Residual Plot
        res_path = os.path.join(ARTIFACTS_DIR, 'residual_plot.png')
        create_residual_plot(y_test, best_pred, res_path)
        mlflow.log_artifact(res_path, "plots")
        
        # Artefak 3: Actual vs Predicted Plot
        avp_path = os.path.join(ARTIFACTS_DIR, 'actual_vs_predicted.png')
        create_actual_vs_predicted_plot(y_test, best_pred, avp_path)
        mlflow.log_artifact(avp_path, "plots")
        
        # Artefak 4: Model Comparison Plot
        comp_path = os.path.join(ARTIFACTS_DIR, 'model_comparison.png')
        create_model_comparison_plot(results, comp_path)
        mlflow.log_artifact(comp_path, "plots")
        
        # Artefak 5: Model Performance Summary (JSON)
        summary = {
            "best_model": best_model_name,
            "best_params": best_params,
            "metrics": best_metrics,
            "all_results": results,
            "feature_names": list(X_train.columns),
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
        }
        summary_path = os.path.join(ARTIFACTS_DIR, 'model_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(summary_path, "reports")
        
        # Artefak 6: GridSearch CV Results
        cv_results_rf = pd.DataFrame(rf_grid.cv_results_)[
            ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
        ].sort_values('rank_test_score')
        cv_results_gb = pd.DataFrame(gb_grid.cv_results_)[
            ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
        ].sort_values('rank_test_score')
        
        cv_path = os.path.join(ARTIFACTS_DIR, 'cv_results.csv')
        cv_combined = pd.concat([
            cv_results_rf.assign(model='RandomForest'),
            cv_results_gb.assign(model='GradientBoosting')
        ])
        cv_combined.to_csv(cv_path, index=False)
        mlflow.log_artifact(cv_path, "reports")
        
        # Log tags
        mlflow.set_tag("author", "M. Faqih Ibrahim Al-fathi")
        mlflow.set_tag("dataset", "California Housing")
        mlflow.set_tag("task", "Regression")
        mlflow.set_tag("tuning_method", "GridSearchCV")
        
        run_id = mlflow.active_run().info.run_id
        
    # ============================================================
    # Print ringkasan
    # ============================================================
    print(f"\n{'=' * 60}")
    print("HASIL AKHIR")
    print(f"{'=' * 60}")
    print(f"Best Model: {best_model_name}")
    print(f"Best Params: {best_params}")
    print(f"\nMetrics:")
    for metric, value in best_metrics.items():
        print(f"  {metric}: {value:.6f}")
    print(f"\nMLflow Run ID: {run_id}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"\nArtefak yang di-log:")
    print("  1. Feature Importance Plot")
    print("  2. Residual Plot")
    print("  3. Actual vs Predicted Plot")
    print("  4. Model Comparison Plot")
    print("  5. Model Performance Summary (JSON)")
    print("  6. CV Results (CSV)")
    print(f"\n[INFO] Jalankan 'mlflow ui' untuk melihat dashboard.")
    print(f"[INFO] Atau konfigurasi DagsHub untuk tracking online.")


if __name__ == "__main__":
    train_with_tuning()
