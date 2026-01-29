"""
Modelos baseline para predicción de demanda eléctrica.
Incluye: Naive, Seasonal, ARIMA, XGBoost.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import pickle
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.preprocessing import load_processed_data

# Rutas
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
    """Calcula todas las métricas de evaluación."""
    metrics = {
        'model': model_name,
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
    }
    return metrics


def prepare_data(df: pd.DataFrame, target_col: str = 'total load actual') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide los datos en train, validation y test."""
    
    # Split temporal (como en config.yaml)
    train_end = '2017-12-31'
    val_end = '2018-06-30'
    
    train = df[df.index <= train_end]
    val = df[(df.index > train_end) & (df.index <= val_end)]
    test = df[df.index > val_end]
    
    print(f"📊 Split de datos:")
    print(f"   Train: {train.index.min()} a {train.index.max()} ({len(train):,} filas)")
    print(f"   Val:   {val.index.min()} a {val.index.max()} ({len(val):,} filas)")
    print(f"   Test:  {test.index.min()} a {test.index.max()} ({len(test):,} filas)")
    
    return train, val, test


def naive_forecast(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> np.ndarray:
    """Predicción naive: último valor conocido."""
    # Para cada punto en test, usar el valor de hace 24h
    predictions = test[f'load_lag_24h'].values
    return predictions


def seasonal_naive_forecast(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> np.ndarray:
    """Predicción seasonal naive: mismo valor hace 1 semana."""
    predictions = test[f'load_lag_168h'].values
    return predictions


def train_xgboost(train: pd.DataFrame, val: pd.DataFrame, target_col: str) -> Tuple[xgb.XGBRegressor, list]:
    """Entrena modelo XGBoost."""
    
    # Seleccionar features (excluir target y columnas relacionadas)
    exclude_cols = [target_col, 'total load forecast', 'price actual', 'price day ahead']
    feature_cols = [c for c in train.columns if c not in exclude_cols and not c.startswith('load_ratio')]
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col]
    
    print(f"   Features: {len(feature_cols)}")
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model, feature_cols


def train_lightgbm(train: pd.DataFrame, val: pd.DataFrame, target_col: str) -> Tuple[lgb.LGBMRegressor, list]:
    """Entrena modelo LightGBM."""
    
    exclude_cols = [target_col, 'total load forecast', 'price actual', 'price day ahead']
    feature_cols = [c for c in train.columns if c not in exclude_cols and not c.startswith('load_ratio')]
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col]
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )
    
    return model, feature_cols


def run_baselines():
    """Ejecuta todos los modelos baseline y compara resultados."""
    print("=" * 60)
    print("⚡ MODELOS BASELINE - EnergiaPredictorES")
    print("=" * 60)
    
    # Cargar datos
    df = load_processed_data()
    target_col = 'total load actual'
    
    # Split
    train, val, test = prepare_data(df, target_col)
    
    # Valores reales
    y_test = test[target_col].values
    
    results = []
    
    # 1. Naive (lag 24h)
    print("\n🔹 Naive (h-24)...")
    pred_naive = naive_forecast(train, test, target_col)
    results.append(evaluate_predictions(y_test, pred_naive, 'Naive (h-24)'))
    
    # 2. Seasonal Naive (lag 168h)
    print("🔹 Seasonal Naive (h-168)...")
    pred_seasonal = seasonal_naive_forecast(train, test, target_col)
    results.append(evaluate_predictions(y_test, pred_seasonal, 'Seasonal Naive (h-168)'))
    
    # 3. Predicción oficial REE (si existe en los datos)
    if 'total load forecast' in test.columns:
        print("🔹 REE Oficial...")
        pred_ree = test['total load forecast'].values
        results.append(evaluate_predictions(y_test, pred_ree, 'REE Oficial'))
    
    # 4. XGBoost
    print("🔹 XGBoost...")
    xgb_model, xgb_features = train_xgboost(train, val, target_col)
    pred_xgb = xgb_model.predict(test[xgb_features])
    results.append(evaluate_predictions(y_test, pred_xgb, 'XGBoost'))
    
    # 5. LightGBM
    print("🔹 LightGBM...")
    lgb_model, lgb_features = train_lightgbm(train, val, target_col)
    pred_lgb = lgb_model.predict(test[lgb_features])
    results.append(evaluate_predictions(y_test, pred_lgb, 'LightGBM'))
    
    # Tabla de resultados
    print("\n" + "=" * 60)
    print("📊 RESULTADOS EN TEST SET")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mape')
    
    print(results_df.to_string(index=False))
    
    # Guardar mejor modelo
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    best_model_name = results_df.iloc[0]['model']
    if 'XGBoost' in best_model_name:
        model_path = MODELS_DIR / "baseline_xgboost.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({'model': xgb_model, 'features': xgb_features}, f)
    else:
        model_path = MODELS_DIR / "baseline_lightgbm.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({'model': lgb_model, 'features': lgb_features}, f)
    
    print(f"\n✅ Mejor modelo guardado: {model_path}")
    
    # Guardar resultados
    results_path = MODELS_DIR / "baseline_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✅ Resultados guardados: {results_path}")
    
    return results_df


if __name__ == "__main__":
    results = run_baselines()
