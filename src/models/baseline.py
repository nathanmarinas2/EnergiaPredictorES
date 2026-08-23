"""
Modelos baseline para predicción de demanda eléctrica.
Incluye: Naive, Seasonal, ARIMA, XGBoost.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import pickle
from datetime import datetime

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

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
        'mae': np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))),
        'rmse': np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
    }
    return metrics


def select_feature_columns(df: pd.DataFrame, target_col: str) -> list:
    """Selecciona solo variables disponibles al emitir un pronóstico diario.

    El escenario de producción es day-ahead: para predecir el día ``t`` solo
    se conocen el calendario de ``t`` y la historia hasta ``t - 1``. Las
    variables de mercado, generación y precios del mismo día se excluyen
    aunque estén presentes en un parquet antiguo, porque podrían ser
    observaciones posteriores al instante real de predicción.
    """
    safe_temporal = {
        'hour', 'day_of_week', 'day_of_month', 'month', 'year',
        'week_of_year', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'month_sin', 'month_cos', 'is_weekend', 'is_peak_morning',
        'is_peak_evening', 'is_peak', 'is_holiday',
    }
    return [
        c for c in df.columns
        if c != target_col
        and (c in safe_temporal or c.startswith('load_lag_') or c.startswith('load_rolling_'))
        and pd.api.types.is_numeric_dtype(df[c])
    ]


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
    """Predicción naive: valor observado un día antes."""
    predictions = test['load_lag_1day'].values
    return predictions


def seasonal_naive_forecast(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> np.ndarray:
    """Predicción seasonal naive: mismo valor hace una semana."""
    predictions = test['load_lag_1week'].values
    return predictions


def train_xgboost(train: pd.DataFrame, val: pd.DataFrame, target_col: str) -> Tuple[xgb.XGBRegressor, list]:
    """Entrena modelo XGBoost."""
    if xgb is None:
        raise ImportError("Instala xgboost para entrenar el modelo XGBoost")
    
    feature_cols = select_feature_columns(train, target_col)
    
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
    if lgb is None:
        raise ImportError("Instala lightgbm para entrenar el modelo LightGBM")
    
    feature_cols = select_feature_columns(train, target_col)
    
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
    
    # 1. Naive (día anterior)
    print("\n🔹 Naive (día anterior)...")
    pred_naive = naive_forecast(train, test, target_col)
    results.append(evaluate_predictions(y_test, pred_naive, 'Naive (día anterior)'))
    
    # 2. Seasonal Naive (semana anterior)
    print("🔹 Seasonal Naive (semana anterior)...")
    pred_seasonal = seasonal_naive_forecast(train, test, target_col)
    results.append(evaluate_predictions(y_test, pred_seasonal, 'Seasonal Naive (semana anterior)'))
    
    # 3. Predicción oficial REE solo si está alineada y en una escala plausible.
    if 'total load forecast' in test.columns:
        forecast = test['total load forecast']
        scale_ratio = np.nanmedian(np.abs(forecast.values)) / np.nanmedian(np.abs(y_test))
        if (
            forecast.notna().all()
            and forecast.index.equals(test.index)
            and 0.25 <= scale_ratio <= 4.0
        ):
            print("🔹 REE Oficial...")
            results.append(evaluate_predictions(y_test, forecast.values, 'REE Oficial'))
        else:
            print(
                "⚠️ REE Oficial omitida: forecast desalineado, incompleto o "
                f"con escala incompatible (ratio mediano={scale_ratio:.2f})"
            )
    
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
    elif 'LightGBM' in best_model_name:
        model_path = MODELS_DIR / "baseline_lightgbm.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({'model': lgb_model, 'features': lgb_features}, f)
    else:
        model_path = None
        print("ℹ️ El mejor resultado es un baseline; no se guarda como modelo ML.")

    if model_path is not None:
        print(f"\n✅ Mejor modelo guardado: {model_path}")
    
    # Guardar resultados
    results_path = MODELS_DIR / "baseline_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✅ Resultados guardados: {results_path}")
    
    return results_df


if __name__ == "__main__":
    results = run_baselines()
