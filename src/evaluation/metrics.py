"""
Métricas de evaluación y comparativas con predicciones oficiales de REE.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.preprocessing import load_processed_data

# Rutas
PROJECT_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula todas las métricas."""
    return {
        'MAPE (%)': mape(y_true, y_pred),
        'RMSE (MW)': rmse(y_true, y_pred),
        'MAE (MW)': mae(y_true, y_pred),
        'SMAPE (%)': smape(y_true, y_pred),
    }


def compare_with_ree(df: pd.DataFrame, model_predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compara las predicciones de nuestros modelos con las oficiales de REE.
    
    Args:
        df: DataFrame con 'total load actual' y 'total load forecast' (REE)
        model_predictions: Dict con nombre del modelo y sus predicciones
    
    Returns:
        DataFrame con métricas comparativas
    """
    print("=" * 60)
    print("📊 COMPARATIVA CON PREDICCIÓN OFICIAL REE")
    print("=" * 60)
    
    # Valores reales
    y_true = df['total load actual'].values
    
    results = []
    
    # Predicción oficial REE
    if 'total load forecast' in df.columns:
        y_ree = df['total load forecast'].values
        ree_metrics = calculate_metrics(y_true, y_ree)
        ree_metrics['Modelo'] = 'REE Oficial'
        results.append(ree_metrics)
    
    # Nuestros modelos
    for model_name, y_pred in model_predictions.items():
        if len(y_pred) == len(y_true):
            model_metrics = calculate_metrics(y_true, y_pred)
            model_metrics['Modelo'] = model_name
            results.append(model_metrics)
    
    # Crear DataFrame
    results_df = pd.DataFrame(results)
    cols = ['Modelo', 'MAPE (%)', 'RMSE (MW)', 'MAE (MW)', 'SMAPE (%)']
    results_df = results_df[cols]
    results_df = results_df.sort_values('MAPE (%)')
    
    print(results_df.to_string(index=False))
    
    # Calcular mejora vs REE
    if 'total load forecast' in df.columns:
        ree_mape = results_df[results_df['Modelo'] == 'REE Oficial']['MAPE (%)'].values[0]
        best_mape = results_df.iloc[0]['MAPE (%)']
        best_model = results_df.iloc[0]['Modelo']
        
        if best_model != 'REE Oficial':
            improvement = ((ree_mape - best_mape) / ree_mape) * 100
            print(f"\n🎯 Mejor modelo: {best_model}")
            print(f"   Mejora vs REE: {improvement:.2f}%")
    
    return results_df


def plot_predictions_comparison(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    start_date: str = None,
    end_date: str = None,
    save_path: Path = None
):
    """
    Grafica comparativa de predicciones vs valores reales.
    """
    if start_date:
        df = df[df.index >= start_date]
        for k in predictions:
            predictions[k] = predictions[k][-len(df):]
    if end_date:
        df = df[df.index <= end_date]
        for k in predictions:
            predictions[k] = predictions[k][:len(df)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Valores reales
    ax.plot(df.index, df['total load actual'], 
            label='Real', color='black', linewidth=1.5, alpha=0.8)
    
    # REE Oficial
    if 'total load forecast' in df.columns:
        ax.plot(df.index, df['total load forecast'], 
                label='REE Oficial', color='blue', linewidth=1, alpha=0.7, linestyle='--')
    
    # Nuestras predicciones
    colors = ['#e63946', '#2a9d8f', '#f4a261', '#9b5de5']
    for i, (name, pred) in enumerate(predictions.items()):
        ax.plot(df.index, pred, 
                label=name, color=colors[i % len(colors)], linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Demanda (MW)')
    ax.set_title('Comparativa de Predicciones de Demanda Eléctrica')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Gráfica guardada: {save_path}")
    
    plt.show()


def plot_error_distribution(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: Path = None
):
    """
    Grafica distribución de errores por modelo.
    """
    fig, axes = plt.subplots(1, len(predictions), figsize=(4*len(predictions), 4))
    
    if len(predictions) == 1:
        axes = [axes]
    
    for ax, (name, pred) in zip(axes, predictions.items()):
        errors = y_true - pred
        
        ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Error (MW)')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'{name}\nMean: {np.mean(errors):.0f}, Std: {np.std(errors):.0f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_error_by_hour(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    save_path: Path = None
):
    """
    Grafica error medio por hora del día.
    """
    y_true = df['total load actual'].values
    hours = df.index.hour
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = ['#e63946', '#2a9d8f', '#f4a261', '#9b5de5', 'blue']
    
    # REE
    if 'total load forecast' in df.columns:
        errors_ree = np.abs(y_true - df['total load forecast'].values)
        error_by_hour_ree = pd.Series(errors_ree).groupby(hours).mean()
        ax.plot(error_by_hour_ree.index, error_by_hour_ree.values, 
                label='REE Oficial', color='blue', marker='o', linewidth=2)
    
    # Nuestros modelos
    for i, (name, pred) in enumerate(predictions.items()):
        errors = np.abs(y_true - pred)
        error_by_hour = pd.Series(errors).groupby(hours).mean()
        ax.plot(error_by_hour.index, error_by_hour.values, 
                label=name, color=colors[i % len(colors)], marker='o', linewidth=2)
    
    ax.set_xlabel('Hora del día')
    ax.set_ylabel('MAE medio (MW)')
    ax.set_title('Error Absoluto Medio por Hora del Día')
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def generate_report(df: pd.DataFrame, predictions: Dict[str, np.ndarray]):
    """Genera reporte completo de evaluación."""
    print("=" * 60)
    print("📋 GENERANDO REPORTE DE EVALUACIÓN")
    print("=" * 60)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Tabla comparativa
    results_df = compare_with_ree(df, predictions)
    results_df.to_csv(REPORTS_DIR / "metrics_comparison.csv", index=False)
    
    # 2. Gráficas
    print("\n📈 Generando gráficas...")
    
    # Última semana de test
    last_week = df.index[-168:]
    df_week = df.loc[last_week]
    preds_week = {k: v[-168:] for k, v in predictions.items()}
    
    plot_predictions_comparison(
        df_week, preds_week,
        save_path=REPORTS_DIR / "predictions_comparison.png"
    )
    
    plot_error_by_hour(
        df, predictions,
        save_path=REPORTS_DIR / "error_by_hour.png"
    )
    
    print(f"\n✅ Reporte guardado en: {REPORTS_DIR}")


if __name__ == "__main__":
    # Ejemplo de uso
    df = load_processed_data()
    
    # Filtrar test set
    test_df = df[df.index > '2018-06-30']
    
    # Predicciones de ejemplo (lag 24h como baseline)
    predictions = {
        'Naive (h-24)': test_df['load_lag_24h'].values,
    }
    
    generate_report(test_df, predictions)
