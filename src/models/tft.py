"""
Temporal Fusion Transformer (TFT) para predicción de demanda eléctrica.
Modelo estado del arte para time series con interpretabilidad.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae, smape

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.preprocessing import infer_time_frequency, load_processed_data

# Rutas
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class EnergyTFT:
    """Wrapper para el modelo TFT de predicción de demanda eléctrica."""
    
    def __init__(
        self,
        prediction_horizon: int = 24,
        context_length: int = 168,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 32,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        max_epochs: int = 100,
        use_wandb: bool = False,
    ):
        self.prediction_horizon = prediction_horizon
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.use_wandb = use_wandb
        
        self.model = None
        self.scaler_target = Scaler()
        self.scaler_covariates = Scaler()
        
    def prepare_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepara los datos en formato Darts TimeSeries."""
        print("📊 Preparando datos para TFT...")
        
        target_col = 'total load actual'
        
        # Covariables conocidas en el futuro (temporales)
        future_cov_cols = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'is_peak'
        ]
        
        # Filtrar columnas existentes
        future_cov_cols = [c for c in future_cov_cols if c in df.columns]
        # Las covariables meteorológicas/generación futuras no están
        # disponibles en el momento real de pronosticar. Se excluyen para no
        # evaluar el TFT con información observada del futuro.
        frequency = infer_time_frequency(df.index)
        self.data_frequency = frequency
        
        print(f"   Target: {target_col}")
        print(f"   Future covariates: {len(future_cov_cols)}")
        print("   Past covariates: 0 (se excluyen para evitar usar observaciones futuras)")
        
        # Crear TimeSeries
        series_target = TimeSeries.from_dataframe(df, value_cols=target_col, freq=frequency)
        series_future = TimeSeries.from_dataframe(df, value_cols=future_cov_cols, freq=frequency)
        
        # Split temporal
        step = pd.Timedelta(hours=1) if frequency == 'h' else pd.Timedelta(days=1)
        train_end = pd.Timestamp('2017-12-31')
        val_end = pd.Timestamp('2018-06-30')
        val_start = train_end + step
        test_start = val_end + step
        
        train_target = series_target.slice(series_target.start_time(), train_end)
        val_target = series_target.slice(val_start, val_end)
        test_target = series_target.slice(test_start, series_target.end_time())
        
        train_future = series_future.slice(series_future.start_time(), train_end)
        val_future = series_future.slice(val_start, val_end)
        test_future = series_future.slice(test_start, series_future.end_time())
        
        # Escalar
        train_target_scaled = self.scaler_target.fit_transform(train_target)
        val_target_scaled = self.scaler_target.transform(val_target)
        test_target_scaled = self.scaler_target.transform(test_target)
        
        print(f"   Train: {len(train_target)} timestamps")
        print(f"   Val: {len(val_target)} timestamps")
        print(f"   Test: {len(test_target)} timestamps")
        
        return {
            'train_target': train_target_scaled,
            'val_target': val_target_scaled,
            'test_target': test_target_scaled,
            'train_future': train_future,
            'val_future': val_future,
            'test_future': test_future,
            'test_target_original': test_target,
        }
    
    def build_model(self):
        """Construye el modelo TFT."""
        print("🏗️  Construyendo modelo TFT...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
            ),
        ]
        
        # Logger
        pl_trainer_kwargs = {
            "callbacks": callbacks,
            "enable_progress_bar": True,
        }
        
        if self.use_wandb:
            wandb_logger = WandbLogger(project="energia-predictor-es", log_model=True)
            pl_trainer_kwargs["logger"] = wandb_logger
        
        self.model = TFTModel(
            input_chunk_length=self.context_length,
            output_chunk_length=self.prediction_horizon,
            hidden_size=self.hidden_size,
            lstm_layers=1,
            num_attention_heads=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            batch_size=self.batch_size,
            n_epochs=self.max_epochs,
            add_relative_index=True,
            optimizer_kwargs={"lr": self.learning_rate},
            pl_trainer_kwargs=pl_trainer_kwargs,
            random_state=42,
            force_reset=True,
        )
        
        print(f"   ✅ Modelo TFT creado")
        unit = "h" if getattr(self, "data_frequency", "h") == "h" else "d"
        print(f"   - Input length: {self.context_length}{unit}")
        print(f"   - Output length: {self.prediction_horizon}{unit}")
        print(f"   - Hidden size: {self.hidden_size}")
        
    def train(self, data: Dict[str, Any]):
        """Entrena el modelo TFT."""
        print("🚀 Entrenando TFT...")
        
        self.model.fit(
            series=data['train_target'],
            future_covariates=data['train_future'],
            val_series=data['val_target'],
            val_future_covariates=data['val_future'],
            verbose=True,
        )
        
        print("   ✅ Entrenamiento completado")
        
    def predict(self, data: Dict[str, Any], n: int = None) -> TimeSeries:
        """Genera predicciones."""
        if n is None:
            n = self.prediction_horizon
            
        history = data['train_target'].append(data['val_target'])
        future_covariates = data['train_future'].append(data['val_future']).append(
            data['test_future']
        )
        predictions = self.model.predict(
            n=n,
            series=history,
            future_covariates=future_covariates,
        )
        
        # Desescalar
        predictions = self.scaler_target.inverse_transform(predictions)
        
        return predictions
    
    def evaluate(self, predictions: TimeSeries, actual: TimeSeries) -> Dict[str, float]:
        """Evalúa las predicciones."""
        # Alinear series
        common_start = max(predictions.start_time(), actual.start_time())
        common_end = min(predictions.end_time(), actual.end_time())
        
        pred_aligned = predictions.slice(common_start, common_end)
        actual_aligned = actual.slice(common_start, common_end)
        
        metrics = {
            'mape': mape(actual_aligned, pred_aligned),
            'rmse': rmse(actual_aligned, pred_aligned),
            'mae': mae(actual_aligned, pred_aligned),
            'smape': smape(actual_aligned, pred_aligned),
        }
        
        return metrics
    
    def save(self, path: Path = None):
        """Guarda el modelo."""
        if path is None:
            path = MODELS_DIR / "tft_model"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path / "model.pt"))
        print(f"✅ Modelo guardado: {path}")
        
    def load(self, path: Path = None):
        """Carga el modelo."""
        if path is None:
            path = MODELS_DIR / "tft_model"
        self.model = TFTModel.load(str(path / "model.pt"))
        print(f"✅ Modelo cargado: {path}")


def main():
    """Pipeline completo de entrenamiento TFT."""
    print("=" * 60)
    print("⚡ TEMPORAL FUSION TRANSFORMER - EnergiaPredictorES")
    print("=" * 60)
    
    # Cargar datos
    df = load_processed_data()
    
    # Crear modelo
    tft = EnergyTFT(
        prediction_horizon=24,
        context_length=168,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        learning_rate=0.001,
        batch_size=128,
        max_epochs=50,  # Reducido para prueba inicial
        use_wandb=False,
    )
    
    # Preparar datos
    data = tft.prepare_data(df)
    
    # Construir y entrenar
    tft.build_model()
    tft.train(data)
    
    # Predecir y evaluar
    print("\n📈 Evaluando en test set...")
    predictions = tft.predict(data, n=len(data['test_target']))
    metrics = tft.evaluate(predictions, data['test_target_original'])
    
    print("\n" + "=" * 60)
    print("📊 RESULTADOS TFT EN TEST SET")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"   {metric.upper()}: {value:.4f}")
    
    # Guardar modelo
    tft.save()
    
    return tft, metrics


if __name__ == "__main__":
    tft, metrics = main()
