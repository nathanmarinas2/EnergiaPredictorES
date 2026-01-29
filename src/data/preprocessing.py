"""
Preprocesamiento de datos de energía eléctrica de España.
Incluye limpieza, merge con datos meteorológicos y feature engineering temporal.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import holidays

# Rutas
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los datasets crudos.
    Soporta múltiples formatos:
    - spain_energy_market.csv (formato long, pivoteado a wide)
    - Kaggle CSV (energy_dataset.csv + weather_features.csv)
    - API REE Parquet (ree_data.parquet)
    """
    print("📂 Cargando datos crudos...")
    
    # Opción 1: spain_energy_market.csv (formato long)
    spain_market_path = RAW_DIR / "spain_energy_market.csv"
    if spain_market_path.exists():
        print("   Usando datos de spain_energy_market.csv...")
        df_long = pd.read_csv(spain_market_path, parse_dates=['datetime'])
        
        # Pivotar de formato long a wide
        df = df_long.pivot_table(
            index='datetime',
            columns='name',
            values='value',
            aggfunc='first'
        )
        df.columns.name = None
        
        # Renombrar columnas clave al formato estándar
        rename_map = {
            'Demanda real': 'total load actual',
            'Demanda programada PBF total': 'total load forecast',
            'Generación programada PBF Eólica': 'generation wind onshore',
            'Generación programada PBF Solar fotovoltaica': 'generation solar',
            'Generación programada PBF Nuclear': 'generation nuclear',
            'Generación programada PBF Ciclo combinado': 'generation fossil gas',
            'Generación programada PBF Carbón': 'generation fossil hard coal',
            'Generación programada PBF UGH + no UGH': 'generation hydro',
            'Precio mercado SPOT Diario ESP': 'price actual',
        }
        
        # Buscar columnas con encoding diferente
        for old_name, new_name in list(rename_map.items()):
            for col in df.columns:
                # Comparar sin acentos
                if old_name.lower().replace('í', 'i').replace('ó', 'o').replace('é', 'e') in col.lower().replace('í', 'i').replace('ó', 'o').replace('é', 'e'):
                    rename_map[col] = new_name
                    break
        
        df = df.rename(columns=rename_map)
        
        print(f"   ✅ Pivoteado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
        print(f"   Período: {df.index.min()} a {df.index.max()}")
        
        return df, None
    
    # Opción 2: REE Parquet
    ree_path = RAW_DIR / "ree_data.parquet"
    if ree_path.exists():
        print("   Usando datos de API REE...")
        df = pd.read_parquet(ree_path)
        df = df.set_index('datetime')
        return df, None
    
    # Opción 3: Kaggle CSVs
    energy_path = RAW_DIR / "energy_dataset.csv"
    weather_path = RAW_DIR / "weather_features.csv"
    
    if not energy_path.exists():
        raise FileNotFoundError(
            f"No se encontraron los datos en {RAW_DIR}. "
            "Descarga spain_energy_market.csv de Kaggle."
        )
    
    energy = pd.read_csv(energy_path, parse_dates=['time'])
    
    weather = None
    if weather_path.exists():
        weather = pd.read_csv(weather_path, parse_dates=['dt_iso'])
        print(f"   ✅ Weather: {weather.shape[0]:,} filas, {weather.shape[1]} columnas")
    
    print(f"   ✅ Energy: {energy.shape[0]:,} filas, {energy.shape[1]} columnas")
    
    return energy, weather


def clean_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset de energía."""
    print("🧹 Limpiando datos de energía...")
    
    df = df.copy()
    
    # Renombrar columna de tiempo
    df = df.rename(columns={'time': 'datetime'})
    df = df.set_index('datetime')
    
    # Columnas de interés
    cols_to_keep = [
        'total load actual',
        'total load forecast',
        'price actual',
        'price day ahead',
        # Generación por tipo
        'generation biomass',
        'generation fossil brown coal/lignite',
        'generation fossil gas',
        'generation fossil hard coal',
        'generation fossil oil',
        'generation hydro pumped storage consumption',
        'generation hydro run-of-river and poundage',
        'generation hydro water reservoir',
        'generation nuclear',
        'generation other',
        'generation other renewable',
        'generation solar',
        'generation waste',
        'generation wind onshore',
    ]
    
    # Filtrar columnas existentes
    cols_available = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_available]
    
    # Rellenar valores faltantes con interpolación
    missing_before = df.isnull().sum().sum()
    df = df.interpolate(method='time', limit=24)  # máximo 24h de interpolación
    df = df.fillna(method='ffill').fillna(method='bfill')
    missing_after = df.isnull().sum().sum()
    
    print(f"   ✅ NaN antes: {missing_before:,}, después: {missing_after}")
    
    return df


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y agrega datos meteorológicos a nivel nacional."""
    print("🌤️  Procesando datos meteorológicos...")
    
    df = df.copy()
    
    # Renombrar columna de tiempo
    df = df.rename(columns={'dt_iso': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    
    # Columnas de interés
    weather_cols = ['temp', 'humidity', 'wind_speed', 'pressure', 'clouds_all']
    
    # Agregar a nivel nacional (promedio de todas las ciudades)
    df_agg = df.groupby('datetime')[weather_cols].mean()
    
    # Convertir temperatura de Kelvin a Celsius
    if df_agg['temp'].mean() > 100:  # Está en Kelvin
        df_agg['temp'] = df_agg['temp'] - 273.15
    
    # Interpolación de valores faltantes
    df_agg = df_agg.interpolate(method='time', limit=6)
    df_agg = df_agg.fillna(method='ffill').fillna(method='bfill')
    
    print(f"   ✅ Weather agregado: {df_agg.shape[0]:,} timestamps")
    
    return df_agg


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features temporales."""
    print("⏰ Añadiendo features temporales...")
    
    df = df.copy()
    
    # Features básicos
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    
    # Features cíclicos (para capturar periodicidad)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Binarios
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Hora punta
    df['is_peak_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
    df['is_peak'] = (df['is_peak_morning'] | df['is_peak_evening']).astype(int)
    
    # Festivos españoles
    min_year = df.index.year.min()
    max_year = df.index.year.max()
    spain_holidays = holidays.Spain(years=range(min_year, max_year + 1))
    df['is_holiday'] = df.index.date
    df['is_holiday'] = df['is_holiday'].apply(lambda x: 1 if x in spain_holidays else 0)
    
    print(f"   ✅ Añadidas {len([c for c in df.columns if 'hour' in c or 'day' in c or 'month' in c or 'is_' in c])} features temporales")
    
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = 'total load actual') -> pd.DataFrame:
    """Añade features de lag y rolling para el target."""
    print("📊 Añadiendo features de lag...")
    
    df = df.copy()
    
    # Lags horarios
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:  # 168 = 1 semana
        df[f'load_lag_{lag}h'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [6, 12, 24, 168]:
        df[f'load_rolling_mean_{window}h'] = df[target_col].shift(1).rolling(window).mean()
        df[f'load_rolling_std_{window}h'] = df[target_col].shift(1).rolling(window).std()
    
    # Diferencias
    df['load_diff_1h'] = df[target_col].diff(1)
    df['load_diff_24h'] = df[target_col].diff(24)
    df['load_diff_168h'] = df[target_col].diff(168)
    
    # Ratio respecto a hace 24h y 168h
    df['load_ratio_24h'] = df[target_col] / df[target_col].shift(24)
    df['load_ratio_168h'] = df[target_col] / df[target_col].shift(168)
    
    # Reemplazar infinitos
    df = df.replace([np.inf, -np.inf], np.nan)
    
    print(f"   ✅ Añadidas {len([c for c in df.columns if 'lag' in c or 'rolling' in c or 'diff' in c or 'ratio' in c])} features de lag")
    
    return df


def merge_and_process() -> pd.DataFrame:
    """Pipeline completo de preprocesamiento."""
    print("=" * 60)
    print("⚡ PREPROCESAMIENTO - EnergiaPredictorES")
    print("=" * 60)
    
    # Cargar datos
    energy, weather = load_raw_data()
    
    # Si tenemos datos de REE (ya vienen como DataFrame indexado)
    if weather is None:
        print("   Usando datos de API REE (sin weather separado)")
        df = energy  # Ya está indexado
        
        # Detectar columna target
        target_candidates = ['Demanda real', 'total load actual', 'Demanda']
        target_col = None
        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                break
        
        if target_col and target_col != 'total load actual':
            df = df.rename(columns={target_col: 'total load actual'})
            print(f"   Renombrada columna '{target_col}' -> 'total load actual'")
        
        # Buscar columna de forecast
        forecast_candidates = ['Demanda prevista', 'Demanda programada', 'total load forecast']
        for candidate in forecast_candidates:
            if candidate in df.columns and 'total load forecast' not in df.columns:
                df = df.rename(columns={candidate: 'total load forecast'})
                break
        
    else:
        # Datos de Kaggle
        energy_clean = clean_energy_data(energy)
        weather_clean = clean_weather_data(weather)
        
        # Merge
        print("🔗 Combinando datasets...")
        df = energy_clean.join(weather_clean, how='left')
        df = df.interpolate(method='time', limit=6)
    
    print(f"   ✅ Dataset combinado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    
    # Verificar que tenemos target
    if 'total load actual' not in df.columns:
        print("❌ No se encontró columna de demanda real")
        print(f"   Columnas disponibles: {df.columns.tolist()}")
        raise ValueError("No se encontró columna de demanda")
    
    # Features temporales
    df = add_temporal_features(df)
    
    # Features de lag
    df = add_lag_features(df)
    
    # Eliminar filas con NaN (primeras filas por los lags)
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    print(f"🗑️  Eliminadas {rows_before - rows_after:,} filas con NaN (por lags)")
    
    # Guardar
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "energy_processed.parquet"
    df.to_parquet(output_path)
    
    print("=" * 60)
    print(f"✅ Dataset guardado: {output_path}")
    print(f"   Filas: {df.shape[0]:,}")
    print(f"   Columnas: {df.shape[1]}")
    print(f"   Período: {df.index.min()} a {df.index.max()}")
    print("=" * 60)
    
    return df


def load_processed_data() -> pd.DataFrame:
    """Carga el dataset procesado."""
    path = PROCESSED_DIR / "energy_processed.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta primero: python src/data/preprocessing.py"
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = merge_and_process()
    
    # Mostrar info básica
    print("\n📊 Primeras columnas:")
    print(df.columns.tolist()[:20])
    print(f"\n📈 Estadísticas del target (total load actual):")
    print(df['total load actual'].describe())
