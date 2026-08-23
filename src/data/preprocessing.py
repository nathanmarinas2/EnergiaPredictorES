"""
Preprocesamiento de datos de energía eléctrica de España.
Incluye limpieza, merge con datos meteorológicos y feature engineering temporal.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
try:
    import holidays
except ImportError:  # Permite probar las funciones de lags sin la dependencia opcional.
    holidays = None

# Rutas
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def infer_time_frequency(index: pd.DatetimeIndex) -> str:
    """Infiere si una serie regular representa horas o días.

    El dataset ``spain_energy_market.csv`` contiene una observación diaria,
    aunque sus timestamps estén a las 22:00/23:00. No se deben interpretar
    esos saltos como horas consecutivas.
    """
    if len(index) < 2:
        raise ValueError("Se necesitan al menos dos timestamps para inferir la frecuencia")

    values = pd.DatetimeIndex(index).sort_values()
    deltas = values.to_series().diff().dropna()
    median_delta = deltas.median()

    if median_delta <= pd.Timedelta(hours=2):
        return "h"
    if median_delta <= pd.Timedelta(days=2):
        return "D"
    raise ValueError(
        "La serie temporal es demasiado irregular para construir lags fiables: "
        f"salto mediano={median_delta}"
    )


def regularize_time_index(df: pd.DataFrame, frequency: str = None) -> pd.DataFrame:
    """Ordena y regulariza el índice sin inventar valores del objetivo.

    Para fuentes diarias normaliza las horas 22:00/23:00 y conserva una sola
    observación por fecha. Los huecos se mantienen como NaN; se rellenan solo
    covariables conocidas mediante forward-fill en el pipeline posterior.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    if df.index.has_duplicates:
        df = df.groupby(level=0).first()

    frequency = frequency or infer_time_frequency(df.index)
    if frequency == "D":
        df.index = df.index.normalize()
        df = df.groupby(level=0).first().sort_index()
        return df.asfreq("D")
    if frequency == "h":
        return df.asfreq("h")
    raise ValueError(f"Frecuencia no soportada: {frequency}")


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
    
    # Solo arrastramos información disponible del pasado. No interpolamos ni
    # hacemos back-fill aquí porque eso puede usar observaciones futuras al
    # construir el conjunto de entrenamiento.
    missing_before = df.isnull().sum().sum()
    fill_cols = [
        c for c in df.columns
        if c not in {'total load actual', 'total load forecast'}
    ]
    df[fill_cols] = df[fill_cols].ffill(limit=24)
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
    
    # Solo forward-fill: una predicción no puede conocer meteorología futura.
    df_agg = df_agg.ffill(limit=6)
    
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
    if holidays is None:
        raise ImportError(
            "La librería 'holidays' es necesaria para crear is_holiday. "
            "Instala las dependencias del proyecto."
        )
    min_year = df.index.year.min()
    max_year = df.index.year.max()
    spain_holidays = holidays.Spain(years=range(min_year, max_year + 1))
    df['is_holiday'] = df.index.date
    df['is_holiday'] = df['is_holiday'].apply(lambda x: 1 if x in spain_holidays else 0)
    
    print(f"   ✅ Añadidas {len([c for c in df.columns if 'hour' in c or 'day' in c or 'month' in c or 'is_' in c])} features temporales")
    
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = 'total load actual',
    frequency: str = None,
) -> pd.DataFrame:
    """Añade features calculables en el instante de predicción.

    Todas las features derivadas del objetivo usan ``shift(1)`` o un lag
    anterior. En particular, nunca se calcula una diferencia/ratio que
    contenga ``y_t``: hacerlo permitiría reconstruir el objetivo directamente.
    """
    print("📊 Añadiendo features de lag...")

    df = df.copy()
    frequency = frequency or infer_time_frequency(df.index)
    if frequency == "h":
        day_period, week_period, unit = 24, 168, "h"
    elif frequency == "D":
        day_period, week_period, unit = 1, 7, "d"
    else:
        raise ValueError(f"Frecuencia no soportada: {frequency}")

    # Lags de pasos, día y semana. Los nombres reflejan la frecuencia real.
    lag_periods = {
        "1step": 1,
        "2step": 2,
        "3step": 3,
        "6step": 6,
        "12step": 12,
        "1day": day_period,
        "2day": day_period * 2,
        "1week": week_period,
    }
    for name, lag in lag_periods.items():
        df[f'load_lag_{name}'] = df[target_col].shift(lag)

    # Rolling statistics
    rolling_periods = {
        "6step": 6,
        "12step": 12,
        "1day": day_period,
        "1week": week_period,
    }
    history = df[target_col].shift(1)
    for name, window in rolling_periods.items():
        df[f'load_rolling_mean_{name}'] = history.rolling(window).mean()
        df[f'load_rolling_std_{name}'] = history.rolling(window).std(ddof=0)

    # Diferencias válidas: entre dos valores ya observados, nunca con y_t.
    df['load_diff_1step'] = df[target_col].shift(1) - df[target_col].shift(2)
    df['load_diff_1day'] = (
        df[target_col].shift(day_period)
        - df[target_col].shift(day_period + 1)
    )
    df['load_diff_1week'] = (
        df[target_col].shift(week_period)
        - df[target_col].shift(week_period + 1)
    )

    # Ratios entre valores históricos. Se conserva la información útil, pero
    # nunca aparece el valor actual del objetivo en el numerador.
    df['load_ratio_1day'] = (
        df[target_col].shift(1) / df[target_col].shift(day_period + 1)
    )
    df['load_ratio_1week'] = (
        df[target_col].shift(1) / df[target_col].shift(week_period + 1)
    )

    # Reemplazar infinitos
    df = df.replace([np.inf, -np.inf], np.nan)
    
    print(f"   ✅ Añadidas {len([c for c in df.columns if 'lag' in c or 'rolling' in c or 'diff' in c or 'ratio' in c])} features de lag ({unit})")
    
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
    
    # Regularizar la frecuencia antes de crear lags. Nunca rellenamos el
    # objetivo con interpolación: los huecos del target se descartan después.
    frequency = infer_time_frequency(df.index)
    df = regularize_time_index(df, frequency)
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in {'total load actual', 'total load forecast'}
    ]
    df[numeric_cols] = df[numeric_cols].ffill()
    df = df.dropna(axis=1, how='all')

    print(f"   ✅ Dataset combinado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    
    # Verificar que tenemos target
    if 'total load actual' not in df.columns:
        print("❌ No se encontró columna de demanda real")
        print(f"   Columnas disponibles: {df.columns.tolist()}")
        raise ValueError("No se encontró columna de demanda")
    
    # Features temporales
    df = add_temporal_features(df)
    
    # Features de lag
    df = add_lag_features(df, frequency=frequency)

    # Solo se eliminan filas que no pueden formar una observación válida del
    # target o de su historial. No se usa dropna() sobre todas las columnas:
    # eso descartaba la mayoría de días por columnas auxiliares incompletas.
    lag_cols = [
        c for c in df.columns
        if c.startswith('load_lag_')
        or c.startswith('load_rolling_')
        or c.startswith('load_diff_')
        or c.startswith('load_ratio_')
    ]
    rows_before = len(df)
    df = df.dropna(subset=['total load actual', *lag_cols])

    # Las columnas auxiliares que aún tienen NaN no se pueden pasar a los
    # modelos sin imputación futura. Se descartan, conservando target y lags.
    remaining_nan_cols = [
        c for c in df.columns
        if c != 'total load actual' and df[c].isna().any()
    ]
    if remaining_nan_cols:
        print(f"🗑️  Eliminadas {len(remaining_nan_cols)} columnas con NaN residual")
        df = df.drop(columns=remaining_nan_cols)

    rows_after = len(df)
    print(f"🗑️  Eliminadas {rows_before - rows_after:,} filas con NaN (por historial del target)")
    
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
