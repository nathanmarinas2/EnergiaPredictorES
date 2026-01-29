"""
Descarga de datos de la API oficial de Red Eléctrica de España (REData).
Documentación: https://www.ree.es/es/apidatos

Descarga datos de:
- Demanda eléctrica real y prevista
- Generación por tecnología
- Precios de mercado
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import time
import json

# Configuración
BASE_URL = "https://apidatos.ree.es"
RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

# Headers requeridos por la API
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "EnergiaPredictorES/1.0 (Academic Project)"
}


class REEDataFetcher:
    """Cliente para la API de Red Eléctrica de España."""
    
    def __init__(self, lang: str = "es"):
        self.lang = lang
        self.base_url = BASE_URL
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        
    def _make_request(
        self,
        category: str,
        widget: str,
        start_date: str,
        end_date: str,
        time_trunc: str = "hour",
        geo_limit: str = None,
        geo_ids: int = None,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """Realiza una petición a la API."""
        
        url = f"{self.base_url}/{self.lang}/datos/{category}/{widget}"
        
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "time_trunc": time_trunc,
        }
        
        if geo_limit:
            params["geo_trunc"] = "electric_system"
            params["geo_limit"] = geo_limit
        if geo_ids:
            params["geo_ids"] = geo_ids
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    print(f"   ⚠️ Rate limit, esperando {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Error {response.status_code}: {response.text[:200]}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"   ⚠️ Timeout, reintentando ({attempt + 1}/{max_retries})...")
                time.sleep(2)
            except Exception as e:
                print(f"   ❌ Error: {e}")
                return None
        
        return None
    
    def _parse_response(self, data: Dict) -> pd.DataFrame:
        """Parsea la respuesta de la API a DataFrame."""
        if not data or "included" not in data:
            return pd.DataFrame()
        
        all_records = []
        
        for indicator in data.get("included", []):
            indicator_id = indicator.get("id", "unknown")
            indicator_title = indicator.get("attributes", {}).get("title", indicator_id)
            
            values = indicator.get("attributes", {}).get("values", [])
            
            for v in values:
                record = {
                    "datetime": v.get("datetime"),
                    "indicator": indicator_title,
                    "value": v.get("value"),
                    "percentage": v.get("percentage"),
                }
                all_records.append(record)
        
        if not all_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        return df
    
    def get_demand(
        self,
        start_date: str,
        end_date: str,
        time_trunc: str = "hour"
    ) -> pd.DataFrame:
        """
        Obtiene datos de demanda eléctrica.
        
        Indicadores incluidos:
        - Demanda real
        - Demanda programada
        - Demanda prevista
        """
        print(f"📥 Descargando demanda: {start_date} a {end_date}")
        
        data = self._make_request(
            category="demanda",
            widget="evolucion",
            start_date=start_date,
            end_date=end_date,
            time_trunc=time_trunc
        )
        
        df = self._parse_response(data)
        
        if not df.empty:
            # Pivotar para tener cada indicador como columna
            df_pivot = df.pivot_table(
                index="datetime",
                columns="indicator",
                values="value",
                aggfunc="first"
            ).reset_index()
            df_pivot.columns.name = None
            print(f"   ✅ {len(df_pivot)} registros")
            return df_pivot
        
        return df
    
    def get_generation(
        self,
        start_date: str,
        end_date: str,
        time_trunc: str = "hour"
    ) -> pd.DataFrame:
        """
        Obtiene datos de generación eléctrica por tecnología.
        
        Indicadores incluidos:
        - Solar fotovoltaica
        - Eólica
        - Nuclear
        - Hidráulica
        - Ciclo combinado
        - etc.
        """
        print(f"📥 Descargando generación: {start_date} a {end_date}")
        
        data = self._make_request(
            category="generacion",
            widget="estructura-generacion",
            start_date=start_date,
            end_date=end_date,
            time_trunc=time_trunc
        )
        
        df = self._parse_response(data)
        
        if not df.empty:
            df_pivot = df.pivot_table(
                index="datetime",
                columns="indicator",
                values="value",
                aggfunc="first"
            ).reset_index()
            df_pivot.columns.name = None
            # Añadir prefijo 'gen_' a las columnas
            df_pivot.columns = ["datetime"] + [f"gen_{c}" for c in df_pivot.columns[1:]]
            print(f"   ✅ {len(df_pivot)} registros")
            return df_pivot
        
        return df
    
    def get_exchange(
        self,
        start_date: str,
        end_date: str,
        time_trunc: str = "hour"
    ) -> pd.DataFrame:
        """Obtiene datos de intercambios internacionales."""
        print(f"📥 Descargando intercambios: {start_date} a {end_date}")
        
        data = self._make_request(
            category="intercambios",
            widget="todas-fronteras-fisicos",
            start_date=start_date,
            end_date=end_date,
            time_trunc=time_trunc
        )
        
        df = self._parse_response(data)
        
        if not df.empty:
            df_pivot = df.pivot_table(
                index="datetime",
                columns="indicator",
                values="value",
                aggfunc="first"
            ).reset_index()
            df_pivot.columns.name = None
            df_pivot.columns = ["datetime"] + [f"exchange_{c}" for c in df_pivot.columns[1:]]
            print(f"   ✅ {len(df_pivot)} registros")
            return df_pivot
        
        return df


def download_historical_data(
    start_year: int = 2019,
    end_year: int = 2025,
    time_trunc: str = "hour"
) -> pd.DataFrame:
    """
    Descarga datos históricos completos de REE.
    
    La API tiene límites, así que descargamos por meses.
    """
    print("=" * 60)
    print("⚡ DESCARGA DE DATOS - API REE (Red Eléctrica de España)")
    print("=" * 60)
    
    fetcher = REEDataFetcher()
    
    all_demand = []
    all_generation = []
    
    # Descargar por meses para evitar timeouts
    current_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    while current_date < end_date:
        month_end = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        if month_end > end_date:
            month_end = end_date
        
        start_str = current_date.strftime("%Y-%m-%dT00:00")
        end_str = month_end.strftime("%Y-%m-%dT23:59")
        
        print(f"\n📅 {current_date.strftime('%Y-%m')}:")
        
        # Demanda
        df_demand = fetcher.get_demand(start_str, end_str, time_trunc)
        if not df_demand.empty:
            all_demand.append(df_demand)
        
        # Generación
        df_gen = fetcher.get_generation(start_str, end_str, time_trunc)
        if not df_gen.empty:
            all_generation.append(df_gen)
        
        # Siguiente mes
        current_date = month_end + timedelta(days=1)
        
        # Pequeña pausa para no saturar la API
        time.sleep(0.5)
    
    # Combinar todo
    print("\n🔗 Combinando datos...")
    
    if all_demand:
        df_demand_full = pd.concat(all_demand, ignore_index=True)
        df_demand_full = df_demand_full.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    else:
        print("❌ No se pudo descargar datos de demanda")
        return pd.DataFrame()
    
    if all_generation:
        df_gen_full = pd.concat(all_generation, ignore_index=True)
        df_gen_full = df_gen_full.drop_duplicates(subset=["datetime"]).sort_values("datetime")
        
        # Merge
        df_full = df_demand_full.merge(df_gen_full, on="datetime", how="left")
    else:
        df_full = df_demand_full
    
    # Guardar
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / "ree_data.parquet"
    df_full.to_parquet(output_path, index=False)
    
    # También guardar como CSV para inspección
    csv_path = RAW_DIR / "ree_data.csv"
    df_full.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"✅ Datos guardados:")
    print(f"   - {output_path}")
    print(f"   - {csv_path}")
    print(f"   Filas: {len(df_full):,}")
    print(f"   Columnas: {len(df_full.columns)}")
    print(f"   Período: {df_full['datetime'].min()} a {df_full['datetime'].max()}")
    print("=" * 60)
    
    return df_full


def download_recent_data(days: int = 30) -> pd.DataFrame:
    """Descarga datos de los últimos N días (para predicción en tiempo real)."""
    print(f"📥 Descargando últimos {days} días...")
    
    fetcher = REEDataFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_str = start_date.strftime("%Y-%m-%dT00:00")
    end_str = end_date.strftime("%Y-%m-%dT23:59")
    
    df_demand = fetcher.get_demand(start_str, end_str, "hour")
    df_gen = fetcher.get_generation(start_str, end_str, "hour")
    
    if df_demand.empty:
        print("❌ No se pudieron descargar datos recientes")
        return pd.DataFrame()
    
    if not df_gen.empty:
        df = df_demand.merge(df_gen, on="datetime", how="left")
    else:
        df = df_demand
    
    return df


def main():
    """Descarga datos históricos completos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Descarga datos de REE")
    parser.add_argument("--start-year", type=int, default=2020, help="Año inicial")
    parser.add_argument("--end-year", type=int, default=2025, help="Año final")
    parser.add_argument("--recent", type=int, default=0, help="Solo últimos N días")
    
    args = parser.parse_args()
    
    if args.recent > 0:
        df = download_recent_data(args.recent)
    else:
        df = download_historical_data(args.start_year, args.end_year)
    
    if not df.empty:
        print("\n📊 Primeras columnas:")
        print(df.columns.tolist())
        print("\n📈 Muestra de datos:")
        print(df.head())


if __name__ == "__main__":
    main()
