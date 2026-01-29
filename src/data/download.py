"""
Descarga del dataset de Kaggle: Energy Consumption, Generation, Prices and Weather
https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather
"""

import os
import zipfile
from pathlib import Path

# Configuración
KAGGLE_DATASET = "nicholasjhana/energy-consumption-generation-prices-and-weather"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def check_kaggle_credentials():
    """Verifica que las credenciales de Kaggle estén configuradas."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.exists():
        print("=" * 60)
        print("❌ No se encontró kaggle.json")
        print("=" * 60)
        print("\nPara descargar datos de Kaggle necesitas:")
        print("1. Ir a https://www.kaggle.com/settings")
        print("2. En 'API', hacer click en 'Create New Token'")
        print("3. Guardar el archivo kaggle.json en:")
        print(f"   {kaggle_json}")
        print("\nAlternativamente, descarga manualmente el dataset desde:")
        print(f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print(f"y descomprime los archivos en: {DATA_DIR}")
        print("=" * 60)
        return False
    return True


def download_dataset():
    """Descarga el dataset desde Kaggle."""
    if not check_kaggle_credentials():
        return False
    
    # Importar kaggle después de verificar credenciales
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle no está instalado. Ejecuta: pip install kaggle")
        return False
    
    # Crear directorio
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Descargando dataset: {KAGGLE_DATASET}")
    print(f"   Destino: {DATA_DIR}")
    
    try:
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=DATA_DIR,
            unzip=True
        )
        print("✅ Descarga completada!")
        
        # Listar archivos descargados
        files = list(DATA_DIR.glob("*.csv"))
        print(f"\n📁 Archivos descargados:")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.2f} MB)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error descargando: {e}")
        return False


def verify_dataset():
    """Verifica que los archivos necesarios existan."""
    required_files = [
        "energy_dataset.csv",
        "weather_features.csv"
    ]
    
    missing = []
    for f in required_files:
        if not (DATA_DIR / f).exists():
            missing.append(f)
    
    if missing:
        print(f"❌ Archivos faltantes: {missing}")
        return False
    
    print("✅ Todos los archivos requeridos están presentes.")
    return True


def main():
    print("=" * 60)
    print("⚡ DESCARGA DE DATOS - EnergiaPredictorES")
    print("=" * 60)
    
    # Verificar si ya existen
    if verify_dataset():
        print("\nℹ️  Los datos ya están descargados.")
        response = input("¿Descargar de nuevo? (s/N): ").strip().lower()
        if response != 's':
            return
    
    # Descargar
    success = download_dataset()
    
    if success:
        verify_dataset()
    else:
        print("\n⚠️  Si la descarga automática falla, puedes descargar manualmente:")
        print(f"   https://www.kaggle.com/datasets/{KAGGLE_DATASET}")


if __name__ == "__main__":
    main()
