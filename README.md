# ⚡ EnergiaPredictorES

## Sistema de Predicción de Demanda Eléctrica Nacional con Deep Learning

Predicción de la demanda eléctrica de España utilizando modelos de Time Series avanzados (Temporal Fusion Transformer, N-BEATS) entrenados con datos históricos de Red Eléctrica de España.

---

## 🎯 Objetivo

Superar las predicciones oficiales de REE utilizando arquitecturas de Deep Learning modernas, demostrando dominio de:
- **Time Series Forecasting** con modelos estado del arte
- **Feature Engineering** temporal y meteorológico
- **Evaluación rigurosa** con métricas estándar de la industria
- **Pipeline ML profesional** reproducible

---

## 📊 Dataset

### Opción A: Kaggle (Recomendado para empezar)
**Fuente:** [Kaggle - Hourly Energy Demand Generation and Weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)

- **Período:** 4 años de datos horarios (2015-2018)
- **Variables:** Demanda real, generación por tipo (eólica, solar, nuclear...), precios, meteorología
- **Granularidad:** Horaria
- **Ventaja:** Dataset limpio y listo para usar

### Opción B: API REE (Para datos actualizados)
**Fuente:** [API REData - Red Eléctrica de España](https://www.ree.es/es/apidatos)

- **Período:** Desde 2014 hasta hoy
- **Variables:** Demanda real/prevista, generación por tecnología, intercambios
- **Granularidad:** Horaria (o diaria/mensual)
- **Ventaja:** Datos en tiempo real, permite predicción operativa
- **Script:** `src/data/download_ree.py`

---

## 🏗️ Arquitectura del Proyecto

```
proyecto_3/
├── data/
│   ├── raw/                    # Datos originales de Kaggle
│   ├── processed/              # Datos preprocesados
│   └── external/               # Datos externos (festivos, etc.)
├── src/
│   ├── data/
│   │   ├── download.py         # Descarga de Kaggle
│   │   └── preprocessing.py    # Limpieza y feature engineering
│   ├── features/
│   │   └── build_features.py   # Variables temporales, lags, etc.
│   ├── models/
│   │   ├── baseline.py         # Modelos baseline (ARIMA, XGBoost)
│   │   ├── tft.py              # Temporal Fusion Transformer
│   │   └── nbeats.py           # N-BEATS
│   ├── evaluation/
│   │   └── metrics.py          # MAPE, RMSE, MAE, comparativas
│   └── visualization/
│       └── plots.py            # Gráficas de predicción
├── notebooks/
│   ├── 01_eda.ipynb            # Análisis exploratorio
│   ├── 02_baseline.ipynb       # Modelos baseline
│   └── 03_deep_learning.ipynb  # Modelos DL
├── models/                     # Modelos entrenados (.pt, .pkl)
├── reports/
│   └── figures/                # Gráficas para el informe
├── requirements.txt
├── config.yaml                 # Configuración de hiperparámetros
└── README.md
```

---

## 🔧 Tecnologías

| Categoría | Herramientas |
|-----------|--------------|
| **Deep Learning** | PyTorch, PyTorch Lightning |
| **Time Series** | Darts, PyTorch Forecasting, NeuralProphet |
| **ML Clásico** | scikit-learn, XGBoost, LightGBM |
| **Data** | Pandas, NumPy, Polars |
| **Tracking** | Weights & Biases (WandB) |
| **Visualización** | Matplotlib, Plotly |

---

## 📈 Modelos Implementados

### Baseline
- **Naive:** Último valor conocido
- **Seasonal Naive:** Mismo valor hace 24h/168h
- **ARIMA/SARIMA:** Modelos clásicos
- **XGBoost:** Gradient Boosting con features temporales

### Deep Learning
- **N-BEATS:** Neural Basis Expansion Analysis
- **TFT (Temporal Fusion Transformer):** Attention + interpretabilidad
- **PatchTST:** Transformers con patches

---

## 📊 Métricas de Evaluación

| Métrica | Descripción |
|---------|-------------|
| **MAPE** | Mean Absolute Percentage Error |
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **SMAPE** | Symmetric MAPE |

**Benchmark:** Comparación con la previsión oficial de REE.

---

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/tuusuario/energia-predictor-es.git
cd energia-predictor-es

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Descargar datos de Kaggle
python src/data/download.py
```

---

## 📝 Uso

```bash
# 1. Preprocesar datos
python src/data/preprocessing.py

# 2. Entrenar baseline
python src/models/baseline.py

# 3. Entrenar TFT
python src/models/tft.py --epochs 50 --lr 0.001

# 4. Evaluar
python src/evaluation/metrics.py --model tft
```

---

## 👨‍💻 Autor

**Nathan Mariñas Pose**  
Estudiante de Ingeniería en IA - Universidad de A Coruña  
[LinkedIn](https://www.linkedin.com/in/nathan-mari%C3%B1as-pose-419b0b385/)

---

## � TODO - Próximos Pasos

### Fase 1: Setup y Datos ✅
- [x] Crear estructura del proyecto
- [x] Descargar dataset de Kaggle (`spain_energy_market.csv`)
- [x] Subir a GitHub

### Fase 2: Preprocesamiento (En progreso)
- [ ] Instalar dependencias: `pip install -r requirements.txt`
- [ ] Ejecutar preprocessing: `python src/data/preprocessing.py`
- [ ] Verificar que se genera `data/processed/energy_processed.parquet`
- [ ] Crear notebook EDA (`notebooks/01_eda.ipynb`) con visualizaciones

### Fase 3: Modelos Baseline
- [ ] Ejecutar baselines: `python src/models/baseline.py`
- [ ] Documentar métricas de XGBoost y LightGBM
- [ ] Comparar con predicción oficial de REE
- [ ] Guardar resultados en `models/baseline_results.csv`

### Fase 4: Deep Learning (TFT)
- [ ] Entrenar TFT: `python src/models/tft.py`
- [ ] Ajustar hiperparámetros si es necesario
- [ ] Comparar TFT vs Baselines vs REE oficial
- [ ] Generar gráficas de predicción

### Fase 5: Documentación Final
- [ ] Añadir gráficas de resultados al README
- [ ] Crear notebook final con análisis completo
- [ ] Escribir sección de "Resultados" con métricas finales
- [ ] (Opcional) Añadir integración con WandB para tracking

### Fase 6: Extras (Opcional)
- [ ] Implementar N-BEATS como alternativa a TFT
- [ ] Añadir datos meteorológicos externos (AEMET)
- [ ] Crear API REST para predicciones en tiempo real
- [ ] Desplegar en cloud (AWS/GCP)

---

## �📄 Licencia

MIT License
