[English](README.md) | **Español**

# EnergiaPredictorES

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Darts](https://img.shields.io/badge/Time%20Series-Darts-00D093)](https://unit8co.github.io/darts/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Sistema avanzado de predicción de demanda eléctrica nacional para España**, implementando un pipeline profesional que combina técnicas de **Machine Learning clásico (Gradient Boosting)** con arquitecturas de **Deep Learning de estado del arte (Temporal Fusion Transformer)**.

---

## Tabla de Contenidos

1.  [Descripción del Proyecto](#descripción-del-proyecto)
2.  [Objetivo y Alcance](#objetivo-y-alcance)
3.  [Datos Utilizados](#datos-utilizados)
4.  [Metodología](#metodología)
    *   [Preprocesamiento de Datos](#preprocesamiento-de-datos)
    *   [Ingeniería de Características](#ingeniería-de-características)
    *   [Modelos Implementados](#modelos-implementados)
5.  [Resultados Experimentales](#resultados-experimentales)
    *   [Métricas de Evaluación](#métricas-de-evaluación)
    *   [Comparativa de Modelos](#comparativa-de-modelos)
    *   [Análisis de Resultados](#análisis-de-resultados)
6.  [Instalación y Uso](#instalación-y-uso)
7.  [Estructura del Proyecto](#estructura-del-proyecto)
8.  [Autor](#autor)
9.  [Licencia](#licencia)

---

## Descripción del Proyecto

Este repositorio contiene la implementación completa de un sistema de predicción de demanda eléctrica para el mercado español. El proyecto aborda el problema de la predicción de series temporales de consumo energético, un dominio crítico para la operación eficiente de las redes eléctricas y la planificación de recursos.

Se implementa un enfoque híbrido que combina:
- **Modelos de Gradient Boosting (LightGBM, XGBoost):** Algoritmos de aprendizaje supervisado altamente efectivos para datos tabulares con características ingenierizadas manualmente.
- **Temporal Fusion Transformer (TFT):** Arquitectura de Deep Learning de última generación diseñada específicamente para predicción de series temporales, que combina mecanismos de atención con redes LSTM para capturar dependencias temporales complejas.

El modelo óptimo logra un error porcentual absoluto medio (MAPE) inferior al **1.2%** en el conjunto de test, demostrando la viabilidad del enfoque para aplicaciones en producción.

---

## Objetivo y Alcance

### Objetivo Principal
Desarrollar un modelo predictivo robusto y preciso capaz de pronosticar la **demanda neta de electricidad (MWh)** en el sistema eléctrico español con un horizonte temporal de 24 horas.

### Objetivos Secundarios
- Comparar rigurosamente el rendimiento de modelos de Machine Learning clásico frente a arquitecturas de Deep Learning.
- Demostrar la importancia del **Feature Engineering** temporal en la mejora del rendimiento predictivo.
- Establecer una línea base (**baseline**) con métodos estadísticos simples para cuantificar la mejora de los modelos avanzados.

### Competencias Demostradas
Este proyecto ilustra competencias en:
- **Ingeniería de Características (Feature Engineering):** Diseño de variables sintéticas para capturar patrones estacionales y tendencias.
- **Modelado Predictivo:** Implementación y optimización de múltiples familias de modelos.
- **Evaluación Experimental:** Diseño de experimentos y análisis comparativo con métricas estándar de la industria.
- **MLOps:** Estructura de proyecto modular, reproducible y escalable.

---

## Datos Utilizados

### Fuente de Datos
Los datos provienen del dataset público de Kaggle [Energy Consumption, Generation, Prices and Weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather), que contiene información histórica del mercado energético español.

### Periodo Temporal
- **Inicio:** 1 de enero de 2014
- **Fin:** 31 de diciembre de 2018
- **Frecuencia:** Horaria (agregada a diaria para el TFT)

### Variables Principales
| Variable | Descripción |
|----------|-------------|
| `total load actual` | Demanda real de electricidad (MWh) - **Variable objetivo** |
| `generation_solar` | Generación solar (MWh) |
| `generation_wind_onshore` | Generación eólica terrestre (MWh) |
| `generation_nuclear` | Generación nuclear (MWh) |
| `temp`, `humidity`, `pressure`, `wind_speed` | Variables meteorológicas |

### Partición de Datos
Para respetar la naturaleza temporal del problema y evitar **data leakage**, se utilizó una partición cronológica estricta:
- **Entrenamiento (Train):** 80% de los datos (primeros ~1252 puntos)
- **Validación (Val):** 10% de los datos (~156 puntos)
- **Test:** 10% de los datos (~157 puntos, últimos 6 meses)

---

## Metodología

### Preprocesamiento de Datos

El pipeline de preprocesamiento (`src/data/`) realiza las siguientes operaciones:

1.  **Limpieza de Timestamps:** Normalización de fechas para manejar correctamente los cambios de horario de verano/invierno en España.
2.  **Tratamiento de Valores Nulos:** Interpolación lineal para rellenar datos faltantes, preservando la continuidad temporal.
3.  **Detección de Anomalías:** Identificación y filtrado de outliers mediante análisis estadístico de Z-score.
4.  **Escalado de Características:** Normalización StandardScaler para los modelos de Deep Learning.

### Ingeniería de Características

Se diseñaron variables sintéticas para capturar los distintos patrones de consumo eléctrico:

#### Características Temporales (Cíclicas)
Para capturar la naturaleza cíclica del tiempo, se aplicó codificación trigonométrica:
```
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
day_sin  = sin(2 * pi * day_of_week / 7)
day_cos  = cos(2 * pi * day_of_week / 7)
month_sin = sin(2 * pi * month / 12)
month_cos = cos(2 * pi * month / 12)
```
Este enfoque evita la discontinuidad artificial entre, por ejemplo, las 23:00 y las 00:00.

#### Características de Calendario
- `is_weekend`: Indicador binario para sábados y domingos.
- `is_holiday`: Detección automática de festivos nacionales y regionales utilizando la librería `holidays`.

#### Características de Rezago (Lag Features)
Se incluyeron valores históricos de la variable objetivo:
- `lag_1h`: Demanda 1 hora antes.
- `lag_24h`: Demanda 24 horas antes (captura patrón diario).
- `lag_168h`: Demanda 168 horas antes (1 semana, captura patrón semanal).

#### Estadísticas de Ventana Móvil
- Media y desviación estándar de la demanda en ventanas de 6, 12 y 24 horas.

### Modelos Implementados

#### 1. Modelos Baseline (Referencia)

Se implementaron dos baselines estadísticos simples para establecer un límite inferior de rendimiento aceptable:

- **Naive (h-24):** Predicción basada en el valor observado hace 24 horas. Captura el patrón diario.
- **Seasonal Naive (h-168):** Predicción basada en el valor observado hace 168 horas (1 semana). Captura el patrón semanal.

#### 2. Modelos de Gradient Boosting

**LightGBM:**
- Algoritmo de boosting basado en gradiente desarrollado por Microsoft.
- Utiliza histogramas para acelerar el entrenamiento.
- Hiperparámetros: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`

**XGBoost:**
- Implementación altamente optimizada de Gradient Boosting.
- Hiperparámetros: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`

#### 3. Temporal Fusion Transformer (TFT)

Arquitectura de Deep Learning desarrollada por Google Research, específicamente diseñada para predicción de series temporales multihorzionte con interpretabilidad.

**Características clave:**
- **Multi-Horizon Forecasting:** Predice múltiples pasos futuros simultáneamente.
- **Variable Selection Networks:** Selecciona automáticamente las características más relevantes.
- **Interpretable Multi-Head Attention:** Identifica las dependencias temporales más importantes.

**Configuración del modelo:**
```python
TFTModel(
    input_chunk_length=30,    # Ventana de entrada: 30 días
    output_chunk_length=7,    # Horizonte de predicción: 7 días
    hidden_size=32,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=32,
    n_epochs=20,
    optimizer_kwargs={'lr': 1e-3}
)
```
**Parámetros totales:** 73,000 (entrenados en GPU Tesla T4)

---

## Resultados Experimentales

### Métricas de Evaluación

Se utilizaron cuatro métricas estándar para evaluar el rendimiento de los modelos:

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **MAE** | Mean Absolute Error | Error promedio en unidades originales (MWh) |
| **RMSE** | Root Mean Squared Error | Penaliza más los errores grandes |
| **MAPE** | Mean Absolute Percentage Error | Error porcentual promedio (%) |
| **sMAPE** | Symmetric MAPE | Versión simétrica del MAPE |

### Comparativa de Modelos

Los modelos fueron evaluados en el conjunto de test independiente (últimos 6 meses del dataset):

| Modelo | MAE (MWh) | RMSE (MWh) | MAPE (%) | sMAPE (%) |
|--------|-----------|------------|----------|-----------|
| **LightGBM** | **325.21** | **435.55** | **1.15** | **1.15** |
| XGBoost | 410.01 | 580.86 | 1.46 | 1.45 |
| TFT | 1523.15 | 1825.43 | 5.13 | 5.22 |
| Seasonal Naive (h-168) | 2710.24 | 3224.62 | 9.52 | 9.57 |
| Naive (h-24) | 2769.68 | 3370.80 | 9.79 | 9.75 |

### Análisis de Resultados

#### Rendimiento General
El modelo **LightGBM** obtiene el mejor rendimiento global con un MAPE de **1.15%**, seguido por XGBoost con un **1.46%**. Ambos modelos superan ampliamente a los baselines estadísticos, que alcanzan errores de aproximadamente el 9.5%.

#### LightGBM vs XGBoost
LightGBM supera a XGBoost en todas las métricas, lo cual es consistente con la literatura reciente que muestra la superioridad de LightGBM en datasets de tamaño moderado. Adicionalmente, LightGBM presenta tiempos de entrenamiento significativamente menores.

#### Modelos de Boosting vs TFT
Contraintuitivamente, los modelos de Gradient Boosting superan al Temporal Fusion Transformer en este caso. Este resultado se puede atribuir a varios factores:

1.  **Tamaño del dataset:** El TFT requiere grandes volúmenes de datos para aprender patrones complejos. Con ~500 puntos diarios, el modelo puede estar subentrenado.
2.  **Feature Engineering explícito:** Los modelos de boosting se benefician de las características ingenierizadas manualmente (lags, cíclicas), mientras que el TFT intenta aprender estas representaciones automáticamente.
3.  **Horizonte de predicción:** El TFT está optimizado para predicción multihorzionte a largo plazo, mientras que los modelos de boosting funcionan bien para predicciones puntuales a corto plazo.

#### Mejora sobre Baseline
La reducción del error respecto al mejor baseline (Seasonal Naive) es:
- **LightGBM:** Reducción del 88% en MAPE (de 9.52% a 1.15%)
- **XGBoost:** Reducción del 85% en MAPE (de 9.52% a 1.46%)

### Conclusión

Para el problema de predicción de demanda eléctrica a corto plazo con datos estructurados, los modelos de **Gradient Boosting** (especialmente LightGBM) enriquecidos con un fuerte **Feature Engineering** temporal demuestran ser extremadamente competitivos, superando incluso a arquitecturas de Deep Learning más complejas como el TFT.

---

## Instalación y Uso

### Prerrequisitos
- Python 3.9 o superior
- Git

### Setup del Entorno

1.  Clonar el repositorio:
    ```bash
    git clone https://github.com/nathanmarinas2/EnergiaPredictorES.git
    cd EnergiaPredictorES
    ```

2.  Crear y activar entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```

3.  Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```

### Ejecución del Pipeline

#### Opción A: Notebook Interactivo (Recomendado)
Abrir `notebooks/EnergiaPredictorES_Colab.ipynb` en Google Colab o Jupyter local para ver el proceso completo con visualizaciones.

#### Opción B: Scripts de Python
```bash
# 1. Descargar datos (requiere credenciales de Kaggle)
python src/data/download.py

# 2. Preprocesar datos
python src/data/preprocessing.py

# 3. Entrenar modelos
python src/models/baseline.py --model lightgbm
```

---

## Estructura del Proyecto

```
EnergiaPredictorES/
|-- config.yaml             # Configuración global (rutas, hiperparámetros)
|-- requirements.txt        # Dependencias de Python
|-- pyproject.toml          # Configuración de proyecto moderna
|-- LICENSE                 # Licencia MIT
|-- README.es.md            # Documentación en Español
|-- README.md               # English Documentation
|
|-- data/
|   |-- raw/                # Datos originales (inmutables)
|   +-- processed/          # Datos transformados para modelado
|
|-- notebooks/
|   +-- EnergiaPredictorES_Colab.ipynb  # Notebook principal con EDA y modelado
|
|-- src/
|   |-- data/
|   |   |-- download.py         # Descarga de datos desde Kaggle
|   |   |-- download_ree.py     # Descarga alternativa desde API REE
|   |   +-- preprocessing.py    # Pipeline de preprocesamiento
|   |
|   |-- models/
|   |   |-- baseline.py         # Implementación de LightGBM, XGBoost
|   |   +-- tft.py              # Implementación del Temporal Fusion Transformer
|   |
|   +-- evaluation/
|       +-- metrics.py          # Funciones de evaluación (MAPE, RMSE, etc.)
|
+-- models/                 # Modelos entrenados (no versionados)
```

---

## Autor

**Nathan Mariñas Pose**

- Ingeniería en Inteligencia Artificial - Universidad de A Coruña
- [LinkedIn](https://www.linkedin.com/in/nathan-mari%C3%B1as-pose-419b0b385/)

---

## Licencia

Este proyecto está bajo la Licencia MIT. Consultar el archivo [LICENSE](LICENSE) para más detalles.
