# 🌪️ AirQuality Analytics (MA2003B_Eq1)

Sistema integral de análisis, preprocesamiento y modelado predictivo de calidad del aire (específicamente Ozono, $O_3$) para el Área Metropolitana de Monterrey (AMM).

## 📖 Acerca del Proyecto

Este repositorio contiene todo el pipeline de datos, desde la limpieza y unificación de archivos crudos del SIMA, hasta el entrenamiento de modelos de Machine Learning avanzados (Gradient Boosting) para predecir concentraciones de ozono en diferentes horizontes temporales (1 a 48 horas).

El proyecto transitó por diferentes enfoques:
- **Modelos Econométricos (ARX/OLS):** Para entender la importancia de los predictores y confirmar el rol crítico de la temperatura y la radiación solar.
- **Deep Learning (LSTM):** Diagnosticado y descartado debido a la gran cantidad de datos faltantes dispersos en los sensores del SIMA, lo que hacía inviable el uso de secuencias continuas.
- **Gradient Boosting (LightGBM):** Solución final que maneja NaNs nativamente, empleando una **Estrategia Directa Multi-Horizonte** con _Feature Engineering_ especializado en la química del ozono, alcanzando un $R^2$ de 0.854 para predecir a 1 hora en datos nunca vistos.

## 🚀 Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto localmente:

1. **Clonar el repositorio:**
   ```bash
   git clone <url-del-repo>
   cd MA2003B_Eq1
   ```

2. **Crear y activar un entorno virtual:**
   * En Windows:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * En macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Nota: `requirements.txt` se mantiene actualizado para asegurar la replicabilidad total del entorno y los modelos).*

## 🗂️ Estructura Principal del Pipeline

### 🧹 Preparación de Datos (Data Engineering)
* `1_join_raw.ipynb`: Consolida bases de datos separadas del SIMA en un historial completo (2020-2025).
* `1.5_xray_lostData.ipynb`: Diagnóstico y radiografía de datos faltantes/perdidos en la red de monitoreo.
* `2_clean_split.ipynb`: Limpieza profunda, aplicación de límites físicos normativos y partición por variables (formato `.parquet` optimizado).
* `3_features.ipynb`: Creación del "Dataset Inteligente", unificando tablas, calculando trigonometría temporal y rezagos (lags).

### 📊 Análisis Exploratorio y Modelado Clásico
* `4_analisis_visual.ipynb` / `.py`: Análisis de distribuciones, series temporales y funciones de correlación cruzada (CCF).
* `etapa2_analisis.ipynb` / `.py`: Implementación de modelos ARX/OLS, evaluación de autocorrelación (Durbin-Watson) y errores robustos HAC.

### 🤖 Machine Learning Avanzado (LightGBM)
* `5_boosting_predictivo.py` / `train_lgbm.py`: Pruebas iniciales de predictibilidad con LightGBM.
* `6_multihorizonte_lgbm.py`: Entrenamiento de 6 modelos independientes de LightGBM (Horizontes de 1h, 4h, 8h, 12h, 24h, 48h).
* `7_multihorizonte_feature_eng.py`: Versión final del pipeline incorporando características específicas de fotoquímica e interacciones químicas.

### 📈 Reportes y Visualización
* `app_laboratorio.py` / `app_presentacion.py`: Scripts orientados a la evaluación final y dashboards de resultados.
* `script_final_tareas.py`: Generación automática de métricas consolidadas (R², RMSE, importancia de variables) para evaluación e informes.
* `data/`: Directorio local de almacenamiento de datos.
* `results/`: Almacena métricas, figuras y reportes generados automáticamente por los scripts (e.g. `reporte_multihorizonte.md`, `reporte_v2_feature_engineering.md`).
