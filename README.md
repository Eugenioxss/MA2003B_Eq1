# 🌪️ Pulso Atmosférico (`MA2003B_Eq1`)

Sistema integral de análisis estadístico, imputación causal espacial, acoplamiento meteorológico numérico (NWP + MOS), aprendizaje profundo, meta-aprendizaje y calibración conformal para el pronóstico multi-horizonte de **Ozono Troposférico ($O_3$)** en las 13 estaciones operativas del Sistema Integral de Monitoreo Ambiental (**SIMA**) del Área Metropolitana de Monterrey (AMM), Nuevo León (`2020–2025`).

---

## 📖 Acerca del Proyecto y Evolución Científica

Este repositorio contiene el pipeline completo de extremo a extremo: desde la consolidación y auditoría de datos crudos del SIMA (`2020–2025`), hasta una arquitectura híbrida de **268 modelos en 14 familias algorítmicas** evaluada bajo un protocolo estricto de **Validación Cruzada Temporal Out-of-Fold (OOF) en 2024** y prueba ciega en **2025** ($108{,}512$ registros horario-estación en $h=1$) para los horizontes $h \in \{1, 4, 8, 12, 24, 48\}$ horas.

El proyecto transitó por seis etapas evolutivas de investigación:
1. **Modelos Econométricos (ARX/OLS con errores HAC):** Confirmaron el rol estructural de la radiación solar ($\text{SR}$), temperatura ($\text{TOUT}$) y persistencia autorregresiva, diagnosticando la fuerte autocorrelación serial y heterocedasticidad diurna.
2. **Baseline Machine Learning (`v2`, Script 7 — LightGBM Directo con rezagos $\le t-1$):** Primer sistema multi-horizonte capaz de manejar `NaN` nativamente con ingeniería de variables fotoquímicas básicas, alcanzando un $\text{RMSE} = 8.051\text{ ppb}$ ($R^2 = 0.854$) en $h=1$ y $13.949\text{ ppb}$ ($R^2 = 0.562$) en $h=48$.
3. **Química de Leighton Contemporánea y Advección Espacial (`Plan A V4`):** Incorporación del estado en el instante de emisión $t$ (lag 0), conservación de oxidantes totales ($\text{O}_x = \text{O}_3 + \text{NO}_2$), razones $\text{NO}_2/\text{NO}_x$ y transporte advectivo *upwind* proyectado sobre el vector de viento entre las 15 estaciones de la cuenca, reduciendo el RMSE en $h=1$ de $8.051$ a **$5.094\text{ ppb}$**.
4. **Imputación Espacial Multivariada 100% Causal (`imp_spatial_lgbm` / `imput2`):** Mientras que el diagnóstico inicial descartó LSTM/SAITS sobre datos crudos por fragmentación de secuencias y fuga temporal de atención bidireccional, se diseñaron **135 regresores espaciales causales LightGBM** (entrenados solo en `2020–2023` con estaciones donantes en $t$ y rezagos $\le t$) que reconstruyen el panel antes de calcular derivadas temporales, más memoria fotoquímica del día previo ($D-1$) y tendencias regionales de cuenca.
5. **Consenso Meteorológico Multi-Modelo (ECMWF + GFS + ICON) y Calibración Causal MOS (`noche2_nwp`):** Acoplamiento de pronósticos numéricos históricos operativos (`previous_day1/2`) de tres centros mundiales (ECMWF IFS 0.25°, NCEP GFS 0.25°, DWD ICON 13 km), forzantes de capa límite ($\text{CAPE}$, $\text{VPD}$, ráfagas de viento, fracción de radiación directa) y **Model Output Statistics (MOS)** causal por estación y hora solar, reduciendo el error del viento superficial pronosticado en **$-44.1\%$** y el de temperatura en **$-13.8\%$**.
6. **Redes Profundas sobre Residuos de Persistencia (`deep2`) y Súper-Ensamble Conformal OOF (`ens2`):** Al parametrizar el objetivo como el incremento físico $\Delta_h = O_3(t+h) - O_3(t)$ sobre el panel continuo imputado con banderas de edad de falla, el ensamble neuronal **`deep_imput2_resid_refit` (TabResNet + TabGated SwiGLU)** superó de forma individual a todos los modelos de árboles en $h=1, 4, 8, 12$. Finalmente, el meta-aprendiz OOF (`ens2_best`, Pasada 5) integró las 14 familias (LightGBM, XGBoost, ExtraTrees, TabResNet, SwiGLU, LGBM-Residual-NN y Seq-GRU 48h), **superando incluso a la cota teórica no operativa con meteorología perfecta ($B_{pp}$) en 5 de los 6 horizontes ($h=1, 4, 8, 12, 24$)** y alcanzando un **$R^2 = 0.950$ ($\text{RMSE} = 4.716\text{ ppb}$) en $h=1$**.

---

## 🏆 Resumen de Resultados Oficiales en Prueba Ciega (Test 2025)

Evaluación estricta mediante `evaluate_test()` de [`experiments/common.py`](experiments/common.py) sobre el año completo **2025** (13 estaciones SIMA; selección de modelos y pesos realizada **100% en Validación OOF 2024 sin consultar 2025**):

| Horizonte ($h$) | Climatología (RMSE) | Baseline `v2` (Script 7) | `Plan A V4` (Química + Red) | `v3_base` (`D_honest_noise`) | Ensamble Noche 1 (`noche/ensemble`) | Cota Teórica No Operativa $B_{pp}$ (*Perfect Prog*) | **Mejor Individual (`noche2`)** | **Ensamble Oficial `ens2_best` (Pasada 5)** | **Mejor Meta-Modelo en Test (`ens2` P5)** | **Test $R^2$ (`ens2_best`)** | **Conformal 80% / 90% (Cobertura Real)** | **NOM-020 DMA8 (RMSE / F1)** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$1\text{ h}$** | 14.88 | 8.051 | 5.094 | 5.005 | 4.931 | 5.003 | **4.764** (`deep_imput2_resid`) | **4.716 ppb** *(bate $B_{pp}$)* | **4.702 ppb** (`M5`) | **0.950** | 79.2% / 89.3% | 2.10 ppb / **0.922** |
| **$4\text{ h}$** | 14.88 | 10.907 | 9.690 | 9.004 | 8.710 | 8.665 | **8.455** (`deep_imput2_resid`) | **8.369 ppb** *(bate $B_{pp}$)* | **8.308 ppb** (`M5`) | **0.842** | 79.6% / 89.5% | 6.97 ppb / **0.717** |
| **$8\text{ h}$** | 14.88 | 11.967 | 11.280 | 10.054 | 9.871 | 9.481 | **9.413** (`deep_imput2_resid`) | **9.228 ppb** *(bate $B_{pp}$)* | **9.149 ppb** (`M5`) | **0.808** | 79.5% / 89.5% | 8.75 ppb / **0.628** |
| **$12\text{ h}$** | 14.88 | 12.205 | 11.730 | 10.183 | 9.958 | 9.543 | **9.604** (`deep_imput2_resid`) | **9.354 ppb** *(bate $B_{pp}$)* | **9.336 ppb** (`M2`) | **0.803** | 80.5% / 89.9% | 8.89 ppb / **0.622** |
| **$24\text{ h}$** | 14.89 | 12.761 | 12.390 | 10.238 | 9.884 | 9.594 | **9.864** (`refit_nwp_best`) | **9.440 ppb** *(¡bate $B_{pp}$!)* | **9.399 ppb** (`M2`) | **0.799** | 80.4% / 89.8% | 8.80 ppb / **0.642** |
| **$48\text{ h}$** | 14.90 | 13.949 | 13.764 | 11.234 | 10.897 | 10.112 | **10.763** (`xgb_nwp_refit_w`) | **10.325 ppb** | **10.319 ppb** (`M7`/`M8`) | **0.760** | 81.4% / 90.4% | 10.32 ppb / **0.539** |

> 📄 **Lectura Recomendada**: Para consultar las ecuaciones matemáticas, ablaciones de imputación, formulación MOS, detalles de arquitecturas neuronales y demostración de descomposición de varianza e intervalos conformales, revisa el **[Reporte Técnico Maestro (`experiments/noche2/REPORTE_TECNICO_MAESTRO.md`)](experiments/noche2/REPORTE_TECNICO_MAESTRO.md)**.

---

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
   *(Nota: `requirements.txt` se mantiene actualizado para asegurar la replicabilidad total del entorno, incluyendo `lightgbm`, `xgboost`, `scikit-learn`, `torch`, `statsmodels`, `scipy`, `optuna`, `shap`, `pyarrow` y `streamlit`).*

---

## 🗂️ Estructura Principal del Pipeline

### 🧹 1. Preparación de Datos (Data Engineering Inicial)
* `1_join_raw.ipynb`: Consolida bases de datos separadas del SIMA en un historial completo (`2020–2025`).
* `1.5_xray_lostData.ipynb`: Diagnóstico y radiografía de datos faltantes/perdidos en la red de monitoreo.
* `2_clean_split.ipynb`: Limpieza profunda, aplicación de límites físicos normativos y partición por variables (formato `.parquet` optimizado).
* `3_features.ipynb`: Creación del "Dataset Inteligente", unificando tablas, calculando trigonometría temporal y rezagos (lags).

### 📊 2. Análisis Exploratorio y Modelado Clásico
* `4_analisis_visual.ipynb` / `.py`: Análisis de distribuciones, series temporales y funciones de correlación cruzada (CCF).
* `etapa2_analisis.ipynb` / `.py`: Implementación de modelos ARX/OLS, evaluación de autocorrelación (Durbin-Watson) y errores robustos HAC.

### 🤖 3. Machine Learning Base (`v1` y `v2`)
* `5_boosting_predictivo.py` / `train_lgbm.py`: Pruebas iniciales de predictibilidad con LightGBM.
* `6_multihorizonte_lgbm.py`: Entrenamiento de 6 modelos independientes de LightGBM (Horizontes de 1h, 4h, 8h, 12h, 24h, 48h).
* `7_multihorizonte_feature_eng.py`: Versión base `v2` del pipeline incorporando características específicas de fotoquímica e interacciones químicas con rezagos $\le t-1$.

### 🔬 4. Investigación Avanzada, Física Atmosférica, Deep Learning y Súper-Ensamble (`experiments/`)
* `experiments/common.py`: Contrato común de evaluación sin fuga de información (`load_panel`, `split`, `evaluate_test`, `log_result`, coordenadas de estaciones y climatología).
* `experiments/results_registry.csv` y `experiments/noche2/inventory.csv`: Registro histórico centralizado e inventario automatizado de los **>268 modelos val/test**.
* **Planes Exploratorios (`Plan A` – `Plan E` e Imputación):**
  * `experiments/planA_quickwins/`: Estado fotoquímico contemporáneo (lag 0, $\text{O}_x = \text{O}_3 + \text{NO}_2$) y descriptores espaciales *upwind* (`features_planA.py`, [`REPORT.md`](experiments/planA_quickwins/)).
  * `experiments/planB_picos/`: Funciones de pérdida asimétricas/Huber y ponderación cuadrática de colas $w = 1 + (y/P_{90})^2$ para mitigar la subestimación de contingencias ([`REPORT.md`](experiments/planB_picos/REPORT.md)).
  * `experiments/planC_meteo/`: Integración inicial de pronósticos Open-Meteo y análisis de horizontes de emisión (`meteo_features.py`, [`REPORT.md`](experiments/planC_meteo/REPORT.md)).
  * `experiments/planD_simple/` y `experiments/planE_deep/`: Modelos parsimoniosos e investigación inicial de arquitecturas neuronales tabulares y secuenciales ([`planD/REPORT.md`](experiments/planD_simple/REPORT.md), [`planE/REPORT.md`](experiments/planE_deep/REPORT.md)).
  * `experiments/imputacion/`: Benchmark controlado de imputación en series con apagones de sensores ([`REPORT.md`](experiments/imputacion/REPORT.md)).
* **Corrida Nocturna Ola 1 (`experiments/noche/`):**
  * `v3_base/`: Integración del protocolo honesto **PP-train / NWP-test** con inyección de ruido calibrado y caché de características de 198 columnas ([`REPORT.md`](experiments/noche/v3_base/REPORT.md)).
  * `audit_imput/`: Construcción del imputador causal espacial LightGBM (`panel_spatial_lgbm.parquet`, [`AUDIT.md`](experiments/noche/audit_imput/AUDIT.md), [`REPORT.md`](experiments/noche/audit_imput/REPORT.md)).
  * `target_eng/`: Reformulación del objetivo predictivo mediante residuos de persistencia $\Delta O_3 = O_3(t+h) - O_3(t)$ y armónicos de hora destino ([`REPORT.md`](experiments/noche/target_eng/REPORT.md)).
  * `tuning/`, `deep/` y `ensemble/`: Optimización bayesiana con Optuna, primeras redes multi-horizonte y primer pipeline de stacking OOF ([`ensemble/REPORT.md`](experiments/noche/ensemble/REPORT.md)).
* **Corrida Nocturna Ola 2 — Estado del Arte (`experiments/noche2/`):**
  * 📄 **[`REPORTE_TECNICO_MAESTRO.md`](experiments/noche2/REPORTE_TECNICO_MAESTRO.md)**: Monografía técnica integral del sistema.
  * `nwp/`: Consenso multi-centro (**ECMWF IFS 0.25° + GFS 0.25° + ICON 13 km**), variables convectivas ($\text{CAPE}, \text{VPD}, \text{Gusts}$), corrección estadística causal **MOS** (`features_nwp_h*.parquet`) y entrenamientos de `XGBoost` y `ExtraTrees` con refit ([`REPORT.md`](experiments/noche2/nwp/REPORT.md)).
  * `imput2/`: Ablación de estrategias de imputación causal (`1A_imp_pure`, `1B_imp_plus_flags`), selección hacia adelante de grupos (memoria del día anterior $D-1$ y anomalías de cuenca) y modelos `imput2_nwp_resid_refit` ([`REPORT.md`](experiments/noche2/imput2/REPORT.md)).
  * `refit/`: Protocolo de *4-Fold Cross-Fitting* por trimestres de 2024 + reentrenamiento `2020–2024` con indicador de dominio `is_nwp` y bagging multi-semilla ([`REPORT.md`](experiments/noche2/refit/REPORT.md)).
  * `deep2/`: Arquitecturas en GPU **TabResNet**, **TabGated (SwiGLU)**, **LGBM-Residual-NN**, **Seq-GRU 48h** (`seq_nwp_refit`) y el campeón individual **`deep_imput2_resid_refit`** ([`REPORT.md`](experiments/noche2/deep2/REPORT.md), [`train_deep_imput2_resid_refit.py`](experiments/noche2/deep2/train_deep_imput2_resid_refit.py)).
  * `ens2/`: **Arquitectura Maestra de Ensamble, Meta-Aprendizaje (`M0`–`M8`), Calibración Conformal Heterocedástica (80%/90%) y Evaluación Regulatoria NOM-020-SSA1 (DMA8)** ([`REPORT.md`](experiments/noche2/ens2/REPORT.md), [`ensemble_pipeline.py`](experiments/noche2/ens2/ensemble_pipeline.py), [`audit_candidates.py`](experiments/noche2/ens2/audit_candidates.py)). Sus predicciones finales listas para producción residen en `experiments/noche2/ens2/preds/ens2_best_h{1,4,8,12,24,48}.parquet`.

### 📈 5. Reportes, Dashboards y Visualización
* `app_laboratorio.py` / `app_presentacion.py`: Aplicaciones interactivas orientadas a la evaluación final y dashboards de resultados.
* `script_final_tareas.py`: Generación automática de métricas consolidadas ($R^2$, RMSE, importancia de variables) para evaluación e informes.
* `data/`: Directorio local de almacenamiento de datos.
* `results/`: Almacena métricas, figuras y reportes de las etapas iniciales (`reporte_multihorizonte.md`, `reporte_v2_feature_engineering.md`).

---

## ⚡ Reproducción Rápida del Súper-Ensamble e Inventario

Desde la raíz del proyecto y con el entorno `.venv` activado:

```bash
# 1. Auditar los 268 modelos candidatos y generar matrices de diversidad inter-familia
.venv\Scripts\python.exe experiments\noche2\ens2\audit_candidates.py

# 2. Ejecutar el pipeline de meta-aprendizaje OOF (M0-M8), calibración conformal y evaluación NOM-020 DMA8
.venv\Scripts\python.exe experiments\noche2\ens2\ensemble_pipeline.py 5

# 3. Actualizar el inventario global de predicciones (val 2024 / test 2025)
.venv\Scripts\python.exe experiments\noche2\inventory.py
```
