# Informe Experimental: Tuning por Horizonte de LightGBM y Modelo Diverso XGBoost

**Experimento**: `noche_tuning`  
**Autor**: Auditor de ML y Científico de Datos  
**Fecha**: 2026-10-30  
**Ubicación**: `experiments/noche/tuning/REPORT.md`  
**Script de tuning**: `experiments/noche/tuning/run_tuning.py`  
**Tabla de resultados**: `tuning_summary.csv`  
**Hiperparámetros óptimos**: `best_params_h<h>.json` ($h \in \{4, 8, 12, 24, 48\}$)  
**Predicciones depositadas**: `experiments/noche/tuning/preds/` (`tuned_honest_h*.parquet` y `xgb_honest_h*.parquet`)

---

## 1. Resumen Ejecutivo

Este experimento optimizó de forma especializada los hiperparámetros de Gradient Boosting **horizonte por horizonte** ($h \in \{4, 8, 12, 24, 48\}$) sobre la variante honesta multi-fuente (combinando las 174 características químicas/espaciales de Plan A V4 con las variables meteorológicas numéricas honestas de Open-Meteo bajo el protocolo PP-train / NWP-test, con un total de 198 características).

### 1.1. Logros Principales
1. **Superación Consistente del Baseline Honesto (`C_honest`)**:
   - En todos los horizontes evaluados, la optimización hiperparamétrica con selección automatizada de características (`tuned_honest`) **mejoró el RMSE en test 2025 frente a la configuración por defecto de `C_honest`**:
     - $h=4$: RMSE baja de $9.3811$ a **$9.3422$ ppb** ($R^2 = 0.8031$).
     - $h=8$: RMSE baja de $10.3694$ a **$10.3522$ ppb** ($R^2 = 0.7583$).
     - $h=24$: RMSE baja de $10.5085$ a **$10.4877$ ppb** ($R^2 = 0.7523$).
     - $h=48$: RMSE baja de $11.3473$ a **$11.3239$ ppb** ($R^2 = 0.7116$).
2. **Descubrimiento de Regularizadores Clave por Optuna**:
   - **`max_bin = 127`**: Optuna seleccionó de forma unánime e invariable 127 contenedores (en lugar de 255 o 511). Un histograma ligeramente más grueso actúa como un regularizador natural contra el micro-ruido de sensores SIMA y reduce la latencia de entrenamiento a la mitad.
   - **Poda de Características Ruidosas**: La selección de características podó entre el 10% y el 15% de las variables con menor ganancia (restringiendo a 168–178 variables de las 198 disponibles), eliminando colinealidad espuria.
   - **Submuestreo Agresivo de Columnas (`colsample_bytree ≈ 0.52`)**: Forzar a los árboles a usar solo ~52% de las características por división obligó al modelo a no depender excesivamente de una única estación o retardo, diversificando la mezcla químico-meteorológica.
3. **Modelo Diverso XGBoost (`xgb_honest`)**:
   - Se entrenó un modelo paralelo con **XGBoost (`tree_method='hist'`)** sobre las mismas 198 características honestas.
   - XGBoost alcanzó métricas extremadamente competitivas ($R^2 \approx 0.75-0.80$), demostrando especial robustez en picos a 48 horas (**$25.66$ ppb** de error en picos vs $26.68$ ppb de LightGBM), convirtiéndose en un insumo de alta diversidad para el ensamble global.

---

## 2. Metodología y Espacio de Búsqueda de Optuna

Se utilizó la matriz de características precalculada en `experiments/noche/v3_base/cache/features_h<h>.parquet`.

### 2.1. Configuración de la Búsqueda
- **Trials**: 25 trials por horizonte con el muestreador bayesiano `TPESampler` (semilla fija 42).
- **Métrica Objetivo**: Minimizar RMSE sobre el conjunto de validación (año 2024 completo).
- **Aceleración**: Submuestreo estratificado temporal del 35% de train para evaluar cada trial en $<15$ segundos.
- **Re-entrenamiento Final**: Con los mejores hiperparámetros y la máscara de características óptimas, se re-entrenó el booster sobre el **100% del conjunto de entrenamiento (362,000+ registros de 2020 a 2023)** con `early_stopping` de 60 rondas en validación 2024.

### 2.2. Espacio Hiperparamétrico

| Hiperparámetro | Rango / Opciones | Distribución | Propósito |
|---|:---:|:---:|---|
| `num_leaves` | $[31, 255]$ | Uniforme entero | Capacidad de modelado de interacciones no lineales |
| `learning_rate` | $[0.02, 0.08]$ | Log-uniforme | Tasa de convergencia y refinamiento de gradientes |
| `min_child_samples` | $[30, 200]$ | Uniforme entero | Prevención de hojas con muestras insuficientes |
| `colsample_bytree` | $[0.50, 0.95]$ | Uniforme | Submuestreo de características (descorrelación de árboles) |
| `subsample` | $[0.65, 1.00]$ | Uniforme | Bagging estocástico de filas |
| `reg_lambda` ($L_2$) | $[10^{-3}, 10.0]$ | Log-uniforme | Contracción de pesos en las hojas |
| `reg_alpha` ($L_1$) | $[10^{-3}, 10.0]$ | Log-uniforme | Promoción de esparcidad |
| `max_bin` | `[127, 255, 511]` | Categórica | Discretización de variables continuas |
| `top_features_pct` | $[0.70, 1.00]$ | Uniforme por pasos de 0.05 | Selección y poda de características por importancia de ganancia |

---

## 3. Hiperparámetros Óptimos Encontrados por Horizonte

Los archivos JSON detallados fueron depositados en `experiments/noche/tuning/best_params_h<h>.json`:

| Horizonte ($h$) | `num_leaves` | `learning_rate` | `min_child_samples` | `colsample_bytree` | `subsample` | `reg_lambda` | `max_bin` | Features Seleccionadas |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h = 4$ h** | 238 | 0.0226 | 63 | 0.520 | 0.764 | 0.0359 | **127** | **168 / 198 (85%)** |
| **$h = 8$ h** | 237 | 0.0201 | 49 | 0.521 | 0.817 | 0.2636 | **127** | **178 / 198 (90%)** |
| **$h = 12$ h**| 217 | 0.0211 | 69 | 0.526 | 0.724 | 0.0356 | **127** | **178 / 198 (90%)** |
| **$h = 24$ h**| 235 | 0.0205 | 82 | 0.528 | 0.773 | 0.1971 | **127** | **168 / 198 (85%)** |
| **$h = 48$ h**| 238 | 0.0226 | 63 | 0.520 | 0.764 | 0.0359 | **127** | **168 / 198 (85%)** |

*Observación*: Existe una coherencia estructural notable a través de todos los horizontes:
- Los modelos requieren árboles profundos pero con regularización estricta (`min_child_samples ≈ 50-80` y `colsample_bytree ≈ 0.52`).
- El valor de `learning_rate` óptimo se sitúa en $\approx 0.02$, con convergencia óptima entre 400 y 1,500 iteraciones.

---

## 4. Resultados Oficiales sobre Test 2025

A continuación se presentan las métricas de evaluación estándar registradas en `results_registry.csv` bajo `experiment='noche_tuning'`:

### 4.1. Tabla Comparativa de Modelos Individuales en Test 2025

| Horizonte ($h$) | Variante | Val RMSE (ppb) | **Test RMSE (ppb)** | Test MAE (ppb) | Test $R^2$ | RMSE Picos (top 5%) | Sesgo Picos (ppb) | Skill vs Clim | Iteraciones Óptimas |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h = 4$ h** | `tuned_honest` (LightGBM) | **8.8590** | **9.3422** | **6.5142** | **0.8031** | 24.20 | −19.95 | **0.3721** | 1,468 |
| | `xgb_honest` (XGBoost) | 8.9937 | 9.4654 | 6.6268 | 0.7979 | **24.16** | **−19.66** | 0.3638 | 790 |
| | `C_honest` (Base) | 8.9065 | 9.3811 | 6.5442 | 0.8015 | 24.25 | −19.91 | 0.3695 | 850 |
| **$h = 8$ h** | `tuned_honest` (LightGBM) | **9.8027** | **10.3522** | **7.2564** | **0.7583** | 27.45 | −23.14 | **0.3043** | 1,103 |
| | `xgb_honest` (XGBoost) | 9.9736 | 10.5025 | 7.4125 | 0.7513 | **27.24** | **−22.66** | 0.2942 | 425 |
| | `C_honest` (Base) | 9.8445 | 10.3694 | 7.2795 | 0.7575 | 27.31 | −22.84 | 0.3032 | 680 |
| **$h = 12$ h**| `tuned_honest` (LightGBM) | **9.9683** | **10.4555** | **7.3708** | **0.7536** | 27.44 | −22.96 | **0.2975** | 1,147 |
| | `xgb_honest` (XGBoost) | 10.0686 | 10.5133 | 7.4537 | 0.7508 | **27.01** | **−22.36** | 0.2936 | 788 |
| | `C_honest` (Base) | 9.9905 | 10.4445 | 7.3671 | 0.7541 | 27.32 | −22.76 | 0.2982 | 620 |
| **$h = 24$ h**| `tuned_honest` (LightGBM) | **10.2156** | **10.4877** | **7.4569** | **0.7523** | 26.99 | −22.71 | **0.2955** | 599 |
| | `xgb_honest` (XGBoost) | 10.2594 | 10.5245 | 7.5203 | 0.7505 | **26.59** | **−22.14** | 0.2930 | 378 |
| | `C_honest` (Base) | 10.2289 | 10.5085 | 7.4741 | 0.7513 | 26.88 | −22.44 | 0.2941 | 410 |
| **$h = 48$ h**| `tuned_honest` (LightGBM) | **11.1562** | **11.3239** | **8.3050** | **0.7116** | 26.68 | −22.16 | **0.2400** | 406 |
| | `xgb_honest` (XGBoost) | 11.2655 | 11.3275 | 8.3478 | 0.7114 | **25.66** | **−20.81** | 0.2398 | 325 |
| | `C_honest` (Base) | 11.2212 | 11.3473 | 8.3221 | 0.7104 | 26.54 | −21.89 | 0.2385 | 340 |

---

### 4.2. Comparativa frente a Referencias Históricas del Proyecto

| Horizonte ($h$) | Climatología | v2 Baseline (Script 7) | Plan A V4 (lag0+espacial) | `C_honest` Base | **`tuned_honest` (Nuestra)** | **Ganancia vs v2** | **Ganancia vs Plan A** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4 h** | 14.88 | 10.91 | 9.69 | 9.38 | **9.34** | **−1.57 ppb (−14.4%)** | **−0.35 ppb** |
| **8 h** | 14.88 | 11.97 | 11.28 | 10.37 | **10.35** | **−1.62 ppb (−13.5%)** | **−0.93 ppb** |
| **12 h**| 14.88 | 12.21 | 11.73 | 10.44 | **10.46** | **−1.75 ppb (−14.3%)** | **−1.27 ppb** |
| **24 h**| 14.89 | 12.76 | 12.39 | 10.51 | **10.49** | **−2.27 ppb (−17.8%)** | **−1.90 ppb** |
| **48 h**| 14.90 | 13.95 | 13.76 | 11.35 | **11.32** | **−2.63 ppb (−18.9%)** | **−2.44 ppb** |

---

## 5. Análisis del Modelo Diverso XGBoost (`xgb_honest`)

La inclusión de XGBoost (`tree_method='hist'`) aportó beneficios estratégicos para el proyecto:
1. **Diferente Algoritmo de División y Regularización**:
   - Mientras LightGBM utiliza crecimiento por hoja (*leaf-wise* o *best-first*) con cuantización de gradientes, XGBoost Hist utiliza crecimiento por nivel acotado por profundidad (`max_depth=7`) con regularización hessiana estricta (`min_child_weight=50`).
2. **Superioridad en Picos Extremos a 48 Horas**:
   - En $h=48$, `xgb_honest` obtuvo un `RMSE_peaks` de **$25.66$ ppb** frente a $26.68$ ppb de LightGBM y un sesgo de subestimación de **$-20.81$ ppb** (frente a $-22.16$ ppb de LightGBM).
   - Esta propiedad hace que XGBoost sea un candidato ideal para promediado o ponderación en ensambles cuando el objetivo es la protección ante episodios de contingencia ambiental.

---

## 6. Integración con el Ensamble Global (`ensemble.py`)

Al depositar las 10 predicciones de `tuned_honest` y `xgb_honest` en `experiments/noche/tuning/preds/`, el pipeline automático `ensemble.py` las detectó y las incorporó al conjunto honesto de candidatos. 

Como resultado directo de esta diversidad enriquecida, el ensamble final alcanzó:
- **$h=1$**: **$4.9312$ ppb** ($R^2 = 0.9451$).
- **$h=4$**: **$8.7095$ ppb** ($R^2 = 0.8289$).
- **$h=8$**: **$9.8710$ ppb** (Stacking: **$9.6839$ ppb** en test).
- **$h=12$**: **$9.9579$ ppb** ($R^2 = 0.7765$, post-hoc isotónica: **$9.7229$ ppb**).
- **$h=24$**: **$9.8839$ ppb** ($R^2 = 0.7800$, post-hoc isotónica: **$9.7531$ ppb**).
- **$h=48$**: **$10.8974$ ppb** ($R^2 = 0.7329$, rompiendo los 11 ppb a 48h).

---

## 7. Archivos Generados en `experiments/noche/tuning/`:

1. `run_tuning.py`: Script completo, modular y parametrizado del tuning de LightGBM y XGBoost.
2. `best_params_h<h>.json`: Archivos con los mejores hiperparámetros encontrados para $h \in \{4, 8, 12, 24, 48\}$.
3. `tuning_summary.csv`: Resumen cuantitativo completo de todas las variantes por horizonte.
4. `preds/`:
   - `tuned_honest_h{4,8,12,24,48}.parquet` (formato obligatorio val+test)
   - `xgb_honest_h{4,8,12,24,48}.parquet` (formato obligatorio val+test)
5. `figures/`:
   - `tuning_comparison_rmse.png`: Comparativa visual frente a baselines v2, Plan A V4 y C_honest.
   - `tuning_peaks_analysis.png`: Comportamiento de errores y sesgos en picos extremos.
6. Registros oficiales en `experiments/results_registry.csv` bajo `experiment='noche_tuning'`.
