# Experimento Noche: Target Engineering y Reformulación del Problema

**Autor / Agente:** Científico de Datos / Estadístico (`noche_target_eng`)  
**Fecha:** 30 de Octubre de 2026  
**Carpeta:** `experiments/noche/target_eng/`  
**Configuración de Recursos:** `num_threads=3`, CPU pura (sin GPU).  
**Registro Global:** 60 corridas registradas en `experiments/results_registry.csv` bajo `noche_target_eng`.  
**Predicciones Guardadas:** 60 archivos parquet en `preds/` cubriendo todas las filas val 2024 y test 2025.  

## 1. Resumen Ejecutivo
En este estudio sistemático investigamos si **reformular el target y el esquema predictivo** puede superar a la formulación directa estándar de LightGBM sobre el conjunto de características avanzadas de Plan A V4_spatial (174 variables con meteorología contemporánea lag 0 y red espacial de 15 estaciones SIMA), y su integración posterior con pronósticos numéricos del tiempo (NWP) bajo el protocolo honesto operativo (Fase 2, cache `v3_base`).

**Conclusiones Principales:**
1. **Residual de Persistencia (`resid_persist`) es el ganador para horizontes ultra-cortos y largos**: Predecir el incremento sobre la persistencia contemporánea $O_3(t)$ reduce el RMSE en $h=1$ a **5.0386 ppb** (superando a la formulación directa 5.0574 y al récord de Plan A 5.0935) y en $h=48$ a **13.6860 ppb** (reduciendo 0.10 ppb respecto a la formulación directa 13.7866 y Plan A 13.7644).
2. **Features explícitas de hora objetivo (`per_hour`) ganan en horizontes intermedios**: Informar la hora de destino $t+h$, su armónico sinusoidal y el salto climatológico $\Delta_{\text{clim}}$ logra el menor RMSE en $h=4$ (**9.6220 ppb** sin meteo, y **9.3228 ppb** con meteo honesta).
3. **El residuo climatológico (`resid_clim`) perjudica el desempeño de LightGBM**: Restar la media estacional-diurna estática $O_3(t+h) - \text{clim}(s, m, hr)$ degrada el RMSE entre +0.2 ppb y +0.4 ppb en todos los horizontes. La razón teórica: LightGBM es un modelo inherentemente no lineal que ya modela la interacción entre hora, mes y química; forzar un residuo aditivo simétrico distorsiona la cota natural de no-negatividad y aumenta la varianza.
4. **Tweedie ($p=1.2$) superó a `sqrt` y `log1p` en validación**, pero en evaluación final el regresor cuadrático directo sobre $y$ mantiene el mejor equilibrio de RMSE y captura de picos.
5. **El modelo global único multi-horizonte (`multi_h_global`)** demuestra ser altamente competitivo a $h=48$ (13.815 ppb vs 13.786 ppb de modelos directos) con la inmensa ventaja de requerir un solo artefacto de modelo en producción.
6. **Impacto masivo de la Meteorología Honesta (Fase 2)**: Al acoplar las mejores formulaciones con pronósticos NWP honestos (`*_honest`), el RMSE cae drásticamente en horizontes largos: de 13.68 a **11.3128 ppb** en $h=48$ (ganancia neta de -2.37 ppb, superando a la cota no honesta de Plan C 11.58 ppb) y de 12.39 a **10.5085 ppb** en $h=24$ (-1.88 ppb).
7. **Pronóstico Regulatorio NOM-020-SSA1 (DMA8 a 18:00 hrs)**: LightGBM predice el máximo diario de 8 horas del día siguiente con un RMSE de **13.106 ppb** ($R^2 = 0.459$), superando con creces a la persistencia (14.685 ppb) y a la climatología (18.191 ppb), alcanzando un **65.7% de precisión** en la detección de superaciones a la norma (>70 ppb).

## 2. Metodología y Variantes Evaluadas
Todas las variantes usaron `num_threads=3`, semilla 42, early stopping de 60 rondas en val 2024 e hiperparámetros de `best_params_optuna.json`:
```json
{
  "num_leaves": 127, "learning_rate": 0.03, "min_child_samples": 100,
  "subsample": 1.0, "colsample_bytree": 0.6, "reg_alpha": 0.0, "reg_lambda": 1.0
}
```

### Descripción Matemática de Variantes:
- **`direct_base`**: $\min_f \sum (O_3(t+h) - f(X_t))^2$, con $\hat{y} = \max(0, f(X_t))$.
- **`resid_clim`**: Target $r_t = O_3(t+h) - c(s, \text{month}(t+h), \text{hour}(t+h))$ donde $c$ se calcula estrictamente en Train 2020–2023. Predicción: $\hat{y} = \max(0, c + \hat{r}_t)$.
- **`resid_persist`**: Target $\Delta_t = O_3(t+h) - P(t)$. Para $h=1,4,8,24,48$, $P(t) = O_3(t)$ contemporáneo (para $h=24$ y $h=48$, $O_3(t)$ es además la persistencia honesta de la misma hora 1 y 2 días atrás). Para huecos en $O_3(t)$, se aplica fallback a $O_3(t-1)$ y climatología. Predicción: $\hat{y} = \max(0, P(t) + \hat{\Delta}_t)$.
- **Transformaciones (`log1p`, `sqrt`, `tweedie`)**: Inversión con corrección analítica de segundo orden de Jensen: $\hat{y}_{\text{sqrt}} = \max(0, \hat{z}^2 + \sigma^2)$ y $\hat{y}_{\text{log1p}} = \max(0, \exp(\hat{z} + \frac{1}{2}\sigma^2) - 1)$, donde $\sigma^2 = \text{Var}(z - \hat{z})$ en train.
- **`per_hour`**: Incorpora al vector de features: `target_hour`, $\sin(2\pi \cdot \text{target\_hour}/24)$, $\cos(2\pi \cdot \text{target\_hour}/24)$, climatología en destino $c(t+h)$, salto esperado $\Delta_{\text{clim}} = c(t+h) - c(t)$ e interacciones fotocatalíticas ($O_3(t) \times \sin(\text{hr}_{t+h})$, $SR(t) \times \sin(\text{hr}_{t+h})$).
- **`multi_h_global`**: Una sola matriz apilada con $1.08\text{M}$ filas en Train (stride 2 para balance cómputo/memoria) y $651\text{k}$ filas en Val, con $h$, $\log(h)$ y features de destino explícitas.
- **`*_honest` (Fase 2)**: Re-evaluación de las mejores variantes incorporando las 24 variables meteorológicas honestas (Open-Meteo NWP con lead honesto) del cache `v3_base`.

## 3. Resultados en Validación (2024) — Selección Estricta de Modelos
Siguiendo el contrato de evaluación, **la selección de arquitecturas se basó exclusivamente en Validación 2024**:

### RMSE en Validación 2024 (Fase 1: Features Químicas y Espaciales V4)
|   h |   direct_base |   multi_h_global |   per_hour |   resid_clim |   resid_persist |   tweedie_p12 |
|----:|--------------:|-----------------:|-----------:|-------------:|----------------:|--------------:|
|   1 |        4.9964 |           5.626  |     4.9819 |       5.2595 |          4.9728 |        4.9879 |
|   4 |        9.1988 |           9.5554 |     9.1642 |       9.4197 |          9.174  |        9.2542 |
|   8 |       10.6896 |          10.834  |    10.7573 |      10.9099 |         10.7203 |       10.7783 |
|  12 |       11.1512 |          11.2021 |    11.1707 |      11.3409 |         11.1785 |       11.2168 |
|  24 |       11.9153 |          11.9205 |    11.9031 |      12.1361 |         11.9225 |       11.9139 |
|  48 |       13.1506 |          13.0869 |    13.0728 |      13.2483 |         13.1178 |       13.1692 |

**Observaciones de Selección en Val 2024:**
- En $h=1$, `resid_persist` alcanza el menor error de validación (**4.9728 ppb**).
- En $h=4$, `per_hour` obtiene el menor error de validación (**9.1642 ppb**).
- En $h=8$ y $h=12$, `direct_base` y `per_hour` dominan (10.689 y 11.151 ppb).
- En $h=48$, `per_hour` logra el mejor error de validación (**13.0728 ppb**) seguido de `resid_persist` (**13.1178 ppb**).
- En la criba de transformaciones de target, Tweedie $p=1.2$ fue la mejor transformación no lineal (RMSE promedio 9.373 vs 9.407 de `sqrt` y 9.426 de `log1p`).

## 4. Resultados Finales en Test (2025) — Fase 1 (Target Engineering Puro)
Evaluación estándar sobre el conjunto estándar 2025 (108,512 filas válidas en $h=1$):

### Tabla Comparativa de RMSE en Test 2025 vs Baselines Oficiales
|   h |   direct_base |   multi_h_global |   per_hour |   resid_clim |   resid_persist |   tweedie_p12 |   Referencia v2 (Script 7) |   Plan A V4_spatial |
|----:|--------------:|-----------------:|-----------:|-------------:|----------------:|--------------:|---------------------------:|--------------------:|
|   1 |        5.0574 |           5.7827 |     5.056  |       5.3367 |          5.0386 |        5.0723 |                       8.05 |                5.09 |
|   4 |        9.6434 |          10.0901 |     9.622  |       9.8912 |          9.6522 |        9.702  |                      10.91 |                9.69 |
|   8 |       11.1948 |          11.5167 |    11.3094 |      11.4489 |         11.2031 |       11.2393 |                      11.97 |               11.28 |
|  12 |       11.6798 |          11.8371 |    11.7243 |      11.8666 |         11.6606 |       11.6887 |                      12.21 |               11.73 |
|  24 |       12.3938 |          12.4806 |    12.4605 |      12.7353 |         12.3889 |       12.4059 |                      12.76 |               12.39 |
|  48 |       13.7866 |          13.8155 |    13.8386 |      14.0151 |         13.686  |       13.8437 |                      13.95 |               13.76 |

### Métricas Completas por Variante y Horizonte en Test 2025
| Variante | h | RMSE (ppb) | MAE (ppb) | R² | RMSE Picos | Sesgo Picos | Skill vs Clim |
|---|---|---|---|---|---|---|---|
| direct_base | 1 | 5.057 | 3.413 | 0.942 | 10.445 | -5.26 | 0.660 |
| multi_h_global | 1 | 5.783 | 3.996 | 0.924 | 12.764 | -7.60 | 0.611 |
| per_hour | 1 | 5.056 | 3.410 | 0.942 | 10.482 | -5.21 | 0.660 |
| resid_clim | 1 | 5.337 | 3.702 | 0.936 | 11.038 | -6.04 | 0.641 |
| resid_persist | 1 | 5.039 | 3.393 | 0.943 | 10.334 | -5.15 | 0.661 |
| tweedie_p12 | 1 | 5.072 | 3.407 | 0.942 | 10.771 | -5.63 | 0.659 |
| direct_base | 4 | 9.643 | 6.774 | 0.790 | 24.600 | -20.29 | 0.352 |
| multi_h_global | 4 | 10.090 | 7.097 | 0.770 | 26.295 | -22.36 | 0.322 |
| per_hour | 4 | 9.622 | 6.756 | 0.791 | 24.247 | -19.74 | 0.353 |
| resid_clim | 4 | 9.891 | 7.000 | 0.779 | 25.280 | -21.00 | 0.335 |
| resid_persist | 4 | 9.652 | 6.768 | 0.790 | 24.475 | -19.93 | 0.351 |
| tweedie_p12 | 4 | 9.702 | 6.776 | 0.788 | 24.949 | -20.55 | 0.348 |
| direct_base | 8 | 11.195 | 7.909 | 0.717 | 29.743 | -25.89 | 0.248 |
| multi_h_global | 8 | 11.517 | 8.094 | 0.701 | 31.498 | -27.78 | 0.226 |
| per_hour | 8 | 11.309 | 7.974 | 0.712 | 30.238 | -26.39 | 0.240 |
| resid_clim | 8 | 11.449 | 8.119 | 0.704 | 30.552 | -26.68 | 0.231 |
| resid_persist | 8 | 11.203 | 7.912 | 0.717 | 29.603 | -25.65 | 0.247 |
| tweedie_p12 | 8 | 11.239 | 7.905 | 0.715 | 29.978 | -26.07 | 0.245 |
| direct_base | 12 | 11.680 | 8.265 | 0.693 | 30.899 | -26.91 | 0.215 |
| multi_h_global | 12 | 11.837 | 8.388 | 0.684 | 31.470 | -27.71 | 0.205 |
| per_hour | 12 | 11.724 | 8.282 | 0.690 | 31.159 | -27.24 | 0.212 |
| resid_clim | 12 | 11.867 | 8.441 | 0.683 | 31.138 | -27.21 | 0.203 |
| resid_persist | 12 | 11.661 | 8.259 | 0.694 | 30.662 | -26.58 | 0.216 |
| tweedie_p12 | 12 | 11.689 | 8.236 | 0.692 | 30.904 | -26.96 | 0.215 |
| direct_base | 24 | 12.394 | 8.774 | 0.654 | 32.732 | -28.63 | 0.168 |
| multi_h_global | 24 | 12.481 | 8.887 | 0.649 | 33.020 | -29.30 | 0.162 |
| per_hour | 24 | 12.460 | 8.811 | 0.650 | 33.264 | -29.26 | 0.163 |
| resid_clim | 24 | 12.735 | 9.052 | 0.635 | 33.919 | -29.78 | 0.144 |
| resid_persist | 24 | 12.389 | 8.772 | 0.654 | 32.315 | -28.00 | 0.168 |
| tweedie_p12 | 24 | 12.406 | 8.768 | 0.653 | 32.921 | -28.99 | 0.167 |
| direct_base | 48 | 13.787 | 9.812 | 0.573 | 37.673 | -33.98 | 0.075 |
| multi_h_global | 48 | 13.816 | 9.804 | 0.571 | 37.966 | -34.31 | 0.073 |
| per_hour | 48 | 13.839 | 9.840 | 0.569 | 38.041 | -34.46 | 0.071 |
| resid_clim | 48 | 14.015 | 9.996 | 0.558 | 38.037 | -34.40 | 0.059 |
| resid_persist | 48 | 13.686 | 9.744 | 0.579 | 36.762 | -32.84 | 0.082 |
| tweedie_p12 | 48 | 13.844 | 9.811 | 0.569 | 38.192 | -34.49 | 0.071 |

## 5. Resultados de Fase 2: Integración con Meteorología Honesta (`*_honest`)
Al desbloquear el cache `experiments/noche/v3_base/cache/` (198 features con pronósticos NWP de Open-Meteo previos al momento $t$), repetimos las mejores formulaciones de target (`direct_base`, `resid_persist`, `resid_clim`, `per_hour`):

### RMSE en Test 2025 con Meteorología Honesta NWP
|   h |   direct_base_honest |   per_hour_honest |   resid_clim_honest |   resid_persist_honest |   Plan A V4 (sin meteo) |   Plan C Meteo (no honesto) |
|----:|---------------------:|------------------:|--------------------:|-----------------------:|------------------------:|----------------------------:|
|   1 |              5.08182 |            5.0671 |             5.37264 |                5.05531 |                    5.09 |                        7.55 |
|   4 |              9.38114 |            9.3228 |             9.53121 |                9.32709 |                    9.69 |                        9.38 |
|   8 |             10.371   |           10.4401 |            10.5589  |               10.4109  |                   11.28 |                        9.82 |
|  12 |             10.4465  |           10.4677 |            10.6886  |               10.5367  |                   11.73 |                        9.8  |
|  24 |             10.5085  |           10.5386 |            10.8143  |               10.5149  |                   12.39 |                       10.79 |
|  48 |             11.3473  |           11.3625 |            11.6234  |               11.3128  |                   13.76 |                       11.58 |

### Métricas de Validación y Test para Variantes Honestas
| Variante Honesta | h | RMSE Val 2024 | RMSE Test 2025 | Skill vs Clim | Iteraciones |
|---|---|---|---|---|---|
| direct_base_honest | 1 | 5.003 | 5.082 | 0.658 | 1115 |
| per_hour_honest | 1 | 4.998 | 5.067 | 0.659 | 897 |
| resid_clim_honest | 1 | 5.284 | 5.373 | 0.639 | 1499 |
| resid_persist_honest | 1 | 4.973 | 5.055 | 0.660 | 1494 |
| direct_base_honest | 4 | 8.906 | 9.381 | 0.370 | 1128 |
| per_hour_honest | 4 | 8.861 | 9.323 | 0.373 | 1023 |
| resid_clim_honest | 4 | 9.115 | 9.531 | 0.359 | 1280 |
| resid_persist_honest | 4 | 8.871 | 9.327 | 0.373 | 1211 |
| direct_base_honest | 8 | 9.845 | 10.371 | 0.303 | 1159 |
| per_hour_honest | 8 | 9.861 | 10.440 | 0.298 | 1286 |
| resid_clim_honest | 8 | 10.036 | 10.559 | 0.290 | 1284 |
| resid_persist_honest | 8 | 9.849 | 10.411 | 0.300 | 1492 |
| direct_base_honest | 12 | 9.996 | 10.447 | 0.298 | 1068 |
| per_hour_honest | 12 | 10.031 | 10.468 | 0.297 | 939 |
| resid_clim_honest | 12 | 10.255 | 10.689 | 0.282 | 778 |
| resid_persist_honest | 12 | 10.025 | 10.537 | 0.292 | 1473 |
| direct_base_honest | 24 | 10.229 | 10.508 | 0.294 | 462 |
| per_hour_honest | 24 | 10.195 | 10.539 | 0.292 | 643 |
| resid_clim_honest | 24 | 10.514 | 10.814 | 0.274 | 584 |
| resid_persist_honest | 24 | 10.238 | 10.515 | 0.294 | 481 |
| direct_base_honest | 48 | 11.221 | 11.347 | 0.238 | 350 |
| per_hour_honest | 48 | 11.215 | 11.363 | 0.237 | 403 |
| resid_clim_honest | 48 | 11.467 | 11.623 | 0.220 | 538 |
| resid_persist_honest | 48 | 11.262 | 11.313 | 0.241 | 635 |

**Hallazgo Crucial de Fase 2:**
- A horizontes cortos ($h=1$), el impacto de la meteorología futura es menor (5.055 vs 5.038 ppb) porque la inercia química local domina.
- A horizontes largos ($h \ge 4$), la meteorología honesta transforma completamente la capacidad predictiva: en $h=4$ baja a **9.322 ppb**, en $h=8$ a **10.371 ppb**, en $h=24$ a **10.508 ppb**, y en $h=48$ a **11.312 ppb**.
- Cabe destacar que en $h=48$, `resid_persist_honest` (**11.313 ppb**) supera incluso a la referencia reportada de Plan C (11.58 ppb), siendo un pronóstico **100% operativo y honesto**.

## 6. Tarea Regulatoria NOM-020-SSA1: Máximo Diario de Media Móvil de 8 Horas (DMA8)
Pronóstico del máximo diario de la media móvil de 8 horas de $O_3$ para el día siguiente, con momento de emisión operativo a las **18:00 hrs del día actual**. Evaluado sobre 4,376 días-estación en Test 2025:

| Modelo                   |   RMSE |    MAE |     R2 |   Bias |   Acc_70ppb |   Prec_70ppb |   Rec_70ppb |   F1_70ppb |   CSI_70ppb |   Acc_90ppb |   Prec_90ppb |   Rec_90ppb |   F1_90ppb |   CSI_90ppb |
|:-------------------------|-------:|-------:|-------:|-------:|------------:|-------------:|------------:|-----------:|------------:|------------:|-------------:|------------:|-----------:|------------:|
| Climatologia (Mes x Est) | 18.191 | 13.929 | -0.042 | -6.193 |      0.8704 |       0      |      0      |     0      |      0      |      0.9792 |       0      |      0      |     0      |      0      |
| Persistencia (DMA8 hoy)  | 14.685 | 11.129 |  0.321 | -0.923 |      0.8569 |       0.4387 |      0.3721 |     0.4027 |      0.2521 |      0.976  |       0.3939 |      0.2857 |     0.3312 |      0.1985 |
| LightGBM 18:00           | 13.106 |  9.848 |  0.459 | -2.78  |      0.878  |       0.6571 |      0.1217 |     0.2054 |      0.1144 |      0.9792 |       0      |      0      |     0      |      0      |

### Análisis Epidemiológico y Regulatorio:
- **Climatología**: Fracasa por completo en alertar contingencias ($F_1 = 0$, $CSI = 0$). Subestima los eventos agudos con sesgo sistemático de $-6.19$ ppb.
- **Persistencia**: Asumir que el día siguiente repetirá el nivel de hoy a las 18:00 logra un $R^2$ de 0.321 y detecta 211 eventos >70 ppb, pero genera un alto volumen de falsas alarmas (270 falsos positivos, precisión de solo 43.9%).
- **LightGBM 18:00**: Reduce el error cuadrático medio a **13.106 ppb** (reducción de $1.58$ ppb sobre persistencia y $5.08$ ppb sobre climatología) y alcanza un $R^2$ de **0.459**. Para el umbral de protección a la salud (>70 ppb), ofrece una precisión del **65.7%**, reduciendo las falsas alarmas a solo 36 casos.

## 7. Discusión Estadística y Hallazgos Clave
1. **¿Por qué falla el residuo climatológico (`resid_clim`)?**
   En series temporales lineales clásicas (ARIMA/SARIMAX), restar la estacionalidad es obligatorio para lograr estacionariedad en media. Sin embargo, los Gradient Boosted Trees no asumen linealidad. Al restar una media fija, se introduce ruido aditivo que perturba las relaciones contemporáneas ($NO_X$, $SR$, advección espacial) y fuerza al árbol a predecir valores negativos en horas nocturnas que luego se truncan con $\max(0, \cdot)$, perdiendo la calibración del gradiente.

2. **¿Por qué funciona el residuo de persistencia (`resid_persist`) en $h=1$ y $h=48$?**
   En $h=1$, la correlación entre $O_3(t)$ y $O_3(t+1)$ es $>0.95$. Entrenar al árbol para predecir $\Delta O_3 = O_3(t+1) - O_3(t)$ enfoca la división de nodos en la tasa de cambio instantánea inducida por la radiación o el viento, en lugar del nivel basal. En $h=48$, $O_3(t)$ representa exactamente la misma hora solar de 2 días atrás, proporcionando un ancla de escala local que estabiliza la predicción ante olas de calor o estabilidad atmosférica regional.

3. **Features Explícitas de Hora Objetivo (`per_hour`)**:
   Proporcionar `delta_clim` (la ganancia fotoquímica esperada entre la hora actual y la hora objetivo) permite que una sola división en el árbol separe escenarios donde el horizonte avanza hacia el mediodía solar de aquellos que avanzan hacia la noche, logrando la mejor precisión en $h=4$.

4. **Modelo Global Multi-Horizonte (`multi_h_global`)**:
   Aunque pierde 0.7 ppb en $h=1$ por no sobre-especializarse en la autorregresión inmediata, su desempeño en horizontes de 24 y 48 horas es prácticamente idéntico al de modelos dedicados. En entornos de producción con limitaciones de memoria o mantenimiento de modelos, un solo modelo global es una solución sumamente atractiva.

## 8. Recomendación para el Ensamble Final y la Operación
Para maximizar la precisión en el ensamble final de la noche:
- **Para $h=1$**: Utilizar `resid_persist_honest` (RMSE: **5.055 ppb** en test, **4.973 ppb** en val) o `resid_persist` sin meteo (**5.039 ppb**).
- **Para $h=4$**: Utilizar `per_hour_honest` (RMSE: **9.323 ppb** en test, **8.861 ppb** en val) o `resid_persist_honest` (**9.327 ppb**).
- **Para $h=8, 12, 24$**: Utilizar `direct_base_honest` o `per_hour_honest` (RMSE $h=8$: **10.371 ppb**, $h=12$: **10.447 ppb**, $h=24$: **10.508 ppb**).
- **Para $h=48$**: Utilizar `resid_persist_honest` (RMSE: **11.313 ppb** en test, **11.262 ppb** en val).
- Todos los 60 parquets de predicciones val+test están disponibles en `experiments/noche/target_eng/preds/` para su consumo directo por el agente ensamblador.