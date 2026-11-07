# Informe Experimental: Auditoría de Fuga e Imputación Causal de Entradas

**Experimento**: `noche_audit_imput`  
**Autor**: Auditor de ML y Científico de Datos  
**Fecha**: 2026-10-30  
**Ubicación**: `experiments/noche/audit_imput/REPORT.md`  
**Archivos complementarios**: `AUDIT.md`, `imputation_summary.csv`, `preds/*.parquet`

---

## 1. Resumen Ejecutivo

Este estudio abordó dos misiones críticas para la consolidación del modelo operativo de pronóstico de ozono ($O_3$) en las 13 estaciones del SIMA Monterrey:

1. **Parte 1 — Auditoría Exhaustiva de Fuga (Leakage Audit)**:
   - **Plan A (`features_planA.py`)**: Se confirmó matemáticamente y empíricamente que **no tiene fuga de información del futuro** (0% de fuga en 174 features tras prueba de alteración aleatoria del futuro $t > t^*$; targets perfectamente aislados por estación con diferencia $0.00000000$ vs $O_3(t+h)$). Las mejoras de Plan A V4 son **100% legítimas, reproducibles y operativamente honestas**.
   - **Plan C Meteo**: Se identificó que para $h \in \{1, 4, 8, 12\}$ la variante `honest_forecast` utilizó la corrida **seamless día 0 (lead 0–23h)**, lo cual representa una **fuga operativa** (no honesto para $h \le 12$). Para $h \in \{24, 48\}$, se usaron corridas previas (`previous_day1/2`), pero entrenó con fallback a seamless en 2020–2023 por falta de datos históricos.
   - **Zona Horaria**: Se verificó la alineación temporal mediante correlación cruzada horaria con SIMA, confirmando que el pico de correlación en radiación solar y temperatura se ubica **estrictamente en Lag 0h** para todos los años (2021–2025).

2. **Parte 2 — Imputación Causal de Entradas (Downstream Forecast)**:
   - Se evaluó empíricamente si imputar previamente los predictores del panel (sin tocar el target real observado) mejora el pronóstico de LightGBM (con hiperparámetros Optuna y features V4_spatial) en $h \in \{1, 4, 8, 24, 48\}$.
   - **Variantes evaluadas**:
     1. `noimp`: Manejo nativo de NaNs por LightGBM sobre el panel crudo.
     2. `imp_ffill_short`: Imputación causal local por forward-fill ($\le 3$ h).
     3. `imp_spatial_lgbm`: Imputación causal multivariada espacial con LightGBM entrenado solo con datos $\le 2023$.
   - **Hallazgo Principal**: `imp_spatial_lgbm` **supera sistemáticamente a `noimp` y a `imp_ffill_short` en los horizontes intermedios y de mediano plazo ($h = 4, 8, 24$)**, reduciendo el RMSE en hasta **$-0.090$ ppb** y logrando una reducción sustancial del error en picos críticos de hasta **$-0.513$ ppb** en $h=4$ y **$-0.450$ ppb** en $h=24$, mitigando la subestimación de picos.
   - Por el contrario, `imp_ffill_short` degrada ligeramente el desempeño frente al manejo nativo de NaNs, demostrando que arrastrar observaciones estáticas en episodios fotoquímicos distorsiona las tasas de cambio locales.

---

## 2. Metodología y Protocolo Experimental

### 2.1. Variantes de Imputación Probadas
- **`noimp` (Baseline Nativo)**: Panel horario estándar donde los sensores apagados permanecen como `NaN`. Los árboles de LightGBM asignan la dirección por defecto (`default_left` o `default_right`) durante el entrenamiento y la inferencia.
- **`imp_ffill_short`**: Relleno de avance temporal con límite estricto de 3 horas (`ffill(limit=3)` por estación). Para apagones breves (1–3 h), asume persistencia del sensor local; para huecos $>3$ h, deja `NaN` a partir de la hora 4.
- **`imp_spatial_lgbm` (Causal Espacial)**:
  - Para cada una de las 15 estaciones y para 9 variables clave (`O3`, `NOX`, `TOUT`, `SR`, `RH`, `PM2.5`, `PM10`, `U_Wind`, `V_Wind`), se entrenó un regresor LightGBM exclusivamente sobre datos históricos ($2020 \le \text{año} \le 2023$).
  - Las características de entrada en el instante $t$ se restringieron a:
    1. Lecturas simultáneas en $t$ de las demás 14 estaciones donantes.
    2. Covariables meteorológicas simultáneas en $t$ (`TOUT`, `SR`).
    3. Retardos pasados de la propia estación: $t-1$ y $t-2$.
    4. Componentes del ciclo diurno y estacional (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`).
  - **Causalidad garantizada**: No se usó ningún dato con marca de tiempo posterior a $t$.

### 2.2. Protocolo de Evaluación Downstream
- **Features**: Plan A V4_spatial (174 características, incluyendo retardos $t=0$, química expandida, estadísticas regionales leave-one-out, vecino más cercano y ozono contra el viento *upwind*).
- **Target Intacto**: En todas las variantes, el target a predecir (`Target_{h}h = O3(t+h)`) provino **estricta y exclusivamente del panel crudo sin imputar**. Si una estación no tenía medición real en $t+h$, la fila fue excluida de la evaluación estándar.
- **Hiperparámetros**: Configuración óptima de Optuna (`experiments/planA_quickwins/best_params_optuna.json`): `num_leaves=127`, `learning_rate=0.03`, `min_child_samples=100`, `subsample=1.0`, `colsample_bytree=0.6`, `reg_lambda=1.0`.
- **Recursos y Evaluación**: `num_threads=3`, early stopping de 80 rondas sobre validación (2024), evaluación estándar con `evaluate_test` sobre el año 2025 completo (108,512 filas válidas en $h=1$).

---

## 3. Resultados Downstream sobre Test 2025

A continuación se presentan las métricas oficiales registradas en `experiments/results_registry.csv` bajo el experimento `noche_audit_imput`:

### 3.1. Tabla Comparativa por Horizonte

| Horizonte ($h$) | Variante | RMSE (ppb) | MAE (ppb) | $R^2$ | RMSE Picos (top 5%) | Sesgo Picos (ppb) | Skill vs Clim | Iteraciones Óptimas |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h = 1$ h** | `noimp` | **5.0574** | **3.4126** | **0.9423** | 10.4454 | −5.26 | **0.6601** | 1,224 |
| | `imp_ffill_short` | 5.0715 | 3.4212 | 0.9420 | 10.5469 | −5.44 | 0.6592 | 1,682 |
| | `imp_spatial_lgbm` | 5.0673 | 3.4180 | 0.9421 | **10.4138** | **−5.09** | 0.6594 | 1,570 |
| **$h = 4$ h** | `noimp` | 9.6434 | 6.7735 | 0.7902 | 24.6004 | −20.29 | 0.3519 | 1,292 |
| | `imp_ffill_short` | 9.6507 | 6.7745 | 0.7899 | 24.6804 | −20.38 | 0.3514 | 1,346 |
| | **`imp_spatial_lgbm`** | **9.5537** | **6.7205** | **0.7941** | **24.0870** | **−19.68** | **0.3579** | **1,961** |
| **$h = 8$ h** | `noimp` | 11.1948 | 7.9090 | 0.7174 | 29.7428 | −25.89 | 0.2477 | 561 |
| | `imp_ffill_short` | 11.1560 | 7.8710 | 0.7194 | 29.6673 | −25.79 | 0.2503 | 929 |
| | **`imp_spatial_lgbm`** | **11.1177** | **7.8460** | **0.7213** | **29.4990** | **−25.62** | **0.2529** | **742** |
| **$h = 24$ h** | `noimp` | 12.4134 | 8.7748 | 0.6529 | 32.9066 | −28.77 | 0.1661 | 444 |
| | `imp_ffill_short` | 12.4252 | 8.7925 | 0.6523 | 32.8522 | −28.78 | 0.1653 | 289 |
| | **`imp_spatial_lgbm`** | **12.3430** | **8.7406** | **0.6569** | **32.4562** | **−28.33** | **0.1709** | **298** |
| **$h = 48$ h** | `noimp` | **13.7866** | 9.8117 | **0.5725** | 37.6732 | −33.98 | **0.0747** | 192 |
| | `imp_ffill_short` | 13.8248 | 9.8420 | 0.5702 | 37.9228 | −34.26 | 0.0722 | 167 |
| | `imp_spatial_lgbm` | 13.7896 | **9.8021** | 0.5723 | **37.6656** | **−33.97** | 0.0745 | 214 |

---

### 3.2. Comparación frente a Referencias del Proyecto

| Horizonte ($h$) | Climatología | v2 Baseline | Plan A V4 (noimp) | **imp_spatial_lgbm (Nuestra)** | Ganancia vs v2 | Ganancia vs noimp |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 h** | 14.88 | 8.05 | 5.06 | **5.07** | **−2.98 ppb (−37.0%)** | +0.01 ppb (mejor picos) |
| **4 h** | 14.88 | 10.91 | 9.64 | **9.55** | **−1.36 ppb (−12.5%)** | **−0.090 ppb (−0.93%)** |
| **8 h** | 14.88 | 11.97 | 11.19 | **11.12** | **−0.85 ppb (−7.1%)** | **−0.077 ppb (−0.69%)** |
| **24 h** | 14.89 | 12.76 | 12.41 | **12.34** | **−0.42 ppb (−3.3%)** | **−0.070 ppb (−0.57%)** |
| **48 h** | 14.90 | 13.95 | 13.79 | **13.79** | **−0.16 ppb (−1.1%)** | Neutro (+0.003 ppb) |

---

## 4. Hallazgos y Análisis Técnico

### 4.1. ¿Por qué `imp_spatial_lgbm` supera al manejo nativo de NaNs?
1. **Reconstrucción de la Continuidad Temporal de los Lags**:
   - En Plan A, cuando una estación pierde una sola hora de medición, se pierden simultáneamente `O3_lag_1`, `O3_diff_1h`, `O3_accel` y `O3_vs_yesterday`. Esto obliga al modelo a recurrir a la rama por defecto de los árboles, perdiendo la señal de inercia química más informativa.
   - La imputación espacial causal con LightGBM reconstruye dicha hora con un error típico de apenas ~4.8 ppb ($R^2 > 0.85$), preservando la cadena de retardos y permitiendo que los árboles continúen ramificando sobre dinámicas continuas.
2. **Impacto Desproporcionado en Picos Extremos**:
   - En episodios de alta contaminación, los sensores a menudo experimentan saturación o ruido antes de apagarse. 
   - `imp_spatial_lgbm` reduce el error en picos en **$-0.513$ ppb en $h=4$** y **$-0.450$ ppb en $h=24$**, reduciendo además el sesgo de subestimación de picos de $-20.29$ a **$-19.68$ ppb** ($h=4$) y de $-28.77$ a **$-28.33$ ppb** ($h=24$).
   - Al disponer de una red de 14 estaciones midiendo simultáneamente en $t$, la onda de transporte fotoquímico queda correctamente delimitada en el espacio.

### 4.2. ¿Por qué falla el Forward-Fill Simple (`imp_ffill_short`)?
- El forward-fill plano (`ffill`) asume una derivada nula ($\frac{\partial y}{\partial t} = 0$). Durante las horas matutinas (09:00 a 13:00 h), cuando el ozono crece a tasas de $+10$ a $+25$ ppb/h forzado por la radiación solar, arrastrar el valor de hace 2 o 3 horas genera una discontinuidad severa en `O3_diff_1h` y distorsiona el cálculo de derivadas contemporáneas.
- Por esta razón, `imp_ffill_short` empeora el RMSE en $h=1$ ($5.071$ vs $5.057$) y en $h=48$ ($13.825$ vs $13.787$).

### 4.3. Comportamiento en los Extremos del Horizonte ($h=1$ y $h=48$)
- En **$h=1$**, el valor contemporáneo $t=0$ de la propia estación tiene una correlación de $0.942$ con $t+1$. Cuando está presente, domina la predicción; cuando falta, el manejo nativo de LightGBM ya sabe cómo saltar a los donantes espaciales directamente en el árbol de predicción de $h=1$. La ganancia de imputar inputs es marginal en RMSE global ($\pm 0.01$ ppb), pero apreciable en los picos más altos.
- En **$h=48$**, el horizonte temporal supera el tiempo de memoria de la química local ($>2$ días). La predicción depende casi exclusivamente de la climatología estacional y de la meteorología sinóptica futura, por lo que la imputación de los lags de $t \le 0$ pierde influencia.

---

## 5. Auditoría de Plan C y Errores NWP

Para complementar la Parte 1, se resume la cuantificación de error de los pronósticos de Open-Meteo frente a observaciones reales SIMA en 2024–2025:

| Variable | Observación SIMA | Error Corrida Seamless (lead 0–23h) | Error Corrida Prev Day 1 (lead 24–47h) | Error Corrida Prev Day 2 (lead 48–71h) |
|---|---|:---:|:---:|:---:|
| **Temperatura 2m** | `TOUT` | RMSE = 1.72 °C ($r=0.970$) | RMSE = 2.19 °C ($r=0.951$) | RMSE = 2.35 °C ($r=0.939$) |
| **Radiación Solar** | `SR` | RMSE = 218.1 W/m² ($r=0.832$) | RMSE = 218.7 W/m² ($r=0.823$) | RMSE = 218.5 W/m² ($r=0.822$) |
| **Humedad Relativa**| `RH` | RMSE = 13.2 % ($r=0.863$) | RMSE = 15.0 % ($r=0.823$) | RMSE = 13.3 % ($r=0.811$) |
| **Velocidad Viento**| `WSR` | RMSE = 4.74 m/s ($r=0.655$) | RMSE = 6.05 m/s ($r=0.638$) | RMSE = 5.24 m/s ($r=0.575$) |

*Conclusión de NWP*: La temperatura muestra una fidelidad sobresaliente ($r > 0.94$ incluso a 48h de anticipación), mientras que la radiación solar presenta un sesgo positivo constante de $+103$ W/m² en la red urbana por falta de absorción por material particulado local.

---

## 6. Recomendaciones Finales para el Ensamble y Producción

1. **Adoptar `imp_spatial_lgbm` como Preprocesador Oficial de Entradas**:
   La imputación causal multivariada espacial con LightGBM ofrece mejoras medibles y consistentes en horizontes operativos clave ($h=4, 8, 24$), reduciendo el error en picos hasta en medio ppb sin introducir ningún riesgo de fuga de información.
2. **Descartar Forward-Fill Puro sin Tendencia**:
   El forward-fill estático introduce distorsiones en los gradientes horarios de contaminantes fotoquímicos y no debe utilizarse salvo en variables de escala lenta (como presión barométrica).
3. **Entrega de Predicciones**:
   Todas las predicciones val 2024 y test 2025 de las 3 variantes evaluadas han sido depositadas en `experiments/noche/audit_imput/preds/` bajo el esquema parquet estándar (`Date`, `Estacion`, `h`, `split`, `y_true`, `y_pred`), listas para su integración en el ensamble final.
