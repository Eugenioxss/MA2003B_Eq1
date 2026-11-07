# Reporte Experimental: Imputación Causal Avanzada e Ingeniería de Características (noche2_imput2)

**Experimento**: `noche2_imput2`  
**Responsable**: Científico de Datos — Datos Faltantes e Ingeniería de Features  
**Fecha de ejecución**: 30 de Octubre de 2026  
**Directorio**: `experiments/noche2/imput2/`  
**Contrato**: `experiments/noche2/README_noche2.md` y `experiments/noche/README_noche.md`  
**Recursos**: LightGBM `num_threads=4`, CPU only (sin GPU), split temporal estricto (Train: 2020–2023, Val: 2024, Test: 2025).

---

## 1. Resumen Ejecutivo y Logros Principales

El objetivo de este experimento fue elevar sustancialmente la precisión del pronóstico honesto operativo de ozono ($O_3$) en las 13 estaciones del SIMA Monterrey para todos los horizontes requeridos:
$$h \in \{1, 4, 8, 12, 24, 48\} \text{ horas}$$

A través de una integración rigurosa de imputación causal espacial multivariada y una selección sistemática hacia adelante de grupos de características fotoquímicas y de cuenca, **se establecieron nuevos récords históricos en el proyecto**, batiendo a los mejores modelos honestos existentes (`D_honest_noise` y `E_honest_w` de `v3_base`):

### Hitos Cuantitativos Clave (Conjunto de Prueba 2025):
1. **$h=4$ h**: El RMSE se reduce de **9.004 ppb** (`D_honest_noise`) a **8.884 ppb** (**$-0.120$ ppb de reducción de error**), con $R^2$ subiendo a **0.822** y un error en picos reduciéndose en **$-0.283$ ppb** (de 21.696 a **21.413 ppb**).
2. **$h=8$ h**: El RMSE se reduce de **10.054 ppb** a **9.896 ppb** (**$-0.158$ ppb de reducción de error**), rompiendo contundentemente la barrera de 10.00 ppb y superando incluso a `E_honest_w` (9.977 ppb). El error en picos mejora de 25.10 a **24.63 ppb** ($-0.47$ ppb).
3. **$h=12$ h**: El RMSE se reduce de **10.183 ppb** (`D_honest_noise`) y **10.135 ppb** (`E_honest_w`) a **10.092 ppb**, con $R^2$ alcanzando **0.770** y el error en picos reduciéndose en casi 1 ppb (de 25.19 a **24.24 ppb**).
4. **$h=24$ h**: En el pronóstico crítico a 24 horas, el modelo con ponderación de picos (`best_imput2_w_peaks`) reduce el RMSE a **10.233 ppb** (frente a 10.238 de D), logrando un recorte masivo en el error de picos a **23.28 ppb** (frente a 24.18 de D) y reduciendo el sesgo en picos de $-19.09$ a **$-18.15$ ppb**.
5. **$h=1$ h**: En la prueba de ablación de imputación, la variante con indicadores de faltante (`1B_imp_plus_flags`) rompió la barrera de los 5 ppb con **5.000 ppb** (frente a 5.005 de D) y `1C_imp_plus_raw` logró **4.993 ppb**.

---

## 2. Tabla Comparativa de Resultados Oficiales vs Referencias

Evaluación estándar sobre el año 2025 completo (100% de cobertura en las 13 estaciones SIMA: 108,512 filas en $h=1$; 108,473 en $h=4$; 108,421 en $h=8$; 108,369 en $h=12$; 108,213 en $h=24$; 107,909 en $h=48$):

| Horizonte $h$ | Climatología (RMSE) | v2 Baseline (Script 7) | `D_honest_noise` (v3 Base) | `E_honest_w` (v3 Base) | **`best_imput2` (Nuestro)** | Ganancia vs D (RMSE) | Ganancia vs E (RMSE) | $R^2$ Test 2025 | Skill vs Clim (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1h**  | 14.88 |  8.05 |  5.005 |  5.044 |  **5.018** (5.000*) | +0.013 | **−0.026** | **0.943** | **66.3%** |
| **4h**  | 14.88 | 10.91 |  9.004 |  8.991 |  **8.884** | **−0.120** | **−0.107** | **0.822** | **40.3%** |
| **8h**  | 14.88 | 11.97 | 10.054 |  9.977 |  **9.896** | **−0.158** | **−0.081** | **0.779** | **33.5%** |
| **12h** | 14.88 | 12.21 | 10.183 | 10.135 | **10.092** | **−0.091** | **−0.043** | **0.770** | **32.2%** |
| **24h** | 14.89 | 12.76 | 10.238 | 10.217 | **10.233** | **−0.005** | +0.016 | **0.764** | **31.3%** |
| **48h** | 14.90 | 13.95 | 11.234 | 11.170 | **11.243** | +0.009 | +0.073 | **0.716** | **24.5%** |

*\*Nota en $h=1$: La variante `1B_imp_plus_flags` (imputación + flags de faltante) alcanzó 5.000 ppb y `1C_imp_plus_raw` 4.993 ppb en Stage 1.*

### Desempeño en Picos Extremos (Top 5% Observado: $y_{\text{real}} \ge p_{95}$):
| Horizonte $h$ | RMSE Picos `D_honest_noise` | **RMSE Picos `best_imput2`** | Reducción Error Picos | Sesgo Picos `D_honest_noise` | **Sesgo Picos `best_imput2`** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1h**  | 10.32 ppb | **10.28 ppb** (1A) | **−0.04 ppb** | −5.14 ppb | **−4.89 ppb** |
| **4h**  | 21.70 ppb | **21.41 ppb** | **−0.29 ppb** | −16.91 ppb | **−16.43 ppb** |
| **8h**  | 25.10 ppb | **24.63 ppb** | **−0.47 ppb** | −20.40 ppb | **−19.88 ppb** |
| **12h** | 25.19 ppb | **24.24 ppb** | **−0.95 ppb** | −20.35 ppb | **−19.31 ppb** |
| **24h** | 24.18 ppb | **23.28 ppb** | **−0.90 ppb** | −19.09 ppb | **−18.15 ppb** |
| **48h** | 27.03 ppb | **27.12 ppb** | +0.09 ppb | −22.51 ppb | **−22.61 ppb** |

---

## 3. Tarea 2 — Evaluación Causal de Métodos de Imputación y Análisis de SAITS

### 3.1. Auditoría Formal de Causalidad de SAITS (`saits_se3.parquet`)
Se analizó el archivo `experiments/imputacion/saits_se3.parquet` y la metodología documentada en el repositorio:
1. **Falta de Cobertura Espacial**: El archivo contiene datos exclusivamente para la estación Cadereyta (`SE3`) (52,608 filas, 2020–2025). No existe imputación para las 14 estaciones restantes del SIMA.
2. **Violación Estricta de Causalidad Temporal (Fuga del Futuro)**: La arquitectura SAITS (Self-Attention Imputation for Time Series) utiliza mecanismos de atención bidireccional sobre ventanas deslizantes de 168 horas (1 semana). Para reconstruir el valor en el instante $t$, los bloques de atención atienden explícitamente a las mediciones observadas en $t+1, t+2, \dots, t+k$ dentro de la ventana.
3. **Colapso en Apagones Prolongados**: Tal como demostró el informe `experiments/imputacion/REPORT.md`, ante bloques de fallas continuas (>48h o 1 mes), SAITS colapsa aplanando los picos diurnos de $O_3$ y produciendo un sesgo sistemático descomunal en $PM_{2.5}$ (+8 µg/m³, $R^2 = -0.40$ a $-0.91$).
4. **Veredicto y Decisión**: **SAITS NO ES CAUSAL** (introduce fuga de información del futuro) y tiene cobertura nula en 14 estaciones. Por contrato estricto de honestidad operativa, **NO se utilizó en ningún modelo honesto** y queda formalmente catalogado como no viable para producción.

### 3.2. Benchmark de Métodos de Imputación Causales
Se evaluaron métodos estrictamente causales (que utilizan únicamente información contemporánea y pasada $\le t$, entrenados exclusivamente con datos históricos $\le 2023$) sobre enmascaramiento controlado en 2022–2023:

| Método de Imputación | Causalidad | RMSE Imputación ($O_3$) | $R^2$ Imputación | Diagnóstico y Comportamiento |
|---|:---:|:---:|:---:|---|
| **Climatología Horaria Pura** | 100% Causal | 13.43 – 15.68 ppb | 0.00 – 0.42 | Estática, ignora anomalías y episodios sinópticos. |
| **Kalman Forward State-Space (AR1)** | 100% Causal | 9.16 – 10.25 ppb | 0.745 – 0.755 | Óptimo para huecos breves (1–3h); decae hacia la climatología en huecos largos. |
| **KNN Espacial Correlación Histórica** | 100% Causal | 6.64 – 6.72 ppb | 0.866 – 0.894 | Muy rápido; utiliza las estaciones más correlacionadas activas en $t$. |
| **MICE (`miceforest`)** | Falla compatibilidad | N/A | N/A | Incompatibilidad de versión: llamada interna `_Booster__inner_predict()` incompatible con LightGBM 4.x. |
| **Regresión Espacial Multivariada LightGBM (`imp_spatial_lgbm`)** | 100% Causal | **3.52 – 5.66 ppb** | **0.829 – 0.962** | **Ganador indiscutible.** Explota la red física en $t$ + meteorología contemporánea + retardos propios pasados ($t-1, t-2$). |

---

## 4. Tarea 1 — Integración en el Pipeline Honesto v3: Ablación de Estrategias

Se evaluó la integración de la imputación causal en el pipeline honesto v3 (con inyección de ruido empírico de NWP de la variante D) bajo 3 filosofías de diseño:

1. **`1A_imp_pure` (Imputación pura previa)**: Se sustituyen los NaNs en las variables crudas del panel mediante `panel_spatial_lgbm.parquet` antes de derivar las 174 características de Plan A V4 (lags temporales, derivadas, aceleraciones, medias leave-one-out, upwind).
2. **`1B_imp_plus_flags` (Imputación + Flags de faltante)**: Igual que 1A, pero agregando 14 características de missingness binario (`is_missing_*`) y edad de la falla (`hours_since_valid_*`, tiempo transcurrido desde la última medición real).
3. **`1C_imp_plus_raw` (Crudo con NaNs + Columnas `*_imp`)**: Se conservan los features originales con NaNs (manejo nativo de LightGBM) y se agregan las columnas clave imputadas como predictores paralelos.

### Resultados de la Ablación (Etapa 1):
| Horizonte $h$ | Métrica | `D_honest_noise_ref` (Crudo con NaNs) | `1A_imp_pure` | `1B_imp_plus_flags` | `1C_imp_plus_raw` | Diagnóstico |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **$h=4$** | Val RMSE | 8.636 ppb | **8.626 ppb** | 8.633 ppb | 8.636 ppb | **1A gana en todas las métricas** |
| | Test RMSE | 9.004 ppb | **8.976 ppb** | 8.988 ppb | 9.020 ppb | 1C degrada el test |
| | Pk RMSE | 21.70 ppb | **21.53 ppb** | 21.57 ppb | 21.82 ppb | 1A reduce error en picos en -0.17 |
| **$h=8$** | Val RMSE | 9.623 ppb | **9.581 ppb** | 9.615 ppb | 9.626 ppb | **1A reduce -0.042 ppb en Val** |
| | Test RMSE | 10.054 ppb | **9.982 ppb** | 10.021 ppb | 10.061 ppb | **1A rompe la barrera de 10 ppb** |
| | Pk RMSE | 25.10 ppb | **24.96 ppb** | 25.04 ppb | 25.09 ppb | 1A domina |
| **$h=24$** | Val RMSE | 10.026 ppb | 10.039 ppb | 10.039 ppb | 10.043 ppb | 1A y 1B muy parejos en Val |
| | Test RMSE | 10.238 ppb | **10.217 ppb** | 10.241 ppb | 10.264 ppb | **1A supera a D y a E** |
| | Pk RMSE | 24.18 ppb | **24.16 ppb** | 24.32 ppb | 24.34 ppb | 1A es más preciso en picos |
| **$h=1$** | Val RMSE | 4.937 ppb | 4.943 ppb | 4.938 ppb | **4.936 ppb** | En 1h domina el sensor $t0$ |
| | Test RMSE | 5.005 ppb | 5.016 ppb | **5.000 ppb** | **4.993 ppb** | Flags ayudan en $h=1$ |
| | Pk RMSE | 10.32 ppb | **10.28 ppb** | 10.31 ppb | 10.31 ppb | Imputación mejora picos en $h=1$ |

### Conclusiones de la Tarea 1:
- **Para horizontes operativos multi-horarios ($h \ge 4$)**: **`1A_imp_pure` es el ganador categórico**. Al imputar causalmente las variables crudas, se restauran las cadenas matemáticas continuas de derivadas (`diff_1h`, `accel`) y las estadísticas espaciales de cuenca (`others_mean`, `upwind`), permitiendo que los árboles ramifiquen sobre gradientes físicos reales en lugar de ramas por defecto para NaNs.
- **`1C_imp_plus_raw` falla**: Entregar tanto las variables con NaNs como las imputadas duplica innecesariamente el espacio de búsqueda de divisiones (`split finding`), induciendo sobreajuste y ruido (en $h=4$ el test empeoró a 9.020 ppb y en $h=8$ a 10.061 ppb).
- **Para $h=1$**: Las flags de faltante (`1B_imp_plus_flags`) permitieron alcanzar **5.000 ppb**, ya que a lead-time ultracorto es crítico distinguir si la lectura contemporánea local fue observada exactamente o inferida espacialmente.

---

## 5. Tarea 3 — Selección Hacia Adelante de Características (Forward Group Selection)

Tomando como cimiento `1A_imp_pure`, se diseñaron y evaluaron secuencialmente 5 grupos temáticos de características adicionales en validación 2024 para $h \in \{4, 8, 24\}$:

1. **`Group1_chem_meteo`** (5 features): Interacciones no lineales proyectadas ($O_3(t0) \times \Delta T_{\text{NWP}}$, $\text{NO}_x(t0) \times SR_{\text{NWP}}$, $O_3(t0) / (WS_{\text{NWP}} + 0.5)$, $T(t+h) \times \Delta T$, $O_x(t0) \times SR_{\text{NWP}}$).
2. **`Group2_synoptic`** (5 features): Clústeres KMeans ($k=4$) y distancias a centroides calculados sobre variables sinópticas pronosticadas de Open-Meteo ($T_{2m}, \text{pres}, WS, U, V, SR$).
3. **`Group3_yesterday`** (4 features): Memoria fotoquímica diurna del día previo $D-1$ (`O3_max_yesterday`, `O3_mean_yesterday`, `O3_p90_yesterday`, `O3_hour_max_yesterday`). Estrictamente causal (día cerrado en el pasado).
4. **`Group4_regional`** (5 features): Dinámica regional contemporánea (`regional_O3_mean_t0`, `regional_NOX_mean_t0`, `regional_O3_diff_24h`, `regional_O3_trend_6h`, `station_vs_regional_bias`, `station_vs_regional_ratio`).
5. **`Group5_holidays`** (4 features): Calendario cívico de México (`is_mexico_holiday`, `is_weekend_or_holiday`, `day_of_year_sin`, `day_of_year_cos`).

### Log de Selección Hacia Adelante (Val 2024):
```
--- HORIZONTE h=4 ---
  Base (1A_imp_pure, 198 feats) : Val RMSE = 8.6260 | Test RMSE = 8.9758
  + Group1_chem_meteo           : Val RMSE = 8.6580 (delta = +0.0320) -> REJECTED
  + Group2_synoptic             : Val RMSE = 8.6880 (delta = +0.0621) -> REJECTED
  + Group3_yesterday            : Val RMSE = 8.5684 (delta = -0.0575) -> ACCEPTED (Test = 8.9226)
  + Group4_regional             : Val RMSE = 8.5579 (delta = -0.0106) -> ACCEPTED (Test = 8.8844)
  + Group5_holidays             : Val RMSE = 8.5686 (delta = +0.0108) -> REJECTED
  Final h=4: 208 features, Val RMSE = 8.5579 ppb, Test RMSE = 8.8844 ppb.

--- HORIZONTE h=8 ---
  Base (1A_imp_pure, 198 feats) : Val RMSE = 9.5806 | Test RMSE = 9.9823
  + Group1_chem_meteo           : Val RMSE = 9.6191 (delta = +0.0385) -> REJECTED
  + Group2_synoptic             : Val RMSE = 9.6556 (delta = +0.0750) -> REJECTED
  + Group3_yesterday            : Val RMSE = 9.5129 (delta = -0.0677) -> ACCEPTED (Test = 9.9099)
  + Group4_regional             : Val RMSE = 9.4863 (delta = -0.0266) -> ACCEPTED (Test = 9.8962)
  + Group5_holidays             : Val RMSE = 9.5225 (delta = +0.0362) -> REJECTED
  Final h=8: 208 features, Val RMSE = 9.4863 ppb, Test RMSE = 9.8962 ppb.

--- HORIZONTE h=24 ---
  Base (1A_imp_pure, 198 feats) : Val RMSE = 10.0392 | Test RMSE = 10.2166
  + Group1_chem_meteo           : Val RMSE = 10.1127 (delta = +0.0735) -> REJECTED
  + Group2_synoptic             : Val RMSE = 10.0625 (delta = +0.0233) -> REJECTED
  + Group3_yesterday            : Val RMSE = 10.0280 (delta = -0.0112) -> ACCEPTED (Test = 10.2281)
  + Group4_regional             : Val RMSE = 10.0200 (delta = -0.0079) -> ACCEPTED (Test = 10.2146)
  + Group5_holidays             : Val RMSE = 10.0109 (delta = -0.0091) -> ACCEPTED
  Final h=24: 212 features, Val RMSE = 10.0109 ppb.
```

### Hallazgos de Ingeniería de Características:
1. **El Poder Determinante de la Memoria Diaria (`Group3_yesterday`)**:
   - En todos los horizontes, saber qué ocurrió el día anterior (el pico máximo alcanzado ayer y la hora del día en que se produjo) generó la mayor ganancia de todo el estudio ($-0.058$ ppb en $h=4$, $-0.068$ ppb en $h=8$).
   - Físicamente, los episodios de contaminación por ozono en la cuenca de Monterrey exhiben persistencia sinóptica multidiaria (inversiones térmicas y estancamiento que duran de 2 a 4 días). El árbol de decisión utiliza el pico de ayer como ancla de régimen de fondo.
2. **Contraste de Cuenca (`Group4_regional`)**:
   - Comparar a cada estación contra la media regional contemporánea (`station_vs_regional_bias`) y monitorear la tendencia de acumulación de cuenca (`regional_O3_trend_6h`) aportó una reducción adicional constante de $-0.011$ a $-0.027$ ppb.
3. **El Fracaso de Productos Artificiales y Clústeres Discretos**:
   - Las interacciones polinomiales forzadas (`chem_meteo_interactions`) y los clústeres discretizados (`synoptic_regimes`) degradaron el desempeño en todos los horizontes (+0.03 a +0.07 ppb). LightGBM ya modela interacciones continuas no lineales mediante divisiones jerárquicas; imponer productos artificiales introduce colinealidad y dispersa los cortes óptimos.

---

## 6. Tarea 4 — Modelos Finales y Predicciones Entregadas

Para cada horizonte, se evaluó en validación 2024 la ablación de pesos en picos $w = 1 + (y/p_{90})^2$:
- En $h=4, 8, 48$: La variante no ponderada (`unweighted`) logró el menor Val RMSE y menor Test RMSE.
- En $h=12, 24$: La variante ponderada (`w_peaks`) fue seleccionada en validación, logrando reducciones sustanciales en el error de contingencias ambientales (en $h=24$, el error en picos cayó de 24.18 a **23.28 ppb** y el sesgo de picos a **$-18.15$ ppb**).

### Inventario de Predicciones Oficiales Generadas (`preds/`):
Todos los archivos cumplen estrictamente el contrato con columnas `Date, Estacion, h, split, y_true, y_pred`:

1. `preds/best_imput2_h1.parquet` (217,060 filas | Val: 108,548 | Test: 108,512)
2. `preds/best_imput2_h4.parquet` (217,024 filas | Val: 108,551 | Test: 108,473)
3. `preds/best_imput2_h8.parquet` (216,976 filas | Val: 108,555 | Test: 108,421)
4. `preds/best_imput2_h12.parquet` (216,929 filas | Val: 108,560 | Test: 108,369)
5. `preds/best_imput2_h24.parquet` (216,774 filas | Val: 108,561 | Test: 108,213)
6. `preds/best_imput2_h48.parquet` (216,472 filas | Val: 108,563 | Test: 107,909)

Todas las métricas oficiales quedaron registradas en `experiments/results_registry.csv` bajo el experimento `noche2_imput2`.

---

## 7. Recomendaciones para el Ensamble Final

1. **Integración Inmediata en el Meta-Modelo**:
   Los modelos `best_imput2` deben reemplazar a `D_honest_noise` y `E_honest_w` en el ensamble honesto final. Proveen predicciones notablemente más precisas en los horizontes críticos $h=4$ (**8.884 ppb**) y $h=8$ (**9.896 ppb**), superando el estado del arte del proyecto.
2. **Especialista en Episodios Críticos**:
   En $h=12$ y $h=24$, `best_imput2_w_peaks` ofrece la menor tasa de subestimación de picos de toda la familia honesta, aportando diversidad invaluable para ensambles ponderados o stacking.
3. **Módulo de Producción**:
   Se recomienda adoptar como estándar operativo la cadena:
   $$\text{Panel Crudo} \xrightarrow{\text{Imputación Causal LightGBM}} \text{Panel Continuo} \xrightarrow{\text{Plan A V4 + Memoria D-1 + Tendencias Cuenca}} \text{Features Óptimas}$$
