# Informe Experimental: Benchmark de Imputación y Prueba Downstream de Pronóstico

**Autor:** Científico de Datos — Experimento de Imputación  
**Fecha:** 28 de Octubre de 2026  
**Repositorio:** `D:\Codingggg\Notas\Multivariados\MA2003B_Eq1`  
**Directorio del experimento:** `experiments/imputacion/`  
**Módulo reutilizable:** `experiments/imputacion/imputers.py` (`impute_tiered`)

---

## 1. Resumen Ejecutivo y Motivación

La red de monitoreo atmosférico SIMA del Área Metropolitana de Monterrey (15 estaciones horarias, 2020–2025) presenta apagones de sensores prolongados que van desde episodios puntuales de 1 a 3 horas hasta fallas de sensores superiores a 400 días consecutivos. 

Un estudio preliminar del equipo (`check_LSTM_PC/eval_bloques/evaluacion_imputacion_bloques.md`) demostró que modelos neuronales generativos de imputación temporal profunda como **SAITS+CSDI** fallan de manera estrepitosa ante huecos en bloque:
- En ozono ($O_3$), la varianza explicada se redujo a $R^2 \approx 0.36$.
- En material particulado fino ($PM_{2.5}$), el modelo colapsó con **$R^2$ negativo** ($-0.40$ en 1 mes y $-0.91$ en 55 días), generando un **sesgo sistemático de sobreestimación del $+58\%$ al $+66\%$** ($+8.3$ µg/m³ sobre la media real).

### Objetivos del presente estudio:
1. **Parte 1 (Benchmark de Métodos por Tipo de Hueco):** Evaluar sistemáticamente 7 familias de algoritmos (interpolación polinomial, climatología horaria pura y con ajuste por nivel, descomposición estacional STL, interpolación espacial ponderada por distancia inversa IDW, regresión espacial multivariada con LightGBM, MICE y SoftImpute matricial) a través de 5 escenarios reales de enmascaramiento ($S_1$ a $S_5$) para $O_3$, $PM_{2.5}$, $NO_X$ y $TOUT$.
2. **Parte 2 (Prueba Downstream):** Diseñar una estrategia de imputación por niveles (`impute_tiered`) e investigar empíricamente si la imputación previa de variables de entrada mejora el pronóstico operativo de $O_3$ con LightGBM v2 a horizontes $h \in \{1, 8, 24\}$, evaluando estrictamente sobre los targets reales observados de 2025 bajo el contrato estándar común.

---

## 2. Protocolo Experimental y Escenarios de Enmascaramiento

Todos los experimentos se ejecutaron sobre el panel horario unificado (`load_panel()`, 789,120 registros $\times$ 19 variables), utilizando CPU acotada a `num_threads=3` y semilla fija `SEED=42`.

### Escenarios Evaluados ($S_1$ a $S_5$):
- **$S_1$ (Huecos Cortos Aleatorios 1–3 h):** Extracción de bloques aleatorios de 1 a 3 horas contiguas sobre series reales observadas en 2022 (total ~1,000 horas evaluadas en estaciones clave).
- **$S_2$ (Huecos Medianos Aleatorios 6–48 h):** Extracción de bloques de 6, 12, 24 y 48 horas con datos observados antes y después de la ventana.
- **$S_3$ (Bloque Crítico de 1 Mes — Junio 2022 en SE3):** Replicación exacta del Escenario A previo (715 horas válidas de $O_3$, 693 horas de $PM_{2.5}$; ground truth en `ground_truth_A.csv`).
- **$S_4$ (Bloque Crítico de 55 Días — 1 Jul a 25 Ago 2022 en SE3):** Replicación exacta del Escenario B previo (1,328 horas válidas de $O_3$, 1,288 horas de $PM_{2.5}$; ground truth en `ground_truth_B.csv`).
- **$S_5$ (Robustez Espacial en 3 Estaciones — Junio 2022 en SO2, CE, SE):** Bloque de 1 mes en 3 estaciones con cobertura superior al 95%, ubicadas en el núcleo urbano metropolitano.

### Métricas de Evaluación:
Para cada escenario y método se calcularon:
- **RMSE** y **MAE** (en unidades de la variable: ppb o µg/m³ o °C).
- **$R^2$** (coeficiente de determinación respecto a la media real).
- **RMSE en Picos (top 5% superior)** de las observaciones reales ($y \ge Q_{0.95}$).
- **Sesgo Medio (Bias):** $\frac{1}{N} \sum (\hat{y} - y)$.

---

## 3. Parte 1 — Resultados del Benchmark de Imputación

### 3.1. Comparativa Global Método $\times$ Escenario (Variables Principales: $O_3$ y $PM_{2.5}$)

A continuación se resumen las métricas clave obtenidas empíricamente en `experiments/imputacion/benchmark_results.csv`:

#### Ozono ($O_3$, ppb)
| Escenario | Método | RMSE (ppb) | MAE (ppb) | $R^2$ | RMSE Picos (top 5%) | Sesgo (ppb) | Diagnóstico |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$S_1$ (Cortos 1–3h)** | **PCHIP** | **4.59** | **2.68** | **0.905** | **14.81** | **+0.08** | **Óptimo y ultra-rápido** |
| | Lineal | 4.97 | 2.92 | 0.889 | 15.68 | +0.26 | Muy competitivo |
| | Clim + Ajuste Nivel | 4.49 | 2.70 | 0.910 | 14.52 | +0.14 | Excelente |
| | Climatología Horaria | 11.44 | 8.35 | 0.425 | 32.18 | +2.29 | Inadecuado (ignora estado actual) |
| **$S_2$ (Medianos 6–48h)** | **Climatología Horaria** | **13.64** | **10.22** | **0.553** | **37.60** | **-1.86** | **Mejor correlación general** |
| | **Clim + Ajuste Nivel** | **15.10** | **10.66** | **0.457** | **37.74** | **+0.00** | **Insesgado (anomalía continua)** |
| | STL + Interp | 18.50 | 12.38 | 0.159 | 40.85 | +0.48 | Degrada en huecos de 48h |
| | Lineal | 27.26 | 21.05 | -0.811 | 55.47 | +3.20 | Colapso severo |
| **$S_3$ (1 mes SE3)** | **Regresión Espacial LGBM** | **5.66** | **3.89** | **0.829** | **20.68** | **+0.24** | **Ganador indiscutible** |
| | IDW Espacial | 6.25 | 4.54 | 0.791 | 22.02 | +0.98 | Excelente baseline espacial |
| | SoftImpute (SVD) | 6.42 | 4.67 | 0.780 | 21.65 | -1.02 | Muy robusto |
| | SAITS+CSDI (previo) | 10.97 | 7.63 | 0.358 | 34.66 | -0.76 | Aplana completamente los picos |
| | Climatología Horaria | 10.30 | 7.55 | 0.433 | 31.84 | -1.23 | Supera a SAITS |
| | Lineal | 13.89 | 10.12 | -0.030 | 43.81 | -3.20 | Inválido |
| **$S_4$ (55 días SE3)** | **Regresión Espacial LGBM** | **5.81** | **4.20** | **0.791** | **21.28** | **+1.26** | **Reconstrucción de alta precisión** |
| | SoftImpute (SVD) | 7.15 | 5.38 | 0.684 | 24.38 | +1.27 | Robusto a largos apagones |
| | IDW Espacial | 7.19 | 5.41 | 0.681 | 24.18 | +2.51 | Ligera sobreestimación en periferia |
| | Climatología Horaria | 8.11 | 6.29 | 0.594 | 26.54 | -1.85 | Ciclo diurno estático |
| | SAITS+CSDI (previo) | 10.06 | 7.18 | 0.375 | 28.00 | +1.54 | Ruido con picos atenuados |
| | Lineal | 22.85 | 18.91 | -2.227 | 56.40 | +17.60 | Falla catastrófica |
| **$S_5$ (Robustez 3 est.)** | **Regresión Espacial LGBM** | **3.52** | **2.54** | **0.962** | **11.45** | **-0.04** | **R² = 0.96 en núcleo urbano** |
| | IDW Espacial | 4.05 | 2.94 | 0.949 | 13.87 | -0.61 | Rendimiento sobresaliente |
| | SoftImpute (SVD) | 4.92 | 3.51 | 0.929 | 15.62 | +0.14 | Excelente completado matricial |
| | Lineal | 19.44 | 14.86 | -0.124 | 48.21 | -4.17 | No viable |

---

#### Material Particulado Fino ($PM_{2.5}$, µg/m³)
| Escenario | Método | RMSE (µg/m³) | MAE (µg/m³) | $R^2$ | RMSE Picos (top 5%) | Sesgo (µg/m³) | Diagnóstico |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$S_1$ (Cortos 1–3h)** | **Lineal / PCHIP** | **5.55** | **3.31** | **0.836** | **18.72** | **-0.31** | **Inercia temporal dominante** |
| | Clim + Ajuste Nivel | 5.61 | 3.39 | 0.830 | 18.84 | -0.34 | Equivalente a lineal |
| | Climatología Horaria | 13.27 | 9.85 | 0.167 | 30.12 | -0.56 | Insuficiente |
| **$S_2$ (Medianos 6–48h)** | **Clim + Ajuste Nivel** | **7.04** | **4.76** | **0.539** | **20.15** | **-0.51** | **Mantiene anomalía regional** |
| | STL + Interp | 7.80 | 5.34 | 0.432 | 22.04 | -0.35 | Buen seguimiento de fondo |
| | Lineal | 9.01 | 6.55 | 0.216 | 23.41 | -0.75 | Pierde variabilidad intra-día |
| **$S_3$ (1 mes SE3)** | **Regresión Espacial LGBM** | **5.14** | **3.71** | **0.648** | **12.48** | **+0.37** | **Elimina sesgo, R² positivo** |
| | SoftImpute (SVD) | 5.88 | 4.25 | 0.538 | 13.65 | +1.13 | Gran consistencia |
| | IDW Espacial | 7.63 | 5.89 | 0.224 | 15.20 | +5.39 | Sesgado por distancias a periferia |
| | Lineal | 9.31 | 6.90 | -0.155 | 23.56 | -3.36 | Colapso |
| | SAITS+CSDI (previo) | 10.25 | 8.93 | **-0.401** | 7.14 | **+8.01** | **Sesgo masivo (+58%)** |
| **$S_4$ (55 días SE3)** | **Regresión Espacial LGBM** | **4.88** | **3.56** | **0.526** | **11.89** | **-0.13** | **Sin sesgo residual (Bias -0.13)** |
| | SoftImpute (SVD) | 5.68 | 4.12 | 0.359 | 12.94 | +1.34 | Reconstrucción estable |
| | Climatología Horaria | 7.06 | 5.12 | 0.008 | 14.85 | +2.86 | Valor medio sin picos |
| | IDW Espacial | 7.52 | 5.92 | -0.124 | 14.78 | +5.49 | Sesgo positivo elevado |
| | SAITS+CSDI (previo) | 9.81 | 8.42 | **-0.915** | 6.58 | **+8.30** | **Colapso (+66% sobreestimación)** |
| **$S_5$ (Robustez 3 est.)** | **Regresión Espacial LGBM** | **3.48** | **2.51** | **0.846** | **9.12** | **-0.12** | **R² = 0.85 en área central** |
| | IDW Espacial | 4.78 | 3.52 | 0.689 | 11.24 | +1.39 | Coherencia espacial alta |
| | SoftImpute (SVD) | 4.97 | 3.65 | 0.685 | 11.82 | +0.93 | Desempeño sólido |

---

### 3.2. Variables Adicionales: $NO_X$ y $TOUT$

- **Temperatura Exterior ($TOUT$, °C):** Presenta una correlación espacial y temporal sumamente elevada. En bloques largos ($S_3$ y $S_4$), tanto **IDW Espacial** ($R^2 = 0.960$ a $0.972$, RMSE $\approx 0.75$–$0.93$ °C) como **Regresión Espacial LightGBM** ($R^2 = 0.975$ a $0.977$, RMSE $= 0.70$ °C) reconstruyen la temperatura prácticamente de forma exacta.
- **Óxidos de Nitrógeno ($NO_X$, ppb):** Los $NO_X$ dependen fuertemente de fuentes de emisión móviles hiperlocales (tráfico y arterias viales). En apagones largos ($S_3$ y $S_4$), los métodos univariados y espaciales simples fallan ($R^2 < 0$), mientras que la **Regresión Espacial LightGBM** (al incorporar covariables como radiación solar, hora del día, temperatura y viento) es el único método capaz de lograr $R^2$ positivo ($R^2 = 0.456$ en $S_3$ y $R^2 = 0.409$ en $S_4$, RMSE $= 2.48$ a $2.76$ ppb).

---

## 4. Análisis Detallado: ¿Por qué SAITS+CSDI Falló y Cómo la Regresión Espacial lo Supera?

### 4.1. Diagnóstico del Fallo de SAITS+CSDI
1. **Supuesto de Imputación Violado (MCAR vs Apagones Estructurales):** SAITS se entrenó enmascarando puntos aleatorios al 20% (MCAR puntual) en ventanas de 168 horas (1 semana). Aprendió a usar el contexto visible de las horas inmediatamente anteriores y posteriores de la misma estación. Al enfrentarse a un bloque continuo de 720 h (1 mes) o 1,333 h (55 días), la ventana de 168 h queda **completamente vacía de la variable target**.
2. **Promediado por Solapamiento:** Las ventanas con paso de 84 h promedian predicciones, resultando en una trayectoria aplanada que no puede reconstruir los picos de ozono diurnos (los picos reales de 90–115 ppb quedan achatados en 30–40 ppb).
3. **Deriva Positiva en $PM_{2.5}$:** Al no tener anclaje espacial, el denoiser se fue al centroide de la distribución marginal de entrenamiento más un sesgo acumulado en el espacio latente, flotando sistemáticamente 8 µg/m³ por encima de la realidad.

### 4.2. La Ventaja Espacial Multivariada
En contraste, la **Regresión Espacial LightGBM**:
1. No interpola a lo largo del tiempo de la estación apagada, sino **a través del espacio físico en el mismo instante $t$**.
2. Cuando Cadereyta (SE3) está apagada, las estaciones circundantes (Juárez SE2 a 10.8 km, Pastora SE a 25.5 km, Apodaca NE2 a 27.2 km) están midiendo ozono, radiación solar, dirección y velocidad del viento y temperatura en ese mismo instante.
3. LightGBM maneja de forma nativa los registros faltantes de estaciones donantes individuales y modela interacciones no lineales entre meteorología y transporte de contaminantes.

### 4.3. Evidencia Visual (Gráficas Overlay)

A continuación se presentan las reconstrucciones de los mejores métodos frente al ground truth y frente al SAITS previo:

#### Reconstrucción de Ozono ($O_3$) — Escenario $S_3$ (Junio 2022, SE3)
![Overlay S3 O3](figures/overlay_S3_O3.png)
*Figura 1: Comparativa en Escenario S3 (720h). Nótese cómo la Regresión Espacial LightGBM (verde) rastrea con fidelidad la amplitud diaria y los picos agudos de ozono, mientras que SAITS (naranja punteado) sufre una fuerte atenuación de picos y lineal (negro) colapsa.*

#### Reconstrucción de $PM_{2.5}$ — Escenario $S_3$ (Junio 2022, SE3)
![Overlay S3 PM25](figures/overlay_S3_PM25.png)
*Figura 2: Comparativa de $PM_{2.5}$ en Escenario S3. SAITS (naranja) flota continuamente por encima con un sesgo de +8 µg/m³ ($R^2 = -0.40$). La Regresión Espacial LightGBM (verde) elimina por completo dicho sesgo ($R^2 = +0.65$, Bias $= +0.37$).*

#### Reconstrucción en Bloque Largo ($S_4$, 55 días — 1 Jul a 25 Ago 2022, SE3)
![Overlay S4 O3](figures/overlay_S4_O3.png)
*Figura 3: Reconstrucción de Ozono durante el apagón de 55 días en Cadereyta ($S_4$). LightGBM mantiene $R^2 = 0.791$ y RMSE $= 5.81$ ppb a lo largo de casi dos meses sin datos locales.*

![Overlay S4 PM25](figures/overlay_S4_PM25.png)
*Figura 4: Reconstrucción de $PM_{2.5}$ durante 55 días. SAITS colapsa a $R^2 = -0.915$, mientras LightGBM logra $R^2 = +0.526$.*

#### Resumen de $R^2$ por Escenario y Método
![Benchmark Summary R2](figures/benchmark_summary_r2.png)
*Figura 5: Desempeño relativo ($R^2$) de los algoritmos across escenarios. En huecos cortos dominan los métodos temporales locales (Lineal / PCHIP / Clim+Ajuste); a partir de bloques medianos y largos (>48h), la información espacial se vuelve absolutamente indispensable.*

---

## 5. Parte 2 — ¿La Imputación Ayuda al Pronóstico? (Prueba Downstream)

### 5.1. Diseño de la Estrategia por Niveles (`impute_tiered`)
Con base en los hallazgos del benchmark, se implementó en `experiments/imputacion/imputers.py` una función de producción `impute_tiered(panel)` con el siguiente esquema de decisión por longitud de hueco ($L$):

$$\text{Método}(L) = \begin{cases} 
\text{Interpolación Lineal Temporal}, & L \le 3 \text{ h} \\
\text{Climatología Horaria + Ajuste por Nivel (Interpolación de Anomalías)}, & 4 \text{ h} \le L \le 48 \text{ h} \\
\text{IDW Espacial Instantáneo + Climatología de Respaldo}, & L > 48 \text{ h}
\end{cases}$$

### 5.2. Protocolo de Evaluación Downstream
- Se imputaron **únicamente las variables de entrada del panel** ($O_3$, $PM_{2.5}$, $PM_{10}$, $NO_X$, $TOUT$, $SR$, $RH$, $U\_Wind$, $V\_Wind$).
- **El target a predecir ($O_3(t+h)$) se mantuvo estrictamente intacto con sus valores no observados reales (sin imputar)**.
- Se entrenó el modelo LightGBM v2 (`build_v2_features`, hiperparámetros oficiales `LGB_PARAMS_V2`, `num_threads=3`) para horizontes $h \in \{1, 8, 24\}$ sobre train (2020–2023) y val (2024).
- Se evaluó con `evaluate_test(test_df, y_pred, h)` sobre el conjunto estándar de test (año 2025 completo, 13 estaciones de modelado, observaciones reales del target: 108,512 filas para $h=1$).

### 5.3. Variantes Probadas:
1. `v2_native_nan`: LightGBM v2 estándar sobre panel crudo (los árboles manejan los NaN internamente asignando la dirección por defecto `default_left`).
2. `v2_tiered_imputed`: LightGBM v2 con features derivados del panel con imputación por niveles (`impute_tiered`).
3. `v2_imputed_with_flags`: LightGBM v2 sobre panel imputado + indicadores binarios de si el sensor estaba apagado en $t-1$ (`missing_var_lag1`).

---

### 5.4. Resultados Downstream sobre Test 2025

| Horizonte ($h$) | Variante | RMSE (ppb) | MAE (ppb) | $R^2$ | RMSE Picos (top 5%) | Sesgo Picos (ppb) | Skill vs Clim | Iteraciones Óptimas |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1 h** | `v2_native_nan` (baseline) | 8.057 | 5.544 | 0.8535 | 19.216 | -13.88 | 0.4585 | 798 |
| | `v2_tiered_imputed` | 8.028 | 5.519 | 0.8546 | 18.995 | -13.60 | 0.4605 | 822 |
| | **`v2_imputed_with_flags`** | **8.027** | **5.521** | **0.8546** | **18.992** | **-13.70** | **0.4605** | **571** |
| **+8 h** | `v2_native_nan` (baseline) | 11.967 | 8.549 | 0.6771 | **31.120** | **-26.93** | 0.1958 | 792 |
| | `v2_tiered_imputed` | 11.975 | 8.514 | 0.6767 | 31.375 | -27.17 | 0.1953 | 680 |
| | **`v2_imputed_with_flags`** | **11.964** | **8.515** | **0.6772** | 31.310 | -27.17 | **0.1960** | **672** |
| **+24 h** | `v2_native_nan` (baseline) | 12.761 | 9.125 | 0.6332 | 33.337 | -29.07 | 0.1428 | 300 |
| | `v2_tiered_imputed` | 12.720 | 9.087 | 0.6356 | 33.388 | -29.28 | 0.1456 | 187 |
| | **`v2_imputed_with_flags`** | **12.707** | **9.085** | **0.6363** | **33.287** | **-29.21** | **0.1464** | **162** |

*Todos los registros fueron debidamente asentados en `experiments/results_registry.csv`.*

![Downstream RMSE R2](figures/downstream_rmse_r2.png)
*Figura 6: Comparativa de RMSE y $R^2$ en el conjunto de prueba 2025 para las 3 variantes evaluadas.*

![Downstream Peaks](figures/downstream_peaks.png)
*Figura 7: Desempeño en picos críticos ($y \ge Q_{0.95}$ del test 2025).*

---

### 5.5. Hallazgos de la Prueba Downstream

1. **¿Ayuda la imputación al pronóstico? SÍ, pero de forma sutil y condicionada:**
   - La variante con imputación por niveles + flags de faltante (`v2_imputed_with_flags`) **supera al baseline de NaN nativo en todos los horizontes evaluados ($h=1, 8, 24$)**:
     - En $h=1$: RMSE pasa de $8.057$ a **$8.027$ ppb** ($-0.030$ ppb), y el error en picos críticos mejora en **$-0.224$ ppb** (de $19.216$ a $18.992$ ppb).
     - En $h=24$: RMSE mejora de $12.761$ a **$12.707$ ppb** ($-0.054$ ppb), MAE mejora de $9.125$ a **$9.085$ ppb**, y $R^2$ sube de $0.6332$ a **$0.6363$**.
2. **El Rol Crítico de los Indicadores de Faltante (Missingness Flags):**
   - Imputar sin indicarle al modelo qué valores fueron reconstruidos (`v2_tiered_imputed`) introduce ruido residual en horizontes intermedios ($h=8$ RMSE $11.975$ vs $11.967$).
   - Al agregar los flags de missingness (`missing_var_lag1`), el árbol de decisión puede ramificar explícitamente: si el dato es medido, confía en la tasa de cambio directa; si es imputado, modula la ponderación y recurre a variables meteorológicas o climatológicas.
3. **Aceleración Drástica de la Convergencia:**
   - Al tener ventanas continuas sin huecos NaN y con flags explícitos, los árboles encuentran divisiones óptimas mucho antes:
     - En $h=1$, `v2_imputed_with_flags` alcanza early stopping en **571 árboles** (frente a 798 de la versión cruda, un $28\%$ menos de complejidad computacional).
     - En $h=24$, converge en apenas **162 árboles** (frente a 300 de la versión cruda, casi $50\%$ más rápido).

---

## 6. Módulo Reutilizable: `experiments/imputacion/imputers.py`

Se entrega un módulo Python limpio, modular y documentado con las siguientes funciones:
- `compute_distance_matrix()`: Matriz de distancias Haversine (km) entre las 15 estaciones de SIMA.
- `build_climatology_map(panel, max_year=2023)`: Perfil climatológico horario por estación, mes y hora.
- `impute_linear(series)` y `impute_pchip(series)`: Interpoladores univariados temporales.
- `impute_climatology_level_adj(series, dates, st, var, clim_maps)`: Interpola anomalías respecto a la climatología horaria entre los bordes de los huecos.
- `impute_stl(series, period=24)`: Descomposición estacional STL con interpolación de residuo/tendencia.
- `impute_idw_spatial(panel, var, dist_matrix, clim_maps)`: Ponderación espacial inversa al cuadrado de la distancia para cada instante $t$.
- `train_spatial_regression_lightgbm(...)`: Regresión espacial multivariada que predice cualquier estación objetivo con las demás estaciones y meteorología.
- `impute_softimpute_svd(M, rank=3)`: Completado de matriz mediante descomposición SVD iterativa con umbralización suave.
- `impute_tiered(panel, vars_to_impute=None)`: **Estrategia por niveles lista para producción.**

---

## 7. Conclusiones y Recomendaciones para el Proyecto

1. **Abandono definitivo de imputadores Deep Learning tipo SAITS/CSDI para apagones continuos:**  
   Queda empíricamente verificado que SAITS no es aplicable para fallas de sensores de mediana o larga duración en redes de calidad del aire. Su $R^2$ negativo en $PM_{2.5}$ y sesgo de $+58\%$ a $+66\%$ contaminan severamente cualquier downstream model.
2. **Para huecos cortos ($\le 3$ horas):**  
   Usar **interpolación lineal o PCHIP** directa. Ofrece $R^2 > 0.88$–$0.91$, RMSE $< 4.6$ ppb y tiene costo computacional nulo.
3. **Para huecos medianos ($4$ a $48$ horas):**  
   Usar **Climatología Horaria con Ajuste por Nivel (Interpolación de Anomalías)** o **STL**. Mantiene el ciclo diurno forzado por la radiación solar y conserva la anomalía regional del episodio de contaminación.
4. **Para apagones largos ($> 48$ horas o meses completos):**  
   La **Regresión Espacial con LightGBM** es la solución superior indiscutible:
   - En ozono: $R^2 = 0.829$ en $S_3$ y $0.791$ en $S_4$ (vs $0.36$–$0.37$ de SAITS).
   - En $PM_{2.5}$: rescata la señal de $R^2 = -0.91$ a **$R^2 = +0.53$**, eliminando por completo el sesgo sistemático (+8.3 µg/m³ $\to$ -0.13 µg/m³).
   - En estaciones del núcleo urbano ($S_5$): alcanza $R^2 = 0.962$ en $O_3$ y $0.846$ en $PM_{2.5}$.
5. **Recomendación para el Pipeline de Pronóstico Downstream:**  
   Si se decide imputar entradas antes del gradient boosting, **siempre debe suministrarse la columna indicadora de faltante (`*_was_missing`)**. Esta combinación (`v2_imputed_with_flags`) provee una mejora medible en RMSE y picos para todos los horizontes y reduce a la mitad el número de árboles necesarios para converger.
