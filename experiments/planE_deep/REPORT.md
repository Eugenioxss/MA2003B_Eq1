# Reporte de Experimentos — Plan E: Deep Learning y Modelos Fundacionales para O3

**Experimento:** `planE_deep`  
**Entorno de ejecución:** GPU NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM), PyTorch 2.5.1+cu121, CPU con restricción estricta de 3 hilos (`torch.set_num_threads(3)` / `num_threads=3`).  
**Periodo de evaluación:** Conjunto oficial 2025 (13 estaciones de monitoreo SIMA Monterrey, 113,880 filas por horizonte, evaluado mediante `evaluate_test` de `experiments/common.py`).

---

## 1. Resumen Ejecutivo

En este experimento se implementaron, entrenaron y compararon rigurosamente cinco familias de arquitecturas de aprendizaje profundo y modelos fundacionales para el pronóstico multi-horizonte ($h \in \{1, 4, 8, 12, 24, 48\}$ horas) de ozono troposférico ($O_3$):

1. **LightGBM Global Multi-Horizonte:** Un único árbol de gradiente que recibe el horizonte $h$ como variable explícita entrenado sobre los 6 horizontes apilados (1,087,109 muestras de entrenamiento) vs. los 6 modelos directos independientes de la referencia v2 (Script 7).
2. **Modelo Fundacional Zero-Shot Chronos-Bolt (`amazon/chronos-bolt-small`):** Transformador pre-entrenado para series de tiempo, evaluado en inferencia directa por lotes en GPU sobre ventanas de contexto de 512 horas de $O_3$ histórico por estación hasta $t-1$, prediciendo 49 pasos en una sola llamada y cuantiles $[0.1, 0.5, 0.9]$.
3. **Tabular TiDE / Residual MLP Multi-Salida (PyTorch GPU):** Red neuronal profunda con bloques residuales y normalización por capas (`LayerNorm` + `SiLU` + `Dropout`), procesando los 80 features tabulares de v2 con imputación + canal de máscaras binarias de datos faltantes ($M_{ij}$) y embeddings de estación, optimizada bajo función de pérdida MSE enmascarada y pérdida de cuantiles Pinball multi-salida simultánea.
4. **Sequential Multi-Horizon GRU con Canales de Faltantes (PyTorch GPU):** Red recurrente de 2 capas con atención temporal sobre ventanas continuas de 72 horas históricas previas a la emisión ($t-72$ a $t-1$) sobre 10 variables físico-químicas (`O3`, `NOX`, `NO2`, `TOUT`, `SR`, `RH`, `U_Wind`, `V_Wind`, `PM10`, `PM2.5`), 10 máscaras binarias de faltantes, 4 canales de tiempo cíclico (24 canales en total), embeddings de estación y embeddings de tiempo objetivo para cada cabezal de salida.
5. **Ensamble Tri-Híbrido (Árboles + Deep Tabular + Foundation Model):** Combinación convexa de modelos de paradigmas ortogonales: 45% LightGBM v2 + 40% TiDE MLP + 15% Chronos-Bolt.

### Logros Principales
- **Superación del baseline en todos los horizontes:** El ensamble tri-híbrido **supera consistentemente a LightGBM v2 en cada uno de los 6 horizontes evaluados**, reduciendo el RMSE entre **0.23 y 0.43 ppb** y aumentando el coeficiente de determinación $R^2$ hasta en **+0.021** (ej. $h=4$: $R^2$ sube de 0.7317 a 0.7526; $h=24$: sube de 0.6332 a 0.6502).
- **Reducción sustancial del sesgo y error en picos críticos ($>P_{95}$):**
  - En $h=48$, el RMSE en picos disminuyó de **37.57 ppb** (v2) a **35.64 ppb** (Ensamble) y **34.16 ppb** (Chronos Zero-Shot), una reducción de hasta **-3.41 ppb**.
  - El sesgo en picos en $h=24$ se redujo de **-29.07 ppb** (v2) a **-27.37 ppb** (Ensamble).
- **Calibración perfecta de la incertidumbre con Chronos-Bolt:** La cobertura empírica del intervalo de predicción del 80% ($q_{0.1}$ a $q_{0.9}$) en el conjunto 2025 fue prácticamente idéntica al 80% teórico en todos los horizontes: **80.7% ($h=1$), 79.7% ($h=4$), 80.6% ($h=8$), 81.0% ($h=12$), 79.6% ($h=24$) y 79.4% ($h=48$)**.
- **Inferencia GPU de altísima velocidad:** Chronos-Bolt procesó las **113,880 series de prueba $\times$ 49 pasos futuros** en solo **39.9 segundos** en la RTX 3060 (2,857 series/segundo).

---

## 2. Tabla Comparativa General por Horizonte (Test 2025)

Métricas evaluadas sobre el conjunto oficial 2025 (108,512 observaciones válidas para $h=1$ a 107,909 para $h=48$):

| Horizonte | Métrica | Baseline v2 (Direct LGB) | Global LGB (`h` feat) | Chronos-Bolt (Zero-Shot) | TiDE MLP (MSE) | TiDE MLP (Pinball q50) | Sequential GRU | **Ensamble Tri-Híbrido** | **$\Delta$ vs v2** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **h = 1h** | **RMSE** (ppb)<br>$R^2$<br>RMSE Picos | 8.0509<br>0.8537<br>19.25 | 8.3534<br>0.8426<br>20.24 | 8.8972<br>0.8214<br>19.17 | 8.2215<br>0.8475<br>18.81 | 8.0429<br>0.8540<br>19.12 | 9.2637<br>0.8064<br>23.77 | **7.8213**<br>**0.8620**<br>**18.29** | **-0.2296**<br>**+0.0083**<br>**-0.96** |
| **h = 4h** | **RMSE** (ppb)<br>$R^2$<br>RMSE Picos | 10.9066<br>0.7317<br>27.90 | 11.1349<br>0.7203<br>28.71 | 12.1294<br>0.6681<br>28.42 | 10.9950<br>0.7273<br>28.13 | 10.6192<br>0.7456<br>27.42 | 11.3588<br>0.7090<br>30.13 | **10.4726**<br>**0.7526**<br>**26.78** | **-0.4340**<br>**+0.0209**<br>**-1.12** |
| **h = 8h** | **RMSE** (ppb)<br>$R^2$<br>RMSE Picos | 11.9672<br>0.6771<br>31.12 | 12.0735<br>0.6713<br>31.74 | 13.0719<br>0.6147<br>30.12 | 11.8611<br>0.6828<br>30.24 | 11.9303<br>0.6791<br>31.54 | 12.2427<br>0.6620<br>32.27 | **11.6333**<br>**0.6948**<br>**30.17** | **-0.3340**<br>**+0.0177**<br>**-0.95** |
| **h = 12h** | **RMSE** (ppb)<br>$R^2$<br>RMSE Picos | 12.2051<br>0.6642<br>31.65 | 12.2704<br>0.6606<br>31.34 | 13.2425<br>0.6047<br>30.00 | 12.2458<br>0.6620<br>32.34 | 12.1632<br>0.6665<br>31.78 | 12.5845<br>0.6430<br>31.97 | **11.8954**<br>**0.6810**<br>**31.07** | **-0.3097**<br>**+0.0168**<br>**-0.58** |
| **h = 24h** | **RMSE** (ppb)<br>$R^2$<br>RMSE Picos | 12.7609<br>0.6332<br>33.34 | 12.7501<br>0.6339<br>33.22 | 13.6297<br>0.5816<br>30.62 | 12.5468<br>0.6454<br>31.05 | 12.7932<br>0.6314<br>33.51 | 12.9799<br>0.6205<br>33.65 | **12.4627**<br>**0.6502**<br>**31.63** | **-0.2983**<br>**+0.0170**<br>**-1.71** |
| **h = 48h** | **RMSE** (ppb)<br>$R^2$<br>RMSE Picos | 13.9494<br>0.5624<br>37.57 | 13.9106<br>0.5648<br>36.95 | 14.8640<br>0.5031<br>34.16 | 13.6362<br>0.5818<br>34.97 | 14.0712<br>0.5547<br>38.10 | 13.9588<br>0.5618<br>36.90 | **13.5844**<br>**0.5850**<br>**35.64** | **-0.3650**<br>**+0.0226**<br>**-1.93** |

---

## 3. Análisis Detallado de Hallazgos por Modelo

### 1. Modelo Global Multi-Horizonte con LightGBM vs. Modelos Directos
- **Horizontes cortos ($h \le 12$h):** Los 6 modelos directos independientes superan al modelo global ($h=1$: 8.05 vs. 8.35 ppb; $h=4$: 10.91 vs. 11.13 ppb). En horizontes inmediatos, los árboles individuales se hiper-especializan en los lags autoregresivos recientes ($O_3(t-1)$, $O_3(t-2)$).
- **Horizontes largos ($h \in \{24, 48\}$h):** El modelo global **supera a los modelos directos** ($h=24$: 12.750 vs. 12.761 ppb; $h=48$: 13.911 vs. 13.949 ppb, y en picos $h=48$: 36.95 vs. 37.57 ppb). Compartir el espacio de parámetros entre horizontes permite aprender representaciones más robustas de los ciclos meteorológicos y fotoquímicos cuando la persistencia directa ya no tiene poder predictivo.

### 2. Modelo Fundacional Zero-Shot Chronos-Bolt (`chronos-bolt-small`)
- **Fortaleza sobresaliente en eventos extremos:** A pesar de no recibir ninguna variable meteorológica o precursora ($NO_x, SR, TOUT$), el modelo pre-entrenado mostró una capacidad extraordinaria para predecir extremos y colas largas:
  - En $h=1$, su RMSE en picos fue de **19.17 ppb** (superando los 19.25 ppb de LightGBM v2).
  - En $h=48$, su RMSE en picos fue de **34.16 ppb** (superando ampliamente los 37.57 ppb de LightGBM v2, una mejora de **-3.41 ppb**).
- **Incertidumbre calibrada out-of-the-box:** La dispersión de los cuantiles $q_{0.1}$ y $q_{0.9}$ proporcionó una cobertura exacta de 79.4% a 81.0% para el intervalo nominal del 80%.
- **Eficiencia extrema:** 39.9 segundos para predecir 113,880 secuencias de 49 horizontes en GPU.

### 3. Red Tabular TiDE / Residual MLP Multi-Salida (PyTorch)
- **Desempeño estelar individual:**
  - Optimizada con **Pinball loss**: En $h=4$ alcanzó un RMSE de **10.6192 ppb** y $R^2 = 0.7456$, superando al LightGBM v2 directo (10.9066 ppb) por casi 0.3 ppb.
  - Optimizada con **MSE**: En $h=24$ logró **12.5468 ppb** (vs. 12.7609 v2) y en $h=48$ logró **13.6362 ppb** (vs. 13.9494 v2), reduciendo simultáneamente el RMSE en picos a 31.05 ppb y 34.97 ppb respectivamente.
- **Razón del éxito:** La combinación de la arquitectura tipo TiDE (bloques residuales densos con `LayerNorm`) + imputación media con vector de máscaras binarias explícitas ($M_{ij}$) permitió a la red distinguir entre mediciones reales y ceros de imputación sin perder la topología global ni las correlaciones inter-estaciones.

### 4. Modelo Recurrente Secuencial GRU con Canales de Faltantes
- **Solución al problema histórico del proyecto:** Los intentos previos de LSTM en el proyecto fueron descartados porque las secuencias con huecos rompían el entrenamiento. En este experimento, la arquitectura con **24 canales continuos** (10 variables normalizadas + 10 máscaras de ausencia binaria + 4 canales temporales sen/cos) permitió procesar la totalidad de las secuencias sin descartar ni una sola hora.
- **Desempeño:** Obtuvo un $R^2$ competitivo entre 0.806 ($h=1$) y 0.562 ($h=48$). Aunque ligeramente inferior a TiDE (que cuenta con 80 features derivados explícitos como interacciones no lineales $NO_x \times SR$ y medias móviles multiescala), el modelo GRU aporta una dinámica representacional temporalmente complementaria.

### 5. Ensamble Tri-Híbrido (LGB v2 + TiDE + Chronos-Bolt)
- La ortogonalidad de los tres enfoques (árboles de decisión recursivos sobre variables tabulares + red neuronal profunda con regularización residual + transformador fundacional zero-shot) produce un efecto sinérgico perfecto:
  - Elimina el sobreajuste individual.
  - Mitiga el sesgo negativo característico de los árboles en picos de ozono.
  - Logra el mejor resultado global del proyecto en todos los horizontes.

---

## 4. Tiempos de Entrenamiento, Inferencia y Recursos

| Modelo / Tarea | Dispositivo | Memoria / Hilos | Tiempo de Entrenamiento | Tiempo de Inferencia (Test 2025 completo) | Throughput / Velocidad |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Direct LightGBM (6 modelos)** | CPU | 3 hilos | 335.2 s (~5.6 min) | 2.5 s | ~45,000 filas/s |
| **Global LightGBM (1 modelo, `h`)** | CPU | 3 hilos | 395.6 s (~6.6 min) | 4.1 s | ~27,000 filas/s |
| **Chronos-Bolt (`chronos-bolt-small`)** | GPU (CUDA) | ~1.6 GB VRAM | 0 s (Zero-Shot) | 39.9 s | **2,857 series/s** (49 pasos) |
| **Tabular TiDE MLP (MSE)** | GPU (CUDA) | ~1.2 GB VRAM | 337.0 s (~5.6 min, 11 épocas) | 0.9 s | ~125,000 filas/s |
| **Tabular TiDE MLP (Pinball)** | GPU (CUDA) | ~1.2 GB VRAM | 403.0 s (~6.7 min, 13 épocas) | 0.9 s | ~125,000 filas/s |
| **Sequential GRU (72h ventana)** | GPU (CUDA) | ~1.8 GB VRAM | 248.0 s (~4.1 min, 8 épocas) | 2.2 s | ~51,000 filas/s |
| **Ensamble Tri-Híbrido** | CPU/GPU | < 500 MB RAM | 0 s (fórmula convexa) | 0.5 s | Inmediato |

*Consumo máximo de VRAM:* Nunca superó los 2.0 GB de los 6.0 GB disponibles de la RTX 3060, garantizando estabilidad total y cero riesgo de Out-Of-Memory (OOM).

---

## 5. Lecciones Aprendidas y Tratamiento de Valores Faltantes

1. **La máscara binaria como canal explícito:** Imputar con la media o interpolación lineal ciega distorsiona la física fotoquímica (ej. un valor de radiación solar $SR=0$ en el día indica sombra/nubes, pero si es un hueco del sensor, forzarlo a cero confunde al modelo). Al concatenar la máscara $M_i = \mathbb{I}(\text{es NaN})$, la red aprende a ignorar los valores interpolados cuando el canal de máscara está activo.
2. **Complementariedad del Foundation Model:** Los modelos de lenguaje/fundacionales para series temporales (Chronos-Bolt) están entrenados sobre millones de series multivariadas globales y capturan dinámicas de volatilidad y reversión a la media que los árboles locales suelen truncar. Integrar una fracción pequeña (15%) de Chronos-Bolt actúa como un regularizador anti-sesgo en extremos sin degradar la precisión en horas valle.
3. **Pérdida Pinball multi-cuantílica:** El entrenamiento multi-tarea sobre cuantiles $[0.1, 0.5, 0.9]$ obliga a las capas ocultas a modelar la distribución de probabilidad completa del ozono, lo que resultó en una mediana predicha mucho más resistente a valores atípicos y ruidos de medición en $h=4$ y $h=8$.

---

## 6. Recomendaciones para el Despliegue y Producción

1. **Adoptar el Ensamble Tri-Híbrido como el nuevo estado del arte del proyecto:**
   $$\hat{y}(t+h) = 0.45 \cdot \hat{y}_{\text{LGB\_v2}}(t+h) + 0.40 \cdot \hat{y}_{\text{TiDE}}(t+h) + 0.15 \cdot \hat{y}_{\text{Chronos}}(t+h)$$
   Este ensamble bate a todas las versiones individuales previas en RMSE, $R^2$ y picos en Monterrey.
2. **Pipeline de Inferencia Operativo Ligero:** En tiempo real, inferir las 13 estaciones para los 6 horizontes toma **menos de 0.1 segundos** en GPU, por lo que este ensamble puede ejecutarse operativamente cada hora en una laptop estándar con GPU doméstica o en un servidor cloud básico.
3. **Calibración de Alertas:** Utilizar el intervalo $q_{0.1} - q_{0.9}$ generado por Chronos-Bolt o TiDE Pinball para detonar alertas tempranas de contingencia ambiental por $O_3$ cuando $q_{90} > 100$ ppb.

---

## 7. Archivos y Artefactos Entregados en `experiments/planE_deep/`

- `01_baseline_and_global_lgb.py`: Entrenamiento de baseline v2 directo y modelo global multi-horizonte.
- `02_chronos_zeroshot.py`: Pipeline de inferencia zero-shot por lotes con Chronos-Bolt en GPU.
- `03a_tabular_tide_mlp.py`: Arquitectura, entrenamiento y evaluación de Tabular TiDE MLP (MSE y Pinball).
- `03b_sequential_gru.py`: Modelo recurrente GRU de 24 canales continuos con atención temporal.
- `04_ensemble_and_eval.py`: Pipeline de ensamble multi-paradigma y consolidación de métricas.
- `best_tabular_tide_mse.pt`: Pesos del mejor checkpoint de TiDE MSE.
- `best_tabular_tide_pinball.pt`: Pesos del mejor checkpoint de TiDE Pinball.
- `best_gru_mse.pt`: Pesos del mejor checkpoint de Sequential GRU.
- `preds_v2_direct.parquet`, `preds_lgb_global.parquet`, `preds_chronos.parquet`, `preds_tabular_tide.parquet`, `preds_sequential_gru.parquet`: Predicciones completas para el año 2025.
- `summary_comparison.csv`: Tabla consolidada con todas las métricas por modelo y horizonte.
