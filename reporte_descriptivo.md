# Etapa 2 — Comprensión de los Datos y Análisis Descriptivo

**Proyecto:** Análisis de Calidad del Aire — SIMA (Sistema Integral de Monitoreo Ambiental)  
**Equipo 1 · MA2003B · Tecnológico de Monterrey**  
**Fecha:** Agosto 2026

---

## 1. Objetivo del Análisis

En la Entrega 1 ("Conociendo el Negocio"), el objetivo general del proyecto fue *analizar la variación de la concentración de partículas contaminantes entre distintas regiones en función de las condiciones climáticas y de la hora del día*. Para esta segunda entrega, se realizó un enfoque (*narrowing*) de dicho objetivo hacia la Pregunta de Investigación 2 del proyecto: **la relación entre la disminución de PM2.5/PM10 y el aumento de O3**. Esto no representa un cambio de objetivo, sino una delimitación hacia una hipótesis específica y comprobable para esta etapa.

**Hipótesis de "Bloqueo Radiativo":** Las altas concentraciones de partículas (PM2.5 y PM10) bloquean la radiación solar (SR); como el ozono (O3) se forma fotoquímicamente a partir de precursores en presencia de radiación solar, se espera una correlación negativa entre partículas y O3, mediada por SR.

## 2. Técnicas Estadísticas Utilizadas

- **Medidas de tendencia central:** promedio, mediana y moda (con la salvedad de que la moda tiene interpretación limitada en variables continuas; se reporta redondeada a 1 decimal).
- **Medidas de dispersión:** rango, varianza y desviación estándar.
- **Medidas de posición no central:** Q1, Q3 y rango intercuartílico (IQR).
- **Detección de outliers:** criterio de IQR — valores fuera de $[Q1 - 1.5 \cdot IQR,\ Q3 + 1.5 \cdot IQR]$.
- **Coeficientes de correlación:** Pearson (relaciones lineales) y Spearman (relaciones monótonas). Dado que los histogramas revelan sesgo positivo en la mayoría de las variables, Spearman es más robusto y se incluye como complemento. La comparación entre ambos permite evaluar si las relaciones detectadas son lineales o no lineales.
- **Distribución de frecuencia:** para la variable cualitativa `Estacion` (conteo absoluto y porcentaje relativo). No se calcula la mediana para `Estacion` por ser una variable nominal sin orden natural inherente.

## 3. Variables y su Rol en el Análisis

| Variable | Descripción | Rol |
|----------|-------------|-----|
| O3 | Ozono troposférico (ppb) | Variable objetivo |
| PM2.5 | Material particulado fino ≤ 2.5 μm (μg/m³) | Predictora |
| PM10 | Material particulado ≤ 10 μm (μg/m³) | Predictora |
| SR | Radiación solar (kW/m²) | Predictora / mediadora |
| TOUT | Temperatura exterior (°C) | Predictora |
| Estacion | Estación de monitoreo del SIMA | Cualitativa — variabilidad espacial |

La variable `Estacion` es clave porque permite capturar la variabilidad espacial: distintas zonas del Área Metropolitana de Monterrey tienen diferentes fuentes de emisión (industria, tráfico vehicular, topografía), lo cual puede modular la relación entre partículas y ozono.

## 4. Dimensión y Calidad de los Datos

### 4.1 Filas por archivo de entrada

| Archivo | Filas |
|---------|------:|
| O3_clean.parquet | 643,436 |
| PM2.5_clean.parquet | 564,250 |
| PM10_clean.parquet | 719,672 |
| SR_clean.parquet | 725,814 |
| TOUT_clean.parquet | 701,649 |

### 4.2 Resultado del inner join

Se utilizó un **inner join** secuencial sobre las llaves `Date` y `Estacion` para unir los 5 archivos.

| Etapa | Filas |
|-------|------:|
| Después de join O3 + PM2.5 | 494,834 |
| + PM10 | 487,723 |
| + SR | 473,319 |
| + TOUT | 456,519 |
| Después de `dropna` | 456,519 (0 removidas) |
| Después de `drop_duplicates` | 456,519 (0 removidas) |

**Dataset final:** 456,519 filas × 7 columnas.

- Pérdida respecto al archivo más restrictivo (PM2.5, 564,250 filas): **19.09%**
- Pérdida respecto al archivo más grande (SR, 725,814 filas): **37.10%**

La pérdida del 19% se debe a que no todas las estaciones ni todos los timestamps tienen registros simultáneos en las 5 variables; el inner join conserva únicamente las intersecciones completas. No se encontraron valores nulos ni duplicados en el dataset resultante. Los 15 nombres de estación fueron consistentes entre los 5 archivos (sin typos ni diferencias de escritura).

## 5. Análisis Estadístico

### 5.1 Tendencia central y dispersión

| Variable | N | Promedio | Mediana | Moda (1 dec) | Rango | Varianza | Desv. Estándar |
|----------|--:|--------:|--------:|-------------:|------:|---------:|---------------:|
| O3 | 456,519 | 27.22 | 23.00 | 16.0 | 183.00 | 370.50 | 19.25 |
| PM2.5 | 456,519 | 20.40 | 17.00 | 10.0 | 999.00 | 234.83 | 15.32 |
| PM10 | 456,519 | 60.92 | 52.00 | 39.0 | 997.00 | 1,664.72 | 40.80 |
| SR | 456,519 | 0.15 | 0.008 | 0.0 | 1.25 | 0.048 | 0.22 |
| TOUT | 456,519 | 23.59 | 24.22 | 24.4 | 51.12 | 49.04 | 7.00 |

*Nota sobre la moda:* Al tratarse de variables continuas, la moda se calculó redondeando los valores a 1 decimal. Su interpretación es limitada (indica el valor discretizado más frecuente, no un pico estadísticamente robusto). La moda de SR = 0.0 refleja la alta proporción de registros nocturnos donde la radiación solar es nula.

### 5.2 Posición no central y outliers

| Variable | Q1 | Q3 | IQR | Outliers (n) | Outliers (%) |
|----------|---:|---:|----:|-------------:|-------------:|
| O3 | 13.00 | 37.00 | 24.00 | 12,570 | 2.75% |
| PM2.5 | 10.47 | 26.35 | 15.88 | 16,875 | 3.70% |
| PM10 | 36.00 | 74.00 | 38.00 | 21,751 | 4.76% |
| SR | 0.00 | 0.25 | 0.25 | 23,859 | 5.23% |
| TOUT | 19.18 | 28.45 | 9.27 | 4,023 | 0.88% |

Las variables de contaminantes (PM2.5 y PM10) presentan un porcentaje de outliers entre 3.7% y 4.8%, consistente con episodios de contaminación elevada. TOUT es la variable más compacta (0.88% de outliers), reflejando el rango térmico acotado de la región.

### 5.3 Distribución de frecuencia por estación

| Estación | Conteo | % Relativo |
|----------|-------:|-----------:|
| CE | 42,013 | 9.20% |
| NE | 27,433 | 6.01% |
| NE2 | 27,554 | 6.04% |
| NE3 | 1,992 | 0.44% |
| NO | 18,208 | 3.99% |
| NO2 | 33,215 | 7.28% |
| NO3 | 3,378 | 0.74% |
| NTE | 30,703 | 6.73% |
| NTE2 | 38,189 | 8.37% |
| SE | 41,044 | 8.99% |
| SE2 | 34,780 | 7.62% |
| SE3 | 42,557 | 9.32% |
| SO | 41,098 | 9.00% |
| SO2 | 42,889 | 9.39% |
| SUR | 31,466 | 6.89% |

Las estaciones NE3 (0.44%) y NO3 (0.74%) tienen una representación significativamente menor. Las conclusiones basadas en estas estaciones deben interpretarse con cautela por el reducido tamaño muestral.

*Nota:* No se calcula la mediana para `Estacion` por ser una variable nominal (categórica sin orden natural). Su análisis se limita a la distribución de frecuencia absoluta y relativa.

### 5.4 Matriz de correlación de Pearson

|       | O3 | PM2.5 | PM10 | SR | TOUT |
|-------|---:|------:|-----:|---:|-----:|
| **O3** | 1.000 | **0.008** | **0.014** | **0.530** | **0.488** |
| **PM2.5** | 0.008 | 1.000 | 0.606 | 0.124 | 0.014 |
| **PM10** | 0.014 | 0.606 | 1.000 | 0.125 | 0.018 |
| **SR** | 0.530 | 0.124 | 0.125 | 1.000 | 0.392 |
| **TOUT** | 0.488 | 0.014 | 0.018 | 0.392 | 1.000 |

### 5.5 Matriz de correlación de Spearman

|       | O3 | PM2.5 | PM10 | SR | TOUT |
|-------|---:|------:|-----:|---:|-----:|
| **O3** | 1.000 | **−0.036** | **0.004** | **0.455** | **0.507** |
| **PM2.5** | −0.036 | 1.000 | 0.633 | 0.176 | 0.072 |
| **PM10** | 0.004 | 0.633 | 1.000 | 0.176 | 0.084 |
| **SR** | 0.455 | 0.176 | 0.176 | 1.000 | 0.346 |
| **TOUT** | 0.507 | 0.072 | 0.084 | 0.346 | 1.000 |

**Hallazgo clave:** La correlación O3–PM2.5 es esencialmente nula tanto en Pearson (0.008) como en Spearman (−0.036), y la correlación O3–PM10 es igualmente débil (0.014 Pearson, 0.004 Spearman). La hipótesis de bloqueo radiativo predecía una correlación negativa entre partículas y O3, pero los datos agregados no la sustentan de forma clara. En contraste, O3 sí tiene una correlación moderada-fuerte con SR (0.530) y TOUT (0.488), consistente con la formación fotoquímica del ozono. La correlación PM–SR es positiva pero débil (~0.12), lo cual contradice parcialmente el mecanismo de bloqueo.

Es importante destacar que **correlación no implica causalidad**: la relación O3–SR podría estar confundida por el ciclo diurno (ambas variables aumentan durante el día) y otros factores no incluidos en este análisis (NOx, velocidad del viento, etc.).

### 5.6 Correlación O3 vs PM por estación (Pearson)

| Estación | O3 vs PM2.5 | O3 vs PM10 | n |
|----------|------------:|-----------:|--:|
| CE | −0.041 | 0.042 | 42,013 |
| NE | −0.079 | 0.004 | 27,433 |
| NE2 | 0.117 | 0.028 | 27,554 |
| **NE3** | **−0.312** | −0.045 | 1,992 |
| NO | −0.119 | −0.070 | 18,208 |
| NO2 | 0.052 | 0.179 | 33,215 |
| NO3 | −0.008 | 0.155 | 3,378 |
| NTE | −0.088 | −0.069 | 30,703 |
| NTE2 | 0.124 | −0.014 | 38,189 |
| SE | 0.075 | 0.210 | 41,044 |
| SE2 | −0.145 | −0.098 | 34,780 |
| SE3 | 0.031 | −0.098 | 42,557 |
| SO | −0.137 | −0.054 | 41,098 |
| SO2 | 0.193 | 0.124 | 42,889 |
| SUR | 0.261 | 0.226 | 31,466 |

La estación NE3 muestra la correlación negativa más fuerte O3–PM2.5 (−0.312), pero su muestra es muy pequeña (n = 1,992) y este resultado debe tomarse con cautela. Las estaciones NO, NTE, SE2 y SO muestran correlaciones negativas leves (−0.07 a −0.15). En contraste, SUR y SO2 muestran correlaciones **positivas** (0.19 a 0.26). Este patrón mixto sugiere que la relación partículas–ozono varía considerablemente entre zonas, posiblemente por diferencias en las fuentes de emisión y condiciones locales.

## 6. Interpretación de Visualizaciones

### 6.1 Histogramas con KDE (`histogramas_distribucion.png`)

Las tres variables de contaminantes presentan **sesgo positivo** (cola derecha), con valores de skewness de ~1.1 para O3, ~2.5 para PM2.5 y ~2.8 para PM10. En todos los casos, la media se ubica a la derecha de la mediana, confirmando la asimetría. Este sesgo es esperado en datos ambientales: la mayoría de las mediciones se concentran en niveles bajos-moderados, con episodios esporádicos de alta contaminación que generan la cola derecha. Dado este sesgo, los coeficientes de Spearman son más apropiados que los de Pearson para evaluar relaciones monótonas.

### 6.2 Boxplots por estación (`boxplot_estaciones.png`)

Los boxplots revelan diferencias espaciales importantes. Las estaciones del sur y sureste tienden a reportar niveles más elevados de O3, consistente con la dirección predominante de los vientos que transportan precursores. PM2.5 y PM10 muestran mayor variabilidad en las estaciones del noreste y centro, donde la actividad industrial es más intensa. Todas las estaciones presentan outliers superiores en las tres variables.

### 6.3 Heatmap de correlación (`heatmap_correlacion.png`)

El heatmap confirma visualmente los hallazgos numéricos: la correlación más fuerte es O3–SR (0.53), seguida de O3–TOUT (0.49) y PM2.5–PM10 (0.61, esperada por ser ambas medidas de material particulado). Las celdas O3–PM2.5 y O3–PM10 aparecen prácticamente en blanco (valores cercanos a cero), indicando ausencia de relación lineal a nivel agregado.

### 6.4 Scatterplot PM2.5 vs O3 coloreado por SR (`scatter_pm_o3_sr.png`)

Esta visualización es la prueba más directa de la hipótesis de bloqueo radiativo. Se observa:

- La nube de puntos no muestra una tendencia lineal clara entre PM2.5 y O3, consistente con la correlación cercana a cero.
- Los puntos con alta radiación solar (colores cálidos) tienden a ubicarse en la zona de O3 alto, independientemente del nivel de PM2.5. Esto sugiere que la formación de O3 depende más directamente de SR que del nivel de partículas.
- No se aprecia el patrón esperado de "altas partículas → baja SR → bajo O3" de forma evidente en los datos agregados.

**Interpretación cautelosa:** Estos resultados no invalidan la hipótesis de bloqueo radiativo, pero sugieren que, a nivel de datos horarios agregados, la relación no es lo suficientemente fuerte para manifestarse como una correlación lineal simple. Factores confusores (ciclo diurno, estacionalidad, concentración de precursores NOx) podrían estar enmascarando el efecto. Correlación no implica causalidad, y la ausencia de correlación lineal tampoco descarta un mecanismo causal que opere de forma no lineal o con rezagos temporales.

### 6.5 Distribución por estación (`piechart_estaciones.png`)

La distribución de registros es razonablemente homogénea entre la mayoría de las estaciones (6–9.4%), con la excepción de NE3 (0.44%) y NO3 (0.74%), que son estaciones con cobertura temporal más limitada en el periodo analizado.

## 7. Resumen y Próximos Pasos

### ¿Qué se logró?
- Se integró un dataset limpio de 456,519 registros con 5 variables numéricas y 15 estaciones.
- Se completó el análisis descriptivo univariado (tendencia central, dispersión, posición, outliers) y bivariado (correlaciones Pearson/Spearman, desglose por estación).
- Se generaron 5 visualizaciones que caracterizan la distribución de los datos y las relaciones entre variables.

### ¿Qué se encontró?
- La hipótesis de bloqueo radiativo **no se sustenta de forma clara** en los datos agregados: la correlación O3–partículas es esencialmente nula a nivel global, aunque varía por estación (de −0.31 a +0.26 para O3–PM2.5).
- El O3 tiene una correlación moderada con SR (0.53) y TOUT (0.49), consistente con su origen fotoquímico.
- El efecto es espacialmente heterogéneo: algunas estaciones muestran correlaciones negativas leves (NE3, SO, SE2), otras positivas (SUR, SO2).

### Próximos pasos
- Incorporar las variables de **NOx** (NO, NO2, NOX) como precursores directos del ozono para evaluar si la relación PM–O3 se aclara al controlar por la concentración de precursores.
- Utilizar las **variables de rezago temporal** construidas en la Etapa 1 (e.g., PM2.5 con lag de 1–3 horas) para capturar posibles efectos retardados del bloqueo radiativo.
- Explorar la relación partículas–O3 **segmentada por franja horaria** (ciclo diurno) y por **temporada** (invierno vs. verano), ya que el mecanismo fotoquímico opera principalmente durante las horas de mayor insolación.
- Avanzar hacia la construcción de un modelo predictivo multivariado en la Etapa 3, partiendo del dataset `dataset_ozono_predictivo.parquet` ya preparado.

### Preguntas nuevas
- ¿El efecto de bloqueo es más pronunciado en horas de alta insolación (10:00–16:00)?
- ¿La correlación negativa observada en NE3 y SO es un artefacto del tamaño muestral o refleja condiciones locales específicas?
- ¿Un modelo con interacción PM × SR captura mejor el mecanismo que correlaciones marginales?

## 8. Anexos y Referencias

### Lista de figuras generadas

| Archivo | Descripción |
|---------|-------------|
| `results/figures/boxplot_estaciones.png` | Boxplots de O3, PM2.5 y PM10 por estación |
| `results/figures/histogramas_distribucion.png` | Histogramas + KDE con indicadores de sesgo |
| `results/figures/heatmap_correlacion.png` | Heatmap de correlación de Pearson |
| `results/figures/piechart_estaciones.png` | Distribución de registros por estación |
| `results/figures/scatter_pm_o3_sr.png` | Scatterplot PM2.5 vs O3 coloreado por SR |

### Repositorio

[https://github.com/Eugenioxss/MA2003B_Eq1](https://github.com/Eugenioxss/MA2003B_Eq1)

### Declaratoria de Uso de IA

**Opción B. Se utilizó IA especificando:**

1. **Herramienta:** Claude / Antigravity.
2. **Uso realizado:** Generación del script de análisis descriptivo (`etapa2_analisis.py`), estructuración del reporte, y asistencia en la interpretación estadística.
3. **Secciones donde se utilizó:** Script de carga/limpieza de datos, cálculo de estadísticas descriptivas, generación de visualizaciones y redacción del presente reporte.
4. **Validación realizada por los estudiantes:** *(completar a mano — indicar qué cifras, gráficas e interpretaciones fueron verificadas por el equipo)*.
5. **Confirmación:** *(completar a mano — firmar responsabilidad sobre el contenido final)*.

---

## Notas Metodológicas (Registro de Decisiones de Ingeniería de Datos)

1. **Tipo de join:** Inner join secuencial sobre (`Date`, `Estacion`). Se eligió inner join (en lugar de left/outer) para garantizar que cada registro del dataset final tenga valores para las 5 variables, evitando imputaciones que podrían sesgar las correlaciones.
2. **Verificación de consistencia:** Los 5 archivos comparten las mismas 15 estaciones con escritura idéntica. No se requirieron correcciones de typos ni normalización de nombres.
3. **Tipos de dato:** `Date` es `datetime64[us]` y `Estacion` es `string` en los 5 archivos — no se requirió conversión.
4. **Manejo de la moda:** Para variables continuas, los valores se redondearon a 1 decimal antes de calcular la moda. En caso de empate, se reporta el primer valor modal. La moda de SR = 0.0 refleja registros nocturnos.
5. **Duplicados y nulos:** `dropna()` y `drop_duplicates(subset=['Date','Estacion'])` no removieron filas, confirmando la integridad del pipeline de limpieza previo (Etapa 1).
6. **Pérdida de datos:** El inner join redujo el dataset en un 19.09% respecto al archivo más restrictivo (PM2.5). Esta pérdida se debe a la asincronía temporal entre los sensores de distintas variables.
7. **Outliers:** Se reportan pero no se removieron del análisis. En datos ambientales, los valores extremos frecuentemente representan eventos reales (episodios de contaminación) y su eliminación podría sesgar las conclusiones.
8. **Correlación por estación:** Solo se calculó para estaciones con n > 10 registros. Todas las 15 estaciones cumplieron este umbral.
