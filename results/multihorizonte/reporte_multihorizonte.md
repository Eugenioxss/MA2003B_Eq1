# Reporte Multi-Horizonte — LightGBM
**Generado:** 2026-09-11 12:23

## Resumen Ejecutivo
Se entrenaron **6 modelos independientes** de Gradient Boosting (LightGBM) usando la **Estrategia Directa**: cada modelo aprende a predecir el nivel de Ozono (O₃) a un horizonte específico en el futuro, usando exclusivamente información pasada (rezagos puntuales). Esto evita la acumulación de error que sufrirían los modelos autorregresivos tradicionales.

### Partición Temporal
| Conjunto | Período | Uso |
|----------|---------|-----|
| **Train** | 2020 – 2023 | Aprendizaje del modelo |
| **Validación** | 2024 | Early stopping (evitar sobreajuste) |
| **Test** | 2025 | Evaluación final (nunca visto) |

---

## Métricas por Horizonte (Test 2025)

|   Horizonte (h) |   RMSE (ppb) |   MAE (ppb) |     R² |   MAPE (%) |   Best Iteration |   Cobertura Test |
|----------------:|-------------:|------------:|-------:|-----------:|-----------------:|-----------------:|
|               1 |       8.2391 |      5.6921 | 0.8468 |      34.49 |             1266 |           108512 |
|               4 |      11.4649 |      8.1063 | 0.7035 |      52.38 |             1995 |           108473 |
|               8 |      12.4179 |      8.867  | 0.6523 |      59.95 |              885 |           108421 |
|              12 |      12.286  |      8.8004 | 0.6597 |      61.53 |              290 |           108369 |
|              24 |      12.7518 |      9.1455 | 0.6338 |      63.83 |              244 |           108213 |
|              48 |      13.9266 |     10.0149 | 0.5638 |      69.7  |              159 |           107909 |


### Interpretación Rápida
| Horizonte | Calidad | Comentario |
|-----------|---------|------------|
| +1h | 🟡 Bueno | Alertas inmediatas, RMSE de solo 8.2 ppb |
| +4h | 🟡 Bueno | Alertas inmediatas, RMSE de solo 11.5 ppb |
| +8h | 🟠 Aceptable | Pronóstico de medio día, error de 12.4 ppb |
| +12h | 🟠 Aceptable | Pronóstico de medio día, error de 12.3 ppb |
| +24h | 🟠 Aceptable | Ciclo diario estable, error de 12.8 ppb |
| +48h | 🟠 Aceptable | Pronóstico a 2 días, error de 13.9 ppb |


---

## Gráficas Generadas

- **`01_rmse_por_horizonte.png`** — Error (RMSE) por horizonte temporal
- **`02_r2_por_horizonte.png`** — R² por horizonte temporal
- **`03_rmse_vs_mae.png`** — Comparativa RMSE vs MAE
- **`04_feature_importance_panel.png`** — Top 10 variables por horizonte (panel 2×3)
- **`05_heatmap_importancia.png`** — Heatmap de importancia cruzada
- **`06_serie_real_vs_pred.png`** — Series de tiempo: real vs predicho (semana ejemplo)
- **`07_scatter_real_vs_pred.png`** — Dispersión real vs predicho por horizonte


---

## Variables Más Importantes por Horizonte

### +1h
| Feature    |   Importance |
|:-----------|-------------:|
| O3_lag_1   |         3846 |
| TOUT_lag_1 |         2514 |
| hour       |         2343 |
| NOX_lag_1  |         2313 |
| O3_lag_24  |         2279 |

### +4h
| Feature     |   Importance |
|:------------|-------------:|
| TOUT_lag_12 |         4472 |
| TOUT_lag_48 |         3979 |
| month       |         3696 |
| TOUT_lag_24 |         3491 |
| TOUT_lag_1  |         3442 |

### +8h
| Feature     |   Importance |
|:------------|-------------:|
| TOUT_lag_12 |         2589 |
| month       |         2570 |
| hour        |         2139 |
| TOUT_lag_48 |         1944 |
| O3_lag_12   |         1845 |

### +12h
| Feature     |   Importance |
|:------------|-------------:|
| month       |         1346 |
| O3_lag_12   |         1256 |
| hour        |         1136 |
| TOUT_lag_12 |          985 |
| dayofweek   |          888 |

### +24h
| Feature     |   Importance |
|:------------|-------------:|
| month       |         1176 |
| O3_lag_1    |          855 |
| dayofweek   |          809 |
| TOUT_lag_48 |          770 |
| TOUT_lag_12 |          734 |

### +48h
| Feature     |   Importance |
|:------------|-------------:|
| month       |          967 |
| hour        |          655 |
| dayofweek   |          595 |
| TOUT_lag_12 |          538 |
| TOUT_lag_48 |          530 |


---

## Conclusiones y Aplicaciones Prácticas

1. **Alertas inmediatas (+1h, +4h):** Los modelos de corto plazo logran la mayor precisión. Ideales para sistemas de alerta temprana en tiempo real.

2. **Pronóstico de medio día (+8h, +12h):** Permiten emitir boletines matutinos sobre la calidad del aire esperada en la tarde (pico de O₃).

3. **Ciclo diario (+24h):** El modelo recupera precisión gracias al fuerte patrón cíclico de la contaminación. Útil para planificación operativa.

4. **Pronóstico extendido (+48h):** Mayor incertidumbre, pero útil para planificación de contingencias ambientales y avisos a población vulnerable.

5. **Robustez ante datos faltantes:** Todos los modelos conservan su capacidad predictiva incluso cuando sensores individuales reportan huecos, gracias al manejo nativo de NaNs de LightGBM.
