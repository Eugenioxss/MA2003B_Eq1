# Reporte Multi-Horizonte v2 -- Feature Engineering
**Generado:** 2026-09-11 13:06

## Que hay de nuevo en v2?
Se conservaron los mismos lags base del Script 6 y se agregaron **28 features de ingenieria** en las siguientes categorias:

| Categoria | Features | Descripcion |
|-----------|----------|-------------|
| Interacciones fisicoquimicas | `NOX_x_SR`, `TOUT_x_SR`, `O3_NOX_ratio`, `wind_speed` | Codifican la reaccion quimica del O3 |
| Tasas de cambio | `O3_diff_1h`, `O3_accel`, `O3_diff_24h`, `TOUT_diff_3h`, `NOX_diff_1h`, `SR_diff_1h` | Direccion y velocidad del cambio |
| Rolling stats | `O3_mean_6h`, `O3_std_6h`, `TOUT_mean_6h`, `NOX_mean_6h`, `SR_sum_6h`, `O3_max_24h`, `O3_min_24h`, `O3_range_24h`, `TOUT_max_24h`, `SR_sum_12h` | Resumenes estadisticos de ventana |
| Anomalia vs ayer | `O3_vs_yesterday`, `TOUT_vs_yesterday` | Desviacion del patron diario |
| Contexto urbano | `is_weekend`, `is_rush_hour` | Patrones de trafico |
| Codificacion ciclica | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` | Continuidad temporal |

**Total features:** 80 (vs 52 en v1)

---

## Comparativa v1 vs v2

|   Horizonte |   RMSE v1 |   RMSE v2 |   Delta RMSE |   R2 v1 |   R2 v2 |   Delta R2 |   Mejora RMSE % |
|------------:|----------:|----------:|-------------:|--------:|--------:|-----------:|----------------:|
|           1 |    8.2391 |    8.0509 |      -0.1882 |  0.8468 |  0.8537 |     0.0069 |       2.28423   |
|           4 |   11.4649 |   10.9066 |      -0.5583 |  0.7035 |  0.7317 |     0.0282 |       4.86965   |
|           8 |   12.4179 |   11.9672 |      -0.4507 |  0.6523 |  0.6771 |     0.0248 |       3.62944   |
|          12 |   12.286  |   12.2051 |      -0.0809 |  0.6597 |  0.6642 |     0.0045 |       0.658473  |
|          24 |   12.7518 |   12.7609 |       0.0091 |  0.6338 |  0.6332 |    -0.0006 |      -0.0713625 |
|          48 |   13.9266 |   13.9494 |       0.0228 |  0.5638 |  0.5624 |    -0.0014 |      -0.163715  |


---

## Graficas Generadas

- **`01_comparativa_rmse_v1_vs_v2.png`** -- RMSE comparado: v1 vs v2
- **`02_comparativa_r2_v1_vs_v2.png`** -- R2 comparado: v1 vs v2
- **`03_feature_importance_v2_panel.png`** -- Top 15 features por horizonte (base=verde, nuevo=naranja)
- **`04_heatmap_features_nuevos.png`** -- Heatmap de solo los features nuevos
- **`05_serie_real_vs_pred_v2.png`** -- Series de tiempo: prediccion v2 vs real
- **`06_scatter_v2.png`** -- Dispersion real vs predicho v2
- **`07_mejora_porcentual.png`** -- Mejora porcentual RMSE por horizonte


---

## Top 5 Features Nuevos Mas Utiles por Horizonte

### +1h
| Feature           |   Importance |
|:------------------|-------------:|
| TOUT_vs_yesterday |         1946 |
| O3_diff_1h        |         1657 |
| TOUT_diff_3h      |         1463 |
| wind_speed        |         1447 |
| O3_std_6h         |         1221 |

### +4h
| Feature           |   Importance |
|:------------------|-------------:|
| TOUT_max_24h      |         2422 |
| TOUT_vs_yesterday |         2404 |
| O3_max_24h        |         2149 |
| O3_range_24h      |         1735 |
| TOUT_diff_3h      |         1625 |

### +8h
| Feature           |   Importance |
|:------------------|-------------:|
| TOUT_max_24h      |         1992 |
| O3_max_24h        |         1734 |
| TOUT_vs_yesterday |         1628 |
| O3_range_24h      |         1290 |
| month_cos         |         1053 |

### +12h
| Feature           |   Importance |
|:------------------|-------------:|
| TOUT_max_24h      |         1244 |
| TOUT_vs_yesterday |         1001 |
| O3_max_24h        |          891 |
| month_cos         |          781 |
| O3_range_24h      |          647 |

### +24h
| Feature           |   Importance |
|:------------------|-------------:|
| TOUT_vs_yesterday |          905 |
| TOUT_max_24h      |          779 |
| month_cos         |          557 |
| O3_max_24h        |          454 |
| month_sin         |          425 |

### +48h
| Feature           |   Importance |
|:------------------|-------------:|
| month_cos         |          390 |
| TOUT_max_24h      |          385 |
| TOUT_vs_yesterday |          288 |
| O3_max_24h        |          246 |
| month_sin         |          229 |
