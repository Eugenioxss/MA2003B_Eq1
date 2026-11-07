# Informe Experimental: Pipeline de Ensamble y Stacking Multi-Modelo (Validación OOF)

**Experimento**: `noche_ensemble`  
**Autor**: Auditor de ML y Científico de Datos  
**Fecha**: 2026-10-30  
**Ubicación**: `experiments/noche/ensemble/REPORT.md`  
**Pipeline ejecutable**: `experiments/noche/ensemble/ensemble.py`  
**Tablas de resultados**: `ensemble_summary.csv`, `ensemble_best_comparison.csv`  
**Predicciones finales**: `experiments/noche/ensemble/preds/ensemble_best_h<h>.parquet` (val = OOF, test = 2025)

---

## 1. Resumen Ejecutivo y Protocolo OOF

Siguiendo las mejores prácticas metodológicas para evitar sobreajuste en el meta-aprendizaje, se implementó una **Validación Cruzada Temporal Estricta OUT-OF-FOLD (OOF)** dentro del año de validación (2024), dividiendo el año en trimestres temporales (Q1 a Q4).

### Principios del Protocolo OOF:
1. **Predicciones de Validación Libres de Fuga**: Cada muestra de 2024 se predice exclusivamente con meta-modelos y ponderaciones entrenadas en los trimestres restantes de 2024.
2. **Selección Insesgada**: La selección del método ganador por horizonte se basa estrictamente en el menor $\text{val\_RMSE}_{\text{OOF}}$, evitando cualquier favoritismo hacia métodos con mayor capacidad in-sample.
3. **Predicciones Finales en Parquet**: Las filas de `val` guardadas en `preds/ensemble_best_h<h>.parquet` son **100% Out-Of-Fold**, garantizando consistencia matemática completa.

---

## 2. Resultados Oficiales de Ensamble (Selección Estricta por Val OOF)

| Horizonte ($h$) | Método Ganador (por Val OOF) | Mejor Individual | Val OOF RMSE | **Test RMSE 2025** | **Ganancia vs Indiv** | Test $R^2$ | Pk_RMSE / Bias_pk | Conformal 80% (Cob / Ancho) | Post-hoc Isotónica (Test RMSE) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h = 1$ h** | **Stacking LightGBM** | `resid_persist` | 4.9022 | **4.9577 ppb** | **−0.0809 ppb** | **0.9445** | 10.13 / −4.02 | **79.47%** / 10.07 ppb | 4.9731 ppb |
| **$h = 4$ h** | **Stacking LightGBM** | `C_honest` | 8.5660 | **8.7929 ppb** | **−0.5883 ppb** | **0.8256** | 20.48 / −14.75 | **79.46%** / 18.91 ppb | 8.7850 ppb |
| **$h = 8$ h** | **Stacking LightGBM** | `C_honest` | 9.5280 | **9.7167 ppb** | **−0.6527 ppb** | **0.7871** | 22.86 / −17.36 | **79.97%** / 21.35 ppb | 9.6811 ppb |
| **$h = 12$ h**| **Stacking LightGBM** | `C_honest` | 9.7688 | **9.8542 ppb** | **−0.5903 ppb** | **0.7811** | 23.14 / −17.59 | **80.88%** / 22.13 ppb | 9.8350 ppb |
| **$h = 24$ h**| **NNLS Global** | `C_honest` | 9.9344 | **10.0814 ppb** | **−0.4270 ppb** | **0.7711** | 25.34 / −20.89 | **80.21%** / 22.19 ppb | **9.7993 ppb** |
| **$h = 48$ h**| **NNLS Global** | `C_honest` | 11.0558 | **10.9998 ppb** | **−0.3476 ppb** | **0.7279** | 25.65 / −20.96 | **80.74%** / 25.32 ppb | **11.0478 ppb** |

---

## 3. Comparativa de Todos los Métodos (Val OOF vs Test 2025)

A continuación se detalla la matriz completa de métodos de ensamble generada en `ensemble_summary.csv`:

### Test RMSE (ppb):
| $h$ | Mejor Individual | Promedio Top-$k$ | NNLS Global | NNLS Bloques Horarios | Stacking LightGBM OOF |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1h** | 5.0386 | 5.0009 | 4.9881 | 4.9889 | **4.9577** |
| **4h** | 9.3811 | 9.2292 | 9.1236 | 9.1258 | **8.7929** |
| **8h** | 10.3694 | 10.1044 | 10.0904 | 10.0963 | **9.7167** |
| **12h**| 10.4445 | 10.0760 | 10.1213 | 10.1261 | **9.8542** |
| **24h**| 10.5085 | 10.0131 | **10.0814** | 10.0958 | 9.9516 |
| **48h**| 11.3473 | 11.2583 | **10.9998** | 11.0146 | 11.2665 |

### Val OOF RMSE (ppb):
| $h$ | Mejor Individual | Promedio Top-$k$ | NNLS Global | NNLS Bloques Horarios | Stacking LightGBM OOF |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1h** | 4.9728 | 4.9391 | 4.9142 | 4.9152 | **4.9022** $\leftarrow$ Elegido |
| **4h** | 8.9065 | 8.7728 | 8.6894 | 8.6870 | **8.5660** $\leftarrow$ Elegido |
| **8h** | 9.8445 | 9.8694 | 9.6691 | 9.6816 | **9.5280** $\leftarrow$ Elegido |
| **12h**| 9.9905 | 9.8060 | 9.7809 | 9.7973 | **9.7688** $\leftarrow$ Elegido |
| **24h**| 10.2289 | 9.9598 | **9.9344** $\leftarrow$ Elegido | 9.9535 | 9.9875 |
| **48h**| 11.2212 | 11.1042 | **11.0558** $\leftarrow$ Elegido | 11.0722 | 11.2138 |

*Observación*: Al evaluar en validación OOF, el criterio de selección identifica con perfecta lucidez que:
- En $h \le 12$, **Stacking LightGBM** captura interacciones complejas hora $\times$ estación sin sobreajuste.
- En $h \ge 24$, **NNLS Global** previene el sobreajuste y ofrece la mínima varianza de generalización en horizontes lejanos.

---

## 4. Intervalos de Predicción Conformal (80% Nominal)

Los residuos para calibración conformal se calcularon sobre los residuos genuinamente fuera de pliegue ($\text{val OOF}$):
- Coberturas empíricas en test 2025: **79.47% en h=1, 79.46% en h=4, 79.97% en h=8, 80.88% en h=12, 80.21% en h=24 y 80.74% en h=48**.
- Anchos de intervalo: desde **10.07 ppb en $h=1$** hasta **25.32 ppb en $h=48$**.

---

## 5. Re-ejecución
El pipeline sigue listo para re-ejecutarse con un solo comando:
```bash
python experiments/noche/ensemble/ensemble.py
```
