# Informe Experimental: Arquitectura de Ensambles, Meta-Aprendizaje y Calibración Conformal (Ola 2 - Pasada 5 Definitiva)

**Experimento**: `noche2_ens2`  
**Autor**: Arquitecto Principal de Ensambles, Meta-Aprendizaje y Calibración Estadística  
**Fecha**: 2026-11-06  
**Ubicación**: `experiments/noche2/ens2/REPORT.md`  
**Pipeline ejecutable**: `experiments/noche2/ens2/ensemble_pipeline.py`  
**Auditoría y Diversidad**: `experiments/noche2/ens2/audit_candidates.py`  
**Predicciones finales**: `experiments/noche2/ens2/preds/ens2_best_h<h>.parquet`  
**Registro global**: Registrado en `experiments/results_registry.csv` e inventariado en `experiments/noche2/inventory.csv`

---

## 1. Resumen Ejecutivo y Resultados Oficiales (Pasada 5 Definitiva)

Siguiendo el contrato de honestidad estricta para la corrida `noche2` (split temporal 2020–2023 train, 2024 val, 2025 test), se ejecutó la **Pasada 5 (definitiva)** del sistema de **ensamble multi-familia, meta-aprendizaje y calibración conformal** bajo el **Protocolo de Validación Cruzada Temporal Out-of-Fold (OOF) en 4 Trimestres (Q1 a Q4 de 2024)**.

En esta **Pasada 5**, el sistema integró los **268 modelos candidatos válidos** de **14 familias metodológicas**, destacando la entrada completa de la familia **`Deep_Imput2_Resid` (`deep_imput2_resid_refit_h{1,4,8,12,24,48}`)** —el ensamble neuronal TabResNet + TabGated SwiGLU entrenado sobre el panel imputado causalmente y parametrizado en residuos de persistencia $\Delta O_3 = O_3(t+h) - O_3(t)$—, así como `refit_nwp_resid_persist_h1`, `imput2_nwp_refit_h1`, `imput2_nwp_refit_w_h1`, `xgb_nwp_refit_w_h1` y `et_nwp_refit_h1`.

### Resultados Oficiales Ganadores (Pasada 5 Definitiva vs Referencias)

| Horizonte ($h$) | Método Ganador (por Val OOF) | Mejor Modelo Individual | Val OOF RMSE | **Test RMSE 2025** | **Ganancia vs Indiv** | **Ganancia vs Noche 1** | Test $R^2$ | Pk_RMSE / Bias_pk | Conformal 80% (Cob / Ancho) | Conformal 90% (Cob / Ancho) | DMA8 Test (RMSE / F1) | Cota PP No Operativa ($B_{pp}$) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h = 1$ h** | **M7 Peak Booster** | `deep2__deep_imput2_resid_refit` (4.764) | **4.6665** | **4.7157 ppb** | **−0.0487 ppb** | **−0.2155 ppb** | **0.9498** | 9.34 / −3.87 | **79.21%** / 9.56 ppb | **89.26%** / 13.45 ppb | 2.10 ppb / **0.9217** | 5.003 ppb *(Batida por −0.287)* |
| **$h = 4$ h** | **M8 Grand Blend** | `deep2__deep_imput2_resid_refit` (8.455) | **8.1341** | **8.3688 ppb** | **−0.0860 ppb** | **−0.3407 ppb** | **0.8420** | 19.64 / −14.32 | **79.59%** / 17.82 ppb | **89.51%** / 23.89 ppb | 6.97 ppb / **0.7168** | 8.665 ppb *(Batida por −0.296)* |
| **$h = 8$ h** | **M7 Peak Booster** | `deep2__deep_imput2_resid_refit` (9.413) | **9.0551** | **9.2275 ppb** | **−0.1858 ppb** | **−0.6435 ppb** | **0.8080** | 21.57 / −16.21 | **79.47%** / 19.99 ppb | **89.54%** / 26.64 ppb | 8.75 ppb / **0.6283** | 9.481 ppb *(Batida por −0.254)* |
| **$h = 12$ h**| **M8 Grand Blend** | `deep2__deep_imput2_resid_refit` (9.604) | **9.2504** | **9.3539 ppb** | **−0.2498 ppb** | **−0.6040 ppb** | **0.8028** | 21.60 / −16.12 | **80.50%** / 20.89 ppb | **89.94%** / 27.64 ppb | 8.89 ppb / **0.6221** | 9.543 ppb *(Batida por −0.189)* |
| **$h = 24$ h**| **M7 Peak Booster** | `refit__refit_nwp_best` (9.864) | **9.4421** | **9.4400 ppb** | **−0.4241 ppb** | **−0.4439 ppb** | **0.7993** | 20.70 / −14.72 | **80.40%** / 21.26 ppb | **89.84%** / 28.14 ppb | 8.80 ppb / **0.6420** | 9.594 ppb *(¡Batida por −0.154!)* |
| **$h = 48$ h**| **M3 NNLS Ridge** | `imput2__imput2_nwp_resid_refit` (10.781) | **10.3768** | **10.3255 ppb** | **−0.4552 ppb** | **−0.5719 ppb** | **0.7602** | 23.09 / −17.97 | **81.44%** / 23.86 ppb | **90.38%** / 31.26 ppb | 10.32 ppb / **0.5391** | 10.112 ppb |

---

## 2. Evolución Progresiva a través de las 5 Pasadas

| Horizonte ($h$) | Ensamble Noche 1 | Pasada 1 (209 mods) | Pasada 2 (215 mods) | Pasada 3 (233 mods) | Pasada 4 (257 mods) | **Pasada 5 Final (268 mods)** | **Ganancia Neta vs Noche 1** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h = 1$ h** | 4.931 ppb | 4.819 ppb | 4.818 ppb | 4.779 ppb | 4.771 ppb | **4.716 ppb** | **−0.216 ppb** |
| **$h = 4$ h** | 8.710 ppb | 8.520 ppb | 8.516 ppb | 8.459 ppb | 8.446 ppb | **8.369 ppb** | **−0.341 ppb** |
| **$h = 8$ h** | 9.871 ppb | 9.297 ppb | 9.258 ppb | 9.279 ppb | 9.255 ppb | **9.228 ppb** | **−0.644 ppb** |
| **$h = 12$ h**| 9.958 ppb | 9.480 ppb | 9.490 ppb | 9.417 ppb | 9.398 ppb | **9.354 ppb** | **−0.604 ppb** |
| **$h = 24$ h**| 9.884 ppb | 9.642 ppb | 9.636 ppb | 9.506 ppb | 9.458 ppb | **9.440 ppb** | **−0.444 ppb** |
| **$h = 48$ h**| 10.897 ppb | 10.590 ppb | 10.558 ppb | 10.338 ppb | 10.310 ppb | **10.325 ppb** (`M7`: **10.319**) | **−0.572 ppb** |

---

## 3. Comparación Exhaustiva de Algoritmos (M0 a M8) en Pasada 5

### Tabla Comparativa: Val OOF RMSE / Test 2025 RMSE (ppb)

| $h$ | M0: Mejor Indiv | M1: Top-$K$ Mean | M2: Caruana | M3: NNLS Ridge | M4: NNLS Régimen | M5: Ridge Context | M6: LGBM Stacking | M6: Blend Híbrido | M7: Peak Booster | **M8: Grand Blend OOF** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1h** | 4.715 / 4.764 | 4.672 / 4.726 | 4.668 / 4.710 | 4.669 / 4.710 | 4.672 / 4.713 | 4.685 / **4.702** | 4.736 / 4.752 | 4.669 / 4.710 | **4.667 / 4.716** $\leftarrow$ | 4.670 / 4.714 |
| **4h** | 8.335 / 8.455 | 8.175 / 8.379 | 8.150 / 8.348 | 8.155 / 8.406 | 8.148 / 8.415 | 8.232 / **8.308** | 8.456 / 8.416 | 8.148 / 8.415 | 8.136 / 8.381 | **8.134 / 8.369** $\leftarrow$ |
| **8h** | 9.362 / 9.413 | 9.111 / 9.318 | 9.078 / 9.209 | 9.072 / 9.267 | 9.066 / 9.268 | 9.265 / **9.149** | 9.658 / 9.329 | 9.066 / 9.268 | **9.055 / 9.228** $\leftarrow$ | 9.057 / 9.211 |
| **12h**| 9.509 / 9.604 | 9.378 / 9.554 | 9.278 / **9.336** | 9.274 / 9.401 | 9.273 / 9.430 | 9.445 / 9.376 | 9.714 / 9.498 | 9.273 / 9.430 | 9.253 / 9.387 | **9.250 / 9.354** $\leftarrow$ |
| **24h**| 9.842 / 9.864 | 9.685 / 9.717 | 9.479 / **9.399** | 9.463 / 9.462 | 9.456 / 9.441 | 9.949 / 9.503 | 10.197 / 9.659 | 9.456 / 9.441 | **9.442 / 9.440** $\leftarrow$ | 9.449 / 9.440 |
| **48h**| 10.735 / 10.781| 10.524 / 10.530| 10.415 / 10.328| **10.377 / 10.326** $\leftarrow$| 10.380 / 10.320| 11.120 / 10.535| 11.843 / 10.689| 10.377 / 10.326| 10.378 / **10.319**| 10.378 / **10.319** |
