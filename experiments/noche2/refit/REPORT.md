# Experimento `noche2_refit`: Reentrenamiento con 2024 y Meteo Honesta Real

**Autor:** Científico de Datos Senior (`noche2_refit`)  
**Fecha:** Noviembre 2026  
**Carpeta de trabajo:** `experiments/noche2/refit/`  
**Recursos:** LightGBM CPU pura, `num_threads=6` (Fase 1) / `num_threads=5` (Fase 2), sin GPU.  
**Registro Oficial:** Registrado en `experiments/results_registry.csv` bajo `noche2_refit`.  
**Predicciones:** Parquets estándar en `experiments/noche2/refit/preds/` con columnas `Date, Estacion, h, split ('val'|'test'), y_true, y_pred`.

---

## 1. Contexto, Motivación e Hipótesis

El modelo base v3 honesto (`D_honest_noise`, `E_honest_w`) se entrenó exclusivamente con el periodo 2020–2023, donde la meteorología futura provenía de reanálisis/seamless (*Perfect-Prog*) con inyección de ruido sintético. En contraste, el año 2024 cuenta con pronósticos numéricos del tiempo (**NWP**) reales emitidos operativamente (`*_previous_day1/2`).

### Hipótesis Científica
1. **Distribución real del error NWP**: Entrenar con 2024 enseña al modelo la estructura estocástica y sistemática real del error del NWP (sesgos diurnos, dispersión, correlación cruzada entre variables meteorológicas).
2. **Proximidad temporal**: Los datos recientes de 2024 mitigan el cambio en la distribución fotoquímica hacia el conjunto de prueba de 2025.
3. **Cross-fitting honesto por trimestres (4 Folds en 2024)**: Para evitar data leakage y obtener métricas de validación out-of-fold (OOF) para todo 2024 (108,548 filas), se entrena con 2020–2023 + los otros 3 trimestres de 2024 para predecir el trimestre restante.
4. **Calibración de ruido libre de contaminación**: El ruido sintético inyectado a 2020–2023 se calibra estrictamente sobre los 3 trimestres de entrenamiento de 2024 (excluyendo el trimestre evaluado).

---

## 2. Fase 1: Exploración de Variantes y Cross-Fitting en 2024

### 2.1 Comparación de Variantes en Horizontes Piloto ($h=24$ y $h=4$)

En la exploración inicial sobre $h=24$ y $h=4$ se evaluaron las siguientes formulaciones:
- **`refit_base`**: 2020–2023 (ruido calibrado sin trimestre Q) + 2024 (otros 3 trimestres, peso 1.0).
- **`refit_isnwp`**: `refit_base` + variable indicadora binaria `is_nwp` (0 para 2020–2023, 1 para 2024 y 2025).
- **`refit_w2`**: `refit_base` + peso extra $\times 2$ para filas de 2024.
- **`refit_w4`**: `refit_base` + peso extra $\times 4$ para filas de 2024.
- **`refit_nonoise`**: 2020–2023 sin ruido sintético + 2024 real.
- **`refit_peak_w`**: `refit_base` + ponderación de picos $w = 1 + (y/p_{90})^2$.
- **`refit_leak_test`**: `refit_base` pero calibrando ruido con todo 2024 (test de fuga).

#### Resultados en $h=24$ (Referencia Base D_honest_noise: Val RMSE = 10.0262):
| Variante | Val OOF RMSE | $\Delta$ vs D | % Mejora | Val Pk RMSE | Hallazgo Clave |
|---|:---:|:---:|:---:|:---:|---|
| **`refit_isnwp`** | **9.8949** | **-0.1313** | **-1.31%** | 21.25 | **Ganador indiscutible.** Diferenciar PP de NWP real es óptimo. |
| `refit_w2` | 9.9270 | -0.0992 | -0.99% | 21.81 | Ponderar 2024 ayuda pero menos que `is_nwp`. |
| `refit_leak_test` | 9.9252 | -0.1010 | -1.01% | 21.88 | Calibrar con todo 2024 da sesgo optimista de 0.022 ppb. |
| `refit_isnwp_w2` | 9.9360 | -0.0902 | -0.90% | 21.41 | Combinar ambos sobrepondera varianza de un solo año. |
| `refit_base` | 9.9473 | -0.0789 | -0.79% | 21.81 | Incorporar 2024 crudo ya supera a Base D por 0.08 ppb. |
| `refit_peak_w` | 9.9507 | -0.0755 | -0.75% | **20.52** | Excelente en picos (Pk RMSE 20.52 vs 21.81). |
| `refit_nonoise` | 9.9721 | -0.0542 | -0.54% | 23.44 | **Eliminar ruido degrada picos** (+1.63 ppb Pk RMSE). |

#### Resultados en $h=4$ (Referencia Base D_honest_noise: Val RMSE = 8.6356):
| Variante | Val OOF RMSE | $\Delta$ vs D | % Mejora | Val Pk RMSE |
|---|:---:|:---:|:---:|:---:|
| **`refit_w2`** | **8.5034** | **-0.1322** | **-1.53%** | 18.36 |
| `refit_isnwp` | 8.5091 | -0.1265 | -1.46% | 18.02 |
| `refit_base` | 8.5095 | -0.1261 | -1.46% | 18.38 |
| `refit_peak_w` | 8.5359 | -0.0997 | -1.15% | **17.47** |

---

## 3. Fase 1: Desempeño Consolidado en Todos los Horizontes ($h \in \{1, 4, 8, 12, 24, 48\}$)

Siguiendo el contrato estricto, la selección de la mejor arquitectura se realizó **exclusivamente por el menor RMSE en Val OOF 2024**:
- $h=1$: `refit_base`
- $h=4$: `refit_w2`
- $h=8$: `refit_w2`
- $h=12$: `refit_isnwp`
- $h=24$: `refit_isnwp`
- $h=48$: `refit_isnwp`

El modelo final se reentrenó sobre **2020–2024 completo** con bagging de 3 semillas ($n_{\text{iter}} = \text{round}(\text{mediana}(\text{best\_iters}) \times 1.1)$) y se evaluó sobre Test 2025.

### Tabla Consolidada Fase 1 vs. Referencias
| $h$ | Variante Ganadora | Val OOF RMSE | Base D Val | $\Delta$ Val vs D | Test RMSE (Refit) | Base D Test | $\Delta$ Test vs D | Ensamble Noche 1 Test | $\Delta$ vs Ensamble | Test $R^2$ | Test Pk RMSE | Test Pk Bias | Ganancia Bagging |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `refit_base` | **4.8663** | 4.9370 | **-0.0707** | **4.8663** | 5.0048 | **-0.1384** | 4.9312 | **-0.0649** | 0.9466 | 9.85 | -4.34 | +0.0237 |
| **4** | `refit_w2` | **8.5034** | 8.6356 | **-0.1322** | **8.7481** | 9.0039 | **-0.2558** | 8.7095 | +0.0385 | 0.8274 | 20.78 | -15.59 | +0.0306 |
| **8** | `refit_w2` | **9.4568** | 9.6229 | **-0.1661** | **9.7368** | 10.0536 | **-0.3168** | 9.8710 | **-0.1342** | 0.7862 | 23.82 | -18.80 | +0.0331 |
| **12** | `refit_isnwp` | **9.6596** | 9.8358 | **-0.1762** | **9.8774** | 10.1832 | **-0.3059** | 9.9579 | **-0.0805** | 0.7801 | 23.23 | -17.75 | +0.0408 |
| **24** | `refit_isnwp` | **9.8949** | 10.0262 | **-0.1313** | **9.9958** | 10.2382 | **-0.2425** | 9.8839 | +0.1119 | 0.7750 | 23.02 | -17.63 | +0.0339 |
| **48** | `refit_isnwp` | **10.9553** | 11.1158 | **-0.1605** | **11.0130** | 11.2337 | **-0.2207** | 10.8974 | +0.1157 | 0.7272 | 25.62 | -20.92 | +0.0143 |

> [!IMPORTANT]
> **Hito alcanzado en Fase 1**:
> - En **$h=1$, $h=8$ y $h=12$**, nuestro modelo individual con refit **supera directamente al Ensamble Multimodelo completo de Noche 1** (p.ej. en $h=1$: 4.866 vs 4.931; en $h=8$: 9.737 vs 9.871).
> - Frente al mejor modelo individual previo (`D_honest_noise`), la ganancia neta en Test 2025 es masiva: **-0.22 a -0.32 ppb** en horizontes medios y largos.
> - El bagging de 3 semillas aporta una reducción adicional y sistemática de **~0.02 a 0.04 ppb** en todos los horizontes.

---

## 4. Fase 2: Integración con Características Meteorológicas Multi-Modelo + MOS (`nwp`)

*(En ejecución)*: Acoplamiento de la matriz `v3_base` con las 12 características generadas por el agente meteorológico en `experiments/noche2/nwp/features_nwp_h<h>.parquet`:
1. Consenso multi-modelo ECMWF + GFS + ICON (`nwp_temp_mean/std`, `nwp_rad_mean/std`, `nwp_ws_mean/std`).
2. Diagnósticos de convección y mezcla superficial (`nwp_cape`, `nwp_wind_gust`, `nwp_vpd_mean`).
3. Calibración empírica causal MOS (`nwp_mos_temp`, `nwp_mos_ws`, `nwp_mos_photothermal`).
4. Evaluación de `refit_nwp_best`, `refit_nwp_peak_w` y reformulación residual de persistencia (`resid_persist`) para $h=1$ y $h=48$.
