# Evaluación de Imputación por Bloques — SAITS+CSDI (MIMA-v6)

## Objetivo

Determinar **empíricamente** si el modelo híbrido SAITS+CSDI del repositorio `MA2003B-Equipo-6-MIMA-v6` reconstruye de forma confiable apagones largos de sensores, o si solo funciona para huecos puntuales dispersos (MCAR) como advierte su propia documentación.

## Protocolo

1. Se construyó el CSV de entrada uniendo los 15 `*_clean.parquet` de `MA2003B_Eq1/data/processed/variables/` (749,169 filas × 17 columnas, 15 estaciones, 2020–2025).
2. Se calculó la cobertura simultánea de O3 y PM2.5 por estación/mes para 2022–2024.
3. Se seleccionó **estación SE3, año 2022** — la mejor cobertura simultánea para ambos escenarios (Esc. A: 96.2%, Esc. B: 96.6%).
4. Se enmascararon dos bloques como NaN (sin alterar el resto del dataset):
   - **Escenario A (moderado):** junio 2022 completo (~720h) de O3 y PM2.5.
   - **Escenario B (peor caso real):** 1 jul – 25 ago 2022 (~1,333h consecutivas / 55 días).
5. Se entrenó el modelo con hiperparámetros por defecto: 50 épocas, `seq_len=168`, `hidden_size=64`, `lr=1e-3`, `mask_ratio=0.2`, `train_stride=24`, `batch_size=32`, sobre RTX 4080.
6. Se imputó el dataset enmascarado y se extrajeron las reconstrucciones.
7. Se compararon contra los valores reales guardados (ground truth).

## Resultados

| Escenario | Variable | N | RMSE | R² | RMSE picos (top 5%) | N picos |
|-----------|----------|--:|-----:|---:|-------------------:|--------:|
| **A** (1 mes) | O3 | 715 | 10.97 | **0.358** | 34.66 | 38 |
| **A** (1 mes) | PM2.5 | 693 | 10.25 | **−0.401** | 7.14 | 35 |
| **B** (55 días) | O3 | 1,328 | 10.06 | **0.375** | 28.00 | 67 |
| **B** (55 días) | PM2.5 | 1,288 | 9.81 | **−0.915** | 6.58 | 70 |

> **R² negativo en PM2.5 significa que el modelo es peor que predecir la media constante.**
> La imputación falla incluso en el escenario "moderado" de 1 mes.

### Sesgo sistemático en PM2.5

| Escenario | Media real | Media imputada | Sesgo |
|-----------|----------:|---------------:|------:|
| A (1 mes) | 13.86 | 21.87 | **+57.8%** |
| B (55 días) | 12.62 | 20.92 | **+65.8%** |

El modelo sobreestima PM2.5 consistentemente en ~8 µg/m³.

## Gráficas Overlay

Las gráficas se encuentran en `eval_bloques/resultados/`:
- `overlay_A_O3.png` — O3 Escenario A
- `overlay_A_PM25.png` — PM2.5 Escenario A
- `overlay_B_O3.png` — O3 Escenario B
- `overlay_B_PM25.png` — PM2.5 Escenario B

### Observaciones visuales

- **O3**: El modelo captura vagamente el ciclo diurno pero aplana completamente los picos. Cuando el ozono real sube a 90–115 ppb, la reconstrucción se queda en ~30–40 ppb.
- **PM2.5**: La reconstrucción flota sistemáticamente por encima de los valores reales. No hay tracking de la señal — es ruido con sesgo positivo. R² = −0.91 en el peor caso.

## Diagnóstico: ¿Por qué falla?

1. **El modelo entrena con enmascaramiento MCAR** (aleatorio puntual al 20%). Aprende a interpolar huecos dispersos usando contexto circundante visible en la misma ventana de 168h.
2. **En un bloque de NaN de 720–1333h**, no hay contexto circundante O3/PM2.5 dentro de la ventana. El modelo solo puede usar las otras 13 variables como proxy — insuficiente.
3. **El denoiser es de 1 sola etapa** (no CSDI completo). No tiene capacidad generativa para producir trayectorias realistas condicionadas.
4. **Las ventanas de 168h con stride 84** promedian predicciones solapadas, suavizando aún más cualquier señal residual.

## Recomendación

**La imputación NO es confiable para apagones de bloques continuos — ni de 55 días (R² O3=0.37, PM2.5=−0.91), ni siquiera de 1 mes (R² O3=0.36, PM2.5=−0.40). El modelo solo funciona para huecos puntuales dispersos MCAR. Para el pipeline antes de la LSTM, usar gradient boosting directo sobre datos crudos o interpolación acotada, no este imputador.**

---

*Evaluación ejecutada el 2026-09-11 sobre estación SE3 (año 2022), con modelo entrenado 50 épocas en RTX 4080, hiperparámetros por defecto del repositorio.*
