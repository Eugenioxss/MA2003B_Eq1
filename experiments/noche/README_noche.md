# Corrida nocturna "noche" — Contrato común (v3)

Raíz del proyecto: `D:\Codingggg\Notas\Multivariados\MA2003B_Eq1`
Python: `.venv\Scripts\python.exe` (ejecutar SIEMPRE desde la raíz del proyecto).
Objetivo: encontrar la mejor forma HONESTA de predecir O3(t+h), h ∈ {1,4,8,12,24,48}, en 13 estaciones SIMA.

## Reglas (además de `experiments/README_experimentos.md`, que sigue vigente)
1. Usa `experiments/common.py` (`load_panel`, `split`, `evaluate_test`, `log_result`, ...). No lo modifiques.
   Puedes importar (sin modificar) `experiments/planA_quickwins/features_planA.py` y `experiments/planC_meteo/meteo_features.py`.
2. Escribe SOLO en tu carpeta `experiments/noche/<tu_experimento>/`. `log_result()` con experiment=`noche_<tu_experimento>`.
3. Split: train 2020–2023, val 2024, **test 2025 solo para evaluación final**. Nada de seleccionar modelos/hiperparámetros con 2025.
4. **Momento de emisión t**: se permiten datos SIMA con timestamp ≤ t (lag 0 permitido, como Plan A `V1_lag0`). Target_h = O3(t+h).
5. **Meteorología futura (Open-Meteo, `experiments/planC_meteo/data/`)** — regla de honestidad:
   - Columnas sin sufijo (`temperature_2m`, ...) = corrida "seamless"/día 0 (lead 0–23 h). Para un valor válido en t+h con h ≥ 1 **NO es honesto** usarlas como pronóstico (la corrida pudo emitirse después de t). Úsalas solo para entrenar en modo *perfect-prog* o como "cota superior".
   - `*_previous_day1` (lead 24–47 h) es honesto para h ≤ 24. `*_previous_day2` (lead 48–71 h) es honesto para h ≤ 48.
   - OJO: `*_previous_day1/2` solo existen completos en 2024–2025 (temperatura desde 2021). BLH solo desde mediados de 2024.
   - Protocolo honesto recomendado (**PP-train / NWP-test**): entrenar con seamless/reanálisis en t+h (2020–2023), y en val/test sustituir por `previous_day1` (h ≤ 24) o `previous_day2` (h = 48). Opcional: inyectar en train ruido con la distribución del error (previous_dayK − seamless) estimada SOLO en 2024.
   - Reportar siempre por separado: `*_pp` (cota superior, no operativa) vs `*_honest` (operativa). Solo `honest` cuenta como resultado final.
   - Meteo en el instante t (≤ t) desde seamless/reanálisis es aceptable.
6. **Formato de predicciones (obligatorio, para el ensamble final)**: por cada variante y h guarda
   `experiments/noche/<exp>/preds/<variante>_h<h>.parquet` con columnas
   `Date` (emisión t), `Estacion`, `h`, `split` ('val' = 2024, 'test' = 2025), `y_true`, `y_pred`
   (y opcionalmente `q10`, `q90`). Incluir TODAS las filas val 2024 y test 2025 con Target observado
   (test debe coincidir con el conjunto estándar de `evaluate_test`, p.ej. 108,512 filas en h=1). Si no hay predicción para una fila, deja NaN.
7. Recursos compartidos (16 hilos, 31 GB RAM, RTX 3060 6 GB): usa EXACTAMENTE los hilos indicados en tu tarea. Solo el agente "deep" usa GPU. No instales paquetes.
8. Corre scripts largos en segundo plano con logs en tu carpeta; guarda checkpoints/resultados parciales para no perder trabajo. Si algo tarda > 60 min, reduce y documenta.
9. Entregable: `experiments/noche/<exp>/REPORT.md` (español): qué se probó, tabla por horizonte (RMSE, R², RMSE_peaks, bias_peaks, skill) vs referencias, hallazgos, problemas y recomendación. Mensaje final breve al orquestador con la tabla.

## Referencias (test 2025, RMSE)
| h | v2 | Plan A V4_spatial (lag0+espacial) | Plan C meteo (OJO: h≤12 usó seamless → NO honesto) |
|---|---|---|---|
| 1 | 8.05 | 5.09 | 7.55 |
| 4 | 10.91 | 9.69 | 9.38 |
| 8 | 11.97 | 11.28 | 9.82 |
| 12 | 12.21 | 11.73 | 9.80 |
| 24 | 12.76 | 12.39 | 10.79 |
| 48 | 13.95 | 13.76 | 11.58 |
Climatología RMSE ≈ 14.88. Plan B: pesos `w = 1 + (y/p90)^2` reducen sesgo en picos con costo < 3 % RMSE.
