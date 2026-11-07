# Corrida "noche2" — Contrato común (v4)

Raíz: `D:\Codingggg\Notas\Multivariados\MA2003B_Eq1` · Python: `.venv\Scripts\python.exe` (ejecutar SIEMPRE desde la raíz).
Objetivo: mejorar el pronóstico HONESTO de O3(t+h), h ∈ {1,4,8,12,24,48}, 13 estaciones SIMA, sobre el mejor estado actual.

## Siguen vigentes
`experiments/README_experimentos.md` y `experiments/noche/README_noche.md` (honestidad meteo, formato de preds, split, `common.py`).
Lee también: `experiments/noche/v3_base/REPORT.md`, `experiments/noche/v3_base/cache/FEATURES.md`, `experiments/noche/ensemble/REPORT.md`.

## Reglas clave
1. Split: train 2020–2023, val 2024, test 2025. **Nunca seleccionar nada con 2025.**
   Excepción permitida: para el pronóstico final de 2025 se puede **reentrenar con train+val (≤2024)** usando
   hiperparámetros/iteraciones elegidos SOLO con val (p.ej. n_iter = best_iter × 1.1). Las filas `val` de tus preds deben
   ser entonces **out-of-fold** (p.ej. cross-fitting por trimestre de 2024: entrenar con ≤2023 + los otros 3 trimestres de 2024).
   Documenta cómo se generó cada fila val.
2. Variables `*_pp` / `B_pp` / `ensemble_pp` / `tide_pp` NO son honestas → no se usan como resultado ni en el ensamble honesto.
3. Escribe SOLO en `experiments/noche2/<tu_exp>/`. Preds en `experiments/noche2/<tu_exp>/preds/<variante>_h<h>.parquet`
   con columnas `Date, Estacion, h, split ('val'|'test'), y_true, y_pred` (opcional q10,q90), todas las filas val 2024 y
   test 2025 con target observado (test = conjunto estándar de `evaluate_test`; h=1 → 108,512 filas).
4. `experiments/common.py` no se modifica. Puedes importar (sin modificar) código de `experiments/noche/*` y `experiments/plan*`.
   `log_result()` con experiment=`noche2_<tu_exp>` (OJO: `results_registry.csv` tiene filas con distinto nº de columnas; no lo reescribas).
5. Recursos: 16 hilos, 31 GB RAM, RTX 3060 6 GB. Usa EXACTAMENTE los hilos de tu tarea. Solo el agente indicado usa GPU.
   Vigila RAM: los caches `v3_base/cache/features_h*.parquet` pesan ~190 MB c/u en disco (varios GB en RAM); carga columnas necesarias, usa float32.
6. Scripts largos en segundo plano con log en tu carpeta y checkpoints. Si algo tarda > 90 min, reduce y documenta.
7. Entregable: `experiments/noche2/<tu_exp>/REPORT.md` (español): qué se probó, tabla por h (val RMSE, test RMSE, R², RMSE_peaks,
   bias_peaks) vs referencias, hallazgos, problemas y recomendación. Mensaje final breve al orquestador con la tabla.

## Estado actual (honesto, test 2025 RMSE) — a batir
| h | mejor individual honesto (val / test) | ensamble honesto actual `noche/ensemble` (val OOF / test) | cota PP no operativa (B_pp test) |
|---|---|---|---|
| 1 | D_honest_noise 4.937 / 5.005 | 4.873 / 4.931 | 5.00 |
| 4 | D_honest_noise 8.636 / 9.004 (E_honest_w 8.991) | 8.497 / 8.710 | 8.67 |
| 8 | D_honest_noise 9.623 / 10.054 (E 9.977) | 9.498 / 9.871 | 9.48 |
| 12 | E_honest_w 9.816 / 10.135 | 9.653 / 9.958 | 9.54 |
| 24 | D_honest_noise 10.026 / 10.238 | 9.785 / 9.884 | 9.59 |
| 48 | D_honest_noise 11.116 / 11.234 | 10.931 / 10.897 | 10.11 |
Inventario completo de preds existentes: `experiments/noche2/inventory.csv` (script `experiments/noche2/inventory.py`).
