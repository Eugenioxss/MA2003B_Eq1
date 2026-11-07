# Experimentos "Siguiente Nivel" — Contrato común

Raíz del proyecto: `D:\Codingggg\Notas\Multivariados\MA2003B_Eq1`
Python: `.venv\Scripts\python.exe` (ejecutar SIEMPRE desde la raíz del proyecto).

## Reglas
1. Usa `experiments/common.py` (`load_panel`, `build_v2_features`, `split`, `evaluate_test`, `log_result`, `LGB_PARAMS_V2`, `STATION_COORDS`). No lo modifiques; si necesitas algo extra, cópialo en tu propio módulo.
2. Escribe SOLO dentro de tu carpeta `experiments/<tu_experimento>/` (scripts, figuras, CSVs, `REPORT.md`). No modifiques archivos existentes del proyecto. Excepción: `log_result()` agrega filas a `experiments/results_registry.csv`.
3. Split: train 2020–2023, val 2024 (early stopping/tuning), **test 2025 solo para la evaluación final**. Nunca uses 2025 para seleccionar modelos/hiperparámetros.
4. Sin fuga de información: todo feature en la fila t debe usar datos con timestamp ≤ t (o pronósticos meteorológicos que hubieran estado disponibles en t). Target_h = O3(t+h).
5. Evalúa con `evaluate_test(test_df, y_pred, h)` sobre el conjunto estándar (13 estaciones, 2025, Target observado; para h=1 son 108,512 filas). Registra con `log_result(experimento, variante, h, metricas, notas)`.
6. Recursos compartidos (hay 6 agentes en paralelo, 16 hilos, 31 GB RAM, 1 GPU RTX 3060 6 GB):
   - LightGBM/XGBoost: `num_threads`/`n_jobs` = el valor indicado en tu tarea.
   - Solo el agente de Deep Learning usa la GPU.
   - No instales paquetes con pip. Ya están: lightgbm, xgboost, sklearn, statsmodels, torch(cu121), shap, optuna, pygam, mapie, miceforest, transformers, accelerate, chronos-forecasting, requests, pyarrow. Si falta algo crítico, avisa al orquestador.
7. Corre los scripts largos en segundo plano y con logs; si un experimento tarda demasiado, reduce (menos horizontes, submuestreo de train, menos trials) y documéntalo.
8. Resultados de referencia (v2, Script 7, test 2025) — RMSE / R²:
   h1 8.05/0.854 · h4 10.91/0.732 · h8 11.97/0.677 · h12 12.21/0.664 · h24 12.76/0.633 · h48 13.95/0.562.
   Climatología (estación×mes×hora): RMSE ≈ 14.88, R² ≈ 0.50. Sesgo en picos (top 5 %) v2: −14 (h1) a −34 ppb (h48).
   Nota: la fila t de v2 usa datos hasta t−1 (features con shift≥1).
9. Entregable final: `experiments/<tu_experimento>/REPORT.md` en español con: qué se probó, tabla de métricas por horizonte vs v2, hallazgos, problemas, y recomendación. Y un mensaje final breve al orquestador con la tabla resumen.
