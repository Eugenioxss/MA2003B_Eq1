"""
experiments/planA_quickwins/run_ablations.py
============================================
Ejecuta las variantes de ablación V0 a V4 para todos los horizontes [1, 4, 8, 12, 24, 48]:
  - V0: Baseline v2 (reproducción exacta Script 7)
  - V1: + lag_0 contemporáneo y rollings/diffs a t
  - V2: + Estacion como categorical_feature
  - V3: + variables no usadas (NO, NO2, CO, RH, PRS, RAINF, SO2, Ox)
  - V4: + features espaciales de la red completa (15 estaciones, leave-one-out, nearest, upwind)

Usa num_threads=4, registra con log_result() y guarda predicciones y métricas.
"""
import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.abspath('experiments'))
from common import load_panel, split, evaluate_test, log_result, LGB_PARAMS_V2, HORIZONS
from planA_quickwins.features_planA import build_planA_features

OUT_DIR = os.path.join('experiments', 'planA_quickwins')
os.makedirs(OUT_DIR, exist_ok=True)

VARIANTS = [
    ('V0', 'V0_baseline_v2', False),
    ('V1', 'V1_lag0', False),
    ('V2', 'V2_estacion_cat', True),
    ('V3', 'V3_unused_vars_ox', True),
    ('V4', 'V4_spatial', True),
]

def main():
    print("=" * 70)
    print("  EXPERIMENTO PLAN A: QUICK WINS - ABLACIONES INCREMENTALES (V0 - V4)")
    print("  LightGBM num_threads=4 | Horizontes:", HORIZONS)
    print("=" * 70)

    t_start = time.time()
    panel = load_panel()
    print(f"Panel cargado con éxito ({panel.shape[0]} filas, {panel.shape[1]} columnas).\n")

    all_metrics = []

    for v_code, v_name, use_cat in VARIANTS:
        print(f"\n=======================================================")
        print(f"  INICIANDO VARIANTE: {v_name} (code={v_code})")
        print(f"=======================================================")
        t_v = time.time()
        
        # 1. Construir features
        df_feat, features = build_planA_features(panel, variant=v_code)
        print(f"Features construidos: {len(features)} variables en {time.time()-t_v:.2f}s")
        
        # 2. Split temporal
        train_df, val_df, test_df = split(df_feat)
        print(f"Filas por split -> Train (<=2023): {len(train_df):,}, Val (2024): {len(val_df):,}, Test (2025): {len(test_df):,}")

        # Configurar parámetros
        params = dict(LGB_PARAMS_V2)
        params['num_threads'] = 4
        cat_cols = ['Estacion'] if use_cat else None

        for h in HORIZONS:
            t_h = time.time()
            target_col = f'Target_{h}h'
            
            # Filtrar no nulos en target
            tr_h = train_df.dropna(subset=[target_col]).reset_index(drop=True)
            va_h = val_df.dropna(subset=[target_col]).reset_index(drop=True)
            ts_h = test_df.dropna(subset=[target_col]).reset_index(drop=True)

            X_tr, y_tr = tr_h[features], tr_h[target_col]
            X_va, y_va = va_h[features], va_h[target_col]
            X_ts, y_ts = ts_h[features], ts_h[target_col]

            # Crear LightGBM Datasets
            lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
            lgb_val = lgb.Dataset(X_va, y_va, reference=lgb_train, categorical_feature=cat_cols)

            # Entrenar con early stopping 80
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=[lgb_val],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=80, verbose=False),
                    lgb.log_evaluation(period=0),
                ]
            )

            # Predecir en test
            y_pred = model.predict(X_ts, num_iteration=model.best_iteration)
            
            # Evaluar con protocolo estándar
            m = evaluate_test(ts_h, y_pred, h)
            m['best_iteration'] = int(model.best_iteration)
            m['train_time_s'] = round(time.time() - t_h, 2)
            m['n_features'] = len(features)
            
            # Registrar en registry global
            notes = f"iters={model.best_iteration}, n_feat={len(features)}, time={m['train_time_s']}s"
            log_result('planA_quickwins', v_name, h, m, notes=notes)

            # Guardar predicciones en test
            pred_df = ts_h[['Date', 'Estacion', target_col]].copy()
            pred_df['y_pred'] = y_pred
            pred_file = os.path.join(OUT_DIR, f"preds_{v_name}_{h}h.parquet")
            pred_df.to_parquet(pred_file, index=False)

            # Guardar modelo
            model_file = os.path.join(OUT_DIR, f"model_{v_name}_{h}h.txt")
            model.save_model(model_file)

            # Guardar en lista de resultados
            res_entry = {
                'variant': v_name,
                'h': h,
                'RMSE': round(m['RMSE'], 4),
                'MAE': round(m['MAE'], 4),
                'R2': round(m['R2'], 4),
                'RMSE_peaks': round(m['RMSE_peaks'], 4),
                'bias_peaks': round(m['bias_peaks'], 4),
                'RMSE_day': round(m.get('RMSE_day', 0.0), 4),
                'skill_vs_clim': round(m.get('skill_vs_clim', 0.0), 4),
                'best_iter': model.best_iteration,
                'train_time_s': m['train_time_s'],
                'n_features': len(features)
            }
            all_metrics.append(res_entry)

            print(f"  [{v_name}] h={h:2d} | RMSE: {m['RMSE']:.4f} | R2: {m['R2']:.4f} | "
                  f"RMSE_pk: {m['RMSE_peaks']:.4f} | Bias_pk: {m['bias_peaks']:.2f} | "
                  f"Skill: {m['skill_vs_clim']:.4f} | Iter: {model.best_iteration} ({m['train_time_s']:.1f}s)")

        # Guardar checkpoint intermedio de resultados
        df_res = pd.DataFrame(all_metrics)
        df_res.to_csv(os.path.join(OUT_DIR, 'ablation_summary.csv'), index=False)
        print(f"Variante {v_name} finalizada en {time.time()-t_v:.1f}s")

    print("\n" + "=" * 70)
    print(f"  TODAS LAS ABLACIONES (V0 - V4) FINALIZADAS EN {time.time()-t_start:.1f}s")
    print("=" * 70)
    print("\nResumen de Métricas:")
    print(pd.DataFrame(all_metrics).to_string(index=False))

if __name__ == '__main__':
    main()
