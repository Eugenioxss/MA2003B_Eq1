"""
experiments/noche2/refit/run_all.py
===================================
Orquestador autónomo para noche2_refit:
1. Completa la exploración por CV de los horizontes restantes (h=8, 12, 1, 48)
   para las variantes clave (refit_base, refit_isnwp, refit_w2, refit_peak_w).
2. Selecciona la mejor variante por h según SOLO val OOF RMSE.
3. Ejecuta el refit final con bagging de 3 semillas (42, 43, 44) sobre 2020-2024.
4. Genera predicciones val (OOF) + test (2025), evalúa métricas completas,
   guarda parquets oficiales y registra en results_registry.csv.
"""
import os
import sys
import time
import json
import gc
import numpy as np
import pandas as pd
from functools import partial

print = partial(print, flush=True)

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
EXP_DIR = os.path.join(ROOT, 'experiments')
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, BASE_DIR)

from common import HORIZONS, evaluate_test, log_result, metrics, SEED
from train_refit import (
    load_cache_for_h, cross_fit_2024, refit_and_evaluate_test,
    VARIANTS, OOF_DIR, PREDS_DIR
)


def run_pipeline():
    start_total = time.time()
    cv_results_path = os.path.join(BASE_DIR, 'cv_results.csv')
    inv_path = os.path.join(ROOT, 'experiments', 'noche2', 'inventory.csv')
    inv = pd.read_csv(inv_path)

    # 1. Exploración CV de horizontes restantes
    remaining_horizons = [8, 12, 1, 48]
    cv_variants = ['refit_base', 'refit_isnwp', 'refit_w2', 'refit_peak_w']

    existing_df = pd.read_csv(cv_results_path) if os.path.exists(cv_results_path) else pd.DataFrame()
    records = existing_df.to_dict('records') if len(existing_df) > 0 else []

    print("=" * 80)
    print("FASE 1: CROSS-FITTING POR TRIMESTRES 2024 (HORIZONTES RESTANTES)")
    print("=" * 80)

    for h in remaining_horizons:
        print(f"\n>>> HORIZONTE h={h}")
        df, chem, t0, honest, pp = load_cache_for_h(h)

        base_d_row = inv[(inv['exp'] == 'v3_base') & (inv['var'] == 'D_honest_noise') & (inv['h'] == h)]
        base_d_val = base_d_row['val_rmse'].values[0] if len(base_d_row) > 0 else np.nan
        base_d_test = base_d_row['test_rmse'].values[0] if len(base_d_row) > 0 else np.nan
        print(f"  Referencia Base D_honest_noise h={h:2d} -> Val RMSE: {base_d_val:6.4f} | Test RMSE: {base_d_test:6.4f}")

        for var_name in cv_variants:
            # Verificar si ya fue evaluado
            if len(existing_df) > 0:
                match = existing_df[(existing_df['horizon'] == h) & (existing_df['variant'] == var_name)]
                if len(match) > 0:
                    print(f"  [CACHE] h={h} {var_name} ya existe -> Val OOF RMSE: {match['val_OOF_RMSE'].values[0]:6.4f}")
                    continue

            cfg = VARIANTS[var_name]
            print(f"\n  --- Evaluando {var_name} (h={h}) ---")
            print(f"      {cfg['desc']}")

            oof_m, preds_oof_df, best_iters = cross_fit_2024(
                h=h, df=df, chem_cols=chem, t0_cols=t0, honest_cols=honest, pp_cols=pp,
                noise_injection=cfg['noise_injection'],
                is_nwp_feature=cfg['is_nwp_feature'],
                weight_2024=cfg['weight_2024'],
                peak_weights=cfg['peak_weights'],
                leak_free_calib=cfg['leak_free_calib']
            )

            oof_p = os.path.join(OOF_DIR, f'oof_{var_name}_h{h}.parquet')
            preds_oof_df.to_parquet(oof_p, index=False)

            diff_d = oof_m['RMSE'] - base_d_val
            pct_diff = (diff_d / base_d_val) * 100.0

            rec = {
                'horizon': h,
                'variant': var_name,
                'val_OOF_RMSE': round(oof_m['RMSE'], 5),
                'base_D_val_RMSE': round(base_d_val, 5),
                'diff_vs_D': round(diff_d, 5),
                'pct_diff_vs_D': round(pct_diff, 3),
                'val_MAE': round(oof_m['MAE'], 5),
                'val_R2': round(oof_m['R2'], 5),
                'val_RMSE_peaks': round(oof_m['RMSE_peaks'], 5),
                'val_bias_peaks': round(oof_m['bias_peaks'], 5),
                'median_best_iter': oof_m['median_best_iter'],
                'mean_best_iter': round(oof_m['mean_best_iter'], 1),
                'best_iters': str(best_iters),
                'cv_time_s': round(oof_m['total_cv_time'], 1),
                'desc': cfg['desc']
            }
            records.append(rec)

            df_out = pd.DataFrame(records)
            df_out = df_out.drop_duplicates(subset=['horizon', 'variant'], keep='last').reset_index(drop=True)
            df_out.to_csv(cv_results_path, index=False)
            existing_df = df_out

            sign = "-" if diff_d < 0 else "+"
            status = "MEJORA" if diff_d < 0 else "PEOR/IGUAL"
            print(f"  [RESULTADO] h={h:2d} | {var_name:15s} | Val OOF: {oof_m['RMSE']:6.4f} (vs D {base_d_val:6.4f} -> {sign}{abs(diff_d):.4f}, {sign}{abs(pct_diff):.2f}%) [{status}] | Pk RMSE: {oof_m['RMSE_peaks']:6.2f} | Time: {oof_m['total_cv_time']:.1f}s")

        del df
        gc.collect()

    print("\n" + "=" * 80)
    print("FASE 2: SELECCION DE GANADORES POR HORIZONTE (SOLO VAL OOF)")
    print("=" * 80)

    cv_all = pd.read_csv(cv_results_path)
    best_per_h = {}
    for h in [1, 4, 8, 12, 24, 48]:
        h_rows = cv_all[cv_all['horizon'] == h]
        # Filtrar solo variantes candidatas válidas (excluir leak_test)
        candidates = h_rows[h_rows['variant'] != 'refit_leak_test']
        best_row = candidates.sort_values('val_OOF_RMSE').iloc[0]
        best_var = best_row['variant']
        best_val = best_row['val_OOF_RMSE']
        diff_d = best_row['diff_vs_D']
        best_per_h[h] = {
            'variant': best_var,
            'val_OOF_RMSE': best_val,
            'diff_vs_D': diff_d,
            'best_iters': json.loads(best_row['best_iters']),
            'row': best_row
        }
        print(f"  h={h:2d} -> Ganador: {best_var:15s} | Val OOF RMSE: {best_val:6.4f} (Mejora vs D: {diff_d:+.4f}) | iters={best_row['best_iters']}")

    print("\n" + "=" * 80)
    print("FASE 3: REFIT COMPLETO (2020-2024) Y EVALUACION TEST 2025 CON BAGGING")
    print("=" * 80)

    final_results = []
    seeds = [42, 43, 44]

    for h in [1, 4, 8, 12, 24, 48]:
        info = best_per_h[h]
        best_var = info['variant']
        cfg = VARIANTS[best_var]
        best_iters = info['best_iters']
        num_trees = int(round(np.median(best_iters) * 1.1))

        print(f"\n>>> REFIT h={h:2d} | Variante: {best_var} | Trees: {num_trees}")

        df, chem, t0, honest, pp = load_cache_for_h(h)

        # Cargar predicciones OOF de val 2024
        oof_p = os.path.join(OOF_DIR, f'oof_{best_var}_h{h}.parquet')
        if not os.path.exists(oof_p):
            raise FileNotFoundError(f"No existe predicción OOF: {oof_p}")
        preds_oof_df = pd.read_parquet(oof_p)

        # Reentrenar con cada semilla
        test_preds_by_seed = {}
        single_seed_m = None

        for s in seeds:
            print(f"  Entrenando seed {s} ({num_trees} arboles)...")
            m_s, p_s = refit_and_evaluate_test(
                h=h, df=df, chem_cols=chem, t0_cols=t0, honest_cols=honest, pp_cols=pp,
                best_iters=best_iters,
                noise_injection=cfg['noise_injection'],
                is_nwp_feature=cfg['is_nwp_feature'],
                weight_2024=cfg['weight_2024'],
                peak_weights=cfg['peak_weights'],
                iter_mult=1.1,
                seed=s,
                variant_name=best_var
            )
            test_preds_by_seed[s] = p_s['y_pred'].values
            if s == 42:
                single_seed_m = m_s
                print(f"    [Seed 42] Test RMSE: {m_s['RMSE']:6.4f} | R2: {m_s['R2']:.4f} | Pk RMSE: {m_s['RMSE_peaks']:6.2f}")

        # Ensamble de 3 semillas (promedio)
        preds_bagging = np.mean([test_preds_by_seed[s] for s in seeds], axis=0).astype('float32')

        te_eval_df = df[df['split'] == 'test'].copy().reset_index(drop=True)
        te_eval_df = te_eval_df[te_eval_df['Target'].notna()].reset_index(drop=True)
        te_eval_df[f'Target_{h}h'] = te_eval_df['Target']

        m_bag = evaluate_test(te_eval_df, preds_bagging, h)
        gain_bag = single_seed_m['RMSE'] - m_bag['RMSE']
        print(f"    [Bagging 3-seeds] Test RMSE: {m_bag['RMSE']:6.4f} | R2: {m_bag['R2']:.4f} | Pk RMSE: {m_bag['RMSE_peaks']:6.2f} | Ganancia bagging: {gain_bag:+.4f}")

        # Guardar parquets de predicciones oficiales
        preds_test_df = pd.DataFrame({
            'Date': te_eval_df['Date'],
            'Estacion': te_eval_df['Estacion'],
            'h': h,
            'split': 'test',
            'y_true': te_eval_df['Target'].astype('float32'),
            'y_pred': preds_bagging
        })

        preds_all = pd.concat([preds_oof_df, preds_test_df], ignore_index=True)
        
        # 1. Guardar con nombre específico de la variante: e.g. refit_isnwp_h24.parquet
        p_var = os.path.join(PREDS_DIR, f'{best_var}_h{h}.parquet')
        preds_all.to_parquet(p_var, index=False)
        
        # 2. Guardar también como refit_best_h{h}.parquet para referencia canónica
        p_best = os.path.join(PREDS_DIR, f'refit_best_h{h}.parquet')
        preds_all.to_parquet(p_best, index=False)

        # Baseline de referencia D y ensamble noche1
        row_d = inv[(inv['exp'] == 'v3_base') & (inv['var'] == 'D_honest_noise') & (inv['h'] == h)].iloc[0]
        row_ens = inv[(inv['exp'] == 'ensemble') & (inv['var'] == 'ensemble_best') & (inv['h'] == h)].iloc[0]

        # Registrar en results_registry.csv
        notes = f"Refit 2020-2024 con {best_var} ({cfg['desc']}). Val=OOF 4-folds 2024, Test=refit 3-seed bagging ({num_trees} arboles)."
        log_result(
            experiment='noche2_refit',
            variant=best_var,
            h=h,
            m=m_bag,
            notes=notes
        )

        res_record = {
            'h': h,
            'best_variant': best_var,
            'val_OOF_RMSE': round(info['val_OOF_RMSE'], 4),
            'base_D_val': round(row_d['val_rmse'], 4),
            'test_RMSE_seed42': round(single_seed_m['RMSE'], 4),
            'test_RMSE_bagging': round(m_bag['RMSE'], 4),
            'base_D_test': round(row_d['test_rmse'], 4),
            'diff_vs_D_test': round(m_bag['RMSE'] - row_d['test_rmse'], 4),
            'ensemble_noche1_test': round(row_ens['test_rmse'], 4),
            'test_R2': round(m_bag['R2'], 4),
            'test_RMSE_peaks': round(m_bag['RMSE_peaks'], 2),
            'test_bias_peaks': round(m_bag['bias_peaks'], 2),
            'bagging_gain': round(gain_bag, 4),
            'num_trees': num_trees
        }
        final_results.append(res_record)

        del df
        gc.collect()

    summary_df = pd.DataFrame(final_results)
    summary_path = os.path.join(BASE_DIR, 'refit_final_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("RESUMEN FINAL noche2_refit")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print(f"\nResumen guardado en {summary_path}")
    print(f"Tiempo total: {(time.time() - start_total) / 60:.1f} minutos")


if __name__ == '__main__':
    run_pipeline()
