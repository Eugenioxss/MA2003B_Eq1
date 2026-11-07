"""
experiments/noche2/refit/finish_fase1.py
Completa el refit final para h=24 y h=48 con bagging de 3 semillas.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
EXP_DIR = os.path.join(ROOT, 'experiments')
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, BASE_DIR)

from common import evaluate_test, log_result, metrics
from train_refit import load_cache_for_h, refit_and_evaluate_test, VARIANTS, OOF_DIR, PREDS_DIR

cv_results_path = os.path.join(BASE_DIR, 'cv_results.csv')
inv_path = os.path.join(ROOT, 'experiments', 'noche2', 'inventory.csv')
inv = pd.read_csv(inv_path)
cv_all = pd.read_csv(cv_results_path)

seeds = [42, 43, 44]

# Horizontes a terminar
for h in [24, 48]:
    h_rows = cv_all[(cv_all['horizon'] == h) & (cv_all['variant'] != 'refit_leak_test')]
    best_row = h_rows.sort_values('val_OOF_RMSE').iloc[0]
    best_var = best_row['variant']
    best_val = best_row['val_OOF_RMSE']
    best_iters = json.loads(best_row['best_iters'])
    num_trees = int(round(np.median(best_iters) * 1.1))
    cfg = VARIANTS[best_var]

    print(f"\n>>> REFIT h={h:2d} | Ganador: {best_var} | Trees: {num_trees} (mediana={np.median(best_iters):.0f})")

    df, chem, t0, honest, pp = load_cache_for_h(h)
    oof_p = os.path.join(OOF_DIR, f'oof_{best_var}_h{h}.parquet')
    preds_oof_df = pd.read_parquet(oof_p)

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

    preds_bagging = np.mean([test_preds_by_seed[s] for s in seeds], axis=0).astype('float32')

    te_eval_df = df[df['split'] == 'test'].copy().reset_index(drop=True)
    te_eval_df = te_eval_df[te_eval_df['Target'].notna()].reset_index(drop=True)
    te_eval_df[f'Target_{h}h'] = te_eval_df['Target']

    m_bag = evaluate_test(te_eval_df, preds_bagging, h)
    gain_bag = single_seed_m['RMSE'] - m_bag['RMSE']
    print(f"    [Bagging 3-seeds] Test RMSE: {m_bag['RMSE']:6.4f} | R2: {m_bag['R2']:.4f} | Pk RMSE: {m_bag['RMSE_peaks']:6.2f} | Ganancia bagging: {gain_bag:+.4f}")

    preds_test_df = pd.DataFrame({
        'Date': te_eval_df['Date'],
        'Estacion': te_eval_df['Estacion'],
        'h': h,
        'split': 'test',
        'y_true': te_eval_df['Target'].astype('float32'),
        'y_pred': preds_bagging
    })

    preds_all = pd.concat([preds_oof_df, preds_test_df], ignore_index=True)
    p_var = os.path.join(PREDS_DIR, f'{best_var}_h{h}.parquet')
    preds_all.to_parquet(p_var, index=False)
    p_best = os.path.join(PREDS_DIR, f'refit_best_h{h}.parquet')
    preds_all.to_parquet(p_best, index=False)

    notes = f"Refit 2020-2024 con {best_var} ({cfg['desc']}). Val=OOF 4-folds 2024, Test=refit 3-seed bagging ({num_trees} arboles)."
    log_result(
        experiment='noche2_refit',
        variant=best_var,
        h=h,
        m=m_bag,
        notes=notes
    )

print("\nRefit h=24 y h=48 completado con exito.")
