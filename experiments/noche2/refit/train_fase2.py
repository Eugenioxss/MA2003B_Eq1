"""
experiments/noche2/refit/train_fase2.py
=======================================
Fase 2 de noche2_refit:
Incorpora al cache base v3 las 12 características meteorológicas multi-modelo + MOS
generadas por el agente NWP en experiments/noche2/nwp/features_nwp_h<h>.parquet.

Variantes a entrenar y evaluar:
1. refit_nwp_best: configuración ganadora de Fase 1 por h + 12 NWP features.
2. refit_nwp_peak_w: refit_nwp_best + pesos de picos w = 1 + (y/p90)^2.
3. refit_nwp_resid_persist (para h=1 y h=48): residual sobre persistencia contemporánea O3(t).

Esquema:
- Cross-fitting honesto de 4 trimestres en 2024 para val OOF.
- Refit completo 2020-2024 con bagging de 3 semillas (42, 43, 44) para test 2025.
- num_threads=5 (según instrucción).
"""
import os
import sys
import time
import json
import gc
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import partial

print = partial(print, flush=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
EXP_DIR = os.path.join(ROOT, 'experiments')
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, BASE_DIR)

from common import HORIZONS, evaluate_test, log_result, metrics, climatology_pred, load_panel, SEED
from train_refit import load_cache_for_h, inject_meteo_noise, BEST_OPTUNA, OOF_DIR, PREDS_DIR

NWP_DIR = os.path.join(ROOT, 'experiments', 'noche2', 'nwp')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Parámetros LightGBM con num_threads=5 para Fase 2
LGB_PARAMS_FASE2 = {
    'objective': 'regression',
    'metric': 'rmse',
    'seed': SEED,
    'verbose': -1,
    'num_threads': 5,  # Contrato Fase 2: 5 hilos
    'feature_pre_filter': False,
    'importance_type': 'gain',
    **BEST_OPTUNA
}

# Configuración ganadora de Fase 1 por horizonte
BEST_CFG_FASE1 = {
    1: {'noise_injection': True, 'is_nwp_feature': False, 'weight_2024': 1.0},
    4: {'noise_injection': True, 'is_nwp_feature': False, 'weight_2024': 2.0},
    8: {'noise_injection': True, 'is_nwp_feature': False, 'weight_2024': 2.0},
    12: {'noise_injection': True, 'is_nwp_feature': True, 'weight_2024': 1.0},
    24: {'noise_injection': True, 'is_nwp_feature': True, 'weight_2024': 1.0},
    48: {'noise_injection': True, 'is_nwp_feature': True, 'weight_2024': 1.0},
}


def load_fase2_dataset(h: int) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str], list[str]]:
    """Carga cache v3_base y cruza con features_nwp_h<h>.parquet."""
    df_base, chem_cols, t0_cols, honest_cols, pp_cols = load_cache_for_h(h)
    
    nwp_path = os.path.join(NWP_DIR, f'features_nwp_h{h}.parquet')
    if not os.path.exists(nwp_path):
        raise FileNotFoundError(f"No existe archivo NWP: {nwp_path}")
    
    df_nwp = pd.read_parquet(nwp_path)
    nwp_cols = [c for c in df_nwp.columns if c not in ['Date', 'Estacion']]
    for c in nwp_cols:
        df_nwp[c] = df_nwp[c].astype('float32')

    # Merge
    df_merged = pd.merge(df_base, df_nwp, on=['Date', 'Estacion'], how='inner')

    # Si es h=1 o h=48, agregar persistencia contemporánea O3(t) de panel_full
    panel = load_panel()
    df_p = panel[['Date', 'Estacion', 'O3']].copy()
    df_p['O3_t0'] = df_p['O3'].astype('float32')
    df_merged = pd.merge(df_merged, df_p[['Date', 'Estacion', 'O3_t0']], on=['Date', 'Estacion'], how='left')

    df_merged['Estacion'] = df_merged['Estacion'].astype('category')

    return df_merged, chem_cols, t0_cols, honest_cols, pp_cols, nwp_cols


def run_cross_fitting_fase2(
    h: int,
    df: pd.DataFrame,
    features: list[str],
    honest_cols: list[str],
    pp_cols: list[str],
    cfg: dict,
    peak_weights: bool = False,
    is_residual_persist: bool = False,
    seed: int = SEED
) -> tuple[dict, pd.DataFrame, list[int]]:
    """Cross-fitting de 4 trimestres para Fase 2."""
    target_col = 'Target'
    cat_cols = ['Estacion']

    train_mask = (df['split'] == 'train') & df[target_col].notna()
    val_mask = (df['split'] == 'val') & df[target_col].notna()
    
    tr_df_base = df[train_mask].copy().reset_index(drop=True)
    val_df_full = df[val_mask].copy().reset_index(drop=True)
    val_df_full['quarter'] = val_df_full['Date'].dt.quarter

    oof_preds = np.zeros(len(val_df_full), dtype='float32')
    best_iters = []
    fold_times = []

    # Features a utilizar en el modelo
    used_features = list(features)
    if cfg['is_nwp_feature'] and 'is_nwp' not in used_features:
        used_features.append('is_nwp')

    for q in [1, 2, 3, 4]:
        t0_fold = time.time()
        q_val_idx = (val_df_full['quarter'] == q).values
        q_tr_idx = (val_df_full['quarter'] != q).values

        # Inyección de ruido calibrado sin Q
        if cfg['noise_injection']:
            calib_df = val_df_full[q_tr_idx]
            tr_part_a = inject_meteo_noise(tr_df_base, calib_df, honest_cols, pp_cols, seed=seed)
        else:
            tr_part_a = tr_df_base.copy()

        tr_part_b = val_df_full[q_tr_idx].copy()

        if cfg['is_nwp_feature']:
            tr_part_a['is_nwp'] = np.float32(0.0)
            tr_part_b['is_nwp'] = np.float32(1.0)
            val_df_full.loc[q_val_idx, 'is_nwp'] = np.float32(1.0)

        tr_fold = pd.concat([tr_part_a, tr_part_b], ignore_index=True)
        tr_fold['Estacion'] = tr_fold['Estacion'].astype('category')

        # Determinar target (directo vs resid_persist)
        if is_residual_persist:
            # P(t) con fallback a O3_lag_1 y luego a climatología
            def get_persist(sub_df):
                p = sub_df['O3_t0'].to_numpy(copy=True, dtype=np.float32)
                nan_mask = ~np.isfinite(p)
                if nan_mask.any():
                    lag1 = sub_df['O3_lag_1'].to_numpy(copy=True, dtype=np.float32)
                    p[nan_mask] = lag1[nan_mask]
                    nan_mask2 = ~np.isfinite(p)
                    if nan_mask2.any():
                        c = climatology_pred(sub_df['Estacion'], sub_df['Date'])
                        p[nan_mask2] = c[nan_mask2].astype('float32')
                return p

            p_tr = get_persist(tr_fold)
            p_va = get_persist(val_df_full[q_val_idx])

            y_tr = (tr_fold[target_col].values - p_tr).astype('float32')
            y_va_raw = val_df_full.loc[q_val_idx, target_col].values
            y_va_for_early_stop = (y_va_raw - p_va).astype('float32')
        else:
            y_tr = tr_fold[target_col].values
            y_va_raw = val_df_full.loc[q_val_idx, target_col].values
            y_va_for_early_stop = y_va_raw

        # Pesos
        weights = np.ones(len(tr_fold), dtype='float32')
        if cfg['weight_2024'] != 1.0:
            weights[len(tr_part_a):] = np.float32(cfg['weight_2024'])

        if peak_weights:
            y_orig = tr_fold[target_col].values
            p90 = np.percentile(y_orig, 90.0)
            pk_w = 1.0 + (y_orig / p90) ** 2
            weights = (weights * pk_w).astype('float32')

        X_tr = tr_fold[used_features]
        va_fold = val_df_full[q_val_idx]
        X_va = va_fold[used_features]

        lgb_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, weight=weights, free_raw_data=False)
        lgb_va = lgb.Dataset(X_va, y_va_for_early_stop, reference=lgb_tr, categorical_feature=cat_cols, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0)
        ]

        params = dict(LGB_PARAMS_FASE2)
        params['seed'] = seed

        model = lgb.train(
            params,
            lgb_tr,
            num_boost_round=2000,
            valid_sets=[lgb_va],
            callbacks=callbacks
        )

        b_iter = model.best_iteration
        best_iters.append(b_iter)
        pred_va_model = model.predict(X_va, num_iteration=b_iter)

        if is_residual_persist:
            pred_va_final = np.maximum(0.0, p_va + pred_va_model)
        else:
            pred_va_final = np.maximum(0.0, pred_va_model)

        oof_preds[q_val_idx] = pred_va_final
        t_fold = time.time() - t0_fold
        fold_times.append(t_fold)
        fold_rmse = float(np.sqrt(np.mean((y_va_raw - pred_va_final) ** 2)))
        print(f"      Fold Q{q} | N={len(va_fold):5d} | BestIter={b_iter:4d} | RMSE={fold_rmse:6.3f} | Tiempo={t_fold:4.1f}s")

    y_true_all = val_df_full[target_col].values
    oof_m = metrics(y_true_all, oof_preds)
    oof_m['median_best_iter'] = int(np.median(best_iters))
    oof_m['mean_best_iter'] = float(np.mean(best_iters))
    oof_m['total_cv_time'] = float(np.sum(fold_times))

    preds_oof_df = pd.DataFrame({
        'Date': val_df_full['Date'],
        'Estacion': val_df_full['Estacion'],
        'h': h,
        'split': 'val',
        'y_true': y_true_all.astype('float32'),
        'y_pred': oof_preds.astype('float32')
    })

    return oof_m, preds_oof_df, best_iters


def refit_test_fase2(
    h: int,
    df: pd.DataFrame,
    features: list[str],
    honest_cols: list[str],
    pp_cols: list[str],
    best_iters: list[int],
    cfg: dict,
    peak_weights: bool = False,
    is_residual_persist: bool = False,
    iter_mult: float = 1.1,
    seeds: list[int] = [42, 43, 44]
) -> tuple[dict, pd.DataFrame]:
    """Refit completo sobre 2020-2024 y bagging de 3 semillas para test 2025."""
    target_col = 'Target'
    cat_cols = ['Estacion']

    train_mask = (df['split'] == 'train') & df[target_col].notna()
    val_mask = (df['split'] == 'val') & df[target_col].notna()
    test_mask = (df['split'] == 'test') & df[target_col].notna()

    tr_df_base = df[train_mask].copy().reset_index(drop=True)
    val_df_full = df[val_mask].copy().reset_index(drop=True)
    te_df = df[test_mask].copy().reset_index(drop=True)

    # Inyección de ruido con todo 2024
    if cfg['noise_injection']:
        tr_part_a = inject_meteo_noise(tr_df_base, val_df_full, honest_cols, pp_cols, seed=SEED)
    else:
        tr_part_a = tr_df_base.copy()

    tr_part_b = val_df_full.copy()

    used_features = list(features)
    if cfg['is_nwp_feature']:
        tr_part_a['is_nwp'] = np.float32(0.0)
        tr_part_b['is_nwp'] = np.float32(1.0)
        te_df['is_nwp'] = np.float32(1.0)
        if 'is_nwp' not in used_features:
            used_features.append('is_nwp')

    tr_full = pd.concat([tr_part_a, tr_part_b], ignore_index=True)
    tr_full['Estacion'] = tr_full['Estacion'].astype('category')
    te_df['Estacion'] = te_df['Estacion'].astype('category')

    # Target
    if is_residual_persist:
        def get_persist(sub_df):
            p = sub_df['O3_t0'].to_numpy(copy=True, dtype=np.float32)
            nan_mask = ~np.isfinite(p)
            if nan_mask.any():
                lag1 = sub_df['O3_lag_1'].to_numpy(copy=True, dtype=np.float32)
                p[nan_mask] = lag1[nan_mask]
                nan_mask2 = ~np.isfinite(p)
                if nan_mask2.any():
                    c = climatology_pred(sub_df['Estacion'], sub_df['Date'])
                    p[nan_mask2] = c[nan_mask2].astype('float32')
            return p

        p_tr = get_persist(tr_full)
        p_te = get_persist(te_df)
        y_tr = (tr_full[target_col].values - p_tr).astype('float32')
    else:
        y_tr = tr_full[target_col].values

    weights = np.ones(len(tr_full), dtype='float32')
    if cfg['weight_2024'] != 1.0:
        weights[len(tr_part_a):] = np.float32(cfg['weight_2024'])

    if peak_weights:
        y_orig = tr_full[target_col].values
        p90 = np.percentile(y_orig, 90.0)
        pk_w = 1.0 + (y_orig / p90) ** 2
        weights = (weights * pk_w).astype('float32')

    X_tr = tr_full[used_features]
    X_te = te_df[used_features]
    y_te_raw = te_df[target_col].values

    num_trees = int(round(np.median(best_iters) * iter_mult))
    print(f"    Reentrenando {num_trees} árboles con {len(seeds)} semillas...")

    test_preds_by_seed = []
    for s in seeds:
        lgb_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, weight=weights, free_raw_data=False)
        params = dict(LGB_PARAMS_FASE2)
        params['seed'] = s

        model = lgb.train(params, lgb_tr, num_boost_round=num_trees)
        pred_raw = model.predict(X_te, num_iteration=num_trees)
        if is_residual_persist:
            pred_final = np.maximum(0.0, p_te + pred_raw)
        else:
            pred_final = np.maximum(0.0, pred_raw)
        test_preds_by_seed.append(pred_final)

    preds_bagging = np.mean(test_preds_by_seed, axis=0).astype('float32')

    te_eval_df = te_df.copy()
    te_eval_df[f'Target_{h}h'] = te_eval_df[target_col]
    m_te = evaluate_test(te_eval_df, preds_bagging, h)
    m_te['num_trees'] = num_trees

    preds_test_df = pd.DataFrame({
        'Date': te_df['Date'],
        'Estacion': te_df['Estacion'],
        'h': h,
        'split': 'test',
        'y_true': y_te_raw.astype('float32'),
        'y_pred': preds_bagging
    })

    return m_te, preds_test_df


def run_fase2(horizons_to_run: list[int]):
    print("=" * 85)
    print("INICIANDO FASE 2: noche2_refit CON NWP FEATURES (num_threads=5)")
    print("=" * 85)

    inv = pd.read_csv(os.path.join(ROOT, 'experiments', 'noche2', 'inventory.csv'))
    summary_path = os.path.join(BASE_DIR, 'fase2_results_summary.csv')
    records = []
    if os.path.exists(summary_path):
        records = pd.read_csv(summary_path).to_dict('records')

    for h in horizons_to_run:
        print(f"\n{'='*80}")
        print(f"  CARGANDO DATOS FASE 2 HORIZONTE h={h}")
        print(f"{'='*80}")
        df, chem, t0, honest, pp, nwp_cols = load_fase2_dataset(h)
        base_features = ['Estacion'] + chem + t0 + honest + nwp_cols

        cfg = BEST_CFG_FASE1[h]

        # Obtener baseline previo de Fase 1 para comparar
        refit_best_p = os.path.join(PREDS_DIR, f'refit_best_h{h}.parquet')
        df_f1 = pd.read_parquet(refit_best_p)
        va_f1 = df_f1[df_f1['split'] == 'val']
        te_f1 = df_f1[df_f1['split'] == 'test']
        rmse_val_f1 = float(np.sqrt(np.mean((va_f1['y_true'] - va_f1['y_pred']) ** 2)))
        rmse_te_f1 = float(np.sqrt(np.mean((te_f1['y_true'] - te_f1['y_pred']) ** 2)))

        print(f"  Referencia Fase 1 (refit_best) h={h:2d} -> Val OOF: {rmse_val_f1:6.4f} | Test: {rmse_te_f1:6.4f}")

        # Lista de variantes a correr para este h
        variants_for_h = [
            ('refit_nwp_best', False, False),
            ('refit_nwp_peak_w', True, False)
        ]
        if h in [1, 48]:
            variants_for_h.append(('refit_nwp_resid_persist', False, True))

        for var_name, peak_w, resid_persist in variants_for_h:
            print(f"\n  --- Ejecutando {var_name} (h={h}) ---")
            oof_m, preds_oof_df, best_iters = run_cross_fitting_fase2(
                h=h,
                df=df,
                features=base_features,
                honest_cols=honest,
                pp_cols=pp,
                cfg=cfg,
                peak_weights=peak_w,
                is_residual_persist=resid_persist
            )

            print(f"    Val OOF RMSE: {oof_m['RMSE']:6.4f} (vs Fase 1 {rmse_val_f1:6.4f} -> {oof_m['RMSE'] - rmse_val_f1:+.4f}) | Pk RMSE: {oof_m['RMSE_peaks']:6.2f}")

            # Refit y test 2025 con 3 semillas
            m_te, preds_test_df = refit_test_fase2(
                h=h,
                df=df,
                features=base_features,
                honest_cols=honest,
                pp_cols=pp,
                best_iters=best_iters,
                cfg=cfg,
                peak_weights=peak_w,
                is_residual_persist=resid_persist,
                iter_mult=1.1,
                seeds=[42, 43, 44]
            )

            print(f"    Test 2025 RMSE: {m_te['RMSE']:6.4f} (vs Fase 1 {rmse_te_f1:6.4f} -> {m_te['RMSE'] - rmse_te_f1:+.4f}) | R2: {m_te['R2']:.4f} | Pk RMSE: {m_te['RMSE_peaks']:6.2f}")

            # Guardar parquet oficial
            preds_all = pd.concat([preds_oof_df, preds_test_df], ignore_index=True)
            p_out = os.path.join(PREDS_DIR, f'{var_name}_h{h}.parquet')
            preds_all.to_parquet(p_out, index=False)
            print(f"    Predicciones guardadas en {p_out}")

            # Registrar en results_registry.csv
            notes = f"Fase 2 refit + NWP features ({var_name}). Val=OOF 4-folds 2024, Test=refit 3-seed bagging."
            log_result(
                experiment='noche2_refit',
                variant=var_name,
                h=h,
                m=m_te,
                notes=notes
            )

            rec = {
                'h': h,
                'variant': var_name,
                'val_OOF_RMSE': round(oof_m['RMSE'], 4),
                'fase1_val_RMSE': round(rmse_val_f1, 4),
                'diff_val_vs_f1': round(oof_m['RMSE'] - rmse_val_f1, 4),
                'test_RMSE': round(m_te['RMSE'], 4),
                'fase1_test_RMSE': round(rmse_te_f1, 4),
                'diff_test_vs_f1': round(m_te['RMSE'] - rmse_te_f1, 4),
                'test_R2': round(m_te['R2'], 4),
                'test_RMSE_peaks': round(m_te['RMSE_peaks'], 2),
                'test_bias_peaks': round(m_te['bias_peaks'], 2),
                'num_trees': m_te['num_trees']
            }
            records.append(rec)
            pd.DataFrame(records).to_csv(summary_path, index=False)

        del df
        gc.collect()

    print("\n" + "="*85)
    print("FASE 2 FINALIZADA")
    print("="*85)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizons', type=str, default='24,4,1,8,12,48')
    args = parser.parse_args()
    horizons = [int(x.strip()) for x in args.horizons.split(',')]
    run_fase2(horizons)
