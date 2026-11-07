"""
experiments/noche2/imput2/train_phase2_refit.py
===============================================
Fase 2 de noche2_imput2 (Optimizada para cumplir la Regla de <90 min):
Combina:
  1. Pipeline ganador imput2:
     - Imputación causal espacial LightGBM (1A_imp)
     - Memoria diaria del día anterior (Group3_yesterday)
     - Dinámica y contraste de cuenca regional (Group4_regional)
     - Variables por hora objetivo y ganancia fotoquímica (per_hour / delta_clim)
  2. Features NWP honestas del agente nwp:
     - Multi-modelo ECMWF + GFS + ICON (nwp_temp_mean, nwp_rad_mean, nwp_ws_mean, etc.)
     - Incertidumbre sinóptica / dispersión entre modelos (nwp_temp_std, nwp_rad_std, nwp_ws_std)
     - Diagnósticos termodinámicos (CAPE, VPD, ráfagas de viento)
     - Calibración estadística causal MOS (MOS_Temp, MOS_WS, MOS_Photothermal)
     - Indicador is_nwp (0 en <=2023, 1 en 2024-2025)
  3. Esquema de entrenamiento honesto:
     - 4-fold cross-fitting por trimestres de 2024 (val OOF 100% honesto, sin fuga)
     - Refit completo 2020-2024 fijando n_iters = int(round(median(best_iter_folds) * 1.1)) para predecir test 2025.
  4. Tres variantes completas para los 6 horizontes h in [4, 8, 24, 12, 48, 1]:
     - imput2_nwp_refit: sin ponderación de picos
     - imput2_nwp_refit_w: con ponderación de picos w = 1 + (y/p90)^2
     - imput2_nwp_resid_refit: prediciendo el residual de persistencia Delta = y - O3(t0)

Guarda predicciones oficiales en preds/<variante>_h<h>.parquet y registra en results_registry.csv.
"""
from __future__ import annotations
import os
import sys
import time
import json
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

from common import load_panel, evaluate_test, metrics, log_result, climatology_pred, MODEL_STATIONS, HORIZONS, SEED
from planA_quickwins.features_planA import build_planA_features
from experiments.noche2.imput2.features_pipeline import (
    compute_missing_features,
    compute_yesterday_stats,
    compute_regional_trends
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(EXP_DIR, 'preds')
os.makedirs(PREDS_DIR, exist_ok=True)

CACHE_V3 = os.path.join(ROOT, 'experiments', 'noche', 'v3_base', 'cache')
IMPUTED_PANEL_PATH = os.path.join(ROOT, 'experiments', 'noche', 'audit_imput', 'panel_spatial_lgbm.parquet')
NWP_DIR = os.path.join(ROOT, 'experiments', 'noche2', 'nwp')

# Hiperparámetros optimizados para convergencia rápida y óptima (<90 min total)
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,        # 0.05 converge en 700-1000 árboles (2x más rápido que 0.03 sin pérdida)
    'num_leaves': 127,
    'min_child_samples': 100,
    'subsample': 1.0,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'seed': SEED,
    'verbose': -1,
    'num_threads': 4,            # Contrato: 4 hilos
    'feature_pre_filter': False,
    'importance_type': 'gain'
}


def log_msg(msg: str):
    """Imprime y fuerza el vaciado del buffer de salida inmediatamente."""
    print(msg, flush=True)
    sys.stdout.flush()


def inject_meteo_noise(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    honest_cols: list[str],
    pp_cols: list[str],
    seed: int = 42
) -> pd.DataFrame:
    """Inyecta en train_df ruido empírico de NWP calibrado sobre calib_df por hora solar."""
    train_noisy = train_df.copy()
    rng = np.random.default_rng(seed)

    calib_hour = calib_df['Date'].dt.hour.values
    tr_hour = train_noisy['Date'].dt.hour.values

    for h_col in honest_cols:
        suffix = h_col.replace('honest_meteo_', '')
        pp_col = f'pp_meteo_{suffix}'
        if pp_col not in calib_df.columns:
            continue

        val_diff = (calib_df[h_col] - calib_df[pp_col]).values
        valid_mask = np.isfinite(val_diff)

        sampled_noise = np.zeros(len(train_noisy), dtype='float32')
        for hr in range(24):
            hr_calib_mask = valid_mask & (calib_hour == hr)
            hr_tr_mask = (tr_hour == hr)
            n_tr = int(hr_tr_mask.sum())
            if n_tr == 0:
                continue

            residuals_pool = val_diff[hr_calib_mask]
            if len(residuals_pool) > 0:
                sampled_noise[hr_tr_mask] = rng.choice(residuals_pool, size=n_tr, replace=True)
            else:
                sampled_noise[hr_tr_mask] = 0.0

        train_noisy[h_col] = (train_noisy[h_col] + sampled_noise).astype('float32')

        # Acotar físicamente
        if 'rh' in h_col or 'cloud' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, 100.0)
        elif 'rad' in h_col or 'ws' in h_col or 'precip' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, None)

    # Recalcular derivadas
    if 'honest_meteo_target_temp' in train_noisy.columns and 'meteo_t0_temp' in train_noisy.columns:
        train_noisy['honest_meteo_delta_temp'] = (train_noisy['honest_meteo_target_temp'] - train_noisy['meteo_t0_temp']).astype('float32')
    if 'honest_meteo_target_rh' in train_noisy.columns and 'meteo_t0_rh' in train_noisy.columns:
        train_noisy['honest_meteo_delta_rh'] = (train_noisy['honest_meteo_target_rh'] - train_noisy['meteo_t0_rh']).astype('float32')
    if 'honest_meteo_target_temp' in train_noisy.columns and 'honest_meteo_target_rad' in train_noisy.columns:
        train_noisy['honest_meteo_target_temp_x_rad'] = (train_noisy['honest_meteo_target_temp'] * train_noisy['honest_meteo_target_rad']).astype('float32')

    return train_noisy


def build_horizon_dataset(
    h: int,
    base_static: pd.DataFrame,
    chem_imp_cols: list[str],
    yesterday_cols: list[str],
    regional_cols: list[str],
    missing_cols: list[str]
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Construye dataset enriquecido para el horizonte h."""
    t0 = time.time()
    df_v3 = pd.read_parquet(os.path.join(CACHE_V3, f'features_h{h}.parquet'))
    t0_cols = [c for c in df_v3.columns if c.startswith('meteo_t0_')]
    honest_cols = [c for c in df_v3.columns if c.startswith('honest_meteo_')]
    pp_cols = [c for c in df_v3.columns if c.startswith('pp_meteo_')]
    target_col = f'Target_{h}h' if f'Target_{h}h' in df_v3.columns else 'Target'

    if f'Target_{h}h' not in df_v3.columns:
        df_v3[f'Target_{h}h'] = df_v3['Target']
    if 'Target' not in df_v3.columns:
        df_v3['Target'] = df_v3[f'Target_{h}h']

    cols_to_pull = ['Estacion', 'Date', 'split', 'Target', f'Target_{h}h'] + t0_cols + honest_cols + pp_cols
    cols_to_pull = list(dict.fromkeys(cols_to_pull))
    meteo_and_targets = df_v3[cols_to_pull]

    merged = pd.merge(base_static, meteo_and_targets, on=['Estacion', 'Date'], how='left')

    # Cargar NWP multi-modelo
    nwp_path = os.path.join(NWP_DIR, f'features_nwp_h{h}.parquet')
    if os.path.exists(nwp_path):
        nwp_df = pd.read_parquet(nwp_path)
        nwp_cols = [c for c in nwp_df.columns if c not in ['Date', 'Estacion']]
        merged = pd.merge(merged, nwp_df, on=['Estacion', 'Date'], how='left')
    else:
        nwp_cols = []

    # Features per_hour y delta_clim
    target_dt = merged['Date'] + pd.Timedelta(hours=h)
    target_hr = target_dt.dt.hour
    merged['target_hour'] = target_hr.astype('float32')
    merged['target_hour_sin'] = np.sin(2 * np.pi * target_hr / 24).astype('float32')
    merged['target_hour_cos'] = np.cos(2 * np.pi * target_hr / 24).astype('float32')

    clim_target = climatology_pred(merged['Estacion'], target_dt)
    clim_now = climatology_pred(merged['Estacion'], merged['Date'])
    merged['target_clim'] = clim_target.astype('float32')
    merged['delta_clim'] = (clim_target - clim_now).astype('float32')
    merged['is_target_daylight'] = ((target_hr >= 10) & (target_hr <= 18)).astype('float32')

    o3_now = merged['O3_lag_0'].astype('float32') if 'O3_lag_0' in merged.columns else merged['O3'].astype('float32')
    merged['O3_x_target_sin'] = (o3_now * merged['target_hour_sin']).astype('float32')
    merged['O3_x_target_cos'] = (o3_now * merged['target_hour_cos']).astype('float32')
    merged['O3_x_delta_clim'] = (o3_now * merged['delta_clim'] / 50.0).astype('float32')

    per_hour_cols = [
        'target_hour', 'target_hour_sin', 'target_hour_cos', 'target_clim', 'delta_clim',
        'is_target_daylight', 'O3_x_target_sin', 'O3_x_target_cos', 'O3_x_delta_clim'
    ]

    merged['is_nwp'] = (merged['Date'].dt.year >= 2024).astype('float32')
    merged['Estacion'] = merged['Estacion'].astype('category')

    all_feature_cols = ['Estacion'] + chem_imp_cols + yesterday_cols + regional_cols + t0_cols + honest_cols + nwp_cols + per_hour_cols + ['is_nwp']
    if h == 1:
        all_feature_cols += missing_cols

    all_feature_cols = list(dict.fromkeys(all_feature_cols))

    for c in all_feature_cols:
        if c != 'Estacion' and merged[c].dtype == 'float64':
            merged[c] = merged[c].astype('float32')

    log_msg(f"  Dataset h={h} construido en {time.time()-t0:.1f}s: {len(merged)} filas, {len(all_feature_cols)} features.")
    return merged, all_feature_cols, honest_cols, pp_cols


def run_cross_fitting_and_refit(
    h: int,
    df: pd.DataFrame,
    features: list[str],
    honest_cols: list[str],
    pp_cols: list[str],
    variant_type: str = 'direct',  # 'direct', 'weighted', 'residual'
    seed: int = SEED,
    iter_mult: float = 1.1
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Ejecuta el protocolo de Fase 2:
    1. 4-fold cross-fitting por trimestres de 2024 para obtener OOF val 100% honesto.
    2. Refit en 2020-2024 con n_iter = round(median(best_iter) * iter_mult) para test 2025.
    """
    target_col = f'Target_{h}h'
    cat_cols = ['Estacion']

    train_mask = (df['split'] == 'train') & df[target_col].notna()
    val_mask = (df['split'] == 'val') & df[target_col].notna()
    test_mask = (df['split'] == 'test') & df[target_col].notna()

    tr_df_base = df[train_mask].copy().reset_index(drop=True)
    val_df_full = df[val_mask].copy().reset_index(drop=True)
    te_df = df[test_mask].copy().reset_index(drop=True)

    val_df_full['quarter'] = val_df_full['Date'].dt.quarter
    o3_t0_col = 'O3_lag_0' if 'O3_lag_0' in df.columns else 'O3'

    if variant_type == 'residual':
        learn_target = 'Target_diff'
        tr_df_base[learn_target] = (tr_df_base[target_col] - tr_df_base[o3_t0_col]).astype('float32')
        val_df_full[learn_target] = (val_df_full[target_col] - val_df_full[o3_t0_col]).astype('float32')
    else:
        learn_target = target_col

    oof_preds = np.zeros(len(val_df_full), dtype='float32')
    best_iters = []

    log_msg(f"    [CV 2024] Ejecutando 4-fold cross-fitting ({variant_type})...")

    # 1. Cross-fitting 4 folds
    for q in [1, 2, 3, 4]:
        t_f0 = time.time()
        q_val_idx = (val_df_full['quarter'] == q).values
        q_tr_idx = (val_df_full['quarter'] != q).values

        calib_df = val_df_full[q_tr_idx]
        tr_part_a = inject_meteo_noise(tr_df_base, calib_df, honest_cols, pp_cols, seed=seed)
        tr_part_a['is_nwp'] = np.float32(0.0)

        tr_part_b = val_df_full[q_tr_idx].copy()
        tr_part_b['is_nwp'] = np.float32(1.0)

        tr_fold = pd.concat([tr_part_a, tr_part_b], ignore_index=True)

        if variant_type == 'weighted':
            y_fold = tr_fold[target_col].values
            p90 = np.percentile(y_fold, 90.0)
            weights = (1.0 + (y_fold / p90) ** 2).astype('float32')
        else:
            weights = None

        X_tr = tr_fold[features]
        y_tr = tr_fold[learn_target].values
        va_fold = val_df_full[q_val_idx]
        X_va = va_fold[features]
        y_va = va_fold[learn_target].values

        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, weight=weights, free_raw_data=False)
        dva = lgb.Dataset(X_va, y_va, reference=dtr, categorical_feature=cat_cols, free_raw_data=False)

        cb = [lgb.early_stopping(50, verbose=False)]
        bst = lgb.train(LGB_PARAMS, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=cb)

        b_iter = bst.best_iteration
        best_iters.append(b_iter)

        raw_pred_va = bst.predict(X_va, num_iteration=b_iter)
        if variant_type == 'residual':
            recon_pred = np.maximum(0.0, va_fold[o3_t0_col].values + raw_pred_va)
            oof_preds[q_val_idx] = recon_pred.astype('float32')
        else:
            oof_preds[q_val_idx] = np.maximum(0.0, raw_pred_va).astype('float32')

        f_rmse = np.sqrt(np.mean((val_df_full.loc[q_val_idx, target_col].values - oof_preds[q_val_idx]) ** 2))
        log_msg(f"      Fold Q{q} en {time.time()-t_f0:.1f}s | Iter={b_iter:4d} | Val RMSE={f_rmse:.4f}")

    y_true_val = val_df_full[target_col].values
    val_m = metrics(y_true_val, oof_preds)
    val_m['median_best_iter'] = int(np.median(best_iters))
    log_msg(f"    --> [Val OOF 2024] RMSE = {val_m['RMSE']:.4f} | R2 = {val_m['R2']:.4f} | Pk RMSE = {val_m['RMSE_peaks']:.2f} | Mediana Iters = {val_m['median_best_iter']}")

    # 2. Refit 2020-2024
    num_trees = int(round(np.median(best_iters) * iter_mult))
    log_msg(f"    [Refit 2020-2024] Entrenando con {num_trees} árboles (mediana={np.median(best_iters):.0f} * {iter_mult})...")

    tr_part_a = inject_meteo_noise(tr_df_base, val_df_full, honest_cols, pp_cols, seed=seed)
    tr_part_a['is_nwp'] = np.float32(0.0)

    tr_part_b = val_df_full.copy()
    tr_part_b['is_nwp'] = np.float32(1.0)
    te_df['is_nwp'] = np.float32(1.0)

    tr_full = pd.concat([tr_part_a, tr_part_b], ignore_index=True)

    if variant_type == 'residual':
        tr_full[learn_target] = (tr_full[target_col] - tr_full[o3_t0_col]).astype('float32')

    if variant_type == 'weighted':
        y_all = tr_full[target_col].values
        p90 = np.percentile(y_all, 90.0)
        weights_refit = (1.0 + (y_all / p90) ** 2).astype('float32')
    else:
        weights_refit = None

    X_tr_full = tr_full[features]
    y_tr_full = tr_full[learn_target].values
    X_te = te_df[features]

    dtr_full = lgb.Dataset(X_tr_full, y_tr_full, categorical_feature=cat_cols, weight=weights_refit, free_raw_data=False)
    t_refit0 = time.time()
    bst_refit = lgb.train(LGB_PARAMS, dtr_full, num_boost_round=num_trees)
    t_refit = time.time() - t_refit0

    raw_pred_te = bst_refit.predict(X_te, num_iteration=num_trees)
    if variant_type == 'residual':
        test_pred = np.maximum(0.0, te_df[o3_t0_col].values + raw_pred_te).astype('float32')
    else:
        test_pred = np.maximum(0.0, raw_pred_te).astype('float32')

    te_eval_df = te_df.copy()
    te_eval_df[f'Target_{h}h'] = te_eval_df[target_col]
    test_m = evaluate_test(te_eval_df, test_pred, h)
    test_m['refit_time'] = t_refit
    test_m['num_trees'] = num_trees

    log_msg(f"    --> [Test 2025] RMSE = {test_m['RMSE']:.4f} | R2 = {test_m['R2']:.4f} | Pk RMSE = {test_m['RMSE_peaks']:.2f} | Pk Bias = {test_m['bias_peaks']:.2f} | Refit Time = {t_refit:.1f}s")

    preds_val_df = pd.DataFrame({
        'Date': val_df_full['Date'],
        'Estacion': val_df_full['Estacion'],
        'h': h,
        'split': 'val',
        'y_true': y_true_val.astype('float32'),
        'y_pred': oof_preds.astype('float32')
    })
    preds_test_df = pd.DataFrame({
        'Date': te_df['Date'],
        'Estacion': te_df['Estacion'],
        'h': h,
        'split': 'test',
        'y_true': te_df[target_col].astype('float32'),
        'y_pred': test_pred.astype('float32')
    })
    preds_all = pd.concat([preds_val_df, preds_test_df], ignore_index=True)

    return val_m, test_m, preds_all


def main():
    log_msg("=" * 80)
    log_msg("  NOCHE2_IMPUT2 — FASE 2: CROSS-FITTING 2024 + REFIT 2020-2024 + NWP MULTI-MODELO")
    log_msg("=" * 80)
    t_all_start = time.time()

    log_msg("\n1. Cargando paneles y precomputando base estática...")
    raw_panel = load_panel()
    imputed_panel = pd.read_parquet(IMPUTED_PANEL_PATH)

    df_planA_imp, chem_imp_cols_raw = build_planA_features(imputed_panel, variant='V4')
    target_cols_to_drop = [c for c in df_planA_imp.columns if c.startswith('Target')]
    df_planA_imp.drop(columns=target_cols_to_drop, inplace=True)
    chem_imp_cols = [c for c in chem_imp_cols_raw if c != 'Estacion' and not c.startswith('Target')]

    yesterday_df = compute_yesterday_stats(imputed_panel)
    regional_df = compute_regional_trends(imputed_panel)
    missing_df = compute_missing_features(raw_panel)

    base_static = pd.merge(df_planA_imp, yesterday_df, on=['Estacion', 'Date'], how='left')
    base_static = pd.merge(base_static, regional_df, on=['Estacion', 'Date'], how='left')
    base_static = pd.merge(base_static, missing_df, on=['Estacion', 'Date'], how='left')

    yesterday_cols = [c for c in yesterday_df.columns if c not in ['Date', 'Estacion']]
    regional_cols = [c for c in regional_df.columns if c not in ['Date', 'Estacion']]
    missing_cols = [c for c in missing_df.columns if c not in ['Date', 'Estacion']]

    del imputed_panel, raw_panel, df_planA_imp, yesterday_df, regional_df, missing_df
    gc.collect()

    ORDERED_HORIZONS = [4, 8, 24, 12, 48, 1]
    VARIANTS = [
        ('imput2_nwp_refit', 'direct', 'Imputacion 1A + NWP multi-modelo + refit 2020-2024'),
        ('imput2_nwp_refit_w', 'weighted', 'Imputacion 1A + NWP + refit 2020-2024 con pesos picos w=1+(y/p90)^2'),
        ('imput2_nwp_resid_refit', 'residual', 'Imputacion 1A + NWP + refit 2020-2024 prediciendo residual persistencia Delta=y-O3(t0)')
    ]

    summary_path = os.path.join(EXP_DIR, 'phase2_summary.csv')
    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        all_records = existing_df.to_dict('records')
        done_set = set(zip(existing_df['horizon'], existing_df['variant']))
    else:
        all_records = []
        done_set = set()

    for h in ORDERED_HORIZONS:
        # Filtrar variantes pendientes
        pending_variants = [
            (v_name, v_type, v_desc) for v_name, v_type, v_desc in VARIANTS
            if (h, v_name) not in done_set or not os.path.exists(os.path.join(PREDS_DIR, f'{v_name}_h{h}.parquet'))
        ]
        if not pending_variants:
            log_msg(f"\nHorizonte h={h} ya completado ({len(VARIANTS)}/{len(VARIANTS)} variantes existentes). Saltando...")
            continue

        log_msg(f"\n{'='*80}\n  INICIANDO HORIZONTE h={h} ({len(pending_variants)} variantes por procesar)\n{'='*80}")
        df_h, feature_cols, honest_cols, pp_cols = build_horizon_dataset(
            h=h,
            base_static=base_static,
            chem_imp_cols=chem_imp_cols,
            yesterday_cols=yesterday_cols,
            regional_cols=regional_cols,
            missing_cols=missing_cols
        )

        for var_name, var_type, var_desc in pending_variants:
            log_msg(f"\n  ---> VARIANTE: {var_name} ({var_desc})")
            t_var = time.time()
            val_m, test_m, preds_df = run_cross_fitting_and_refit(
                h=h,
                df=df_h,
                features=feature_cols,
                honest_cols=honest_cols,
                pp_cols=pp_cols,
                variant_type=var_type,
                seed=SEED,
                iter_mult=1.1
            )
            t_var_dur = time.time() - t_var

            # Guardar parquet oficial
            out_parquet = os.path.join(PREDS_DIR, f'{var_name}_h{h}.parquet')
            preds_df.to_parquet(out_parquet, index=False)
            log_msg(f"    Archivo guardado: {out_parquet} ({len(preds_df)} filas)")

            # Registrar en results_registry.csv
            notes = f"noche2_imput2_phase2: {var_desc} (val=OOF 4-folds 2024, test=refit {test_m['num_trees']} trees, {len(feature_cols)} feats)"
            log_result(
                experiment='noche2_imput2',
                variant=var_name,
                h=h,
                m=test_m,
                notes=notes
            )

            rec = {
                'horizon': h,
                'variant': var_name,
                'n_features': len(feature_cols),
                'val_OOF_RMSE': val_m['RMSE'],
                'val_OOF_R2': val_m['R2'],
                'test_RMSE': test_m['RMSE'],
                'test_R2': test_m['R2'],
                'test_MAE': test_m['MAE'],
                'test_RMSE_peaks': test_m['RMSE_peaks'],
                'test_bias_peaks': test_m['bias_peaks'],
                'skill_vs_clim': test_m.get('skill_vs_clim', 0.0),
                'num_trees': test_m['num_trees'],
                'time_s': round(t_var_dur, 1)
            }
            all_records.append(rec)
            pd.DataFrame(all_records).to_csv(os.path.join(EXP_DIR, 'phase2_summary.csv'), index=False)

        del df_h
        gc.collect()

    log_msg("\n" + "=" * 80)
    log_msg("  FASE 2 COMPLETADA EXITOSAMENTE EN {:.1f} MINUTOS".format((time.time() - t_all_start) / 60))
    log_msg("=" * 80)
    summary_df = pd.DataFrame(all_records)
    log_msg(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
