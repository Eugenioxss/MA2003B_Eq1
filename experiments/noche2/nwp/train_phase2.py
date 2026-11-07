"""
experiments/noche2/nwp/train_phase2.py
======================================
Fase 2 del experimento noche2_nwp:
Aporta diversidad estructural de algoritmos al ensamble maestro mediante:
1. XGBoost (tree_method="hist", n_jobs=4)
   - Sin pesos: xgb_nwp_refit_h<h>.parquet
   - Con pesos de picos: xgb_nwp_refit_w_h<h>.parquet
2. LightGBM ExtraTrees (extra_trees=True, colsample_bytree=0.5, num_threads=4)
   - et_nwp_refit_h<h>.parquet

Esquema riguroso:
- 4-fold cross-fitting por trimestres (Q1..Q4) de 2024 para generar val OOF 100% honesto
- Refit completo 2020-2024 (con bagging de 2 semillas y n_estimators = int(median_best_iter * 1.1)) para test 2025
- Indicador is_nwp=1 en 2024-2025 e is_nwp=0 en 2020-2023 con ruido/dispersión empírica calibrada
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from functools import partial
print = partial(print, flush=True)

sys.path.insert(0, '.')
from experiments.common import evaluate_test, metrics, log_result, SEED, MODEL_STATIONS

CACHE_DIR = 'experiments/noche/v3_base/cache'
NWP_DIR = 'experiments/noche2/nwp'
PREDS_DIR = os.path.join(NWP_DIR, 'preds')
os.makedirs(PREDS_DIR, exist_ok=True)

OPTUNA_PATH = 'experiments/planA_quickwins/best_params_optuna.json'
with open(OPTUNA_PATH, 'r') as f:
    BEST_OPTUNA = json.load(f)


def load_combined_data(h: int) -> tuple[pd.DataFrame, list[str]]:
    """Carga cache base + features_nwp_h<h>.parquet y devuelve dataframe con columnas."""
    cache_path = os.path.join(CACHE_DIR, f'features_h{h}.parquet')
    nwp_path = os.path.join(NWP_DIR, f'features_nwp_h{h}.parquet')

    df_base = pd.read_parquet(cache_path)
    df_nwp = pd.read_parquet(nwp_path)

    df = pd.merge(df_base, df_nwp, on=['Date', 'Estacion'], how='inner')
    df['Estacion'] = df['Estacion'].astype('category')
    df['quarter'] = df['Date'].dt.quarter

    t0_cols = [c for c in df.columns if c.startswith('meteo_t0_')]
    honest_cols = [c for c in df.columns if c.startswith('honest_meteo_')]
    pp_cols = [c for c in df.columns if c.startswith('pp_meteo_')]
    nwp_cols = [c for c in df_nwp.columns if c not in ['Date', 'Estacion']]
    exclude = {'Date', 'Estacion', 'split', 'Target', f'Target_{h}h', 'quarter'} | set(t0_cols) | set(honest_cols) | set(pp_cols) | set(nwp_cols)
    chem_cols = [c for c in df.columns if c not in exclude]

    features = ['Estacion'] + chem_cols + t0_cols + honest_cols + nwp_cols + ['is_nwp']
    return df, features, honest_cols, pp_cols, nwp_cols


def inject_noise_and_spread_calib(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    honest_cols: list[str],
    pp_cols: list[str],
    seed: int = 42
) -> pd.DataFrame:
    """Inyecta ruido empírico calibrado estrictamente sobre calib_df."""
    train_noisy = train_df.copy()
    rng = np.random.default_rng(seed)

    calib_hour = calib_df['Date'].dt.hour.values
    tr_hour = train_noisy['Date'].dt.hour.values

    # 1. Ruido en honest_meteo
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
            pool = val_diff[hr_calib_mask]
            if len(pool) > 0:
                sampled_noise[hr_tr_mask] = rng.choice(pool, size=n_tr, replace=True)

        train_noisy[h_col] = (train_noisy[h_col] + sampled_noise).astype('float32')
        if 'rh' in h_col or 'cloud' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, 100.0)
        elif 'rad' in h_col or 'ws' in h_col or 'precip' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, None)

    # 2. Inyección de spreads y estabilidad condicionada a la hora solar
    spread_cols = [c for c in ['nwp_temp_std', 'nwp_rad_std', 'nwp_ws_std', 'nwp_vpd_mean', 'nwp_cape', 'nwp_wind_gust'] if c in calib_df.columns]
    for col in spread_cols:
        val_vals = calib_df[col].values
        valid_mask = np.isfinite(val_vals)
        sampled_vals = np.zeros(len(train_noisy), dtype='float32')
        for hr in range(24):
            hr_calib_mask = valid_mask & (calib_hour == hr)
            hr_tr_mask = (tr_hour == hr)
            n_tr = int(hr_tr_mask.sum())
            if n_tr == 0:
                continue
            pool = val_vals[hr_calib_mask]
            if len(pool) > 0:
                sampled_vals[hr_tr_mask] = rng.choice(pool, size=n_tr, replace=True)
        train_noisy[col] = sampled_vals.astype('float32')

    if 'nwp_mos_temp' in train_noisy.columns and 'honest_meteo_target_rad' in train_noisy.columns:
        train_noisy['nwp_mos_photothermal'] = (train_noisy['nwp_mos_temp'] * train_noisy['honest_meteo_target_rad']).astype('float32')

    return train_noisy


def run_cross_fitting_and_refit(
    model_type: str,  # 'xgb', 'xgb_w', 'et'
    h: int,
    df: pd.DataFrame,
    features: list[str],
    honest_cols: list[str],
    pp_cols: list[str]
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Ejecuta el protocolo completo:
    1. 4-fold cross-fitting por trimestres de 2024 para val OOF
    2. Refit con 2020-2024 para test 2025
    """
    target_col = 'Target'
    use_peak_weights = (model_type == 'xgb_w')

    train_base_mask = (df['split'] == 'train') & df[target_col].notna()
    val_full_mask = (df['split'] == 'val') & df[target_col].notna()
    test_mask = (df['split'] == 'test') & df[target_col].notna()

    tr_base = df[train_base_mask].copy().reset_index(drop=True)
    val_full = df[val_full_mask].copy().reset_index(drop=True)
    te_full = df[test_mask].copy().reset_index(drop=True)

    oof_preds = np.zeros(len(val_full), dtype='float32')
    best_iters = []

    print(f"  [{model_type} h={h}] Iniciando 4-fold cross-fitting en 2024 (n_val={len(val_full)})...")

    for q in [1, 2, 3, 4]:
        t_fold = time.time()
        q_val_idx = (val_full['quarter'] == q).values
        q_tr_idx = (val_full['quarter'] != q).values

        # 1. Calibrar ruido estrictamente sin el trimestre validado
        calib_df = val_full[q_tr_idx]
        tr_part_a = inject_noise_and_spread_calib(tr_base, calib_df, honest_cols, pp_cols, seed=SEED)
        tr_part_a['is_nwp'] = np.float32(0.0)

        tr_part_b = val_full[q_tr_idx].copy()
        tr_part_b['is_nwp'] = np.float32(1.0)

        tr_fold = pd.concat([tr_part_a, tr_part_b], ignore_index=True)

        val_fold = val_full[q_val_idx].copy()
        val_fold['is_nwp'] = np.float32(1.0)

        # Pesos de muestra
        weights = None
        if use_peak_weights:
            y_fold = tr_fold[target_col].values
            p90 = np.percentile(y_fold, 90.0)
            weights = (1.0 + (y_fold / p90) ** 2).astype('float32')

        X_tr = tr_fold[features]
        y_tr = tr_fold[target_col].values
        X_va = val_fold[features]
        y_va = val_fold[target_col].values

        if model_type in ['xgb', 'xgb_w']:
            model = xgb.XGBRegressor(
                tree_method='hist',
                enable_categorical=True,
                n_estimators=1500,
                learning_rate=0.06,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=SEED,
                n_jobs=4,
                early_stopping_rounds=40,
                eval_metric='rmse'
            )
            model.fit(
                X_tr, y_tr,
                sample_weight=weights,
                eval_set=[(X_va, y_va)],
                verbose=False
            )
            b_iter = model.best_iteration
            pred_va = model.predict(X_va)

        elif model_type == 'et':
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'seed': SEED,
                'verbose': -1,
                'num_threads': 4,
                'extra_trees': True,
                'colsample_bytree': 0.5,
                'subsample': 0.8,
                'subsample_freq': 1,
                'learning_rate': 0.05,
                'num_leaves': 63,
                'min_child_samples': 50
            }
            lgb_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=['Estacion'], weight=weights, free_raw_data=False)
            lgb_va = lgb.Dataset(X_va, y_va, reference=lgb_tr, categorical_feature=['Estacion'], free_raw_data=False)
            model = lgb.train(
                lgb_params,
                lgb_tr,
                num_boost_round=1500,
                valid_sets=[lgb_va],
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)]
            )
            b_iter = model.best_iteration
            pred_va = model.predict(X_va, num_iteration=b_iter)

        best_iters.append(b_iter)
        oof_preds[q_val_idx] = pred_va
        fold_rmse = float(np.sqrt(np.mean((y_va - pred_va) ** 2)))
        print(f"    Fold Q{q} | BestIter={b_iter:4d} | RMSE={fold_rmse:6.3f} | Tiempo={time.time()-t_fold:4.1f}s")

    # Métricas Val OOF
    y_true_val = val_full[target_col].values
    m_va = metrics(y_true_val, oof_preds)
    m_va['median_best_iter'] = int(np.median(best_iters))
    print(f"  [{model_type} h={h}] Val OOF RMSE: {m_va['RMSE']:.4f} | MAE: {m_va['MAE']:.4f} | R2: {m_va['R2']:.4f} | Mediana Iter: {m_va['median_best_iter']}")

    # -------------------------------------------------------------
    # REFIT CON 2020-2024 PARA TEST 2025
    # -------------------------------------------------------------
    num_trees = int(round(np.median(best_iters) * 1.1))
    print(f"  [{model_type} h={h}] Reentrenando 2020-2024 para test 2025 con {num_trees} árboles...")

    tr_full_a = inject_noise_and_spread_calib(tr_base, val_full, honest_cols, pp_cols, seed=SEED)
    tr_full_a['is_nwp'] = np.float32(0.0)

    tr_full_b = val_full.copy()
    tr_full_b['is_nwp'] = np.float32(1.0)

    tr_refit = pd.concat([tr_full_a, tr_full_b], ignore_index=True)
    te_full['is_nwp'] = np.float32(1.0)

    weights_refit = None
    if use_peak_weights:
        y_refit = tr_refit[target_col].values
        p90 = np.percentile(y_refit, 90.0)
        weights_refit = (1.0 + (y_refit / p90) ** 2).astype('float32')

    X_refit = tr_refit[features]
    y_refit = tr_refit[target_col].values
    X_test = te_full[features]
    y_test = te_full[target_col].values

    # Bagging de 2 semillas (SEED y SEED+1) para robustez
    test_preds_seeds = []
    seeds = [SEED, SEED + 1]

    for s in seeds:
        if model_type in ['xgb', 'xgb_w']:
            m_refit = xgb.XGBRegressor(
                tree_method='hist',
                enable_categorical=True,
                n_estimators=num_trees,
                learning_rate=0.06,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=s,
                n_jobs=4
            )
            m_refit.fit(X_refit, y_refit, sample_weight=weights_refit, verbose=False)
            test_preds_seeds.append(m_refit.predict(X_test))

        elif model_type == 'et':
            lgb_params_refit = {
                'objective': 'regression',
                'metric': 'rmse',
                'seed': s,
                'verbose': -1,
                'num_threads': 4,
                'extra_trees': True,
                'colsample_bytree': 0.5,
                'subsample': 0.8,
                'subsample_freq': 1,
                'learning_rate': 0.05,
                'num_leaves': 63,
                'min_child_samples': 50
            }
            lgb_tr_refit = lgb.Dataset(X_refit, y_refit, categorical_feature=['Estacion'], weight=weights_refit, free_raw_data=False)
            m_refit = lgb.train(lgb_params_refit, lgb_tr_refit, num_boost_round=num_trees)
            test_preds_seeds.append(m_refit.predict(X_test, num_iteration=num_trees))

    y_pred_te = np.mean(test_preds_seeds, axis=0).astype('float32')

    # Evaluar con evaluate_test de common
    te_eval_df = te_full.copy()
    te_eval_df[f'Target_{h}h'] = te_eval_df[target_col]
    m_te = evaluate_test(te_eval_df, y_pred_te, h)

    print(f"  [{model_type} h={h}] Test RMSE: {m_te['RMSE']:.4f} | R2: {m_te['R2']:.4f} | Picos RMSE: {m_te['RMSE_peaks']:.2f} | Bias picos: {m_te['bias_peaks']:.2f}")

    # Dataframe de predicciones combinadas
    preds_val = pd.DataFrame({
        'Date': val_full['Date'],
        'Estacion': val_full['Estacion'],
        'h': h,
        'split': 'val',
        'y_true': y_true_val.astype('float32'),
        'y_pred': oof_preds.astype('float32')
    })
    preds_test = pd.DataFrame({
        'Date': te_full['Date'],
        'Estacion': te_full['Estacion'],
        'h': h,
        'split': 'test',
        'y_true': y_test.astype('float32'),
        'y_pred': y_pred_te.astype('float32')
    })
    preds_all = pd.concat([preds_val, preds_test], ignore_index=True)

    return m_va, m_te, preds_all


def main():
    horizons = [24, 48, 8, 12, 1, 4]  # En orden de prioridad
    summary_records = []

    print("=" * 80)
    print("  FASE 2: DIVERSIDAD ESTRUCTURAL (XGBoost + LightGBM ExtraTrees)")
    print("  Esquema: 4-Fold OOF 2024 + Refit 2020-2024 con is_nwp y NWP multi-modelo")
    print("  Recursos: n_jobs=4 / num_threads=4")
    print("=" * 80)

    for h in horizons:
        t_h_start = time.time()
        print(f"\n{'='*70}\n  INICIANDO HORIZONTE h={h}\n{'='*70}")
        df, features, honest_cols, pp_cols, nwp_cols = load_combined_data(h)

        # 1. XGBoost sin pesos
        preds_xgb_path = os.path.join(PREDS_DIR, f'xgb_nwp_refit_h{h}.parquet')
        if os.path.exists(preds_xgb_path):
            print(f"  [xgb h={h}] Ya existe {preds_xgb_path}. Cargando...", flush=True)
            p_df = pd.read_parquet(preds_xgb_path)
            pv = p_df[p_df['split'] == 'val']
            pt = p_df[p_df['split'] == 'test']
            va_xgb = metrics(pv['y_true'].values, pv['y_pred'].values)
            pt_eval = pt.copy()
            pt_eval[f'Target_{h}h'] = pt_eval['y_true']
            te_xgb = evaluate_test(pt_eval, pt['y_pred'].values, h)
        else:
            va_xgb, te_xgb, preds_xgb = run_cross_fitting_and_refit('xgb', h, df, features, honest_cols, pp_cols)
            preds_xgb.to_parquet(preds_xgb_path, index=False)
            log_result('noche2_nwp', 'xgb_nwp_refit', h, te_xgb, notes='XGBoost hist 4-fold OOF + 2024 refit con NWP')

        # 2. XGBoost con pesos de picos
        preds_xgb_w_path = os.path.join(PREDS_DIR, f'xgb_nwp_refit_w_h{h}.parquet')
        if os.path.exists(preds_xgb_w_path):
            print(f"  [xgb_w h={h}] Ya existe {preds_xgb_w_path}. Cargando...", flush=True)
            p_df = pd.read_parquet(preds_xgb_w_path)
            pv = p_df[p_df['split'] == 'val']
            pt = p_df[p_df['split'] == 'test']
            va_xgb_w = metrics(pv['y_true'].values, pv['y_pred'].values)
            pt_eval = pt.copy()
            pt_eval[f'Target_{h}h'] = pt_eval['y_true']
            te_xgb_w = evaluate_test(pt_eval, pt['y_pred'].values, h)
        else:
            va_xgb_w, te_xgb_w, preds_xgb_w = run_cross_fitting_and_refit('xgb_w', h, df, features, honest_cols, pp_cols)
            preds_xgb_w.to_parquet(preds_xgb_w_path, index=False)
            log_result('noche2_nwp', 'xgb_nwp_refit_w', h, te_xgb_w, notes='XGBoost hist w=1+(y/p90)^2 4-fold OOF + 2024 refit con NWP')

        # 3. LightGBM ExtraTrees
        preds_et_path = os.path.join(PREDS_DIR, f'et_nwp_refit_h{h}.parquet')
        if os.path.exists(preds_et_path):
            print(f"  [et h={h}] Ya existe {preds_et_path}. Cargando...", flush=True)
            p_df = pd.read_parquet(preds_et_path)
            pv = p_df[p_df['split'] == 'val']
            pt = p_df[p_df['split'] == 'test']
            va_et = metrics(pv['y_true'].values, pv['y_pred'].values)
            pt_eval = pt.copy()
            pt_eval[f'Target_{h}h'] = pt_eval['y_true']
            te_et = evaluate_test(pt_eval, pt['y_pred'].values, h)
        else:
            va_et, te_et, preds_et = run_cross_fitting_and_refit('et', h, df, features, honest_cols, pp_cols)
            preds_et.to_parquet(preds_et_path, index=False)
            log_result('noche2_nwp', 'et_nwp_refit', h, te_et, notes='LightGBM ExtraTrees 4-fold OOF + 2024 refit con NWP')

        rec = {
            'h': h,
            'xgb_val_rmse': va_xgb['RMSE'], 'xgb_test_rmse': te_xgb['RMSE'], 'xgb_test_r2': te_xgb['R2'],
            'xgb_w_val_rmse': va_xgb_w['RMSE'], 'xgb_w_test_rmse': te_xgb_w['RMSE'], 'xgb_w_test_r2': te_xgb_w['R2'],
            'xgb_w_peaks_rmse': te_xgb_w['RMSE_peaks'], 'xgb_w_peaks_bias': te_xgb_w['bias_peaks'],
            'et_val_rmse': va_et['RMSE'], 'et_test_rmse': te_et['RMSE'], 'et_test_r2': te_et['R2'],
            'horizon_time_s': time.time() - t_h_start
        }
        summary_records.append(rec)

        # Actualizar CSV de resumen parcial tras cada horizonte
        pd.DataFrame(summary_records).sort_values('h').to_csv(os.path.join(NWP_DIR, 'metrics_phase2.csv'), index=False)
        print(f"\n[Completado h={h} en {round((time.time() - t_h_start)/60, 1)} min]")

    print("\n" + "=" * 90)
    print("  TABLA GLOBAL FINAL FASE 2: NOCHE2_NWP")
    print("=" * 90)
    res_df = pd.DataFrame(summary_records).sort_values('h')
    print(res_df.to_string(index=False))
    print("=" * 90)


if __name__ == '__main__':
    main()
