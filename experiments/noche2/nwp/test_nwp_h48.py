"""
experiments/noche2/nwp/test_nwp_h48.py
============================================
Evalúa NWP en h=48 (lead time honesto previous_day2):
- D_nwp_noise
- E_nwp_w
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, '.')
from experiments.common import evaluate_test, metrics, SEED, MODEL_STATIONS

CACHE_DIR = 'experiments/noche/v3_base/cache'
DATA_DIR = 'experiments/noche2/nwp/data'
OPTUNA_PATH = 'experiments/planA_quickwins/best_params_optuna.json'

with open(OPTUNA_PATH, 'r') as f:
    BEST_OPTUNA = json.load(f)

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'seed': SEED,
    'verbose': -1,
    'num_threads': 3,
    'feature_pre_filter': False,
    'importance_type': 'gain',
    **BEST_OPTUNA
}


def load_and_prepare(h: int = 48):
    print(f"Cargando cache base y agregando features NWP para h={h}...", flush=True)
    df_base = pd.read_parquet(os.path.join(CACHE_DIR, f'features_h{h}.parquet'))

    t0_cols = [c for c in df_base.columns if c.startswith('meteo_t0_')]
    honest_cols = [c for c in df_base.columns if c.startswith('honest_meteo_')]
    pp_cols = [c for c in df_base.columns if c.startswith('pp_meteo_')]
    exclude = {'Date', 'Estacion', 'split', 'Target', f'Target_{h}h'} | set(t0_cols) | set(honest_cols) | set(pp_cols)
    chem_cols = [c for c in df_base.columns if c not in exclude]

    nwp_dfs = []
    for st in MODEL_STATIONS:
        p = os.path.join(DATA_DIR, f'nwp_multimodel_{st}.parquet')
        if os.path.exists(p):
            nwp_dfs.append(pd.read_parquet(p))
    nwp_all = pd.concat(nwp_dfs, ignore_index=True).sort_values(['Estacion', 'Date'])

    suf = '_previous_day1' if h <= 24 else '_previous_day2'
    t_ec = f'temperature_2m{suf}_ecmwf_ifs025'
    t_gfs = f'temperature_2m{suf}_gfs_seamless'
    t_icon = f'temperature_2m{suf}_icon_seamless'

    rad_ec = f'shortwave_radiation{suf}_ecmwf_ifs025'
    rad_gfs = f'shortwave_radiation{suf}_gfs_seamless'
    rad_icon = f'shortwave_radiation{suf}_icon_seamless'

    ws_ec = f'wind_speed_10m{suf}_ecmwf_ifs025'
    ws_gfs = f'wind_speed_10m{suf}_gfs_seamless'
    ws_icon = f'wind_speed_10m{suf}_icon_seamless'

    cape_col = f'cape{suf}_gfs_seamless'
    gust_gfs = f'wind_gusts_10m{suf}_gfs_seamless'
    gust_icon = f'wind_gusts_10m{suf}_icon_seamless'

    vpd_ec = f'vapour_pressure_deficit{suf}_ecmwf_ifs025'
    vpd_gfs = f'vapour_pressure_deficit{suf}_gfs_seamless'
    vpd_icon = f'vapour_pressure_deficit{suf}_icon_seamless'

    nwp_all['nwp_temp_mean'] = nwp_all[[t_ec, t_gfs, t_icon]].mean(axis=1)
    nwp_all['nwp_temp_std'] = nwp_all[[t_ec, t_gfs, t_icon]].std(axis=1).fillna(0.0)

    nwp_all['nwp_rad_mean'] = nwp_all[[rad_ec, rad_gfs, rad_icon]].mean(axis=1)
    nwp_all['nwp_rad_std'] = nwp_all[[rad_ec, rad_gfs, rad_icon]].std(axis=1).fillna(0.0)

    nwp_all['nwp_ws_mean'] = nwp_all[[ws_ec, ws_gfs, ws_icon]].mean(axis=1)
    nwp_all['nwp_ws_std'] = nwp_all[[ws_ec, ws_gfs, ws_icon]].std(axis=1).fillna(0.0)

    nwp_all['nwp_vpd_mean'] = nwp_all[[vpd_ec, vpd_gfs, vpd_icon]].mean(axis=1)
    nwp_all['nwp_cape'] = nwp_all[cape_col].fillna(0.0)
    nwp_all['nwp_wind_gust'] = nwp_all[[gust_gfs, gust_icon]].max(axis=1).fillna(nwp_all['nwp_ws_mean'])

    nwp_features_list = []
    nwp_cols = ['nwp_temp_mean', 'nwp_temp_std', 'nwp_rad_mean', 'nwp_rad_std', 
                'nwp_ws_mean', 'nwp_ws_std', 'nwp_vpd_mean', 'nwp_cape', 'nwp_wind_gust']

    for st, g in nwp_all.groupby('Estacion', sort=False):
        g_sub = g[['Date', 'Estacion'] + nwp_cols].copy()
        for c in nwp_cols:
            g_sub[c] = g_sub[c].shift(-h).astype('float32')
        nwp_features_list.append(g_sub)
    nwp_shifted = pd.concat(nwp_features_list, ignore_index=True)

    df_merged = pd.merge(df_base, nwp_shifted, on=['Date', 'Estacion'], how='left')

    is_train = df_merged['split'] == 'train'
    df_merged.loc[is_train, 'nwp_temp_mean'] = df_merged.loc[is_train, 'honest_meteo_target_temp']
    df_merged.loc[is_train, 'nwp_rad_mean'] = df_merged.loc[is_train, 'honest_meteo_target_rad']
    df_merged.loc[is_train, 'nwp_ws_mean'] = df_merged.loc[is_train, 'honest_meteo_target_ws']

    # MOS Causal
    df_merged['hour'] = df_merged['Date'].dt.hour
    train_df = df_merged[is_train]
    bias_t_series = train_df.groupby(['Estacion', 'hour'], observed=False)['TOUT_lag_0'].mean() - train_df.groupby(['Estacion', 'hour'], observed=False)['meteo_t0_temp'].mean()
    bias_ws_series = train_df.groupby(['Estacion', 'hour'], observed=False)['wind_speed_t0'].mean() - train_df.groupby(['Estacion', 'hour'], observed=False)['meteo_t0_ws'].mean()

    target_hour = (df_merged['hour'] + h) % 24
    df_merged['target_hour'] = target_hour

    bias_df = bias_t_series.reset_index().rename(columns={'hour': 'target_hour', 0: 'mos_bias_temp'})
    bias_ws_df = bias_ws_series.reset_index().rename(columns={'hour': 'target_hour', 0: 'mos_bias_ws'})

    df_merged = pd.merge(df_merged, bias_df, on=['Estacion', 'target_hour'], how='left')
    df_merged = pd.merge(df_merged, bias_ws_df, on=['Estacion', 'target_hour'], how='left')

    df_merged['nwp_mos_temp'] = (df_merged['honest_meteo_target_temp'] + df_merged['mos_bias_temp'].fillna(0.0)).astype('float32')
    df_merged['nwp_mos_ws'] = np.clip(df_merged['honest_meteo_target_ws'] + df_merged['mos_bias_ws'].fillna(0.0), 0.1, None).astype('float32')
    df_merged['nwp_mos_photothermal'] = (df_merged['nwp_mos_temp'] * df_merged['honest_meteo_target_rad']).astype('float32')

    df_merged['Estacion'] = df_merged['Estacion'].astype('category')
    return df_merged, chem_cols, t0_cols, honest_cols, pp_cols, nwp_cols


def inject_all_noise(train_df: pd.DataFrame, val_df: pd.DataFrame, honest_cols: list[str], pp_cols: list[str], nwp_cols: list[str], seed: int = 42) -> pd.DataFrame:
    train_noisy = train_df.copy()
    rng = np.random.default_rng(seed)

    val_hour = val_df['Date'].dt.hour.values
    tr_hour = train_noisy['Date'].dt.hour.values

    for h_col in honest_cols:
        suffix = h_col.replace('honest_meteo_', '')
        pp_col = f'pp_meteo_{suffix}'
        if pp_col not in val_df.columns:
            continue
        val_diff = (val_df[h_col] - val_df[pp_col]).values
        valid_mask = np.isfinite(val_diff)

        sampled_noise = np.zeros(len(train_noisy), dtype='float32')
        for hr in range(24):
            hr_val_mask = valid_mask & (val_hour == hr)
            hr_tr_mask = (tr_hour == hr)
            n_tr = int(hr_tr_mask.sum())
            if n_tr == 0:
                continue
            pool = val_diff[hr_val_mask]
            if len(pool) > 0:
                sampled_noise[hr_tr_mask] = rng.choice(pool, size=n_tr, replace=True)

        train_noisy[h_col] = (train_noisy[h_col] + sampled_noise).astype('float32')
        if 'rh' in h_col or 'cloud' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, 100.0)
        elif 'rad' in h_col or 'ws' in h_col or 'precip' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, None)

    for col in ['nwp_temp_std', 'nwp_rad_std', 'nwp_ws_std', 'nwp_vpd_mean', 'nwp_cape', 'nwp_wind_gust']:
        if col in val_df.columns:
            val_vals = val_df[col].values
            valid_mask = np.isfinite(val_vals)
            sampled_vals = np.zeros(len(train_noisy), dtype='float32')
            for hr in range(24):
                hr_val_mask = valid_mask & (val_hour == hr)
                hr_tr_mask = (tr_hour == hr)
                n_tr = int(hr_tr_mask.sum())
                if n_tr == 0:
                    continue
                pool = val_vals[hr_val_mask]
                if len(pool) > 0:
                    sampled_vals[hr_tr_mask] = rng.choice(pool, size=n_tr, replace=True)
            train_noisy[col] = sampled_vals.astype('float32')

    if 'nwp_mos_temp' in train_noisy.columns and 'honest_meteo_target_rad' in train_noisy.columns:
        train_noisy['nwp_mos_photothermal'] = (train_noisy['nwp_mos_temp'] * train_noisy['honest_meteo_target_rad']).astype('float32')

    return train_noisy


def main():
    h = 48
    df_merged, chem_cols, t0_cols, honest_cols, pp_cols, nwp_cols = load_and_prepare(h)

    mos_cols = ['nwp_mos_temp', 'nwp_mos_ws', 'nwp_mos_photothermal']
    nwp_feature_set = mos_cols + nwp_cols
    all_features = ['Estacion'] + chem_cols + t0_cols + honest_cols + nwp_feature_set

    tr_df = df_merged[(df_merged['split'] == 'train') & df_merged['Target'].notna()].copy().reset_index(drop=True)
    va_df = df_merged[(df_merged['split'] == 'val') & df_merged['Target'].notna()].copy().reset_index(drop=True)
    te_df = df_merged[(df_merged['split'] == 'test') & df_merged['Target'].notna()].copy().reset_index(drop=True)

    # 1. D_nwp_noise
    print("\n--- Ejecutando D_nwp_noise para h=48 ---", flush=True)
    tr_noisy = inject_all_noise(tr_df, va_df, honest_cols, pp_cols, nwp_cols, seed=SEED)

    lgb_tr_d = lgb.Dataset(tr_noisy[all_features], tr_noisy['Target'].values, categorical_feature=['Estacion'], free_raw_data=False)
    lgb_va = lgb.Dataset(va_df[all_features], va_df['Target'].values, reference=lgb_tr_d, categorical_feature=['Estacion'], free_raw_data=False)

    t0 = time.time()
    model_d = lgb.train(
        LGB_PARAMS,
        lgb_tr_d,
        num_boost_round=2000,
        valid_sets=[lgb_va],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]
    )
    t_d = time.time() - t0

    y_va_d = model_d.predict(va_df[all_features], num_iteration=model_d.best_iteration)
    y_te_d = model_d.predict(te_df[all_features], num_iteration=model_d.best_iteration)

    m_va_d = metrics(va_df['Target'].values, y_va_d)
    m_te_d = evaluate_test(te_df, y_te_d, h)

    print(f"[D_nwp_noise h=48] Best Iter: {model_d.best_iteration}, Tiempo: {t_d:.1f}s")
    print(f"  Val  RMSE: {m_va_d['RMSE']:.4f} | MAE: {m_va_d['MAE']:.4f} | R2: {m_va_d['R2']:.4f}")
    print(f"  Test RMSE: {m_te_d['RMSE']:.4f} | MAE: {m_te_d['MAE']:.4f} | R2: {m_te_d['R2']:.4f} | RMSE_p95: {m_te_d['RMSE_peaks']:.2f} | Bias_p95: {m_te_d['bias_peaks']:.2f}")

    # 2. E_nwp_w
    print("\n--- Ejecutando E_nwp_w para h=48 ---", flush=True)
    p90 = np.percentile(tr_noisy['Target'].values, 90)
    sample_weights = 1.0 + (tr_noisy['Target'].values / p90) ** 2

    lgb_tr_e = lgb.Dataset(tr_noisy[all_features], tr_noisy['Target'].values, categorical_feature=['Estacion'], weight=sample_weights, free_raw_data=False)

    t0 = time.time()
    model_e = lgb.train(
        LGB_PARAMS,
        lgb_tr_e,
        num_boost_round=2000,
        valid_sets=[lgb_va],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]
    )
    t_e = time.time() - t0

    y_va_e = model_e.predict(va_df[all_features], num_iteration=model_e.best_iteration)
    y_te_e = model_e.predict(te_df[all_features], num_iteration=model_e.best_iteration)

    m_va_e = metrics(va_df['Target'].values, y_va_e)
    m_te_e = evaluate_test(te_df, y_te_e, h)

    print(f"[E_nwp_w h=48] Best Iter: {model_e.best_iteration}, Tiempo: {t_e:.1f}s")
    print(f"  Val  RMSE: {m_va_e['RMSE']:.4f} | MAE: {m_va_e['MAE']:.4f} | R2: {m_va_e['R2']:.4f}")
    print(f"  Test RMSE: {m_te_e['RMSE']:.4f} | MAE: {m_te_e['MAE']:.4f} | R2: {m_te_e['R2']:.4f} | RMSE_p95: {m_te_e['RMSE_peaks']:.2f} | Bias_p95: {m_te_e['bias_peaks']:.2f}")

    print("\n" + "=" * 85)
    print(f"  TABLA FINAL COMPARATIVA h={h}")
    print("=" * 85)
    print(f"{'Modelo / Variante':<25} {'Val RMSE':<10} {'Test RMSE':<10} {'Test R2':<8} {'Test p95':<10} {'Bias p95':<10}")
    print("-" * 85)
    print(f"{'v3 D_honest_noise (ref)':<25} {'11.1160':<10} {'11.2340':<10} {'0.7160':<8} {'27.03':<10} {'-22.51':<10}")
    print(f"{'v3 E_honest_w (ref)':<25} {'--':<10} {'11.1700':<10} {'0.7190':<8} {'25.16':<10} {'-20.18':<10}")
    print(f"{'NWP D_nwp_noise (nuevo)':<25} {m_va_d['RMSE']:<10.4f} {m_te_d['RMSE']:<10.4f} {m_te_d['R2']:<8.4f} {m_te_d['RMSE_peaks']:<10.2f} {m_te_d['bias_peaks']:<10.2f}")
    print(f"{'NWP E_nwp_w (nuevo)':<25} {m_va_e['RMSE']:<10.4f} {m_te_e['RMSE']:<10.4f} {m_te_e['R2']:<8.4f} {m_te_e['RMSE_peaks']:<10.2f} {m_te_e['bias_peaks']:<10.2f}")
    print("=" * 85)


if __name__ == '__main__':
    main()
