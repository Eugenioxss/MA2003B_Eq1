"""
experiments/noche2/nwp/train_all_nwp.py
=======================================
Pipeline completo de entrenamiento, evaluación y exportación de pronósticos NWP
para horizontes h in [24, 48, 8, 12, 1, 4]:
- Features: Multi-modelo (ECMWF IFS 0.25, GFS Seamless, ICON Seamless),
  Estabilidad atmosférica (CAPE, ráfagas, VPD) y MOS Causal (TOUT, WSR, fototermia)
- Variantes:
  1. D_nwp_noise: Estimador de media condicional con inyección de ruido/dispersión empírica NWP
  2. E_nwp_w: Especialista en picos con ponderación w = 1 + (y/p90)^2
- Exporta:
  - features_nwp_h<h>.parquet (reutilizable para ensambles y DL)
  - preds/D_nwp_noise_h<h>.parquet y preds/E_nwp_w_h<h>.parquet
  - metrics_summary.csv
  - Registra en experiments/results_registry.csv
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import partial
print = partial(print, flush=True)

sys.path.insert(0, '.')
from experiments.common import evaluate_test, metrics, log_result, SEED, MODEL_STATIONS

CACHE_DIR = 'experiments/noche/v3_base/cache'
DATA_DIR = 'experiments/noche2/nwp/data'
BASE_DIR = 'experiments/noche2/nwp'
PREDS_DIR = os.path.join(BASE_DIR, 'preds')
os.makedirs(PREDS_DIR, exist_ok=True)

OPTUNA_PATH = 'experiments/planA_quickwins/best_params_optuna.json'
with open(OPTUNA_PATH, 'r') as f:
    BEST_OPTUNA = json.load(f)

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'seed': SEED,
    'verbose': -1,
    'num_threads': 3,  # Contrato: 3 hilos LightGBM
    'feature_pre_filter': False,
    'importance_type': 'gain',
    **BEST_OPTUNA
}

NWP_COLS_EXPORT = [
    'nwp_temp_mean', 'nwp_temp_std', 'nwp_rad_mean', 'nwp_rad_std',
    'nwp_ws_mean', 'nwp_ws_std', 'nwp_vpd_mean', 'nwp_cape', 'nwp_wind_gust',
    'nwp_mos_temp', 'nwp_mos_ws', 'nwp_mos_photothermal'
]


def build_and_export_nwp_features(df_base: pd.DataFrame, h: int) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str], list[str]]:
    """Construye features NWP honestas y exporta features_nwp_h<h>.parquet."""
    print(f"\n[{h}h] Procesando features meteorológicas NWP y MOS...", flush=True)
    t0 = time.time()

    # Columnas base del cache
    t0_cols = [c for c in df_base.columns if c.startswith('meteo_t0_')]
    honest_cols = [c for c in df_base.columns if c.startswith('honest_meteo_')]
    pp_cols = [c for c in df_base.columns if c.startswith('pp_meteo_')]
    exclude = {'Date', 'Estacion', 'split', 'Target', f'Target_{h}h'} | set(t0_cols) | set(honest_cols) | set(pp_cols)
    chem_cols = [c for c in df_base.columns if c not in exclude]

    # Cargar datos NWP multi-modelo descargados (2024-2025)
    nwp_dfs = []
    for st in MODEL_STATIONS:
        p = os.path.join(DATA_DIR, f'nwp_multimodel_{st}.parquet')
        if os.path.exists(p):
            nwp_dfs.append(pd.read_parquet(p))
    nwp_all = pd.concat(nwp_dfs, ignore_index=True).sort_values(['Estacion', 'Date'])

    # Selección estricta de lead time honesto:
    # Para h <= 24: lead 24-47h (previous_day1)
    # Para h = 48: lead 48-71h (previous_day2)
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

    # Multi-modelo consensus y spread
    nwp_all['nwp_temp_mean'] = nwp_all[[t_ec, t_gfs, t_icon]].mean(axis=1)
    nwp_all['nwp_temp_std'] = nwp_all[[t_ec, t_gfs, t_icon]].std(axis=1).fillna(0.0)

    nwp_all['nwp_rad_mean'] = nwp_all[[rad_ec, rad_gfs, rad_icon]].mean(axis=1)
    nwp_all['nwp_rad_std'] = nwp_all[[rad_ec, rad_gfs, rad_icon]].std(axis=1).fillna(0.0)

    nwp_all['nwp_ws_mean'] = nwp_all[[ws_ec, ws_gfs, ws_icon]].mean(axis=1)
    nwp_all['nwp_ws_std'] = nwp_all[[ws_ec, ws_gfs, ws_icon]].std(axis=1).fillna(0.0)

    nwp_all['nwp_vpd_mean'] = nwp_all[[vpd_ec, vpd_gfs, vpd_icon]].mean(axis=1)
    nwp_all['nwp_cape'] = nwp_all[cape_col].fillna(0.0)
    nwp_all['nwp_wind_gust'] = nwp_all[[gust_gfs, gust_icon]].max(axis=1).fillna(nwp_all['nwp_ws_mean'])

    # Shifting -h por estación (para alinear condición pronosticada en t+h al momento t de emisión)
    nwp_cols = ['nwp_temp_mean', 'nwp_temp_std', 'nwp_rad_mean', 'nwp_rad_std', 
                'nwp_ws_mean', 'nwp_ws_std', 'nwp_vpd_mean', 'nwp_cape', 'nwp_wind_gust']

    nwp_features_list = []
    for st, g in nwp_all.groupby('Estacion', sort=False):
        g_sub = g[['Date', 'Estacion'] + nwp_cols].copy()
        for c in nwp_cols:
            g_sub[c] = g_sub[c].shift(-h).astype('float32')
        nwp_features_list.append(g_sub)
    nwp_shifted = pd.concat(nwp_features_list, ignore_index=True)

    df_merged = pd.merge(df_base, nwp_shifted, on=['Date', 'Estacion'], how='left')

    # Imputación de train (<= 2023) con base física consistente
    is_train = df_merged['split'] == 'train'
    df_merged.loc[is_train, 'nwp_temp_mean'] = df_merged.loc[is_train, 'honest_meteo_target_temp']
    df_merged.loc[is_train, 'nwp_rad_mean'] = df_merged.loc[is_train, 'honest_meteo_target_rad']
    df_merged.loc[is_train, 'nwp_ws_mean'] = df_merged.loc[is_train, 'honest_meteo_target_ws']

    # MOS Causal ajustado estrictamente con Train (<= 2023)
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

    # Exportar features_nwp_h<h>.parquet reusable
    export_df = df_merged[['Date', 'Estacion'] + NWP_COLS_EXPORT].copy()
    export_path = os.path.join(BASE_DIR, f'features_nwp_h{h}.parquet')
    export_df.to_parquet(export_path, index=False)
    print(f"[{h}h] Exportado {export_path} ({len(export_df)} filas, {len(export_df.columns)} columnas, {round(time.time() - t0, 1)}s)", flush=True)

    df_merged['Estacion'] = df_merged['Estacion'].astype('category')
    return df_merged, chem_cols, t0_cols, honest_cols, pp_cols, nwp_cols


def inject_noise_and_spread(train_df: pd.DataFrame, val_df: pd.DataFrame, honest_cols: list[str], pp_cols: list[str], seed: int = 42) -> pd.DataFrame:
    """Inyecta ruido empírico de NWP y dispersión multi-modelo observada en val 2024 a train."""
    train_noisy = train_df.copy()
    rng = np.random.default_rng(seed)

    val_hour = val_df['Date'].dt.hour.values
    tr_hour = train_noisy['Date'].dt.hour.values

    # 1. Ruido en honest_meteo
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

    # 2. Inyección de spreads y estabilidad condicionada a la hora solar
    spread_cols = ['nwp_temp_std', 'nwp_rad_std', 'nwp_ws_std', 'nwp_vpd_mean', 'nwp_cape', 'nwp_wind_gust']
    for col in spread_cols:
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

    # Actualizar derivadas MOS en train
    if 'nwp_mos_temp' in train_noisy.columns and 'honest_meteo_target_rad' in train_noisy.columns:
        train_noisy['nwp_mos_photothermal'] = (train_noisy['nwp_mos_temp'] * train_noisy['honest_meteo_target_rad']).astype('float32')

    return train_noisy


def train_single_horizon(h: int) -> dict:
    print(f"\n{'='*75}")
    print(f"  INICIANDO HORIZONTE h={h}")
    print(f"{'='*75}")

    cache_file = os.path.join(CACHE_DIR, f'features_h{h}.parquet')
    df_base = pd.read_parquet(cache_file)

    df_merged, chem_cols, t0_cols, honest_cols, pp_cols, nwp_cols = build_and_export_nwp_features(df_base, h)

    mos_cols = ['nwp_mos_temp', 'nwp_mos_ws', 'nwp_mos_photothermal']
    all_nwp_features = mos_cols + nwp_cols
    feature_cols = ['Estacion'] + chem_cols + t0_cols + honest_cols + all_nwp_features

    tr_df = df_merged[(df_merged['split'] == 'train') & df_merged['Target'].notna()].copy().reset_index(drop=True)
    va_df = df_merged[(df_merged['split'] == 'val') & df_merged['Target'].notna()].copy().reset_index(drop=True)
    te_df = df_merged[(df_merged['split'] == 'test') & df_merged['Target'].notna()].copy().reset_index(drop=True)

    print(f"[{h}h] Aplicando inyección de ruido y dispersión a Train...", flush=True)
    tr_noisy = inject_noise_and_spread(tr_df, va_df, honest_cols, pp_cols, seed=SEED)

    # -------------------------------------------------------------
    # 1. VARIANTE D_nwp_noise
    # -------------------------------------------------------------
    print(f"\n[{h}h] --- Entrenando D_nwp_noise ({len(feature_cols)} features) ---", flush=True)
    lgb_tr_d = lgb.Dataset(tr_noisy[feature_cols], tr_noisy['Target'].values, categorical_feature=['Estacion'], free_raw_data=False)
    lgb_va_d = lgb.Dataset(va_df[feature_cols], va_df['Target'].values, reference=lgb_tr_d, categorical_feature=['Estacion'], free_raw_data=False)

    t0_d = time.time()
    model_d = lgb.train(
        LGB_PARAMS,
        lgb_tr_d,
        num_boost_round=2000,
        valid_sets=[lgb_va_d],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]
    )
    t_d = time.time() - t0_d

    y_va_d = model_d.predict(va_df[feature_cols], num_iteration=model_d.best_iteration)
    y_te_d = model_d.predict(te_df[feature_cols], num_iteration=model_d.best_iteration)

    m_va_d = metrics(va_df['Target'].values, y_va_d)
    m_te_d = evaluate_test(te_df, y_te_d, h)
    m_te_d['train_time'] = t_d
    m_te_d['best_iter'] = model_d.best_iteration

    # Guardar predicciones de D_nwp_noise
    preds_val_d = pd.DataFrame({
        'Date': va_df['Date'],
        'Estacion': va_df['Estacion'],
        'h': h,
        'split': 'val',
        'y_true': va_df['Target'].values.astype('float32'),
        'y_pred': y_va_d.astype('float32')
    })
    preds_test_d = pd.DataFrame({
        'Date': te_df['Date'],
        'Estacion': te_df['Estacion'],
        'h': h,
        'split': 'test',
        'y_true': te_df['Target'].values.astype('float32'),
        'y_pred': y_te_d.astype('float32')
    })
    preds_all_d = pd.concat([preds_val_d, preds_test_d], ignore_index=True)
    preds_all_d.to_parquet(os.path.join(PREDS_DIR, f'D_nwp_noise_h{h}.parquet'), index=False)

    log_result('noche2_nwp', 'D_nwp_noise', h, m_te_d, notes='Multi-modelo ECMWF/GFS/ICON + MOS + noise injection')

    print(f"[{h}h D_nwp_noise] Val RMSE: {m_va_d['RMSE']:.4f} | Test RMSE: {m_te_d['RMSE']:.4f} | R2: {m_te_d['R2']:.4f} | Picos RMSE: {m_te_d['RMSE_peaks']:.2f}")

    # -------------------------------------------------------------
    # 2. VARIANTE E_nwp_w
    # -------------------------------------------------------------
    print(f"\n[{h}h] --- Entrenando E_nwp_w (Pesos w = 1 + (y/p90)^2) ---", flush=True)
    p90 = np.percentile(tr_noisy['Target'].values, 90)
    sample_weights = 1.0 + (tr_noisy['Target'].values / p90) ** 2

    lgb_tr_e = lgb.Dataset(tr_noisy[feature_cols], tr_noisy['Target'].values, categorical_feature=['Estacion'], weight=sample_weights, free_raw_data=False)
    lgb_va_e = lgb.Dataset(va_df[feature_cols], va_df['Target'].values, reference=lgb_tr_e, categorical_feature=['Estacion'], free_raw_data=False)

    t0_e = time.time()
    model_e = lgb.train(
        LGB_PARAMS,
        lgb_tr_e,
        num_boost_round=2000,
        valid_sets=[lgb_va_e],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]
    )
    t_e = time.time() - t0_e

    y_va_e = model_e.predict(va_df[feature_cols], num_iteration=model_e.best_iteration)
    y_te_e = model_e.predict(te_df[feature_cols], num_iteration=model_e.best_iteration)

    m_va_e = metrics(va_df['Target'].values, y_va_e)
    m_te_e = evaluate_test(te_df, y_te_e, h)
    m_te_e['train_time'] = t_e
    m_te_e['best_iter'] = model_e.best_iteration

    # Guardar predicciones de E_nwp_w
    preds_val_e = pd.DataFrame({
        'Date': va_df['Date'],
        'Estacion': va_df['Estacion'],
        'h': h,
        'split': 'val',
        'y_true': va_df['Target'].values.astype('float32'),
        'y_pred': y_va_e.astype('float32')
    })
    preds_test_e = pd.DataFrame({
        'Date': te_df['Date'],
        'Estacion': te_df['Estacion'],
        'h': h,
        'split': 'test',
        'y_true': te_df['Target'].values.astype('float32'),
        'y_pred': y_te_e.astype('float32')
    })
    preds_all_e = pd.concat([preds_val_e, preds_test_e], ignore_index=True)
    preds_all_e.to_parquet(os.path.join(PREDS_DIR, f'E_nwp_w_h{h}.parquet'), index=False)

    log_result('noche2_nwp', 'E_nwp_w', h, m_te_e, notes='Multi-modelo ECMWF/GFS/ICON + MOS + noise + peak weights')

    print(f"[{h}h E_nwp_w] Val RMSE: {m_va_e['RMSE']:.4f} | Test RMSE: {m_te_e['RMSE']:.4f} | R2: {m_te_e['R2']:.4f} | Picos RMSE: {m_te_e['RMSE_peaks']:.2f} | Bias picos: {m_te_e['bias_peaks']:.2f}")

    return {
        'h': h,
        'd_val_rmse': m_va_d['RMSE'], 'd_test_rmse': m_te_d['RMSE'], 'd_test_r2': m_te_d['R2'],
        'd_test_peaks_rmse': m_te_d['RMSE_peaks'], 'd_test_peaks_bias': m_te_d['bias_peaks'],
        'e_val_rmse': m_va_e['RMSE'], 'e_test_rmse': m_te_e['RMSE'], 'e_test_r2': m_te_e['R2'],
        'e_test_peaks_rmse': m_te_e['RMSE_peaks'], 'e_test_peaks_bias': m_te_e['bias_peaks'],
        'd_iter': model_d.best_iteration, 'e_iter': model_e.best_iteration,
        'd_time': t_d, 'e_time': t_e
    }


def main():
    horizons = [24, 48, 8, 12, 1, 4]  # En orden de prioridad según contrato
    summary_results = []

    print("=" * 80)
    print("  EJECUCION INTEGRAL NWP NOCHE2 (h = 24, 48, 8, 12, 1, 4)")
    print(f"  LightGBM num_threads = {LGB_PARAMS['num_threads']}")
    print(f"  Carpeta destino: {BASE_DIR}")
    print("=" * 80)

    for h in horizons:
        res = train_single_horizon(h)
        summary_results.append(res)
        
        # Guardar resumen parcial actualizado
        res_df = pd.DataFrame(summary_results).sort_values('h')
        res_df.to_csv(os.path.join(BASE_DIR, 'metrics_summary.csv'), index=False)

    print("\n" + "=" * 95)
    print("  TABLA GLOBAL FINAL: EXPERIMENTO NOCHE2_NWP")
    print("=" * 95)
    final_df = pd.DataFrame(summary_results).sort_values('h')
    print(final_df.to_string(index=False))
    print("=" * 95)


if __name__ == '__main__':
    main()
