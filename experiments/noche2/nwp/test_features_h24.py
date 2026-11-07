"""
experiments/noche2/nwp/test_features_h24.py
============================================
Prueba de features NWP en h=24:
1. Baseline: C_honest (10.2289 val / 10.5085 test)
2. Variante NWP 1: + MOS causal (T, WS)
3. Variante NWP 2: + Multi-modelo (ECMWF, GFS, ICON) mean + spread + CAPE + ráfagas
4. Variante NWP 3: Combinación completa (MOS + Multi-modelo + Estabilidad)
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
    'num_threads': 3,  # Contrato: 3 hilos
    'feature_pre_filter': False,
    'importance_type': 'gain',
    **BEST_OPTUNA
}


def load_h24_base():
    print("Cargando features_h24.parquet base...", flush=True)
    df = pd.read_parquet(os.path.join(CACHE_DIR, 'features_h24.parquet'))
    df['Estacion'] = df['Estacion'].astype('category')
    return df


def build_nwp_features(df_base: pd.DataFrame, h: int = 24):
    print(f"Construyendo features NWP multi-modelo y MOS causal para h={h}...", flush=True)
    # 1. Cargar datos NWP descargados (2024-2025)
    nwp_dfs = []
    for st in MODEL_STATIONS:
        p = os.path.join(DATA_DIR, f'nwp_multimodel_{st}.parquet')
        if os.path.exists(p):
            nwp_dfs.append(pd.read_parquet(p))
    nwp_all = pd.concat(nwp_dfs, ignore_index=True).sort_values(['Estacion', 'Date'])

    # Calcular features derivadas a nivel t+h por estacion
    suf = '_previous_day1' if h <= 24 else '_previous_day2'
    
    # Columnas por modelo
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

    # Shifting -h por estacion para que el valor en t sea el pronostico para t+h
    nwp_features_list = []
    nwp_cols = ['nwp_temp_mean', 'nwp_temp_std', 'nwp_rad_mean', 'nwp_rad_std', 
                'nwp_ws_mean', 'nwp_ws_std', 'nwp_vpd_mean', 'nwp_cape', 'nwp_wind_gust']

    for st, g in nwp_all.groupby('Estacion', sort=False):
        g_sub = g[['Date', 'Estacion'] + nwp_cols].copy()
        for c in nwp_cols:
            g_sub[c] = g_sub[c].shift(-h).astype('float32')
        nwp_features_list.append(g_sub)
    nwp_shifted = pd.concat(nwp_features_list, ignore_index=True)

    # Unir con df_base por Date y Estacion
    df_merged = pd.merge(df_base, nwp_shifted, on=['Date', 'Estacion'], how='left')

    # Para train (<= 2023), imputar valores consistentes de seamless/reanalisis
    is_train = df_merged['split'] == 'train'
    df_merged.loc[is_train, 'nwp_temp_mean'] = df_merged.loc[is_train, 'honest_meteo_target_temp']
    df_merged.loc[is_train, 'nwp_temp_std'] = 0.0
    df_merged.loc[is_train, 'nwp_rad_mean'] = df_merged.loc[is_train, 'honest_meteo_target_rad']
    df_merged.loc[is_train, 'nwp_rad_std'] = 0.0
    df_merged.loc[is_train, 'nwp_ws_mean'] = df_merged.loc[is_train, 'honest_meteo_target_ws']
    df_merged.loc[is_train, 'nwp_ws_std'] = 0.0
    df_merged.loc[is_train, 'nwp_vpd_mean'] = 1.0
    df_merged.loc[is_train, 'nwp_cape'] = 0.0
    df_merged.loc[is_train, 'nwp_wind_gust'] = df_merged.loc[is_train, 'honest_meteo_target_ws']

    # 2. MOS Causal: entrenar con train (<= 2023)
    print("Ajustando calibración MOS causal...", flush=True)
    df_merged['hour'] = df_merged['Date'].dt.hour
    
    train_df = df_merged[is_train]
    
    # Calculamos sesgo horario por estacion para TOUT y wind_speed en train
    bias_t_series = train_df.groupby(['Estacion', 'hour'], observed=False)['TOUT_lag_0'].mean() - train_df.groupby(['Estacion', 'hour'], observed=False)['meteo_t0_temp'].mean()
    bias_ws_series = train_df.groupby(['Estacion', 'hour'], observed=False)['wind_speed_t0'].mean() - train_df.groupby(['Estacion', 'hour'], observed=False)['meteo_t0_ws'].mean()

    # Mapear a target_hour = (hour + h) % 24
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
    return df_merged


def train_eval(df: pd.DataFrame, feature_cols: list[str], name: str):
    print(f"\n--- Evaluando {name} ({len(feature_cols)} features) ---", flush=True)
    tr = df[(df['split'] == 'train') & df['Target'].notna()]
    va = df[(df['split'] == 'val') & df['Target'].notna()]
    te = df[(df['split'] == 'test') & df['Target'].notna()]

    lgb_tr = lgb.Dataset(tr[feature_cols], tr['Target'].values, categorical_feature=['Estacion'], free_raw_data=False)
    lgb_va = lgb.Dataset(va[feature_cols], va['Target'].values, reference=lgb_tr, categorical_feature=['Estacion'], free_raw_data=False)

    t0 = time.time()
    model = lgb.train(
        LGB_PARAMS,
        lgb_tr,
        num_boost_round=2000,
        valid_sets=[lgb_va],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]
    )
    t_train = time.time() - t0

    y_pred_va = model.predict(va[feature_cols], num_iteration=model.best_iteration)
    y_pred_te = model.predict(te[feature_cols], num_iteration=model.best_iteration)

    m_va = metrics(va['Target'].values, y_pred_va)
    m_te = evaluate_test(te, y_pred_te, 24)

    print(f"[{name}] Best Iter: {model.best_iteration}, Tiempo: {t_train:.1f}s")
    print(f"  Val  RMSE: {m_va['RMSE']:.4f} | MAE: {m_va['MAE']:.4f} | R2: {m_va['R2']:.4f}")
    print(f"  Test RMSE: {m_te['RMSE']:.4f} | MAE: {m_te['MAE']:.4f} | R2: {m_te['R2']:.4f} | RMSE_p95: {m_te['RMSE_peaks']:.2f} | Bias_p95: {m_te['bias_peaks']:.2f}")

    # Feature importances top 10
    imp = pd.Series(model.feature_importance(importance_type='gain'), index=feature_cols).sort_values(ascending=False)
    nwp_imp = imp[[c for c in imp.index if 'nwp' in c]]
    print("Top NWP features por ganancia:")
    for c, v in nwp_imp.items():
        print(f"    {c:<25}: {v:.2e}")

    return m_va, m_te, y_pred_va, y_pred_te


def main():
    df_base = load_h24_base()

    # Columnas v3_base C_honest
    t0_cols = [c for c in df_base.columns if c.startswith('meteo_t0_')]
    honest_cols = [c for c in df_base.columns if c.startswith('honest_meteo_')]
    pp_cols = [c for c in df_base.columns if c.startswith('pp_meteo_')]
    exclude = {'Date', 'Estacion', 'split', 'Target', 'Target_24h'} | set(t0_cols) | set(honest_cols) | set(pp_cols)
    chem_cols = [c for c in df_base.columns if c not in exclude]

    base_features = ['Estacion'] + chem_cols + t0_cols + honest_cols

    m_va_base = {'RMSE': 10.2289, 'MAE': 7.3352, 'R2': 0.7431}
    m_te_base = {'RMSE': 10.5085, 'MAE': 7.4741, 'R2': 0.7513, 'RMSE_peaks': 26.88, 'bias_peaks': -22.44}

    # Agregar features NWP
    df_nwp = build_nwp_features(df_base, h=24)

    # Variante A: Solo MOS
    mos_cols = ['nwp_mos_temp', 'nwp_mos_ws', 'nwp_mos_photothermal']
    features_mos = base_features + mos_cols
    m_va_mos, m_te_mos, _, _ = train_eval(df_nwp, features_mos, "NWP_MOS_only")

    # Variante B: Solo Multi-modelo & Estabilidad
    mm_cols = ['nwp_temp_mean', 'nwp_temp_std', 'nwp_rad_mean', 'nwp_rad_std', 
               'nwp_ws_mean', 'nwp_ws_std', 'nwp_vpd_mean', 'nwp_cape', 'nwp_wind_gust']
    features_mm = base_features + mm_cols
    m_va_mm, m_te_mm, _, _ = train_eval(df_nwp, features_mm, "NWP_MultiModel_only")

    # Variante C: Todo junto (MOS + MultiModel + Estabilidad)
    all_nwp_cols = mos_cols + mm_cols
    features_all = base_features + all_nwp_cols
    m_va_all, m_te_all, _, _ = train_eval(df_nwp, features_all, "NWP_Full_Suite")

    print("\n" + "=" * 80)
    print("  RESUMEN COMPARATIVO EN HORIZONTE h=24")
    print("=" * 80)
    print(f"{'Configuracion':<22} {'Val RMSE':<10} {'Test RMSE':<10} {'Test R2':<8} {'Test p95':<10} {'Bias p95':<10}")
    print("-" * 80)
    print(f"{'C_honest_baseline':<22} {m_va_base['RMSE']:<10.4f} {m_te_base['RMSE']:<10.4f} {m_te_base['R2']:<8.4f} {m_te_base['RMSE_peaks']:<10.2f} {m_te_base['bias_peaks']:<10.2f}")
    print(f"{'NWP_MOS_only':<22} {m_va_mos['RMSE']:<10.4f} {m_te_mos['RMSE']:<10.4f} {m_te_mos['R2']:<8.4f} {m_te_mos['RMSE_peaks']:<10.2f} {m_te_mos['bias_peaks']:<10.2f}")
    print(f"{'NWP_MultiModel_only':<22} {m_va_mm['RMSE']:<10.4f} {m_te_mm['RMSE']:<10.4f} {m_te_mm['R2']:<8.4f} {m_te_mm['RMSE_peaks']:<10.2f} {m_te_mm['bias_peaks']:<10.2f}")
    print(f"{'NWP_Full_Suite':<22} {m_va_all['RMSE']:<10.4f} {m_te_all['RMSE']:<10.4f} {m_te_all['R2']:<8.4f} {m_te_all['RMSE_peaks']:<10.2f} {m_te_all['bias_peaks']:<10.2f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
