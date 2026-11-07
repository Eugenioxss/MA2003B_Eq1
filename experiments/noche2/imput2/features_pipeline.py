"""
experiments/noche2/imput2/features_pipeline.py
==============================================
Construye las matrices de características enriquecidas para el experimento noche2_imput2:
  1. Base química y espacial con imputación causal LightGBM (audit_imput)
  2. Missingness flags e indicadores de tiempo desde la última medición válida
  3. Meteorología honesta contemporánea y pronosticada (Open-Meteo PP-train / NWP-test)
  4. Features de régimen sinóptico (clusters de patrones de viento/temperatura NWP)
  5. Interacciones química-meteo avanzadas (O3(t0)*delta_T, NOx(t0)*SR_NWP, etc.)
  6. Estadísticos del día anterior (máx O3 ayer, hora del máximo, media)
  7. Tendencias regionales y contrastes espaciales de cuenca
  8. Días festivos y fines de semana de México (LFT y calendario cívico)
"""
from __future__ import annotations
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import load_panel, ALL_STATIONS, MODEL_STATIONS, STATION_COORDS, HORIZONS, SEED
from planA_quickwins.features_planA import build_planA_features

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_V3_DIR = os.path.join(ROOT, 'experiments', 'noche', 'v3_base', 'cache')
IMPUTED_PANEL_PATH = os.path.join(ROOT, 'experiments', 'noche', 'audit_imput', 'panel_spatial_lgbm.parquet')

# Días festivos oficiales y de asueto en México (2020-2025)
MEXICO_HOLIDAYS_SET = {
    # 2020
    '2020-01-01', '2020-02-03', '2020-03-16', '2020-04-09', '2020-04-10', '2020-05-01', '2020-09-16', '2020-11-16', '2020-12-24', '2020-12-25', '2020-12-31',
    # 2021
    '2021-01-01', '2021-02-01', '2021-03-15', '2021-04-01', '2021-04-02', '2021-05-01', '2021-09-16', '2021-11-15', '2021-12-24', '2021-12-25', '2021-12-31',
    # 2022
    '2022-01-01', '2022-02-07', '2022-03-21', '2022-04-14', '2022-04-15', '2022-05-01', '2022-09-16', '2022-11-21', '2022-12-24', '2022-12-25', '2022-12-31',
    # 2023
    '2023-01-01', '2023-02-06', '2023-03-20', '2023-04-06', '2023-04-07', '2023-05-01', '2023-09-16', '2023-11-20', '2023-12-24', '2023-12-25', '2023-12-31',
    # 2024
    '2024-01-01', '2024-02-05', '2024-03-18', '2024-03-28', '2024-03-29', '2024-05-01', '2024-09-16', '2024-10-01', '2024-11-18', '2024-12-24', '2024-12-25', '2024-12-31',
    # 2025
    '2025-01-01', '2025-02-03', '2025-03-17', '2025-04-17', '2025-04-18', '2025-05-01', '2025-09-16', '2025-11-17', '2025-12-24', '2025-12-25', '2025-12-31',
}


def compute_missing_features(raw_panel: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores de faltante y horas desde la última observación válida."""
    print("  Calculando missingness flags y age of missingness...", flush=True)
    df_sorted = raw_panel.sort_values(['Estacion', 'Date']).reset_index(drop=True)
    
    key_vars = ['O3', 'NOX', 'TOUT', 'SR', 'PM10', 'PM2.5', 'RH', 'U_Wind']
    out_cols = {'Date': df_sorted['Date'], 'Estacion': df_sorted['Estacion']}
    
    # Flags binarios instantáneos en t0
    total_missing = np.zeros(len(df_sorted), dtype='float32')
    for v in key_vars:
        if v in df_sorted.columns:
            m = df_sorted[v].isna().astype('float32')
            out_cols[f'is_missing_{v}'] = m
            total_missing += m
    out_cols['missing_count_total'] = total_missing
    
    # Horas desde la última observación válida (age)
    for v in ['O3', 'NOX', 'TOUT', 'SR', 'PM2.5']:
        if v in df_sorted.columns:
            is_valid = df_sorted[v].notna()
            # Contador acumulado por bloques
            grp = df_sorted.groupby('Estacion', sort=False)
            
            # Vectorizado por estación
            ages = np.zeros(len(df_sorted), dtype='float32')
            for est, sub_df in df_sorted.groupby('Estacion', sort=False):
                sub_idx = sub_df.index
                sub_valid = sub_df[v].notna()
                sub_blocks = sub_valid.cumsum()
                sub_age = (~sub_valid).groupby(sub_blocks).cumsum().clip(upper=72).astype('float32')
                ages[sub_idx] = sub_age.values
            out_cols[f'hours_since_valid_{v}'] = ages
            
    return pd.DataFrame(out_cols)


def compute_yesterday_stats(imputed_panel: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticos del día calendario anterior (D-1): máx, media, hora del máx, p90."""
    print("  Calculando estadísticos del día anterior (D-1)...", flush=True)
    df = imputed_panel[['Estacion', 'Date', 'O3']].copy()
    df['day'] = df['Date'].dt.date
    
    # Agrupar por estación y día
    daily = df.groupby(['Estacion', 'day'])['O3'].agg(
        O3_max_yesterday='max',
        O3_mean_yesterday='mean',
        O3_p90_yesterday=lambda s: np.percentile(s, 90.0) if len(s) > 0 else np.nan
    ).reset_index()
    
    # Hora del máximo
    idx_max = df.groupby(['Estacion', 'day'])['O3'].idxmax()
    daily['O3_hour_max_yesterday'] = df.loc[idx_max, 'Date'].dt.hour.values.astype('float32')
    
    # Para ser estrictamente causal: el día D toma las métricas del día D-1
    daily['day_target'] = pd.to_datetime(daily['day']) + pd.Timedelta(days=1)
    daily['day_target'] = daily['day_target'].dt.date
    daily.drop(columns=['day'], inplace=True)
    
    # Merge de vuelta al panel
    df_merged = pd.merge(
        df[['Estacion', 'Date', 'day']],
        daily,
        left_on=['Estacion', 'day'],
        right_on=['Estacion', 'day_target'],
        how='left'
    )
    
    res = df_merged[['Date', 'Estacion', 'O3_max_yesterday', 'O3_mean_yesterday', 'O3_p90_yesterday', 'O3_hour_max_yesterday']].copy()
    for col in ['O3_max_yesterday', 'O3_mean_yesterday', 'O3_p90_yesterday', 'O3_hour_max_yesterday']:
        res[col] = res[col].astype('float32')
    return res


def compute_regional_trends(imputed_panel: pd.DataFrame) -> pd.DataFrame:
    """Calcula tendencias regionales de cuenca y contraste estación vs cuenca."""
    print("  Calculando tendencias regionales de cuenca...", flush=True)
    # Media regional en cada hora t
    reg_o3 = imputed_panel.groupby('Date')['O3'].mean().rename('regional_O3_mean_t0').astype('float32')
    reg_nox = imputed_panel.groupby('Date')['NOX'].mean().rename('regional_NOX_mean_t0').astype('float32')
    
    reg_df = pd.concat([reg_o3, reg_nox], axis=1)
    reg_df['regional_O3_diff_24h'] = (reg_df['regional_O3_mean_t0'] - reg_df['regional_O3_mean_t0'].shift(24)).astype('float32')
    reg_df['regional_O3_trend_6h'] = (reg_df['regional_O3_mean_t0'] - reg_df['regional_O3_mean_t0'].shift(6)).astype('float32')
    reg_df = reg_df.reset_index()
    
    merged = pd.merge(imputed_panel[['Date', 'Estacion', 'O3']], reg_df, on='Date', how='left')
    merged['station_vs_regional_bias'] = (merged['O3'] - merged['regional_O3_mean_t0']).astype('float32')
    merged['station_vs_regional_ratio'] = (merged['O3'] / (merged['regional_O3_mean_t0'] + 1.0)).astype('float32')
    
    drop_cols = ['O3']
    return merged.drop(columns=drop_cols)


def compute_mexico_holidays(dates: pd.Series) -> pd.DataFrame:
    """Genera indicadores de calendario festivo y días inhábiles de México."""
    print("  Calculando calendario festivo de México...", flush=True)
    date_strs = dates.dt.strftime('%Y-%m-%d')
    is_holiday = date_strs.isin(MEXICO_HOLIDAYS_SET).astype('float32')
    is_weekend = (dates.dt.dayofweek >= 5).astype('float32')
    is_weekend_or_holiday = np.maximum(is_holiday, is_weekend).astype('float32')
    
    doy = dates.dt.dayofyear.values
    doy_sin = np.sin(2 * np.pi * doy / 365.25).astype('float32')
    doy_cos = np.cos(2 * np.pi * doy / 365.25).astype('float32')
    
    return pd.DataFrame({
        'Date': dates,
        'is_mexico_holiday': is_holiday,
        'is_weekend_or_holiday': is_weekend_or_holiday,
        'day_of_year_sin': doy_sin,
        'day_of_year_cos': doy_cos
    })


def compute_synoptic_clusters(train_meteo: pd.DataFrame, all_meteo: pd.DataFrame, h: int, n_clusters: int = 4) -> pd.DataFrame:
    """Ajusta clusters KMeans sobre patrones de viento/temperatura/presión de train y los aplica a todo el panel."""
    print(f"  Ajustando regímenes sinópticos (KMeans k={n_clusters}) en train <= 2023...", flush=True)
    cluster_cols = [
        'honest_meteo_target_temp', 'honest_meteo_target_pres',
        'honest_meteo_target_ws', 'honest_meteo_target_u', 'honest_meteo_target_v',
        'honest_meteo_target_rad'
    ]
    # Filtrar solo train y filas completas
    tr_sub = train_meteo[cluster_cols].dropna()
    
    # Estandarización
    mu = tr_sub.mean()
    sigma = tr_sub.std().replace(0, 1.0)
    
    X_tr = (tr_sub - mu) / sigma
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    kmeans.fit(X_tr)
    
    # Predecir sobre all_meteo
    X_all = (all_meteo[cluster_cols] - mu) / sigma
    X_all_clean = X_all.fillna(0.0)
    
    clusters = kmeans.predict(X_all_clean).astype('float32')
    distances = kmeans.transform(X_all_clean)
    
    out = pd.DataFrame({
        'Date': all_meteo['Date'],
        'Estacion': all_meteo['Estacion'],
        'synoptic_cluster': clusters
    })
    for k in range(n_clusters):
        out[f'synoptic_dist_c{k}'] = distances[:, k].astype('float32')
        
    return out


def compute_chem_meteo_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula interacciones no lineales entre química en t0 y meteorología futura proyectada en t+h."""
    print("  Calculando interacciones química-meteo proyectadas...", flush=True)
    out = pd.DataFrame({'Date': df['Date'], 'Estacion': df['Estacion']})
    
    o3_t0 = df['O3_lag_0'].astype('float32') if 'O3_lag_0' in df.columns else df['O3_lag_1'].astype('float32')
    nox_t0 = df['NOX_lag_0'].astype('float32') if 'NOX_lag_0' in df.columns else df['NOX_lag_1'].astype('float32')
    
    if 'honest_meteo_delta_temp' in df.columns:
        out['chem_O3_t0_x_delta_T'] = (o3_t0 * df['honest_meteo_delta_temp']).astype('float32')
    if 'honest_meteo_target_rad' in df.columns:
        out['chem_NOX_t0_x_rad_target'] = (nox_t0 * df['honest_meteo_target_rad']).astype('float32')
    if 'honest_meteo_target_ws' in df.columns:
        out['chem_O3_t0_div_ws_target'] = (o3_t0 / (df['honest_meteo_target_ws'] + 0.5)).astype('float32')
    if 'honest_meteo_target_temp' in df.columns and 'honest_meteo_delta_temp' in df.columns:
        out['chem_temp_target_x_delta_T'] = (df['honest_meteo_target_temp'] * df['honest_meteo_delta_temp']).astype('float32')
    if 'Ox_lag_0' in df.columns and 'honest_meteo_target_rad' in df.columns:
        out['chem_Ox_t0_x_rad_target'] = (df['Ox_lag_0'] * df['honest_meteo_target_rad']).astype('float32')
        
    return out
