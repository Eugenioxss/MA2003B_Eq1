"""
experiments/planA_quickwins/features_planA.py
=============================================
Feature engineering para Experimento Plan A: Quick Wins.
Incluye funciones incrementales para reproducir ablaciones:
  - V0: v2 baseline
  - V1: + lag_0 y rollings/diffs contemporáneos (t=0)
  - V2: + Estacion como categorical feature
  - V3: + variables previamente no usadas (NO, NO2, CO, RH, PRS, RAINF, SO2, Ox=O3+NO2)
  - V4: + features espaciales (red 15 estaciones, leave-one-out, nearest neighbor, upwind)
  - V5: usa features V4 con hiperparámetros optimizados
"""
from __future__ import annotations
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('experiments'))
from common import ALL_STATIONS, MODEL_STATIONS, STATION_COORDS, HORIZONS, add_targets

LAT_KM = 110.8
LON_KM = 100.3


def compute_spatial_matrices(panel: pd.DataFrame) -> pd.DataFrame:
    """Calcula matrices espaciales para todas las 15 estaciones y fechas."""
    panel_calc = panel.copy()
    if 'Ox' not in panel_calc.columns:
        panel_calc['Ox'] = panel_calc['O3'] + panel_calc['NO2']

    o3_mat = panel_calc.pivot(index='Date', columns='Estacion', values='O3')
    ox_mat = panel_calc.pivot(index='Date', columns='Estacion', values='Ox')
    tout_mat = panel_calc.pivot(index='Date', columns='Estacion', values='TOUT')
    u_mat = panel_calc.pivot(index='Date', columns='Estacion', values='U_Wind')
    v_mat = panel_calc.pivot(index='Date', columns='Estacion', values='V_Wind')
    
    dates = o3_mat.index
    
    # Precomputar vecinos mas cercanos para MODEL_STATIONS usando ALL_STATIONS
    nearest_map = {}
    for s1 in MODEL_STATIONS:
        lat1, lon1, _ = STATION_COORDS[s1]
        best_dist = 1e9
        best_s = None
        for s2 in ALL_STATIONS:
            if s1 == s2:
                continue
            lat2, lon2, _ = STATION_COORDS[s2]
            d = np.sqrt(((lat1 - lat2) * LAT_KM) ** 2 + ((lon1 - lon2) * LON_KM) ** 2)
            if d < best_dist:
                best_dist = d
                best_s = s2
        nearest_map[s1] = (best_s, float(best_dist))

    spatial_dict = {}
    for s in MODEL_STATIONS:
        lat1, lon1, _ = STATION_COORDS[s]
        other_stations = [s2 for s2 in ALL_STATIONS if s2 != s]
        
        o3_sub = o3_mat[other_stations]
        ox_sub = ox_mat[other_stations]
        tout_sub = tout_mat[other_stations]
        
        cols = {}
        # 1) Leave-one-out stats de O3, Ox, TOUT
        cols['O3_others_mean_lag_0'] = o3_sub.mean(axis=1)
        cols['O3_others_max_lag_0'] = o3_sub.max(axis=1)
        cols['O3_others_std_lag_0'] = o3_sub.std(axis=1)
        cols['O3_others_median_lag_0'] = o3_sub.median(axis=1)
        cols['O3_diff_regional_lag_0'] = o3_mat[s] - cols['O3_others_mean_lag_0']
        
        cols['O3_others_mean_lag_1'] = cols['O3_others_mean_lag_0'].shift(1)
        cols['O3_others_max_lag_1'] = cols['O3_others_max_lag_0'].shift(1)
        cols['O3_others_std_lag_1'] = cols['O3_others_std_lag_0'].shift(1)
        cols['O3_others_median_lag_1'] = cols['O3_others_median_lag_0'].shift(1)
        cols['O3_diff_regional_lag_1'] = cols['O3_diff_regional_lag_0'].shift(1)
        
        cols['TOUT_others_mean_lag_0'] = tout_sub.mean(axis=1)
        cols['TOUT_others_mean_lag_1'] = cols['TOUT_others_mean_lag_0'].shift(1)
        
        cols['Ox_others_mean_lag_0'] = ox_sub.mean(axis=1)
        cols['Ox_others_mean_lag_1'] = cols['Ox_others_mean_lag_0'].shift(1)
        cols['Ox_others_max_lag_0'] = ox_sub.max(axis=1)
        cols['Ox_others_max_lag_1'] = cols['Ox_others_max_lag_0'].shift(1)
        
        # 2) Nearest station
        near_s, dist = nearest_map[s]
        cols['O3_nearest_lag_0'] = o3_mat[near_s]
        cols['O3_nearest_lag_1'] = o3_mat[near_s].shift(1)
        cols['dist_nearest'] = dist
        
        # 3) Upwind O3
        dx = np.array([(STATION_COORDS[s2][1] - lon1) * LON_KM for s2 in other_stations])
        dy = np.array([(STATION_COORDS[s2][0] - lat1) * LAT_KM for s2 in other_stations])
        dist_others = np.sqrt(dx ** 2 + dy ** 2)
        
        u_s = u_mat[s].values[:, None]
        v_s = v_mat[s].values[:, None]
        ws = np.sqrt(u_s ** 2 + v_s ** 2)
        
        proj = (-u_s) * dx[None, :] + (-v_s) * dy[None, :]
        cos_phi = proj / (ws * dist_others[None, :] + 1e-6)
        weights = np.maximum(0.0, cos_phi) / (dist_others[None, :] + 1e-6)
        weights[ws[:, 0] < 0.5, :] = 0.0
        
        o3_other_vals = o3_sub.values
        valid = np.isfinite(o3_other_vals)
        w_valid = weights * valid
        w_sum = w_valid.sum(axis=1, keepdims=True)
        
        with np.errstate(invalid='ignore', divide='ignore'):
            upwind_o3 = np.sum(w_valid * np.nan_to_num(o3_other_vals), axis=1, keepdims=True) / w_sum
            
        regional_mean = np.nanmean(o3_other_vals, axis=1, keepdims=True)
        no_upwind = (w_sum[:, 0] == 0) | (~np.isfinite(upwind_o3[:, 0]))
        upwind_o3[no_upwind, 0] = regional_mean[no_upwind, 0]
        
        cols['O3_upwind_lag_0'] = pd.Series(upwind_o3[:, 0], index=dates)
        cols['O3_upwind_lag_1'] = cols['O3_upwind_lag_0'].shift(1)
        
        cols['Estacion'] = s
        cols['Date'] = dates
        spatial_dict[s] = pd.DataFrame(cols, index=dates)
        
    df_spatial = pd.concat(spatial_dict.values(), ignore_index=True)
    return df_spatial


def build_planA_features(panel: pd.DataFrame,
                          variant: str = 'V4',
                          stations: list[str] = MODEL_STATIONS) -> tuple[pd.DataFrame, list[str]]:
    """
    Construye features para el plan A según la variante solicitada:
      - 'V0': baseline v2 idéntico a Script 7
      - 'V1': + lag_0 y derivadas/rollings de t
      - 'V2': + Estacion como categórica
      - 'V3': + variables no usadas (NO, NO2, CO, RH, PRS, RAINF, SO2, Ox)
      - 'V4': + features espaciales de la red de 15 estaciones
      - 'V5': idéntico en features a V4 (V5 usa hiperparámetros optimizados)
    """
    assert variant in ['V0', 'V1', 'V2', 'V3', 'V4', 'V5'], f"Variante desconocida: {variant}"
    
    panel_work = panel.copy()
    if variant in ['V3', 'V4', 'V5']:
        panel_work['Ox'] = panel_work['O3'] + panel_work['NO2']
    
    df_spatial = None
    if variant in ['V4', 'V5']:
        df_spatial = compute_spatial_matrices(panel_work)

    df = panel_work[panel_work['Estacion'].isin(stations)].copy()
    df = df.sort_values(['Estacion', 'Date']).reset_index(drop=True)
    
    out = []
    for est, g in df.groupby('Estacion', sort=False):
        g = g.copy()
        s = lambda c, k: g[c].shift(k)
        new_cols = {}
        
        # ==============================================================
        # 1. Base v2 (V0)
        # ==============================================================
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            new_cols[f'O3_lag_{lag}'] = s('O3', lag)
        for lag in [1, 2, 4, 12, 24, 48]:
            for f in ['NOX', 'SR', 'TOUT', 'PM10', 'PM2.5', 'U_Wind', 'V_Wind']:
                new_cols[f'{f}_lag_{lag}'] = s(f, lag)
                
        new_cols['hour'] = g['Date'].dt.hour
        new_cols['month'] = g['Date'].dt.month
        new_cols['dayofweek'] = g['Date'].dt.dayofweek
        new_cols['NOX_x_SR'] = s('NOX', 1) * s('SR', 1)
        new_cols['TOUT_x_SR'] = s('TOUT', 1) * s('SR', 1)
        new_cols['O3_NOX_ratio'] = s('O3', 1) / (s('NOX', 1) + 1)
        new_cols['wind_speed'] = np.sqrt(s('U_Wind', 1) ** 2 + s('V_Wind', 1) ** 2)
        new_cols['O3_diff_1h'] = s('O3', 1) - s('O3', 2)
        new_cols['O3_accel'] = (s('O3', 1) - s('O3', 2)) - (s('O3', 2) - s('O3', 3))
        new_cols['O3_diff_24h'] = s('O3', 1) - s('O3', 25)
        new_cols['TOUT_diff_3h'] = s('TOUT', 1) - s('TOUT', 4)
        new_cols['NOX_diff_1h'] = s('NOX', 1) - s('NOX', 2)
        new_cols['SR_diff_1h'] = s('SR', 1) - s('SR', 2)
        
        o3s, ts, ns, ss = s('O3', 1), s('TOUT', 1), s('NOX', 1), s('SR', 1)
        new_cols['O3_mean_6h'] = o3s.rolling(6, min_periods=2).mean()
        new_cols['O3_std_6h'] = o3s.rolling(6, min_periods=2).std()
        new_cols['TOUT_mean_6h'] = ts.rolling(6, min_periods=2).mean()
        new_cols['NOX_mean_6h'] = ns.rolling(6, min_periods=2).mean()
        new_cols['SR_sum_6h'] = ss.rolling(6, min_periods=2).sum()
        new_cols['O3_max_24h'] = o3s.rolling(24, min_periods=6).max()
        new_cols['O3_min_24h'] = o3s.rolling(24, min_periods=6).min()
        new_cols['O3_range_24h'] = new_cols['O3_max_24h'] - new_cols['O3_min_24h']
        new_cols['TOUT_max_24h'] = ts.rolling(24, min_periods=6).max()
        new_cols['SR_sum_12h'] = ss.rolling(12, min_periods=3).sum()
        new_cols['O3_vs_yesterday'] = s('O3', 1) / (s('O3', 25) + 1)
        new_cols['TOUT_vs_yesterday'] = s('TOUT', 1) - s('TOUT', 25)
        new_cols['is_weekend'] = (g['Date'].dt.dayofweek >= 5).astype(int)
        new_cols['is_rush_hour'] = g['Date'].dt.hour.isin([7, 8, 9, 17, 18, 19]).astype(int)
        new_cols['hour_sin'] = np.sin(2 * np.pi * g['Date'].dt.hour / 24)
        new_cols['hour_cos'] = np.cos(2 * np.pi * g['Date'].dt.hour / 24)
        new_cols['month_sin'] = np.sin(2 * np.pi * g['Date'].dt.month / 12)
        new_cols['month_cos'] = np.cos(2 * np.pi * g['Date'].dt.month / 12)

        # ==============================================================
        # 2. V1: + lag_0 (valores contemporáneos al emitir en t)
        # ==============================================================
        if variant != 'V0':
            new_cols['O3_lag_0'] = g['O3']
            for f in ['NOX', 'SR', 'TOUT', 'PM10', 'PM2.5', 'U_Wind', 'V_Wind']:
                new_cols[f'{f}_lag_0'] = g[f]
                
            new_cols['NOX_x_SR_t0'] = g['NOX'] * g['SR']
            new_cols['TOUT_x_SR_t0'] = g['TOUT'] * g['SR']
            new_cols['O3_NOX_ratio_t0'] = g['O3'] / (g['NOX'] + 1)
            new_cols['wind_speed_t0'] = np.sqrt(g['U_Wind'] ** 2 + g['V_Wind'] ** 2)
            
            new_cols['O3_diff_1h_t0'] = g['O3'] - s('O3', 1)
            new_cols['O3_accel_t0'] = (g['O3'] - s('O3', 1)) - (s('O3', 1) - s('O3', 2))
            new_cols['O3_diff_24h_t0'] = g['O3'] - s('O3', 24)
            new_cols['TOUT_diff_3h_t0'] = g['TOUT'] - s('TOUT', 3)
            new_cols['NOX_diff_1h_t0'] = g['NOX'] - s('NOX', 1)
            new_cols['SR_diff_1h_t0'] = g['SR'] - s('SR', 1)
            
            o0, t0, n0, s0 = g['O3'], g['TOUT'], g['NOX'], g['SR']
            new_cols['O3_mean_6h_t0'] = o0.rolling(6, min_periods=2).mean()
            new_cols['O3_std_6h_t0'] = o0.rolling(6, min_periods=2).std()
            new_cols['TOUT_mean_6h_t0'] = t0.rolling(6, min_periods=2).mean()
            new_cols['NOX_mean_6h_t0'] = n0.rolling(6, min_periods=2).mean()
            new_cols['SR_sum_6h_t0'] = s0.rolling(6, min_periods=2).sum()
            new_cols['O3_max_24h_t0'] = o0.rolling(24, min_periods=6).max()
            new_cols['O3_min_24h_t0'] = o0.rolling(24, min_periods=6).min()
            new_cols['O3_range_24h_t0'] = new_cols['O3_max_24h_t0'] - new_cols['O3_min_24h_t0']
            new_cols['TOUT_max_24h_t0'] = t0.rolling(24, min_periods=6).max()
            new_cols['SR_sum_12h_t0'] = s0.rolling(12, min_periods=3).sum()
            new_cols['O3_vs_yesterday_t0'] = g['O3'] / (s('O3', 24) + 1)
            new_cols['TOUT_vs_yesterday_t0'] = g['TOUT'] - s('TOUT', 24)

        # ==============================================================
        # 3. V3: + variables no usadas (NO, NO2, CO, RH, PRS, RAINF, SO2, Ox)
        # ==============================================================
        if variant in ['V3', 'V4', 'V5']:
            for var_nu in ['NO', 'NO2', 'CO', 'RH', 'PRS', 'SO2', 'Ox']:
                for lag in [0, 1, 2, 24]:
                    new_cols[f'{var_nu}_lag_{lag}'] = s(var_nu, lag) if lag > 0 else g[var_nu]
                    
            new_cols['Ox_mean_6h_t0'] = g['Ox'].rolling(6, min_periods=2).mean()
            new_cols['Ox_max_24h_t0'] = g['Ox'].rolling(24, min_periods=6).max()
            new_cols['Ox_diff_1h_t0'] = g['Ox'] - s('Ox', 1)
            new_cols['Ox_diff_24h_t0'] = g['Ox'] - s('Ox', 24)
            
            new_cols['RH_mean_6h_t0'] = g['RH'].rolling(6, min_periods=2).mean()
            new_cols['RH_diff_1h_t0'] = g['RH'] - s('RH', 1)
            new_cols['RH_diff_24h_t0'] = g['RH'] - s('RH', 24)
            
            new_cols['PRS_diff_3h_t0'] = g['PRS'] - s('PRS', 3)
            new_cols['PRS_diff_24h_t0'] = g['PRS'] - s('PRS', 24)
            
            new_cols['RAINF_lag_0'] = g['RAINF']
            new_cols['RAINF_lag_1'] = s('RAINF', 1)
            new_cols['RAINF_sum_6h_t0'] = g['RAINF'].rolling(6, min_periods=1).sum()
            new_cols['RAINF_sum_24h_t0'] = g['RAINF'].rolling(24, min_periods=1).sum()
            new_cols['is_raining_t0'] = (g['RAINF'] > 0).astype(int)

        df_feat = pd.DataFrame(new_cols, index=g.index)
        out.append(pd.concat([g, df_feat], axis=1))

    df_out = pd.concat(out, ignore_index=True)

    # ==============================================================
    # 4. V4 / V5: Merge de features espaciales
    # ==============================================================
    if variant in ['V4', 'V5'] and df_spatial is not None:
        df_out = pd.merge(df_out, df_spatial, on=['Estacion', 'Date'], how='left')

    # Agregar targets
    df_out = add_targets(df_out)
    
    # Lista de features
    base_cols = set(panel.columns) | {'Ox'}
    feat_cols = [c for c in df_out.columns if c not in base_cols and not c.startswith('Target_')]
    
    if variant in ['V2', 'V3', 'V4', 'V5']:
        df_out['Estacion'] = df_out['Estacion'].astype('category')
        feat_cols = ['Estacion'] + feat_cols
        
    return df_out, feat_cols
