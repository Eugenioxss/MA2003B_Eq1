"""
experiments/imputacion/imputers.py
==================================
Módulo reutilizable de métodos de imputación para la red de calidad del aire SIMA Monterrey.
Incluye:
  1. Interpolación temporal: Lineal y PCHIP
  2. Perfil climatológico (estación x mes x hora) y Climatología + Ajuste por Nivel
  3. Descomposición estacional STL + interpolación
  4. IDW espacial (Inverse Distance Weighting) usando coordenadas de las estaciones
  5. Regresión espacial multivariada (LightGBM)
  6. SoftImpute (completado de matriz de rango bajo mediante SVD iterativo)
  7. miceforest (MICE con LightGBM)
  8. Estrategia por niveles: impute_tiered(panel)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
from scipy.sparse.linalg import svds
from statsmodels.tsa.seasonal import STL
import lightgbm as lgb
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import STATION_COORDS, MODEL_STATIONS, ALL_STATIONS

# ---------------------------------------------------------------------------
# Utilidades espaciales
# ---------------------------------------------------------------------------
def compute_distance_matrix() -> pd.DataFrame:
    """Calcula la matriz de distancias Haversine (en km) entre todas las estaciones SIMA."""
    stations = sorted(list(STATION_COORDS.keys()))
    coords = {s: (STATION_COORDS[s][0], STATION_COORDS[s][1]) for s in stations}
    n = len(stations)
    dist = np.zeros((n, n), dtype=float)
    for i, s1 in enumerate(stations):
        lat1, lon1 = np.radians(coords[s1][0]), np.radians(coords[s1][1])
        for j, s2 in enumerate(stations):
            if i == j:
                continue
            lat2, lon2 = np.radians(coords[s2][0]), np.radians(coords[s2][1])
            dphi = lat2 - lat1
            dlam = lon2 - lon1
            a = np.sin(dphi / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlam / 2)**2
            dist[i, j] = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    return pd.DataFrame(dist, index=stations, columns=stations)


# ---------------------------------------------------------------------------
# Climatología
# ---------------------------------------------------------------------------
def build_climatology_map(panel: pd.DataFrame, vars_to_clim: list[str] | None = None, max_year: int = 2023) -> dict:
    """Calcula el perfil climatológico medio por (Estacion, mes, hora) usando datos <= max_year."""
    if vars_to_clim is None:
        vars_to_clim = ['O3', 'PM2.5', 'PM10', 'NOX', 'NO2', 'CO', 'SO2', 'TOUT', 'SR', 'RH', 'PRS', 'RAINF', 'WSR', 'WDR', 'U_Wind', 'V_Wind']
    tr = panel[panel['Date'].dt.year <= max_year]
    clim_maps = {}
    for v in vars_to_clim:
        if v in tr.columns:
            m = tr.groupby(['Estacion', tr['Date'].dt.month, tr['Date'].dt.hour])[v].mean()
            clim_maps[v] = m.to_dict()
    return clim_maps


def get_climatology_series(dates: pd.Series, station: str, var: str, clim_maps: dict) -> np.ndarray:
    """Devuelve la serie climatológica para una estación y fechas dadas."""
    m_dict = clim_maps.get(var, {})
    months = dates.dt.month.values
    hours = dates.dt.hour.values
    out = np.empty(len(dates), dtype=float)
    for i in range(len(dates)):
        out[i] = m_dict.get((station, months[i], hours[i]), np.nan)
    return out


# ---------------------------------------------------------------------------
# Métodos individuales de imputación
# ---------------------------------------------------------------------------
def impute_linear(series: pd.Series | np.ndarray) -> np.ndarray:
    """Interpolación lineal acotada."""
    s = pd.Series(series).copy()
    return s.interpolate(method='linear', limit_direction='both').values


def impute_pchip(series: pd.Series | np.ndarray) -> np.ndarray:
    """Interpolación cúbica Hermite monotónica (PCHIP)."""
    s = pd.Series(series).copy()
    try:
        imp = s.interpolate(method='pchip', limit_direction='both').values
        # Fallback a lineal si quedan NaNs en bordes
        if np.isnan(imp).any():
            imp = pd.Series(imp).interpolate(method='linear', limit_direction='both').values
        return np.maximum(0, imp)
    except Exception:
        return np.maximum(0, s.interpolate(method='linear', limit_direction='both').values)


def impute_climatology(dates: pd.Series, station: str, var: str, clim_maps: dict) -> np.ndarray:
    """Imputación pura por perfil climatológico horario (estación x mes x hora)."""
    return np.maximum(0, get_climatology_series(dates, station, var, clim_maps))


def impute_climatology_level_adj(series: pd.Series | np.ndarray, dates: pd.Series, station: str, var: str, clim_maps: dict) -> np.ndarray:
    """Climatología + ajuste por nivel (interpolación lineal de anomalías entre bordes de huecos)."""
    s = np.asarray(series, dtype=float).copy()
    clim = get_climatology_series(dates, station, var, clim_maps)
    nan_mask = np.isnan(s)
    if not nan_mask.any():
        return s
    
    anomaly = s - clim
    s_anom = pd.Series(anomaly)
    # Interpolar anomalía linealmente
    anom_interp = s_anom.interpolate(method='linear', limit_direction='both').values
    pred = clim + anom_interp
    out = s.copy()
    out[nan_mask] = np.maximum(0, pred[nan_mask])
    return out


def impute_stl(series: pd.Series | np.ndarray, period: int = 24) -> np.ndarray:
    """Descomposición STL (periodo=24h) + interpolación lineal del residuo/tendencia."""
    s = np.asarray(series, dtype=float).copy()
    nan_mask = np.isnan(s)
    if not nan_mask.any():
        return s
    # Relleno preliminar lineal para que STL pueda descomponer
    s_lin = pd.Series(s).interpolate(method='linear', limit_direction='both').bfill().ffill().values
    try:
        stl = STL(s_lin, period=period, robust=False).fit()
        deseas = s - stl.seasonal
        deseas_interp = pd.Series(deseas).interpolate(method='linear', limit_direction='both').values
        reconstructed = np.maximum(0, deseas_interp + stl.seasonal)
        out = s.copy()
        out[nan_mask] = reconstructed[nan_mask]
        return out
    except Exception:
        # Fallback a lineal
        return np.maximum(0, s_lin)


def impute_idw_spatial(panel: pd.DataFrame, var: str, dist_matrix: pd.DataFrame, clim_maps: dict | None = None, return_wide: bool = False) -> pd.DataFrame:
    """Imputación IDW espacial en cada instante t usando las demás estaciones observadas en t."""
    piv = panel.pivot(index='Date', columns='Estacion', values=var).astype(float).copy()
    dates = piv.index
    stations = piv.columns.tolist()
    
    # Precalcular pesos para cada par
    weights = {}
    for s1 in stations:
        w_dict = {}
        for s2 in stations:
            if s1 != s2:
                d = dist_matrix.loc[s1, s2]
                w_dict[s2] = 1.0 / (d ** 2) if d > 0 else 1.0
        weights[s1] = w_dict
        
    piv_imp = piv.copy()
    for st in stations:
        mask_nan = piv[st].isna()
        if not mask_nan.any():
            continue
        
        other_st = [s for s in stations if s != st]
        w_arr = np.array([weights[st][s] for s in other_st])
        other_vals = piv.loc[mask_nan, other_st].to_numpy(dtype=float, copy=True)
        
        # Valid mask
        val_mask = np.isfinite(other_vals)
        # Sum weights where valid
        w_mat = np.tile(w_arr, (len(other_vals), 1))
        w_mat[~val_mask] = 0.0
        w_sums = w_mat.sum(axis=1)
        
        # Weighted sum
        other_vals_zeros = np.where(val_mask, other_vals, 0.0)
        v_sums = (w_mat * other_vals_zeros).sum(axis=1)
        
        preds = np.zeros(len(other_vals))
        ok_w = w_sums > 0
        preds[ok_w] = v_sums[ok_w] / w_sums[ok_w]
        
        # Fallback para casos donde ninguna otra estación tenga dato
        if (~ok_w).any() and clim_maps is not None:
            nan_dates = dates[mask_nan][~ok_w]
            clim_fallback = get_climatology_series(pd.Series(nan_dates), st, var, clim_maps)
            preds[~ok_w] = clim_fallback
            
        piv_imp.loc[mask_nan, st] = np.maximum(0, preds).astype(float)
        
    if return_wide:
        return piv_imp

    # Unpivot de vuelta a formato panel
    piv_unstacked = piv_imp.unstack().reset_index(name=var)
    return piv_unstacked


def impute_softimpute_svd(M: np.ndarray, rank: int = 3, max_iter: int = 20, tol: float = 1e-4) -> np.ndarray:
    """SoftImpute / SVD iterativo sobre matriz tiempo x (estaciones/variables).
    M: matriz float con NaNs. Devuelve matriz completa sin NaNs."""
    M_work = M.copy().astype(float)
    obs = np.isfinite(M_work)
    
    col_means = np.nanmean(M_work, axis=0)
    col_stds = np.nanstd(M_work, axis=0)
    col_stds[np.isnan(col_stds) | (col_stds == 0)] = 1.0
    col_means[np.isnan(col_means)] = 0.0
    
    # Estandarizar
    X = (M_work - col_means) / col_stds
    X[~obs] = 0.0
    
    k = min(rank, X.shape[1] - 1)
    if k < 1:
        k = 1
        
    for it in range(max_iter):
        U, s, Vt = svds(X, k=k)
        X_rec = U @ np.diag(s) @ Vt
        diff = np.linalg.norm(X[~obs] - X_rec[~obs]) / (np.linalg.norm(X[~obs]) + 1e-8)
        X[~obs] = X_rec[~obs]
        if diff < tol:
            break
            
    M_rec = X * col_stds + col_means
    return np.maximum(0, M_rec)


def train_spatial_regression_lightgbm(
    panel: pd.DataFrame,
    target_var: str,
    target_station: str,
    donor_stations: list[str] | None = None,
    exclude_dates: list | pd.DatetimeIndex | None = None,
    max_train_year: int = 2023,
    num_threads: int = 3,
) -> lgb.Booster:
    """Entrena un modelo LightGBM para predecir target_station[target_var]
    a partir de las demás estaciones y variables meteorológicas en el mismo instante."""
    if donor_stations is None:
        donor_stations = [s for s in ALL_STATIONS if s != target_station]
        
    key_covars = [target_var, 'TOUT', 'SR', 'RH', 'U_Wind', 'V_Wind'] if target_var != 'TOUT' else [target_var, 'SR', 'RH', 'U_Wind', 'V_Wind']
    
    dfs = []
    for kv in key_covars:
        p = panel.pivot(index='Date', columns='Estacion', values=kv)
        cols = [s for s in donor_stations if s in p.columns]
        dfs.append(p[cols].add_prefix(f'{kv}_'))
    X_wide = pd.concat(dfs, axis=1)
    
    X_wide['hour_sin'] = np.sin(2 * np.pi * X_wide.index.hour / 24)
    X_wide['hour_cos'] = np.cos(2 * np.pi * X_wide.index.hour / 24)
    X_wide['month_sin'] = np.sin(2 * np.pi * X_wide.index.month / 12)
    X_wide['month_cos'] = np.cos(2 * np.pi * X_wide.index.month / 12)
    
    p_target = panel.pivot(index='Date', columns='Estacion', values=target_var)[target_station]
    
    train_cond = (X_wide.index.year <= max_train_year) & p_target.notna()
    if exclude_dates is not None:
        train_cond = train_cond & (~X_wide.index.isin(exclude_dates))
        
    X_train = X_wide[train_cond]
    y_train = p_target[train_cond]
    
    # Subsplit train / val temporal para early stopping
    val_cond = train_cond & (X_wide.index.year == max_train_year)
    tr_cond = train_cond & (X_wide.index.year < max_train_year)
    
    dtrain = lgb.Dataset(X_wide[tr_cond], p_target[tr_cond])
    dval = lgb.Dataset(X_wide[val_cond], p_target[val_cond], reference=dtrain)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_threads': num_threads,
        'seed': 42,
        'verbose': -1,
    }
    
    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=800,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    return bst


# ---------------------------------------------------------------------------
# ESTRATEGIA POR NIVELES (Tiered Imputation)
# ---------------------------------------------------------------------------
def impute_tiered(
    panel: pd.DataFrame,
    vars_to_impute: list[str] | None = None,
    dist_matrix: pd.DataFrame | None = None,
    clim_maps: dict | None = None,
    num_threads: int = 3,
) -> pd.DataFrame:
    """
    Aplica la estrategia de imputación por niveles recomendada para el panel SIMA:
      - Huecos cortos (<= 3 horas): Interpolación lineal temporal
      - Huecos medianos (4 a 48 horas): Climatología horaria con ajuste por nivel (interpolación de anomalías)
      - Huecos largos (> 48 horas): IDW espacial instantáneo + Climatología residual si faltan donantes
    
    Preserva intactas las fechas, estaciones y estructura original del panel.
    """
    if vars_to_impute is None:
        vars_to_impute = ['O3', 'PM2.5', 'PM10', 'NOX', 'TOUT', 'SR', 'RH', 'U_Wind', 'V_Wind']
        
    if dist_matrix is None:
        dist_matrix = compute_distance_matrix()
    if clim_maps is None:
        clim_maps = build_climatology_map(panel, vars_to_impute)
        
    df_out = panel.copy().sort_values(['Estacion', 'Date']).reset_index(drop=True)
    
    for v in vars_to_impute:
        if v not in df_out.columns:
            continue
            
        # Paso 1: Identificar longitudes de hueco por estación
        # Primero aplicamos IDW espacial completo para tener la referencia espacial de huecos largos
        piv_spatial = impute_idw_spatial(df_out, v, dist_matrix, clim_maps, return_wide=True)
        
        # Procesar estación por estación
        out_v = []
        for st, g in df_out.groupby('Estacion', sort=False):
            dates = g['Date']
            s = g[v].values.astype(float)
            isnan = np.isnan(s)
            
            if not isnan.any():
                out_v.extend(s)
                continue
                
            # Calcular longitudes de bloques continuos de NaN
            gap_lens = np.zeros(len(s), dtype=int)
            i = 0
            n = len(s)
            while i < n:
                if isnan[i]:
                    start = i
                    while i < n and isnan[i]:
                        i += 1
                    end = i
                    length = end - start
                    gap_lens[start:end] = length
                else:
                    i += 1
                    
            # 1. Componente <= 3h: lineal
            s_lin = pd.Series(s).interpolate(method='linear', limit_direction='both').values
            
            # 2. Componente 4-48h: climatología + ajuste por nivel
            s_clim_adj = impute_climatology_level_adj(s, dates, st, v, clim_maps)
            
            # 3. Componente > 48h: IDW espacial
            s_spatial = piv_spatial.loc[dates, st].values
            
            # Combinación según nivel
            s_final = s.copy()
            mask_short = isnan & (gap_lens <= 3)
            mask_med = isnan & (gap_lens > 3) & (gap_lens <= 48)
            mask_long = isnan & (gap_lens > 48)
            
            s_final[mask_short] = s_lin[mask_short]
            s_final[mask_med] = s_clim_adj[mask_med]
            s_final[mask_long] = s_spatial[mask_long]
            
            # Fallback final si quedara algún NaN residual (ej. extremos)
            s_final = pd.Series(s_final).bfill().ffill().values
            s_final = np.maximum(0, s_final)
            out_v.extend(s_final)
            
        df_out[v] = out_v
        
    return df_out
