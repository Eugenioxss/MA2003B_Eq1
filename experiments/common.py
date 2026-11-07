"""
experiments/common.py
=====================
Utilidades COMPARTIDAS por todos los experimentos del "siguiente nivel".
Todos los experimentos DEBEN usar este módulo para que los resultados sean comparables.

Convenciones (contrato de evaluación)
-------------------------------------
- Panel horario completo por estación (reindex 2020-01-01 00:00 .. 2025-12-31 23:00), huecos = NaN.
- Estaciones de modelado: MODEL_STATIONS (se excluyen NE3 y NO3, igual que scripts 6/7).
- Fila t = momento de emisión del pronóstico. Target_h = O3(t+h).
- Split temporal: train = 2020-2023, val = 2024 (early stopping / tuning), test = 2025 (solo evaluación final).
- Conjunto de evaluación estándar: filas de 2025, estación en MODEL_STATIONS, Target_h observado.
  Si un modelo no puede predecir alguna fila, se reporta la cobertura y se evalúa también sobre la intersección.
- Métricas: RMSE, MAE, R2, RMSE y sesgo en picos (y_real >= p95 del test), RMSE en horas diurnas (10-18h),
  skill = 1 - RMSE / RMSE_climatologia.
- Registrar SIEMPRE resultados con `log_result(...)` -> experiments/results_registry.csv

Uso:
    import sys; sys.path.insert(0, 'experiments')
    from common import *
    panel = load_panel()            # DataFrame con Date, Estacion y las 15 variables (+U_Wind, V_Wind)
"""
from __future__ import annotations
import os, json, time
from functools import reduce
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAR_DIR = os.path.join(ROOT, 'data', 'processed', 'variables')
EXP_DIR = os.path.join(ROOT, 'experiments')
CACHE_DIR = os.path.join(EXP_DIR, '_cache')
REGISTRY = os.path.join(EXP_DIR, 'results_registry.csv')
os.makedirs(CACHE_DIR, exist_ok=True)

ALL_VARS = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'PRS', 'RAINF', 'RH', 'SO2', 'SR', 'TOUT', 'WSR', 'WDR']
ALL_STATIONS = ['CE', 'NE', 'NE2', 'NE3', 'NO', 'NO2', 'NO3', 'NTE', 'NTE2', 'SE', 'SE2', 'SE3', 'SO', 'SO2', 'SUR']
EXCLUDED = ['NE3', 'NO3']
MODEL_STATIONS = [s for s in ALL_STATIONS if s not in EXCLUDED]
HORIZONS = [1, 4, 8, 12, 24, 48]
SEED = 42

# Coordenadas (doc "Ubicación de las estaciones de monitoreo SIMA 2025"): lat, lon (decimal), elevación msnm
def _dms(d, m, s): return d + m / 60 + s / 3600
STATION_COORDS = {
    'SE':   (_dms(25, 39, 55), -_dms(100, 14, 37), 500),
    'NE':   (_dms(25, 44, 41), -_dms(100, 15, 11), 474),
    'CE':   (_dms(25, 40, 33), -_dms(100, 20, 18), 562),
    'NO':   (_dms(25, 45, 46), -_dms(100, 22,  9), 568),
    'SO':   (_dms(25, 40, 32), -_dms(100, 27, 29), 674),
    'NO2':  (_dms(25, 48,  1), -_dms(100, 35,  4), 702),
    'NTE':  (_dms(25, 47, 55), -_dms(100, 19, 38), 503),
    'NE2':  (_dms(25, 46, 38), -_dms(100, 11, 17), 432),
    'SE2':  (_dms(25, 38, 45), -_dms(100,  5, 43), 387),
    'SO2':  (_dms(25, 39, 54), -_dms(100, 24, 46), 636),
    'SE3':  (_dms(25, 36,  4), -_dms( 99, 59, 57), 334),
    'SUR':  (_dms(25, 37,  1), -_dms(100, 16, 26), 555),
    'NTE2': (_dms(25, 43, 47), -_dms(100, 18, 36), 520),
    'NE3':  (_dms(25, 47, 26), -_dms(100,  4, 42), 346),
    'NO3':  (_dms(25, 46,  6), -_dms(100, 27, 49), 607),
}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def load_panel(stations=None, use_cache=True) -> pd.DataFrame:
    """Panel horario completo (todas las variables), una fila por (Estacion, Date). Huecos = NaN.
    Incluye U_Wind, V_Wind derivados de WSR/WDR (se conservan WSR y WDR)."""
    cache = os.path.join(CACHE_DIR, 'panel_full.parquet')
    if use_cache and os.path.exists(cache):
        panel = pd.read_parquet(cache)
    else:
        dfs = []
        for v in ALL_VARS:
            d = pd.read_parquet(os.path.join(VAR_DIR, f'{v}_clean.parquet'))
            d['Estacion'] = d['Estacion'].astype(str)
            d = d.drop_duplicates(['Estacion', 'Date'])
            dfs.append(d)
        df = reduce(lambda l, r: pd.merge(l, r, on=['Date', 'Estacion'], how='outer'), dfs)
        idx = pd.MultiIndex.from_product(
            [ALL_STATIONS, pd.date_range('2020-01-01 00:00', '2025-12-31 23:00', freq='h')],
            names=['Estacion', 'Date'])
        panel = df.set_index(['Estacion', 'Date']).reindex(idx).reset_index()
        # Códigos centinela residuales documentados (997/999)
        for v in ['PM10', 'PM2.5']:
            panel.loc[panel[v] >= 997, v] = np.nan
        rad = np.deg2rad(panel['WDR'])
        panel['U_Wind'] = (-panel['WSR'] * np.sin(rad)).astype('float32')
        panel['V_Wind'] = (-panel['WSR'] * np.cos(rad)).astype('float32')
        panel = panel.sort_values(['Estacion', 'Date']).reset_index(drop=True)
        panel.to_parquet(cache, index=False)
    if stations is not None:
        panel = panel[panel['Estacion'].isin(stations)].reset_index(drop=True)
    return panel


def add_targets(df: pd.DataFrame, horizons=HORIZONS, col='O3') -> pd.DataFrame:
    """Agrega Target_{h}h = col(t+h) por estación. df debe ser panel completo ordenado (Estacion, Date)."""
    g = df.groupby('Estacion', sort=False)[col]
    for h in horizons:
        df[f'Target_{h}h'] = g.shift(-h)
    return df


def build_v2_features(panel: pd.DataFrame, stations=MODEL_STATIONS) -> tuple[pd.DataFrame, list[str]]:
    """Reproduce EXACTAMENTE los features del Script 7 (v2) sobre el panel completo.
    Devuelve (df, lista_features). Útil como baseline 'v2' y como punto de partida."""
    df = panel[panel['Estacion'].isin(stations)].copy()
    df = df.sort_values(['Estacion', 'Date']).reset_index(drop=True)
    out = []
    for est, g in df.groupby('Estacion', sort=False):
        g = g.copy()
        s = lambda c, k: g[c].shift(k)
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            g[f'O3_lag_{lag}'] = s('O3', lag)
        for lag in [1, 2, 4, 12, 24, 48]:
            for f in ['NOX', 'SR', 'TOUT', 'PM10', 'PM2.5', 'U_Wind', 'V_Wind']:
                g[f'{f}_lag_{lag}'] = s(f, lag)
        g['hour'] = g['Date'].dt.hour
        g['month'] = g['Date'].dt.month
        g['dayofweek'] = g['Date'].dt.dayofweek
        g['NOX_x_SR'] = s('NOX', 1) * s('SR', 1)
        g['TOUT_x_SR'] = s('TOUT', 1) * s('SR', 1)
        g['O3_NOX_ratio'] = s('O3', 1) / (s('NOX', 1) + 1)
        g['wind_speed'] = np.sqrt(s('U_Wind', 1) ** 2 + s('V_Wind', 1) ** 2)
        g['O3_diff_1h'] = s('O3', 1) - s('O3', 2)
        g['O3_accel'] = (s('O3', 1) - s('O3', 2)) - (s('O3', 2) - s('O3', 3))
        g['O3_diff_24h'] = s('O3', 1) - s('O3', 25)
        g['TOUT_diff_3h'] = s('TOUT', 1) - s('TOUT', 4)
        g['NOX_diff_1h'] = s('NOX', 1) - s('NOX', 2)
        g['SR_diff_1h'] = s('SR', 1) - s('SR', 2)
        o3s, ts, ns, ss = s('O3', 1), s('TOUT', 1), s('NOX', 1), s('SR', 1)
        g['O3_mean_6h'] = o3s.rolling(6, min_periods=2).mean()
        g['O3_std_6h'] = o3s.rolling(6, min_periods=2).std()
        g['TOUT_mean_6h'] = ts.rolling(6, min_periods=2).mean()
        g['NOX_mean_6h'] = ns.rolling(6, min_periods=2).mean()
        g['SR_sum_6h'] = ss.rolling(6, min_periods=2).sum()
        g['O3_max_24h'] = o3s.rolling(24, min_periods=6).max()
        g['O3_min_24h'] = o3s.rolling(24, min_periods=6).min()
        g['O3_range_24h'] = g['O3_max_24h'] - g['O3_min_24h']
        g['TOUT_max_24h'] = ts.rolling(24, min_periods=6).max()
        g['SR_sum_12h'] = ss.rolling(12, min_periods=3).sum()
        g['O3_vs_yesterday'] = s('O3', 1) / (s('O3', 25) + 1)
        g['TOUT_vs_yesterday'] = s('TOUT', 1) - s('TOUT', 25)
        g['is_weekend'] = (g['Date'].dt.dayofweek >= 5).astype(int)
        g['is_rush_hour'] = g['Date'].dt.hour.isin([7, 8, 9, 17, 18, 19]).astype(int)
        g['hour_sin'] = np.sin(2 * np.pi * g['Date'].dt.hour / 24)
        g['hour_cos'] = np.cos(2 * np.pi * g['Date'].dt.hour / 24)
        g['month_sin'] = np.sin(2 * np.pi * g['Date'].dt.month / 12)
        g['month_cos'] = np.cos(2 * np.pi * g['Date'].dt.month / 12)
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df = add_targets(df)
    base_cols = set(panel.columns)
    feats = [c for c in df.columns if c not in base_cols and not c.startswith('Target_')]
    return df, feats


def split(df: pd.DataFrame):
    y = df['Date'].dt.year
    return df[y <= 2023], df[y == 2024], df[y == 2025]


# ---------------------------------------------------------------------------
# Baselines y métricas
# ---------------------------------------------------------------------------
_CLIM = None
def climatology_pred(estacion: pd.Series, target_time: pd.Series, panel: pd.DataFrame | None = None) -> np.ndarray:
    """Promedio de O3 por estación x mes x hora calculado SOLO con 2020-2023."""
    global _CLIM
    if _CLIM is None:
        p = panel if panel is not None else load_panel()
        tr = p[p['Date'].dt.year <= 2023]
        _CLIM = tr.groupby([tr['Estacion'], tr['Date'].dt.month, tr['Date'].dt.hour])['O3'].mean()
    tt = pd.to_datetime(target_time)
    idx = pd.MultiIndex.from_arrays([np.asarray(estacion), tt.dt.month.values, tt.dt.hour.values])
    return _CLIM.reindex(idx).to_numpy()


def metrics(y_true, y_pred, clim=None, hours=None) -> dict:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    out = {
        'n': int(ok.sum()), 'coverage': float(ok.mean()),
        'RMSE': rmse, 'MAE': float(np.mean(np.abs(yt - yp))),
        'R2': float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)),
        'bias': float(np.mean(yp - yt)),
    }
    thr = np.quantile(yt, 0.95); pk = yt >= thr
    out['RMSE_peaks'] = float(np.sqrt(np.mean((yt[pk] - yp[pk]) ** 2)))
    out['bias_peaks'] = float(np.mean(yp[pk] - yt[pk]))
    if hours is not None:
        hh = np.asarray(hours)[ok]; d = (hh >= 10) & (hh <= 18)
        out['RMSE_day'] = float(np.sqrt(np.mean((yt[d] - yp[d]) ** 2)))
    if clim is not None:
        c = np.asarray(clim, float)[ok]; okc = np.isfinite(c)
        rc = float(np.sqrt(np.mean((yt[okc] - c[okc]) ** 2)))
        out['RMSE_clim'] = rc
        out['skill_vs_clim'] = float(1 - rmse / rc)
    return out


def evaluate_test(test_df: pd.DataFrame, y_pred, h: int) -> dict:
    """test_df: filas de test con columnas Date, Estacion, Target_{h}h. Devuelve métricas estándar."""
    tt = test_df['Date'] + pd.Timedelta(hours=h)
    clim = climatology_pred(test_df['Estacion'], tt)
    return metrics(test_df[f'Target_{h}h'].values, y_pred, clim=clim, hours=tt.dt.hour.values)


def log_result(experiment: str, variant: str, h: int, m: dict, notes: str = ''):
    """Agrega una fila al registro global de resultados (CSV)."""
    row = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'experiment': experiment,
           'variant': variant, 'h': h, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()},
           'notes': notes}
    df = pd.DataFrame([row])
    header = not os.path.exists(REGISTRY)
    df.to_csv(REGISTRY, mode='a', header=header, index=False)
    return row


# Parámetros LightGBM del Script 7 (referencia)
LGB_PARAMS_V2 = {
    'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05, 'num_leaves': 63,
    'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,  # (sin subsample_freq: igual que Script 7)
    'seed': SEED, 'verbose': -1,
}
