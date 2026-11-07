"""
experiments/planC_meteo/meteo_features.py
=========================================
Módulo reutilizable para cargar, procesar y unir los datos meteorológicos
de Open-Meteo (reanálisis ERA5 y pronóstico numérico histórico / previous runs)
al dataset de características v2 del SIMA.
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd

# Directorios
METEO_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

BASE_METEO_COLS = [
    'temperature_2m', 'relative_humidity_2m', 'shortwave_radiation',
    'cloud_cover', 'wind_speed_10m', 'wind_direction_10m',
    'surface_pressure', 'precipitation', 'dew_point_2m', 'boundary_layer_height'
]


def load_meteo_raw(source: str = 'reanalysis', stations: list[str] | None = None) -> pd.DataFrame:
    """
    Carga todos los archivos parquet descargados para la fuente especificada.
    source: 'reanalysis' o 'forecast'.
    Devuelve DataFrame ordenado por [Estacion, Date].
    """
    prefix = f"{source}_*.parquet"
    pattern = os.path.join(METEO_DATA_DIR, prefix)
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos para {source} en {METEO_DATA_DIR}")

    dfs = []
    for f in sorted(files):
        df_part = pd.read_parquet(f)
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)
    if stations is not None:
        df = df[df['Estacion'].isin(stations)].reset_index(drop=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Estacion', 'Date']).reset_index(drop=True)

    # Derivar componentes U y V de viento para las variables base
    if 'wind_speed_10m' in df.columns and 'wind_direction_10m' in df.columns:
        rad = np.deg2rad(df['wind_direction_10m'])
        df['U_Wind_10m'] = (-df['wind_speed_10m'] * np.sin(rad)).astype('float32')
        df['V_Wind_10m'] = (-df['wind_speed_10m'] * np.cos(rad)).astype('float32')

    # Derivar componentes U y V para previous_day1 y previous_day2 si existen
    for p in ['previous_day1', 'previous_day2']:
        ws_col = f'wind_speed_10m_{p}'
        wd_col = f'wind_direction_10m_{p}'
        if ws_col in df.columns and wd_col in df.columns:
            rad_p = np.deg2rad(df[wd_col])
            df[f'U_Wind_10m_{p}'] = (-df[ws_col] * np.sin(rad_p)).astype('float32')
            df[f'V_Wind_10m_{p}'] = (-df[ws_col] * np.cos(rad_p)).astype('float32')

    return df


def build_meteo_features_for_horizon(
    meteo_df: pd.DataFrame,
    h: int,
    source: str = 'reanalysis',
    use_previous_runs: bool = True
) -> pd.DataFrame:
    """
    Construye las covariables meteorológicas para el horizonte h:
    - Estado en t (instante de emisión del pronóstico)
    - Pronóstico / reanálisis en el momento objetivo (t + h)
    - Estadísticas en ventana móvil de 4 horas [t+h-3 .. t+h] (máx temp, suma radiación, etc.)
    - Deltas meteorológicas entre t+h y t (cambio esperado de temperatura, BLH, etc.)

    Para source='forecast' y use_previous_runs=True:
    - h=24 usa '_previous_day1' (emitido ~24h antes) si está disponible, con fallback al base.
    - h=48 usa '_previous_day2' (emitido ~48h antes) si está disponible, con fallback al base.
    - h in [1, 4, 8, 12] usa las variables base del pronóstico numérico.
    """
    df = meteo_df.sort_values(['Estacion', 'Date']).copy()

    # Determinar sufijo de variables para el objetivo según el horizonte y modo
    prev_suffix = ''
    if source == 'forecast' and use_previous_runs:
        if h == 24 and 'temperature_2m_previous_day1' in df.columns:
            prev_suffix = '_previous_day1'
        elif h == 48 and 'temperature_2m_previous_day2' in df.columns:
            prev_suffix = '_previous_day2'

    out_groups = []
    for est, g in df.groupby('Estacion', sort=False):
        g = g.copy()

        # Variable mapping para t+h
        def get_col(base_name):
            if prev_suffix:
                cand = f"{base_name}{prev_suffix}"
                if cand in g.columns:
                    # Fallback suave al base si cand es nulo (por ejemplo en 2020)
                    return g[cand].fillna(g[base_name]) if base_name in g.columns else g[cand]
            return g[base_name]

        temp_series = get_col('temperature_2m')
        rh_series = get_col('relative_humidity_2m')
        rad_series = get_col('shortwave_radiation')
        cloud_series = get_col('cloud_cover')
        ws_series = get_col('wind_speed_10m')
        pres_series = get_col('surface_pressure')
        precip_series = get_col('precipitation')
        dew_series = get_col('dew_point_2m') if 'dew_point_2m' in g.columns else pd.Series(np.nan, index=g.index)
        blh_series = get_col('boundary_layer_height') if 'boundary_layer_height' in g.columns else pd.Series(np.nan, index=g.index)

        # Viento U, V
        u_col_name = f'U_Wind_10m{prev_suffix}' if prev_suffix and f'U_Wind_10m{prev_suffix}' in g.columns else 'U_Wind_10m'
        v_col_name = f'V_Wind_10m{prev_suffix}' if prev_suffix and f'V_Wind_10m{prev_suffix}' in g.columns else 'V_Wind_10m'
        u_series = g[u_col_name] if u_col_name in g.columns else pd.Series(np.nan, index=g.index)
        v_series = g[v_col_name] if v_col_name in g.columns else pd.Series(np.nan, index=g.index)

        # 1. Features en t (instante de emisión, shift(0))
        g['meteo_t0_temp'] = g['temperature_2m']
        g['meteo_t0_rh'] = g['relative_humidity_2m']
        g['meteo_t0_rad'] = g['shortwave_radiation']
        g['meteo_t0_ws'] = g['wind_speed_10m']
        g['meteo_t0_pres'] = g['surface_pressure']
        if 'boundary_layer_height' in g.columns:
            g['meteo_t0_blh'] = g['boundary_layer_height']
        if 'U_Wind_10m' in g.columns:
            g['meteo_t0_u'] = g['U_Wind_10m']
            g['meteo_t0_v'] = g['V_Wind_10m']

        # 2. Features en el momento objetivo (t + h) -> shift(-h)
        g['meteo_target_temp'] = temp_series.shift(-h)
        g['meteo_target_rh'] = rh_series.shift(-h)
        g['meteo_target_rad'] = rad_series.shift(-h)
        g['meteo_target_cloud'] = cloud_series.shift(-h)
        g['meteo_target_ws'] = ws_series.shift(-h)
        g['meteo_target_pres'] = pres_series.shift(-h)
        g['meteo_target_precip'] = precip_series.shift(-h)
        g['meteo_target_dew'] = dew_series.shift(-h)
        g['meteo_target_blh'] = blh_series.shift(-h)
        g['meteo_target_u'] = u_series.shift(-h)
        g['meteo_target_v'] = v_series.shift(-h)

        # 3. Ventana móvil [t+h-3 .. t+h] (ventana de 4 horas terminando en t+h)
        # rolling(4).max() en una serie da el max sobre los últimos 4 pasos (tau-3..tau).
        # Para que esté alineado con la fila t donde tau = t+h, shift(-h) a la serie rolling:
        g['meteo_target_temp_max_3h'] = temp_series.rolling(4, min_periods=2).max().shift(-h)
        g['meteo_target_rad_sum_3h'] = rad_series.rolling(4, min_periods=2).sum().shift(-h)
        g['meteo_target_rh_mean_3h'] = rh_series.rolling(4, min_periods=2).mean().shift(-h)
        g['meteo_target_ws_mean_3h'] = ws_series.rolling(4, min_periods=2).mean().shift(-h)
        g['meteo_target_blh_mean_3h'] = blh_series.rolling(4, min_periods=2).mean().shift(-h)

        # 4. Deltas: cambio de t a t+h
        g['meteo_delta_temp'] = g['meteo_target_temp'] - g['meteo_t0_temp']
        g['meteo_delta_blh'] = g['meteo_target_blh'] - g.get('meteo_t0_blh', np.nan)
        g['meteo_delta_rh'] = g['meteo_target_rh'] - g['meteo_t0_rh']

        # Interacciones termodinámicas/fotoquímicas en t+h:
        # Radiación * Temperatura
        g['meteo_target_temp_x_rad'] = g['meteo_target_temp'] * g['meteo_target_rad']
        # Ventilación / Potencial de dilución: Viento * BLH (m^2/s)
        g['meteo_target_ventilation'] = g['meteo_target_ws'] * g['meteo_target_blh']

        out_groups.append(g)

    res = pd.concat(out_groups, ignore_index=True)
    
    # Mantener solo columnas Date, Estacion y las columnas meteo generadas
    meteo_feature_cols = [c for c in res.columns if c.startswith('meteo_')]
    return res[['Date', 'Estacion'] + meteo_feature_cols], meteo_feature_cols


def merge_meteo_with_v2(
    v2_df: pd.DataFrame,
    meteo_df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    Une las características meteorológicas al DataFrame base v2 por ['Date', 'Estacion'].
    Devuelve (merged_df, lista_de_columnas_meteo).
    """
    meteo_cols = [c for c in meteo_df.columns if c not in ['Date', 'Estacion']]
    merged = pd.merge(v2_df, meteo_df, on=['Date', 'Estacion'], how='left')
    return merged, meteo_cols
