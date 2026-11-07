"""
Test full data preparation and batch speed for TiDE.
"""
import sys, os, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import (
    load_panel, MODEL_STATIONS, STATION_COORDS, HORIZONS, SEED
)

t0 = time.time()
print("1. Loading panel...")
panel = load_panel(MODEL_STATIONS).sort_values(['Estacion', 'Date']).reset_index(drop=True)

stations = MODEL_STATIONS
num_stations = len(stations)
st_to_idx = {s: i for i, s in enumerate(stations)}
dates = pd.date_range('2020-01-01 00:00', '2025-12-31 23:00', freq='h')
num_hours = len(dates)

print(f"Panel ready in {time.time()-t0:.2f}s")

# 2. Climatology (Station x Month x Hour) on Train (<= 2023)
t0 = time.time()
train_mask = panel['Date'].dt.year <= 2023
clim_table = panel[train_mask].groupby(['Estacion', panel.loc[train_mask, 'Date'].dt.month, panel.loc[train_mask, 'Date'].dt.hour])['O3'].mean()

# Fast 3D lookup array for Climatology: (13, 12, 24)
clim_array = np.zeros((num_stations, 12, 24), dtype=np.float32)
for (s, m, h), val in clim_table.items():
    clim_array[st_to_idx[s], m - 1, h] = val

# Precompute climatology for every (station, hour)
# month-1 for all dates, hour for all dates
months = dates.month.values - 1  # 0..11
hours = dates.hour.values        # 0..23
clim_full = np.zeros((num_stations, num_hours), dtype=np.float32)
for s_idx in range(num_stations):
    clim_full[s_idx] = clim_array[s_idx, months, hours]

# Residual std on train
train_hours_count = 35064
o3_train = panel.loc[train_mask, 'O3'].values
clim_train_vals = clim_full[:, :train_hours_count].ravel()
res_train = o3_train - clim_train_vals
valid_res = np.isfinite(res_train)
sigma_res = float(np.std(res_train[valid_res]))
print(f"Climatology array ready in {time.time()-t0:.2f}s, sigma_res: {sigma_res:.4f}")

# 3. Physical variables in past
PAST_VARS = ['O3', 'NOX', 'NO2', 'NO', 'CO', 'PM10', 'PM2.5', 'PRS', 'RAINF', 'RH', 'SO2', 'SR', 'TOUT', 'WSR', 'U_Wind', 'V_Wind']
print(f"Past variables ({len(PAST_VARS)}): {PAST_VARS}")

# Compute train mean and std for each variable
var_stats = {}
for v in PAST_VARS:
    vals = panel.loc[train_mask, v].values
    val_ok = vals[np.isfinite(vals)]
    mean_val = float(np.mean(val_ok)) if len(val_ok) > 0 else 0.0
    std_val = float(np.std(val_ok)) if len(val_ok) > 0 and np.std(val_ok) > 1e-6 else 1.0
    var_stats[v] = (mean_val, std_val)

# Build past tensor per station: (num_stations, num_hours, num_channels)
# channels: 16 normalized + 16 masks + 4 calendar = 36 channels
num_past_channels = len(PAST_VARS) * 2 + 4
past_tensor = np.zeros((num_stations, num_hours, num_past_channels), dtype=np.float32)

hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
month_sin = np.sin(2 * np.pi * (months + 1) / 12).astype(np.float32)
month_cos = np.cos(2 * np.pi * (months + 1) / 12).astype(np.float32)

for s_idx, s in enumerate(stations):
    sub = panel[panel['Estacion'] == s]
    for v_idx, v in enumerate(PAST_VARS):
        raw = sub[v].values
        mask = np.isnan(raw).astype(np.float32)
        mean_v, std_v = var_stats[v]
        norm = np.where(mask == 0, (raw - mean_v) / std_v, 0.0).astype(np.float32)
        past_tensor[s_idx, :, v_idx] = norm
        past_tensor[s_idx, :, len(PAST_VARS) + v_idx] = mask
    
    # calendar channels
    base_cal = len(PAST_VARS) * 2
    past_tensor[s_idx, :, base_cal + 0] = hour_sin
    past_tensor[s_idx, :, base_cal + 1] = hour_cos
    past_tensor[s_idx, :, base_cal + 2] = month_sin
    past_tensor[s_idx, :, base_cal + 3] = month_cos

print(f"Past tensor shape: {past_tensor.shape}, size: {past_tensor.nbytes / 1e6:.1f} MB")

# 4. Load Forecast Meteo
t0 = time.time()
METEO_DIR = os.path.join(ROOT, 'experiments', 'planC_meteo', 'data')
METEO_BASE = ['temperature_2m', 'relative_humidity_2m', 'shortwave_radiation', 'cloud_cover',
              'wind_speed_10m', 'surface_pressure', 'precipitation', 'dew_point_2m']

# We will build:
# meteo_seamless: (num_stations, num_hours, num_meteo_channels)
# meteo_day1:     (num_stations, num_hours, num_meteo_channels)
# meteo_day2:     (num_stations, num_hours, num_meteo_channels)
# meteo_channels: 8 base + U + V + temp_x_rad + 7 calendar = 18 channels

# Compute meteo stats on train (seamless)
meteo_dfs = {}
for s in stations:
    fpath = os.path.join(METEO_DIR, f'forecast_{s}.parquet')
    df_m = pd.read_parquet(fpath)
    meteo_dfs[s] = df_m

# Fit scalers on train seamless across all stations
train_meteo_vals = {v: [] for v in METEO_BASE}
for s in stations:
    df_m = meteo_dfs[s]
    sub_tr = df_m[df_m['Date'].dt.year <= 2023]
    for v in METEO_BASE:
        train_meteo_vals[v].append(sub_tr[v].values)

meteo_stats = {}
for v in METEO_BASE:
    all_v = np.concatenate(train_meteo_vals[v])
    meteo_stats[v] = (float(np.nanmean(all_v)), float(np.nanstd(all_v)))

num_meteo_feats = len(METEO_BASE) + 3 + 7  # 8 base + U + V + TxR + 7 calendar = 18
meteo_seamless = np.zeros((num_stations, num_hours, num_meteo_feats), dtype=np.float32)
meteo_day1     = np.zeros((num_stations, num_hours, num_meteo_feats), dtype=np.float32)
meteo_day2     = np.zeros((num_stations, num_hours, num_meteo_feats), dtype=np.float32)

dow = dates.dayofweek.values
dow_sin = np.sin(2 * np.pi * dow / 7).astype(np.float32)
dow_cos = np.cos(2 * np.pi * dow / 7).astype(np.float32)
is_weekend = (dow >= 5).astype(np.float32)

for s_idx, s in enumerate(stations):
    df_m = meteo_dfs[s]
    
    # Process each source: seamless, day1, day2
    for mode, target_arr in [('seamless', meteo_seamless), ('day1', meteo_day1), ('day2', meteo_day2)]:
        suffix = '' if mode == 'seamless' else f'_previous_{mode}'
        ws_col = f'wind_speed_10m{suffix}' if suffix and f'wind_speed_10m{suffix}' in df_m.columns else 'wind_speed_10m'
        wd_col = f'wind_direction_10m{suffix}' if suffix and f'wind_direction_10m{suffix}' in df_m.columns else 'wind_direction_10m'
        
        ws = df_m[ws_col].fillna(df_m['wind_speed_10m']).values
        wd = df_m[wd_col].fillna(df_m['wind_direction_10m']).values
        rad = np.deg2rad(wd)
        u_wind = (-ws * np.sin(rad)).astype(np.float32)
        v_wind = (-ws * np.cos(rad)).astype(np.float32)
        
        temp_col = f'temperature_2m{suffix}' if suffix and f'temperature_2m{suffix}' in df_m.columns else 'temperature_2m'
        rad_col = f'shortwave_radiation{suffix}' if suffix and f'shortwave_radiation{suffix}' in df_m.columns else 'shortwave_radiation'
        temp = df_m[temp_col].fillna(df_m['temperature_2m']).values
        sw_rad = df_m[rad_col].fillna(df_m['shortwave_radiation']).values
        txr = (temp * sw_rad).astype(np.float32)
        
        # Populate base
        for v_idx, v in enumerate(METEO_BASE):
            col_name = f'{v}{suffix}' if suffix and f'{v}{suffix}' in df_m.columns else v
            vals = df_m[col_name].fillna(df_m[v]).values
            m_mean, m_std = meteo_stats[v]
            target_arr[s_idx, :, v_idx] = ((vals - m_mean) / (m_std + 1e-6)).astype(np.float32)
        
        # Populate U, V, TxR
        target_arr[s_idx, :, len(METEO_BASE) + 0] = u_wind / 5.0
        target_arr[s_idx, :, len(METEO_BASE) + 1] = v_wind / 5.0
        target_arr[s_idx, :, len(METEO_BASE) + 2] = (txr - 2000.0) / 4000.0
        
        # Populate calendar (7)
        c_base = len(METEO_BASE) + 3
        target_arr[s_idx, :, c_base + 0] = hour_sin
        target_arr[s_idx, :, c_base + 1] = hour_cos
        target_arr[s_idx, :, c_base + 2] = month_sin
        target_arr[s_idx, :, c_base + 3] = month_cos
        target_arr[s_idx, :, c_base + 4] = dow_sin
        target_arr[s_idx, :, c_base + 5] = dow_cos
        target_arr[s_idx, :, c_base + 6] = is_weekend

print(f"Meteo tensors ready in {time.time()-t0:.2f}s, shape: {meteo_seamless.shape}")
print("Data preparation test passed successfully!")
