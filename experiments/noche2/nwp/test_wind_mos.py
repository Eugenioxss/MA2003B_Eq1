import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from experiments.common import load_panel, MODEL_STATIONS

print("Loading data for Wind and Radiation MOS...")
panel = load_panel()

df_meteo_list = []
for st in MODEL_STATIONS:
    p = os.path.join('experiments', 'planC_meteo', 'data', f'forecast_{st}.parquet')
    if os.path.exists(p):
        df = pd.read_parquet(p)
        df['Estacion'] = st
        df_meteo_list.append(df)

df_meteo = pd.concat(df_meteo_list, ignore_index=True)
merged = pd.merge(panel, df_meteo, on=['Date', 'Estacion'], how='inner')
merged['hour'] = merged['Date'].dt.hour
merged['year'] = merged['Date'].dt.year

train = merged[merged['year'] <= 2023]
val = merged[merged['year'] == 2024]
test = merged[merged['year'] == 2025]

# 1. Wind Speed MOS: WSR ~ wind_speed_10m by (Estacion, hour)
mos_ws_params = {}
for (st, hr), g in train.groupby(['Estacion', 'hour']):
    valid = g[['WSR', 'wind_speed_10m']].dropna()
    if len(valid) > 20:
        b, a = np.polyfit(valid['wind_speed_10m'], valid['WSR'], deg=1)
        b = np.clip(b, 0.1, 2.0)
        mos_ws_params[(st, hr)] = (a, b)
    else:
        mos_ws_params[(st, hr)] = (0.0, 1.0)

# Evaluate Wind Speed on Val 2024
val_ws = val[['Estacion', 'hour', 'WSR', 'wind_speed_10m_previous_day1']].dropna().copy()
preds_ws_raw = val_ws['wind_speed_10m_previous_day1'].values
preds_ws_mos = []
for _, row in val_ws.iterrows():
    a, b = mos_ws_params.get((row['Estacion'], int(row['hour'])), (0.0, 1.0))
    preds_ws_mos.append(max(0.1, a + b * row['wind_speed_10m_previous_day1']))
preds_ws_mos = np.array(preds_ws_mos)
y_ws_val = val_ws['WSR'].values

rmse_ws_raw = np.sqrt(np.mean((preds_ws_raw - y_ws_val)**2))
rmse_ws_mos = np.sqrt(np.mean((preds_ws_mos - y_ws_val)**2))
bias_ws_raw = np.mean(preds_ws_raw - y_ws_val)
bias_ws_mos = np.mean(preds_ws_mos - y_ws_val)

print(f"\n--- Wind Speed Val 2024 ---")
print(f"Raw NWP prev_day1: Bias = {bias_ws_raw:+.3f} km/h, RMSE = {rmse_ws_raw:.3f} km/h")
print(f"MOS-calibrated:    Bias = {bias_ws_mos:+.3f} km/h, RMSE = {rmse_ws_mos:.3f} km/h")
print(f"Error reduction:   RMSE dropped by {rmse_ws_raw - rmse_ws_mos:.3f} km/h ({(1 - rmse_ws_mos/rmse_ws_raw)*100:.1f}%)")

# Test on 2025 Test set
test_ws = test[['Estacion', 'hour', 'WSR', 'wind_speed_10m_previous_day1']].dropna().copy()
preds_te_ws_raw = test_ws['wind_speed_10m_previous_day1'].values
preds_te_ws_mos = []
for _, row in test_ws.iterrows():
    a, b = mos_ws_params.get((row['Estacion'], int(row['hour'])), (0.0, 1.0))
    preds_te_ws_mos.append(max(0.1, a + b * row['wind_speed_10m_previous_day1']))
preds_te_ws_mos = np.array(preds_te_ws_mos)
y_ws_test = test_ws['WSR'].values

rmse_te_ws_raw = np.sqrt(np.mean((preds_te_ws_raw - y_ws_test)**2))
rmse_te_ws_mos = np.sqrt(np.mean((preds_te_ws_mos - y_ws_test)**2))
bias_te_ws_raw = np.mean(preds_te_ws_raw - y_ws_test)
bias_te_ws_mos = np.mean(preds_te_ws_mos - y_ws_test)

print(f"\n--- Wind Speed Test 2025 ---")
print(f"Raw NWP prev_day1: Bias = {bias_te_ws_raw:+.3f} km/h, RMSE = {rmse_te_ws_raw:.3f} km/h")
print(f"MOS-calibrated:    Bias = {bias_te_ws_mos:+.3f} km/h, RMSE = {rmse_te_ws_mos:.3f} km/h")
print(f"Error reduction:   RMSE dropped by {rmse_te_ws_raw - rmse_te_ws_mos:.3f} km/h ({(1 - rmse_te_ws_mos/rmse_te_ws_raw)*100:.1f}%)")
