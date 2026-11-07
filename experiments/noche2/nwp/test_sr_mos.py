import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from experiments.common import load_panel, MODEL_STATIONS

print("Loading data for Radiation MOS...")
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

# Fit MOS for Solar Radiation: SR ~ shortwave_radiation / 1000 by (Estacion, hour)
# for daytime hours (7 to 20)
mos_sr_params = {}
for (st, hr), g in train.groupby(['Estacion', 'hour']):
    if 6 <= hr <= 20:
        valid = g[['SR', 'shortwave_radiation']].dropna()
        if len(valid) > 20:
            x = valid['shortwave_radiation'] / 1000.0
            y = valid['SR']
            b, a = np.polyfit(x, y, deg=1)
            b = np.clip(b, 0.1, 1.5)
            mos_sr_params[(st, hr)] = (a, b)
        else:
            mos_sr_params[(st, hr)] = (0.0, 1.0)
    else:
        mos_sr_params[(st, hr)] = (0.0, 0.0)

# Evaluate on Val 2024
val_sr = val[['Estacion', 'hour', 'SR', 'shortwave_radiation_previous_day1']].dropna().copy()
val_sr = val_sr[val_sr['hour'].between(7, 19)]
preds_sr_raw = (val_sr['shortwave_radiation_previous_day1'] / 1000.0).values
preds_sr_mos = []
for _, row in val_sr.iterrows():
    a, b = mos_sr_params.get((row['Estacion'], int(row['hour'])), (0.0, 1.0))
    preds_sr_mos.append(max(0.0, a + b * (row['shortwave_radiation_previous_day1'] / 1000.0)))
preds_sr_mos = np.array(preds_sr_mos)
y_sr_val = val_sr['SR'].values

rmse_sr_raw = np.sqrt(np.mean((preds_sr_raw - y_sr_val)**2))
rmse_sr_mos = np.sqrt(np.mean((preds_sr_mos - y_sr_val)**2))
bias_sr_raw = np.mean(preds_sr_raw - y_sr_val)
bias_sr_mos = np.mean(preds_sr_mos - y_sr_val)

print(f"\n--- Solar Radiation Val 2024 (Daytime) ---")
print(f"Raw NWP prev_day1 (/1000): Bias = {bias_sr_raw:+.3f} kW/m2, RMSE = {rmse_sr_raw:.3f} kW/m2")
print(f"MOS-calibrated:           Bias = {bias_sr_mos:+.3f} kW/m2, RMSE = {rmse_sr_mos:.3f} kW/m2")
print(f"Error reduction:          RMSE dropped by {rmse_sr_raw - rmse_sr_mos:.3f} kW/m2 ({(1 - rmse_sr_mos/rmse_sr_raw)*100:.1f}%)")

# Test 2025
test_sr = test[['Estacion', 'hour', 'SR', 'shortwave_radiation_previous_day1']].dropna().copy()
test_sr = test_sr[test_sr['hour'].between(7, 19)]
preds_te_sr_raw = (test_sr['shortwave_radiation_previous_day1'] / 1000.0).values
preds_te_sr_mos = []
for _, row in test_sr.iterrows():
    a, b = mos_sr_params.get((row['Estacion'], int(row['hour'])), (0.0, 1.0))
    preds_te_sr_mos.append(max(0.0, a + b * (row['shortwave_radiation_previous_day1'] / 1000.0)))
preds_te_sr_mos = np.array(preds_te_sr_mos)
y_sr_test = test_sr['SR'].values

rmse_te_sr_raw = np.sqrt(np.mean((preds_te_sr_raw - y_sr_test)**2))
rmse_te_sr_mos = np.sqrt(np.mean((preds_te_sr_mos - y_sr_test)**2))
bias_te_sr_raw = np.mean(preds_te_sr_raw - y_sr_test)
bias_te_sr_mos = np.mean(preds_te_sr_mos - y_sr_test)

print(f"\n--- Solar Radiation Test 2025 (Daytime) ---")
print(f"Raw NWP prev_day1 (/1000): Bias = {bias_te_sr_raw:+.3f} kW/m2, RMSE = {rmse_te_sr_raw:.3f} kW/m2")
print(f"MOS-calibrated:           Bias = {bias_te_sr_mos:+.3f} kW/m2, RMSE = {rmse_te_sr_mos:.3f} kW/m2")
print(f"Error reduction:          RMSE dropped by {rmse_te_sr_raw - rmse_te_sr_mos:.3f} kW/m2 ({(1 - rmse_te_sr_mos/rmse_te_sr_raw)*100:.1f}%)")
