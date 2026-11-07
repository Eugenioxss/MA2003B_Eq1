import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from experiments.common import load_panel, MODEL_STATIONS

print("Loading data for MOS test...")
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
merged['month'] = merged['Date'].dt.month
merged['year'] = merged['Date'].dt.year

# Train MOS on 2020-2023 (or <=2023)
train = merged[merged['year'] <= 2023]
val = merged[merged['year'] == 2024]
test = merged[merged['year'] == 2025]

# Fit MOS for Temperature: TOUT ~ temperature_2m by (Estacion, hour)
# We use linear regression: TOUT = a + b * temp_2m
mos_temp_params = {}
for (st, hr), g in train.groupby(['Estacion', 'hour']):
    valid = g[['TOUT', 'temperature_2m']].dropna()
    if len(valid) > 20:
        b, a = np.polyfit(valid['temperature_2m'], valid['TOUT'], deg=1)
        # bound slope to reasonable range [0.5, 1.5]
        b = np.clip(b, 0.5, 1.5)
        mos_temp_params[(st, hr)] = (a, b)
    else:
        mos_temp_params[(st, hr)] = (0.0, 1.0)

# Evaluate on Val 2024: previous_day1 vs MOS(previous_day1)
val_valid = val[['Estacion', 'hour', 'TOUT', 'temperature_2m_previous_day1']].dropna().copy()
preds_raw = val_valid['temperature_2m_previous_day1'].values
preds_mos = []
for _, row in val_valid.iterrows():
    a, b = mos_temp_params.get((row['Estacion'], int(row['hour'])), (0.0, 1.0))
    preds_mos.append(a + b * row['temperature_2m_previous_day1'])

preds_mos = np.array(preds_mos)
y_true = val_valid['TOUT'].values

rmse_raw = np.sqrt(np.mean((preds_raw - y_true)**2))
mae_raw = np.mean(np.abs(preds_raw - y_true))
bias_raw = np.mean(preds_raw - y_true)

rmse_mos = np.sqrt(np.mean((preds_mos - y_true)**2))
mae_mos = np.mean(np.abs(preds_mos - y_true))
bias_mos = np.mean(preds_mos - y_true)

print("\n--- Temperature Val 2024 Evaluation ---")
print(f"Raw NWP prev_day1: Bias = {bias_raw:+.3f}°C, MAE = {mae_raw:.3f}°C, RMSE = {rmse_raw:.3f}°C")
print(f"MOS-calibrated:    Bias = {bias_mos:+.3f}°C, MAE = {mae_mos:.3f}°C, RMSE = {rmse_mos:.3f}°C")
print(f"Error reduction:   RMSE dropped by {rmse_raw - rmse_mos:.3f}°C ({(1 - rmse_mos/rmse_raw)*100:.1f}%)")

# Test on 2025 Test set
test_valid = test[['Estacion', 'hour', 'TOUT', 'temperature_2m_previous_day1']].dropna().copy()
preds_te_raw = test_valid['temperature_2m_previous_day1'].values
preds_te_mos = []
for _, row in test_valid.iterrows():
    a, b = mos_temp_params.get((row['Estacion'], int(row['hour'])), (0.0, 1.0))
    preds_te_mos.append(a + b * row['temperature_2m_previous_day1'])

preds_te_mos = np.array(preds_te_mos)
y_te_true = test_valid['TOUT'].values

rmse_te_raw = np.sqrt(np.mean((preds_te_raw - y_te_true)**2))
rmse_te_mos = np.sqrt(np.mean((preds_te_mos - y_te_true)**2))
bias_te_raw = np.mean(preds_te_raw - y_te_true)
bias_te_mos = np.mean(preds_te_mos - y_te_true)

print("\n--- Temperature Test 2025 Evaluation ---")
print(f"Raw NWP prev_day1: Bias = {bias_te_raw:+.3f}°C, RMSE = {rmse_te_raw:.3f}°C")
print(f"MOS-calibrated:    Bias = {bias_te_mos:+.3f}°C, RMSE = {rmse_te_mos:.3f}°C")
