import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from experiments.common import load_panel, MODEL_STATIONS

print("Loading SIMA panel and NWP data...")
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
print(f"Merged rows: {len(merged)}")

# Filter val 2024 to inspect NWP vs SIMA
val = merged[merged['Date'].dt.year == 2024]

print("\n--- Val 2024 NWP vs SIMA Ground Truth ---")
# 1. Temperature: TOUT vs temperature_2m_previous_day1
valid_t = val[['TOUT', 'temperature_2m_previous_day1', 'temperature_2m']].dropna()
bias_t1 = (valid_t['temperature_2m_previous_day1'] - valid_t['TOUT']).mean()
rmse_t1 = np.sqrt(((valid_t['temperature_2m_previous_day1'] - valid_t['TOUT'])**2).mean())
corr_t1 = valid_t['TOUT'].corr(valid_t['temperature_2m_previous_day1'])
print(f"Temperature (prev_day1 vs TOUT): Bias = {bias_t1:+.2f}°C, RMSE = {rmse_t1:.2f}°C, Corr = {corr_t1:.3f}")

# 2. Solar radiation: SR vs shortwave_radiation_previous_day1
valid_sr = val[['SR', 'shortwave_radiation_previous_day1', 'shortwave_radiation']].dropna()
bias_sr1 = (valid_sr['shortwave_radiation_previous_day1'] - valid_sr['SR']).mean()
rmse_sr1 = np.sqrt(((valid_sr['shortwave_radiation_previous_day1'] - valid_sr['SR'])**2).mean())
corr_sr1 = valid_sr['SR'].corr(valid_sr['shortwave_radiation_previous_day1'])
print(f"Radiation (prev_day1 vs SR): Bias = {bias_sr1:+.2f} W/m2, RMSE = {rmse_sr1:.2f} W/m2, Corr = {corr_sr1:.3f}")

# 3. Wind speed: WSR vs wind_speed_10m_previous_day1
valid_ws = val[['WSR', 'wind_speed_10m_previous_day1', 'wind_speed_10m']].dropna()
bias_ws1 = (valid_ws['wind_speed_10m_previous_day1'] - valid_ws['WSR']).mean()
rmse_ws1 = np.sqrt(((valid_ws['wind_speed_10m_previous_day1'] - valid_ws['WSR'])**2).mean())
corr_ws1 = valid_ws['WSR'].corr(valid_ws['wind_speed_10m_previous_day1'])
print(f"Wind Speed (prev_day1 vs WSR): Bias = {bias_ws1:+.2f} m/s, RMSE = {rmse_ws1:.2f} m/s, Corr = {corr_ws1:.3f}")

# 4. Relative humidity: RH vs relative_humidity_2m_previous_day1
valid_rh = val[['RH', 'relative_humidity_2m_previous_day1', 'relative_humidity_2m']].dropna()
bias_rh1 = (valid_rh['relative_humidity_2m_previous_day1'] - valid_rh['RH']).mean()
rmse_rh1 = np.sqrt(((valid_rh['relative_humidity_2m_previous_day1'] - valid_rh['RH'])**2).mean())
corr_rh1 = valid_rh['RH'].corr(valid_rh['relative_humidity_2m_previous_day1'])
print(f"Humidity (prev_day1 vs RH): Bias = {bias_rh1:+.2f}%, RMSE = {rmse_rh1:.2f}%, Corr = {corr_rh1:.3f}")

# Let's inspect by station
print("\n--- Bias by Station in 2024 ---")
for st in MODEL_STATIONS[:5]:
    sub = val[val['Estacion'] == st]
    t_bias = (sub['temperature_2m_previous_day1'] - sub['TOUT']).mean()
    ws_bias = (sub['wind_speed_10m_previous_day1'] - sub['WSR']).mean()
    print(f"Station {st}: T bias = {t_bias:+.2f}°C, Wind bias = {ws_bias:+.2f} m/s")
