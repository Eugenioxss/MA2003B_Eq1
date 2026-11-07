import sys
import pandas as pd
import numpy as np

# Compare forecast_CE and forecast_SE in planC_meteo
df_ce = pd.read_parquet('experiments/planC_meteo/data/forecast_CE.parquet')
df_se = pd.read_parquet('experiments/planC_meteo/data/forecast_SE.parquet')

diff_t = np.abs(df_ce['temperature_2m'] - df_se['temperature_2m']).max()
diff_rad = np.abs(df_ce['shortwave_radiation'] - df_se['shortwave_radiation']).max()
diff_ws = np.abs(df_ce['wind_speed_10m'] - df_se['wind_speed_10m']).max()
print(f"CE vs SE diffs: Temp max diff = {diff_t}, Rad max diff = {diff_rad}, WS max diff = {diff_ws}")

df_ne = pd.read_parquet('experiments/planC_meteo/data/forecast_NE.parquet')
df_nte2 = pd.read_parquet('experiments/planC_meteo/data/forecast_NTE2.parquet')
diff_t2 = np.abs(df_ne['temperature_2m'] - df_nte2['temperature_2m']).max()
print(f"NE vs NTE2 diffs: Temp max diff = {diff_t2}")
