import sys
import os
import pandas as pd
sys.path.insert(0, '.')
from experiments.common import load_panel

panel = load_panel()
print("WSR quantiles in SIMA:")
print(panel['WSR'].describe())
print("\nOpen-Meteo wind_speed_10m:")
df_fc = pd.read_parquet('experiments/planC_meteo/data/forecast_CE.parquet')
print(df_fc['wind_speed_10m'].describe())
