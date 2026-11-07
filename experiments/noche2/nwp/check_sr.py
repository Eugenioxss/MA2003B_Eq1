import sys
import os
import pandas as pd
sys.path.insert(0, '.')
from experiments.common import load_panel

panel = load_panel()
print("SR in SIMA (daytime):")
day_sima = panel[panel['Date'].dt.hour == 13]['SR']
print(day_sima.describe())

df_fc = pd.read_parquet('experiments/planC_meteo/data/forecast_CE.parquet')
print("\nOpen-Meteo shortwave_radiation (daytime at 13h):")
day_fc = df_fc[df_fc['Date'].dt.hour == 13]['shortwave_radiation']
print(day_fc.describe())
