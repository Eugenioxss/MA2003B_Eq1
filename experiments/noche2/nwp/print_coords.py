import sys
import os
import pandas as pd

sys.path.insert(0, '.')
from experiments.common import STATION_COORDS, MODEL_STATIONS

print(f"{'Station':<8} {'Lat':<10} {'Lon':<10} {'Elev':<6}")
print("-" * 38)
for st in MODEL_STATIONS:
    lat, lon, elev = STATION_COORDS[st]
    print(f"{st:<8} {lat:<10.4f} {lon:<10.4f} {elev:<6}")
