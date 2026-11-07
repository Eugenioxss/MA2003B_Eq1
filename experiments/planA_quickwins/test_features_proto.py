import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('experiments'))
from common import load_panel, ALL_STATIONS, MODEL_STATIONS, STATION_COORDS

print("Testing feature construction logic...")
t0 = time.time()
panel = load_panel()
print(f"Panel loaded: {panel.shape} in {time.time()-t0:.2f}s")

# Let's inspect columns
print("Panel columns:", panel.columns.tolist())
