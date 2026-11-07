import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from experiments.common import MODEL_STATIONS

dfs = {}
for st in MODEL_STATIONS:
    p = f'experiments/planC_meteo/data/forecast_{st}.parquet'
    if os.path.exists(p):
        dfs[st] = pd.read_parquet(p)['temperature_2m'].values

print("Comparing temperature_2m across all pairs of stations:")
unique_groups = []
seen = set()
for s1 in MODEL_STATIONS:
    if s1 in seen:
        continue
    grp = [s1]
    seen.add(s1)
    for s2 in MODEL_STATIONS:
        if s2 in seen:
            continue
        if np.allclose(dfs[s1], dfs[s2], equal_nan=True):
            grp.append(s2)
            seen.add(s2)
    unique_groups.append(grp)

for i, grp in enumerate(unique_groups, 1):
    print(f"Group {i} ({grp[0]} as rep): {grp}")
