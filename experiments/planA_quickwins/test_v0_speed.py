import sys
import os
import time

sys.path.insert(0, os.path.abspath('experiments'))
from common import load_panel, build_v2_features, split, evaluate_test, log_result, LGB_PARAMS_V2, MODEL_STATIONS, HORIZONS
import lightgbm as lgb
import pandas as pd
import numpy as np

print("Testing load_panel...")
t0 = time.time()
panel = load_panel()
print(f"Panel loaded in {time.time() - t0:.2f}s. Shape: {panel.shape}")
print(f"Columns: {panel.columns.tolist()}")
print(f"Stations in panel: {panel['Estacion'].unique().tolist()}")

t0 = time.time()
df_v2, feats = build_v2_features(panel)
print(f"v2 features built in {time.time() - t0:.2f}s. Shape: {df_v2.shape}, n_features: {len(feats)}")

tr, val, ts = split(df_v2)
print(f"Split sizes - Train: {len(tr)}, Val: {len(val)}, Test: {len(ts)}")

# Quick test for h=1
target = 'Target_1h'
tr_h = tr.dropna(subset=[target])
val_h = val.dropna(subset=[target])
ts_h = ts.dropna(subset=[target])
print(f"h=1 valid counts - Train: {len(tr_h)}, Val: {len(val_h)}, Test: {len(ts_h)}")

params = dict(LGB_PARAMS_V2)
params['num_threads'] = 4
lgb_tr = lgb.Dataset(tr_h[feats], tr_h[target])
lgb_val = lgb.Dataset(val_h[feats], val_h[target], reference=lgb_tr)

t0 = time.time()
model = lgb.train(
    params,
    lgb_tr,
    num_boost_round=2000,
    valid_sets=[lgb_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=80, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
t_train = time.time() - t0
y_pred = model.predict(ts_h[feats], num_iteration=model.best_iteration)
m = evaluate_test(ts_h, y_pred, 1)
print(f"h=1 trained in {t_train:.2f}s, best_iteration: {model.best_iteration}")
print(f"Metrics h=1: RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}, RMSE_peaks={m['RMSE_peaks']:.4f}, skill={m['skill_vs_clim']:.4f}")
