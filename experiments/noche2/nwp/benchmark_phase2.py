import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import os
import sys

sys.path.insert(0, '.')
from experiments.common import SEED

CACHE_DIR = 'experiments/noche/v3_base/cache'
DATA_DIR = 'experiments/noche2/nwp'

print("Loading data for benchmark on h=24...")
df_base = pd.read_parquet(os.path.join(CACHE_DIR, 'features_h24.parquet'))
df_nwp = pd.read_parquet(os.path.join(DATA_DIR, 'features_nwp_h24.parquet'))

df = pd.merge(df_base, df_nwp, on=['Date', 'Estacion'], how='inner')
df['Estacion'] = df['Estacion'].astype('category')
df['quarter'] = df['Date'].dt.quarter

t0_cols = [c for c in df.columns if c.startswith('meteo_t0_')]
honest_cols = [c for c in df.columns if c.startswith('honest_meteo_')]
nwp_cols = [c for c in df_nwp.columns if c not in ['Date', 'Estacion']]
exclude = {'Date', 'Estacion', 'split', 'Target', 'Target_24h', 'quarter'} | set(t0_cols) | set(honest_cols) | set(nwp_cols)
chem_cols = [c for c in df.columns if c not in exclude and not c.startswith('pp_meteo_')]

features = ['Estacion'] + chem_cols + t0_cols + honest_cols + nwp_cols + ['is_nwp']

# Filter train (2020-2023 + Q2,Q3,Q4 2024) and val (Q1 2024)
is_train_base = (df['split'] == 'train') & df['Target'].notna()
is_val_q1 = (df['split'] == 'val') & (df['quarter'] == 1) & df['Target'].notna()
is_val_others = (df['split'] == 'val') & (df['quarter'] != 1) & df['Target'].notna()

tr_a = df[is_train_base].copy()
tr_a['is_nwp'] = 0.0

tr_b = df[is_val_others].copy()
tr_b['is_nwp'] = 1.0

tr_fold = pd.concat([tr_a, tr_b], ignore_index=True)
val_fold = df[is_val_q1].copy()
val_fold['is_nwp'] = 1.0

print(f"Train rows: {len(tr_fold)}, Val rows: {len(val_fold)}, Features: {len(features)}")

# 1. Benchmark XGBoost
print("\n--- Testing XGBoost (n_jobs=4, hist) ---")
t0 = time.time()
xgb_model = xgb.XGBRegressor(
    tree_method='hist',
    enable_categorical=True,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=4,
    early_stopping_rounds=40,
    eval_metric='rmse'
)

xgb_model.fit(
    tr_fold[features], tr_fold['Target'],
    eval_set=[(val_fold[features], val_fold['Target'])],
    verbose=False
)
t_xgb = time.time() - t0
p_xgb = xgb_model.predict(val_fold[features])
rmse_xgb = np.sqrt(np.mean((val_fold['Target'] - p_xgb)**2))
print(f"XGBoost finished in {t_xgb:.1f}s | Best iter: {xgb_model.best_iteration} | Fold Q1 RMSE: {rmse_xgb:.4f}")

# 2. Benchmark LightGBM ExtraTrees
print("\n--- Testing LightGBM ExtraTrees (num_threads=4) ---")
t0 = time.time()
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'seed': SEED,
    'verbose': -1,
    'num_threads': 4,
    'extra_trees': True,
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'subsample_freq': 1,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 50
}
lgb_tr = lgb.Dataset(tr_fold[features], tr_fold['Target'], categorical_feature=['Estacion'], free_raw_data=False)
lgb_va = lgb.Dataset(val_fold[features], val_fold['Target'], reference=lgb_tr, categorical_feature=['Estacion'], free_raw_data=False)
lgb_model = lgb.train(
    lgb_params,
    lgb_tr,
    num_boost_round=1000,
    valid_sets=[lgb_va],
    callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)]
)
t_lgb = time.time() - t0
p_lgb = lgb_model.predict(val_fold[features], num_iteration=lgb_model.best_iteration)
rmse_lgb = np.sqrt(np.mean((val_fold['Target'] - p_lgb)**2))
print(f"LGB ExtraTrees finished in {t_lgb:.1f}s | Best iter: {lgb_model.best_iteration} | Fold Q1 RMSE: {rmse_lgb:.4f}")
