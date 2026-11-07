import os, sys, time, pandas as pd, numpy as np, lightgbm as lgb

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import SEED

df = pd.read_parquet(os.path.join(ROOT, 'experiments/noche/v3_base/cache/features_h1.parquet'))
df['Estacion'] = df['Estacion'].astype('category')
t0_cols = [c for c in df.columns if c.startswith('meteo_t0_')]
honest_cols = [c for c in df.columns if c.startswith('honest_meteo_')]
pp_cols = [c for c in df.columns if c.startswith('pp_meteo_')]
exclude = {'Date', 'Estacion', 'split', 'Target', 'Target_1h'} | set(t0_cols) | set(honest_cols) | set(pp_cols)
chem_cols = [c for c in df.columns if c not in exclude]
features = ['Estacion'] + chem_cols + t0_cols + honest_cols

tr = df[(df['split'] == 'train') & df['Target'].notna()]
tr_sub = tr[tr['Date'].dt.year.isin([2021, 2022, 2023])]
val_sub = tr[tr['Date'].dt.year == 2020]

t0 = time.time()
lgb_tr = lgb.Dataset(tr_sub[features], tr_sub['Target'], categorical_feature=['Estacion'], free_raw_data=False)
lgb_va = lgb.Dataset(val_sub[features], val_sub['Target'], reference=lgb_tr, categorical_feature=['Estacion'], free_raw_data=False)

params = {'objective': 'regression', 'metric': 'rmse', 'seed': SEED, 'verbose': -1, 'num_threads': 3, 'learning_rate': 0.05, 'num_leaves': 63}
m = lgb.train(params, lgb_tr, num_boost_round=300, valid_sets=[lgb_va], callbacks=[lgb.early_stopping(50, verbose=False)])
pred = m.predict(val_sub[features])
rmse = float(np.sqrt(np.mean((val_sub['Target'] - pred)**2)))
print(f"Done in {time.time()-t0:.1f}s! Best iter: {m.best_iteration}, 2020 RMSE: {rmse:.4f}")
