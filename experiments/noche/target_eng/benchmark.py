import time
import json
import lightgbm as lgb
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
EXP_DIR = os.path.join(ROOT, 'experiments')
if EXP_DIR not in sys.path: sys.path.insert(0, EXP_DIR)

from common import split, evaluate_test
from noche.target_eng.data_prep import get_v4_data

df, feats = get_v4_data()
tr, va, ts = split(df)
tr1 = tr.dropna(subset=['Target_1h'])
va1 = va.dropna(subset=['Target_1h'])
ts1 = ts.dropna(subset=['Target_1h'])

with open(os.path.join(EXP_DIR, 'planA_quickwins', 'best_params_optuna.json')) as f:
    p = json.load(f)
p.update({'objective': 'regression', 'metric': 'rmse', 'num_threads': 3, 'seed': 42, 'verbose': -1})

ds_tr = lgb.Dataset(tr1[feats], tr1['Target_1h'], categorical_feature=['Estacion'])
ds_va = lgb.Dataset(va1[feats], va1['Target_1h'], reference=ds_tr, categorical_feature=['Estacion'])

t0 = time.time()
m = lgb.train(p, ds_tr, num_boost_round=1500, valid_sets=[ds_va], callbacks=[lgb.early_stopping(60, verbose=False)])
t_train = time.time() - t0
preds = m.predict(ts1[feats])
res = evaluate_test(ts1, preds, 1)
print(f"Trained h=1 in {t_train:.1f}s, iters={m.best_iteration}, test RMSE: {res['RMSE']:.4f}")
