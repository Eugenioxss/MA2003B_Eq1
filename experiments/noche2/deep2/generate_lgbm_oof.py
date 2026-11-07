"""
experiments/noche2/deep2/generate_lgbm_oof.py
=============================================
Genera predicciones Out-Of-Fold (OOF) de LightGBM (D_honest_noise) para train 2020-2023
mediante cross-fitting por año (4 folds: deja un año fuera, entrena con los otros 3).
Para val 2024 y test 2025 reutiliza las predicciones oficiales de D_honest_noise_h<h>.parquet.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import partial

print = partial(print, flush=True)

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

from common import HORIZONS, SEED
from experiments.noche.v3_base.train_v3 import load_cache_for_h, inject_meteo_noise, OPTUNA_PATH, BEST_OPTUNA

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'seed': SEED,
    'verbose': -1,
    'num_threads': 3,
    'feature_pre_filter': False,
    'learning_rate': 0.08,
    'num_leaves': 63,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OOF_DIR = os.path.join(BASE_DIR, 'oof')
os.makedirs(OOF_DIR, exist_ok=True)

YEARS = [2020, 2021, 2022, 2023]

def generate_oof_for_h(h: int):
    t0 = time.time()
    out_file = os.path.join(OOF_DIR, f'lgbm_oof_h{h}.parquet')
    if os.path.exists(out_file):
        print(f"OOF para h={h} ya existe en {out_file}, omitiendo.")
        return

    print(f"\n{'='*65}\n  Generando OOF LightGBM para h={h}\n{'='*65}")
    df, chem, t0_cols, honest, pp = load_cache_for_h(h)
    features = ['Estacion'] + chem + t0_cols + honest

    tr_mask = (df['split'] == 'train') & df['Target'].notna()
    va_mask = (df['split'] == 'val') & df['Target'].notna()
    
    tr_df = df[tr_mask].copy().reset_index(drop=True)
    va_df = df[va_mask].copy().reset_index(drop=True)

    # Inyección de ruido en train completo para consistencia
    tr_df = inject_meteo_noise(tr_df, va_df, honest, pp, t0_cols, seed=SEED)

    oof_preds = np.zeros(len(tr_df), dtype=np.float32)

    for val_year in YEARS:
        t_fold = time.time()
        train_idx = tr_df[tr_df['Date'].dt.year != val_year].index
        val_idx = tr_df[tr_df['Date'].dt.year == val_year].index

        X_tr = tr_df.loc[train_idx, features]
        y_tr = tr_df.loc[train_idx, 'Target'].values
        X_va = tr_df.loc[val_idx, features]
        y_va = tr_df.loc[val_idx, 'Target'].values

        lgb_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=['Estacion'], free_raw_data=False)
        lgb_va = lgb.Dataset(X_va, y_va, reference=lgb_tr, categorical_feature=['Estacion'], free_raw_data=False)

        model = lgb.train(
            LGB_PARAMS,
            lgb_tr,
            num_boost_round=600,
            valid_sets=[lgb_va],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )

        fold_preds = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = fold_preds.astype(np.float32)
        fold_rmse = float(np.sqrt(np.mean((y_va - fold_preds)**2)))
        print(f"    Año {val_year} -> Best iter: {model.best_iteration:03d} | RMSE: {fold_rmse:.4f} ({time.time()-t_fold:.1f}s)")

    oof_rmse = float(np.sqrt(np.mean((tr_df['Target'].values - oof_preds)**2)))
    print(f"  OOF Train 2020-2023 Total RMSE h={h}: {oof_rmse:.4f} ({time.time()-t0:.1f}s)")

    # Cargar val y test oficiales de D_honest_noise
    lgb_official_path = os.path.join(ROOT, f'experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet')
    lgb_off = pd.read_parquet(lgb_official_path)

    val_off = lgb_off[lgb_off['split'] == 'val'].reset_index(drop=True)
    test_off = lgb_off[lgb_off['split'] == 'test'].reset_index(drop=True)

    # DataFrame completo OOF
    df_oof_train = pd.DataFrame({
        'Date': tr_df['Date'],
        'Estacion': tr_df['Estacion'],
        'h': h,
        'split': 'train',
        'y_true': tr_df['Target'].values.astype(np.float32),
        'y_pred_lgbm': oof_preds.astype(np.float32)
    })
    df_val = pd.DataFrame({
        'Date': val_off['Date'],
        'Estacion': val_off['Estacion'],
        'h': h,
        'split': 'val',
        'y_true': val_off['y_true'].astype(np.float32),
        'y_pred_lgbm': val_off['y_pred'].astype(np.float32)
    })
    df_test = pd.DataFrame({
        'Date': test_off['Date'],
        'Estacion': test_off['Estacion'],
        'h': h,
        'split': 'test',
        'y_true': test_off['y_true'].astype(np.float32),
        'y_pred_lgbm': test_off['y_pred'].astype(np.float32)
    })

    full_oof = pd.concat([df_oof_train, df_val, df_test], ignore_index=True)
    full_oof.to_parquet(out_file, index=False)
    print(f"  Guardado exitoso: {out_file} (Filas: {len(full_oof)})")

if __name__ == '__main__':
    for h in HORIZONS:
        generate_oof_for_h(h)
    print("\nGeneracion de OOF para todos los horizontes finalizada exitosamente.")
