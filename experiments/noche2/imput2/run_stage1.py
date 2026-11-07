"""
experiments/noche2/imput2/run_stage1.py
=======================================
Etapa 1: Integración de Imputación Causal en Pipeline v3 Honesto.
Compara en h in [4, 8, 24, 1]:
  - D_honest_noise_ref: Referencia oficial v3 (panel crudo con NaNs)
  - 1A_imp_pure: Features Plan A V4 construidos sobre panel_spatial_lgbm
  - 1B_imp_plus_flags: 1A + missingness flags + age of missingness
  - 1C_imp_plus_raw: Features originales con NaNs + features clave *_imp como columnas adicionales
"""
from __future__ import annotations
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import partial
print = partial(print, flush=True)

ROOT = 'D:/Codingggg/Notas/Multivariados\MA2003B_Eq1'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import load_panel, evaluate_test, metrics, MODEL_STATIONS, SEED
from planA_quickwins.features_planA import build_planA_features
from experiments.noche.v3_base.train_v3 import inject_meteo_noise
from experiments.noche2.imput2.features_pipeline import compute_missing_features

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_V3 = os.path.join(ROOT, 'experiments', 'noche', 'v3_base', 'cache')
IMPUTED_PANEL_PATH = os.path.join(ROOT, 'experiments', 'noche', 'audit_imput', 'panel_spatial_lgbm.parquet')

OPTUNA_PATH = os.path.join(ROOT, 'experiments', 'planA_quickwins', 'best_params_optuna.json')
with open(OPTUNA_PATH, 'r') as f:
    BEST_OPTUNA = json.load(f)

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'seed': SEED,
    'verbose': -1,
    'num_threads': 4,
    'feature_pre_filter': False,
    'importance_type': 'gain',
    **BEST_OPTUNA
}

# Referencias oficiales de D_honest_noise de metrics_summary.csv
REF_METRICS = {
    1: {'val_RMSE': 4.93699, 'test_RMSE': 5.00475, 'test_R2': 0.94348, 'test_RMSE_peaks': 10.31989, 'test_bias_peaks': -5.13936, 'best_iter': 1491},
    4: {'val_RMSE': 8.63560, 'test_RMSE': 9.00390, 'test_R2': 0.81713, 'test_RMSE_peaks': 21.69600, 'test_bias_peaks': -16.90620, 'best_iter': 1992},
    8: {'val_RMSE': 9.62287, 'test_RMSE': 10.05359, 'test_R2': 0.77208, 'test_RMSE_peaks': 25.09881, 'test_bias_peaks': -20.39603, 'best_iter': 1669},
    12: {'val_RMSE': 9.83580, 'test_RMSE': 10.18324, 'test_R2': 0.76624, 'test_RMSE_peaks': 25.18699, 'test_bias_peaks': -20.34608, 'best_iter': 974},
    24: {'val_RMSE': 10.02623, 'test_RMSE': 10.23822, 'test_R2': 0.76391, 'test_RMSE_peaks': 24.17781, 'test_bias_peaks': -19.08762, 'best_iter': 990},
    48: {'val_RMSE': 11.11582, 'test_RMSE': 11.23373, 'test_R2': 0.71619, 'test_RMSE_peaks': 27.03067, 'test_bias_peaks': -22.51129, 'best_iter': 504},
}

print("=" * 75)
print("  ETAPA 1: INTEGRACIÓN DE IMPUTACIÓN CAUSAL EN PIPELINE v3 HONESTO")
print("=" * 75)

# 1. Cargar paneles
print("Cargando paneles...")
raw_panel = load_panel()
imputed_panel = pd.read_parquet(IMPUTED_PANEL_PATH)

print("Construyendo features Plan A sobre panel imputado...")
df_planA_imp, chem_imp_cols_raw = build_planA_features(imputed_panel, variant='V4')
# Eliminar columnas de Target que build_planA_features agrega automáticamente
target_cols_to_drop = [c for c in df_planA_imp.columns if c.startswith('Target')]
df_planA_imp.drop(columns=target_cols_to_drop, inplace=True)

# Excluir 'Estacion' para que nunca quede duplicada
chem_imp_cols = [c for c in chem_imp_cols_raw if c != 'Estacion' and not c.startswith('Target')]

print("Calculando missingness flags...")
missing_df = compute_missing_features(raw_panel)
missing_cols = [c for c in missing_df.columns if c not in ['Date', 'Estacion']]

# Mergear df_planA_imp con missing flags
df_planA_imp = pd.merge(df_planA_imp, missing_df, on=['Estacion', 'Date'], how='left')

EVAL_HORIZONS = [4, 8, 24, 1]
results = []

for h in EVAL_HORIZONS:
    print(f"\n{'='*70}\n  EVALUANDO HORIZONTE h={h}\n{'='*70}")
    # 0. Agregar referencia oficial
    ref = REF_METRICS[h]
    print(f"[h={h}] REF D_honest_noise   | Val RMSE: {ref['val_RMSE']:6.3f} | Test RMSE: {ref['test_RMSE']:6.3f} | Pk RMSE: {ref['test_RMSE_peaks']:6.2f} (iter={ref['best_iter']})")
    results.append({'h': h, 'variant': 'D_honest_noise_ref', **ref})
    
    # Cargar cache v3
    df_v3 = pd.read_parquet(os.path.join(CACHE_V3, f'features_h{h}.parquet'))
    df_v3['Estacion'] = df_v3['Estacion'].astype('category')
    
    t0_cols = [c for c in df_v3.columns if c.startswith('meteo_t0_')]
    honest_cols = [c for c in df_v3.columns if c.startswith('honest_meteo_')]
    pp_cols = [c for c in df_v3.columns if c.startswith('pp_meteo_')]
    target_col = f'Target_{h}h' if f'Target_{h}h' in df_v3.columns else 'Target'
    exclude = {'Date', 'Estacion', 'split', 'Target', f'Target_{h}h'} | set(t0_cols) | set(honest_cols) | set(pp_cols)
    chem_raw_cols = [c for c in df_v3.columns if c not in exclude and c != 'Estacion']
    
    if f'Target_{h}h' not in df_v3.columns:
        df_v3[f'Target_{h}h'] = df_v3['Target']
    if 'Target' not in df_v3.columns:
        df_v3['Target'] = df_v3[f'Target_{h}h']
        
    # Unir features imputadas con meteo y targets observados
    cols_to_pull = ['Estacion', 'Date', 'split', target_col] + t0_cols + honest_cols + pp_cols
    cols_to_pull = list(dict.fromkeys(cols_to_pull))
    meteo_and_targets = df_v3[cols_to_pull]
    
    df_merged = pd.merge(df_planA_imp, meteo_and_targets, on=['Estacion', 'Date'], how='left')
    df_merged['Estacion'] = df_merged['Estacion'].astype('category')
    
    tr_m = df_merged[(df_merged['split'] == 'train') & df_merged[target_col].notna()].copy().reset_index(drop=True)
    va_m = df_merged[(df_merged['split'] == 'val') & df_merged[target_col].notna()].copy().reset_index(drop=True)
    te_m = df_merged[(df_merged['split'] == 'test') & df_merged[target_col].notna()].copy().reset_index(drop=True)
    
    # Inyección de ruido en meteo train
    tr_m = inject_meteo_noise(tr_m, va_m, honest_cols, pp_cols, t0_cols, seed=SEED)
    
    callbacks = [lgb.early_stopping(80, verbose=False)]
    
    # -------------------------------------------------------------
    # 1A: Pure Imputed Features (imputación antes de features)
    # -------------------------------------------------------------
    feat_1a = ['Estacion'] + chem_imp_cols + t0_cols + honest_cols
    feat_1a = list(dict.fromkeys(feat_1a))
    
    dtr_1a = lgb.Dataset(tr_m[feat_1a], tr_m[target_col].values, categorical_feature=['Estacion'], free_raw_data=False)
    dva_1a = lgb.Dataset(va_m[feat_1a], va_m[target_col].values, reference=dtr_1a, categorical_feature=['Estacion'], free_raw_data=False)
    
    bst_1a = lgb.train(LGB_PARAMS, dtr_1a, num_boost_round=2000, valid_sets=[dva_1a], callbacks=callbacks)
    pred_va_1a = bst_1a.predict(va_m[feat_1a], num_iteration=bst_1a.best_iteration)
    pred_te_1a = bst_1a.predict(te_m[feat_1a], num_iteration=bst_1a.best_iteration)
    
    m_va_1a = metrics(va_m[target_col].values, pred_va_1a)
    m_te_1a = evaluate_test(te_m, pred_te_1a, h)
    print(f"[h={h}] 1A_imp_pure          | Val RMSE: {m_va_1a['RMSE']:6.3f} | Test RMSE: {m_te_1a['RMSE']:6.3f} | Pk RMSE: {m_te_1a['RMSE_peaks']:6.2f} (iter={bst_1a.best_iteration})")
    
    results.append({
        'h': h, 'variant': '1A_imp_pure',
        'val_RMSE': m_va_1a['RMSE'], 'test_RMSE': m_te_1a['RMSE'],
        'test_R2': m_te_1a['R2'], 'test_RMSE_peaks': m_te_1a['RMSE_peaks'],
        'test_bias_peaks': m_te_1a['bias_peaks'], 'best_iter': bst_1a.best_iteration
    })
    
    # -------------------------------------------------------------
    # 1B: Imputed + Missingness Flags & Age
    # -------------------------------------------------------------
    feat_1b = ['Estacion'] + chem_imp_cols + missing_cols + t0_cols + honest_cols
    feat_1b = list(dict.fromkeys(feat_1b))
    
    dtr_1b = lgb.Dataset(tr_m[feat_1b], tr_m[target_col].values, categorical_feature=['Estacion'], free_raw_data=False)
    dva_1b = lgb.Dataset(va_m[feat_1b], va_m[target_col].values, reference=dtr_1b, categorical_feature=['Estacion'], free_raw_data=False)
    
    bst_1b = lgb.train(LGB_PARAMS, dtr_1b, num_boost_round=2000, valid_sets=[dva_1b], callbacks=callbacks)
    pred_va_1b = bst_1b.predict(va_m[feat_1b], num_iteration=bst_1b.best_iteration)
    pred_te_1b = bst_1b.predict(te_m[feat_1b], num_iteration=bst_1b.best_iteration)
    
    m_va_1b = metrics(va_m[target_col].values, pred_va_1b)
    m_te_1b = evaluate_test(te_m, pred_te_1b, h)
    print(f"[h={h}] 1B_imp_plus_flags    | Val RMSE: {m_va_1b['RMSE']:6.3f} | Test RMSE: {m_te_1b['RMSE']:6.3f} | Pk RMSE: {m_te_1b['RMSE_peaks']:6.2f} (iter={bst_1b.best_iteration})")
    
    results.append({
        'h': h, 'variant': '1B_imp_plus_flags',
        'val_RMSE': m_va_1b['RMSE'], 'test_RMSE': m_te_1b['RMSE'],
        'test_R2': m_te_1b['R2'], 'test_RMSE_peaks': m_te_1b['RMSE_peaks'],
        'test_bias_peaks': m_te_1b['bias_peaks'], 'best_iter': bst_1b.best_iteration
    })
    
    # -------------------------------------------------------------
    # 1C: Raw with NaNs + Key Imputed Features as extra columns
    # -------------------------------------------------------------
    key_imp_cols = [c for c in chem_imp_cols if c.startswith(('O3_lag_', 'O3_diff_', 'O3_others_', 'O3_nearest_'))]
    sub_imp = df_planA_imp[['Estacion', 'Date'] + key_imp_cols].copy()
    sub_imp.rename(columns={c: f'{c}_imp' for c in key_imp_cols}, inplace=True)
    
    df_1c = pd.merge(df_v3, sub_imp, on=['Estacion', 'Date'], how='left')
    df_1c['Estacion'] = df_1c['Estacion'].astype('category')
    
    tr_1c = df_1c[(df_1c['split'] == 'train') & df_1c[target_col].notna()].copy().reset_index(drop=True)
    va_1c = df_1c[(df_1c['split'] == 'val') & df_1c[target_col].notna()].copy().reset_index(drop=True)
    te_1c = df_1c[(df_1c['split'] == 'test') & df_1c[target_col].notna()].copy().reset_index(drop=True)
    
    tr_1c = inject_meteo_noise(tr_1c, va_1c, honest_cols, pp_cols, t0_cols, seed=SEED)
    
    extra_imp_cols = [f'{c}_imp' for c in key_imp_cols]
    feat_1c = ['Estacion'] + chem_raw_cols + extra_imp_cols + t0_cols + honest_cols
    feat_1c = list(dict.fromkeys(feat_1c))
    
    dtr_1c = lgb.Dataset(tr_1c[feat_1c], tr_1c[target_col].values, categorical_feature=['Estacion'], free_raw_data=False)
    dva_1c = lgb.Dataset(va_1c[feat_1c], va_1c[target_col].values, reference=dtr_1c, categorical_feature=['Estacion'], free_raw_data=False)
    
    bst_1c = lgb.train(LGB_PARAMS, dtr_1c, num_boost_round=2000, valid_sets=[dva_1c], callbacks=callbacks)
    pred_va_1c = bst_1c.predict(va_1c[feat_1c], num_iteration=bst_1c.best_iteration)
    pred_te_1c = bst_1c.predict(te_1c[feat_1c], num_iteration=bst_1c.best_iteration)
    
    m_va_1c = metrics(va_1c[target_col].values, pred_va_1c)
    m_te_1c = evaluate_test(te_1c, pred_te_1c, h)
    print(f"[h={h}] 1C_imp_plus_raw      | Val RMSE: {m_va_1c['RMSE']:6.3f} | Test RMSE: {m_te_1c['RMSE']:6.3f} | Pk RMSE: {m_te_1c['RMSE_peaks']:6.2f} (iter={bst_1c.best_iteration})")
    
    results.append({
        'h': h, 'variant': '1C_imp_plus_raw',
        'val_RMSE': m_va_1c['RMSE'], 'test_RMSE': m_te_1c['RMSE'],
        'test_R2': m_te_1c['R2'], 'test_RMSE_peaks': m_te_1c['RMSE_peaks'],
        'test_bias_peaks': m_te_1c['bias_peaks'], 'best_iter': bst_1c.best_iteration
    })

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(EXP_DIR, 'stage1_results.csv'), index=False)
print("\n" + "=" * 75)
print("  RESUMEN ETAPA 1 (Guardado en stage1_results.csv)")
print("=" * 75)
print(res_df.to_string(index=False))
