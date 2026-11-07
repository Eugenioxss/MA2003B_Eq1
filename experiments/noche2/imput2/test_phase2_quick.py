"""
Test rapido de 1 fold para verificar que la integracion de Fase 2 no tenga errores de columnas o tipos.
"""
import os, sys, time, gc
ROOT = 'D:/Codingggg/Notas/Multivariados/MA2003B_Eq1'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

from experiments.noche2.imput2.train_phase2_refit import (
    build_horizon_dataset, run_cross_fitting_and_refit,
    build_planA_features, compute_missing_features,
    compute_yesterday_stats, compute_regional_trends,
    load_panel, IMPUTED_PANEL_PATH, LGB_PARAMS
)
import pandas as pd

print("Cargando base estatica para test...")
raw_panel = load_panel()
imputed_panel = pd.read_parquet(IMPUTED_PANEL_PATH)

df_planA_imp, chem_imp_cols_raw = build_planA_features(imputed_panel, variant='V4')
target_cols_to_drop = [c for c in df_planA_imp.columns if c.startswith('Target')]
df_planA_imp.drop(columns=target_cols_to_drop, inplace=True)
chem_imp_cols = [c for c in chem_imp_cols_raw if c != 'Estacion' and not c.startswith('Target')]

yesterday_df = compute_yesterday_stats(imputed_panel)
regional_df = compute_regional_trends(imputed_panel)
missing_df = compute_missing_features(raw_panel)

base_static = pd.merge(df_planA_imp, yesterday_df, on=['Estacion', 'Date'], how='left')
base_static = pd.merge(base_static, regional_df, on=['Estacion', 'Date'], how='left')

yesterday_cols = [c for c in yesterday_df.columns if c not in ['Date', 'Estacion']]
regional_cols = [c for c in regional_df.columns if c not in ['Date', 'Estacion']]
missing_cols = [c for c in missing_df.columns if c not in ['Date', 'Estacion']]

print("Construyendo dataset h=4...")
df_h, feature_cols, honest_cols, pp_cols = build_horizon_dataset(
    h=4,
    base_static=base_static,
    chem_imp_cols=chem_imp_cols,
    yesterday_cols=yesterday_cols,
    regional_cols=regional_cols,
    missing_cols=missing_cols
)

print(f"Features: {len(feature_cols)}, Columnas df: {len(df_h.columns)}")
print("Ejecutando 1 prueba rapida de cross_fitting...")

# Reducir boost_round temporalmente para prueba rápida
test_params = dict(LGB_PARAMS)
test_params['verbose'] = -1

val_m, test_m, preds_df = run_cross_fitting_and_refit(
    h=4,
    df=df_h,
    features=feature_cols,
    honest_cols=honest_cols,
    pp_cols=pp_cols,
    variant_type='direct',
    seeds=[42],
    iter_mult=1.0
)

print("Test exitoso!")
print("Val OOF RMSE:", val_m['RMSE'], "Test RMSE:", test_m['RMSE'])
print("Preds df shape:", preds_df.shape)
print("Preds head:\n", preds_df.head())
