"""
Prueba rápida de construcción y alineación de features para h=4.
"""
import os
import sys
import time
import pandas as pd
import numpy as np

ROOT = 'D:/Codingggg/Notas/Multivariados/MA2003B_Eq1'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import load_panel
from planA_quickwins.features_planA import build_planA_features

from experiments.noche2.imput2.features_pipeline import (
    compute_missing_features,
    compute_yesterday_stats,
    compute_regional_trends,
    compute_mexico_holidays,
    compute_synoptic_clusters,
    compute_chem_meteo_interactions
)

print("Iniciando prueba de integración de features...")
t0 = time.time()

# 1. Cargar panel crudo y panel imputado causal
raw_panel = load_panel()
imputed_panel = pd.read_parquet(os.path.join(ROOT, 'experiments', 'noche', 'audit_imput', 'panel_spatial_lgbm.parquet'))

print(f"Paneles cargados ({time.time()-t0:.2f}s).")

# 2. Features Plan A V4 sobre panel imputado
t1 = time.time()
df_planA, feat_cols_planA = build_planA_features(imputed_panel, variant='V4')
print(f"Plan A features construidas sobre panel imputado: {len(feat_cols_planA)} cols ({time.time()-t1:.2f}s).")

# 3. Missing features del panel crudo
missing_df = compute_missing_features(raw_panel)

# 4. Estadísticos del día anterior
yesterday_df = compute_yesterday_stats(imputed_panel)

# 5. Tendencias regionales
regional_df = compute_regional_trends(imputed_panel)

# 6. Días festivos México
holidays_df = compute_mexico_holidays(df_planA['Date'].drop_duplicates())

# 7. Cargar features meteo del cache v3_base para h=4
meteo_h4 = pd.read_parquet(os.path.join(ROOT, 'experiments', 'noche', 'v3_base', 'cache', 'features_h4.parquet'))
meteo_cols = [c for c in meteo_h4.columns if c.startswith(('meteo_t0_', 'honest_meteo_', 'pp_meteo_'))]
meteo_sub = meteo_h4[['Estacion', 'Date', 'split', 'Target'] + meteo_cols].copy()

# Merge todo
merged = pd.merge(df_planA, missing_df, on=['Estacion', 'Date'], how='left')
merged = pd.merge(merged, yesterday_df, on=['Estacion', 'Date'], how='left')
merged = pd.merge(merged, regional_df, on=['Estacion', 'Date'], how='left')
merged = pd.merge(merged, holidays_df, on='Date', how='left')
merged = pd.merge(merged, meteo_sub, on=['Estacion', 'Date'], how='left')

# Clusters sinópticos
tr_mask = (merged['split'] == 'train')
synoptic_df = compute_synoptic_clusters(merged[tr_mask], merged, h=4, n_clusters=4)
merged = pd.merge(merged, synoptic_df, on=['Estacion', 'Date'], how='left')

# Interacciones química-meteo
interactions_df = compute_chem_meteo_interactions(merged)
merged = pd.merge(merged, interactions_df, on=['Estacion', 'Date'], how='left')

print(f"Integración exitosa! Shape final: {merged.shape}, Tiempo total: {time.time()-t0:.2f}s")
print("Muestra de columnas nuevas:")
new_cols = [c for c in merged.columns if not c.startswith(('meteo_t0_', 'honest_meteo_', 'pp_meteo_')) and c not in feat_cols_planA and c not in ['Date', 'Estacion', 'split', 'Target']]
print(new_cols)
