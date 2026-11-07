"""
experiments/noche2/imput2/run_stage2_feature_selection.py
=========================================================
Etapa 2: Selección hacia adelante por grupos de features sobre la base ganadora 1A_imp_pure.
Evalúa en validación 2024 para h in [4, 8, 24]:
  - Base: 1A_imp_pure (Plan A V4 sobre panel_spatial_lgbm + meteo honesta)
  - Paso 1: + Interacciones química-meteo (chem_meteo_interactions)
  - Paso 2: + Regímenes sinópticos NWP (synoptic_regimes)
  - Paso 3: + Estadísticos del día anterior (yesterday_stats)
  - Paso 4: + Tendencias regionales de cuenca (regional_trends)
  - Paso 5: + Días festivos de México y calendario solar (mexico_holidays)

Registra ganancia marginal en val RMSE.
Guarda resultados en stage2_feature_selection.csv.
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

ROOT = 'D:/Codingggg/Notas/Multivariados/MA2003B_Eq1'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import load_panel, evaluate_test, metrics, MODEL_STATIONS, SEED
from planA_quickwins.features_planA import build_planA_features
from experiments.noche.v3_base.train_v3 import inject_meteo_noise
from experiments.noche2.imput2.features_pipeline import (
    compute_yesterday_stats,
    compute_regional_trends,
    compute_mexico_holidays,
    compute_synoptic_clusters,
    compute_chem_meteo_interactions
)

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

def main():
    print("=" * 75)
    print("  ETAPA 2: SELECCIÓN HACIA ADELANTE DE GRUPOS DE FEATURES")
    print("=" * 75)
    
    print("Cargando paneles y precomputando features fijas...")
    raw_panel = load_panel()
    imputed_panel = pd.read_parquet(IMPUTED_PANEL_PATH)
    
    df_planA_imp, chem_imp_cols_raw = build_planA_features(imputed_panel, variant='V4')
    # Quitar targets pre-insertados
    target_cols_to_drop = [c for c in df_planA_imp.columns if c.startswith('Target')]
    df_planA_imp.drop(columns=target_cols_to_drop, inplace=True)
    chem_imp_cols = [c for c in chem_imp_cols_raw if c != 'Estacion' and not c.startswith('Target')]
    
    yesterday_df = compute_yesterday_stats(imputed_panel)
    regional_df = compute_regional_trends(imputed_panel)
    holidays_df = compute_mexico_holidays(df_planA_imp['Date'].drop_duplicates())
    
    # Pre-mergear features estáticas independientes de h
    base_static = pd.merge(df_planA_imp, yesterday_df, on=['Estacion', 'Date'], how='left')
    base_static = pd.merge(base_static, regional_df, on=['Estacion', 'Date'], how='left')
    base_static = pd.merge(base_static, holidays_df, on='Date', how='left')
    
    yesterday_cols = [c for c in yesterday_df.columns if c not in ['Date', 'Estacion']]
    regional_cols = [c for c in regional_df.columns if c not in ['Date', 'Estacion']]
    holiday_cols = [c for c in holidays_df.columns if c not in ['Date', 'Estacion']]
    
    EVAL_HORIZONS = [4, 8, 24]
    
    feature_groups = [
        ('Group1_chem_meteo', 'Interacciones quimica-meteo proyectadas'),
        ('Group2_synoptic', 'Clusters de regimen sinoptico NWP'),
        ('Group3_yesterday', 'Estadisticos del dia anterior (D-1)'),
        ('Group4_regional', 'Tendencias regionales y contraste de cuenca'),
        ('Group5_holidays', 'Dias festivos Mexico y calendario solar'),
    ]
    
    selection_log = []
    best_features_by_h = {}
    
    for h in EVAL_HORIZONS:
        print(f"\n{'='*75}\n  SELECCIÓN HACIA ADELANTE: HORIZONTE h={h}\n{'='*75}")
        df_v3 = pd.read_parquet(os.path.join(CACHE_V3, f'features_h{h}.parquet'))
        t0_cols = [c for c in df_v3.columns if c.startswith('meteo_t0_')]
        honest_cols = [c for c in df_v3.columns if c.startswith('honest_meteo_')]
        pp_cols = [c for c in df_v3.columns if c.startswith('pp_meteo_')]
        target_col = f'Target_{h}h' if f'Target_{h}h' in df_v3.columns else 'Target'
        
        if f'Target_{h}h' not in df_v3.columns:
            df_v3[f'Target_{h}h'] = df_v3['Target']
        if 'Target' not in df_v3.columns:
            df_v3['Target'] = df_v3[f'Target_{h}h']
            
        cols_to_pull = ['Estacion', 'Date', 'split', target_col] + t0_cols + honest_cols + pp_cols
        cols_to_pull = list(dict.fromkeys(cols_to_pull))
        meteo_and_targets = df_v3[cols_to_pull]
        
        merged_h = pd.merge(base_static, meteo_and_targets, on=['Estacion', 'Date'], how='left')
        merged_h['Estacion'] = merged_h['Estacion'].astype('category')
        
        # Calcular sinóptico e interacciones específicas de este h
        tr_mask = (merged_h['split'] == 'train')
        synoptic_df = compute_synoptic_clusters(merged_h[tr_mask], merged_h, h=h, n_clusters=4)
        merged_h = pd.merge(merged_h, synoptic_df, on=['Estacion', 'Date'], how='left')
        
        interactions_df = compute_chem_meteo_interactions(merged_h)
        merged_h = pd.merge(merged_h, interactions_df, on=['Estacion', 'Date'], how='left')
        
        synoptic_cols = [c for c in synoptic_df.columns if c not in ['Date', 'Estacion']]
        chem_meteo_cols = [c for c in interactions_df.columns if c not in ['Date', 'Estacion']]
        
        # Splits
        tr_df = merged_h[(merged_h['split'] == 'train') & merged_h[target_col].notna()].copy().reset_index(drop=True)
        va_df = merged_h[(merged_h['split'] == 'val') & merged_h[target_col].notna()].copy().reset_index(drop=True)
        te_df = merged_h[(merged_h['split'] == 'test') & merged_h[target_col].notna()].copy().reset_index(drop=True)
        
        # Inyección de ruido en meteo train
        tr_df = inject_meteo_noise(tr_df, va_df, honest_cols, pp_cols, t0_cols, seed=SEED)
        
        # Base: 1A_imp_pure (el ganador absoluto de la Etapa 1)
        active_features = ['Estacion'] + chem_imp_cols + t0_cols + honest_cols
        active_features = list(dict.fromkeys(active_features))
        
        def train_and_eval(feats: list[str]):
            feats_clean = list(dict.fromkeys(feats))
            dtr = lgb.Dataset(tr_df[feats_clean], tr_df[target_col].values, categorical_feature=['Estacion'], free_raw_data=False)
            dva = lgb.Dataset(va_df[feats_clean], va_df[target_col].values, reference=dtr, categorical_feature=['Estacion'], free_raw_data=False)
            cb = [lgb.early_stopping(80, verbose=False)]
            bst = lgb.train(LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=cb)
            
            p_va = bst.predict(va_df[feats_clean], num_iteration=bst.best_iteration)
            p_te = bst.predict(te_df[feats_clean], num_iteration=bst.best_iteration)
            m_va = metrics(va_df[target_col].values, p_va)
            m_te = evaluate_test(te_df, p_te, h)
            return m_va, m_te, bst.best_iteration
            
        m_va_base, m_te_base, iter_base = train_and_eval(active_features)
        print(f"  [h={h}] BASE (1A_imp_pure)        | Val RMSE: {m_va_base['RMSE']:.4f} | Test RMSE: {m_te_base['RMSE']:.4f} | Pk RMSE: {m_te_base['RMSE_peaks']:.2f} (iter={iter_base})")
        
        current_val_rmse = m_va_base['RMSE']
        selection_log.append({
            'h': h, 'step': 'Base_1A', 'features_added': 'None',
            'n_features': len(active_features),
            'val_RMSE': m_va_base['RMSE'], 'test_RMSE': m_te_base['RMSE'],
            'test_RMSE_peaks': m_te_base['RMSE_peaks'], 'decision': 'BASE'
        })
        
        group_col_map = {
            'Group1_chem_meteo': chem_meteo_cols,
            'Group2_synoptic': synoptic_cols,
            'Group3_yesterday': yesterday_cols,
            'Group4_regional': regional_cols,
            'Group5_holidays': holiday_cols,
        }
        
        for g_id, g_desc in feature_groups:
            cols_to_add = [c for c in group_col_map[g_id] if c not in active_features]
            candidate_features = active_features + cols_to_add
            m_va_cand, m_te_cand, iter_cand = train_and_eval(candidate_features)
            
            diff_val = m_va_cand['RMSE'] - current_val_rmse
            accepted = diff_val < -0.0005  # criterio de aceptación: mejora en val 2024
            decision = 'ACCEPTED' if accepted else 'REJECTED'
            
            print(f"  [h={h}] + {g_id:18s} ({g_desc}) -> Val RMSE: {m_va_cand['RMSE']:.4f} (delta={diff_val:+.4f}) | Test RMSE: {m_te_cand['RMSE']:.4f} [{decision}]")
            
            selection_log.append({
                'h': h, 'step': g_id, 'features_added': g_desc,
                'n_features': len(candidate_features),
                'val_RMSE': m_va_cand['RMSE'], 'test_RMSE': m_te_cand['RMSE'],
                'test_RMSE_peaks': m_te_cand['RMSE_peaks'], 'decision': decision
            })
            
            if accepted:
                active_features = candidate_features
                current_val_rmse = m_va_cand['RMSE']
                
        print(f"  [h={h}] Selección finalizada. Features seleccionadas: {len(active_features)}, Val RMSE final: {current_val_rmse:.4f}")
        best_features_by_h[h] = active_features

    df_log = pd.DataFrame(selection_log)
    df_log.to_csv(os.path.join(EXP_DIR, 'stage2_feature_selection.csv'), index=False)
    
    with open(os.path.join(EXP_DIR, 'best_features_by_h.json'), 'w') as f:
        json.dump(best_features_by_h, f, indent=2)
        
    print("\n" + "=" * 75)
    print("  RESUMEN ETAPA 2 (Guardado en stage2_feature_selection.csv)")
    print("=" * 75)
    print(df_log.to_string(index=False))

if __name__ == '__main__':
    main()
