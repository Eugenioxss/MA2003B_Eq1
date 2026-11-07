"""
experiments/noche2/imput2/run_stage4_final.py
=============================================
Etapa 4: Entrenamiento final y generación de predicciones oficiales para todos los horizontes:
  h in [4, 8, 24, 12, 48, 1]
  - Aplica la mejor combinación de features seleccionada en Etapa 2
  - Evalúa ablación de sample weighting w = 1 + (y/p90)^2 (E_honest_w)
  - Guarda predicciones oficiales en:
      experiments/noche2/imput2/preds/best_imput2_h{h}.parquet
      con columnas: Date, Estacion, h, split ('val'|'test'), y_true, y_pred
  - Registra resultados oficiales en experiments/results_registry.csv con experiment='noche2_imput2'
  - Genera final_metrics_summary.csv
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
from common import load_panel, evaluate_test, metrics, log_result, MODEL_STATIONS, HORIZONS, SEED
from planA_quickwins.features_planA import build_planA_features
from experiments.noche.v3_base.train_v3 import inject_meteo_noise
from experiments.noche2.imput2.features_pipeline import (
    compute_missing_features,
    compute_yesterday_stats,
    compute_regional_trends,
    compute_mexico_holidays,
    compute_synoptic_clusters,
    compute_chem_meteo_interactions
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(EXP_DIR, 'preds')
os.makedirs(PREDS_DIR, exist_ok=True)

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

def run_final():
    print("=" * 75)
    print("  ETAPA 4: ENTRENAMIENTO FINAL Y PREDICCIONES OFICIALES NOCHE2_IMPUT2")
    print("=" * 75)
    
    # 1. Cargar selección de features de Stage 2
    feat_json_path = os.path.join(EXP_DIR, 'best_features_by_h.json')
    best_features_map = {}
    if os.path.exists(feat_json_path):
        with open(feat_json_path, 'r') as f:
            best_features_map = json.load(f)
            
    print("Cargando paneles y precomputando features estáticas...")
    raw_panel = load_panel()
    imputed_panel = pd.read_parquet(IMPUTED_PANEL_PATH)
    
    df_planA_imp, chem_imp_cols_raw = build_planA_features(imputed_panel, variant='V4')
    target_cols_to_drop = [c for c in df_planA_imp.columns if c.startswith('Target')]
    df_planA_imp.drop(columns=target_cols_to_drop, inplace=True)
    chem_imp_cols = [c for c in chem_imp_cols_raw if c != 'Estacion' and not c.startswith('Target')]
    
    missing_df = compute_missing_features(raw_panel)
    yesterday_df = compute_yesterday_stats(imputed_panel)
    regional_df = compute_regional_trends(imputed_panel)
    holidays_df = compute_mexico_holidays(df_planA_imp['Date'].drop_duplicates())
    
    # Pre-merge estático
    base_static = pd.merge(df_planA_imp, missing_df, on=['Estacion', 'Date'], how='left')
    base_static = pd.merge(base_static, yesterday_df, on=['Estacion', 'Date'], how='left')
    base_static = pd.merge(base_static, regional_df, on=['Estacion', 'Date'], how='left')
    base_static = pd.merge(base_static, holidays_df, on='Date', how='left')
    
    missing_cols = [c for c in missing_df.columns if c not in ['Date', 'Estacion']]
    yesterday_cols = [c for c in yesterday_df.columns if c not in ['Date', 'Estacion']]
    regional_cols = [c for c in regional_df.columns if c not in ['Date', 'Estacion']]
    holiday_cols = [c for c in holidays_df.columns if c not in ['Date', 'Estacion']]
    
    final_metrics_records = []
    
    # Orden de horizontes: prioritarios 4, 8, 24, luego 12, 48, 1
    ORDERED_HORIZONS = [4, 8, 24, 12, 48, 1]
    
    for h in ORDERED_HORIZONS:
        print(f"\n{'='*75}\n  GENERANDO MODELO FINAL: HORIZONTE h={h}\n{'='*75}")
        t0_h = time.time()
        
        df_v3 = pd.read_parquet(os.path.join(CACHE_V3, f'features_h{h}.parquet'))
        t0_cols = [c for c in df_v3.columns if c.startswith('meteo_t0_')]
        honest_cols = [c for c in df_v3.columns if c.startswith('honest_meteo_')]
        pp_cols = [c for c in df_v3.columns if c.startswith('pp_meteo_')]
        target_col = f'Target_{h}h' if f'Target_{h}h' in df_v3.columns else 'Target'
        
        if f'Target_{h}h' not in df_v3.columns:
            df_v3[f'Target_{h}h'] = df_v3['Target']
        if 'Target' not in df_v3.columns:
            df_v3['Target'] = df_v3[f'Target_{h}h']
            
        cols_to_pull = ['Estacion', 'Date', 'split', 'Target', f'Target_{h}h'] + t0_cols + honest_cols + pp_cols
        cols_to_pull = list(dict.fromkeys(cols_to_pull))
        meteo_and_targets = df_v3[cols_to_pull]
        
        merged_h = pd.merge(base_static, meteo_and_targets, on=['Estacion', 'Date'], how='left')
        merged_h['Estacion'] = merged_h['Estacion'].astype('category')
        
        # Splits
        tr_df = merged_h[(merged_h['split'] == 'train') & merged_h[target_col].notna()].copy().reset_index(drop=True)
        va_df = merged_h[(merged_h['split'] == 'val') & merged_h[target_col].notna()].copy().reset_index(drop=True)
        te_df = merged_h[(merged_h['split'] == 'test') & merged_h[target_col].notna()].copy().reset_index(drop=True)
        
        # Inyección de ruido meteo en train
        tr_df = inject_meteo_noise(tr_df, va_df, honest_cols, pp_cols, t0_cols, seed=SEED)
        
        # Obtener mejores features
        str_h = str(h)
        if str_h in best_features_map:
            features_to_use = best_features_map[str_h]
        elif h == 1:
            # Ganador en h=1: Imputación + missing flags + yesterday stats
            features_to_use = ['Estacion'] + chem_imp_cols + missing_cols + t0_cols + honest_cols + yesterday_cols
        else:
            # Para h=12 y h=48: usar la fórmula óptima de h=4 y h=8
            features_to_use = ['Estacion'] + chem_imp_cols + t0_cols + honest_cols + yesterday_cols + regional_cols
            
        features_to_use = list(dict.fromkeys(features_to_use))
        
        # Pesos de muestra w = 1 + (y/p90)^2
        y_tr = tr_df[target_col].values
        p90 = np.percentile(y_tr, 90.0)
        sample_weights = 1.0 + (y_tr / p90) ** 2
        
        # Datasets
        dtr_unweighted = lgb.Dataset(tr_df[features_to_use], y_tr, categorical_feature=['Estacion'], free_raw_data=False)
        dtr_weighted = lgb.Dataset(tr_df[features_to_use], y_tr, categorical_feature=['Estacion'], weight=sample_weights, free_raw_data=False)
        dva = lgb.Dataset(va_df[features_to_use], va_df[target_col].values, reference=dtr_unweighted, categorical_feature=['Estacion'], free_raw_data=False)
        
        cb = [lgb.early_stopping(80, verbose=False)]
        
        # Modelo A: sin pesos
        bst_unw = lgb.train(LGB_PARAMS, dtr_unweighted, num_boost_round=2000, valid_sets=[dva], callbacks=cb)
        p_va_unw = bst_unw.predict(va_df[features_to_use], num_iteration=bst_unw.best_iteration)
        m_va_unw = metrics(va_df[target_col].values, p_va_unw)
        
        # Modelo B: con pesos
        bst_w = lgb.train(LGB_PARAMS, dtr_weighted, num_boost_round=2000, valid_sets=[dva], callbacks=cb)
        p_va_w = bst_w.predict(va_df[features_to_use], num_iteration=bst_w.best_iteration)
        m_va_w = metrics(va_df[target_col].values, p_va_w)
        
        print(f"  [h={h}] Sin pesos: Val RMSE = {m_va_unw['RMSE']:.4f} (Pk RMSE: {m_va_unw['RMSE_peaks']:.2f}) | Con pesos: Val RMSE = {m_va_w['RMSE']:.4f} (Pk RMSE: {m_va_w['RMSE_peaks']:.2f})")
        
        # Selección: preferir con pesos si Val RMSE es menor o si mejora sustancialmente picos sin degradar RMSE en >0.03
        use_weights = (m_va_w['RMSE'] <= m_va_unw['RMSE']) or (m_va_w['RMSE_peaks'] < m_va_unw['RMSE_peaks'] - 0.5 and m_va_w['RMSE'] <= m_va_unw['RMSE'] + 0.02)
        best_bst = bst_w if use_weights else bst_unw
        best_va_m = m_va_w if use_weights else m_va_unw
        best_va_pred = p_va_w if use_weights else p_va_unw
        weight_str = 'w_peaks' if use_weights else 'unweighted'
        
        p_te = best_bst.predict(te_df[features_to_use], num_iteration=best_bst.best_iteration)
        best_te_m = evaluate_test(te_df, p_te, h)
        train_time = time.time() - t0_h
        
        print(f"  [h={h}] SELECCIONADO ({weight_str}): Val RMSE = {best_va_m['RMSE']:.4f} | Test RMSE = {best_te_m['RMSE']:.4f} | Pk RMSE = {best_te_m['RMSE_peaks']:.2f} | Pk Bias = {best_te_m['bias_peaks']:.2f} (iter={best_bst.best_iteration}, time={train_time:.1f}s)")
        
        # Guardar predicciones oficiales según contrato
        preds_val = pd.DataFrame({
            'Date': va_df['Date'],
            'Estacion': va_df['Estacion'],
            'h': h,
            'split': 'val',
            'y_true': va_df[target_col].astype('float32'),
            'y_pred': best_va_pred.astype('float32')
        })
        preds_test = pd.DataFrame({
            'Date': te_df['Date'],
            'Estacion': te_df['Estacion'],
            'h': h,
            'split': 'test',
            'y_true': te_df[target_col].astype('float32'),
            'y_pred': p_te.astype('float32')
        })
        preds_all = pd.concat([preds_val, preds_test], ignore_index=True)
        pred_path = os.path.join(PREDS_DIR, f'best_imput2_h{h}.parquet')
        preds_all.to_parquet(pred_path, index=False)
        print(f"  Predicciones oficiales guardadas en: {pred_path} ({len(preds_all)} filas)")
        
        # Registrar en results_registry.csv
        notes = f"noche2_imput2: Imputacion causal espacial + memoria ayer D-1 + tendencias regionales ({weight_str}, {len(features_to_use)} feats)"
        log_result(
            experiment='noche2_imput2',
            variant=f'best_imput2_{weight_str}',
            h=h,
            m=best_te_m,
            notes=notes
        )
        
        final_metrics_records.append({
            'h': h,
            'variant': f'best_imput2_{weight_str}',
            'n_features': len(features_to_use),
            'val_RMSE': best_va_m['RMSE'],
            'test_RMSE': best_te_m['RMSE'],
            'test_R2': best_te_m['R2'],
            'test_MAE': best_te_m['MAE'],
            'test_RMSE_peaks': best_te_m['RMSE_peaks'],
            'test_bias_peaks': best_te_m['bias_peaks'],
            'skill_vs_clim': best_te_m.get('skill_vs_clim', 0.0),
            'best_iter': best_bst.best_iteration,
            'train_time_s': train_time
        })

    df_final = pd.DataFrame(final_metrics_records)
    df_final.to_csv(os.path.join(EXP_DIR, 'final_metrics_summary.csv'), index=False)
    print("\n" + "=" * 75)
    print("  RESUMEN FINAL NOCHE2_IMPUT2 (Guardado en final_metrics_summary.csv)")
    print("=" * 75)
    print(df_final.to_string(index=False))

if __name__ == '__main__':
    run_final()
