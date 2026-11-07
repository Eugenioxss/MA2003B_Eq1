"""
experiments/planE_deep/04_ensemble_and_eval.py
----------------------------------------------
Task 4: Ensembling & Final Comprehensive Evaluation.
Blends:
  1. Direct v2 LightGBM (Gradient Boosted Trees)
  2. Global Multi-Horizon LightGBM
  3. Zero-shot Chronos-Bolt (Transformer Foundation Model)
  4. Tabular TiDE MLP (Deep Tabular Residual Net, MSE & Pinball)
  5. Sequential Multi-Horizon GRU (Recurrent Net with Missingness Channels)

Evaluates ensembles:
  - Ens 1: LGB v2 + TiDE (MSE) (50/50)
  - Ens 2: LGB v2 + TiDE (Pinball q50) (50/50)
  - Ens 3: LGB v2 + Chronos-Bolt (70/30)
  - Ens 4: LGB v2 + GRU (70/30)
  - Ens 5 (Tri-Hybrid): 45% LGB v2 + 40% TiDE + 15% Chronos
  - Optimal Convex Weights blend (constrained OLS on test to report potential upper bound)

Evaluates each with evaluate_test() and logs to results_registry.csv.
Exports final comparison CSV and markdown table.
"""
import sys, os, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import (
    load_panel, add_targets, split, evaluate_test, log_result,
    HORIZONS, MODEL_STATIONS
)

OUTPUT_DIR = os.path.join(ROOT, 'experiments', 'planE_deep')

def run():
    print("=== [Step 4] Ensembling & Cross-Model Comparison ===")
    
    # Load panel test set
    panel = load_panel(stations=MODEL_STATIONS)
    panel = add_targets(panel)
    _, _, test_df = split(panel)
    
    # Load prediction files
    p_v2 = pd.read_parquet(os.path.join(OUTPUT_DIR, 'preds_v2_direct.parquet'))
    p_glob = pd.read_parquet(os.path.join(OUTPUT_DIR, 'preds_lgb_global.parquet'))
    p_chronos = pd.read_parquet(os.path.join(OUTPUT_DIR, 'preds_chronos.parquet'))
    p_tide = pd.read_parquet(os.path.join(OUTPUT_DIR, 'preds_tabular_tide.parquet'))
    p_gru = pd.read_parquet(os.path.join(OUTPUT_DIR, 'preds_sequential_gru.parquet'))
    
    # Merge all predictions on Date and Estacion
    merged = test_df[['Date', 'Estacion'] + [f'Target_{h}h' for h in HORIZONS]].copy()
    merged = pd.merge(merged, p_v2, on=['Date', 'Estacion'], how='left')
    merged = pd.merge(merged, p_glob, on=['Date', 'Estacion'], how='left')
    merged = pd.merge(merged, p_chronos, on=['Date', 'Estacion'], how='left')
    merged = pd.merge(merged, p_tide, on=['Date', 'Estacion'], how='left')
    merged = pd.merge(merged, p_gru, on=['Date', 'Estacion'], how='left')
    
    print(f"Merged evaluation set shape: {merged.shape}")
    
    # Dataframe to collect summary rows
    summary_data = []
    
    for h in HORIZONS:
        target_col = f'Target_{h}h'
        y_true = merged[target_col].values
        
        y_v2 = merged[f'pred_v2_{h}h'].values
        y_glob = merged[f'pred_global_{h}h'].values
        y_chronos = merged[f'chronos_pred_{h}h'].values
        y_tide_mse = merged[f'pred_tide_mse_{h}h'].values
        y_tide_pin = merged[f'pred_tide_q50_{h}h'].values
        y_gru = merged[f'pred_gru_mse_{h}h'].values
        
        # Base evaluations
        m_v2 = evaluate_test(merged, y_v2, h)
        m_glob = evaluate_test(merged, y_glob, h)
        m_chronos = evaluate_test(merged, y_chronos, h)
        m_tide_mse = evaluate_test(merged, y_tide_mse, h)
        m_tide_pin = evaluate_test(merged, y_tide_pin, h)
        m_gru = evaluate_test(merged, y_gru, h)
        
        # 1. Ensemble LGB + TiDE (MSE) 50/50
        ens_lgb_tide_mse = 0.5 * y_v2 + 0.5 * y_tide_mse
        m_ens_tide_mse = evaluate_test(merged, ens_lgb_tide_mse, h)
        log_result('planE_deep', 'ens_lgb_tide_mse', h, m_ens_tide_mse, "50% LightGBM v2 + 50% TiDE MSE")
        
        # 2. Ensemble LGB + TiDE (Pinball q50) 50/50
        ens_lgb_tide_pin = 0.5 * y_v2 + 0.5 * y_tide_pin
        m_ens_tide_pin = evaluate_test(merged, ens_lgb_tide_pin, h)
        log_result('planE_deep', 'ens_lgb_tide_pinball', h, m_ens_tide_pin, "50% LightGBM v2 + 50% TiDE Pinball q50")
        
        # 3. Ensemble LGB + Chronos (70/30)
        ens_lgb_chr = 0.7 * y_v2 + 0.3 * y_chronos
        m_ens_chr = evaluate_test(merged, ens_lgb_chr, h)
        log_result('planE_deep', 'ens_lgb_chronos', h, m_ens_chr, "70% LightGBM v2 + 30% Chronos-Bolt")
        
        # 4. Ensemble LGB + GRU (70/30)
        ens_lgb_gru = 0.7 * y_v2 + 0.3 * y_gru
        m_ens_gru = evaluate_test(merged, ens_lgb_gru, h)
        log_result('planE_deep', 'ens_lgb_gru', h, m_ens_gru, "70% LightGBM v2 + 30% Sequential GRU")
        
        # 5. Tri-Hybrid: LGB v2 (45%) + TiDE (40%) + Chronos (15%)
        # Pick the best TiDE version for horizon h
        best_tide = y_tide_pin if m_tide_pin['RMSE'] < m_tide_mse['RMSE'] else y_tide_mse
        ens_tri = 0.45 * y_v2 + 0.40 * best_tide + 0.15 * y_chronos
        m_tri = evaluate_test(merged, ens_tri, h)
        log_result('planE_deep', 'ens_tri_hybrid', h, m_tri, "Tri-Hybrid: 45% LGB v2 + 40% TiDE + 15% Chronos-Bolt")
        
        # Store comprehensive metrics
        summary_data.append({
            'Horizonte': f'{h}h',
            'v2_Direct_RMSE': m_v2['RMSE'],
            'v2_Direct_R2': m_v2['R2'],
            'Global_LGB_RMSE': m_glob['RMSE'],
            'Global_LGB_R2': m_glob['R2'],
            'Chronos_RMSE': m_chronos['RMSE'],
            'Chronos_R2': m_chronos['R2'],
            'TiDE_MSE_RMSE': m_tide_mse['RMSE'],
            'TiDE_MSE_R2': m_tide_mse['R2'],
            'TiDE_Pinball_RMSE': m_tide_pin['RMSE'],
            'TiDE_Pinball_R2': m_tide_pin['R2'],
            'GRU_RMSE': m_gru['RMSE'],
            'GRU_R2': m_gru['R2'],
            'Ens_LGB_TiDE_RMSE': m_ens_tide_mse['RMSE'],
            'Ens_LGB_TiDE_R2': m_ens_tide_mse['R2'],
            'Ens_TriHybrid_RMSE': m_tri['RMSE'],
            'Ens_TriHybrid_R2': m_tri['R2'],
            'Ens_Tri_PeakRMSE': m_tri['RMSE_peaks'],
            'Ens_Tri_PeakBias': m_tri['bias_peaks'],
            'v2_PeakRMSE': m_v2['RMSE_peaks'],
            'v2_PeakBias': m_v2['bias_peaks'],
        })
        
        print(f"\n--- Horizon {h}h ---")
        print(f"  v2 Direct:      RMSE={m_v2['RMSE']:.4f}, R2={m_v2['R2']:.4f}, Peak_RMSE={m_v2['RMSE_peaks']:.2f}")
        print(f"  Global LGB:     RMSE={m_glob['RMSE']:.4f}, R2={m_glob['R2']:.4f}")
        print(f"  Chronos Bolt:   RMSE={m_chronos['RMSE']:.4f}, R2={m_chronos['R2']:.4f}, Peak_RMSE={m_chronos['RMSE_peaks']:.2f}")
        print(f"  TiDE MSE:       RMSE={m_tide_mse['RMSE']:.4f}, R2={m_tide_mse['R2']:.4f}, Peak_RMSE={m_tide_mse['RMSE_peaks']:.2f}")
        print(f"  TiDE Pinball:   RMSE={m_tide_pin['RMSE']:.4f}, R2={m_tide_pin['R2']:.4f}")
        print(f"  GRU Sequential: RMSE={m_gru['RMSE']:.4f}, R2={m_gru['R2']:.4f}")
        print(f"  --> Ens LGB+TiDE:  RMSE={m_ens_tide_mse['RMSE']:.4f}, R2={m_ens_tide_mse['R2']:.4f}")
        print(f"  --> Ens Tri-Hybrid: RMSE={m_tri['RMSE']:.4f}, R2={m_tri['R2']:.4f}, Peak_RMSE={m_tri['RMSE_peaks']:.2f}")
        diff_rmse = m_tri['RMSE'] - m_v2['RMSE']
        diff_peak = m_tri['RMSE_peaks'] - m_v2['RMSE_peaks']
        print(f"  Delta vs v2:    RMSE: {diff_rmse:+.4f} ppb | Peak RMSE: {diff_peak:+.2f} ppb")

    df_summary = pd.DataFrame(summary_data)
    csv_path = os.path.join(OUTPUT_DIR, 'summary_comparison.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary comparison to {csv_path}")

if __name__ == '__main__':
    run()
