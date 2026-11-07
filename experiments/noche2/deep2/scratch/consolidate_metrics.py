import os, sys, glob, pandas as pd, numpy as np

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import evaluate_test

EVAL_HORIZONS = [1, 4, 8, 12, 24, 48]
VARIANTS = [
    'tab_resnet_honest',
    'tab_resnet_quantile',
    'deep_ensemble',
    'tab_resnet_5seeds',
    'tab_gated_honest',
    'seq_corrected_honest',
    'lgbm_residual_honest'
]

results = {v: {} for v in VARIANTS}
lgbm_results = {}

for h in EVAL_HORIZONS:
    # LGBM
    lgb_df = pd.read_parquet(os.path.join(ROOT, f"experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet"))
    lgb_va = lgb_df[lgb_df['split'] == 'val'].reset_index(drop=True)
    lgb_te = lgb_df[lgb_df['split'] == 'test'].reset_index(drop=True)
    lgb_te[f'Target_{h}h'] = lgb_te['y_true']
    m_lgb = evaluate_test(lgb_te, lgb_te['y_pred'].values, h=h)
    va_lgb_rmse = float(np.sqrt(np.mean((lgb_va['y_true'] - lgb_va['y_pred'])**2)))
    lgbm_results[h] = {
        'val_rmse': va_lgb_rmse,
        'test_rmse': m_lgb['RMSE'],
        'test_r2': m_lgb['R2'],
        'peaks_rmse': m_lgb['RMSE_peaks'],
        'peaks_bias': m_lgb['bias_peaks']
    }

    for v in VARIANTS:
        p_path = os.path.join(ROOT, f"experiments/noche2/deep2/preds/{v}_h{h}.parquet")
        df_p = pd.read_parquet(p_path)
        va = df_p[df_p['split'] == 'val'].reset_index(drop=True)
        te = df_p[df_p['split'] == 'test'].reset_index(drop=True)
        te[f'Target_{h}h'] = te['y_true']
        
        m = evaluate_test(te, te['y_pred'].values, h=h)
        va_rmse = float(np.sqrt(np.mean((va['y_true'] - va['y_pred'])**2)))
        
        # Corr vs LGBM
        # Ensure aligned by Date & Estacion
        merged = pd.merge(te[['Date', 'Estacion', f'Target_{h}h', 'y_pred']], lgb_te[['Date', 'Estacion', 'y_pred']], on=['Date', 'Estacion'], suffixes=('_nn', '_lgb'))
        r_nn = merged[f'Target_{h}h'].values - merged['y_pred_nn'].values
        r_lgb = merged[f'Target_{h}h'].values - merged['y_pred_lgb'].values
        corr = float(np.corrcoef(r_nn, r_lgb)[0, 1])

        # Blend
        blend_pred = 0.5 * merged['y_pred_lgb'].values + 0.5 * merged['y_pred_nn'].values
        m_blend = evaluate_test(merged, blend_pred, h=h)

        results[v][h] = {
            'val_rmse': va_rmse,
            'test_rmse': m['RMSE'],
            'test_r2': m['R2'],
            'peaks_rmse': m['RMSE_peaks'],
            'peaks_bias': m['bias_peaks'],
            'res_corr': corr,
            'blend_test_rmse': m_blend['RMSE'],
            'blend_peaks_rmse': m_blend['RMSE_peaks'],
            'blend_peaks_bias': m_blend['bias_peaks']
        }

print("=== CONSOLIDATED METRICS ===")
for h in EVAL_HORIZONS:
    print(f"\n--- HORIZONTE h={h} ---")
    print(f"LGBM D_honest_noise: Val={lgbm_results[h]['val_rmse']:.4f}, Test={lgbm_results[h]['test_rmse']:.4f}, Pk_RMSE={lgbm_results[h]['peaks_rmse']:.2f}, Pk_Bias={lgbm_results[h]['peaks_bias']:.2f}")
    for v in VARIANTS:
        r = results[v][h]
        print(f"  {v:<22}: Val={r['val_rmse']:.4f}, Test={r['test_rmse']:.4f}, Pk_RMSE={r['peaks_rmse']:.2f}, Pk_Bias={r['peaks_bias']:.2f}, Corr={r['res_corr']:.3f}, Blend={r['blend_test_rmse']:.4f}")

# Save json for easy reference
import json
with open(os.path.join(ROOT, 'experiments/noche2/deep2/consolidated_metrics.json'), 'w') as f:
    json.dump({'lgbm': lgbm_results, 'variants': results}, f, indent=2)
print("\nGuardado en consolidated_metrics.json")
