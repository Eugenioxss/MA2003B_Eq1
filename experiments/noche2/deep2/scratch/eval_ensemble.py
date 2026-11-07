import os, sys, pandas as pd, numpy as np

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import evaluate_test, log_result

rows = []
for h in [1, 4, 8, 12, 24, 48]:
    p_h = pd.read_parquet(f'experiments/noche2/deep2/preds/tab_resnet_honest_h{h}.parquet')
    p_q = pd.read_parquet(f'experiments/noche2/deep2/preds/tab_resnet_quantile_h{h}.parquet')
    
    va_h = p_h[p_h['split'] == 'val'].reset_index(drop=True)
    te_h = p_h[p_h['split'] == 'test'].reset_index(drop=True)
    va_q = p_q[p_q['split'] == 'val'].reset_index(drop=True)
    te_q = p_q[p_q['split'] == 'test'].reset_index(drop=True)
    
    te_h[f'Target_{h}h'] = te_h['y_true']
    
    deep_va = 0.5 * va_h['y_pred'].values + 0.5 * va_q['y_pred'].values
    deep_te = 0.5 * te_h['y_pred'].values + 0.5 * te_q['y_pred'].values
    
    val_rmse = float(np.sqrt(np.mean((deep_va - va_h['y_true'].values)**2)))
    m_deep = evaluate_test(te_h, deep_te, h=h)
    
    # Save deep_ensemble preds
    va_df = pd.DataFrame({
        'Date': va_h['Date'], 'Estacion': va_h['Estacion'], 'h': h, 'split': 'val',
        'y_true': va_h['y_true'], 'y_pred': deep_va.astype('float32'),
        'q10': va_q['q10'], 'q90': va_q['q90']
    })
    te_df = pd.DataFrame({
        'Date': te_h['Date'], 'Estacion': te_h['Estacion'], 'h': h, 'split': 'test',
        'y_true': te_h['y_true'], 'y_pred': deep_te.astype('float32'),
        'q10': te_q['q10'], 'q90': te_q['q90']
    })
    df_all = pd.concat([va_df, te_df], ignore_index=True)
    df_all.to_parquet(f'experiments/noche2/deep2/preds/deep_ensemble_h{h}.parquet', index=False)
    
    # LGBM
    lgb_df = pd.read_parquet(f'experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet')
    lgb_te = lgb_df[lgb_df['split'] == 'test']['y_pred'].values
    m_lgb = evaluate_test(te_h, lgb_te, h=h)
    
    # Res corr vs LGBM
    lgb_res = te_h[f'Target_{h}h'].values - lgb_te
    deep_res = te_h[f'Target_{h}h'].values - deep_te
    corr = float(np.corrcoef(lgb_res, deep_res)[0, 1])
    
    # Blend LGBM + Deep
    blend = 0.5 * lgb_te + 0.5 * deep_te
    m_blend = evaluate_test(te_h, blend, h=h)
    
    # Log result
    log_result('noche2_deep2', 'deep_ensemble', h, m_deep, notes=f'Ensamble 50/50 TabResNet Honest + Quantile. corr={corr:.3f}')
    
    rows.append({
        'h': h,
        'val_rmse': val_rmse,
        'lgb_test_rmse': m_lgb['RMSE'],
        'deep_test_rmse': m_deep['RMSE'],
        'blend_test_rmse': m_blend['RMSE'],
        'deep_peaks_rmse': m_deep['RMSE_peaks'],
        'deep_peaks_bias': m_deep['bias_peaks'],
        'blend_peaks_rmse': m_blend['RMSE_peaks'],
        'blend_peaks_bias': m_blend['bias_peaks'],
        'res_corr_vs_lgb': corr
    })

res_df = pd.DataFrame(rows)
res_df.to_csv('experiments/noche2/deep2/summary_deep_ensemble.csv', index=False)
print(res_df.to_string())
