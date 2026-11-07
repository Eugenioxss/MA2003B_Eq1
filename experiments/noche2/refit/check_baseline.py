import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from common import metrics, evaluate_test

def check_baselines():
    inv = pd.read_csv('experiments/noche2/inventory.csv')
    print("=" * 80)
    print("VERIFICACION DE BASELINES D_honest_noise y E_honest_w VS INVENTORY.CSV")
    print("=" * 80)
    
    records = []
    for var in ['D_honest_noise', 'E_honest_w']:
        for h in [1, 4, 8, 12, 24, 48]:
            p = f'experiments/noche/v3_base/preds/{var}_h{h}.parquet'
            df = pd.read_parquet(p)
            va = df[df['split'] == 'val']
            te = df[df['split'] == 'test']
            te = te.copy()
            te[f'Target_{h}h'] = te['y_true']
            m_va = metrics(va['y_true'].values, va['y_pred'].values)
            m_te = evaluate_test(te, te['y_pred'].values, h)
            row = inv[(inv['exp'] == 'v3_base') & (inv['var'] == var) & (inv['h'] == h)].iloc[0]
            val_match = abs(m_va['RMSE'] - row['val_rmse']) < 1e-4
            test_match = abs(m_te['RMSE'] - row['test_rmse']) < 1e-4
            records.append({
                'var': var, 'h': h,
                'val_calc': m_va['RMSE'], 'val_inv': row['val_rmse'], 'val_ok': val_match,
                'test_calc': m_te['RMSE'], 'test_inv': row['test_rmse'], 'test_ok': test_match,
                'test_R2': m_te['R2'], 'test_RMSE_peaks': m_te['RMSE_peaks'], 'test_bias_peaks': m_te['bias_peaks']
            })
            print(f"{var:14s} h={h:2d} | Val: {m_va['RMSE']:7.4f} (inv {row['val_rmse']:7.4f}) [{'OK' if val_match else 'DIFF'}] | Test: {m_te['RMSE']:7.4f} (inv {row['test_rmse']:7.4f}) [{'OK' if test_match else 'DIFF'}] | R2: {m_te['R2']:.4f}")
    
    df_res = pd.DataFrame(records)
    df_res.to_csv('experiments/noche2/refit/baseline_verification.csv', index=False)
    print("=" * 80)

if __name__ == '__main__':
    check_baselines()
