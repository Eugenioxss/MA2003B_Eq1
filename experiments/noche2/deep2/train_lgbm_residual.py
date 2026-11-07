"""
experiments/noche2/deep2/train_lgbm_residual.py
================================================
Entrena una red TabResNet para predecir el residuo de LightGBM: r = y - y_pred_lgbm.
- En train: r_tr = y_tr - y_oof_lgbm
- En val: r_va = y_va - y_val_lgbm
- En test: y_final = y_test_lgbm + r_pred_nn
"""
import os, sys, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial

print = partial(print, flush=True)

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

from common import (
    MODEL_STATIONS, HORIZONS, SEED, evaluate_test, log_result
)
from dataset import TabularDatasetManager
from model import TabResNet

torch.set_num_threads(3)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(BASE_DIR, 'preds')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
OOF_DIR = os.path.join(BASE_DIR, 'oof')
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS = [42, 123, 999]

def train_residual_for_h(h: int, device: torch.device):
    print(f"\n{'='*75}\n  ENTRENANDO RESIDUAL LEARNING SOBRE LGBM PARA h={h:02d}\n{'='*75}")
    t0 = time.time()
    oof_path = os.path.join(OOF_DIR, f'lgbm_oof_h{h}.parquet')
    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"No existe el archivo OOF: {oof_path}")

    oof_df = pd.read_parquet(oof_path)
    dm = TabularDatasetManager(h=h, device=device, seed=SEED)

    # Alinear predicciones de LGBM
    oof_tr = oof_df[oof_df['split'] == 'train'].reset_index(drop=True)
    oof_va = oof_df[oof_df['split'] == 'val'].reset_index(drop=True)
    oof_te = oof_df[oof_df['split'] == 'test'].reset_index(drop=True)

    y_lgb_tr = oof_tr['y_pred_lgbm'].values.astype(np.float32)
    y_lgb_va = oof_va['y_pred_lgbm'].values.astype(np.float32)
    y_lgb_te = oof_te['y_pred_lgbm'].values.astype(np.float32)

    # Residual targets: r = y - y_lgb
    r_tr_np = dm.tr_df['Target'].values.astype(np.float32) - y_lgb_tr
    r_va_np = dm.va_df['Target'].values.astype(np.float32) - y_lgb_va
    r_te_np = dm.te_df['Target'].values.astype(np.float32) - y_lgb_te

    r_tr = torch.tensor(r_tr_np, dtype=torch.float32, device=device).unsqueeze(1)
    r_va = torch.tensor(r_va_np, dtype=torch.float32, device=device).unsqueeze(1)
    r_te = torch.tensor(r_te_np, dtype=torch.float32, device=device).unsqueeze(1)

    # Incorporar y_lgb estandarizado como feature continua adicional
    lgb_mean = float(np.mean(y_lgb_tr))
    lgb_std = float(np.std(y_lgb_tr)) + 1e-4

    feat_lgb_tr = torch.tensor((y_lgb_tr - lgb_mean) / lgb_std, dtype=torch.float32, device=device).unsqueeze(1)
    feat_lgb_va = torch.tensor((y_lgb_va - lgb_mean) / lgb_std, dtype=torch.float32, device=device).unsqueeze(1)
    feat_lgb_te = torch.tensor((y_lgb_te - lgb_mean) / lgb_std, dtype=torch.float32, device=device).unsqueeze(1)

    X_tr = torch.cat([dm.X_tr, feat_lgb_tr], dim=1)
    X_va = torch.cat([dm.X_va, feat_lgb_va], dim=1)
    X_te = torch.cat([dm.X_te, feat_lgb_te], dim=1)

    input_dim = X_tr.shape[1]
    n_epochs = 10 if h <= 4 else 8
    delta = 1.5 if h <= 4 else 2.0
    use_w = (h >= 8)

    va_res_preds, te_res_preds = [], []

    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        model = TabResNet(len(MODEL_STATIONS), input_dim, embed_dim=16, hr_embed_dim=8, hidden_dim=256, num_blocks=3, dropout=0.05).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
        batch_size = 2048
        n_train = len(r_tr)
        n_batches = (n_train + batch_size - 1) // batch_size

        best_val = float('inf')
        best_vp, best_tp, best_st = None, None, None

        for ep in range(1, n_epochs + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            for b in range(n_batches):
                idx = perm[b*batch_size : (b+1)*batch_size]
                optimizer.zero_grad()
                pred_r = model(dm.tr_st[idx], dm.tr_hr[idx], X_tr[idx])
                err = pred_r - r_tr[idx]
                abs_err = torch.abs(err)
                loss_raw = torch.where(abs_err < delta, 0.5 * (err**2), delta * (abs_err - 0.5 * delta))
                loss = torch.mean(dm.weights_tr[idx] * loss_raw) if use_w else torch.mean(loss_raw)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                vp = model(dm.va_st, dm.va_hr, X_va)
                tp = model(dm.te_st, dm.te_hr, X_te)
                
                # Evaluar RMSE de la predicción final corregida en val: y_final = y_lgb + r_pred
                y_final_va = torch.clamp(torch.tensor(y_lgb_va, device=device).unsqueeze(1) + vp, min=0.0)
                v_rmse = torch.sqrt(torch.mean((y_final_va - dm.y_va)**2)).item()

            if v_rmse < best_val:
                best_val = v_rmse
                best_vp = vp.cpu().numpy().ravel()
                best_tp = tp.cpu().numpy().ravel()
                best_st = copy.deepcopy(model.state_dict())

        va_res_preds.append(best_vp)
        te_res_preds.append(best_tp)
        if s == SEEDS[0]:
            torch.save(best_st, os.path.join(CKPT_DIR, f"lgbm_residual_honest_h{h}_s{s}.pt"))
        print(f"    Seed {s} -> Best Val RMSE (LGBM+Res): {best_val:.4f}")

    final_r_va = np.mean(va_res_preds, axis=0)
    final_r_te = np.mean(te_res_preds, axis=0)

    # Predicción final corregida
    final_y_va = np.clip(y_lgb_va + final_r_va, 0.0, None)
    final_y_te = np.clip(y_lgb_te + final_r_te, 0.0, None)

    # Guardar preds parquet oficiales
    va_df_out = pd.DataFrame({
        'Date': dm.va_df['Date'], 'Estacion': dm.va_df['Estacion'], 'h': h, 'split': 'val',
        'y_true': dm.va_df['Target'].values.astype('float32'), 'y_pred': final_y_va.astype('float32')
    })
    te_df_out = pd.DataFrame({
        'Date': dm.te_df['Date'], 'Estacion': dm.te_df['Estacion'], 'h': h, 'split': 'test',
        'y_true': dm.te_df['Target'].values.astype('float32'), 'y_pred': final_y_te.astype('float32')
    })
    pd.concat([va_df_out, te_df_out], ignore_index=True).to_parquet(
        os.path.join(PREDS_DIR, f"lgbm_residual_honest_h{h}.parquet"), index=False
    )

    val_rmse = float(np.sqrt(np.mean((final_y_va - dm.va_df['Target'].values)**2)))
    m_te = evaluate_test(dm.te_df, final_y_te, h=h)

    # Correlación de residuos vs LGBM
    lgb_res = dm.te_df[f'Target_{h}h'].values - y_lgb_te
    corrected_res = dm.te_df[f'Target_{h}h'].values - final_y_te
    res_corr = float(np.corrcoef(lgb_res, corrected_res)[0, 1])

    lgb_test_rmse = float(np.sqrt(np.mean(lgb_res**2)))

    log_result(
        experiment='noche2_deep2', variant='lgbm_residual_honest', h=h, m=m_te,
        notes=f"Residual learning over LightGBM. lgb={lgb_test_rmse:.3f} -> corr_res={m_te['RMSE']:.3f}, corr={res_corr:.3f}"
    )

    print(f"\n  [h={h:02d} RESIDUAL SUMMARY] Val RMSE: {val_rmse:.4f} | Test RMSE: {m_te['RMSE']:.4f} (LGBM: {lgb_test_rmse:.4f})")
    print(f"     Peaks RMSE: {m_te['RMSE_peaks']:.2f} | Bias Peaks: {m_te['bias_peaks']:.2f} | Res Corr vs LGBM: {res_corr:.4f} | Time: {time.time()-t0:.1f}s")

    return {
        'h': h, 'val_rmse': val_rmse, 'test_rmse': m_te['RMSE'], 'lgb_test_rmse': lgb_test_rmse,
        'test_r2': m_te['R2'], 'test_peaks_rmse': m_te['RMSE_peaks'], 'test_peaks_bias': m_te['bias_peaks'],
        'res_corr_vs_lgb': res_corr
    }

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando ejecucion train_lgbm_residual.py en {device}")
    rows = []
    for h in HORIZONS:
        res = train_residual_for_h(h, device)
        rows.append(res)
    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(BASE_DIR, 'summary_lgbm_residual_honest.csv'), index=False)
    print("\nResidual learning finalizado exitosamente.")
