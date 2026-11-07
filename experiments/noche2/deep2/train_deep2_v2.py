"""
experiments/noche2/deep2/train_deep2_v2.py
==========================================
Entrenamiento extendido para noche2_deep2:
1. tab_resnet_5seeds: Ensamble robusto de 5 semillas [42, 123, 999, 2024, 777].
2. tab_gated_honest: Arquitectura alternativa Gated MLP (SwiGLU + LayerNorm) para máxima diversidad de representación.
3. deep_mega_ensemble: Combinación de TabResNet 5-seeds + TabResNetQuantile + TabGated.
"""
import os, sys, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS_5 = [42, 123, 999, 2024, 777]

class TabGatedMLP(nn.Module):
    def __init__(self, num_stations, in_dim, embed_dim=16, hr_embed_dim=8, hidden_dim=384, num_layers=3, dropout=0.05):
        super().__init__()
        self.st_emb = nn.Embedding(num_stations, embed_dim)
        self.hr_emb = nn.Embedding(24, hr_embed_dim)
        total_in = in_dim + embed_dim + hr_embed_dim
        
        self.in_proj = nn.Linear(total_in, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'fc_gate': nn.Linear(hidden_dim, hidden_dim),
                'fc_val': nn.Linear(hidden_dim, hidden_dim),
                'fc_out': nn.Linear(hidden_dim, hidden_dim),
                'drop': nn.Dropout(dropout)
            }))
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_head = nn.Linear(hidden_dim, 1)

    def forward(self, st_idx, hr_idx, x_num):
        emb_st = self.st_emb(st_idx)
        emb_hr = self.hr_emb(hr_idx)
        x = torch.cat([x_num, emb_st, emb_hr], dim=1)
        h = self.in_proj(x)
        for layer in self.layers:
            norm_h = layer['norm'](h)
            gated = F.silu(layer['fc_gate'](norm_h)) * layer['fc_val'](norm_h)
            out = layer['drop'](layer['fc_out'](gated))
            h = h + out
        h = self.out_norm(h)
        return self.out_head(h)


def run_5seeds_experiment(device):
    print("\n" + "="*75)
    print("  EJECUTANDO: tab_resnet_5seeds (5 semillas)")
    print("="*75)
    
    rows = []
    for h in HORIZONS:
        t0 = time.time()
        print(f"\n>>> [5-Seeds] Procesando h = {h:02d} ...")
        dm = TabularDatasetManager(h=h, device=device, seed=SEED)
        use_w = (h >= 8)
        delta = 1.5 if h <= 4 else 2.0
        n_epochs = 10 if h <= 4 else 8

        va_preds, te_preds = [], []
        for s in SEEDS_5:
            torch.manual_seed(s)
            np.random.seed(s)
            model = TabResNet(len(MODEL_STATIONS), dm.input_dim, embed_dim=16, hr_embed_dim=8, hidden_dim=256, num_blocks=3, dropout=0.05).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
            batch_size = 2048
            n_train = len(dm.y_tr)
            n_batches = (n_train + batch_size - 1) // batch_size
            
            best_val = float('inf')
            best_vp, best_tp, best_st = None, None, None
            for ep in range(1, n_epochs + 1):
                model.train()
                perm = torch.randperm(n_train, device=device)
                for b in range(n_batches):
                    idx = perm[b*batch_size : (b+1)*batch_size]
                    optimizer.zero_grad()
                    pred = model(dm.tr_st[idx], dm.tr_hr[idx], dm.X_tr[idx])
                    err = pred - dm.y_tr[idx]
                    abs_err = torch.abs(err)
                    loss_raw = torch.where(abs_err < delta, 0.5 * (err**2), delta * (abs_err - 0.5 * delta))
                    loss = torch.mean(dm.weights_tr[idx] * loss_raw) if use_w else torch.mean(loss_raw)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
                model.eval()
                with torch.no_grad():
                    vp = model(dm.va_st, dm.va_hr, dm.X_va)
                    tp = model(dm.te_st, dm.te_hr, dm.X_te)
                    v_rmse = torch.sqrt(torch.mean((vp - dm.y_va)**2)).item()
                if v_rmse < best_val:
                    best_val = v_rmse
                    best_vp = vp.cpu().numpy().ravel()
                    best_tp = tp.cpu().numpy().ravel()
                    best_st = copy.deepcopy(model.state_dict())
            va_preds.append(best_vp)
            te_preds.append(best_tp)
            if s == SEEDS_5[0]:
                torch.save(best_st, os.path.join(CKPT_DIR, f"tab_resnet_5seeds_h{h}_s{s}.pt"))
            print(f"    Seed {s} -> Best Val RMSE: {best_val:.4f}")

        final_va = np.mean(va_preds, axis=0)
        final_te = np.mean(te_preds, axis=0)
        
        # Guardar preds parquet
        va_df_out = pd.DataFrame({
            'Date': dm.va_df['Date'], 'Estacion': dm.va_df['Estacion'], 'h': h, 'split': 'val',
            'y_true': dm.va_df['Target'].values.astype('float32'), 'y_pred': final_va.astype('float32')
        })
        te_df_out = pd.DataFrame({
            'Date': dm.te_df['Date'], 'Estacion': dm.te_df['Estacion'], 'h': h, 'split': 'test',
            'y_true': dm.te_df['Target'].values.astype('float32'), 'y_pred': final_te.astype('float32')
        })
        pd.concat([va_df_out, te_df_out], ignore_index=True).to_parquet(
            os.path.join(PREDS_DIR, f"tab_resnet_5seeds_h{h}.parquet"), index=False
        )

        val_rmse = float(np.sqrt(np.mean((final_va - dm.va_df['Target'].values)**2)))
        m_te = evaluate_test(dm.te_df, final_te, h=h)
        
        # Corr vs LGBM
        lgb_df = pd.read_parquet(os.path.join(ROOT, f"experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet"))
        lgb_te = lgb_df[lgb_df['split'] == 'test']['y_pred'].values
        lgb_res = dm.te_df[f'Target_{h}h'].values - lgb_te
        nn_res = dm.te_df[f'Target_{h}h'].values - final_te
        res_corr = float(np.corrcoef(lgb_res, nn_res)[0, 1])
        
        blend_te = 0.5 * lgb_te + 0.5 * final_te
        m_blend = evaluate_test(dm.te_df, blend_te, h=h)

        log_result(
            experiment='noche2_deep2', variant='tab_resnet_5seeds', h=h, m=m_te,
            notes=f"5 seeds average. corr={res_corr:.3f}, blend_rmse={m_blend['RMSE']:.3f}"
        )

        print(f"  [h={h:02d} 5-SEEDS] Val: {val_rmse:.4f} | Test: {m_te['RMSE']:.4f} | Blend: {m_blend['RMSE']:.4f} | Peaks RMSE: {m_te['RMSE_peaks']:.2f} | Time: {time.time()-t0:.1f}s")
        rows.append({
            'h': h, 'val_rmse': val_rmse, 'test_rmse': m_te['RMSE'], 'blend_test_rmse': m_blend['RMSE'],
            'test_r2': m_te['R2'], 'test_peaks_rmse': m_te['RMSE_peaks'], 'test_peaks_bias': m_te['bias_peaks'],
            'res_corr_vs_lgb': res_corr
        })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(BASE_DIR, 'summary_tab_resnet_5seeds.csv'), index=False)
    return df_res


def run_gated_mlp_experiment(device):
    print("\n" + "="*75)
    print("  EJECUTANDO: tab_gated_honest (SwiGLU + LayerNorm)")
    print("="*75)
    
    rows = []
    for h in HORIZONS:
        t0 = time.time()
        print(f"\n>>> [TabGated] Procesando h = {h:02d} ...")
        dm = TabularDatasetManager(h=h, device=device, seed=SEED)
        use_w = (h >= 8)
        delta = 1.5 if h <= 4 else 2.0
        n_epochs = 10 if h <= 4 else 8

        va_preds, te_preds = [], []
        for s in [42, 123, 999]:
            torch.manual_seed(s)
            np.random.seed(s)
            model = TabGatedMLP(len(MODEL_STATIONS), dm.input_dim, hidden_dim=384, num_layers=3, dropout=0.05).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
            batch_size = 2048
            n_train = len(dm.y_tr)
            n_batches = (n_train + batch_size - 1) // batch_size
            
            best_val = float('inf')
            best_vp, best_tp, best_st = None, None, None
            for ep in range(1, n_epochs + 1):
                model.train()
                perm = torch.randperm(n_train, device=device)
                for b in range(n_batches):
                    idx = perm[b*batch_size : (b+1)*batch_size]
                    optimizer.zero_grad()
                    pred = model(dm.tr_st[idx], dm.tr_hr[idx], dm.X_tr[idx])
                    err = pred - dm.y_tr[idx]
                    abs_err = torch.abs(err)
                    loss_raw = torch.where(abs_err < delta, 0.5 * (err**2), delta * (abs_err - 0.5 * delta))
                    loss = torch.mean(dm.weights_tr[idx] * loss_raw) if use_w else torch.mean(loss_raw)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
                model.eval()
                with torch.no_grad():
                    vp = model(dm.va_st, dm.va_hr, dm.X_va)
                    tp = model(dm.te_st, dm.te_hr, dm.X_te)
                    v_rmse = torch.sqrt(torch.mean((vp - dm.y_va)**2)).item()
                if v_rmse < best_val:
                    best_val = v_rmse
                    best_vp = vp.cpu().numpy().ravel()
                    best_tp = tp.cpu().numpy().ravel()
                    best_st = copy.deepcopy(model.state_dict())
            va_preds.append(best_vp)
            te_preds.append(best_tp)
            if s == 42:
                torch.save(best_st, os.path.join(CKPT_DIR, f"tab_gated_h{h}_s{s}.pt"))
            print(f"    Seed {s} -> Best Val RMSE: {best_val:.4f}")

        final_va = np.mean(va_preds, axis=0)
        final_te = np.mean(te_preds, axis=0)

        va_df_out = pd.DataFrame({
            'Date': dm.va_df['Date'], 'Estacion': dm.va_df['Estacion'], 'h': h, 'split': 'val',
            'y_true': dm.va_df['Target'].values.astype('float32'), 'y_pred': final_va.astype('float32')
        })
        te_df_out = pd.DataFrame({
            'Date': dm.te_df['Date'], 'Estacion': dm.te_df['Estacion'], 'h': h, 'split': 'test',
            'y_true': dm.te_df['Target'].values.astype('float32'), 'y_pred': final_te.astype('float32')
        })
        pd.concat([va_df_out, te_df_out], ignore_index=True).to_parquet(
            os.path.join(PREDS_DIR, f"tab_gated_honest_h{h}.parquet"), index=False
        )

        val_rmse = float(np.sqrt(np.mean((final_va - dm.va_df['Target'].values)**2)))
        m_te = evaluate_test(dm.te_df, final_te, h=h)

        lgb_df = pd.read_parquet(os.path.join(ROOT, f"experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet"))
        lgb_te = lgb_df[lgb_df['split'] == 'test']['y_pred'].values
        lgb_res = dm.te_df[f'Target_{h}h'].values - lgb_te
        nn_res = dm.te_df[f'Target_{h}h'].values - final_te
        res_corr = float(np.corrcoef(lgb_res, nn_res)[0, 1])

        blend_te = 0.5 * lgb_te + 0.5 * final_te
        m_blend = evaluate_test(dm.te_df, blend_te, h=h)

        log_result(
            experiment='noche2_deep2', variant='tab_gated_honest', h=h, m=m_te,
            notes=f"TabGated SwiGLU 3-seeds. corr={res_corr:.3f}, blend_rmse={m_blend['RMSE']:.3f}"
        )

        print(f"  [h={h:02d} TabGated] Val: {val_rmse:.4f} | Test: {m_te['RMSE']:.4f} | Blend: {m_blend['RMSE']:.4f} | Peaks RMSE: {m_te['RMSE_peaks']:.2f} | Time: {time.time()-t0:.1f}s")
        rows.append({
            'h': h, 'val_rmse': val_rmse, 'test_rmse': m_te['RMSE'], 'blend_test_rmse': m_blend['RMSE'],
            'test_r2': m_te['R2'], 'test_peaks_rmse': m_te['RMSE_peaks'], 'test_peaks_bias': m_te['bias_peaks'],
            'res_corr_vs_lgb': res_corr
        })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(BASE_DIR, 'summary_tab_gated_honest.csv'), index=False)
    return df_res


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando ejecucion de variantes adicionales en GPU: {device}")
    run_5seeds_experiment(device)
    run_gated_mlp_experiment(device)
    print("\nVariantes adicionales completadas con exito.")
