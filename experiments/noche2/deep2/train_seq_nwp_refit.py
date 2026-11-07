"""
experiments/noche2/deep2/train_seq_nwp_refit.py
================================================
Modelo secuencial honesto corregido con consenso NWP y Refit 2020-2024:
- Target anclado a persistencia en t: y_pred = ReLU(O3(t) + delta)
- Perdida multi-horizonte normalizada por varianza inversa (1 / sigma_h^2)
- Encoder GRU de 48h sobre la red SIMA completa + meteorologia honesta futura t+1..t+48
- Esquema 4-fold OOF en 2024 por trimestres (Q1..Q4) para validacion
- Refit completo 2020-2024 con 3 semillas para test 2025
"""
import os, sys, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functools import partial

print = partial(print, flush=True)

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

from common import MODEL_STATIONS, HORIZONS, SEED, evaluate_test, log_result
from experiments.noche.deep.dataset import MultiHorizonDataBundle, FastDataLoader
from model_seq_corrected import SeqCorrectedModel

torch.set_num_threads(3)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(BASE_DIR, 'preds')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

EVAL_HORIZONS = [1, 4, 8, 12, 24, 48]
EVAL_INDICES = [h - 1 for h in EVAL_HORIZONS]

# Limites temporales en horas (0 = 2020-01-01 00:00)
T_TRAIN_START = 48
T_2024_START  = 35064
T_2025_START  = 43848
T_2025_END    = 52608

# Horas de inicio y fin de cada trimestre de 2024 (bisiesto: 366 dias = 8784 horas)
# Q1: Ene 31d + Feb 29d + Mar 31d = 91d = 2184h -> [35064, 37248)
# Q2: Abr 30d + May 31d + Jun 30d = 91d = 2184h -> [37248, 39432)
# Q3: Jul 31d + Ago 31d + Sep 30d = 92d = 2208h -> [39432, 41640)
# Q4: Oct 31d + Nov 30d + Dic 31d = 92d = 2208h -> [41640, 43848)
Q_RANGES = [
    (1, 35064, 37248),
    (2, 37248, 39432),
    (3, 39432, 41640),
    (4, 41640, 43848),
]


class CustomPairsLoader:
    """Loader rapido sobre un array explicito de pares (station_idx, time_idx)."""
    def __init__(self, bundle, pairs, batch_size=512, mode='honest', shuffle=True, add_noise=False, allow_clamp=True):
        self.bundle = bundle
        self.pairs = pairs
        self.n_samples = len(pairs)
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        self.add_noise = add_noise
        self.allow_clamp = allow_clamp

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.n_samples)
        else:
            indices = np.arange(self.n_samples)

        b = self.bundle
        L = b.seq_len
        H = b.future_len
        num_hours = b.num_hours

        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[start_idx : start_idx + self.batch_size]
            batch_pairs = self.pairs[batch_idx]

            s_arr = batch_pairs[:, 0]
            t_arr = batch_pairs[:, 1]

            past_batch = torch.stack([b.torch_past[s, t - L + 1 : t + 1] for s, t in zip(s_arr, t_arr)])
            spatial_batch = torch.stack([b.torch_spatial[t, s] for s, t in zip(s_arr, t_arr)])
            station_batch = torch.as_tensor(s_arr, dtype=torch.long)

            future_list, target_list, mask_list = [], [], []
            for s, t in zip(s_arr, t_arr):
                leads = np.clip(np.arange(t + 1, t + 1 + H), 0, num_hours - 1)
                f = torch.empty((H, 18), dtype=torch.float32)
                f[:24] = b.torch_day1[s, leads[:24]]
                f[24:] = b.torch_day2[s, leads[24:]]
                if self.add_noise:
                    f = f + torch.randn_like(f) * b.torch_noise_stds
                tgt = b.torch_targets[s, leads]
                msk = b.torch_masks[s, leads].clone()
                beyond = (np.arange(t + 1, t + 1 + H) >= num_hours)
                msk[beyond] = 0.0

                future_list.append(f)
                target_list.append(tgt)
                mask_list.append(msk)

            future_batch = torch.stack(future_list)
            target_batch = torch.stack(target_list)
            mask_batch = torch.stack(mask_list)

            yield past_batch, future_batch, spatial_batch, station_batch, target_batch, mask_batch


def run_seq_nwp_refit(device: torch.device):
    print("\n" + "="*75)
    print("  EJECUTANDO: seq_nwp_refit (Modelo Secuencial + 4-Fold OOF 2024 + Refit)")
    print("="*75)
    t0_global = time.time()

    bundle = MultiHorizonDataBundle(seq_len=48, future_len=48)
    o3_mean, o3_std = bundle.var_stats['O3']

    # 1. Ponderacion de varianza por horizonte
    tgt_tr = bundle.targets[:, :T_2024_START]
    msk_tr = bundle.target_masks[:, :T_2024_START]
    stds_per_h = np.zeros(48, dtype=np.float32)
    for lead in range(1, 49):
        y_lead = tgt_tr[:, lead:]
        y_base = tgt_tr[:, :-lead]
        m_lead = (msk_tr[:, lead:] == 1) & (msk_tr[:, :-lead] == 1)
        diff = (y_lead - y_base)[m_lead]
        stds_per_h[lead - 1] = float(np.std(diff)) if len(diff) > 0 else 10.0
    w_unnorm = 1.0 / (stds_per_h ** 2)
    w_norm = (w_unnorm / np.mean(w_unnorm)).astype(np.float32)
    torch_w = torch.tensor(w_norm, dtype=torch.float32, device=device).unsqueeze(0)

    # 2. Construir lista de pares completos para 2020-2023 y 2024
    print("Construyendo pares de entrenamiento y validacion...")
    all_stations_idx = np.arange(len(bundle.stations))

    # Pares base 2020-2023
    t_base = np.arange(T_TRAIN_START, T_2024_START)
    grid_base = np.meshgrid(all_stations_idx, t_base)
    pairs_base = np.stack([grid_base[0].ravel(), grid_base[1].ravel()], axis=1)

    # Filtrar pares donde O3(t) no sea nulo si se desea, o permitir fallback
    print(f"  Pares base 2020-2023: {len(pairs_base):,}")

    # Pares por trimestre de 2024
    q_pairs = {}
    for q, q_start, q_end in Q_RANGES:
        t_q = np.arange(q_start, q_end)
        g_q = np.meshgrid(all_stations_idx, t_q)
        q_pairs[q] = np.stack([g_q[0].ravel(), g_q[1].ravel()], axis=1)
        print(f"  Pares Q{q} (2024): {len(q_pairs[q]):,}")

    # Pares Test 2025
    t_test = np.arange(T_2025_START, T_2025_END)
    g_test = np.meshgrid(all_stations_idx, t_test)
    pairs_test = np.stack([g_test[0].ravel(), g_test[1].ravel()], axis=1)
    print(f"  Pares Test 2025: {len(pairs_test):,}")

    # --- 3. 4-Fold Cross-Fitting en 2024 ---
    q_oof_preds = {}
    fold_best_epochs = []

    for q, q_start, q_end in Q_RANGES:
        t_q0 = time.time()
        print(f"\n>>> Entrenando Fold Q{q} de 2024 ...")
        # Train fold = pairs_base + pairs de los otros 3 trimestres
        other_q_pairs = np.concatenate([q_pairs[oq] for oq in [1, 2, 3, 4] if oq != q], axis=0)
        train_fold_pairs = np.concatenate([pairs_base, other_q_pairs], axis=0)
        val_fold_pairs = q_pairs[q]

        tr_loader = CustomPairsLoader(bundle, train_fold_pairs, batch_size=2048, shuffle=True, add_noise=True)
        va_loader = CustomPairsLoader(bundle, val_fold_pairs, batch_size=2048, shuffle=False, add_noise=False)

        # Entrenar modelo
        torch.manual_seed(42)
        np.random.seed(42)
        model = SeqCorrectedModel(
            past_channels=36, past_len=48, future_channels=18, future_len=48,
            spatial_dim=37, num_stations=len(MODEL_STATIONS), station_emb_dim=16,
            gru_hidden=128, future_proj_dim=32, hidden_dim=256, dropout=0.10
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        n_epochs_fold = 4
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs_fold, eta_min=1e-5)

        best_v_rmse = float('inf')
        best_q_pred = None
        best_ep = 1

        for ep in range(1, n_epochs_fold + 1):
            model.train()
            for past, fut, spat, st, tgt, msk in tr_loader:
                past = past.to(device, non_blocking=True)
                fut = fut.to(device, non_blocking=True)
                spat = spat.to(device, non_blocking=True)
                st = st.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                msk = msk.to(device, non_blocking=True)

                base_o3 = (past[:, -1, 0:1] * o3_std + o3_mean).clamp(min=0.0)
                optimizer.zero_grad()
                yp = model(past, fut, spat, st, base_o3)

                err = yp - tgt
                abs_err = torch.abs(err)
                delta = 2.0
                huber_raw = torch.where(abs_err < delta, 0.5 * (err**2), delta * (abs_err - 0.5 * delta))
                loss = ((huber_raw * torch_w) * msk).sum() / (msk.sum() + 1e-6)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validar en Q
            model.eval()
            val_sq = []
            q_preds_list = []
            with torch.no_grad():
                for past, fut, spat, st, tgt, msk in va_loader:
                    past = past.to(device, non_blocking=True)
                    fut = fut.to(device, non_blocking=True)
                    spat = spat.to(device, non_blocking=True)
                    st = st.to(device, non_blocking=True)
                    tgt = tgt.to(device, non_blocking=True)
                    msk = msk.to(device, non_blocking=True)

                    base_o3 = (past[:, -1, 0:1] * o3_std + o3_mean).clamp(min=0.0)
                    yp = model(past, fut, spat, st, base_o3)
                    q_preds_list.append(yp.cpu().numpy())

                    for h_idx in EVAL_INDICES:
                        valid = (msk[:, h_idx] == 1)
                        if valid.any():
                            val_sq.append(((yp[valid, h_idx] - tgt[valid, h_idx])**2).cpu().numpy())

            ep_rmse = float(np.sqrt(np.mean(np.concatenate(val_sq))))
            if ep_rmse < best_v_rmse:
                best_v_rmse = ep_rmse
                best_q_pred = np.concatenate(q_preds_list, axis=0)
                best_ep = ep

        print(f"  Fold Q{q} terminado en {time.time()-t_q0:.1f}s | Mejor Epoca: {best_ep} | Val RMSE promedio: {best_v_rmse:.4f}")
        q_oof_preds[q] = best_q_pred
        fold_best_epochs.append(best_ep)

    refit_epochs = max(2, int(np.round(np.median(fold_best_epochs))))
    print(f"\nMediana de epocas en 2024: {refit_epochs}. Iniciando Refit completo 2020-2024...")

    # Concatenar predicciones OOF de 2024
    all_2024_pairs = np.concatenate([q_pairs[q] for q in [1, 2, 3, 4]], axis=0)
    all_2024_preds = np.concatenate([q_oof_preds[q] for q in [1, 2, 3, 4]], axis=0)
    
    st_val_list = [bundle.stations[s] for s in all_2024_pairs[:, 0]]
    dt_val_list = [bundle.dates[t] for t in all_2024_pairs[:, 1]]
    df_val_oof = pd.DataFrame({'Date': dt_val_list, 'Estacion': st_val_list})
    for h, h_idx in zip(EVAL_HORIZONS, EVAL_INDICES):
        df_val_oof[f'pred_h{h}'] = all_2024_preds[:, h_idx].astype(np.float32)

    # --- 4. Refit Completo 2020-2024 con 3 Semillas ---
    train_full_pairs = np.concatenate([pairs_base, all_2024_pairs], axis=0)
    tr_full_loader = CustomPairsLoader(bundle, train_full_pairs, batch_size=2048, shuffle=True, add_noise=True)
    te_loader = CustomPairsLoader(bundle, pairs_test, batch_size=2048, shuffle=False, add_noise=False)

    test_seed_preds = []
    for s in [42, 123, 999]:
        t_s0 = time.time()
        torch.manual_seed(s)
        np.random.seed(s)
        model = SeqCorrectedModel(
            past_channels=36, past_len=48, future_channels=18, future_len=48,
            spatial_dim=37, num_stations=len(MODEL_STATIONS), station_emb_dim=16,
            gru_hidden=128, future_proj_dim=32, hidden_dim=256, dropout=0.10
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=refit_epochs, eta_min=1e-5)

        for ep in range(1, refit_epochs + 1):
            model.train()
            for past, fut, spat, st, tgt, msk in tr_full_loader:
                past = past.to(device, non_blocking=True)
                fut = fut.to(device, non_blocking=True)
                spat = spat.to(device, non_blocking=True)
                st = st.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                msk = msk.to(device, non_blocking=True)

                base_o3 = (past[:, -1, 0:1] * o3_std + o3_mean).clamp(min=0.0)
                optimizer.zero_grad()
                yp = model(past, fut, spat, st, base_o3)
                err = yp - tgt
                abs_err = torch.abs(err)
                loss_raw = torch.where(abs_err < 2.0, 0.5 * (err**2), 2.0 * (abs_err - 1.0))
                loss = ((loss_raw * torch_w) * msk).sum() / (msk.sum() + 1e-6)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Inferencia Test 2025
        model.eval()
        s_preds = []
        with torch.no_grad():
            for past, fut, spat, st, tgt, msk in te_loader:
                past = past.to(device, non_blocking=True)
                fut = fut.to(device, non_blocking=True)
                spat = spat.to(device, non_blocking=True)
                st = st.to(device, non_blocking=True)
                base_o3 = (past[:, -1, 0:1] * o3_std + o3_mean).clamp(min=0.0)
                yp = model(past, fut, spat, st, base_o3)
                s_preds.append(yp.cpu().numpy())
        test_seed_preds.append(np.concatenate(s_preds, axis=0))
        print(f"  Semilla {s} refit completada en {time.time()-t_s0:.1f}s")

    test_pred_avg = np.mean(test_seed_preds, axis=0) # (N, 48)

    st_te_list = [bundle.stations[s] for s in pairs_test[:, 0]]
    dt_te_list = [bundle.dates[t] for t in pairs_test[:, 1]]
    df_test_preds = pd.DataFrame({'Date': dt_te_list, 'Estacion': st_te_list})
    for h, h_idx in zip(EVAL_HORIZONS, EVAL_INDICES):
        df_test_preds[f'pred_h{h}'] = test_pred_avg[:, h_idx].astype(np.float32)

    # --- 5. Alinear y Evaluar por Horizonte ---
    print("\nAlineando predicciones y evaluando con evaluate_test...")
    summary_rows = []
    for h in EVAL_HORIZONS:
        fpath_h = os.path.join(ROOT, f"experiments/noche/v3_base/cache/features_h{h}.parquet")
        df_cache = pd.read_parquet(fpath_h)
        
        va_df = df_cache[(df_cache['split'] == 'val') & df_cache['Target'].notna()].copy()
        te_df = df_cache[(df_cache['split'] == 'test') & df_cache['Target'].notna()].copy()

        va_merged = pd.merge(va_df[['Date', 'Estacion', 'Target']], df_val_oof[['Date', 'Estacion', f'pred_h{h}']], on=['Date', 'Estacion'], how='inner')
        te_merged = pd.merge(te_df[['Date', 'Estacion', f'Target_{h}h']], df_test_preds[['Date', 'Estacion', f'pred_h{h}']], on=['Date', 'Estacion'], how='inner')

        va_p = va_merged[f'pred_h{h}'].values
        va_t = va_merged['Target'].values
        val_rmse = float(np.sqrt(np.mean((va_p - va_t)**2)))

        te_p = te_merged[f'pred_h{h}'].values
        te_t = te_merged[f'Target_{h}h'].values
        m_te = evaluate_test(te_merged, te_p, h=h)

        # Correlacion vs LGBM
        lgb_df = pd.read_parquet(os.path.join(ROOT, f"experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet"))
        lgb_te = lgb_df[lgb_df['split'] == 'test'].reset_index(drop=True)
        te_with_lgb = pd.merge(te_merged, lgb_te[['Date', 'Estacion', 'y_pred']], on=['Date', 'Estacion'], how='inner')
        
        lgb_res = te_with_lgb[f'Target_{h}h'].values - te_with_lgb['y_pred'].values
        seq_res = te_with_lgb[f'Target_{h}h'].values - te_with_lgb[f'pred_h{h}'].values
        res_corr = float(np.corrcoef(lgb_res, seq_res)[0, 1])

        blend_pred = 0.5 * te_with_lgb['y_pred'].values + 0.5 * te_with_lgb[f'pred_h{h}'].values
        m_blend = evaluate_test(te_with_lgb, blend_pred, h=h)

        # Guardar parquet oficial
        va_out = pd.DataFrame({
            'Date': va_merged['Date'], 'Estacion': va_merged['Estacion'], 'h': h, 'split': 'val',
            'y_true': va_t.astype('float32'), 'y_pred': va_p.astype('float32')
        })
        te_out = pd.DataFrame({
            'Date': te_merged['Date'], 'Estacion': te_merged['Estacion'], 'h': h, 'split': 'test',
            'y_true': te_t.astype('float32'), 'y_pred': te_p.astype('float32')
        })
        pd.concat([va_out, te_out], ignore_index=True).to_parquet(
            os.path.join(PREDS_DIR, f"seq_nwp_refit_h{h}.parquet"), index=False
        )

        log_result(
            experiment='noche2_deep2', variant='seq_nwp_refit', h=h, m=m_te,
            notes=f"Sequential GRU + 4-fold OOF 2024 + Refit 2020-2024. corr={res_corr:.3f}, blend={m_blend['RMSE']:.3f}"
        )

        print(f"h={h:02d} | Val OOF: {val_rmse:.4f} | Test RMSE: {m_te['RMSE']:.4f} | Blend: {m_blend['RMSE']:.4f} | Res Corr: {res_corr:.4f} | Pk RMSE: {m_te['RMSE_peaks']:.2f}")
        summary_rows.append({
            'h': h, 'val_rmse': val_rmse, 'test_rmse': m_te['RMSE'], 'blend_test_rmse': m_blend['RMSE'],
            'test_r2': m_te['R2'], 'test_peaks_rmse': m_te['RMSE_peaks'], 'test_peaks_bias': m_te['bias_peaks'],
            'res_corr_vs_lgb': res_corr
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(BASE_DIR, 'summary_seq_nwp_refit.csv'), index=False)
    print(f"\nseq_nwp_refit completado exitosamente en {time.time()-t0_global:.1f}s.")
    return df_sum


if __name__ == '__main__':
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_seq_nwp_refit(dev)
