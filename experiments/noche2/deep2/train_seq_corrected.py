"""
experiments/noche2/deep2/train_seq_corrected.py
================================================
Entrenamiento del modelo secuencial honesto corregido:
- Anclaje a persistencia fisica en t: y_pred = ReLU(O3(t) + delta)
- Perdida multi-horizonte normalizada por la varianza de cada horizonte
- Encoder temporal GRU sobre 48 horas de la red SIMA
- Inyeccion de ruido de pronostico numerico en train
- Evaluacion oficial en h in {1, 4, 8, 12, 24, 48}
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

def train_seq_corrected(device: torch.device):
    print("\n" + "="*75)
    print("  INICIANDO ENTRENAMIENTO: seq_corrected_honest")
    print(f"  Dispositivo: {device}")
    print("="*75)

    t0 = time.time()
    bundle = MultiHorizonDataBundle(seq_len=48, future_len=48)

    # 1. Calcular pesos normalizados de varianza por horizonte en train
    print("Calculando pesos de varianza por horizonte...")
    # Residuos respecto a persistencia O3(t)
    train_hours = 35064
    tgt_tr = bundle.targets[:, :train_hours]
    msk_tr = bundle.target_masks[:, :train_hours]
    
    stds_per_h = np.zeros(48, dtype=np.float32)
    for lead in range(1, 49):
        # target(t+lead) - target(t)
        y_lead = tgt_tr[:, lead:]
        y_base = tgt_tr[:, :-lead]
        m_lead = (msk_tr[:, lead:] == 1) & (msk_tr[:, :-lead] == 1)
        diff = (y_lead - y_base)[m_lead]
        stds_per_h[lead - 1] = float(np.std(diff)) if len(diff) > 0 else 10.0

    print("Desviacion estandar de delta por horizonte:")
    for h in EVAL_HORIZONS:
        print(f"  h={h:02d}: {stds_per_h[h-1]:.2f} ppb")

    # Pesos w_h = 1 / sigma_h^2 normalizados a que sum(w_h) = 48
    w_unnorm = 1.0 / (stds_per_h ** 2)
    w_norm = (w_unnorm / np.mean(w_unnorm)).astype(np.float32)
    torch_w = torch.tensor(w_norm, dtype=torch.float32, device=device).unsqueeze(0) # (1, 48)

    # 2. Inicializar modelo
    torch.manual_seed(42)
    np.random.seed(42)
    model = SeqCorrectedModel(
        past_channels=36,
        past_len=48,
        future_channels=18,
        future_len=48,
        spatial_dim=37,
        num_stations=len(MODEL_STATIONS),
        station_emb_dim=16,
        gru_hidden=128,
        future_proj_dim=32,
        hidden_dim=256,
        dropout=0.10
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 12
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # DataLoaders
    # Train: t from 48 to 35064 - 48
    train_loader = FastDataLoader(
        bundle, t_start=48, t_end=35064 - 48,
        batch_size=512, mode='honest', shuffle=True, add_noise=True, allow_clamp=False
    )
    val_loader = FastDataLoader(
        bundle, t_start=35064, t_end=43848,
        batch_size=512, mode='honest', shuffle=False, add_noise=False, allow_clamp=True
    )
    test_loader = FastDataLoader(
        bundle, t_start=43848, t_end=52608,
        batch_size=512, mode='honest', shuffle=False, add_noise=False, allow_clamp=True
    )

    best_val_rmse = float('inf')
    best_state = None

    print(f"\nIniciando bucle de entrenamiento ({epochs} epocas)...")
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        loss_acc = 0.0
        n_samples = 0

        for past, fut, spat, st, clim, tgt, msk in train_loader:
            past = past.to(device, non_blocking=True)
            fut = fut.to(device, non_blocking=True)
            spat = spat.to(device, non_blocking=True)
            st = st.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)

            # Base O3 en t para cada muestra del batch
            # past[:, -1, 0] es O3 normalizado en t. Reconstruimos O3 crudo o usamos target en t:
            # En FastDataLoader, spatial contiene reg stats, y target_batch es t+1..t+48.
            # O3 en t exacto:
            o3_t = []
            for s_idx, t_idx in zip(st.cpu().numpy(), train_loader.pairs[train_loader.batch_slice_start : train_loader.batch_slice_start + len(st), 1] if hasattr(train_loader, 'batch_slice_start') else train_loader.pairs[:len(st), 1]):
                pass
            # Mas directo y ultra-rapido:
            # past[:, -1, 0] es (O3(t) - mean) / std. Por lo tanto O3(t) = past[:, -1, 0] * std + mean
            o3_mean, o3_std = bundle.var_stats['O3']
            base_o3 = (past[:, -1, 0:1] * o3_std + o3_mean).clamp(min=0.0)

            optimizer.zero_grad()
            y_pred = model(past, fut, spat, st, base_o3)

            # Masked Huber loss ponderada por horizonte
            err = y_pred - tgt
            abs_err = torch.abs(err)
            delta = 2.0
            huber_raw = torch.where(abs_err < delta, 0.5 * (err**2), delta * (abs_err - 0.5 * delta))
            
            # Ponderacion de varianza por horizonte
            weighted_huber = huber_raw * torch_w
            loss = (weighted_huber * msk).sum() / (msk.sum() + 1e-6)

            loss.backward()
            optimizer.step()
            loss_acc += loss.item() * len(st)
            n_samples += len(st)

        scheduler.step()

        # Validacion rapida
        model.eval()
        va_sq_err = {h: [] for h in EVAL_HORIZONS}
        with torch.no_grad():
            for past, fut, spat, st, clim, tgt, msk in val_loader:
                past = past.to(device, non_blocking=True)
                fut = fut.to(device, non_blocking=True)
                spat = spat.to(device, non_blocking=True)
                st = st.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                msk = msk.to(device, non_blocking=True)

                base_o3 = (past[:, -1, 0:1] * o3_std + o3_mean).clamp(min=0.0)
                yp = model(past, fut, spat, st, base_o3)

                for h, h_idx in zip(EVAL_HORIZONS, EVAL_INDICES):
                    valid = (msk[:, h_idx] == 1)
                    if valid.any():
                        sq = (yp[valid, h_idx] - tgt[valid, h_idx]) ** 2
                        va_sq_err[h].append(sq.cpu().numpy())

        ep_va_rmse = {h: float(np.sqrt(np.mean(np.concatenate(va_sq_err[h])))) for h in EVAL_HORIZONS}
        mean_va_rmse = float(np.mean(list(ep_va_rmse.values())))

        if mean_va_rmse < best_val_rmse:
            best_val_rmse = mean_va_rmse
            best_state = copy.deepcopy(model.state_dict())

        print(f"  Ep {ep:02d} ({time.time()-t_ep:.1f}s) | Val RMSEs: h1={ep_va_rmse[1]:.2f}, h4={ep_va_rmse[4]:.2f}, h8={ep_va_rmse[8]:.2f}, h12={ep_va_rmse[12]:.2f}, h24={ep_va_rmse[24]:.2f}, h48={ep_va_rmse[48]:.2f} | Mean: {mean_va_rmse:.2f}")

    print("\nCargando mejor estado de validacion...")
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(CKPT_DIR, "seq_corrected_honest.pt"))

    # Generar predicciones completas val y test
    print("Generando predicciones oficiales de prueba...")
    def run_inference(loader):
        model.eval()
        preds_dict = {h: [] for h in EVAL_HORIZONS}
        with torch.no_grad():
            for past, fut, spat, st, clim, tgt, msk in loader:
                past = past.to(device, non_blocking=True)
                fut = fut.to(device, non_blocking=True)
                spat = spat.to(device, non_blocking=True)
                st = st.to(device, non_blocking=True)
                base_o3 = (past[:, -1, 0:1] * o3_std + o3_mean).clamp(min=0.0)
                yp = model(past, fut, spat, st, base_o3)
                for h, h_idx in zip(EVAL_HORIZONS, EVAL_INDICES):
                    preds_dict[h].append(yp[:, h_idx].cpu().numpy())
        return {h: np.concatenate(preds_dict[h]) for h in EVAL_HORIZONS}

    va_preds_all = run_inference(val_loader)
    te_preds_all = run_inference(test_loader)

    # Evaluar por horizonte y guardar segun contrato
    summary_rows = []
    for h in EVAL_HORIZONS:
        # Reconstruir dataframe oficial con evaluate_test
        fpath_h = os.path.join(ROOT, f"experiments/noche/v3_base/cache/features_h{h}.parquet")
        df_cache = pd.read_parquet(fpath_h)
        
        va_df = df_cache[(df_cache['split'] == 'val') & df_cache['Target'].notna()].copy().reset_index(drop=True)
        te_df = df_cache[(df_cache['split'] == 'test') & df_cache['Target'].notna()].copy().reset_index(drop=True)

        va_p = va_preds_all[h][:len(va_df)]
        te_p = te_preds_all[h][:len(te_df)]

        val_rmse = float(np.sqrt(np.mean((va_p - va_df['Target'].values)**2)))
        m_te = evaluate_test(te_df, te_p, h=h)

        # Correlacion de residuos vs LGBM
        lgb_path = os.path.join(ROOT, f"experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet")
        if os.path.exists(lgb_path):
            lgb_df = pd.read_parquet(lgb_path)
            lgb_te = lgb_df[lgb_df['split'] == 'test']['y_pred'].values
            lgb_res = te_df[f'Target_{h}h'].values - lgb_te
            nn_res = te_df[f'Target_{h}h'].values - te_p
            res_corr = float(np.corrcoef(lgb_res, nn_res)[0, 1])

            blend_te = 0.5 * lgb_te + 0.5 * te_p
            blend_m = evaluate_test(te_df, blend_te, h=h)
            blend_rmse = blend_m['RMSE']
            lgb_rmse = float(np.sqrt(np.mean(lgb_res**2)))
        else:
            res_corr, blend_rmse, lgb_rmse = float('nan'), float('nan'), float('nan')

        # Guardar preds parquet segun contrato
        va_out = pd.DataFrame({
            'Date': va_df['Date'], 'Estacion': va_df['Estacion'], 'h': h, 'split': 'val',
            'y_true': va_df['Target'].values.astype('float32'), 'y_pred': va_p.astype('float32')
        })
        te_out = pd.DataFrame({
            'Date': te_df['Date'], 'Estacion': te_df['Estacion'], 'h': h, 'split': 'test',
            'y_true': te_df['Target'].values.astype('float32'), 'y_pred': te_p.astype('float32')
        })
        pd.concat([va_out, te_out], ignore_index=True).to_parquet(
            os.path.join(PREDS_DIR, f"seq_corrected_honest_h{h}.parquet"), index=False
        )

        log_result(
            experiment='noche2_deep2', variant='seq_corrected_honest', h=h, m=m_te,
            notes=f"Sequential GRU + Persistence Anchor + Variance Normalization. corr={res_corr:.3f}, blend={blend_rmse:.3f}"
        )

        print(f"\n  [h={h:02d} SEQ SUMMARY] Val RMSE: {val_rmse:.4f} | Test RMSE: {m_te['RMSE']:.4f} (LGBM: {lgb_rmse:.4f})")
        print(f"     Peaks RMSE: {m_te['RMSE_peaks']:.2f} | Bias Peaks: {m_te['bias_peaks']:.2f} | Res Corr: {res_corr:.4f} | Blend: {blend_rmse:.4f}")

        summary_rows.append({
            'h': h, 'val_rmse': val_rmse, 'test_rmse': m_te['RMSE'], 'blend_test_rmse': blend_rmse,
            'test_r2': m_te['R2'], 'test_peaks_rmse': m_te['RMSE_peaks'], 'test_peaks_bias': m_te['bias_peaks'],
            'res_corr_vs_lgb': res_corr
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(BASE_DIR, 'summary_seq_corrected_honest.csv'), index=False)
    print(f"\nEntrenamiento secuencial corregido finalizado en {time.time()-t0:.1f}s.")
    return summary_df


if __name__ == '__main__':
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_seq_corrected(dev)
