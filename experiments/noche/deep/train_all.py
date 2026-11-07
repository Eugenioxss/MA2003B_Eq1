"""
experiments/noche/deep/train_all.py
-----------------------------------
Master training, evaluation, logging, and prediction generation script for noche_deep.
Runs:
  1. TiDE-MLP (MSE Loss) -> evaluated as 'tide_honest' and 'tide_pp'
  2. TiDE-MLP (Pinball Quantile Loss: q10, q50, q90) -> evaluated as 'tide_quantile_honest' and 'tide_quantile_pp'
  3. GRU-TiDE (Recurrent Encoder, MSE Loss) -> evaluated as 'gru_honest'
  4. Deep Multi-Horizon Ensemble -> evaluated as 'ensemble_honest' and 'ensemble_pp'

Strictly follows:
  - GPU RTX 3060 with torch.set_num_threads(3)
  - Train: 2020-2023, Val: 2024, Test: 2025 (test used ONLY for final evaluation)
  - Climatology residual target (station x month x hour) fit strictly on train
  - Honest meteo protocol: NWP previous_day1 (leads 1-24) and previous_day2 (leads 25-48)
  - Saves all predictions to experiments/noche/deep/preds/<variante>_h<h>.parquet
  - Logs official metrics with log_result(experiment='noche_deep', ...)
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import (
    load_panel, add_targets, split, evaluate_test, log_result,
    HORIZONS, MODEL_STATIONS, SEED
)
from experiments.noche.deep.dataset import MultiHorizonDataBundle, FastDataLoader
from experiments.noche.deep.model_tide import TiDEModel
from experiments.noche.deep.model_gru import GRUTiDEModel

# Strict resource controls
torch.set_num_threads(3)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(ROOT, 'experiments', 'noche', 'deep')
PREDS_DIR = os.path.join(OUTPUT_DIR, 'preds')
CKPT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

EVAL_HORIZONS = [1, 4, 8, 12, 24, 48]
EVAL_INDICES = [h - 1 for h in EVAL_HORIZONS]  # 0-indexed corresponding to h=1..48

def log_print(msg):
    t_str = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{t_str}] {msg}"
    print(formatted, flush=True)
    with open(os.path.join(OUTPUT_DIR, 'train.log'), 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')

def masked_mse_loss(y_pred, y_true, mask):
    diff2 = (y_pred - y_true) ** 2
    return (diff2 * mask).sum() / (mask.sum() + 1e-6)

def masked_pinball_loss(q_preds, y_true, mask, quantiles=(0.1, 0.5, 0.9)):
    total_loss = 0.0
    for i, q in enumerate(quantiles):
        yp = q_preds[:, :, i]
        err = y_true - yp
        loss_q = torch.maximum(q * err, (q - 1.0) * err)
        total_loss += (loss_q * mask).sum() / (mask.sum() + 1e-6)
    return total_loss

@torch.no_grad()
def predict_full_split(model, bundle, split='val', mode='honest', is_quantile=False, batch_size=512):
    """
    Runs full inference on split ('val' 2024 or 'test' 2025).
    Returns a DataFrame with ['Date', 'Estacion'] and predicted columns.
    """
    model.eval()
    if split == 'val':
        t_start, t_end = 35064, 43848
    elif split == 'test':
        t_start, t_end = 43848, 52608
    else:
        raise ValueError(f"Unknown split: {split}")

    loader = FastDataLoader(
        bundle, t_start=t_start, t_end=t_end,
        batch_size=batch_size, mode=mode,
        shuffle=False, add_noise=False, allow_clamp=True
    )

    all_preds = []
    for past, fut, spat, st, clim, tgt, msk in loader:
        past = past.to(DEVICE, non_blocking=True)
        fut = fut.to(DEVICE, non_blocking=True)
        spat = spat.to(DEVICE, non_blocking=True)
        st = st.to(DEVICE, non_blocking=True)
        clim = clim.to(DEVICE, non_blocking=True)
        
        out = model(past, fut, spat, st, clim, sigma_res=bundle.sigma_res)
        all_preds.append(out.cpu().numpy())

    concat_preds = np.concatenate(all_preds, axis=0)  # (N, 48) or (N, 48, 3)

    # Reconstruct Date and Estacion in exact order of loader
    pairs = loader.pairs
    stations_list = [bundle.stations[s] for s in pairs[:, 0]]
    dates_list = [bundle.dates[t] for t in pairs[:, 1]]

    df_out = pd.DataFrame({'Date': dates_list, 'Estacion': stations_list})
    
    if not is_quantile:
        for h, h_idx in zip(EVAL_HORIZONS, EVAL_INDICES):
            df_out[f'pred_h{h}'] = concat_preds[:, h_idx].astype(np.float32)
    else:
        for h, h_idx in zip(EVAL_HORIZONS, EVAL_INDICES):
            df_out[f'q10_h{h}'] = concat_preds[:, h_idx, 0].astype(np.float32)
            df_out[f'pred_h{h}'] = concat_preds[:, h_idx, 1].astype(np.float32)  # q50
            df_out[f'q90_h{h}'] = concat_preds[:, h_idx, 2].astype(np.float32)

    return df_out

def evaluate_and_save_variant(
    variant_name,
    df_val_preds,
    df_test_preds,
    val_ground_truth,
    test_ground_truth,
    notes=""
):
    """
    Evaluates test performance, logs to registry, and saves official parquet predictions.
    """
    log_print(f"\n=======================================================")
    log_print(f"Evaluating and Logging Variant: {variant_name}")
    log_print(f"Notes: {notes}")
    log_print(f"=======================================================")

    results = {}
    is_quantile = 'q10_h1' in df_val_preds.columns

    for h in EVAL_HORIZONS:
        # Merge val predictions
        m_val = pd.merge(
            val_ground_truth[['Date', 'Estacion', f'Target_{h}h']],
            df_val_preds[['Date', 'Estacion', f'pred_h{h}'] + ([f'q10_h{h}', f'q90_h{h}'] if is_quantile else [])],
            on=['Date', 'Estacion'],
            how='left'
        )
        m_val['h'] = h
        m_val['split'] = 'val'
        m_val = m_val.rename(columns={f'Target_{h}h': 'y_true', f'pred_h{h}': 'y_pred'})
        if is_quantile:
            m_val = m_val.rename(columns={f'q10_h{h}': 'q10', f'q90_h{h}': 'q90'})

        # Merge test predictions
        m_test = pd.merge(
            test_ground_truth[['Date', 'Estacion', f'Target_{h}h']],
            df_test_preds[['Date', 'Estacion', f'pred_h{h}'] + ([f'q10_h{h}', f'q90_h{h}'] if is_quantile else [])],
            on=['Date', 'Estacion'],
            how='left'
        )
        m_test['h'] = h
        m_test['split'] = 'test'
        m_test = m_test.rename(columns={f'Target_{h}h': 'y_true', f'pred_h{h}': 'y_pred'})
        if is_quantile:
            m_test = m_test.rename(columns={f'q10_h{h}': 'q10', f'q90_h{h}': 'q90'})

        # Official evaluation with evaluate_test()
        m_metrics = evaluate_test(test_ground_truth, m_test['y_pred'].values, h)
        results[h] = m_metrics

        log_print(
            f"h={h:2d} | RMSE: {m_metrics['RMSE']:6.3f} | R2: {m_metrics['R2']:6.4f} | "
            f"RMSE_peaks: {m_metrics['RMSE_peaks']:6.2f} | bias_peaks: {m_metrics['bias_peaks']:6.2f} | "
            f"skill: {m_metrics.get('skill_vs_clim', 0.0):6.4f}"
        )

        # Log to registry
        log_result('noche_deep', variant_name, h, m_metrics, notes=notes)

        # Save parquet predictions (val + test with Target observado)
        cols_save = ['Date', 'Estacion', 'h', 'split', 'y_true', 'y_pred']
        if is_quantile:
            cols_save += ['q10', 'q90']

        # Include all rows with observed target
        val_clean = m_val[m_val['y_true'].notna()][cols_save]
        test_clean = m_test[m_test['y_true'].notna()][cols_save]
        combined = pd.concat([val_clean, test_clean], ignore_index=True)

        parquet_path = os.path.join(PREDS_DIR, f"{variant_name}_h{h}.parquet")
        combined.to_parquet(parquet_path, index=False)
        log_print(f"  Saved {parquet_path} (val={len(val_clean)}, test={len(test_clean)})")

    return results

def train_tide_model(bundle, val_df, test_df, num_epochs=12, batch_size=512, lr=1e-3):
    log_print("\n>>> TRAINING MODEL 1: TiDE-MLP (MSE Loss) with Honest Protocol and Noise Injection")
    model = TiDEModel(
        past_channels=36,
        past_len=96,
        future_channels=18,
        future_len=48,
        spatial_dim=37,
        num_stations=13,
        station_emb_dim=16,
        future_proj_dim=16,
        hidden_dim=384,
        decoder_dim=64,
        num_encoder_blocks=3,
        num_heads=1,
        dropout=0.15
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_val_rmse = 999.0
    best_ckpt_path = os.path.join(CKPT_DIR, "best_tide_mse.pt")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        train_loader = FastDataLoader(
            bundle, t_start=95, t_end=35016,
            batch_size=batch_size, mode='honest',
            shuffle=True, add_noise=True
        )

        total_loss, total_count = 0.0, 0
        for past, fut, spat, st, clim, tgt, msk in train_loader:
            past = past.to(DEVICE, non_blocking=True)
            fut = fut.to(DEVICE, non_blocking=True)
            spat = spat.to(DEVICE, non_blocking=True)
            st = st.to(DEVICE, non_blocking=True)
            clim = clim.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)
            msk = msk.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(past, fut, spat, st, clim, sigma_res=bundle.sigma_res)
            loss = masked_mse_loss(y_pred, tgt, msk)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * len(tgt)
            total_count += len(tgt)

        train_rmse = np.sqrt(total_loss / total_count)
        scheduler.step()

        # Validation on 2024
        df_val = predict_full_split(model, bundle, split='val', mode='honest', is_quantile=False)
        val_rmses = []
        for h in EVAL_HORIZONS:
            m = pd.merge(val_df[['Date', 'Estacion', f'Target_{h}h']], df_val[['Date', 'Estacion', f'pred_h{h}']], on=['Date', 'Estacion'])
            ok = m[f'Target_{h}h'].notna()
            h_rmse = np.sqrt(np.mean((m.loc[ok, f'Target_{h}h'] - m.loc[ok, f'pred_h{h}']) ** 2))
            val_rmses.append(h_rmse)
        mean_val_rmse = np.mean(val_rmses)

        epoch_time = time.time() - t0
        log_print(
            f"Epoch {epoch:2d}/{num_epochs:2d} | Train RMSE: {train_rmse:5.2f} | "
            f"Val RMSE (avg 6h): {mean_val_rmse:5.3f} (h1:{val_rmses[0]:.2f}, h4:{val_rmses[1]:.2f}, h24:{val_rmses[4]:.2f}, h48:{val_rmses[5]:.2f}) | "
            f"Time: {epoch_time:4.1f}s"
        )

        if mean_val_rmse < best_val_rmse:
            best_val_rmse = mean_val_rmse
            torch.save(model.state_dict(), best_ckpt_path)
            log_print(f"  [*] Best model saved (val RMSE = {best_val_rmse:.4f})")

    # Load best checkpoint
    model.load_state_dict(torch.load(best_ckpt_path))
    log_print(f"Loaded best TiDE MSE checkpoint with Val RMSE: {best_val_rmse:.4f}")

    # 1. Evaluate Honest
    df_val_honest = predict_full_split(model, bundle, split='val', mode='honest')
    df_test_honest = predict_full_split(model, bundle, split='test', mode='honest')
    res_honest = evaluate_and_save_variant(
        'tide_honest', df_val_honest, df_test_honest, val_df, test_df,
        notes="TiDE-MLP honest (day1 1-24h, day2 25-48h, train NWP noise, spatial lag0)"
    )

    # 2. Evaluate PP (cota superior no operativa)
    df_val_pp = predict_full_split(model, bundle, split='val', mode='pp')
    df_test_pp = predict_full_split(model, bundle, split='test', mode='pp')
    res_pp = evaluate_and_save_variant(
        'tide_pp', df_val_pp, df_test_pp, val_df, test_df,
        notes="TiDE-MLP PP (seamless Open-Meteo in test, upper bound non-operational)"
    )

    return model, df_val_honest, df_test_honest, df_val_pp, df_test_pp, res_honest, res_pp

def train_tide_quantile_model(bundle, val_df, test_df, num_epochs=12, batch_size=512, lr=1e-3):
    log_print("\n>>> TRAINING MODEL 2: TiDE-MLP (Pinball Quantile Loss: q10, q50, q90)")
    model = TiDEModel(
        past_channels=36,
        past_len=96,
        future_channels=18,
        future_len=48,
        spatial_dim=37,
        num_stations=13,
        station_emb_dim=16,
        future_proj_dim=16,
        hidden_dim=384,
        decoder_dim=64,
        num_encoder_blocks=3,
        num_heads=3,  # q10, q50, q90
        dropout=0.15
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_val_loss = 999.0
    best_ckpt_path = os.path.join(CKPT_DIR, "best_tide_quantile.pt")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        train_loader = FastDataLoader(
            bundle, t_start=95, t_end=35016,
            batch_size=batch_size, mode='honest',
            shuffle=True, add_noise=True
        )

        total_loss, total_count = 0.0, 0
        for past, fut, spat, st, clim, tgt, msk in train_loader:
            past = past.to(DEVICE, non_blocking=True)
            fut = fut.to(DEVICE, non_blocking=True)
            spat = spat.to(DEVICE, non_blocking=True)
            st = st.to(DEVICE, non_blocking=True)
            clim = clim.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)
            msk = msk.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            q_preds = model(past, fut, spat, st, clim, sigma_res=bundle.sigma_res)
            loss = masked_pinball_loss(q_preds, tgt, msk)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * len(tgt)
            total_count += len(tgt)

        train_loss = total_loss / total_count
        scheduler.step()

        # Validation on 2024 using q50
        df_val = predict_full_split(model, bundle, split='val', mode='honest', is_quantile=True)
        val_rmses = []
        for h in EVAL_HORIZONS:
            m = pd.merge(val_df[['Date', 'Estacion', f'Target_{h}h']], df_val[['Date', 'Estacion', f'pred_h{h}']], on=['Date', 'Estacion'])
            ok = m[f'Target_{h}h'].notna()
            h_rmse = np.sqrt(np.mean((m.loc[ok, f'Target_{h}h'] - m.loc[ok, f'pred_h{h}']) ** 2))
            val_rmses.append(h_rmse)
        mean_val_rmse = np.mean(val_rmses)

        epoch_time = time.time() - t0
        log_print(
            f"Epoch {epoch:2d}/{num_epochs:2d} | Train Pinball: {train_loss:5.3f} | "
            f"Val RMSE q50: {mean_val_rmse:5.3f} (h1:{val_rmses[0]:.2f}, h4:{val_rmses[1]:.2f}, h24:{val_rmses[4]:.2f}, h48:{val_rmses[5]:.2f}) | "
            f"Time: {epoch_time:4.1f}s"
        )

        if mean_val_rmse < best_val_loss:
            best_val_loss = mean_val_rmse
            torch.save(model.state_dict(), best_ckpt_path)
            log_print(f"  [*] Best quantile model saved (val RMSE = {best_val_loss:.4f})")

    model.load_state_dict(torch.load(best_ckpt_path))
    log_print(f"Loaded best TiDE Quantile checkpoint with Val RMSE: {best_val_loss:.4f}")

    # 1. Evaluate Honest Quantile
    df_val_q_honest = predict_full_split(model, bundle, split='val', mode='honest', is_quantile=True)
    df_test_q_honest = predict_full_split(model, bundle, split='test', mode='honest', is_quantile=True)
    res_q_honest = evaluate_and_save_variant(
        'tide_quantile_honest', df_val_q_honest, df_test_q_honest, val_df, test_df,
        notes="TiDE-MLP pinball loss (q10/q50/q90), honest NWP meteo"
    )

    # 2. Evaluate PP Quantile
    df_val_q_pp = predict_full_split(model, bundle, split='val', mode='pp', is_quantile=True)
    df_test_q_pp = predict_full_split(model, bundle, split='test', mode='pp', is_quantile=True)
    res_q_pp = evaluate_and_save_variant(
        'tide_quantile_pp', df_val_q_pp, df_test_q_pp, val_df, test_df,
        notes="TiDE-MLP pinball loss (q10/q50/q90), PP seamless meteo in test"
    )

    return model, df_val_q_honest, df_test_q_honest, df_val_q_pp, df_test_q_pp, res_q_honest, res_q_pp

def train_gru_model(bundle, val_df, test_df, num_epochs=10, batch_size=512, lr=1e-3):
    log_print("\n>>> TRAINING MODEL 3: GRU-TiDE (Recurrent Encoder + Multi-Horizon Decoder)")
    model = GRUTiDEModel(
        past_channels=36,
        past_len=96,
        future_channels=18,
        future_len=48,
        spatial_dim=37,
        num_stations=13,
        station_emb_dim=16,
        future_proj_dim=16,
        hidden_dim=256,
        decoder_dim=64,
        dropout=0.15
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_val_rmse = 999.0
    best_ckpt_path = os.path.join(CKPT_DIR, "best_gru_tide.pt")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        train_loader = FastDataLoader(
            bundle, t_start=95, t_end=35016,
            batch_size=batch_size, mode='honest',
            shuffle=True, add_noise=True
        )

        total_loss, total_count = 0.0, 0
        for past, fut, spat, st, clim, tgt, msk in train_loader:
            past = past.to(DEVICE, non_blocking=True)
            fut = fut.to(DEVICE, non_blocking=True)
            spat = spat.to(DEVICE, non_blocking=True)
            st = st.to(DEVICE, non_blocking=True)
            clim = clim.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)
            msk = msk.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(past, fut, spat, st, clim, sigma_res=bundle.sigma_res)
            loss = masked_mse_loss(y_pred, tgt, msk)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * len(tgt)
            total_count += len(tgt)

        train_rmse = np.sqrt(total_loss / total_count)
        scheduler.step()

        # Validation on 2024
        df_val = predict_full_split(model, bundle, split='val', mode='honest')
        val_rmses = []
        for h in EVAL_HORIZONS:
            m = pd.merge(val_df[['Date', 'Estacion', f'Target_{h}h']], df_val[['Date', 'Estacion', f'pred_h{h}']], on=['Date', 'Estacion'])
            ok = m[f'Target_{h}h'].notna()
            h_rmse = np.sqrt(np.mean((m.loc[ok, f'Target_{h}h'] - m.loc[ok, f'pred_h{h}']) ** 2))
            val_rmses.append(h_rmse)
        mean_val_rmse = np.mean(val_rmses)

        epoch_time = time.time() - t0
        log_print(
            f"Epoch {epoch:2d}/{num_epochs:2d} | Train RMSE: {train_rmse:5.2f} | "
            f"Val RMSE (avg 6h): {mean_val_rmse:5.3f} (h1:{val_rmses[0]:.2f}, h4:{val_rmses[1]:.2f}, h24:{val_rmses[4]:.2f}, h48:{val_rmses[5]:.2f}) | "
            f"Time: {epoch_time:4.1f}s"
        )

        if mean_val_rmse < best_val_rmse:
            best_val_rmse = mean_val_rmse
            torch.save(model.state_dict(), best_ckpt_path)
            log_print(f"  [*] Best GRU model saved (val RMSE = {best_val_rmse:.4f})")

    model.load_state_dict(torch.load(best_ckpt_path))
    log_print(f"Loaded best GRU checkpoint with Val RMSE: {best_val_rmse:.4f}")

    df_val_gru = predict_full_split(model, bundle, split='val', mode='honest')
    df_test_gru = predict_full_split(model, bundle, split='test', mode='honest')
    res_gru = evaluate_and_save_variant(
        'gru_honest', df_val_gru, df_test_gru, val_df, test_df,
        notes="GRU-TiDE honest (2-layer GRU encoder + attention + multi-horizon decoder)"
    )

    return model, df_val_gru, df_test_gru, res_gru

def build_ensemble(
    df_val_tide, df_test_tide,
    df_val_q, df_test_q,
    df_val_gru, df_test_gru,
    val_df, test_df,
    mode='honest'
):
    log_print(f"\n>>> BUILDING DEEP ENSEMBLE ({mode.upper()})")
    # Convex blend weights: 50% TiDE-MSE + 30% TiDE-Quantile (q50) + 20% GRU
    w_tide, w_q, w_gru = 0.50, 0.30, 0.20

    df_val_ens = df_val_tide[['Date', 'Estacion']].copy()
    df_test_ens = df_test_tide[['Date', 'Estacion']].copy()

    for h in EVAL_HORIZONS:
        df_val_ens[f'pred_h{h}'] = (
            w_tide * df_val_tide[f'pred_h{h}'] +
            w_q    * df_val_q[f'pred_h{h}'] +
            w_gru  * df_val_gru[f'pred_h{h}']
        )
        df_test_ens[f'pred_h{h}'] = (
            w_tide * df_test_tide[f'pred_h{h}'] +
            w_q    * df_test_q[f'pred_h{h}'] +
            w_gru  * df_test_gru[f'pred_h{h}']
        )
        # Pass through calibrated q10 and q90 from quantile model
        df_val_ens[f'q10_h{h}'] = df_val_q[f'q10_h{h}']
        df_val_ens[f'q90_h{h}'] = df_val_q[f'q90_h{h}']
        df_test_ens[f'q10_h{h}'] = df_test_q[f'q10_h{h}']
        df_test_ens[f'q90_h{h}'] = df_test_q[f'q90_h{h}']

    var_name = f'ensemble_{mode}'
    notes = f"Deep Ensemble 50% TiDE-MSE + 30% TiDE-Quantile + 20% GRU-TiDE ({mode})"
    res_ens = evaluate_and_save_variant(
        var_name, df_val_ens, df_test_ens, val_df, test_df, notes=notes
    )
    return res_ens

def main():
    start_time = time.time()
    log_print("=================================================================")
    log_print("STARTING EXPERIMENT: noche_deep (Multi-Horizon Deep Learning O3)")
    log_print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    log_print("=================================================================")

    # 1. Prepare panel and split for official evaluation
    panel = load_panel(MODEL_STATIONS)
    panel = add_targets(panel)
    tr_df, val_df, test_df = split(panel)
    log_print(f"Panel split: train={len(tr_df)}, val={len(val_df)}, test={len(test_df)}")

    # 2. Build MultiHorizonDataBundle
    bundle = MultiHorizonDataBundle(seq_len=96, future_len=48)

    # 3. Train Model 1: TiDE MSE (Honest & PP)
    m1, val_tide_h, test_tide_h, val_tide_pp, test_tide_pp, res_tide_h, res_tide_pp = train_tide_model(
        bundle, val_df, test_df, num_epochs=12, batch_size=512, lr=1e-3
    )

    # 4. Train Model 2: TiDE Quantile (Honest & PP)
    m2, val_q_h, test_q_h, val_q_pp, test_q_pp, res_q_h, res_q_pp = train_tide_quantile_model(
        bundle, val_df, test_df, num_epochs=12, batch_size=512, lr=1e-3
    )

    # 5. Train Model 3: GRU-TiDE (Honest)
    m3, val_gru_h, test_gru_h, res_gru_h = train_gru_model(
        bundle, val_df, test_df, num_epochs=10, batch_size=512, lr=1e-3
    )

    # 6. Deep Ensemble (Honest)
    res_ens_h = build_ensemble(
        val_tide_h, test_tide_h,
        val_q_h, test_q_h,
        val_gru_h, test_gru_h,
        val_df, test_df,
        mode='honest'
    )

    # 7. Deep Ensemble (PP - Upper bound)
    # Using GRU with honest meteo as fallback for PP ensemble
    res_ens_pp = build_ensemble(
        val_tide_pp, test_tide_pp,
        val_q_pp, test_q_pp,
        val_gru_h, test_gru_h,
        val_df, test_df,
        mode='pp'
    )

    elapsed = time.time() - start_time
    log_print(f"\nALL RUNS COMPLETED IN {elapsed/60:.2f} MINUTES ({elapsed:.1f}s)!")

if __name__ == '__main__':
    main()
