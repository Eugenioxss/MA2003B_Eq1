"""
experiments/planE_deep/03b_sequential_gru.py
--------------------------------------------
Task 3b: Sequential Multi-Horizon Model (GRU with Missingness Masks).
- Past window: L = 72 hours (up to t-1)
- Variables (10): [O3, NOX, NO2, TOUT, SR, RH, U_Wind, V_Wind, PM10, PM2.5]
- Channels: 10 normalized variables + 10 binary missingness masks + cyclic time (hour_sin/cos, month_sin/cos)
  Total: 24 input channels per time step!
- Station Embedding (13 stations)
- Target-time calendar embeddings for each horizon [1, 4, 8, 12, 24, 48]
- Multi-output decoder predicting all 6 horizons simultaneously
- Masked loss (ignores missing target values)
- Evaluates both MSE model and Pinball quantile model (0.1, 0.5, 0.9)
- Early stopping on Val 2024
- Evaluates on 2025 test set using evaluate_test()
- Logs to results_registry.csv and saves test predictions
"""
import sys, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import (
    load_panel, add_targets, split, evaluate_test, log_result,
    HORIZONS, MODEL_STATIONS, SEED
)

OUTPUT_DIR = os.path.join(ROOT, 'experiments', 'planE_deep')
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(3)

SEQ_VARS = ['O3', 'NOX', 'NO2', 'TOUT', 'SR', 'RH', 'U_Wind', 'V_Wind', 'PM10', 'PM2.5']
SEQ_LEN = 72  # 3 days of hourly history up to t-1

# ---------------------------------------------------------------------------
# Fast Sliced Sequence Dataset
# ---------------------------------------------------------------------------
class FastSequentialDataset(Dataset):
    def __init__(self, station_tensors, sample_indices):
        """
        station_tensors: dict of {st_idx: torch.Tensor of shape (T_station, num_channels)}
        sample_indices: list of tuples (st_idx, t_idx, target_vals, target_cal_feats)
        """
        self.station_tensors = station_tensors
        self.samples = sample_indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        st_idx, t_idx, targets, target_cal, _ = self.samples[idx]
        # Past window ends at t_idx - 1 (length SEQ_LEN)
        seq = self.station_tensors[st_idx][t_idx - SEQ_LEN : t_idx]
        return seq, st_idx, targets, target_cal

# ---------------------------------------------------------------------------
# Sequential GRU Architecture with Missingness Channels
# ---------------------------------------------------------------------------
class MultiHorizonGRU(nn.Module):
    def __init__(self, in_channels=24, hidden_dim=128, num_stations=13, station_emb_dim=16, num_outputs=6, dropout=0.2):
        super().__init__()
        self.num_outputs = num_outputs
        self.station_emb = nn.Embedding(num_stations, station_emb_dim)
        
        # Input projection
        self.in_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # GRU Encoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Temporal attention pooling
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Decoders for each horizon
        # Each decoder takes pooled representation + station embedding + target time calendar features (4 dims)
        decoder_in_dim = hidden_dim + station_emb_dim + 4
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(decoder_in_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 1 if num_outputs == 6 else 3)  # 1 for MSE, 3 for Pinball
            ) for _ in range(6)
        ])

    def forward(self, seq, st_idx, target_cal):
        # seq: (B, L, C)
        # st_idx: (B,)
        # target_cal: (B, 6, 4)
        B, L, _ = seq.shape
        x = self.in_proj(seq)
        gru_out, _ = self.gru(x)  # (B, L, H)
        
        # Attention weights
        attn_scores = torch.softmax(self.attn(gru_out), dim=1)  # (B, L, 1)
        pooled = torch.sum(gru_out * attn_scores, dim=1)  # (B, H)
        
        st_e = self.station_emb(st_idx)  # (B, S_emb)
        base_rep = torch.cat([pooled, st_e], dim=-1)  # (B, H + S_emb)
        
        outputs = []
        for h_i, dec in enumerate(self.decoders):
            cal_h = target_cal[:, h_i, :]  # (B, 4)
            inp_h = torch.cat([base_rep, cal_h], dim=-1)
            out_h = dec(inp_h)  # (B, 1) or (B, 3)
            outputs.append(out_h)
            
        if self.num_outputs == 6:
            # (B, 6)
            return torch.cat(outputs, dim=-1)
        else:
            # (B, 6, 3)
            return torch.stack(outputs, dim=1)

def masked_mse_loss(preds, targets):
    valid = torch.isfinite(targets)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    diff = preds[valid] - targets[valid]
    return torch.mean(diff ** 2)

def masked_pinball_loss(preds, targets, quantiles=[0.1, 0.5, 0.9]):
    valid = torch.isfinite(targets)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    total_loss = 0.0
    for q_idx, q in enumerate(quantiles):
        p_q = preds[:, :, q_idx]
        diff = targets[valid] - p_q[valid]
        loss_q = torch.maximum(q * diff, (q - 1.0) * diff)
        total_loss += torch.mean(loss_q)
    return total_loss

def run():
    print("=== [Step 3b] Sequential GRU with Missingness Channels ===")
    t0 = time.time()
    
    panel = load_panel(stations=MODEL_STATIONS)
    panel = add_targets(panel)
    panel = panel.sort_values(['Estacion', 'Date']).reset_index(drop=True)
    
    # Pre-calculate train statistics for normalization
    train_mask_raw = panel['Date'].dt.year <= 2023
    val_mask_raw = panel['Date'].dt.year == 2024
    test_mask_raw = panel['Date'].dt.year == 2025
    
    means = {}
    stds = {}
    for col in SEQ_VARS:
        tr_vals = panel.loc[train_mask_raw, col].values
        m = np.nanmean(tr_vals)
        s = np.nanstd(tr_vals)
        if s < 1e-5 or np.isnan(s): s = 1.0
        if np.isnan(m): m = 0.0
        means[col] = float(m)
        stds[col] = float(s)
        
    st_map = {s: i for i, s in enumerate(MODEL_STATIONS)}

    # Build continuous station tensors
    # For each station: 10 normalized vars + 10 missing masks + 4 cyclic calendar = 24 channels
    station_tensors = {}
    print("Building station tensor continuous sequences...")
    for est, grp in panel.groupby('Estacion'):
        st_i = st_map[est]
        grp = grp.sort_values('Date').reset_index(drop=True)
        N_g = len(grp)
        
        # Missing masks (1 if missing, 0 if present)
        masks = np.zeros((N_g, len(SEQ_VARS)), dtype=np.float32)
        norm_vars = np.zeros((N_g, len(SEQ_VARS)), dtype=np.float32)
        
        for c_i, col in enumerate(SEQ_VARS):
            raw = grp[col].values.astype(np.float32)
            masks[:, c_i] = np.isnan(raw).astype(np.float32)
            # Impute NaNs with linear interpolation + train mean fallback
            s_col = pd.Series(raw)
            interp = s_col.interpolate(method='linear', limit=6).bfill().ffill().fillna(means[col]).values
            norm_vars[:, c_i] = (interp - means[col]) / stds[col]
            
        # Calendar features for historical sequence
        dates = pd.to_datetime(grp['Date'])
        hour_sin = np.sin(2 * np.pi * dates.dt.hour.values / 24).astype(np.float32)
        hour_cos = np.cos(2 * np.pi * dates.dt.hour.values / 24).astype(np.float32)
        month_sin = np.sin(2 * np.pi * dates.dt.month.values / 12).astype(np.float32)
        month_cos = np.cos(2 * np.pi * dates.dt.month.values / 12).astype(np.float32)
        cal = np.stack([hour_sin, hour_cos, month_sin, month_cos], axis=1)
        
        full_tensor = np.concatenate([norm_vars, masks, cal], axis=1)  # (N_g, 24)
        station_tensors[st_i] = torch.from_numpy(full_tensor).float()

    # Build sample indices for Train, Val, Test
    print("Indexing sequences...")
    train_samples = []
    val_samples = []
    test_samples = []
    
    target_cols = [f'Target_{h}h' for h in HORIZONS]

    for est, grp in panel.groupby('Estacion'):
        st_i = st_map[est]
        grp = grp.sort_values('Date').reset_index(drop=True)
        dates = pd.to_datetime(grp['Date'])
        years = dates.dt.year.values
        targets = grp[target_cols].values.astype(np.float32)
        
        # For each horizon, precalculate target calendar features
        # target_cal shape: (N_g, 6, 4)
        t_cal = np.zeros((len(grp), 6, 4), dtype=np.float32)
        for h_i, h in enumerate(HORIZONS):
            target_dates = dates + pd.Timedelta(hours=h)
            t_cal[:, h_i, 0] = np.sin(2 * np.pi * target_dates.dt.hour.values / 24)
            t_cal[:, h_i, 1] = np.cos(2 * np.pi * target_dates.dt.hour.values / 24)
            t_cal[:, h_i, 2] = np.sin(2 * np.pi * target_dates.dt.month.values / 12)
            t_cal[:, h_i, 3] = np.cos(2 * np.pi * target_dates.dt.month.values / 12)

        # Iterate over valid indices where t >= SEQ_LEN
        grp_dates = grp['Date'].values
        for t_idx in range(SEQ_LEN, len(grp)):
            yr = years[t_idx]
            sample = (st_i, t_idx, targets[t_idx], t_cal[t_idx], grp_dates[t_idx])
            if yr <= 2023:
                train_samples.append(sample)
            elif yr == 2024:
                val_samples.append(sample)
            elif yr == 2025:
                test_samples.append(sample)

    print(f"Dataset indexed. Train: {len(train_samples):,}, Val: {len(val_samples):,}, Test: {len(test_samples):,}")

    train_ds = FastSequentialDataset(station_tensors, train_samples)
    val_ds = FastSequentialDataset(station_tensors, val_samples)
    test_ds = FastSequentialDataset(station_tensors, test_samples)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device} ({torch.cuda.get_device_name(0)})")

    # ----------------------------------------------------
    # Model: MultiHorizonGRU (MSE Loss)
    # ----------------------------------------------------
    print("\n--- Training Sequential MultiHorizonGRU (MSE Loss) ---")
    model_gru = MultiHorizonGRU(in_channels=24, hidden_dim=128, num_outputs=6).to(device)
    optimizer = torch.optim.AdamW(model_gru.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_gru_path = os.path.join(OUTPUT_DIR, 'best_gru_mse.pt')
    patience = 6
    patience_cnt = 0
    epochs = 25

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model_gru.train()
        train_loss = 0.0
        n_b = 0
        for b_seq, b_st, b_y, b_cal in train_loader:
            b_seq, b_st, b_y, b_cal = b_seq.to(device), b_st.to(device), b_y.to(device), b_cal.to(device)
            optimizer.zero_grad()
            preds = model_gru(b_seq, b_st, b_cal)
            loss = masked_mse_loss(preds, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_gru.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_b += 1
        train_loss /= n_b

        model_gru.eval()
        val_loss = 0.0
        val_b = 0
        with torch.no_grad():
            for b_seq, b_st, b_y, b_cal in val_loader:
                b_seq, b_st, b_y, b_cal = b_seq.to(device), b_st.to(device), b_y.to(device), b_cal.to(device)
                preds = model_gru(b_seq, b_st, b_cal)
                loss = masked_mse_loss(preds, b_y)
                val_loss += loss.item()
                val_b += 1
        val_loss /= val_b
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save(model_gru.state_dict(), best_gru_path)
        else:
            patience_cnt += 1

        lr_curr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d} ({time.time()-t_ep:.1f}s) | Train MSE: {train_loss:.2f} | Val MSE: {val_loss:.2f} | lr: {lr_curr:.1e} {'*' if improved else ''}")
        if patience_cnt >= patience:
            print(f"Early stopping reached after {epoch} epochs.")
            break

    # Evaluate best GRU on test
    print("\nEvaluating best GRU on 2025 test set...")
    model_gru.load_state_dict(torch.load(best_gru_path, weights_only=True))
    model_gru.eval()
    all_preds_gru = []
    with torch.no_grad():
        for b_seq, b_st, b_y, b_cal in test_loader:
            b_seq, b_st, b_cal = b_seq.to(device), b_st.to(device), b_cal.to(device)
            preds = model_gru(b_seq, b_st, b_cal)
            all_preds_gru.append(preds.cpu().numpy())
    preds_gru = np.concatenate(all_preds_gru, axis=0)  # (N_test, 6)

    # Reconstruct test dataframe matching order
    test_meta = [{'Date': s[4], 'Estacion': MODEL_STATIONS[s[0]]} for s in test_samples]
    df_preds_gru = pd.DataFrame(test_meta)

    # Merge with full test panel
    test_df = panel[panel['Date'].dt.year == 2025].copy()
    for idx_h, h in enumerate(HORIZONS):
        df_preds_gru[f'pred_gru_mse_{h}h'] = preds_gru[:, idx_h]

    eval_test = pd.merge(test_df, df_preds_gru, on=['Date', 'Estacion'], how='left')

    print("\n=== GRU Evaluation Results (Test 2025) ===")
    for h in HORIZONS:
        p_h = eval_test[f'pred_gru_mse_{h}h'].values
        m = evaluate_test(eval_test, p_h, h)
        log_result('planE_deep', 'nn_sequential_gru_mse', h, m, f"Sequential GRU 72h window (best val MSE={best_val_loss:.2f})")
        print(f"  GRU h={h:2d}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}, RMSE_peaks={m['RMSE_peaks']:.4f}")

    eval_test[['Date', 'Estacion'] + [f'pred_gru_mse_{h}h' for h in HORIZONS]].to_parquet(
        os.path.join(OUTPUT_DIR, 'preds_sequential_gru.parquet'), index=False
    )
    print(f"GRU predictions saved to {os.path.join(OUTPUT_DIR, 'preds_sequential_gru.parquet')}")

if __name__ == '__main__':
    run()
