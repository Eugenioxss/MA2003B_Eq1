"""
experiments/planE_deep/03a_tabular_tide_mlp.py
---------------------------------------------
Task 3a: Multi-output Tabular Residual MLP / TiDE-like architecture.
- Inputs: 80 tabular v2 features + binary missingness masks (NaN -> 0 + mask)
- Feature standardization fit ONLY on train set (2020-2023)
- Station embedding (13 stations)
- Output: simultaneous prediction of all 6 horizons [1, 4, 8, 12, 24, 48]
- Masked loss (ignores missing target values during training & validation)
- Evaluates MSE model as well as Pinball (quantiles 0.1, 0.5, 0.9)
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
    load_panel, build_v2_features, split, evaluate_test, log_result,
    HORIZONS, MODEL_STATIONS, SEED
)

OUTPUT_DIR = os.path.join(ROOT, 'experiments', 'planE_deep')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(3)

# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, X, mask, station_idx, targets):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)
        self.station_idx = torch.as_tensor(station_idx, dtype=torch.long)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.station_idx[idx], self.targets[idx]

# ---------------------------------------------------------------------------
# Model Architecture: TiDE-like Tabular Residual Network
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.act(out + residual)
        return out

class TabularTiDE(nn.Module):
    def __init__(self, num_features, num_stations=13, station_emb_dim=16, hidden_dim=256, num_blocks=3, num_outputs=6, dropout=0.1):
        super().__init__()
        self.station_emb = nn.Embedding(num_stations, station_emb_dim)
        in_dim = num_features * 2 + station_emb_dim  # features + missing masks + station emb
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])
        self.head = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x, mask, station_idx):
        st_emb = self.station_emb(station_idx)
        inp = torch.cat([x, mask, st_emb], dim=-1)
        feat = self.input_proj(inp)
        for blk in self.blocks:
            feat = blk(feat)
        out = self.head(feat)
        return out

def masked_mse_loss(preds, targets):
    valid = torch.isfinite(targets)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    diff = preds[valid] - targets[valid]
    return torch.mean(diff ** 2)

def masked_pinball_loss(preds, targets, quantiles=[0.1, 0.5, 0.9]):
    # preds: (B, 6, 3), targets: (B, 6)
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
    print("=== [Step 3a] Tabular Residual MLP / TiDE Multi-Output Model ===")
    t0 = time.time()
    
    # 1. Load data and features
    panel = load_panel()
    df, features = build_v2_features(panel)
    train_df, val_df, test_df = split(df)

    # Station mapping
    st_map = {s: i for i, s in enumerate(MODEL_STATIONS)}
    
    # 2. Extract features, masks, and targets
    print("Pre-processing tabular features and missing masks...")
    X_tr_raw = train_df[features].values.astype(np.float32)
    X_va_raw = val_df[features].values.astype(np.float32)
    X_te_raw = test_df[features].values.astype(np.float32)

    M_tr = np.isnan(X_tr_raw).astype(np.float32)
    M_va = np.isnan(X_va_raw).astype(np.float32)
    M_te = np.isnan(X_te_raw).astype(np.float32)

    # Fit standardization statistics on train only (using nanmean / nanstd)
    feat_mean = np.nanmean(X_tr_raw, axis=0)
    feat_std = np.nanstd(X_tr_raw, axis=0)
    feat_std[feat_std < 1e-6] = 1.0
    feat_mean[np.isnan(feat_mean)] = 0.0

    # Impute NaNs with mean and standardize
    X_tr_norm = np.where(np.isnan(X_tr_raw), 0.0, (X_tr_raw - feat_mean) / feat_std)
    X_va_norm = np.where(np.isnan(X_va_raw), 0.0, (X_va_raw - feat_mean) / feat_std)
    X_te_norm = np.where(np.isnan(X_te_raw), 0.0, (X_te_raw - feat_mean) / feat_std)

    st_tr = train_df['Estacion'].map(st_map).values
    st_va = val_df['Estacion'].map(st_map).values
    st_te = test_df['Estacion'].map(st_map).values

    target_cols = [f'Target_{h}h' for h in HORIZONS]
    y_tr = train_df[target_cols].values.astype(np.float32)
    y_va = val_df[target_cols].values.astype(np.float32)
    y_te = test_df[target_cols].values.astype(np.float32)

    train_ds = TabularDataset(X_tr_norm, M_tr, st_tr, y_tr)
    val_ds = TabularDataset(X_va_norm, M_va, st_va, y_va)
    test_ds = TabularDataset(X_te_norm, M_te, st_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # ----------------------------------------------------
    # Model 1: MSE Optimization
    # ----------------------------------------------------
    print("\n--- Training Tabular TiDE (MSE Loss) ---")
    model_mse = TabularTiDE(num_features=len(features), num_outputs=6).to(device)
    optimizer = torch.optim.AdamW(model_mse.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_weights_path = os.path.join(OUTPUT_DIR, 'best_tabular_tide_mse.pt')
    patience = 8
    patience_cnt = 0

    epochs = 35
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model_mse.train()
        train_loss = 0.0
        n_batches = 0
        for bx, bm, bst, by in train_loader:
            bx, bm, bst, by = bx.to(device), bm.to(device), bst.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model_mse(bx, bm, bst)
            loss = masked_mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_mse.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # Validation
        model_mse.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for bx, bm, bst, by in val_loader:
                bx, bm, bst, by = bx.to(device), bm.to(device), bst.to(device), by.to(device)
                pred = model_mse(bx, bm, bst)
                loss = masked_mse_loss(pred, by)
                val_loss += loss.item()
                val_batches += 1
        val_loss /= val_batches
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save(model_mse.state_dict(), best_weights_path)
        else:
            patience_cnt += 1

        lr_curr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d} ({time.time()-t_ep:.1f}s) | Train MSE: {train_loss:.2f} | Val MSE: {val_loss:.2f} | lr: {lr_curr:.1e} {'*' if improved else ''}")
        if patience_cnt >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # Load best weights and predict on test
    print("\nEvaluating best MSE model on 2025 test set...")
    model_mse.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model_mse.eval()
    all_preds_mse = []
    with torch.no_grad():
        for bx, bm, bst, by in test_loader:
            bx, bm, bst = bx.to(device), bm.to(device), bst.to(device)
            pred = model_mse(bx, bm, bst)
            all_preds_mse.append(pred.cpu().numpy())
    preds_mse = np.concatenate(all_preds_mse, axis=0)  # (N_test, 6)

    # Save test predictions dataframe
    df_preds_mlp = test_df[['Date', 'Estacion']].copy()
    mlp_metrics = {}
    for idx_h, h in enumerate(HORIZONS):
        p_h = preds_mse[:, idx_h]
        df_preds_mlp[f'pred_tide_mse_{h}h'] = p_h
        m = evaluate_test(test_df, p_h, h)
        mlp_metrics[h] = m
        log_result('planE_deep', 'nn_tabular_tide_mse', h, m, f"Tabular TiDE MLP Multi-Output (best val MSE={best_val_loss:.2f})")
        print(f"  MSE h={h:2d}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}, RMSE_peaks={m['RMSE_peaks']:.4f}")

    # ----------------------------------------------------
    # Model 2: Pinball / Quantile Optimization (0.1, 0.5, 0.9)
    # ----------------------------------------------------
    print("\n--- Training Tabular TiDE with Quantile Loss (0.1, 0.5, 0.9) ---")
    model_pinball = TabularTiDE(num_features=len(features), num_outputs=6 * 3).to(device)
    opt_pin = torch.optim.AdamW(model_pinball.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_pin = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_pin, mode='min', factor=0.5, patience=2)

    best_pin_loss = float('inf')
    best_pin_path = os.path.join(OUTPUT_DIR, 'best_tabular_tide_pinball.pt')
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model_pinball.train()
        train_loss = 0.0
        n_batches = 0
        for bx, bm, bst, by in train_loader:
            bx, bm, bst, by = bx.to(device), bm.to(device), bst.to(device), by.to(device)
            opt_pin.zero_grad()
            pred = model_pinball(bx, bm, bst).reshape(-1, 6, 3)
            loss = masked_pinball_loss(pred, by, quantiles=[0.1, 0.5, 0.9])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_pinball.parameters(), 1.0)
            opt_pin.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        model_pinball.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for bx, bm, bst, by in val_loader:
                bx, bm, bst, by = bx.to(device), bm.to(device), bst.to(device), by.to(device)
                pred = model_pinball(bx, bm, bst).reshape(-1, 6, 3)
                loss = masked_pinball_loss(pred, by, quantiles=[0.1, 0.5, 0.9])
                val_loss += loss.item()
                val_batches += 1
        val_loss /= val_batches
        sched_pin.step(val_loss)

        improved = val_loss < best_pin_loss
        if improved:
            best_pin_loss = val_loss
            patience_cnt = 0
            torch.save(model_pinball.state_dict(), best_pin_path)
        else:
            patience_cnt += 1

        lr_curr = opt_pin.param_groups[0]['lr']
        print(f"Epoch {epoch:02d} ({time.time()-t_ep:.1f}s) | Train Pinball: {train_loss:.3f} | Val Pinball: {val_loss:.3f} | lr: {lr_curr:.1e} {'*' if improved else ''}")
        if patience_cnt >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # Load best pinball weights and evaluate
    print("\nEvaluating best Pinball model on 2025 test set...")
    model_pinball.load_state_dict(torch.load(best_pin_path, weights_only=True))
    model_pinball.eval()
    all_preds_pin = []
    with torch.no_grad():
        for bx, bm, bst, by in test_loader:
            bx, bm, bst = bx.to(device), bm.to(device), bst.to(device)
            pred = model_pinball(bx, bm, bst).reshape(-1, 6, 3)
            all_preds_pin.append(pred.cpu().numpy())
    preds_pin = np.concatenate(all_preds_pin, axis=0)  # (N_test, 6, 3)

    for idx_h, h in enumerate(HORIZONS):
        q10 = preds_pin[:, idx_h, 0]
        q50 = preds_pin[:, idx_h, 1]
        q90 = preds_pin[:, idx_h, 2]
        df_preds_mlp[f'pred_tide_q10_{h}h'] = q10
        df_preds_mlp[f'pred_tide_q50_{h}h'] = q50
        df_preds_mlp[f'pred_tide_q90_{h}h'] = q90

        y_true = test_df[f'Target_{h}h'].values
        valid = np.isfinite(y_true) & np.isfinite(q50)
        cov_80 = float(np.mean((y_true[valid] >= q10[valid]) & (y_true[valid] <= q90[valid])))

        m = evaluate_test(test_df, q50, h)
        m['coverage_80'] = cov_80
        log_result('planE_deep', 'nn_tabular_tide_pinball', h, m, f"Tabular TiDE Pinball q50 (80% cov={cov_80*100:.1f}%)")
        print(f"  Pinball h={h:2d}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}, Cov80%={cov_80*100:.1f}%")

    df_preds_mlp.to_parquet(os.path.join(OUTPUT_DIR, 'preds_tabular_tide.parquet'), index=False)
    print(f"All predictions saved to {os.path.join(OUTPUT_DIR, 'preds_tabular_tide.parquet')}")

if __name__ == '__main__':
    run()
