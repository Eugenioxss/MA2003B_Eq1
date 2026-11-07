import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import MODEL_STATIONS, SEED, evaluate_test

torch.set_num_threads(3)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inject_meteo_noise(train_df: pd.DataFrame, val_df: pd.DataFrame, honest_cols: list[str], seed: int = 42) -> pd.DataFrame:
    train_noisy = train_df.copy()
    rng = np.random.default_rng(seed)
    val_hour = val_df['Date'].dt.hour.values
    tr_hour = train_noisy['Date'].dt.hour.values

    for h_col in honest_cols:
        suffix = h_col.replace('honest_meteo_', '')
        pp_col = f'pp_meteo_{suffix}'
        if pp_col not in val_df.columns:
            continue
        val_diff = (val_df[h_col] - val_df[pp_col]).values
        valid_mask = np.isfinite(val_diff)
        sampled_noise = np.zeros(len(train_noisy), dtype='float32')
        for hr in range(24):
            hr_val_mask = valid_mask & (val_hour == hr)
            hr_tr_mask = (tr_hour == hr)
            n_tr = int(hr_tr_mask.sum())
            if n_tr == 0:
                continue
            pool = val_diff[hr_val_mask]
            if len(pool) > 0:
                sampled_noise[hr_tr_mask] = rng.choice(pool, size=n_tr, replace=True)
            else:
                sampled_noise[hr_tr_mask] = 0.0

        train_noisy[h_col] = (train_noisy[h_col] + sampled_noise).astype('float32')
        if 'rh' in h_col or 'cloud' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, 100.0)
        elif 'rad' in h_col or 'ws' in h_col or 'precip' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, None)

    if 'honest_meteo_target_temp' in train_noisy.columns and 'meteo_t0_temp' in train_noisy.columns:
        train_noisy['honest_meteo_delta_temp'] = train_noisy['honest_meteo_target_temp'] - train_noisy['meteo_t0_temp']
    if 'honest_meteo_target_rh' in train_noisy.columns and 'meteo_t0_rh' in train_noisy.columns:
        train_noisy['honest_meteo_delta_rh'] = train_noisy['honest_meteo_target_rh'] - train_noisy['meteo_t0_rh']
    if 'honest_meteo_target_temp' in train_noisy.columns and 'honest_meteo_target_rad' in train_noisy.columns:
        train_noisy['honest_meteo_target_temp_x_rad'] = train_noisy['honest_meteo_target_temp'] * train_noisy['honest_meteo_target_rad']

    return train_noisy

for h in [4, 8, 24, 48]:
    print(f"\n{'='*60}\nEvaluating TabResNet for h = {h}\n{'='*60}")
    fpath = os.path.join(ROOT, f"experiments/noche/v3_base/cache/features_h{h}.parquet")
    df = pd.read_parquet(fpath)

    t0_cols = [c for c in df.columns if c.startswith('meteo_t0_')]
    honest_cols = [c for c in df.columns if c.startswith('honest_meteo_')]
    pp_cols = [c for c in df.columns if c.startswith('pp_meteo_')]
    exclude = {'Date', 'Estacion', 'split', 'Target', f'Target_{h}h'} | set(t0_cols) | set(honest_cols) | set(pp_cols)
    chem_cols = [c for c in df.columns if c not in exclude]
    num_cols = chem_cols + t0_cols + honest_cols

    for c in num_cols:
        bad = (df[c] < -500) | (df[c] > 9000)
        if bad.any():
            df.loc[bad, c] = np.nan

    tr_mask = (df['split'] == 'train') & df['Target'].notna()
    va_mask = (df['split'] == 'val') & df['Target'].notna()
    te_mask = (df['split'] == 'test') & df['Target'].notna()

    tr_df = df[tr_mask].copy().reset_index(drop=True)
    va_df = df[va_mask].copy().reset_index(drop=True)
    te_df = df[te_mask].copy().reset_index(drop=True)

    # Noise injection
    tr_df = inject_meteo_noise(tr_df, va_df, honest_cols, seed=SEED)

    st_map = {s: i for i, s in enumerate(MODEL_STATIONS)}
    tr_st = torch.tensor([st_map[s] for s in tr_df['Estacion']], dtype=torch.long, device=device)
    va_st = torch.tensor([st_map[s] for s in va_df['Estacion']], dtype=torch.long, device=device)
    te_st = torch.tensor([st_map[s] for s in te_df['Estacion']], dtype=torch.long, device=device)

    X_tr_raw = tr_df[num_cols].values.astype(np.float32)
    X_va_raw = va_df[num_cols].values.astype(np.float32)
    X_te_raw = te_df[num_cols].values.astype(np.float32)

    y_tr = torch.tensor(tr_df['Target'].values, dtype=torch.float32, device=device).unsqueeze(1)
    y_va = torch.tensor(va_df['Target'].values, dtype=torch.float32, device=device).unsqueeze(1)
    y_te = torch.tensor(te_df['Target'].values, dtype=torch.float32, device=device).unsqueeze(1)

    means = np.nanmean(X_tr_raw, axis=0)
    stds = np.nanstd(X_tr_raw, axis=0)
    stds[stds < 1e-4] = 1.0

    null_in_tr = np.isnan(X_tr_raw).any(axis=0)
    null_col_indices = np.where(null_in_tr)[0]

    def preprocess(X_raw):
        mask = np.isnan(X_raw[:, null_col_indices]).astype(np.float32)
        X_norm = (X_raw - means) / stds
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        X_norm = np.clip(X_norm, -10.0, 10.0)
        return np.hstack([X_norm, mask])

    X_tr_proc = torch.tensor(preprocess(X_tr_raw), dtype=torch.float32, device=device)
    X_va_proc = torch.tensor(preprocess(X_va_raw), dtype=torch.float32, device=device)
    X_te_proc = torch.tensor(preprocess(X_te_raw), dtype=torch.float32, device=device)

    class TabResNetBlock(nn.Module):
        def __init__(self, dim, dropout=0.05):
            super().__init__()
            self.bn1 = nn.BatchNorm1d(dim)
            self.fc1 = nn.Linear(dim, dim)
            self.act = nn.SiLU()
            self.drop1 = nn.Dropout(dropout)
            self.bn2 = nn.BatchNorm1d(dim)
            self.fc2 = nn.Linear(dim, dim)
            self.drop2 = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop2(self.act(self.fc2(self.bn2(self.drop1(self.act(self.fc1(self.bn1(x))))))))

    class TabResNet(nn.Module):
        def __init__(self, num_stations, num_features, embed_dim=16, hidden_dim=256, num_blocks=3, dropout=0.05):
            super().__init__()
            self.st_emb = nn.Embedding(num_stations, embed_dim)
            self.input_layer = nn.Linear(num_features + embed_dim, hidden_dim)
            self.blocks = nn.ModuleList([TabResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
            self.head_bn = nn.BatchNorm1d(hidden_dim)
            self.head_act = nn.SiLU()
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, st_idx, x_num):
            emb = self.st_emb(st_idx)
            x = torch.cat([x_num, emb], dim=1)
            h = self.input_layer(x)
            for block in self.blocks:
                h = block(h)
            return self.head(self.head_act(self.head_bn(h)))

    model = TabResNet(len(MODEL_STATIONS), X_tr_proc.shape[1], embed_dim=16, hidden_dim=256, num_blocks=3, dropout=0.05).to(device)
    criterion = nn.HuberLoss(delta=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 12
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    batch_size = 2048
    n_train = len(y_tr)
    n_batches = (n_train + batch_size - 1) // batch_size

    best_val_rmse = 999.0
    best_va_preds = None
    best_te_preds = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        loss_acc = 0.0
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            optimizer.zero_grad()
            pred = model(tr_st[idx], X_tr_proc[idx])
            loss = criterion(pred, y_tr[idx])
            loss.backward()
            optimizer.step()
            loss_acc += loss.item() * len(idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(va_st, X_va_proc)
            val_rmse = torch.sqrt(torch.mean((va_pred - y_va) ** 2)).item()
            te_pred = model(te_st, X_te_proc)
            test_rmse = torch.sqrt(torch.mean((te_pred - y_te) ** 2)).item()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_va_preds = va_pred.cpu().numpy().ravel()
            best_te_preds = te_pred.cpu().numpy().ravel()

        print(f"  h={h:02d} Ep {epoch:02d} | Val RMSE: {val_rmse:.4f} | Test RMSE: {test_rmse:.4f} | Best Val: {best_val_rmse:.4f}")

    m_te = evaluate_test(te_df, best_te_preds, h=h)
    print(f"Final Test Metrics for h={h} (best val model):")
    print(f"  RMSE: {m_te['RMSE']:.4f} | R2: {m_te['R2']:.4f} | RMSE_peaks: {m_te['RMSE_peaks']:.2f} | bias_peaks: {m_te['bias_peaks']:.2f}")

    # Check correlation with D_honest_noise
    p_lgb = pd.read_parquet(os.path.join(ROOT, f"experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet"))
    lgb_te = p_lgb[p_lgb['split'] == 'test']['y_pred'].values
    lgb_res = te_df[f'Target_{h}h'].values - lgb_te
    nn_res = te_df[f'Target_{h}h'].values - best_te_preds
    corr = np.corrcoef(lgb_res, nn_res)[0, 1]
    print(f"  Correlation of residuals vs D_honest_noise: {corr:.4f}")

    # Test blend 50/50
    blend = 0.5 * lgb_te + 0.5 * best_te_preds
    blend_m = evaluate_test(te_df, blend, h=h)
    print(f"  Blend 50/50 RMSE: {blend_m['RMSE']:.4f} (LGBM alone: {np.sqrt(np.mean(lgb_res**2)):.4f}, NN alone: {m_te['RMSE']:.4f})")
