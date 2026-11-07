import os, sys, time, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, numpy as np, pandas as pd

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import MODEL_STATIONS, evaluate_test

device = torch.device('cuda')
h = 1
df = pd.read_parquet(f'experiments/noche/v3_base/cache/features_h{h}.parquet')

t0_cols = [c for c in df.columns if c.startswith('meteo_t0_')]
honest_cols = [c for c in df.columns if c.startswith('honest_meteo_')]
exclude = {'Date', 'Estacion', 'split', 'Target', f'Target_{h}h'} | set(t0_cols) | set(honest_cols) | set([c for c in df.columns if c.startswith('pp_meteo_')])
num_cols = [c for c in df.columns if c not in exclude] + t0_cols + honest_cols

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

st_map = {s: i for i, s in enumerate(MODEL_STATIONS)}
tr_st = torch.tensor([st_map[s] for s in tr_df['Estacion']], dtype=torch.long, device=device)
va_st = torch.tensor([st_map[s] for s in va_df['Estacion']], dtype=torch.long, device=device)
te_st = torch.tensor([st_map[s] for s in te_df['Estacion']], dtype=torch.long, device=device)

tr_hr = torch.tensor(tr_df['Date'].dt.hour.values, dtype=torch.long, device=device)
va_hr = torch.tensor(va_df['Date'].dt.hour.values, dtype=torch.long, device=device)
te_hr = torch.tensor(te_df['Date'].dt.hour.values, dtype=torch.long, device=device)

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

class MonotoneQuantileHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc_pred = nn.Linear(in_features, 1)
        self.fc_d10 = nn.Linear(in_features, 1)
        self.fc_d90 = nn.Linear(in_features, 1)

    def forward(self, h):
        pred = self.fc_pred(h)
        d10 = F.softplus(self.fc_d10(h))
        d90 = F.softplus(self.fc_d90(h))
        q10 = pred - d10
        q90 = pred + d90
        return pred, q10, q90

class TabResNetQuantile(nn.Module):
    def __init__(self, num_stations, num_features, embed_dim=16, hr_embed_dim=8, hidden_dim=256, num_blocks=3, dropout=0.05):
        super().__init__()
        self.st_emb = nn.Embedding(num_stations, embed_dim)
        self.hr_emb = nn.Embedding(24, hr_embed_dim)
        self.input_layer = nn.Linear(num_features + embed_dim + hr_embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([TabResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head_bn = nn.BatchNorm1d(hidden_dim)
        self.head_act = nn.SiLU()
        self.qhead = MonotoneQuantileHead(hidden_dim)

    def forward(self, st_idx, hr_idx, x_num):
        emb_st = self.st_emb(st_idx)
        emb_hr = self.hr_emb(hr_idx)
        x = torch.cat([x_num, emb_st, emb_hr], dim=1)
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.head_act(self.head_bn(h))
        return self.qhead(h)

def pinball_loss(y_pred, y_true, q):
    err = y_true - y_pred
    return torch.mean(torch.max(q * err, (q - 1.0) * err))

torch.manual_seed(42)
model = TabResNetQuantile(len(MODEL_STATIONS), X_tr_proc.shape[1], embed_dim=16, hr_embed_dim=8, hidden_dim=256, num_blocks=3, dropout=0.05).to(device)
huber = nn.HuberLoss(delta=1.5)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 12
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
batch_size = 2048
n_train = len(y_tr)
n_batches = (n_train + batch_size - 1) // batch_size

print(f"Training TabResNetQuantile on h=1...")
best_val = 999.0
best_te = 999.0

for ep in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(n_train, device=device)
    loss_acc = 0.0
    for b in range(n_batches):
        idx = perm[b*batch_size : (b+1)*batch_size]
        optimizer.zero_grad()
        pred, q10, q90 = model(tr_st[idx], tr_hr[idx], X_tr_proc[idx])
        l_pred = huber(pred, y_tr[idx])
        l_q10 = pinball_loss(q10, y_tr[idx], 0.10)
        l_q90 = pinball_loss(q90, y_tr[idx], 0.90)
        loss = l_pred + 0.5 * (l_q10 + l_q90)
        loss.backward()
        optimizer.step()
        loss_acc += loss.item() * len(idx)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        vp, v10, v90 = model(va_st, va_hr, X_va_proc)
        tp, t10, t90 = model(te_st, te_hr, X_te_proc)
        v_rmse = torch.sqrt(torch.mean((vp - y_va)**2)).item()
        t_rmse = torch.sqrt(torch.mean((tp - y_te)**2)).item()

        # Coverage in test of [q10, q90] (target ~80%)
        cov_test = torch.mean(((y_te >= t10) & (y_te <= t90)).float()).item()

    if v_rmse < best_val:
        best_val = v_rmse
        best_te = t_rmse
    print(f"  Ep {ep:02d} | Val RMSE: {v_rmse:.4f} | Test RMSE: {t_rmse:.4f} | 80% CI Test Cov: {cov_test*100:.1f}% | Best Val: {best_val:.4f}")

# Final evaluate_test
model.eval()
with torch.no_grad():
    tp, t10, t90 = model(te_st, te_hr, X_te_proc)
    te_pred_np = tp.cpu().numpy().ravel()
m = evaluate_test(te_df, te_pred_np, h=1)
print("\nFinal Test Metrics with Quantile Regularization:")
for k in ['RMSE', 'R2', 'RMSE_peaks', 'bias_peaks', 'RMSE_day']:
    print(f"  {k}: {m[k]:.4f}")
