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
print(f"Using device: {device}")

# 1. Load data
h = 1
fpath = os.path.join(ROOT, f"experiments/noche/v3_base/cache/features_h{h}.parquet")
print(f"Loading {fpath}...")
df = pd.read_parquet(fpath)

# Clean sentinels (< -500 or > 9000)
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

# Separate splits
tr_mask = (df['split'] == 'train') & df['Target'].notna()
va_mask = (df['split'] == 'val') & df['Target'].notna()
te_mask = (df['split'] == 'test') & df['Target'].notna()

tr_df = df[tr_mask].copy().reset_index(drop=True)
va_df = df[va_mask].copy().reset_index(drop=True)
te_df = df[te_mask].copy().reset_index(drop=True)

# Station mapping
st_map = {s: i for i, s in enumerate(MODEL_STATIONS)}
tr_st = torch.tensor([st_map[s] for s in tr_df['Estacion']], dtype=torch.long, device=device)
va_st = torch.tensor([st_map[s] for s in va_df['Estacion']], dtype=torch.long, device=device)
te_st = torch.tensor([st_map[s] for s in te_df['Estacion']], dtype=torch.long, device=device)

# Compute train mean and std
X_tr_raw = tr_df[num_cols].values.astype(np.float32)
X_va_raw = va_df[num_cols].values.astype(np.float32)
X_te_raw = te_df[num_cols].values.astype(np.float32)

y_tr = torch.tensor(tr_df['Target'].values, dtype=torch.float32, device=device).unsqueeze(1)
y_va = torch.tensor(va_df['Target'].values, dtype=torch.float32, device=device).unsqueeze(1)
y_te = torch.tensor(te_df['Target'].values, dtype=torch.float32, device=device).unsqueeze(1)

means = np.nanmean(X_tr_raw, axis=0)
stds = np.nanstd(X_tr_raw, axis=0)
stds[stds < 1e-4] = 1.0

# Identify columns with missingness in train
null_in_tr = np.isnan(X_tr_raw).any(axis=0)
null_col_indices = np.where(null_in_tr)[0]
print(f"Num continuous cols: {len(num_cols)}, cols with missingness: {len(null_col_indices)}")

def preprocess(X_raw):
    mask = np.isnan(X_raw[:, null_col_indices]).astype(np.float32)
    # Standardize
    X_norm = (X_raw - means) / stds
    X_norm = np.nan_to_num(X_norm, nan=0.0)
    X_norm = np.clip(X_norm, -10.0, 10.0)
    # Concatenate features and mask
    return np.hstack([X_norm, mask])

X_tr_proc = torch.tensor(preprocess(X_tr_raw), dtype=torch.float32, device=device)
X_va_proc = torch.tensor(preprocess(X_va_raw), dtype=torch.float32, device=device)
X_te_proc = torch.tensor(preprocess(X_te_raw), dtype=torch.float32, device=device)

print(f"Processed input dim: {X_tr_proc.shape[1]}")

# Architecture
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
        res = x
        out = self.drop1(self.act(self.fc1(self.bn1(x))))
        out = self.drop2(self.act(self.fc2(self.bn2(out))))
        return res + out

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
        h = self.head_act(self.head_bn(h))
        return self.head(h)

model = TabResNet(len(MODEL_STATIONS), X_tr_proc.shape[1], embed_dim=16, hidden_dim=256, num_blocks=3, dropout=0.05).to(device)
criterion = nn.HuberLoss(delta=1.5)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 15
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

# Training loop
batch_size = 2048
n_train = len(y_tr)
n_batches = (n_train + batch_size - 1) // batch_size

print(f"Starting training with batch_size={batch_size}, {n_batches} batches/epoch...")
best_val = 999.0
best_test = 999.0
for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(n_train, device=device)
    loss_acc = 0.0
    t0 = time.time()
    for b in range(n_batches):
        idx = perm[b*batch_size : (b+1)*batch_size]
        optimizer.zero_grad()
        pred = model(tr_st[idx], X_tr_proc[idx])
        loss = criterion(pred, y_tr[idx])
        loss.backward()
        optimizer.step()
        loss_acc += loss.item() * len(idx)
    
    scheduler.step()
    train_loss = loss_acc / n_train
    ep_time = time.time() - t0
    
    # Evaluate Val & Test
    model.eval()
    with torch.no_grad():
        va_pred = model(va_st, X_va_proc)
        val_rmse = torch.sqrt(torch.mean((va_pred - y_va) ** 2)).item()
        te_pred = model(te_st, X_te_proc)
        test_rmse = torch.sqrt(torch.mean((te_pred - y_te) ** 2)).item()
    
    if val_rmse < best_val:
        best_val = val_rmse
        best_test = test_rmse
    
    print(f"Epoch {epoch:02d} ({ep_time:.1f}s) | Train Huber: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Test RMSE: {test_rmse:.4f} | Best Val: {best_val:.4f}")

# Official evaluate_test check
model.eval()
with torch.no_grad():
    te_pred = model(te_st, X_te_proc).cpu().numpy().ravel()
m_te = evaluate_test(te_df, te_pred, h=1)
print("\nFinal Test Metrics with evaluate_test:")
for k, v in m_te.items():
    print(f"  {k}: {v}")
