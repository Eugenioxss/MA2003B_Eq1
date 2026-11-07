import os, sys, time, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, numpy as np, pandas as pd

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import MODEL_STATIONS, SEED, evaluate_test

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

# Lightweight FT-Transformer / Tabular Transformer with feature grouping
# Since 360 individual tokens would be O(360^2) attention, we project groups of features or use Gated MLP / TabNet
class TabGatedMLP(nn.Module):
    """Gated MLP / GLU network (similar to modern Transformer FFN blocks)"""
    def __init__(self, num_stations, in_dim, embed_dim=16, hr_embed_dim=8, hidden_dim=384, num_layers=4, dropout=0.05):
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
            # SwiGLU: SiLU(gate) * val
            gated = F.silu(layer['fc_gate'](norm_h)) * layer['fc_val'](norm_h)
            out = layer['drop'](layer['fc_out'](gated))
            h = h + out
        h = self.out_norm(h)
        return self.out_head(h)

torch.manual_seed(42)
model = TabGatedMLP(len(MODEL_STATIONS), X_tr_proc.shape[1], hidden_dim=384, num_layers=3, dropout=0.05).to(device)
criterion = nn.HuberLoss(delta=1.5)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 10
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

batch_size = 2048
n_train = len(y_tr)
n_batches = (n_train + batch_size - 1) // batch_size

print(f"Testing TabGatedMLP on h=1...")
for ep in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(n_train, device=device)
    loss_acc = 0.0
    t0 = time.time()
    for b in range(n_batches):
        idx = perm[b*batch_size : (b+1)*batch_size]
        optimizer.zero_grad()
        pred = model(tr_st[idx], tr_hr[idx], X_tr_proc[idx])
        loss = criterion(pred, y_tr[idx])
        loss.backward()
        optimizer.step()
        loss_acc += loss.item() * len(idx)
    scheduler.step()
    
    model.eval()
    with torch.no_grad():
        vp = model(va_st, va_hr, X_va_proc)
        tp = model(te_st, te_hr, X_te_proc)
        v_rmse = torch.sqrt(torch.mean((vp - y_va)**2)).item()
        t_rmse = torch.sqrt(torch.mean((tp - y_te)**2)).item()
    print(f"  Ep {ep:02d} ({time.time()-t0:.1f}s) | Val: {v_rmse:.4f} | Test: {t_rmse:.4f}")
