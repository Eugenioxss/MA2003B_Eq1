"""
experiments/noche2/deep2/dataset.py
===================================
Pipeline de datos y preprocesamiento robusto para modelos neuronales tabulares (TabResNet).
- Limpieza rigurosa de códigos centinela (< -500 o > 9000).
- Inyección estocástica de ruido numérico de pronóstico (NWP) en train basada en val 2024.
- Normalización estadística estricta fit en train (2020-2023) con clipping [-10, 10].
- Generación de máscaras binarias de faltantes para columnas con datos incompletos.
- Embeddings discretos para Estacion (13 clases) y Hora (24 clases).
"""
import os, sys
import numpy as np
import pandas as pd
import torch

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import MODEL_STATIONS, HORIZONS, SEED

CACHE_DIR = os.path.join(ROOT, 'experiments', 'noche', 'v3_base', 'cache')

def inject_meteo_noise(train_df: pd.DataFrame, val_df: pd.DataFrame, honest_cols: list[str], seed: int = 42) -> pd.DataFrame:
    """
    Agrega a train_df en las columnas honest_meteo ruido empírico muestreado
    de (val[honest_meteo] - val[pp_meteo]) en 2024, condicionado por hora del día.
    """
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

        # Acotar físicamente
        if 'rh' in h_col or 'cloud' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, 100.0)
        elif 'rad' in h_col or 'ws' in h_col or 'precip' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, None)

    # Recalcular derivadas si aplica
    if 'honest_meteo_target_temp' in train_noisy.columns and 'meteo_t0_temp' in train_noisy.columns:
        train_noisy['honest_meteo_delta_temp'] = train_noisy['honest_meteo_target_temp'] - train_noisy['meteo_t0_temp']
    if 'honest_meteo_target_rh' in train_noisy.columns and 'meteo_t0_rh' in train_noisy.columns:
        train_noisy['honest_meteo_delta_rh'] = train_noisy['honest_meteo_target_rh'] - train_noisy['meteo_t0_rh']
    if 'honest_meteo_target_temp' in train_noisy.columns and 'honest_meteo_target_rad' in train_noisy.columns:
        train_noisy['honest_meteo_target_temp_x_rad'] = train_noisy['honest_meteo_target_temp'] * train_noisy['honest_meteo_target_rad']

    return train_noisy


class TabularDatasetManager:
    def __init__(self, h: int, device: torch.device, seed: int = SEED):
        self.h = h
        self.device = device
        self.seed = seed
        self._load_and_process()

    def _load_and_process(self):
        fpath = os.path.join(CACHE_DIR, f'features_h{self.h}.parquet')
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Cache no encontrado: {fpath}")

        df = pd.read_parquet(fpath)

        # Identificar columnas
        t0_cols = [c for c in df.columns if c.startswith('meteo_t0_')]
        honest_cols = [c for c in df.columns if c.startswith('honest_meteo_')]
        pp_cols = [c for c in df.columns if c.startswith('pp_meteo_')]
        exclude = {'Date', 'Estacion', 'split', 'Target', f'Target_{self.h}h'} | set(t0_cols) | set(honest_cols) | set(pp_cols)
        chem_cols = [c for c in df.columns if c not in exclude]
        self.num_cols = chem_cols + t0_cols + honest_cols

        # Limpiar códigos centinela (< -500 o > 9000)
        for c in self.num_cols:
            bad = (df[c] < -500) | (df[c] > 9000)
            if bad.any():
                df.loc[bad, c] = np.nan

        # Filtrar splits con target válido
        tr_mask = (df['split'] == 'train') & df['Target'].notna()
        va_mask = (df['split'] == 'val') & df['Target'].notna()
        te_mask = (df['split'] == 'test') & df['Target'].notna()

        self.tr_df = df[tr_mask].copy().reset_index(drop=True)
        self.va_df = df[va_mask].copy().reset_index(drop=True)
        self.te_df = df[te_mask].copy().reset_index(drop=True)

        # Inyectar ruido de pronóstico numérico en train
        self.tr_df = inject_meteo_noise(self.tr_df, self.va_df, honest_cols, seed=self.seed)

        # Embeddings: Estacion (0..12) y Hora (0..23)
        st_map = {s: i for i, s in enumerate(MODEL_STATIONS)}
        self.tr_st = torch.tensor([st_map[s] for s in self.tr_df['Estacion']], dtype=torch.long, device=self.device)
        self.va_st = torch.tensor([st_map[s] for s in self.va_df['Estacion']], dtype=torch.long, device=self.device)
        self.te_st = torch.tensor([st_map[s] for s in self.te_df['Estacion']], dtype=torch.long, device=self.device)

        self.tr_hr = torch.tensor(self.tr_df['Date'].dt.hour.values, dtype=torch.long, device=self.device)
        self.va_hr = torch.tensor(self.va_df['Date'].dt.hour.values, dtype=torch.long, device=self.device)
        self.te_hr = torch.tensor(self.te_df['Date'].dt.hour.values, dtype=torch.long, device=self.device)

        # Targets
        y_tr_vals = self.tr_df['Target'].values.astype(np.float32)
        self.y_tr = torch.tensor(y_tr_vals, dtype=torch.float32, device=self.device).unsqueeze(1)
        self.y_va = torch.tensor(self.va_df['Target'].values, dtype=torch.float32, device=self.device).unsqueeze(1)
        self.y_te = torch.tensor(self.te_df['Target'].values, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Pesos opcionales de picos w = 1 + (y / p90)^2
        p90 = np.percentile(y_tr_vals, 90.0)
        self.weights_tr = torch.tensor(1.0 + (y_tr_vals / p90)**2, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Estadísticas de train para estandarización
        X_tr_raw = self.tr_df[self.num_cols].values.astype(np.float32)
        X_va_raw = self.va_df[self.num_cols].values.astype(np.float32)
        X_te_raw = self.te_df[self.num_cols].values.astype(np.float32)

        self.means = np.nanmean(X_tr_raw, axis=0)
        self.stds = np.nanstd(X_tr_raw, axis=0)
        self.stds[self.stds < 1e-4] = 1.0

        # Columnas con faltantes en train -> máscaras
        null_in_tr = np.isnan(X_tr_raw).any(axis=0)
        self.null_col_indices = np.where(null_in_tr)[0]

        def transform(X_raw):
            mask = np.isnan(X_raw[:, self.null_col_indices]).astype(np.float32)
            X_norm = (X_raw - self.means) / self.stds
            X_norm = np.nan_to_num(X_norm, nan=0.0)
            X_norm = np.clip(X_norm, -10.0, 10.0)
            return np.hstack([X_norm, mask])

        self.X_tr = torch.tensor(transform(X_tr_raw), dtype=torch.float32, device=self.device)
        self.X_va = torch.tensor(transform(X_va_raw), dtype=torch.float32, device=self.device)
        self.X_te = torch.tensor(transform(X_te_raw), dtype=torch.float32, device=self.device)

        self.input_dim = self.X_tr.shape[1]
