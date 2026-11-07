"""
experiments/noche2/deep2/dataset_nwp_refit.py
=============================================
Pipeline de datos para Refit 2020-2024 y Multi-Model NWP Features:
- Cruza features v3_base con features_nwp_h<h>.parquet (ECMWF, GFS, ICON, CAPE, MOS).
- Agrega variable indicadora is_nwp = (Date.year >= 2024).
- Limpia valores centinela (< -500 o > 9000).
- Soporta 4-fold cross-fitting en 2024 (Q1, Q2, Q3, Q4) y refit completo 2020-2024.
"""
import os, sys
import numpy as np
import pandas as pd
import torch

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import MODEL_STATIONS, SEED

CACHE_V3 = os.path.join(ROOT, 'experiments', 'noche', 'v3_base', 'cache')
CACHE_NWP = os.path.join(ROOT, 'experiments', 'noche2', 'nwp')

def load_combined_nwp_df(h: int) -> tuple[pd.DataFrame, list[str]]:
    f_v3 = os.path.join(CACHE_V3, f'features_h{h}.parquet')
    f_nwp = os.path.join(CACHE_NWP, f'features_nwp_h{h}.parquet')
    
    df_v3 = pd.read_parquet(f_v3)
    df_nwp = pd.read_parquet(f_nwp)
    
    # Merge on Date and Estacion
    df = pd.merge(df_v3, df_nwp, on=['Date', 'Estacion'], how='inner')
    
    # Columnas
    t0_cols = [c for c in df_v3.columns if c.startswith('meteo_t0_')]
    honest_cols = [c for c in df_v3.columns if c.startswith('honest_meteo_')]
    pp_cols = [c for c in df_v3.columns if c.startswith('pp_meteo_')]
    exclude = {'Date', 'Estacion', 'split', 'Target', f'Target_{h}h'} | set(t0_cols) | set(honest_cols) | set(pp_cols)
    chem_cols = [c for c in df_v3.columns if c not in exclude]
    nwp_cols = [c for c in df_nwp.columns if c not in ['Date', 'Estacion']]
    
    # Indicador is_nwp (0 para 2020-2023, 1 para 2024 y 2025)
    df['is_nwp'] = (df['Date'].dt.year >= 2024).astype('float32')
    
    num_cols = chem_cols + t0_cols + honest_cols + nwp_cols + ['is_nwp']
    
    # Limpiar codigos centinela (< -500 o > 9000)
    for c in num_cols:
        bad = (df[c] < -500) | (df[c] > 9000)
        if bad.any():
            df.loc[bad, c] = np.nan

    return df, num_cols


class NWPFoldManager:
    """Gestiona los tensores y preprocesamiento de un fold especifico para GPU."""
    def __init__(self, tr_df: pd.DataFrame, va_df: pd.DataFrame, te_df: pd.DataFrame, num_cols: list[str], device: torch.device):
        self.device = device
        st_map = {s: i for i, s in enumerate(MODEL_STATIONS)}

        # Embeddings
        self.tr_st = torch.tensor([st_map[s] for s in tr_df['Estacion']], dtype=torch.long, device=device)
        self.va_st = torch.tensor([st_map[s] for s in va_df['Estacion']], dtype=torch.long, device=device)
        self.te_st = torch.tensor([st_map[s] for s in te_df['Estacion']], dtype=torch.long, device=device) if te_df is not None else None

        self.tr_hr = torch.tensor(tr_df['Date'].dt.hour.values, dtype=torch.long, device=device)
        self.va_hr = torch.tensor(va_df['Date'].dt.hour.values, dtype=torch.long, device=device)
        self.te_hr = torch.tensor(te_df['Date'].dt.hour.values, dtype=torch.long, device=device) if te_df is not None else None

        # Targets
        y_tr_vals = tr_df['Target'].values.astype(np.float32)
        self.y_tr = torch.tensor(y_tr_vals, dtype=torch.float32, device=device).unsqueeze(1)
        self.y_va = torch.tensor(va_df['Target'].values, dtype=torch.float32, device=device).unsqueeze(1)
        self.y_te = torch.tensor(te_df['Target'].values, dtype=torch.float32, device=device).unsqueeze(1) if te_df is not None else None

        # Pesos opcionales de picos
        p90 = np.percentile(y_tr_vals, 90.0)
        self.weights_tr = torch.tensor(1.0 + (y_tr_vals / p90)**2, dtype=torch.float32, device=device).unsqueeze(1)

        # Estadisticas ajustadas estrictamente en train de este fold
        X_tr_raw = tr_df[num_cols].values.astype(np.float32)
        X_va_raw = va_df[num_cols].values.astype(np.float32)
        X_te_raw = te_df[num_cols].values.astype(np.float32) if te_df is not None else None

        means = np.nanmean(X_tr_raw, axis=0)
        stds = np.nanstd(X_tr_raw, axis=0)
        stds[stds < 1e-4] = 1.0

        null_in_tr = np.isnan(X_tr_raw).any(axis=0)
        null_col_indices = np.where(null_in_tr)[0]

        def transform(X_raw):
            if X_raw is None:
                return None
            mask = np.isnan(X_raw[:, null_col_indices]).astype(np.float32)
            X_norm = (X_raw - means) / stds
            X_norm = np.nan_to_num(X_norm, nan=0.0)
            X_norm = np.clip(X_norm, -10.0, 10.0)
            return np.hstack([X_norm, mask])

        self.X_tr = torch.tensor(transform(X_tr_raw), dtype=torch.float32, device=device)
        self.X_va = torch.tensor(transform(X_va_raw), dtype=torch.float32, device=device)
        self.X_te = torch.tensor(transform(X_te_raw), dtype=torch.float32, device=device) if X_te_raw is not None else None

        self.input_dim = self.X_tr.shape[1]
