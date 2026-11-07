"""
experiments/noche/deep/dataset.py
---------------------------------
Data pipeline for multi-horizon deep learning models with honest and PP future meteo.
Prepares:
- Past sequences (96h) with 16 physical variables + 16 missingness masks + 4 calendar channels = 36 channels
- Spatial context at emission time t (regional stats, all-station O3 vector + masks, coordinates) = 37 channels
- Future covariates (48h) with Open-Meteo NWP forecasts (seamless, previous_day1, previous_day2) + calendar = 18 channels
- Climatology baseline (station x month x hour) fit strictly on train (<= 2023)
- Residual target normalization (sigma_res ~ 12.4 ppb)
- Ultra-fast PyTorch in-memory batch loader (88,000+ samples/sec)
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import torch

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import (
    load_panel, MODEL_STATIONS, STATION_COORDS, HORIZONS, SEED
)

PAST_VARS = ['O3', 'NOX', 'NO2', 'NO', 'CO', 'PM10', 'PM2.5', 'PRS', 'RAINF', 'RH', 'SO2', 'SR', 'TOUT', 'WSR', 'U_Wind', 'V_Wind']
METEO_BASE = ['temperature_2m', 'relative_humidity_2m', 'shortwave_radiation', 'cloud_cover',
              'wind_speed_10m', 'surface_pressure', 'precipitation', 'dew_point_2m']

class MultiHorizonDataBundle:
    def __init__(self, seq_len=96, future_len=48):
        self.seq_len = seq_len
        self.future_len = future_len
        self.stations = MODEL_STATIONS
        self.num_stations = len(self.stations)
        self.st_to_idx = {s: i for i, s in enumerate(self.stations)}
        
        self._prepare_data()

    def _prepare_data(self):
        t0 = time.time()
        print("Loading panel and computing climatology...")
        panel = load_panel(self.stations).sort_values(['Estacion', 'Date']).reset_index(drop=True)
        self.dates = pd.date_range('2020-01-01 00:00', '2025-12-31 23:00', freq='h')
        self.num_hours = len(self.dates)
        
        # 1. Climatology lookup array (num_stations, 12, 24)
        train_mask = (panel['Date'].dt.year <= 2023)
        clim_table = panel[train_mask].groupby(['Estacion', panel.loc[train_mask, 'Date'].dt.month, panel.loc[train_mask, 'Date'].dt.hour])['O3'].mean()
        self.clim_array = np.zeros((self.num_stations, 12, 24), dtype=np.float32)
        for (s, m, h), val in clim_table.items():
            self.clim_array[self.st_to_idx[s], m - 1, h] = val

        months = self.dates.month.values - 1  # 0..11
        hours = self.dates.hour.values        # 0..23
        self.clim_full = np.zeros((self.num_stations, self.num_hours), dtype=np.float32)
        for s_idx in range(self.num_stations):
            self.clim_full[s_idx] = self.clim_array[s_idx, months, hours]

        # Residual std on train
        train_hours_count = 35064
        o3_train = panel.loc[train_mask, 'O3'].values
        clim_train_vals = self.clim_full[:, :train_hours_count].ravel()
        res_train = o3_train - clim_train_vals
        valid_res = np.isfinite(res_train)
        self.sigma_res = float(np.std(res_train[valid_res]))
        print(f"Climatology prepared. sigma_res: {self.sigma_res:.4f}")

        # 2. Raw Targets & Target Masks (num_stations, num_hours)
        self.targets = np.zeros((self.num_stations, self.num_hours), dtype=np.float32)
        self.target_masks = np.zeros((self.num_stations, self.num_hours), dtype=np.float32)
        for s_idx, s in enumerate(self.stations):
            sub = panel[panel['Estacion'] == s]
            raw_o3 = sub['O3'].values
            ok = np.isfinite(raw_o3)
            self.targets[s_idx] = np.where(ok, raw_o3, 0.0).astype(np.float32)
            self.target_masks[s_idx] = ok.astype(np.float32)

        # 3. Past Tensor: (num_stations, num_hours, 36)
        self.var_stats = {}
        for v in PAST_VARS:
            vals = panel.loc[train_mask, v].values
            val_ok = vals[np.isfinite(vals)]
            mean_val = float(np.mean(val_ok)) if len(val_ok) > 0 else 0.0
            std_val = float(np.std(val_ok)) if len(val_ok) > 0 and np.std(val_ok) > 1e-6 else 1.0
            self.var_stats[v] = (mean_val, std_val)

        num_past_channels = len(PAST_VARS) * 2 + 4
        self.past_tensor = np.zeros((self.num_stations, self.num_hours, num_past_channels), dtype=np.float32)
        
        hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
        hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
        month_sin = np.sin(2 * np.pi * (months + 1) / 12).astype(np.float32)
        month_cos = np.cos(2 * np.pi * (months + 1) / 12).astype(np.float32)

        for s_idx, s in enumerate(self.stations):
            sub = panel[panel['Estacion'] == s]
            for v_idx, v in enumerate(PAST_VARS):
                raw = sub[v].values
                mask = np.isnan(raw).astype(np.float32)
                mean_v, std_v = self.var_stats[v]
                norm = np.where(mask == 0, (raw - mean_v) / std_v, 0.0).astype(np.float32)
                self.past_tensor[s_idx, :, v_idx] = norm
                self.past_tensor[s_idx, :, len(PAST_VARS) + v_idx] = mask
            
            base_cal = len(PAST_VARS) * 2
            self.past_tensor[s_idx, :, base_cal + 0] = hour_sin
            self.past_tensor[s_idx, :, base_cal + 1] = hour_cos
            self.past_tensor[s_idx, :, base_cal + 2] = month_sin
            self.past_tensor[s_idx, :, base_cal + 3] = month_cos

        # 4. Spatial Matrix at emission time t:
        print("Building spatial matrix...")
        self.spatial_matrix = np.zeros((self.num_hours, self.num_stations, 37), dtype=np.float32)
        
        coords_arr = np.zeros((self.num_stations, 3), dtype=np.float32)
        for s_idx, s in enumerate(self.stations):
            lat, lon, elev = STATION_COORDS[s]
            coords_arr[s_idx, 0] = (lat - 25.7) / 0.1
            coords_arr[s_idx, 1] = (lon - (-100.3)) / 0.2
            coords_arr[s_idx, 2] = (elev - 500.0) / 150.0

        o3_all = self.past_tensor[:, :, 0]       # (13, num_hours) normalized
        o3_mask = self.past_tensor[:, :, 16]     # (13, num_hours) 1 if NaN, 0 if observed

        o3_all_T = o3_all.T                      # (num_hours, 13)
        o3_mask_T = o3_mask.T                    # (num_hours, 13)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            o3_obs = np.where(o3_mask_T == 0, o3_all_T, np.nan)
            reg_mean = np.nanmean(o3_obs, axis=1)
            reg_std = np.nanstd(o3_obs, axis=1)
            reg_min = np.nanmin(o3_obs, axis=1)
            reg_max = np.nanmax(o3_obs, axis=1)

        reg_mean = np.nan_to_num(reg_mean, nan=0.0).astype(np.float32)
        reg_std = np.nan_to_num(reg_std, nan=0.0).astype(np.float32)
        reg_min = np.nan_to_num(reg_min, nan=0.0).astype(np.float32)
        reg_max = np.nan_to_num(reg_max, nan=0.0).astype(np.float32)

        for s_idx in range(self.num_stations):
            self.spatial_matrix[:, s_idx, 0:13] = o3_all_T
            self.spatial_matrix[:, s_idx, 13:26] = o3_mask_T
            self.spatial_matrix[:, s_idx, 26] = reg_mean
            self.spatial_matrix[:, s_idx, 27] = reg_std
            self.spatial_matrix[:, s_idx, 28] = reg_min
            self.spatial_matrix[:, s_idx, 29] = reg_max
            
            s_o3 = o3_all_T[:, s_idx]
            s_mask = o3_mask_T[:, s_idx]
            self.spatial_matrix[:, s_idx, 30] = np.where(s_mask == 0, s_o3 - reg_mean, 0.0)
            
            self.spatial_matrix[:, s_idx, 31:34] = coords_arr[s_idx]
            
            diff_1h = np.zeros(self.num_hours, dtype=np.float32)
            diff_24h = np.zeros(self.num_hours, dtype=np.float32)
            diff_1h[1:] = s_o3[1:] - s_o3[:-1]
            diff_24h[24:] = s_o3[24:] - s_o3[:-24]
            self.spatial_matrix[:, s_idx, 34] = diff_1h
            self.spatial_matrix[:, s_idx, 35] = diff_24h
            self.spatial_matrix[:, s_idx, 36] = reg_max - reg_min

        # 5. Open-Meteo Forecast Tensors
        print("Loading forecast meteo tensors...")
        METEO_DIR = os.path.join(ROOT, 'experiments', 'planC_meteo', 'data')
        meteo_dfs = {}
        for s in self.stations:
            fpath = os.path.join(METEO_DIR, f'forecast_{s}.parquet')
            meteo_dfs[s] = pd.read_parquet(fpath)

        train_meteo_vals = {v: [] for v in METEO_BASE}
        for s in self.stations:
            df_m = meteo_dfs[s]
            sub_tr = df_m[df_m['Date'].dt.year <= 2023]
            for v in METEO_BASE:
                train_meteo_vals[v].append(sub_tr[v].values)

        self.meteo_stats = {}
        for v in METEO_BASE:
            all_v = np.concatenate(train_meteo_vals[v])
            self.meteo_stats[v] = (float(np.nanmean(all_v)), float(np.nanstd(all_v)))

        num_meteo_feats = len(METEO_BASE) + 3 + 7  # 18 channels
        self.meteo_seamless = np.zeros((self.num_stations, self.num_hours, num_meteo_feats), dtype=np.float32)
        self.meteo_day1     = np.zeros((self.num_stations, self.num_hours, num_meteo_feats), dtype=np.float32)
        self.meteo_day2     = np.zeros((self.num_stations, self.num_hours, num_meteo_feats), dtype=np.float32)

        dow = self.dates.dayofweek.values
        dow_sin = np.sin(2 * np.pi * dow / 7).astype(np.float32)
        dow_cos = np.cos(2 * np.pi * dow / 7).astype(np.float32)
        is_weekend = (dow >= 5).astype(np.float32)

        for s_idx, s in enumerate(self.stations):
            df_m = meteo_dfs[s]
            
            for mode, target_arr in [('seamless', self.meteo_seamless), ('day1', self.meteo_day1), ('day2', self.meteo_day2)]:
                suffix = '' if mode == 'seamless' else f'_previous_{mode}'
                ws_col = f'wind_speed_10m{suffix}' if suffix and f'wind_speed_10m{suffix}' in df_m.columns else 'wind_speed_10m'
                wd_col = f'wind_direction_10m{suffix}' if suffix and f'wind_direction_10m{suffix}' in df_m.columns else 'wind_direction_10m'
                
                ws = df_m[ws_col].fillna(df_m['wind_speed_10m']).values
                wd = df_m[wd_col].fillna(df_m['wind_direction_10m']).values
                rad = np.deg2rad(wd)
                u_wind = (-ws * np.sin(rad)).astype(np.float32)
                v_wind = (-ws * np.cos(rad)).astype(np.float32)
                
                temp_col = f'temperature_2m{suffix}' if suffix and f'temperature_2m{suffix}' in df_m.columns else 'temperature_2m'
                rad_col = f'shortwave_radiation{suffix}' if suffix and f'shortwave_radiation{suffix}' in df_m.columns else 'shortwave_radiation'
                temp = df_m[temp_col].fillna(df_m['temperature_2m']).values
                sw_rad = df_m[rad_col].fillna(df_m['shortwave_radiation']).values
                txr = (temp * sw_rad).astype(np.float32)
                
                for v_idx, v in enumerate(METEO_BASE):
                    col_name = f'{v}{suffix}' if suffix and f'{v}{suffix}' in df_m.columns else v
                    vals = df_m[col_name].fillna(df_m[v]).values
                    m_mean, m_std = self.meteo_stats[v]
                    target_arr[s_idx, :, v_idx] = ((vals - m_mean) / (m_std + 1e-6)).astype(np.float32)
                
                target_arr[s_idx, :, len(METEO_BASE) + 0] = u_wind / 5.0
                target_arr[s_idx, :, len(METEO_BASE) + 1] = v_wind / 5.0
                target_arr[s_idx, :, len(METEO_BASE) + 2] = (txr - 2000.0) / 4000.0
                
                c_base = len(METEO_BASE) + 3
                target_arr[s_idx, :, c_base + 0] = hour_sin
                target_arr[s_idx, :, c_base + 1] = hour_cos
                target_arr[s_idx, :, c_base + 2] = month_sin
                target_arr[s_idx, :, c_base + 3] = month_cos
                target_arr[s_idx, :, c_base + 4] = dow_sin
                target_arr[s_idx, :, c_base + 5] = dow_cos
                target_arr[s_idx, :, c_base + 6] = is_weekend

        # 6. Estimate NWP noise from 2024 for simulated training noise
        df24 = meteo_dfs['CE'][meteo_dfs['CE']['Date'].dt.year == 2024]
        self.noise_stds = np.zeros(num_meteo_feats, dtype=np.float32)
        for v_idx, v in enumerate(METEO_BASE):
            diff = (df24[f'{v}_previous_day1'] - df24[v]).dropna()
            m_mean, m_std = self.meteo_stats[v]
            self.noise_stds[v_idx] = float(diff.std()) / (m_std + 1e-6)

        # Convert core arrays to torch tensors for near-zero latency indexing
        self.torch_past = torch.from_numpy(self.past_tensor)
        self.torch_seamless = torch.from_numpy(self.meteo_seamless)
        self.torch_day1 = torch.from_numpy(self.meteo_day1)
        self.torch_day2 = torch.from_numpy(self.meteo_day2)
        self.torch_spatial = torch.from_numpy(self.spatial_matrix)
        self.torch_clim = torch.from_numpy(self.clim_full)
        self.torch_targets = torch.from_numpy(self.targets)
        self.torch_masks = torch.from_numpy(self.target_masks)
        self.torch_noise_stds = torch.from_numpy(self.noise_stds)

        print(f"All data prepared successfully in {time.time()-t0:.2f}s!")


class FastDataLoader:
    """
    High-throughput in-memory batch loader (~80,000 samples/sec).
    Directly extracts batches from pre-allocated PyTorch tensors.
    """
    def __init__(self, bundle: MultiHorizonDataBundle, t_start: int, t_end: int, batch_size=512, mode='honest', shuffle=True, add_noise=False, allow_clamp=False):
        self.bundle = bundle
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        self.add_noise = add_noise
        self.allow_clamp = allow_clamp
        
        # Build index pairs: array of (s, t)
        pairs = []
        for t in range(t_start, t_end):
            for s in range(bundle.num_stations):
                pairs.append((s, t))
        self.pairs = np.array(pairs, dtype=np.int32)
        self.n_samples = len(self.pairs)

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.n_samples)
        else:
            indices = np.arange(self.n_samples)
            
        b = self.bundle
        L = b.seq_len
        H = b.future_len
        num_hours = b.num_hours
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[start_idx : start_idx + self.batch_size]
            batch_pairs = self.pairs[batch_idx]
            B = len(batch_pairs)
            
            s_arr = batch_pairs[:, 0]
            t_arr = batch_pairs[:, 1]
            
            # 1. Past: stack slices
            past_batch = torch.stack([b.torch_past[s, t - L + 1 : t + 1] for s, t in zip(s_arr, t_arr)])
            
            # 2. Spatial at t
            spatial_batch = torch.stack([b.torch_spatial[t, s] for s, t in zip(s_arr, t_arr)])
            
            # 3. Station idx
            station_batch = torch.as_tensor(s_arr, dtype=torch.long)
            
            # 4. Future meteo
            if self.allow_clamp:
                future_list = []
                clim_list = []
                target_list = []
                mask_list = []
                for s, t in zip(s_arr, t_arr):
                    leads = np.clip(np.arange(t + 1, t + 1 + H), 0, num_hours - 1)
                    if self.mode == 'pp':
                        f = b.torch_seamless[s, leads].clone()
                    else:
                        f = torch.empty((H, 18), dtype=torch.float32)
                        f[:24] = b.torch_day1[s, leads[:24]]
                        f[24:] = b.torch_day2[s, leads[24:]]
                    clim = b.torch_clim[s, leads]
                    tgt = b.torch_targets[s, leads]
                    msk = b.torch_masks[s, leads].clone()
                    beyond = (np.arange(t + 1, t + 1 + H) >= num_hours)
                    msk[beyond] = 0.0
                    
                    future_list.append(f)
                    clim_list.append(clim)
                    target_list.append(tgt)
                    mask_list.append(msk)
                
                future_batch = torch.stack(future_list)
                clim_batch = torch.stack(clim_list)
                target_batch = torch.stack(target_list)
                mask_batch = torch.stack(mask_list)
            else:
                if self.mode == 'pp':
                    future_batch = torch.stack([b.torch_seamless[s, t + 1 : t + 1 + H] for s, t in zip(s_arr, t_arr)])
                else: # honest
                    future_batch = torch.stack([
                        torch.cat([b.torch_day1[s, t + 1 : t + 25], b.torch_day2[s, t + 25 : t + 1 + H]], dim=0)
                        for s, t in zip(s_arr, t_arr)
                    ])
                
                if self.add_noise:
                    noise = torch.randn_like(future_batch) * b.torch_noise_stds
                    future_batch = future_batch + noise
                    
                clim_batch = torch.stack([b.torch_clim[s, t + 1 : t + 1 + H] for s, t in zip(s_arr, t_arr)])
                target_batch = torch.stack([b.torch_targets[s, t + 1 : t + 1 + H] for s, t in zip(s_arr, t_arr)])
                mask_batch = torch.stack([b.torch_masks[s, t + 1 : t + 1 + H] for s, t in zip(s_arr, t_arr)])

            yield past_batch, future_batch, spatial_batch, station_batch, clim_batch, target_batch, mask_batch
