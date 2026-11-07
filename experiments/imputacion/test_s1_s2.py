"""
Test S1 (1-3h gaps) and S2 (6-48h gaps) on real observed data.
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import load_panel, STATION_COORDS, MODEL_STATIONS

def calc_metrics(y_true, y_pred):
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {'n': 0, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'RMSE_peaks': np.nan, 'bias': np.nan}
    yt = y_true[ok]
    yp = y_pred[ok]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(np.mean(np.abs(yt - yp)))
    r2 = float(r2_score(yt, yp))
    bias = float(np.mean(yp - yt))
    thr = np.quantile(yt, 0.95)
    pk = yt >= thr
    rmse_peaks = float(np.sqrt(mean_squared_error(yt[pk], yp[pk]))) if pk.sum() > 0 else np.nan
    return {'n': int(ok.sum()), 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'RMSE_peaks': rmse_peaks, 'bias': bias}

def generate_gaps(series, gap_lengths, total_target_hours=1000, seed=42):
    """
    Finds contiguous valid segments in `series` and masks gaps of sizes in `gap_lengths`.
    Returns mask (True where masked), ground_truth series.
    """
    rng = np.random.RandomState(seed)
    valid = series.notna().values
    n = len(series)
    masked = np.zeros(n, dtype=bool)
    
    # We require at least 1 valid observation before and after each gap
    # Pick random candidate start positions
    accum_hours = 0
    attempts = 0
    max_attempts = 20000
    
    while accum_hours < total_target_hours and attempts < max_attempts:
        attempts += 1
        L = rng.choice(gap_lengths)
        # pick a random index
        idx = rng.randint(2, n - L - 2)
        # Check if idx-1 is valid, idx..idx+L-1 are valid and not already masked, and idx+L is valid
        if valid[idx-1] and valid[idx+L] and valid[idx:idx+L].all() and not masked[idx-1:idx+L+1].any():
            masked[idx:idx+L] = True
            accum_hours += L
            
    return masked

def test_s1_s2():
    panel = load_panel()
    # Test on year 2022 across 4 representative stations: SE3, CE, SO2, SE
    stations = ['SE3', 'CE', 'SO2', 'SE']
    
    for var in ['O3', 'PM2.5']:
        print(f"\n=========================== {var} ===========================")
        for scenario_name, gap_lens in [('S1 (1-3h)', [1, 2, 3]), ('S2 (6-48h)', [6, 12, 24, 48])]:
            print(f"\n--- Scenario {scenario_name} ---")
            results = {'Lineal': [], 'PCHIP': [], 'Clim': [], 'Clim+Adj': [], 'STL+Interp': []}
            
            for st in stations:
                st_df = panel[(panel['Estacion'] == st) & (panel['Date'].dt.year == 2022)].sort_values('Date').reset_index(drop=True)
                s = st_df[var].copy()
                mask = generate_gaps(s, gap_lens, total_target_hours=250, seed=42 + len(st))
                
                real = s[mask].values
                
                # 1. Linear
                s_masked = s.copy()
                s_masked[mask] = np.nan
                pred_lin = s_masked.interpolate(method='linear')[mask].values
                
                # 2. PCHIP
                try:
                    pred_pchip = s_masked.interpolate(method='pchip')[mask].values
                except Exception:
                    pred_pchip = pred_lin
                
                # 3. Clim
                tr = panel[(panel['Estacion'] == st) & (panel['Date'].dt.year <= 2023)]
                clim_map = tr.groupby([tr['Date'].dt.month, tr['Date'].dt.hour])[var].mean().to_dict()
                clim_full = [clim_map.get((m, h), np.nan) for m, h in zip(st_df['Date'].dt.month, st_df['Date'].dt.hour)]
                pred_clim = np.array(clim_full)[mask]
                
                # 4. Clim + Adj (anomaly interpolation)
                st_df['clim'] = clim_full
                st_df['anomaly'] = s_masked - st_df['clim']
                anom_interp = st_df['anomaly'].interpolate(method='linear')
                pred_clim_adj = np.maximum(0, st_df['clim'] + anom_interp)[mask].values
                
                # 5. STL + Interp
                # Fill missing with linear first to get a continuous series for STL
                s_filled = s_masked.interpolate(method='linear').ffill().bfill()
                try:
                    res = STL(s_filled, period=24, robust=True).fit()
                    # Anomaly = s_masked - seasonal
                    deseas = s_masked - res.seasonal
                    deseas_interp = deseas.interpolate(method='linear')
                    pred_stl = np.maximum(0, deseas_interp + res.seasonal)[mask].values
                except Exception as e:
                    pred_stl = pred_clim_adj
                
                results['Lineal'].append((real, pred_lin))
                results['PCHIP'].append((real, pred_pchip))
                results['Clim'].append((real, pred_clim))
                results['Clim+Adj'].append((real, pred_clim_adj))
                results['STL+Interp'].append((real, pred_stl))
            
            # Aggregate metrics across stations
            for m_name, pairs in results.items():
                all_real = np.concatenate([p[0] for p in pairs])
                all_pred = np.concatenate([p[1] for p in pairs])
                met = calc_metrics(all_real, all_pred)
                print(f"{m_name:<12}: RMSE={met['RMSE']:.2f}, R2={met['R2']:.3f}, MAE={met['MAE']:.2f}, RMSE_peaks={met['RMSE_peaks']:.2f}, Bias={met['bias']:+.2f} (n={met['n']})")

if __name__ == '__main__':
    test_s1_s2()
