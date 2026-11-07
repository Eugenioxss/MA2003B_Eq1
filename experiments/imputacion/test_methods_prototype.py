"""
Test and benchmark methods on S3 (SE3 June 2022) for O3 and PM2.5.
"""
import sys
import os
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
from scipy.sparse.linalg import svds
from sklearn.linear_model import Ridge
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import load_panel, STATION_COORDS, MODEL_STATIONS

def get_station_distances():
    stations = sorted(list(STATION_COORDS.keys()))
    coords = {s: (STATION_COORDS[s][0], STATION_COORDS[s][1]) for s in stations}
    dist = {}
    for s1 in stations:
        dist[s1] = {}
        lat1, lon1 = np.radians(coords[s1][0]), np.radians(coords[s1][1])
        for s2 in stations:
            if s1 == s2:
                dist[s1][s2] = 0.0
                continue
            lat2, lon2 = np.radians(coords[s2][0]), np.radians(coords[s2][1])
            dphi = lat2 - lat1
            dlam = lon2 - lon1
            a = np.sin(dphi/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlam/2)**2
            d = 2 * 6371.0 * np.arcsin(np.sqrt(a))
            dist[s1][s2] = d
    return dist

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

def run_s3_test():
    print("Loading panel...")
    panel = load_panel()
    gta = pd.read_csv('check_LSTM_PC/eval_bloques/ground_truth_A.csv', parse_dates=['time'])
    
    # Mask S3 window: SE3 in June 2022
    mask_s3 = (panel['Estacion'] == 'SE3') & (panel['Date'].dt.year == 2022) & (panel['Date'].dt.month == 6)
    
    target_vars = ['O3', 'PM2.5']
    dist = get_station_distances()
    
    for var in target_vars:
        print(f"\n===================== S3 Benchmark: Variable {var} =====================")
        # Ground truth
        real_series = gta[['time', var]].dropna().rename(columns={'time': 'Date', var: 'real'})
        
        # 1. Climatología (estación x mes x hora) sin el mes enmascarado
        tr = panel[(panel['Estacion'] == 'SE3') & (panel['Date'].dt.year <= 2023) & (~mask_s3)]
        clim_map = tr.groupby([tr['Date'].dt.month, tr['Date'].dt.hour])[var].mean()
        
        # Pred climatología
        s3_dates = panel.loc[mask_s3, ['Date']].copy()
        s3_dates['month'] = s3_dates['Date'].dt.month
        s3_dates['hour'] = s3_dates['Date'].dt.hour
        clim_pred = s3_dates.set_index(['month', 'hour']).index.map(clim_map).values
        s3_dates['pred_clim'] = clim_pred
        
        # Eval clim
        df_eval = real_series.merge(s3_dates, on='Date')
        m_clim = calc_metrics(df_eval['real'].values, df_eval['pred_clim'].values)
        print(f"Climatologia:             RMSE={m_clim['RMSE']:.2f}, R2={m_clim['R2']:.3f}, MAE={m_clim['MAE']:.2f}, RMSE_peaks={m_clim['RMSE_peaks']:.2f}, Bias={m_clim['bias']:+.2f}")
        
        # 2. Climatología + ajuste por nivel (anomalía en bordes)
        se3_full = panel[panel['Estacion'] == 'SE3'].sort_values('Date').reset_index(drop=True)
        # Find index of June 2022 in se3_full
        idx_s3 = se3_full.index[(se3_full['Date'].dt.year == 2022) & (se3_full['Date'].dt.month == 6)]
        t0_idx = idx_s3[0] - 1
        t1_idx = idx_s3[-1] + 1
        
        # Border anomalies
        clim_full = se3_full['Date'].dt.month.astype(str) + '_' + se3_full['Date'].dt.hour.astype(str)
        # get clim map dict
        clim_dict = clim_map.to_dict()
        se3_clim = [clim_dict.get((m, h), np.nan) for m, h in zip(se3_full['Date'].dt.month, se3_full['Date'].dt.hour)]
        se3_full['clim'] = se3_clim
        se3_full['anomaly'] = se3_full[var] - se3_full['clim']
        
        # Interpolate anomaly across the gap
        anom_copy = se3_full['anomaly'].copy()
        anom_copy.iloc[idx_s3] = np.nan
        anom_interp = anom_copy.interpolate(method='linear')
        pred_clim_adj = np.maximum(0, se3_full.iloc[idx_s3]['clim'].values + anom_interp.iloc[idx_s3].values)
        m_clim_adj = calc_metrics(df_eval['real'].values, pred_clim_adj[df_eval.index])
        print(f"Climatologia + ajuste niv: RMSE={m_clim_adj['RMSE']:.2f}, R2={m_clim_adj['R2']:.3f}, MAE={m_clim_adj['MAE']:.2f}, RMSE_peaks={m_clim_adj['RMSE_peaks']:.2f}, Bias={m_clim_adj['bias']:+.2f}")
        
        # 3. Lineal / PCHIP
        var_copy = se3_full[var].copy()
        var_copy.iloc[idx_s3] = np.nan
        pred_linear = var_copy.interpolate(method='linear').iloc[idx_s3].values
        m_lin = calc_metrics(df_eval['real'].values, pred_linear[df_eval.index])
        print(f"Lineal:                   RMSE={m_lin['RMSE']:.2f}, R2={m_lin['R2']:.3f}, MAE={m_lin['MAE']:.2f}, RMSE_peaks={m_lin['RMSE_peaks']:.2f}, Bias={m_lin['bias']:+.2f}")
        
        # 4. IDW espacial
        # Pivot panel for variable across stations
        piv = panel.pivot(index='Date', columns='Estacion', values=var)
        # For dates in June 2022
        dates_s3 = se3_full.iloc[idx_s3]['Date'].values
        piv_s3 = piv.loc[dates_s3].copy()
        piv_s3['SE3'] = np.nan # masked
        
        other_st = [s for s in piv.columns if s != 'SE3']
        dists = np.array([dist['SE3'][s] for s in other_st])
        weights = 1.0 / (dists ** 2)
        
        idw_preds = []
        for dt, row in piv_s3[other_st].iterrows():
            vals = row.values
            val_mask = np.isfinite(vals)
            if val_mask.sum() > 0:
                w = weights[val_mask]
                v = vals[val_mask]
                idw_preds.append(np.sum(w * v) / np.sum(w))
            else:
                # fallback to clim
                m, h = dt.month, dt.hour
                idw_preds.append(clim_dict.get((m, h), np.nan))
        idw_preds = np.array(idw_preds)
        m_idw = calc_metrics(df_eval['real'].values, idw_preds[df_eval.index])
        print(f"IDW Espacial:             RMSE={m_idw['RMSE']:.2f}, R2={m_idw['R2']:.3f}, MAE={m_idw['MAE']:.2f}, RMSE_peaks={m_idw['RMSE_peaks']:.2f}, Bias={m_idw['bias']:+.2f}")
        
        # 5. Regresión Espacial LightGBM
        # Train a LightGBM predicting SE3[var] using other stations' var + meteo + calendar
        # Prepare wide table for training: 2020-2023 excluding June 2022
        # Donor features: var at other stations, plus TOUT, SR, RH, U_Wind, V_Wind at donor stations
        # Let's pivot key vars
        print("Training LightGBM spatial regression...")
        key_vars = [var, 'TOUT', 'SR', 'RH', 'U_Wind', 'V_Wind'] if var != 'TOUT' else [var, 'SR', 'RH', 'U_Wind', 'V_Wind']
        donor_features = []
        for kv in key_vars:
            p_kv = panel.pivot(index='Date', columns='Estacion', values=kv)
            cols = [c for c in p_kv.columns if c != 'SE3']
            p_kv_sub = p_kv[cols].add_prefix(f'{kv}_')
            donor_features.append(p_kv_sub)
        X_wide = pd.concat(donor_features, axis=1)
        X_wide['hour'] = X_wide.index.hour
        X_wide['month'] = X_wide.index.month
        X_wide['hour_sin'] = np.sin(2 * np.pi * X_wide['hour'] / 24)
        X_wide['hour_cos'] = np.cos(2 * np.pi * X_wide['hour'] / 24)
        X_wide['month_sin'] = np.sin(2 * np.pi * X_wide['month'] / 12)
        X_wide['month_cos'] = np.cos(2 * np.pi * X_wide['month'] / 12)
        
        y_series = piv['SE3']
        # Train mask: 2020-2023 excluding June 2022
        tr_mask = (X_wide.index.year <= 2023) & (~((X_wide.index.year == 2022) & (X_wide.index.month == 6))) & y_series.notna()
        X_tr = X_wide.loc[tr_mask]
        y_tr = y_series.loc[tr_mask]
        
        val_mask_lgb = (X_wide.index.year == 2023) & y_series.notna()
        # To avoid leak, train on 2020-2022 (except Jun 2022) and val on 2023
        tr_split_mask = (X_wide.index.year <= 2022) & (~((X_wide.index.year == 2022) & (X_wide.index.month == 6))) & y_series.notna()
        
        lgb_tr = lgb.Dataset(X_wide.loc[tr_split_mask], y_series.loc[tr_split_mask])
        lgb_va = lgb.Dataset(X_wide.loc[val_mask_lgb], y_series.loc[val_mask_lgb], reference=lgb_tr)
        
        params = {
            'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05,
            'num_leaves': 31, 'min_child_samples': 30, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'num_threads': 3, 'seed': 42, 'verbose': -1
        }
        bst = lgb.train(params, lgb_tr, num_boost_round=1000, valid_sets=[lgb_va],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        
        X_s3 = X_wide.loc[dates_s3]
        lgb_preds = bst.predict(X_s3)
        m_lgb = calc_metrics(df_eval['real'].values, lgb_preds[df_eval.index])
        print(f"Regresion Espacial LightGBM: RMSE={m_lgb['RMSE']:.2f}, R2={m_lgb['R2']:.3f}, MAE={m_lgb['MAE']:.2f}, RMSE_peaks={m_lgb['RMSE_peaks']:.2f}, Bias={m_lgb['bias']:+.2f}")
        
        # 6. SoftImpute (Iterative Truncated SVD)
        # Matrix T x N for this variable in 2022
        print("Running SoftImpute SVD...")
        piv_2022 = piv[piv.index.year == 2022].copy()
        piv_2022.loc[dates_s3, 'SE3'] = np.nan
        
        # SoftImpute on piv_2022
        M = piv_2022.values.astype(float)
        obs_mask = np.isfinite(M)
        col_means = np.nanmean(M, axis=0)
        col_stds = np.nanstd(M, axis=0)
        col_stds[col_stds == 0] = 1.0
        
        # Standardize
        M_norm = (M - col_means) / col_stds
        M_norm[~obs_mask] = 0.0
        
        # Iterate SVD rank 3
        X_k = M_norm.copy()
        for it in range(20):
            U, s, Vt = svds(X_k, k=min(4, M.shape[1]-1))
            X_rec = U @ np.diag(s) @ Vt
            X_k[~obs_mask] = X_rec[~obs_mask]
            
        M_imp = X_k * col_stds + col_means
        M_imp = np.maximum(0, M_imp)
        se3_col_idx = list(piv_2022.columns).index('SE3')
        soft_s3_preds = M_imp[:, se3_col_idx]
        
        # Align with dates_s3
        s3_idx_in_2022 = [piv_2022.index.get_loc(d) for d in dates_s3]
        soft_preds = soft_s3_preds[s3_idx_in_2022]
        m_soft = calc_metrics(df_eval['real'].values, soft_preds[df_eval.index])
        print(f"SoftImpute (SVD iterativo):  RMSE={m_soft['RMSE']:.2f}, R2={m_soft['R2']:.3f}, MAE={m_soft['MAE']:.2f}, RMSE_peaks={m_soft['RMSE_peaks']:.2f}, Bias={m_soft['bias']:+.2f}")
        
        # SAITS reference from audit
        saits_se3 = pd.read_parquet('experiments/imputacion/saits_se3.parquet')
        saits_s3 = saits_se3.merge(df_eval[['Date', 'real']], left_on='time', right_on='Date')
        m_saits = calc_metrics(saits_s3['real'].values, saits_s3[var].values)
        print(f"SAITS+CSDI (estudio previo): RMSE={m_saits['RMSE']:.2f}, R2={m_saits['R2']:.3f}, MAE={m_saits['MAE']:.2f}, RMSE_peaks={m_saits['RMSE_peaks']:.2f}, Bias={m_saits['bias']:+.2f}")

if __name__ == '__main__':
    run_s3_test()
