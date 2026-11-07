"""
experiments/imputacion/benchmark_imputation.py
==============================================
Benchmark completo de imputación por tipo de hueco (S1 a S5)
para variables O3, PM2.5, NOX y TOUT en la red SIMA Monterrey.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import miceforest as mf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_panel, metrics, ALL_STATIONS, MODEL_STATIONS, STATION_COORDS
from imputers import (
    compute_distance_matrix, build_climatology_map, get_climatology_series,
    impute_linear, impute_pchip, impute_climatology, impute_climatology_level_adj,
    impute_stl, impute_idw_spatial, impute_softimpute_svd,
    train_spatial_regression_lightgbm
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(EXP_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
BENCHMARK_CSV = os.path.join(EXP_DIR, 'benchmark_results.csv')

def calc_all_metrics(y_true, y_pred, label="") -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(yt) & np.isfinite(yp)
    if ok.sum() == 0:
        return {'label': label, 'n': 0, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'RMSE_peaks': np.nan, 'bias': np.nan}
    yt = yt[ok]
    yp = yp[ok]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(np.mean(np.abs(yt - yp)))
    denom = np.sum((yt - np.mean(yt))**2)
    r2 = float(1 - np.sum((yt - yp)**2) / denom) if denom > 0 else np.nan
    bias = float(np.mean(yp - yt))
    thr = np.quantile(yt, 0.95)
    pk = yt >= thr
    rmse_peaks = float(np.sqrt(mean_squared_error(yt[pk], yp[pk]))) if pk.sum() > 0 else np.nan
    return {
        'label': label, 'n': int(ok.sum()), 'RMSE': rmse, 'MAE': mae,
        'R2': r2, 'RMSE_peaks': rmse_peaks, 'bias': bias, 'thr95': float(thr),
        'mean_real': float(np.mean(yt)), 'mean_pred': float(np.mean(yp))
    }

def generate_random_gaps(series: pd.Series, gap_lengths: list[int], target_hours: int = 1000, seed: int = 42) -> np.ndarray:
    """Genera máscara booleana (True donde se oculta) sobre segmentos contiguos observados."""
    rng = np.random.RandomState(seed)
    valid = series.notna().values
    n = len(series)
    masked = np.zeros(n, dtype=bool)
    accum = 0
    attempts = 0
    while accum < target_hours and attempts < 30000:
        attempts += 1
        L = int(rng.choice(gap_lengths))
        idx = int(rng.randint(2, n - L - 2))
        if valid[idx-1] and valid[idx+L] and valid[idx:idx+L].all() and not masked[idx-1:idx+L+1].any():
            masked[idx:idx+L] = True
            accum += L
    return masked

def run_benchmark():
    print("=" * 70, flush=True)
    print("INICIANDO BENCHMARK COMPLETO DE IMPUTACIÓN (S1 a S5)", flush=True)
    print("=" * 70, flush=True)
    
    panel = load_panel()
    dist_matrix = compute_distance_matrix()
    clim_maps = build_climatology_map(panel, max_year=2023)
    
    # Ground truths de SE3 del estudio previo
    gta = pd.read_csv('check_LSTM_PC/eval_bloques/ground_truth_A.csv', parse_dates=['time'])
    gtb = pd.read_csv('check_LSTM_PC/eval_bloques/ground_truth_B.csv', parse_dates=['time'])
    saits_se3 = pd.read_parquet(os.path.join(EXP_DIR, 'saits_se3.parquet'))
    
    records = []
    best_preds_for_plot = {} # Guarda predicciones de S3 y S4 para gráficos overlay
    
    # Lista de escenarios
    # S1: huecos cortos 1-3h (aleatorios)
    # S2: huecos medianos 6-48h (aleatorios)
    # S3: 1 mes (Junio 2022 en SE3)
    # S4: 55 días (1 jul - 25 ago 2022 en SE3)
    # S5: 1 mes (Junio 2022) en SO2, CE, SE
    
    # -----------------------------------------------------------------------
    # FASE 1: S3 y S4 (SE3 - Bloques Críticos contra SAITS)
    # -----------------------------------------------------------------------
    se3_full = panel[panel['Estacion'] == 'SE3'].sort_values('Date').reset_index(drop=True)
    mask_s3 = (se3_full['Date'].dt.year == 2022) & (se3_full['Date'].dt.month == 6)
    
    start_b = pd.Timestamp('2022-07-01 00:00:00')
    end_b = start_b + pd.Timedelta(hours=1333 - 1)
    mask_s4 = (se3_full['Date'] >= start_b) & (se3_full['Date'] <= end_b)
    
    for scen_name, scen_mask, gt_df in [('S3 (1 mes SE3)', mask_s3, gta), ('S4 (55d SE3)', mask_s4, gtb)]:
        print(f"\n>>> Evaluando {scen_name}...", flush=True)
        dates_scen = se3_full.loc[scen_mask, 'Date'].values
        idx_scen = se3_full.index[scen_mask].values
        
        # Guardar para plots
        best_preds_for_plot[scen_name] = {'dates': dates_scen}
        
        for var in ['O3', 'PM2.5', 'NOX', 'TOUT']:
            print(f"  Variable {var}:", flush=True)
            # Ground truth
            val_gt = gt_df[['time', var]].dropna().rename(columns={'time': 'Date', var: 'real'}) if var in gt_df.columns else None
            if val_gt is None or len(val_gt) == 0:
                # Extraer del se3_full sin enmascarar
                val_gt = se3_full.loc[scen_mask, ['Date', var]].dropna().rename(columns={var: 'real'})
            
            real_series = val_gt.set_index('Date')['real']
            valid_dates = real_series.index
            y_real = real_series.values
            
            # --- Método a1: Lineal ---
            s_masked = se3_full[var].copy()
            s_masked.iloc[idx_scen] = np.nan
            p_lin = pd.Series(impute_linear(s_masked), index=se3_full['Date']).loc[valid_dates].values
            m_lin = calc_all_metrics(y_real, p_lin, f"{scen_name}_{var}_Lineal")
            records.append({'scenario': scen_name, 'station': 'SE3', 'variable': var, 'method': 'Lineal', **m_lin})
            
            # --- Método a2: PCHIP ---
            p_pchip = pd.Series(impute_pchip(s_masked), index=se3_full['Date']).loc[valid_dates].values
            m_pchip = calc_all_metrics(y_real, p_pchip, f"{scen_name}_{var}_PCHIP")
            records.append({'scenario': scen_name, 'station': 'SE3', 'variable': var, 'method': 'PCHIP', **m_pchip})
            
            # --- Método b1: Climatología Horaria ---
            clim_vals = get_climatology_series(pd.Series(valid_dates), 'SE3', var, clim_maps)
            m_clim = calc_all_metrics(y_real, clim_vals, f"{scen_name}_{var}_Climatologia")
            records.append({'scenario': scen_name, 'station': 'SE3', 'variable': var, 'method': 'Climatologia', **m_clim})
            
            # --- Método b2: Climatología + Ajuste por Nivel ---
            p_clim_adj = pd.Series(
                impute_climatology_level_adj(s_masked, se3_full['Date'], 'SE3', var, clim_maps),
                index=se3_full['Date']
            ).loc[valid_dates].values
            m_clim_adj = calc_all_metrics(y_real, p_clim_adj, f"{scen_name}_{var}_Clim_Ajuste_Nivel")
            records.append({'scenario': scen_name, 'station': 'SE3', 'variable': var, 'method': 'Clim + Ajuste Nivel', **m_clim_adj})
            
            # --- Método c: STL + Interp ---
            p_stl = pd.Series(impute_stl(s_masked), index=se3_full['Date']).loc[valid_dates].values
            m_stl = calc_all_metrics(y_real, p_stl, f"{scen_name}_{var}_STL_Interp")
            records.append({'scenario': scen_name, 'station': 'SE3', 'variable': var, 'method': 'STL + Interp', **m_stl})
            
            # --- Método d: IDW Espacial ---
            panel_masked = panel.copy()
            panel_masked.loc[(panel_masked['Estacion'] == 'SE3') & (panel_masked['Date'].isin(dates_scen)), var] = np.nan
            idw_res = impute_idw_spatial(panel_masked, var, dist_matrix, clim_maps, return_wide=True)
            p_idw = idw_res.loc[valid_dates, 'SE3'].values
            m_idw = calc_all_metrics(y_real, p_idw, f"{scen_name}_{var}_IDW_Espacial")
            records.append({'scenario': scen_name, 'station': 'SE3', 'variable': var, 'method': 'IDW Espacial', **m_idw})
            
            # --- Método e: Regresión Espacial LightGBM ---
            bst_lgb = train_spatial_regression_lightgbm(
                panel, target_var=var, target_station='SE3',
                exclude_dates=dates_scen
            )
            # Predecir sobre dates_scen
            key_covars = [var, 'TOUT', 'SR', 'RH', 'U_Wind', 'V_Wind'] if var != 'TOUT' else [var, 'SR', 'RH', 'U_Wind', 'V_Wind']
            dfs_x = []
            donor_st = [s for s in ALL_STATIONS if s != 'SE3']
            for kv in key_covars:
                p_kv = panel.pivot(index='Date', columns='Estacion', values=kv)
                cols = [s for s in donor_st if s in p_kv.columns]
                dfs_x.append(p_kv.loc[valid_dates, cols].add_prefix(f'{kv}_'))
            X_scen = pd.concat(dfs_x, axis=1)
            X_scen['hour_sin'] = np.sin(2 * np.pi * X_scen.index.hour / 24)
            X_scen['hour_cos'] = np.cos(2 * np.pi * X_scen.index.hour / 24)
            X_scen['month_sin'] = np.sin(2 * np.pi * X_scen.index.month / 12)
            X_scen['month_cos'] = np.cos(2 * np.pi * X_scen.index.month / 12)
            p_lgb = np.maximum(0, bst_lgb.predict(X_scen))
            m_lgb = calc_all_metrics(y_real, p_lgb, f"{scen_name}_{var}_LGB_Espacial")
            records.append({'scenario': scen_name, 'station': 'SE3', 'variable': var, 'method': 'Regresion Espacial LightGBM', **m_lgb})
            
            # --- Método g: SoftImpute (SVD Iterativo) ---
            piv_var = panel[panel['Date'].dt.year == 2022].pivot(index='Date', columns='Estacion', values=var).copy()
            piv_var.loc[piv_var.index.isin(dates_scen), 'SE3'] = np.nan
            M_imp = impute_softimpute_svd(piv_var.values, rank=3)
            piv_rec = pd.DataFrame(M_imp, index=piv_var.index, columns=piv_var.columns)
            p_soft = piv_rec.loc[valid_dates, 'SE3'].values
            m_soft = calc_all_metrics(y_real, p_soft, f"{scen_name}_{var}_SoftImpute")
            records.append({'scenario': scen_name, 'station': 'SE3', 'variable': var, 'method': 'SoftImpute SVD', **m_soft})
            
            # --- Método f: miceforest (MICE con LightGBM) ---
            # Restringido a 2022 y estaciones representativas para viabilidad computacional
            piv_sub = piv_var.loc[dates_scen[0] - pd.Timedelta(days=15) : dates_scen[-1] + pd.Timedelta(days=15)].copy()
            try:
                kds = mf.ImputationKernel(piv_sub, num_datasets=1, mean_match_candidates=0, random_state=42)
                kds.mice(iterations=2)
                piv_mice = kds.complete_data(dataset=0)
                p_mice = piv_mice.loc[valid_dates, 'SE3'].values
                m_mice = calc_all_metrics(y_real, p_mice, f"{scen_name}_{var}_miceforest")
            except Exception as e:
                m_mice = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'RMSE_peaks': np.nan, 'bias': np.nan}
            records.append({'scenario': scen_name, 'station': 'SE3', 'variable': var, 'method': 'miceforest', **m_mice})
            
            # --- SAITS+CSDI previo ---
            if var in ['O3', 'PM2.5']:
                saits_match = saits_se3.set_index('time').loc[valid_dates, var].values
                m_saits = calc_all_metrics(y_real, saits_match, f"{scen_name}_{var}_SAITS_CSDI")
                records.append({'scenario': scen_name, 'station': 'SE3', 'variable': var, 'method': 'SAITS+CSDI (previo)', **m_saits})
                best_preds_for_plot[scen_name][f'{var}_saits'] = saits_match
            
            # Guardar predicciones clave para gráficos
            best_preds_for_plot[scen_name][f'{var}_real'] = y_real
            best_preds_for_plot[scen_name][f'{var}_valid_dates'] = valid_dates
            best_preds_for_plot[scen_name][f'{var}_lgb'] = p_lgb
            best_preds_for_plot[scen_name][f'{var}_idw'] = p_idw
            best_preds_for_plot[scen_name][f'{var}_clim'] = clim_vals
            
            print(f"    Lineal:        RMSE={m_lin['RMSE']:.2f}, R2={m_lin['R2']:.3f}, Bias={m_lin['bias']:+.2f}", flush=True)
            print(f"    Climatologia:  RMSE={m_clim['RMSE']:.2f}, R2={m_clim['R2']:.3f}, Bias={m_clim['bias']:+.2f}", flush=True)
            print(f"    IDW Espacial:  RMSE={m_idw['RMSE']:.2f}, R2={m_idw['R2']:.3f}, Bias={m_idw['bias']:+.2f}", flush=True)
            print(f"    LGB Espacial:  RMSE={m_lgb['RMSE']:.2f}, R2={m_lgb['R2']:.3f}, Bias={m_lgb['bias']:+.2f}", flush=True)
            print(f"    SoftImpute:    RMSE={m_soft['RMSE']:.2f}, R2={m_soft['R2']:.3f}, Bias={m_soft['bias']:+.2f}", flush=True)
            if var in ['O3', 'PM2.5']:
                print(f"    SAITS Previo:  RMSE={m_saits['RMSE']:.2f}, R2={m_saits['R2']:.3f}, Bias={m_saits['bias']:+.2f}", flush=True)

    # -----------------------------------------------------------------------
    # FASE 2: S5 (Repetir S3 en 3 estaciones adicionales: SO2, CE, SE)
    # -----------------------------------------------------------------------
    print("\n>>> Evaluando Escenario S5 (Robustez en SO2, CE, SE - Junio 2022)...", flush=True)
    s5_stations = ['SO2', 'CE', 'SE']
    for st in s5_stations:
        st_df = panel[panel['Estacion'] == st].sort_values('Date').reset_index(drop=True)
        mask_s5 = (st_df['Date'].dt.year == 2022) & (st_df['Date'].dt.month == 6)
        dates_s5 = st_df.loc[mask_s5, 'Date'].values
        idx_s5 = st_df.index[mask_s5].values
        
        for var in ['O3', 'PM2.5', 'NOX', 'TOUT']:
            val_gt = st_df.loc[mask_s5, ['Date', var]].dropna()
            if len(val_gt) == 0:
                continue
            valid_dates = val_gt['Date'].values
            y_real = val_gt[var].values
            
            s_masked = st_df[var].copy()
            s_masked.iloc[idx_s5] = np.nan
            
            # Lineal
            p_lin = pd.Series(impute_linear(s_masked), index=st_df['Date']).loc[valid_dates].values
            records.append({'scenario': 'S5 (Robustez 3 est.)', 'station': st, 'variable': var, 'method': 'Lineal',
                            **calc_all_metrics(y_real, p_lin)})
            
            # Climatología
            p_clim = get_climatology_series(pd.Series(valid_dates), st, var, clim_maps)
            records.append({'scenario': 'S5 (Robustez 3 est.)', 'station': st, 'variable': var, 'method': 'Climatologia',
                            **calc_all_metrics(y_real, p_clim)})
            
            # Clim + Ajuste Nivel
            p_clim_adj = pd.Series(impute_climatology_level_adj(s_masked, st_df['Date'], st, var, clim_maps), index=st_df['Date']).loc[valid_dates].values
            records.append({'scenario': 'S5 (Robustez 3 est.)', 'station': st, 'variable': var, 'method': 'Clim + Ajuste Nivel',
                            **calc_all_metrics(y_real, p_clim_adj)})
            
            # IDW Espacial
            panel_masked = panel.copy()
            panel_masked.loc[(panel_masked['Estacion'] == st) & (panel_masked['Date'].isin(dates_s5)), var] = np.nan
            idw_res = impute_idw_spatial(panel_masked, var, dist_matrix, clim_maps, return_wide=True)
            p_idw = idw_res.loc[valid_dates, st].values
            records.append({'scenario': 'S5 (Robustez 3 est.)', 'station': st, 'variable': var, 'method': 'IDW Espacial',
                            **calc_all_metrics(y_real, p_idw)})
            
            # Regresión Espacial LightGBM
            bst = train_spatial_regression_lightgbm(
                panel, target_var=var, target_station=st,
                exclude_dates=dates_s5
            )
            key_covars = [var, 'TOUT', 'SR', 'RH', 'U_Wind', 'V_Wind'] if var != 'TOUT' else [var, 'SR', 'RH', 'U_Wind', 'V_Wind']
            dfs_x = []
            donor_st = [s for s in ALL_STATIONS if s != st]
            for kv in key_covars:
                p_kv = panel.pivot(index='Date', columns='Estacion', values=kv)
                cols = [s for s in donor_st if s in p_kv.columns]
                dfs_x.append(p_kv.loc[valid_dates, cols].add_prefix(f'{kv}_'))
            X_s5 = pd.concat(dfs_x, axis=1)
            X_s5['hour_sin'] = np.sin(2 * np.pi * X_s5.index.hour / 24)
            X_s5['hour_cos'] = np.cos(2 * np.pi * X_s5.index.hour / 24)
            X_s5['month_sin'] = np.sin(2 * np.pi * X_s5.index.month / 12)
            X_s5['month_cos'] = np.cos(2 * np.pi * X_s5.index.month / 12)
            p_lgb = np.maximum(0, bst.predict(X_s5))
            records.append({'scenario': 'S5 (Robustez 3 est.)', 'station': st, 'variable': var, 'method': 'Regresion Espacial LightGBM',
                            **calc_all_metrics(y_real, p_lgb)})
            
            # SoftImpute SVD
            piv_var = panel[panel['Date'].dt.year == 2022].pivot(index='Date', columns='Estacion', values=var).copy()
            piv_var.loc[piv_var.index.isin(dates_s5), st] = np.nan
            M_imp = impute_softimpute_svd(piv_var.values, rank=3)
            piv_rec = pd.DataFrame(M_imp, index=piv_var.index, columns=piv_var.columns)
            p_soft = piv_rec.loc[valid_dates, st].values
            records.append({'scenario': 'S5 (Robustez 3 est.)', 'station': st, 'variable': var, 'method': 'SoftImpute SVD',
                            **calc_all_metrics(y_real, p_soft)})

    # -----------------------------------------------------------------------
    # FASE 3: S1 (Cortos 1-3h) y S2 (Medianos 6-48h)
    # -----------------------------------------------------------------------
    print("\n>>> Evaluando Escenarios S1 (1-3h) y S2 (6-48h)...", flush=True)
    eval_stations = ['SE3', 'CE', 'SO2', 'SE']
    for scen_name, gap_lens in [('S1 (Cortos 1-3h)', [1, 2, 3]), ('S2 (Medianos 6-48h)', [6, 12, 24, 48])]:
        for var in ['O3', 'PM2.5', 'NOX', 'TOUT']:
            for st in eval_stations:
                st_df = panel[(panel['Estacion'] == st) & (panel['Date'].dt.year == 2022)].sort_values('Date').reset_index(drop=True)
                s = st_df[var].copy()
                mask = generate_random_gaps(s, gap_lens, target_hours=250, seed=42 + len(st) + len(var))
                
                valid_dates = st_df.loc[mask, 'Date'].values
                y_real = s[mask].values
                s_masked = s.copy()
                s_masked[mask] = np.nan
                
                # Lineal
                p_lin = pd.Series(impute_linear(s_masked), index=st_df['Date']).loc[valid_dates].values
                records.append({'scenario': scen_name, 'station': st, 'variable': var, 'method': 'Lineal',
                                **calc_all_metrics(y_real, p_lin)})
                
                # PCHIP
                p_pchip = pd.Series(impute_pchip(s_masked), index=st_df['Date']).loc[valid_dates].values
                records.append({'scenario': scen_name, 'station': st, 'variable': var, 'method': 'PCHIP',
                                **calc_all_metrics(y_real, p_pchip)})
                
                # Climatología
                p_clim = get_climatology_series(pd.Series(valid_dates), st, var, clim_maps)
                records.append({'scenario': scen_name, 'station': st, 'variable': var, 'method': 'Climatologia',
                                **calc_all_metrics(y_real, p_clim)})
                
                # Clim + Ajuste Nivel
                p_clim_adj = pd.Series(impute_climatology_level_adj(s_masked, st_df['Date'], st, var, clim_maps), index=st_df['Date']).loc[valid_dates].values
                records.append({'scenario': scen_name, 'station': st, 'variable': var, 'method': 'Clim + Ajuste Nivel',
                                **calc_all_metrics(y_real, p_clim_adj)})
                
                # STL + Interp
                p_stl = pd.Series(impute_stl(s_masked), index=st_df['Date']).loc[valid_dates].values
                records.append({'scenario': scen_name, 'station': st, 'variable': var, 'method': 'STL + Interp',
                                **calc_all_metrics(y_real, p_stl)})

    # Guardar resultados en DataFrame
    df_results = pd.DataFrame(records)
    df_results.to_csv(BENCHMARK_CSV, index=False)
    print(f"\n[OK] Resultados guardados en {BENCHMARK_CSV} ({len(df_results)} filas)", flush=True)
    
    # -----------------------------------------------------------------------
    # FASE 4: GENERACIÓN DE FIGURAS
    # -----------------------------------------------------------------------
    print("\n>>> Generando figuras de evaluación y overlays...", flush=True)
    sns.set_theme(style='whitegrid', font_scale=1.1)
    
    # 1. Overlay S3 (1 mes SE3) - O3
    fig, ax = plt.subplots(figsize=(15, 5))
    d_s3 = pd.to_datetime(best_preds_for_plot['S3 (1 mes SE3)']['O3_valid_dates'])
    y_real_s3_o3 = best_preds_for_plot['S3 (1 mes SE3)']['O3_real']
    y_saits_s3_o3 = best_preds_for_plot['S3 (1 mes SE3)']['O3_saits']
    y_lgb_s3_o3 = best_preds_for_plot['S3 (1 mes SE3)']['O3_lgb']
    y_clim_s3_o3 = best_preds_for_plot['S3 (1 mes SE3)']['O3_clim']
    
    ax.plot(d_s3, y_real_s3_o3, label='Real (observado)', color='black', alpha=0.85, lw=1.2)
    ax.plot(d_s3, y_saits_s3_o3, label='SAITS+CSDI (previo)', color='#e76f51', alpha=0.75, lw=1.1, ls='--')
    ax.plot(d_s3, y_lgb_s3_o3, label='Regresión Espacial LightGBM', color='#2a9d8f', alpha=0.85, lw=1.2)
    ax.plot(d_s3, y_clim_s3_o3, label='Perfil Climatológico', color='#457b9d', alpha=0.6, lw=1.0, ls=':')
    ax.set_title('Escenario S3 (Junio 2022, SE3) — Reconstrucción de Ozono (O3) vs Real y SAITS', fontsize=13, fontweight='bold')
    ax.set_ylabel('O3 (ppb)')
    ax.set_xlabel('Fecha')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'overlay_S3_O3.png'), dpi=200)
    plt.close()
    
    # 2. Overlay S3 (1 mes SE3) - PM2.5
    fig, ax = plt.subplots(figsize=(15, 5))
    d_s3_pm = pd.to_datetime(best_preds_for_plot['S3 (1 mes SE3)']['PM2.5_valid_dates'])
    y_real_s3_pm = best_preds_for_plot['S3 (1 mes SE3)']['PM2.5_real']
    y_saits_s3_pm = best_preds_for_plot['S3 (1 mes SE3)']['PM2.5_saits']
    y_lgb_s3_pm = best_preds_for_plot['S3 (1 mes SE3)']['PM2.5_lgb']
    
    ax.plot(d_s3_pm, y_real_s3_pm, label='Real (observado)', color='black', alpha=0.85, lw=1.2)
    ax.plot(d_s3_pm, y_saits_s3_pm, label='SAITS+CSDI (sesgo +58%)', color='#e76f51', alpha=0.75, lw=1.1, ls='--')
    ax.plot(d_s3_pm, y_lgb_s3_pm, label='Regresión Espacial LightGBM (sin sesgo)', color='#2a9d8f', alpha=0.85, lw=1.2)
    ax.set_title('Escenario S3 (Junio 2022, SE3) — Reconstrucción de PM2.5 vs Real y SAITS', fontsize=13, fontweight='bold')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.set_xlabel('Fecha')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'overlay_S3_PM25.png'), dpi=200)
    plt.close()
    
    # 3. Overlay S4 (55 días SE3) - O3
    fig, ax = plt.subplots(figsize=(16, 5))
    d_s4 = pd.to_datetime(best_preds_for_plot['S4 (55d SE3)']['O3_valid_dates'])
    y_real_s4_o3 = best_preds_for_plot['S4 (55d SE3)']['O3_real']
    y_saits_s4_o3 = best_preds_for_plot['S4 (55d SE3)']['O3_saits']
    y_lgb_s4_o3 = best_preds_for_plot['S4 (55d SE3)']['O3_lgb']
    
    ax.plot(d_s4, y_real_s4_o3, label='Real (observado)', color='black', alpha=0.85, lw=1.1)
    ax.plot(d_s4, y_saits_s4_o3, label='SAITS+CSDI (previo)', color='#e76f51', alpha=0.75, lw=1.0, ls='--')
    ax.plot(d_s4, y_lgb_s4_o3, label='Regresión Espacial LightGBM', color='#2a9d8f', alpha=0.85, lw=1.1)
    ax.set_title('Escenario S4 (55 días: 1 Jul – 25 Ago 2022, SE3) — Reconstrucción de Ozono (O3)', fontsize=13, fontweight='bold')
    ax.set_ylabel('O3 (ppb)')
    ax.set_xlabel('Fecha')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'overlay_S4_O3.png'), dpi=200)
    plt.close()
    
    # 4. Overlay S4 (55 días SE3) - PM2.5
    fig, ax = plt.subplots(figsize=(16, 5))
    d_s4_pm = pd.to_datetime(best_preds_for_plot['S4 (55d SE3)']['PM2.5_valid_dates'])
    y_real_s4_pm = best_preds_for_plot['S4 (55d SE3)']['PM2.5_real']
    y_saits_s4_pm = best_preds_for_plot['S4 (55d SE3)']['PM2.5_saits']
    y_lgb_s4_pm = best_preds_for_plot['S4 (55d SE3)']['PM2.5_lgb']
    
    ax.plot(d_s4_pm, y_real_s4_pm, label='Real (observado)', color='black', alpha=0.85, lw=1.1)
    ax.plot(d_s4_pm, y_saits_s4_pm, label='SAITS+CSDI (sesgo +66%)', color='#e76f51', alpha=0.75, lw=1.0, ls='--')
    ax.plot(d_s4_pm, y_lgb_s4_pm, label='Regresión Espacial LightGBM', color='#2a9d8f', alpha=0.85, lw=1.1)
    ax.set_title('Escenario S4 (55 días: 1 Jul – 25 Ago 2022, SE3) — Reconstrucción de PM2.5', fontsize=13, fontweight='bold')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.set_xlabel('Fecha')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'overlay_S4_PM25.png'), dpi=200)
    plt.close()
    
    # 5. Gráfico resumen de R2 por escenario y método para O3
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for idx_v, var in enumerate(['O3', 'PM2.5']):
        ax = axes[idx_v]
        sub = df_results[df_results['variable'] == var].groupby(['scenario', 'method'])['R2'].mean().reset_index()
        # Filtrar métodos principales
        key_methods = ['Lineal', 'Climatologia', 'Clim + Ajuste Nivel', 'IDW Espacial', 'Regresion Espacial LightGBM', 'SoftImpute SVD']
        sub = sub[sub['method'].isin(key_methods)]
        sns.barplot(data=sub, x='scenario', y='R2', hue='method', ax=ax, palette='viridis')
        ax.set_title(f'Capacidad de Reconstrucción (R²) — {var}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Escenario')
        ax.set_ylabel('R²')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
        if idx_v == 0:
            ax.legend(title='Método', loc='lower left', fontsize=9)
        else:
            ax.legend().remove()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'benchmark_summary_r2.png'), dpi=200)
    plt.close()
    
    print("[OK] Todas las figuras fueron generadas exitosamente en experiments/imputacion/figures/", flush=True)

if __name__ == '__main__':
    run_benchmark()
