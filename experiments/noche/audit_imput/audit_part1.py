"""
experiments/noche/audit_imput/audit_part1.py
===========================================
Auditoría exhaustiva de fuga de información (Leakage Audit):
Parte 1a: Auditoría de features_planA.py y run_ablations.py
Parte 1b: Auditoría de planC_meteo (train_models.py, meteo_features.py, Open-Meteo data)
"""
import sys
import os
import time
import json
import numpy as np
import pandas as pd

# Raiz del proyecto
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import load_panel, MODEL_STATIONS, ALL_STATIONS, HORIZONS, add_targets
from planA_quickwins.features_planA import build_planA_features, compute_spatial_matrices

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def audit_part1a():
    print("=" * 70)
    print("PARTE 1a: AUDITORIA DE FUGA EN PLAN A (features_planA.py)")
    print("=" * 70)

    # 1. Cargar panel
    t0 = time.time()
    panel = load_panel(use_cache=True)
    print(f"Panel cargado: {len(panel)} filas, {len(panel.columns)} columnas en {time.time()-t0:.2f}s")
    
    # Comprobar integridad de la grilla temporal
    print("\n--- [1] Verificación de la grilla temporal y huecos ---")
    st_counts = panel.groupby('Estacion')['Date'].agg(['count', 'min', 'max'])
    print(st_counts.head(5))
    expected_hours = len(pd.date_range('2020-01-01 00:00', '2025-12-31 23:00', freq='h'))
    all_exact = (st_counts['count'] == expected_hours).all()
    print(f"¿Todas las estaciones tienen exactamente {expected_hours} horas consecutivas? {all_exact}")
    
    # Comprobar si hay fechas duplicadas o faltantes
    is_monotonic = True
    has_gaps = False
    for st, g in panel.groupby('Estacion'):
        d = g['Date']
        if not d.is_monotonic_increasing:
            is_monotonic = False
        diffs = d.diff().dropna()
        if (diffs != pd.Timedelta(hours=1)).any():
            has_gaps = True
    print(f"¿Fechas estrictamente crecientes en cada estación? {is_monotonic}")
    print(f"¿Hay saltos temporales (filas faltantes en lugar de NaNs)? {has_gaps}")

    # 2. Construir features V4
    print("\n--- [2] Construyendo features V4 para análisis ---")
    t0 = time.time()
    df_v4, feat_cols = build_planA_features(panel, variant='V4')
    print(f"V4 construido en {time.time()-t0:.2f}s: {df_v4.shape[0]} filas, {len(feat_cols)} features.")

    # 3. Comprobar alineación de targets por estación
    print("\n--- [3] Verificación de alineación de targets ---")
    cross_station_leakage = False
    for h in [1, 4, 8, 12, 24, 48]:
        col = f'Target_{h}h'
        # En cada transicion de estacion, los ultimos h valores deben ser NaN
        for st, g in df_v4.groupby('Estacion', observed=False):
            tail_vals = g[col].iloc[-h:]
            if not tail_vals.isna().all():
                print(f"ALERTA: Target_{h}h en estación {st} NO tiene NaNs en las últimas {h} filas!")
                cross_station_leakage = True
            # Comprobar que para t, Target_h coincide con O3(t+h)
            o3_vals = g['O3'].values
            target_vals = g[col].values
            diff = np.nanmax(np.abs(target_vals[:-h] - o3_vals[h:]))
            if diff > 1e-6:
                print(f"ALERTA: Desalineación en Target_{h}h estación {st}: max diff = {diff}")
                cross_station_leakage = True
    print(f"¿Hay fuga de target entre estaciones o desalineación? {cross_station_leakage}")

    # 4. Prueba de cálculo manual para filas aleatorias
    print("\n--- [4] Recálculo manual de features para filas aleatorias ---")
    np.random.seed(12345)
    sample_indices = np.random.choice(len(df_v4), size=10, replace=False)
    manual_checks = []
    
    # Precomputar panel como diccionario indexado por (Estacion, Date) para lookups rapidos
    p_lookup = panel.set_index(['Estacion', 'Date'])
    
    for idx in sample_indices:
        row = df_v4.iloc[idx]
        st = row['Estacion']
        dt = row['Date']
        
        # Check O3_lag_0, O3_lag_1, O3_lag_24
        val_t0 = row['O3_lag_0']
        val_l1 = row['O3_lag_1']
        val_l24 = row['O3_lag_24']
        
        raw_t0 = p_lookup.loc[(st, dt), 'O3']
        raw_l1 = p_lookup.loc[(st, dt - pd.Timedelta(hours=1)), 'O3'] if (st, dt - pd.Timedelta(hours=1)) in p_lookup.index else np.nan
        raw_l24 = p_lookup.loc[(st, dt - pd.Timedelta(hours=24)), 'O3'] if (st, dt - pd.Timedelta(hours=24)) in p_lookup.index else np.nan
        
        diff_t0 = abs(val_t0 - raw_t0) if np.isfinite(val_t0) and np.isfinite(raw_t0) else 0.0
        diff_l1 = abs(val_l1 - raw_l1) if np.isfinite(val_l1) and np.isfinite(raw_l1) else 0.0
        diff_l24 = abs(val_l24 - raw_l24) if np.isfinite(val_l24) and np.isfinite(raw_l24) else 0.0
        
        # Check rolling 6h O3 mean t0
        raw_6h = [p_lookup.loc[(st, dt - pd.Timedelta(hours=k)), 'O3'] for k in range(6) if (st, dt - pd.Timedelta(hours=k)) in p_lookup.index]
        valid_raw_6h = [x for x in raw_6h if np.isfinite(x)]
        manual_mean_6h = np.mean(valid_raw_6h) if len(valid_raw_6h) >= 2 else np.nan
        feat_mean_6h = row['O3_mean_6h_t0']
        diff_mean_6h = abs(feat_mean_6h - manual_mean_6h) if np.isfinite(feat_mean_6h) and np.isfinite(manual_mean_6h) else 0.0
        
        # Check spatial others mean lag 0
        other_st = [s for s in ALL_STATIONS if s != st]
        other_o3 = [p_lookup.loc[(s, dt), 'O3'] for s in other_st if (s, dt) in p_lookup.index]
        valid_other_o3 = [x for x in other_o3 if np.isfinite(x)]
        manual_others_mean = np.mean(valid_other_o3) if len(valid_other_o3) > 0 else np.nan
        feat_others_mean = row['O3_others_mean_lag_0']
        diff_others_mean = abs(feat_others_mean - manual_others_mean) if np.isfinite(feat_others_mean) and np.isfinite(manual_others_mean) else 0.0

        manual_checks.append({
            'Estacion': st, 'Date': str(dt),
            'diff_t0': diff_t0, 'diff_l1': diff_l1, 'diff_l24': diff_l24,
            'diff_mean_6h': diff_mean_6h, 'diff_others_mean': diff_others_mean
        })
    df_checks = pd.DataFrame(manual_checks)
    print("Resultados de recálculo manual en 10 muestras aleatorias:")
    print(df_checks[['Estacion', 'Date', 'diff_t0', 'diff_l1', 'diff_l24', 'diff_mean_6h', 'diff_others_mean']])
    max_manual_diff = df_checks[['diff_t0', 'diff_l1', 'diff_l24', 'diff_mean_6h', 'diff_others_mean']].max().max()
    print(f"Diferencia máxima observada en recálculo manual: {max_manual_diff:.8f}")

    # 5. Prueba de Permutación / Alteración del Futuro
    print("\n--- [5] Prueba de Alteración del Futuro (Future Perturbation Test) ---")
    # Elegir 3 puntos de corte t* aleatorios
    cutoffs = [
        pd.Timestamp('2024-06-15 12:00:00'),
        pd.Timestamp('2024-11-01 00:00:00'),
        pd.Timestamp('2025-03-20 18:00:00'),
    ]
    
    perturbation_results = []
    
    for t_star in cutoffs:
        print(f"\nProbando punto de corte t* = {t_star}...")
        panel_corrupted = panel.copy()
        
        # Corromper brutalmente todo dato con Date > t_star
        future_mask = panel_corrupted['Date'] > t_star
        for col in ['O3', 'NOX', 'NO', 'NO2', 'CO', 'SO2', 'TOUT', 'SR', 'RH', 'PRS', 'RAINF', 'PM10', 'PM2.5', 'WSR', 'WDR', 'U_Wind', 'V_Wind']:
            if col in panel_corrupted.columns:
                n_mut = future_mask.sum()
                vals = np.random.uniform(500, 1000, size=n_mut).astype(panel_corrupted[col].dtype)
                panel_corrupted.loc[future_mask, col] = vals
                
        # Reconstruir features V4 con el panel corrompido
        df_v4_corr, _ = build_planA_features(panel_corrupted, variant='V4')
        
        # Comparar features para filas con Date <= t_star
        past_mask = df_v4['Date'] <= t_star
        sub_orig = df_v4.loc[past_mask, feat_cols]
        sub_corr = df_v4_corr.loc[past_mask, feat_cols]
        
        # Evaluar diferencias numéricas
        diffs = []
        leaking_cols = []
        for c in feat_cols:
            if c == 'Estacion':
                continue
            s1 = sub_orig[c].values.astype(float)
            s2 = sub_corr[c].values.astype(float)
            # Donde ambos son finitos
            both_finite = np.isfinite(s1) & np.isfinite(s2)
            both_nan = np.isnan(s1) & np.isnan(s2)
            nan_mismatch = (~both_finite) & (~both_nan)
            if nan_mismatch.any():
                leaking_cols.append(c)
                print(f"  FUGA DETECTADA (NaN mismatch) en feature {c} con corte {t_star}!")
                continue
            if both_finite.any():
                max_d = np.max(np.abs(s1[both_finite] - s2[both_finite]))
                if max_d > 1e-6:
                    leaking_cols.append(c)
                    print(f"  FUGA DETECTADA (diferencia numérica {max_d}) en feature {c} con corte {t_star}!")
                diffs.append(max_d)
        
        max_past_diff = max(diffs) if diffs else 0.0
        print(f"  Corte {t_star}: Features evaluadas={len(feat_cols)}, Columnas con fuga={len(leaking_cols)}, Max diff en t <= t* = {max_past_diff:.8e}")
        perturbation_results.append({
            'cutoff': str(t_star),
            'n_features': len(feat_cols),
            'leaking_cols': leaking_cols,
            'max_past_diff': float(max_past_diff)
        })
        
    return df_checks, perturbation_results


def audit_part1b():
    print("\n" + "=" * 70)
    print("PARTE 1b: AUDITORIA DE METEOROLOGIA EXTERNA (planC_meteo)")
    print("=" * 70)
    
    data_dir = os.path.join(ROOT, 'experiments', 'planC_meteo', 'data')
    
    # 1. Cargar datos de pronostico historico (forecast) y reanalisis
    fcst_files = [os.path.join(data_dir, f'forecast_{s}.parquet') for s in MODEL_STATIONS]
    rean_files = [os.path.join(data_dir, f'reanalysis_{s}.parquet') for s in MODEL_STATIONS]
    
    df_fcst = pd.concat([pd.read_parquet(f) for f in fcst_files], ignore_index=True)
    df_rean = pd.concat([pd.read_parquet(f) for f in rean_files], ignore_index=True)
    df_fcst['Date'] = pd.to_datetime(df_fcst['Date'])
    df_rean['Date'] = pd.to_datetime(df_rean['Date'])
    
    print(f"Forecast cargado: {len(df_fcst)} filas. Columnas: {list(df_fcst.columns)}")
    print(f"Reanalysis cargado: {len(df_rean)} filas. Columnas: {list(df_rean.columns)}")
    
    # 2. Análisis de disponibilidad por año de previous_day1 y previous_day2
    print("\n--- [1] Disponibilidad de previous_day1 y previous_day2 por año ---")
    df_fcst['year'] = df_fcst['Date'].dt.year
    avail_summary = df_fcst.groupby('year').agg({
        'temperature_2m': lambda x: f"{x.notna().mean()*100:.1f}%",
        'temperature_2m_previous_day1': lambda x: f"{x.notna().mean()*100:.1f}%",
        'temperature_2m_previous_day2': lambda x: f"{x.notna().mean()*100:.1f}%",
        'shortwave_radiation_previous_day1': lambda x: f"{x.notna().mean()*100:.1f}%",
        'boundary_layer_height': lambda x: f"{x.notna().mean()*100:.1f}%",
    })
    print(avail_summary)
    
    # 3. Comparación previous_day1 y previous_day2 vs Seamless (2024-2025)
    print("\n--- [2] Error de NWP: previous_day1 y previous_day2 vs Seamless (2024-2025) ---")
    eval_fcst = df_fcst[df_fcst['year'].isin([2024, 2025])].copy()
    
    vars_compare = ['temperature_2m', 'shortwave_radiation', 'relative_humidity_2m', 'wind_speed_10m']
    fcst_vs_seamless = []
    
    for v in vars_compare:
        for p in ['previous_day1', 'previous_day2']:
            col_p = f"{v}_{p}"
            if col_p in eval_fcst.columns and v in eval_fcst.columns:
                diff = eval_fcst[col_p] - eval_fcst[v]
                valid = diff.dropna()
                rmse = np.sqrt(np.mean(valid ** 2))
                mae = np.mean(np.abs(valid))
                bias = np.mean(valid)
                fcst_vs_seamless.append({
                    'variable': v,
                    'run': p,
                    'n': len(valid),
                    'coverage': len(valid) / len(eval_fcst),
                    'RMSE': float(rmse),
                    'MAE': float(mae),
                    'bias': float(bias)
                })
    df_fcst_vs_seamless = pd.DataFrame(fcst_vs_seamless)
    print(df_fcst_vs_seamless.to_string(index=False))

    # 4. Comparación Open-Meteo vs Observaciones SIMA (2024-2025)
    print("\n--- [3] Error de Open-Meteo vs Observaciones SIMA (2024-2025) ---")
    panel = load_panel(stations=MODEL_STATIONS, use_cache=True)
    panel_eval = panel[panel['Date'].dt.year.isin([2024, 2025])].copy()
    panel_eval['WSR_calc'] = np.sqrt(panel_eval['U_Wind']**2 + panel_eval['V_Wind']**2)
    
    # Merge con forecast
    m_eval = pd.merge(panel_eval, eval_fcst, on=['Estacion', 'Date'], how='inner')
    
    # Mappings: Open-Meteo -> SIMA
    var_pairs = [
        ('temperature_2m', 'TOUT', 'Temperatura (°C)'),
        ('temperature_2m_previous_day1', 'TOUT', 'Temp prev_day1 (°C)'),
        ('temperature_2m_previous_day2', 'TOUT', 'Temp prev_day2 (°C)'),
        ('shortwave_radiation', 'SR', 'Radiación Solar (W/m2 o kW/m2)'),
        ('shortwave_radiation_previous_day1', 'SR', 'SR prev_day1'),
        ('shortwave_radiation_previous_day2', 'SR', 'SR prev_day2'),
        ('relative_humidity_2m', 'RH', 'Humedad Relativa (%)'),
        ('relative_humidity_2m_previous_day1', 'RH', 'RH prev_day1 (%)'),
        ('relative_humidity_2m_previous_day2', 'RH', 'RH prev_day2 (%)'),
        ('wind_speed_10m', 'WSR', 'Velocidad de viento (m/s)'),
        ('wind_speed_10m_previous_day1', 'WSR', 'WS prev_day1 (m/s)'),
        ('wind_speed_10m_previous_day2', 'WSR', 'WS prev_day2 (m/s)'),
    ]
    
    sima_eval_results = []
    for om_col, sima_col, desc in var_pairs:
        if om_col in m_eval.columns and sima_col in m_eval.columns:
            y_om = m_eval[om_col].values.astype(float)
            y_si = m_eval[sima_col].values.astype(float)
            
            # SIMA SR unidades check:
            # En SIMA, a veces SR está en kW/m2 (0-1.2) o W/m2 (0-1200)
            if sima_col == 'SR':
                # Si SIMA SR max es < 5, está en kW/m2 y Open-Meteo en W/m2
                max_sima_sr = np.nanquantile(y_si, 0.99)
                if max_sima_sr < 5.0:
                    y_si = y_si * 1000.0 # convertir kW/m2 a W/m2 para comparar
            
            ok = np.isfinite(y_om) & np.isfinite(y_si)
            if ok.sum() > 0:
                diff = y_om[ok] - y_si[ok]
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                mae = float(np.mean(np.abs(diff)))
                bias = float(np.mean(diff))
                corr = float(np.corrcoef(y_om[ok], y_si[ok])[0, 1])
                sima_eval_results.append({
                    'open_meteo_var': om_col,
                    'sima_var': sima_col,
                    'n': int(ok.sum()),
                    'RMSE': rmse,
                    'MAE': mae,
                    'bias': bias,
                    'corr': corr
                })
    df_sima_eval = pd.DataFrame(sima_eval_results)
    print(df_sima_eval.to_string(index=False))

    # 5. Verificación de Zona Horaria mediante Correlación Cruzada
    print("\n--- [4] Verificación de Zona Horaria (Correlación Cruzada con SIMA) ---")
    # Para radiación solar (SR) y temperatura (TOUT), la correlación con la hora solar es muy nítida
    cross_corr_results = {}
    
    lags = list(range(-6, 7)) # -6h a +6h
    for var_om, var_si in [('shortwave_radiation', 'SR'), ('temperature_2m', 'TOUT')]:
        corrs_by_lag = {}
        for lag in lags:
            # Shift Open-Meteo by lag: y_om(t + lag) vs y_si(t)
            # Si lag = +1, om está desplazado 1 hora al futuro
            if var_si == 'SR':
                y_si = m_eval[var_si].values.copy()
                if np.nanquantile(y_si, 0.99) < 5.0:
                    y_si = y_si * 1000.0
            else:
                y_si = m_eval[var_si].values
                
            y_om_shifted = pd.Series(m_eval[var_om]).shift(lag).values
            ok = np.isfinite(y_om_shifted) & np.isfinite(y_si)
            if ok.sum() > 1000:
                c = float(np.corrcoef(y_om_shifted[ok], y_si[ok])[0, 1])
            else:
                c = np.nan
            corrs_by_lag[lag] = c
        cross_corr_results[f"{var_om}_vs_{var_si}"] = corrs_by_lag
        print(f"Correlación cruzada ({var_om} vs {var_si}):")
        for lag in lags:
            bar = "*" * int(max(0, corrs_by_lag[lag]) * 30)
            mark = " <-- PICO MÁXIMO" if corrs_by_lag[lag] == max(corrs_by_lag.values()) else ""
            print(f"  Lag {lag:+2d}h: corr = {corrs_by_lag[lag]:.4f} | {bar}{mark}")
            
    # Verificar también si hay desalineación estacional (horario de verano antes de 2023)
    print("\n--- Comprobando correlación cruzada por año (2020-2022 vs 2023-2025) ---")
    m_all = pd.merge(panel, pd.concat([df_fcst, df_rean[['Date', 'Estacion']]], ignore_index=True).drop_duplicates(['Date', 'Estacion']), on=['Date', 'Estacion'], how='inner')
    # Use reanalysis for 2020-2025 full coverage
    m_rean = pd.merge(panel, df_rean, on=['Date', 'Estacion'], how='inner')
    
    yearly_lags = {}
    for yr in [2021, 2022, 2023, 2024, 2025]:
        sub_yr = m_rean[m_rean['Date'].dt.year == yr]
        y_si = sub_yr['SR'].values.copy()
        if np.nanquantile(y_si, 0.99) < 5.0:
            y_si = y_si * 1000.0
        lag_corrs = {}
        for lag in range(-3, 4):
            y_om_shifted = pd.Series(sub_yr['shortwave_radiation']).shift(lag).values
            ok = np.isfinite(y_om_shifted) & np.isfinite(y_si)
            c = float(np.corrcoef(y_om_shifted[ok], y_si[ok])[0, 1]) if ok.sum() > 500 else np.nan
            lag_corrs[lag] = c
        best_lag = max(lag_corrs, key=lag_corrs.get)
        yearly_lags[yr] = {'best_lag': best_lag, 'max_corr': lag_corrs[best_lag], 'lag0_corr': lag_corrs[0]}
        print(f"  Año {yr}: Mejor lag = {best_lag:+d}h (corr={lag_corrs[best_lag]:.4f}), Lag 0h corr = {lag_corrs[0]:.4f}")

    return avail_summary, df_fcst_vs_seamless, df_sima_eval, cross_corr_results, yearly_lags


if __name__ == '__main__':
    t_start = time.time()
    df_checks, pert_results = audit_part1a()
    avail_sum, df_fcst_vs_seamless, df_sima_eval, cross_corrs, yearly_lags = audit_part1b()
    
    # Guardar métricas en JSON para armar AUDIT.md
    out_metrics = {
        'manual_checks': df_checks.to_dict(orient='records'),
        'perturbation_results': pert_results,
        'fcst_vs_seamless': df_fcst_vs_seamless.to_dict(orient='records'),
        'sima_eval': df_sima_eval.to_dict(orient='records'),
        'cross_corrs': cross_corrs,
        'yearly_lags': yearly_lags
    }
    with open(os.path.join(OUTPUT_DIR, 'audit_metrics.json'), 'w') as f:
        json.dump(out_metrics, f, indent=2)
    print(f"\nAuditoría Parte 1 finalizada en {time.time()-t_start:.1f}s.")
