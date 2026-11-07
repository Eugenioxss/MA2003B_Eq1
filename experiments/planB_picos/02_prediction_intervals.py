"""
experiments/planB_picos/02_prediction_intervals.py
==================================================
Genera y evalúa intervalos de predicción con cobertura nominal del 80% (alpha=0.20)
en el conjunto de prueba (2025).

Métodos comparados:
 1. Conformal Split (Global): Residuos absolutos de validación (2024), cuantil empírico global.
 2. Conformal Split Condicional por Hora: Residuos de 2024 condicionales a la hora del día (0-23h).
 3. Regresión Cuantílica LightGBM (QR): Modelos directos para q0.10, q0.50 y q0.90.
 4. Conformalized Quantile Regression (CQR): Calibración conforme sobre los cuantiles QR con 2024.

Métricas evaluadas en 2025:
 - Cobertura global (nominal 80%)
 - Ancho medio de banda (ppb)
 - Ancho mediano (ppb)
 - Cobertura en horas pico (y_test >= p95)
 - Cobertura en horas diurnas (10h - 18h)
 - Winkler Score (penalización balanceada de ancho y violación de cobertura)

Guarda métricas en experiments/planB_picos/metrics_intervals.csv
y predicciones de intervalos en experiments/planB_picos/preds/
"""

import os, sys, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import load_panel, build_v2_features, split, LGB_PARAMS_V2, HORIZONS

EXP_NAME = 'planB_picos'
OUT_DIR = os.path.join(ROOT, 'experiments', EXP_NAME)
PRED_DIR = os.path.join(OUT_DIR, 'preds')
os.makedirs(PRED_DIR, exist_ok=True)
METRICS_CSV = os.path.join(OUT_DIR, 'metrics_intervals.csv')

def winkler_score(y_true, lower, upper, alpha=0.20):
    """Calcula el Winkler Score para intervalos al nivel (1 - alpha)."""
    width = upper - lower
    under = np.maximum(0.0, lower - y_true)
    over = np.maximum(0.0, y_true - upper)
    pen = (2.0 / alpha) * (under + over)
    return np.mean(width + pen)

def eval_interval(y_true, lower, upper, hours, alpha=0.20):
    """Calcula métricas clave de un intervalo de predicción."""
    ok = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    yt = y_true[ok]
    lo = np.maximum(0.0, lower[ok])
    up = np.maximum(lo, upper[ok])
    hh = np.asarray(hours)[ok]

    covered = (yt >= lo) & (yt <= up)
    width = up - lo

    # Picos top 5%
    thr = np.percentile(yt, 95)
    pk = yt >= thr
    cov_pk = float(np.mean(covered[pk])) if np.sum(pk) > 0 else np.nan

    # Horas diurnas 10-18h
    day = (hh >= 10) & (hh <= 18)
    cov_day = float(np.mean(covered[day])) if np.sum(day) > 0 else np.nan

    ws = float(winkler_score(yt, lo, up, alpha=alpha))

    return {
        'n': int(np.sum(ok)),
        'nominal_cov': 1.0 - alpha,
        'empirical_cov': float(np.mean(covered)),
        'width_mean': float(np.mean(width)),
        'width_median': float(np.median(width)),
        'width_p95': float(np.percentile(width, 95)),
        'coverage_peaks': cov_pk,
        'coverage_day': cov_day,
        'winkler_score': ws,
    }

def run_intervals():
    print("=" * 70, flush=True)
    print("INICIANDO EVALUACIÓN DE INTERVALOS DE PREDICCIÓN (80% NOMINAL)", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    panel = load_panel()
    df, feats = build_v2_features(panel)
    train_df, val_df, test_df = split(df)
    train_sub = train_df.iloc[::2].copy()
    print(f"Features preparados en {time.time()-t0:.1f}s | Train sub: {len(train_sub):,}", flush=True)

    horizons = [1, 4, 8, 12, 24, 48]
    all_results = []

    for h in horizons:
        print(f"\n=======================================================", flush=True)
        print(f"PROCESANDO HORIZONTE +{h}h", flush=True)
        print(f"=======================================================", flush=True)

        tc = f'Target_{h}h'
        tr_h = train_df.dropna(subset=[tc])
        va_h = val_df.dropna(subset=[tc])
        te_h = test_df.dropna(subset=[tc])

        y_tr = tr_h[tc].values
        y_va = va_h[tc].values
        y_te = te_h[tc].values

        # Hora local del target
        tt_va = (va_h['Date'] + pd.Timedelta(hours=h)).dt.hour.values
        tt_te = (te_h['Date'] + pd.Timedelta(hours=h)).dt.hour.values

        # -------------------------------------------------------------
        # Cargar modelo base v2 para predicciones puntuales
        # -------------------------------------------------------------
        v2_model_path = os.path.join(ROOT, 'results', 'multihorizonte_v2', 'models', f'lgbm_v2_{h}h.txt')
        if not os.path.exists(v2_model_path):
            print(f"Alerta: No se encontró modelo v2 en {v2_model_path}, saltando...", flush=True)
            continue

        booster_v2 = lgb.Booster(model_file=v2_model_path)
        yp_va_v2 = booster_v2.predict(va_h[feats].values)
        yp_te_v2 = booster_v2.predict(te_h[feats].values)

        res_va_v2 = np.abs(y_va - yp_va_v2)

        # =============================================================
        # 1. SPLIT CONFORMAL (GLOBAL)
        # =============================================================
        alpha = 0.20
        n_va = len(res_va_v2)
        q_idx = int(np.ceil((n_va + 1) * (1.0 - alpha)))
        q_idx = min(n_va - 1, max(0, q_idx))
        sorted_res = np.sort(res_va_v2)
        q_conf_global = sorted_res[q_idx]

        lo_conf_global = np.maximum(0.0, yp_te_v2 - q_conf_global)
        up_conf_global = yp_te_v2 + q_conf_global

        m_cg = eval_interval(y_te, lo_conf_global, up_conf_global, tt_te, alpha=alpha)
        m_cg['method'] = 'Split Conformal (Global)'
        m_cg['h'] = h
        all_results.append(m_cg)
        print(f"  [Split Conformal Global] Cov={m_cg['empirical_cov']*100:.1f}% (Peak Cov={m_cg['coverage_peaks']*100:.1f}%), Ancho Medio={m_cg['width_mean']:.2f} ppb, Winkler={m_cg['winkler_score']:.2f}", flush=True)

        # =============================================================
        # 2. SPLIT CONFORMAL CONDICIONAL POR HORA DEL DÍA
        # =============================================================
        q_conf_hour = {}
        for hour in range(24):
            idx_hr = (tt_va == hour)
            res_hr = res_va_v2[idx_hr]
            n_hr = len(res_hr)
            if n_hr > 0:
                qh_idx = int(np.ceil((n_hr + 1) * (1.0 - alpha)))
                qh_idx = min(n_hr - 1, max(0, qh_idx))
                q_conf_hour[hour] = np.sort(res_hr)[qh_idx]
            else:
                q_conf_hour[hour] = q_conf_global

        q_te_hourly = np.array([q_conf_hour[hr] for hr in tt_te])
        lo_conf_hour = np.maximum(0.0, yp_te_v2 - q_te_hourly)
        up_conf_hour = yp_te_v2 + q_te_hourly

        m_ch = eval_interval(y_te, lo_conf_hour, up_conf_hour, tt_te, alpha=alpha)
        m_ch['method'] = 'Conformal Condicional (Hora)'
        m_ch['h'] = h
        all_results.append(m_ch)
        print(f"  [Conformal Condicional] Cov={m_ch['empirical_cov']*100:.1f}% (Peak Cov={m_ch['coverage_peaks']*100:.1f}%), Ancho Medio={m_ch['width_mean']:.2f} ppb, Winkler={m_ch['winkler_score']:.2f}", flush=True)

        # =============================================================
        # 3. REGRESIÓN CUANTÍLICA LIGHTGBM (alpha 0.10, 0.90)
        # =============================================================
        params_qr = dict(LGB_PARAMS_V2)
        params_qr['num_threads'] = 4
        params_qr['learning_rate'] = 0.08
        params_qr['num_leaves'] = 45
        params_qr['objective'] = 'quantile'
        params_qr['metric'] = 'quantile'
        params_qr['verbose'] = -1

        tr_h_sub = train_sub.dropna(subset=[tc])
        dtrain = lgb.Dataset(tr_h_sub[feats].values, tr_h_sub[tc].values)
        dval = lgb.Dataset(va_h[feats].values, y_va, reference=dtrain)

        # Cuantil inferior 0.10
        params_qr_low = dict(params_qr, alpha=0.10)
        m_low = lgb.train(params_qr_low, dtrain, num_boost_round=800, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        q10_va = m_low.predict(va_h[feats].values)
        q10_te = m_low.predict(te_h[feats].values)

        # Cuantil superior 0.90
        params_qr_high = dict(params_qr, alpha=0.90)
        m_high = lgb.train(params_qr_high, dtrain, num_boost_round=800, valid_sets=[dval],
                           callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        q90_va = m_high.predict(va_h[feats].values)
        q90_te = m_high.predict(te_h[feats].values)

        # Corrección de cruce y no negatividad
        lo_qr = np.maximum(0.0, np.minimum(q10_te, q90_te))
        up_qr = np.maximum(lo_qr, q90_te)

        m_qr = eval_interval(y_te, lo_qr, up_qr, tt_te, alpha=alpha)
        m_qr['method'] = 'Quantile Regression (LightGBM)'
        m_qr['h'] = h
        all_results.append(m_qr)
        print(f"  [Quantile Regression]   Cov={m_qr['empirical_cov']*100:.1f}% (Peak Cov={m_qr['coverage_peaks']*100:.1f}%), Ancho Medio={m_qr['width_mean']:.2f} ppb, Winkler={m_qr['winkler_score']:.2f}", flush=True)

        # =============================================================
        # 4. CONFORMALIZED QUANTILE REGRESSION (CQR)
        # =============================================================
        # Calibración sobre validación 2024
        scores_va = np.maximum(q10_va - y_va, y_va - q90_va)
        n_s = len(scores_va)
        q_cqr_idx = int(np.ceil((n_s + 1) * (1.0 - alpha)))
        q_cqr_idx = min(n_s - 1, max(0, q_cqr_idx))
        sorted_scores = np.sort(scores_va)
        q_cqr = sorted_scores[q_cqr_idx]

        lo_cqr = np.maximum(0.0, q10_te - q_cqr)
        up_cqr = np.maximum(lo_cqr, q90_te + q_cqr)

        m_cqr = eval_interval(y_te, lo_cqr, up_cqr, tt_te, alpha=alpha)
        m_cqr['method'] = 'CQR (Conformalized QR)'
        m_cqr['h'] = h
        m_cqr['cqr_calib_q'] = float(q_cqr)
        all_results.append(m_cqr)
        print(f"  [CQR (Conformalized QR)] Cov={m_cqr['empirical_cov']*100:.1f}% (Peak Cov={m_cqr['coverage_peaks']*100:.1f}%), Ancho Medio={m_cqr['width_mean']:.2f} ppb, Winkler={m_cqr['winkler_score']:.2f}", flush=True)

        # -------------------------------------------------------------
        # Guardar predicciones de intervalos para el horizonte
        # -------------------------------------------------------------
        df_preds = te_h[['Date', 'Estacion', tc]].copy()
        df_preds['y_real'] = y_te
        df_preds['y_point_v2'] = yp_te_v2

        df_preds['lo_conf_global'] = lo_conf_global
        df_preds['up_conf_global'] = up_conf_global

        df_preds['lo_conf_hour'] = lo_conf_hour
        df_preds['up_conf_hour'] = up_conf_hour

        df_preds['lo_qr'] = lo_qr
        df_preds['up_qr'] = up_qr

        df_preds['lo_cqr'] = lo_cqr
        df_preds['up_cqr'] = up_cqr

        pred_file = os.path.join(PRED_DIR, f'intervals_{h}h.parquet')
        df_preds.to_parquet(pred_file, index=False)
        print(f"  Predicciones guardadas en {pred_file}", flush=True)

        del dtrain, dval, m_low, m_high
        gc.collect()

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(METRICS_CSV, index=False)
    print("\n" + "=" * 70, flush=True)
    print(f"EVALUACIÓN DE INTERVALOS COMPLETADA. Guardado en {METRICS_CSV}", flush=True)
    print("=" * 70, flush=True)

if __name__ == '__main__':
    run_intervals()
