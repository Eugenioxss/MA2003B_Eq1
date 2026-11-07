"""
experiments/planB_picos/04_peak_correction.py
=============================================
Explora técnicas de corrección post-hoc y modelos de 2 etapas para corregir
el sesgo sistemático en picos sin degradar excesivamente el RMSE global.

Métodos evaluados:
 1. Regresión Isotónica (Isotonic Regression):
    Ajusta una transformación no decreciente f(y_hat) sobre las predicciones de validación 2024
    y la aplica directamente sobre el test 2025.
 2. Corrección Lineal de Cola (Tail Linear Scaling):
    Para predicciones por encima del p90 de validación, aplica un factor de escala
    calibrado en validación 2024: y_corr = y_hat + alpha * max(0, y_hat - p90).
 3. Modelo Híbrido de 2 Etapas (Soft Blending):
    Pondera la predicción estándar v2 con un modelo ponderado en picos (weight_step_p90)
    usando la probabilidad de excedencia del clasificador:
    y_final = (1 - P_alerta) * y_v2 + P_alerta * y_picos.

Métricas evaluadas en 2025:
 - RMSE global, MAE, R2, bias
 - RMSE_peaks y bias_peaks
 - Trade-off delta_RMSE vs delta_bias_peaks

Guarda métricas en experiments/planB_picos/metrics_postprocessing.csv
"""

import os, sys, time
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import load_panel, build_v2_features, split, evaluate_test, log_result, HORIZONS

EXP_NAME = 'planB_picos'
OUT_DIR = os.path.join(ROOT, 'experiments', EXP_NAME)
PRED_DIR = os.path.join(OUT_DIR, 'preds')
METRICS_CSV = os.path.join(OUT_DIR, 'metrics_postprocessing.csv')

def run_corrections():
    print("=" * 70, flush=True)
    print("INICIANDO EVALUACIÓN DE CORRECCIÓN DE PICOS Y POST-PROCESAMIENTO", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    panel = load_panel()
    df, feats = build_v2_features(panel)
    train_df, val_df, test_df = split(df)
    print(f"Datos cargados en {time.time()-t0:.1f}s", flush=True)

    import lightgbm as lgb

    horizons = [1, 4, 8, 12, 24, 48]
    all_metrics = []

    for h in horizons:
        tc = f'Target_{h}h'
        va_h = val_df.dropna(subset=[tc])
        te_h = test_df.dropna(subset=[tc])

        y_va = va_h[tc].values
        y_te = te_h[tc].values

        v2_path = os.path.join(ROOT, 'results', 'multihorizonte_v2', 'models', f'lgbm_v2_{h}h.txt')
        if not os.path.exists(v2_path):
            continue
        booster_v2 = lgb.Booster(model_file=v2_path)
        yp_va_v2 = booster_v2.predict(va_h[feats].values)
        yp_te_v2 = booster_v2.predict(te_h[feats].values)

        m_v2 = evaluate_test(te_h, yp_te_v2, h)
        all_metrics.append({
            'h': h, 'method': 'v2_baseline_uncalibrated',
            'RMSE': m_v2['RMSE'], 'MAE': m_v2['MAE'], 'R2': m_v2['R2'],
            'bias': m_v2['bias'], 'RMSE_peaks': m_v2['RMSE_peaks'], 'bias_peaks': m_v2['bias_peaks']
        })

        # -------------------------------------------------------------
        # 1. REGRESIÓN ISOTÓNICA SOBRE 2024
        # -------------------------------------------------------------
        # Entrenar regresión isotónica sobre validación 2024
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(yp_va_v2, y_va)
        yp_te_iso = iso.predict(yp_te_v2)

        m_iso = evaluate_test(te_h, yp_te_iso, h)
        all_metrics.append({
            'h': h, 'method': 'postproc_isotonic',
            'RMSE': m_iso['RMSE'], 'MAE': m_iso['MAE'], 'R2': m_iso['R2'],
            'bias': m_iso['bias'], 'RMSE_peaks': m_iso['RMSE_peaks'], 'bias_peaks': m_iso['bias_peaks']
        })
        log_result(EXP_NAME, 'postproc_isotonic', h, m_iso, notes="Regresión isotónica ajustada en val 2024")

        # -------------------------------------------------------------
        # 2. CORRECCIÓN LINEAL DE COLA (Tail Linear Expansion)
        # -------------------------------------------------------------
        # Buscar pendiente alpha para expandir valores altos: y_corr = y_hat + alpha * max(0, y_hat - thr)
        thr_tail = np.percentile(yp_va_v2, 85)
        def loss_tail(alpha):
            pred_corr = yp_va_v2 + alpha * np.maximum(0.0, yp_va_v2 - thr_tail)
            # Función objetivo balanceada en validación: RMSE_global + 0.5 * RMSE_peaks
            pks = y_va >= np.percentile(y_va, 95)
            rmse_tot = np.sqrt(np.mean((y_va - pred_corr)**2))
            rmse_pk = np.sqrt(np.mean((y_va[pks] - pred_corr[pks])**2))
            return rmse_tot + 0.3 * rmse_pk

        res = minimize_scalar(loss_tail, bounds=(0.0, 1.5), method='bounded')
        opt_alpha = float(res.x)

        yp_te_tail = yp_te_v2 + opt_alpha * np.maximum(0.0, yp_te_v2 - thr_tail)
        m_tail = evaluate_test(te_h, yp_te_tail, h)
        all_metrics.append({
            'h': h, 'method': f'postproc_tail_scaling (alpha={opt_alpha:.2f})',
            'RMSE': m_tail['RMSE'], 'MAE': m_tail['MAE'], 'R2': m_tail['R2'],
            'bias': m_tail['bias'], 'RMSE_peaks': m_tail['RMSE_peaks'], 'bias_peaks': m_tail['bias_peaks']
        })
        log_result(EXP_NAME, 'postproc_tail_scaling', h, m_tail, notes=f"Expansión de cola alpha={opt_alpha:.2f} sobre p85={thr_tail:.1f}ppb")

        print(f"h={h:2d} | v2: RMSE={m_v2['RMSE']:.2f}, PeakBias={m_v2['bias_peaks']:.1f} | "
              f"Iso: RMSE={m_iso['RMSE']:.2f}, PeakBias={m_iso['bias_peaks']:.1f} | "
              f"Tail: RMSE={m_tail['RMSE']:.2f}, PeakBias={m_tail['bias_peaks']:.1f}", flush=True)

    df_res = pd.DataFrame(all_metrics)
    df_res.to_csv(METRICS_CSV, index=False)
    print(f"\nResultados de corrección guardados en {METRICS_CSV}", flush=True)

if __name__ == '__main__':
    run_corrections()
