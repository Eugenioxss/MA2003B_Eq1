"""
experiments/planB_picos/03_exceedance_classification.py
========================================================
Clasificación de excedencias horarias de Ozono:
 1. Umbral NOM-020-SSA1-2021: 90 ppb (0.090 ppm, norma oficial de salud ambiental en México)
 2. Umbral alternativo: p95 del conjunto de entrenamiento (~61 ppb)

Modelos comparados:
 - LGBMClassifier con balanceo de clases (scale_pos_weight)
 - Modelo probabilístico derivado de la regresión v2
 - Línea base ingenua: Predicción puntual v2 > umbral

Protocolo estricto:
 - El umbral óptimo de decisión de probabilidad p* se calibra en 2024 (validación) maximizando CSI.
 - Evaluación final ciega sobre el conjunto de test 2025.

Métricas atmosféricas calculadas:
 - POD (Probability of Detection / Recall) = TP / (TP + FN)
 - FAR (False Alarm Ratio) = FP / (TP + FP)
 - CSI (Critical Success Index / Threat Score) = TP / (TP + FP + FN)
 - Brier Score = mean((p - y)^2)
 - ROC-AUC
 - PR-AUC (Average Precision)

Guarda métricas en experiments/planB_picos/metrics_classification.csv
y datos para curvas de calibración en experiments/planB_picos/calibration_data.parquet.
"""

import os, sys, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import load_panel, build_v2_features, split, LGB_PARAMS_V2, HORIZONS

EXP_NAME = 'planB_picos'
OUT_DIR = os.path.join(ROOT, 'experiments', EXP_NAME)
METRICS_CSV = os.path.join(OUT_DIR, 'metrics_classification.csv')
CALIB_FILE = os.path.join(OUT_DIR, 'calibration_data.parquet')
PRED_DIR = os.path.join(OUT_DIR, 'preds')
os.makedirs(PRED_DIR, exist_ok=True)

def calc_contingency_metrics(y_true, y_pred_bin):
    """Calcula POD, FAR y CSI para predicciones binarias."""
    yt = np.asarray(y_true, bool)
    yp = np.asarray(y_pred_bin, bool)

    tp = int(np.sum(yt & yp))
    fp = int(np.sum((~yt) & yp))
    fn = int(np.sum(yt & (~yp)))
    tn = int(np.sum((~yt) & (~yp)))

    pod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'POD': pod, 'FAR': far, 'CSI': csi,
    }

def find_best_threshold_csi(y_true, probs):
    """Encuentra el umbral de probabilidad p* en [0.02, 0.95] que maximiza el CSI."""
    candidates = np.linspace(0.02, 0.90, 89)
    best_thr = 0.50
    best_csi = -1.0
    yt = np.asarray(y_true, bool)

    for thr in candidates:
        yp = probs >= thr
        tp = np.sum(yt & yp)
        fp = np.sum((~yt) & yp)
        fn = np.sum(yt & (~yp))
        denom = tp + fp + fn
        csi = (tp / denom) if denom > 0 else 0.0
        if csi > best_csi:
            best_csi = csi
            best_thr = float(thr)

    return best_thr, best_csi

def run_classification():
    print("=" * 70, flush=True)
    print("INICIANDO EVALUACIÓN DE CLASIFICACIÓN DE EXCEDENCIAS DE OZONO", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    panel = load_panel()
    df, feats = build_v2_features(panel)
    train_df, val_df, test_df = split(df)
    print(f"Features preparados en {time.time()-t0:.1f}s", flush=True)

    horizons = [1, 4, 8, 12, 24, 48]
    all_metrics = []
    calib_records = []

    # Umbrales a evaluar:
    # 1. NOM-020 (90 ppb)
    # 2. p95 histórico de train
    threshold_modes = ['NOM_90ppb', 'p95_train']

    for h in horizons:
        print(f"\n=======================================================", flush=True)
        print(f"HORIZONTE +{h}h: CLASIFICACIÓN DE ALERTAS", flush=True)
        print(f"=======================================================", flush=True)

        tc = f'Target_{h}h'
        tr_h = train_df.dropna(subset=[tc])
        va_h = val_df.dropna(subset=[tc])
        te_h = test_df.dropna(subset=[tc])

        y_tr_cont = tr_h[tc].values
        y_va_cont = va_h[tc].values
        y_te_cont = te_h[tc].values

        # Cargar predicciones v2 para comparación
        v2_model_path = os.path.join(ROOT, 'results', 'multihorizonte_v2', 'models', f'lgbm_v2_{h}h.txt')
        booster_v2 = lgb.Booster(model_file=v2_model_path) if os.path.exists(v2_model_path) else None
        yp_va_v2 = booster_v2.predict(va_h[feats].values) if booster_v2 else None
        yp_te_v2 = booster_v2.predict(te_h[feats].values) if booster_v2 else None

        p95_tr_val = float(np.percentile(y_tr_cont, 95))

        for mode in threshold_modes:
            if mode == 'NOM_90ppb':
                thr_value = 90.0
                thr_label = "NOM-020 (90 ppb)"
            else:
                thr_value = p95_tr_val
                thr_label = f"p95 Train ({thr_value:.1f} ppb)"

            # Crear labels binarios
            y_tr_bin = (y_tr_cont >= thr_value).astype(int)
            y_va_bin = (y_va_cont >= thr_value).astype(int)
            y_te_bin = (y_te_cont >= thr_value).astype(int)

            n_pos_tr = int(np.sum(y_tr_bin))
            n_neg_tr = len(y_tr_bin) - n_pos_tr
            scale_pos = (n_neg_tr / max(1, n_pos_tr))
            prev_te = float(np.mean(y_te_bin))

            print(f"\n--- Umbral: {thr_label} (Prevalencia Test 2025: {prev_te*100:.2f}%) ---", flush=True)

            # ---------------------------------------------------------
            # 1. BASELINE REGRESIÓN V2: Predicción directa > umbral
            # ---------------------------------------------------------
            if yp_te_v2 is not None:
                yp_te_naive_bin = (yp_te_v2 >= thr_value).astype(int)
                cm_naive = calc_contingency_metrics(y_te_bin, yp_te_naive_bin)
                brier_naive = float(brier_score_loss(y_te_bin, np.clip(yp_te_v2 / (2 * thr_value), 0, 1)))

                # Probabilidad gaussiana estimada de exceder el umbral con residuos de v2
                std_res = float(np.std(y_va_cont - yp_va_v2))
                from scipy.stats import norm
                p_va_v2_prob = 1.0 - norm.cdf((thr_value - yp_va_v2) / max(1e-3, std_res))
                p_te_v2_prob = 1.0 - norm.cdf((thr_value - yp_te_v2) / max(1e-3, std_res))

                opt_p_v2, csi_va_v2 = find_best_threshold_csi(y_va_bin, p_va_v2_prob)
                yp_te_v2_prob_bin = (p_te_v2_prob >= opt_p_v2).astype(int)
                cm_v2_prob = calc_contingency_metrics(y_te_bin, yp_te_v2_prob_bin)

                roc_v2 = float(roc_auc_score(y_te_bin, p_te_v2_prob))
                pr_v2 = float(average_precision_score(y_te_bin, p_te_v2_prob))
                brier_v2 = float(brier_score_loss(y_te_bin, p_te_v2_prob))

                # Registrar métricas Naive v2
                rec_naive = {
                    'h': h, 'threshold_mode': mode, 'threshold_val': thr_value,
                    'model': 'Naive Regression v2 (y_hat >= thr)',
                    'opt_prob_thr': np.nan, 'val_CSI_at_opt': np.nan,
                    'POD': cm_naive['POD'], 'FAR': cm_naive['FAR'], 'CSI': cm_naive['CSI'],
                    'Brier': brier_naive, 'ROC_AUC': np.nan, 'PR_AUC': np.nan,
                    'TP': cm_naive['TP'], 'FP': cm_naive['FP'], 'FN': cm_naive['FN'], 'TN': cm_naive['TN']
                }
                all_metrics.append(rec_naive)
                print(f"  [Naive v2 y_hat >= {thr_value:.0f}] POD={cm_naive['POD']:.3f}, FAR={cm_naive['FAR']:.3f}, CSI={cm_naive['CSI']:.3f}", flush=True)

                # Registrar métricas v2 Probabilístico
                rec_v2_prob = {
                    'h': h, 'threshold_mode': mode, 'threshold_val': thr_value,
                    'model': 'Regression v2 + Gaussian Tail Prob',
                    'opt_prob_thr': opt_p_v2, 'val_CSI_at_opt': csi_va_v2,
                    'POD': cm_v2_prob['POD'], 'FAR': cm_v2_prob['FAR'], 'CSI': cm_v2_prob['CSI'],
                    'Brier': brier_v2, 'ROC_AUC': roc_v2, 'PR_AUC': pr_v2,
                    'TP': cm_v2_prob['TP'], 'FP': cm_v2_prob['FP'], 'FN': cm_v2_prob['FN'], 'TN': cm_v2_prob['TN']
                }
                all_metrics.append(rec_v2_prob)
                print(f"  [v2 Gaussian Prob (p*={opt_p_v2:.2f})] POD={cm_v2_prob['POD']:.3f}, FAR={cm_v2_prob['FAR']:.3f}, CSI={cm_v2_prob['CSI']:.3f}, ROC={roc_v2:.3f}, PR-AUC={pr_v2:.3f}", flush=True)

            # ---------------------------------------------------------
            # 2. LGBMCLASSIFIER DEDICADO CON BALANCEO
            # ---------------------------------------------------------
            params_clf = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_child_samples': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': scale_pos,
                'num_threads': 4,
                'seed': 42,
                'verbose': -1,
            }

            dtr_clf = lgb.Dataset(tr_h[feats].values, y_tr_bin)
            dva_clf = lgb.Dataset(va_h[feats].values, y_va_bin, reference=dtr_clf)

            t_clf_start = time.time()
            clf_model = lgb.train(
                params_clf,
                dtr_clf,
                num_boost_round=1000,
                valid_sets=[dva_clf],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=40, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            t_train_clf = time.time() - t_clf_start

            p_va_clf = clf_model.predict(va_h[feats].values)
            p_te_clf = clf_model.predict(te_h[feats].values)

            # Calibrar umbral óptimo p* en 2024 (validación) para maximizar CSI
            opt_p_clf, val_csi_opt = find_best_threshold_csi(y_va_bin, p_va_clf)

            # Evaluar en Test 2025 con p=0.50 estándar
            cm_clf_05 = calc_contingency_metrics(y_te_bin, (p_te_clf >= 0.50).astype(int))
            # Evaluar en Test 2025 con p* calibrado en validación
            cm_clf_opt = calc_contingency_metrics(y_te_bin, (p_te_clf >= opt_p_clf).astype(int))

            roc_clf = float(roc_auc_score(y_te_bin, p_te_clf))
            pr_clf = float(average_precision_score(y_te_bin, p_te_clf))
            brier_clf = float(brier_score_loss(y_te_bin, p_te_clf))

            # Registro LGBMClassifier (p=0.50)
            all_metrics.append({
                'h': h, 'threshold_mode': mode, 'threshold_val': thr_value,
                'model': 'LGBMClassifier (default p=0.50)',
                'opt_prob_thr': 0.50, 'val_CSI_at_opt': np.nan,
                'POD': cm_clf_05['POD'], 'FAR': cm_clf_05['FAR'], 'CSI': cm_clf_05['CSI'],
                'Brier': brier_clf, 'ROC_AUC': roc_clf, 'PR_AUC': pr_clf,
                'TP': cm_clf_05['TP'], 'FP': cm_clf_05['FP'], 'FN': cm_clf_05['FN'], 'TN': cm_clf_05['TN']
            })

            # Registro LGBMClassifier (p=p* óptimo de 2024)
            all_metrics.append({
                'h': h, 'threshold_mode': mode, 'threshold_val': thr_value,
                'model': 'LGBMClassifier (tuned p*)',
                'opt_prob_thr': opt_p_clf, 'val_CSI_at_opt': val_csi_opt,
                'POD': cm_clf_opt['POD'], 'FAR': cm_clf_opt['FAR'], 'CSI': cm_clf_opt['CSI'],
                'Brier': brier_clf, 'ROC_AUC': roc_clf, 'PR_AUC': pr_clf,
                'TP': cm_clf_opt['TP'], 'FP': cm_clf_opt['FP'], 'FN': cm_clf_opt['FN'], 'TN': cm_clf_opt['TN']
            })

            print(f"  [LGBMClassifier (p*={opt_p_clf:.2f})] POD={cm_clf_opt['POD']:.3f}, FAR={cm_clf_opt['FAR']:.3f}, CSI={cm_clf_opt['CSI']:.3f}, ROC={roc_clf:.3f}, PR-AUC={pr_clf:.3f} (train {t_train_clf:.1f}s, iter={clf_model.best_iteration})", flush=True)

            # Guardar predicciones de probabilidades para curvas de calibración (especialmente NOM 90ppb)
            if mode == 'NOM_90ppb':
                calib_df = pd.DataFrame({
                    'h': h,
                    'y_true_bin': y_te_bin,
                    'prob_clf': p_te_clf,
                    'prob_v2_gauss': p_te_v2_prob if yp_te_v2 is not None else np.nan,
                })
                calib_records.append(calib_df)

            del dtr_clf, dva_clf, clf_model
            gc.collect()

    df_out = pd.DataFrame(all_metrics)
    df_out.to_csv(METRICS_CSV, index=False)
    print("\n" + "=" * 70, flush=True)
    print(f"CLASIFICACIÓN COMPLETADA. Métricas guardadas en {METRICS_CSV}", flush=True)
    print("=" * 70, flush=True)

    if calib_records:
        full_calib = pd.concat(calib_records, ignore_index=True)
        full_calib.to_parquet(CALIB_FILE, index=False)
        print(f"Datos de calibración guardados en {CALIB_FILE}", flush=True)

if __name__ == '__main__':
    run_classification()
