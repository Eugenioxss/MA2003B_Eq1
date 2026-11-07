"""
experiments/noche2/ens2/ensemble_pipeline.py
============================================
Pipeline Principal de Ensamble, Meta-Aprendizaje y Calibración Conformal (Corrida Noche 2).

Metodología:
1. Selección y descubrimiento dinámico de modelos honestos (excluye _pp, pp_, leak_test, ensambles previos).
2. Protocolo de Validación Temporal Estricto Out-of-Fold (OOF) 4-Fold por Trimestres (Q1-Q4 2024).
3. Evaluación de 7 Familias de Algoritmos de Ensamble:
   - M1: top_k_mean
   - M2: greedy_caruana
   - M3: nnls_ridge_global
   - M4: nnls_by_regime
   - M5: ridge_meta_context
   - M6: lgbm_stacking_oof + blend_convex_lgbm
   - M7: peak_tail_booster
4. Selección insesgada por menor Val RMSE OOF en 2024.
5. Calibración Conformal Heterocedástica (80% y 90% cobertura nominal).
6. Tarea Regulatoria NOM-020-SSA1 (Evaluación DMA8: Daily Maximum 8h Average).
7. Generación de preds/ens2_best_h<h>.parquet (val OOF + test 2025).
"""
from __future__ import annotations
import os
import sys
import glob
import re
import time
import json
import warnings
warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import evaluate_test, log_result, HORIZONS, MODEL_STATIONS, climatology_pred, metrics

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(EXP_DIR, 'preds')
FIGS_DIR = os.path.join(EXP_DIR, 'figures')
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Descubrimiento y Alineación de Modelos Honestos
# ---------------------------------------------------------------------------
def classify_family(exp: str, var: str) -> str:
    v = var.lower()
    e = exp.lower()
    if 'xgb' in v:
        return 'XGBoost'
    if 'et_' in v or 'extratrees' in v:
        return 'ExtraTrees'
    if 'deep_imput2' in v:
        return 'Deep_Imput2_Resid'
    if 'imput2' in e or 'audit_imput' in e or 'imput' in v:
        return 'Imputation_Spatial'
    if 'deep2' in e or 'deep' in e or 'resnet' in v or 'gated' in v or 'swiglu' in v or 'gru' in v or 'seq' in v:
        if 'resnet' in v:
            return 'Deep_TabResNet'
        elif 'gated' in v or 'swiglu' in v:
            return 'Deep_GatedSwiGLU'
        elif 'residual' in v:
            return 'Deep_LGBM_Residual'
        elif 'gru' in v or 'seq' in v:
            return 'Deep_Seq_GRU'
        else:
            return 'Deep_Other'
    if 'refit' in e or 'refit' in v:
        return 'LightGBM_Refit'
    if 'nwp' in e or 'nwp' in v:
        return 'LightGBM_NWP'
    if 'target_eng' in e or 'eng' in v:
        return 'Target_Eng'
    if 'tuning' in e:
        return 'LGBM_Tuning'
    if 'v3_base' in e:
        return 'LGBM_v3_Base'
    return 'Other'


def discover_candidate_models(h: int) -> dict[str, str]:
    """Descubre parquets honestos disponibles para el horizonte h."""
    patterns = [
        os.path.join(ROOT, 'experiments', 'noche', '*', 'preds', f'*_h{h}.parquet'),
        os.path.join(ROOT, 'experiments', 'noche2', '*', 'preds', f'*_h{h}.parquet')
    ]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat))
        
    candidates = {}
    for fpath in all_files:
        fname = os.path.basename(fpath)
        m = re.match(r'(.+)_h\d+\.parquet$', fname)
        if not m:
            continue
        var = m.group(1)
        parts = fpath.replace('\\', '/').split('/')
        exp = parts[-3]
        
        # Filtros de honestidad estricta
        if any(bad in var.lower() for bad in ['_pp', 'pp_', 'leak_test', 'ensemble_', 'ens2_']):
            continue
            
        try:
            head = pd.read_parquet(fpath, columns=['split', 'y_true', 'y_pred'])
            splits = head['split'].unique()
            if 'val' not in splits or 'test' not in splits:
                continue
            val_cov = head[head['split'] == 'val']['y_pred'].notna().mean()
            test_cov = head[head['split'] == 'test']['y_pred'].notna().mean()
            if val_cov < 0.99 or test_cov < 0.99:
                continue
                
            unique_name = f"{exp}__{var}"
            candidates[unique_name] = fpath
        except Exception:
            continue
            
    return candidates


def select_diverse_candidates_for_h(h: int, candidate_paths: dict[str, str], max_per_family: int = 4) -> tuple[pd.DataFrame, list[str]]:
    """
    Carga todos los modelos candidatos, evalúa su RMSE en val y selecciona
    un conjunto óptimo y diverso con los mejores representantes de cada familia.
    """
    dfs = {}
    y_true_s = None
    
    for name, path in sorted(candidate_paths.items()):
        df = pd.read_parquet(path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Estacion'] = df['Estacion'].astype(str)
        indexed = df.set_index(['Date', 'Estacion', 'split'])
        dfs[name] = indexed['y_pred']
        if y_true_s is None and 'y_true' in indexed.columns:
            y_true_s = indexed['y_true']
            
    m_wide = pd.DataFrame(dfs)
    if y_true_s is not None:
        m_wide['y_true'] = y_true_s
        m_wide[f'Target_{h}h'] = y_true_s
        
    m_wide = m_wide.reset_index()
    
    # Agregar variables temporales clave
    m_wide['target_time'] = m_wide['Date'] + pd.Timedelta(hours=h)
    m_wide['target_hour'] = m_wide['target_time'].dt.hour
    m_wide['target_month'] = m_wide['target_time'].dt.month
    m_wide['target_dayofweek'] = m_wide['target_time'].dt.dayofweek
    
    # 4 Folds temporales por trimestres dentro de 2024:
    # 0: Q1 (Ene-Mar), 1: Q2 (Abr-Jun), 2: Q3 (Jul-Sep), 3: Q4 (Oct-Dic)
    m_wide['fold'] = (m_wide['target_month'] - 1) // 3
    
    # Régimen fotoquímico / bloque horario:
    # 0: Pico fotoquímico diurno (10:00–18:00)
    # 1: Noche (22:00–06:00)
    # 2: Transición matutina (07:00–09:00)
    # 3: Transición vespertina (19:00–21:00)
    def assign_regime(hr):
        if 10 <= hr <= 18:
            return 0
        elif hr >= 22 or hr <= 6:
            return 1
        elif 7 <= hr <= 9:
            return 2
        else:
            return 3
    m_wide['regime'] = m_wide['target_hour'].apply(assign_regime)
    
    all_models = list(dfs.keys())
    
    # Imputar si hubiera algún NaN aislado con la media de la fila
    nan_counts = m_wide[all_models].isna().sum()
    if (nan_counts > 0).any():
        row_mean = m_wide[all_models].mean(axis=1)
        for c in all_models:
            m_wide[c] = m_wide[c].fillna(row_mean)
            
    # Evaluar RMSE en Val 2024 para cada modelo y seleccionar top por familia
    val_mask = m_wide['split'] == 'val'
    y_val = m_wide.loc[val_mask, 'y_true'].values
    ok_v = np.isfinite(y_val)
    
    model_scores = []
    for mod in all_models:
        yp = m_wide.loc[val_mask, mod].values
        rmse = np.sqrt(np.mean((y_val[ok_v] - yp[ok_v]) ** 2))
        parts = mod.split('__')
        exp, var = parts[0], parts[1]
        family = classify_family(exp, var)
        model_scores.append({'model_name': mod, 'family': family, 'val_rmse': rmse})
        
    df_scores = pd.DataFrame(model_scores).sort_values('val_rmse')
    
    selected_models = []
    for fam, group in df_scores.groupby('family'):
        top_fam = group.head(max_per_family)['model_name'].tolist()
        selected_models.extend(top_fam)
        
    # Asegurar que los mejores globales absolutos siempre estén incluidos
    best_global = df_scores.head(20)['model_name'].tolist()
    for m in best_global:
        if m not in selected_models:
            selected_models.append(m)
            
    selected_models = sorted(list(set(selected_models)))
    print(f"Candidatos seleccionados ({len(selected_models)} de {len(all_models)} disponibles en {len(df_scores['family'].unique())} familias):")
    for fam in df_scores['family'].unique():
        f_mods = [m for m in selected_models if m in df_scores[df_scores['family'] == fam]['model_name'].values]
        if f_mods:
            print(f"  * {fam}: {f_mods}")
            
    return m_wide, selected_models


# ---------------------------------------------------------------------------
# 2. Métodos de Ensamble y Meta-Aprendizaje (Cross-Fitting OOF)
# ---------------------------------------------------------------------------

def fit_m0_best_individual(val_df: pd.DataFrame, test_df: pd.DataFrame, models: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """M0: Mejor modelo individual según Val 2024."""
    y_val = val_df['y_true'].values
    rmses = {}
    for m in models:
        yp = val_df[m].values
        ok = np.isfinite(y_val) & np.isfinite(yp)
        rmses[m] = float(np.sqrt(np.mean((y_val[ok] - yp[ok]) ** 2)))
    best_m = min(rmses, key=rmses.get)
    pred_val_oof = val_df[best_m].values.copy()
    pred_test = test_df[best_m].values.copy()
    info = {'method': 'M0_best_individual', 'best_model': best_m, 'val_rmse_oof': rmses[best_m]}
    return info, pred_val_oof, pred_test


def fit_m1_top_k_mean(val_df: pd.DataFrame, test_df: pd.DataFrame, models: list[str], k_grid: list[int] = [2, 3, 4, 5, 7, 10]) -> tuple[dict, np.ndarray, np.ndarray]:
    """M1: Promedio simple de los Top-K modelos evaluado estrictamente OOF."""
    folds = sorted(val_df['fold'].unique())
    y_val = val_df['y_true'].values
    
    k_scores = {}
    k_preds = {}
    
    for k in k_grid:
        if k > len(models):
            continue
        p_oof = np.zeros(len(val_df), dtype=float)
        for f in folds:
            tr_m = val_df['fold'] != f
            va_m = val_df['fold'] == f
            y_tr = val_df.loc[tr_m, 'y_true'].values
            
            rmses_fold = {}
            for m in models:
                yp = val_df.loc[tr_m, m].values
                ok = np.isfinite(y_tr) & np.isfinite(yp)
                rmses_fold[m] = np.mean((y_tr[ok] - yp[ok]) ** 2)
            top_k = sorted(rmses_fold, key=rmses_fold.get)[:k]
            p_oof[va_m] = val_df.loc[va_m, top_k].mean(axis=1).values
            
        ok_all = np.isfinite(y_val) & np.isfinite(p_oof)
        k_scores[k] = float(np.sqrt(np.mean((y_val[ok_all] - p_oof[ok_all]) ** 2)))
        k_preds[k] = p_oof
        
    best_k = min(k_scores, key=k_scores.get)
    pred_val_oof = k_preds[best_k]
    
    # Para test: top_k sobre todo val 2024
    all_rmses = {}
    for m in models:
        yp = val_df[m].values
        ok = np.isfinite(y_val) & np.isfinite(yp)
        all_rmses[m] = np.mean((y_val[ok] - yp[ok]) ** 2)
    final_top_k = sorted(all_rmses, key=all_rmses.get)[:best_k]
    pred_test = test_df[final_top_k].mean(axis=1).values
    
    info = {'method': 'M1_top_k_mean', 'best_k': best_k, 'models': final_top_k, 'val_rmse_oof': k_scores[best_k]}
    return info, pred_val_oof, pred_test


def fit_m2_greedy_caruana(val_df: pd.DataFrame, test_df: pd.DataFrame, models: list[str], max_iter: int = 40) -> tuple[dict, np.ndarray, np.ndarray]:
    """M2: Selección forward codiciosa con reemplazo (Caruana Ensemble Selection)."""
    folds = sorted(val_df['fold'].unique())
    y_val = val_df['y_true'].values
    pred_val_oof = np.zeros(len(val_df), dtype=float)
    
    def run_caruana(X_tr, y_tr, X_val, iters):
        ok = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
        X_tr = X_tr[ok]
        y_tr = y_tr[ok]
        
        # Iniciar con el mejor modelo individual
        best_i = int(np.argmin(np.mean((X_tr - y_tr[:, None]) ** 2, axis=0)))
        current_pred_tr = X_tr[:, best_i].copy()
        current_pred_va = X_val[:, best_i].copy()
        selected_indices = [best_i]
        
        for step in range(2, iters + 1):
            best_cand = None
            best_err = float('inf')
            weight_new = 1.0 / step
            weight_old = (step - 1.0) / step
            
            for j in range(X_tr.shape[1]):
                trial_pred = weight_old * current_pred_tr + weight_new * X_tr[:, j]
                err = np.mean((y_tr - trial_pred) ** 2)
                if err < best_err:
                    best_err = err
                    best_cand = j
                    
            selected_indices.append(best_cand)
            current_pred_tr = weight_old * current_pred_tr + weight_new * X_tr[:, best_cand]
            current_pred_va = weight_old * current_pred_va + weight_new * X_val[:, best_cand]
            
        return current_pred_va, selected_indices
        
    for f in folds:
        tr_m = val_df['fold'] != f
        va_m = val_df['fold'] == f
        X_tr = val_df.loc[tr_m, models].values
        y_tr = val_df.loc[tr_m, 'y_true'].values
        X_va = val_df.loc[va_m, models].values
        p_va, _ = run_caruana(X_tr, y_tr, X_va, max_iter)
        pred_val_oof[va_m] = p_va
        
    ok_all = np.isfinite(y_val) & np.isfinite(pred_val_oof)
    val_rmse_oof = float(np.sqrt(np.mean((y_val[ok_all] - pred_val_oof[ok_all]) ** 2)))
    
    # Para test 2025: correr sobre todo val 2024
    X_val_all = val_df[models].values
    X_te = test_df[models].values
    pred_test, final_indices = run_caruana(X_val_all, y_val, X_te, max_iter)
    
    counts = pd.Series(final_indices).value_counts(normalize=True).to_dict()
    weights = {models[idx]: float(round(w, 4)) for idx, w in sorted(counts.items(), key=lambda x: -x[1])}
    
    info = {'method': 'M2_greedy_caruana', 'weights': weights, 'val_rmse_oof': val_rmse_oof}
    return info, pred_val_oof, pred_test


def fit_m3_nnls_ridge_global(val_df: pd.DataFrame, test_df: pd.DataFrame, models: list[str], lambdas: list[float] = [0.0, 10.0, 50.0, 200.0]) -> tuple[dict, np.ndarray, np.ndarray]:
    """M3: Simplex Ridge (NNLS con regularización L2 y suma normalizada a 1)."""
    folds = sorted(val_df['fold'].unique())
    y_val = val_df['y_true'].values
    
    lam_scores = {}
    lam_oof_preds = {}
    
    def solve_simplex_ridge(X, y, lam):
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X = X[ok]
        y = y[ok]
        if lam > 0:
            reg_X = np.sqrt(lam) * np.eye(X.shape[1])
            reg_y = np.zeros(X.shape[1])
            X_aug = np.vstack([X, reg_X])
            y_aug = np.concatenate([y, reg_y])
        else:
            X_aug = X
            y_aug = y
        w, _ = nnls(X_aug, y_aug)
        sw = w.sum()
        return (w / sw) if sw > 0 else (np.ones(X.shape[1]) / X.shape[1])
        
    for lam in lambdas:
        p_oof = np.zeros(len(val_df), dtype=float)
        for f in folds:
            tr_m = val_df['fold'] != f
            va_m = val_df['fold'] == f
            X_tr = val_df.loc[tr_m, models].values
            y_tr = val_df.loc[tr_m, 'y_true'].values
            X_va = val_df.loc[va_m, models].values
            
            w_norm = solve_simplex_ridge(X_tr, y_tr, lam)
            p_oof[va_m] = X_va @ w_norm
            
        ok_all = np.isfinite(y_val) & np.isfinite(p_oof)
        lam_scores[lam] = float(np.sqrt(np.mean((y_val[ok_all] - p_oof[ok_all]) ** 2)))
        lam_oof_preds[lam] = p_oof
        
    best_lam = min(lam_scores, key=lam_scores.get)
    pred_val_oof = lam_oof_preds[best_lam]
    
    # Entrenar sobre todo val 2024 para test
    X_val_all = val_df[models].values
    w_final = solve_simplex_ridge(X_val_all, y_val, best_lam)
    pred_test = test_df[models].values @ w_final
    
    weights_dict = {models[i]: float(round(w_final[i], 4)) for i in range(len(models)) if w_final[i] > 0.001}
    info = {'method': 'M3_nnls_ridge_global', 'best_lambda': best_lam, 'weights': weights_dict, 'val_rmse_oof': lam_scores[best_lam]}
    return info, pred_val_oof, pred_test


def fit_m4_nnls_by_regime(val_df: pd.DataFrame, test_df: pd.DataFrame, models: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """M4: Pesos convexos diferenciados por 4 regímenes solares/horarios."""
    folds = sorted(val_df['fold'].unique())
    y_val = val_df['y_true'].values
    pred_val_oof = np.zeros(len(val_df), dtype=float)
    regimes = [0, 1, 2, 3]
    
    for f in folds:
        tr_m = val_df['fold'] != f
        va_m = val_df['fold'] == f
        
        for r in regimes:
            mr_tr = tr_m & (val_df['regime'] == r)
            mr_va = va_m & (val_df['regime'] == r)
            
            X_tr = val_df.loc[mr_tr, models].values
            y_tr = val_df.loc[mr_tr, 'y_true'].values
            X_va = val_df.loc[mr_va, models].values
            
            ok = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
            if ok.sum() > len(models):
                w_r, _ = nnls(X_tr[ok], y_tr[ok])
                sw = w_r.sum()
                w_norm = w_r / sw if sw > 0 else np.ones(len(models)) / len(models)
            else:
                w_norm = np.ones(len(models)) / len(models)
            pred_val_oof[mr_va] = X_va @ w_norm
            
    ok_all = np.isfinite(y_val) & np.isfinite(pred_val_oof)
    val_rmse_oof = float(np.sqrt(np.mean((y_val[ok_all] - pred_val_oof[ok_all]) ** 2)))
    
    # Para test 2025: pesos por régimen sobre todo val 2024
    pred_test = np.zeros(len(test_df), dtype=float)
    regime_weights = {}
    for r in regimes:
        mr_all = val_df['regime'] == r
        mr_te = test_df['regime'] == r
        
        X_tr = val_df.loc[mr_all, models].values
        y_tr = val_df.loc[mr_all, 'y_true'].values
        X_te = test_df.loc[mr_te, models].values
        
        ok = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
        w_r, _ = nnls(X_tr[ok], y_tr[ok])
        sw = w_r.sum()
        w_norm = w_r / sw if sw > 0 else np.ones(len(models)) / len(models)
        pred_test[mr_te] = X_te @ w_norm
        regime_weights[f'regime_{r}'] = {models[i]: float(round(w_norm[i], 4)) for i in range(len(models)) if w_norm[i] > 0.005}
        
    info = {'method': 'M4_nnls_by_regime', 'regime_weights': regime_weights, 'val_rmse_oof': val_rmse_oof}
    return info, pred_val_oof, pred_test


def fit_m5_ridge_meta_context(val_df: pd.DataFrame, test_df: pd.DataFrame, models: list[str], alphas: list[float] = [1.0, 10.0, 50.0, 200.0]) -> tuple[dict, np.ndarray, np.ndarray]:
    """M5: Meta-aprendiz Ridge con contexto horario, dispersión inter-modelo y estación."""
    def build_meta_matrix(df: pd.DataFrame) -> pd.DataFrame:
        X = df[models].copy()
        X['meta_mean'] = df[models].mean(axis=1)
        X['meta_std'] = df[models].std(axis=1).fillna(0)
        X['meta_min'] = df[models].min(axis=1)
        X['meta_max'] = df[models].max(axis=1)
        X['meta_spread'] = X['meta_max'] - X['meta_min']
        X['hour_sin'] = np.sin(2 * np.pi * df['target_hour'] / 24).astype('float32')
        X['hour_cos'] = np.cos(2 * np.pi * df['target_hour'] / 24).astype('float32')
        X['month_sin'] = np.sin(2 * np.pi * df['target_month'] / 12).astype('float32')
        X['month_cos'] = np.cos(2 * np.pi * df['target_month'] / 12).astype('float32')
        
        # One-hot estación
        dummies = pd.get_dummies(df['Estacion'], prefix='est', drop_first=True, dtype='float32')
        return pd.concat([X, dummies], axis=1)
        
    X_meta_val = build_meta_matrix(val_df)
    X_meta_test = build_meta_matrix(test_df)
    y_val = val_df['y_true'].values
    folds = sorted(val_df['fold'].unique())
    
    alpha_scores = {}
    alpha_oof = {}
    
    for alpha in alphas:
        p_oof = np.zeros(len(val_df), dtype=float)
        for f in folds:
            tr_m = val_df['fold'] != f
            va_m = val_df['fold'] == f
            
            X_tr = X_meta_val[tr_m].values
            y_tr = y_val[tr_m]
            X_va = X_meta_val[va_m].values
            
            ok = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
            clf = Ridge(alpha=alpha, fit_intercept=True)
            clf.fit(X_tr[ok], y_tr[ok])
            p_oof[va_m] = clf.predict(X_va)
            
        ok_all = np.isfinite(y_val) & np.isfinite(p_oof)
        alpha_scores[alpha] = float(np.sqrt(np.mean((y_val[ok_all] - p_oof[ok_all]) ** 2)))
        alpha_oof[alpha] = p_oof
        
    best_alpha = min(alpha_scores, key=alpha_scores.get)
    pred_val_oof = np.maximum(0.0, alpha_oof[best_alpha])
    
    # Predicción test con todo val
    clf_final = Ridge(alpha=best_alpha, fit_intercept=True)
    ok_tot = np.isfinite(y_val) & np.isfinite(X_meta_val.values).all(axis=1)
    clf_final.fit(X_meta_val.values[ok_tot], y_val[ok_tot])
    pred_test = np.maximum(0.0, clf_final.predict(X_meta_test.values))
    
    info = {'method': 'M5_ridge_meta_context', 'best_alpha': best_alpha, 'val_rmse_oof': alpha_scores[best_alpha]}
    return info, pred_val_oof, pred_test


def fit_m6_lgbm_stacking_and_blend(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models: list[str],
    pred_val_linear: np.ndarray,
    pred_test_linear: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    """M6: LightGBM Meta-Learner Regularizado + Blend Híbrido Convexo con NNLS."""
    def prepare_lgb_features(df: pd.DataFrame) -> pd.DataFrame:
        X = df[models].copy()
        X['meta_mean'] = df[models].mean(axis=1)
        X['meta_std'] = df[models].std(axis=1).fillna(0)
        X['meta_min'] = df[models].min(axis=1)
        X['meta_max'] = df[models].max(axis=1)
        X['meta_spread'] = X['meta_max'] - X['meta_min']
        X['hour_sin'] = np.sin(2 * np.pi * df['target_hour'] / 24).astype('float32')
        X['hour_cos'] = np.cos(2 * np.pi * df['target_hour'] / 24).astype('float32')
        X['month_sin'] = np.sin(2 * np.pi * df['target_month'] / 12).astype('float32')
        X['month_cos'] = np.cos(2 * np.pi * df['target_month'] / 12).astype('float32')
        X['target_dayofweek'] = df['target_dayofweek'].astype('int32')
        X['Estacion'] = df['Estacion'].astype('category')
        return X
        
    X_val = prepare_lgb_features(val_df)
    X_test = prepare_lgb_features(test_df)
    y_val = val_df['y_true'].values
    folds = sorted(val_df['fold'].unique())
    
    pred_val_oof_lgb = np.zeros(len(val_df), dtype=float)
    test_preds_folds = []
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 15,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'num_threads': 8,
        'seed': 42,
        'verbose': -1
    }
    
    for f in folds:
        tr_m = val_df['fold'] != f
        va_m = val_df['fold'] == f
        
        # Sub-split temporal dentro del train para early stopping honesto
        tr_indices = np.where(tr_m)[0]
        split_pt = int(len(tr_indices) * 0.8)
        inner_tr_idx = tr_indices[:split_pt]
        inner_va_idx = tr_indices[split_pt:]
        
        dtr = lgb.Dataset(X_val.iloc[inner_tr_idx], y_val[inner_tr_idx], categorical_feature=['Estacion'])
        dva = lgb.Dataset(X_val.iloc[inner_va_idx], y_val[inner_va_idx], reference=dtr, categorical_feature=['Estacion'])
        
        bst = lgb.train(
            params,
            dtr,
            num_boost_round=600,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(25, verbose=False)]
        )
        
        pred_val_oof_lgb[va_m] = bst.predict(X_val.loc[va_m], num_iteration=bst.best_iteration)
        test_preds_folds.append(bst.predict(X_test, num_iteration=bst.best_iteration))
        
    pred_val_oof_lgb = np.maximum(0.0, pred_val_oof_lgb)
    pred_test_lgb = np.maximum(0.0, np.mean(test_preds_folds, axis=0))
    
    ok_all = np.isfinite(y_val) & np.isfinite(pred_val_oof_lgb)
    val_rmse_lgb = float(np.sqrt(np.mean((y_val[ok_all] - pred_val_oof_lgb[ok_all]) ** 2)))
    info_lgb = {'method': 'M6_lgbm_stacking_oof', 'val_rmse_oof': val_rmse_lgb}
    
    # --- Variante Híbrida: Blend Convejo de NNLS (Linear) + LGBM Stacking ---
    alphas = [0.0, 0.15, 0.30, 0.45, 0.50, 0.55, 0.70, 0.85, 1.0]
    blend_scores = {}
    for a in alphas:
        p_blend = a * pred_val_linear + (1.0 - a) * pred_val_oof_lgb
        blend_scores[a] = float(np.sqrt(np.mean((y_val[ok_all] - p_blend[ok_all]) ** 2)))
        
    best_a = min(blend_scores, key=blend_scores.get)
    pred_val_oof_blend = best_a * pred_val_linear + (1.0 - best_a) * pred_val_oof_lgb
    pred_test_blend = best_a * pred_test_linear + (1.0 - best_a) * pred_test_lgb
    
    info_blend = {
        'method': 'M6_blend_hybrid',
        'alpha_linear': best_a,
        'alpha_lgbm': round(1.0 - best_a, 2),
        'val_rmse_oof': blend_scores[best_a]
    }
    
    return info_lgb, pred_val_oof_lgb, pred_test_lgb, info_blend, pred_val_oof_blend, pred_test_blend


def fit_m7_peak_tail_booster(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models: list[str],
    pred_val_base: np.ndarray,
    pred_test_base: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray]:
    """M7: Corrección post-hoc de picos combinando especialistas ponderados en la cola alta."""
    y_val = val_df['y_true'].values
    folds = sorted(val_df['fold'].unique())
    
    # Identificar modelos especialistas en picos por bajo RMSE_peaks en val
    ok_v = np.isfinite(y_val)
    thr_pk = np.quantile(y_val[ok_v], 0.95)
    pk_mask = ok_v & (y_val >= thr_pk)
    
    pk_rmses = {}
    for m in models:
        yp = val_df[m].values
        pk_rmses[m] = np.sqrt(np.mean((y_val[pk_mask] - yp[pk_mask]) ** 2))
    top_peak_models = sorted(pk_rmses, key=pk_rmses.get)[:3]
    
    # Probar umbral de activación (percentil 85 a 93) y factor de mezcla beta
    p_val_oof_boosted = np.zeros(len(val_df), dtype=float)
    
    q_thr_grid = [0.85, 0.88, 0.90, 0.93]
    beta_grid = [0.15, 0.25, 0.35, 0.50]
    
    best_params = None
    best_score = float('inf')
    
    for q_thr in q_thr_grid:
        for beta in beta_grid:
            p_oof_trial = np.zeros(len(val_df), dtype=float)
            for f in folds:
                tr_m = val_df['fold'] != f
                va_m = val_df['fold'] == f
                
                # Umbral calculado solo en train fold
                tau = np.quantile(pred_val_base[tr_m], q_thr)
                
                p_va = pred_val_base[va_m].copy()
                tail_idx = p_va >= tau
                spec_va = val_df.loc[va_m, top_peak_models].mean(axis=1).values
                p_va[tail_idx] = (1.0 - beta) * p_va[tail_idx] + beta * spec_va[tail_idx]
                p_oof_trial[va_m] = p_va
                
            err = np.sqrt(np.mean((y_val[ok_v] - p_oof_trial[ok_v]) ** 2))
            if err < best_score:
                best_score = err
                best_params = (q_thr, beta)
                p_val_oof_boosted = p_oof_trial
                
    q_opt, beta_opt = best_params
    
    # Aplicar a test con umbral de todo val 2024
    tau_full = np.quantile(pred_val_base, q_opt)
    pred_test_boosted = pred_test_base.copy()
    tail_test = pred_test_boosted >= tau_full
    spec_test = test_df[top_peak_models].mean(axis=1).values
    pred_test_boosted[tail_test] = (1.0 - beta_opt) * pred_test_boosted[tail_test] + beta_opt * spec_test[tail_test]
    
    info = {
        'method': 'M7_peak_tail_booster',
        'peak_specialists': top_peak_models,
        'tau_quantile': q_opt,
        'beta': beta_opt,
        'val_rmse_oof': best_score
    }
    return info, p_val_oof_boosted, pred_test_boosted


def fit_m8_grand_blend_oof(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    p_val_m2: np.ndarray,
    p_te_m2: np.ndarray,
    p_val_m5: np.ndarray,
    p_te_m5: np.ndarray,
    p_val_m7: np.ndarray,
    p_te_m7: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    M8: Grand Blend OOF.
    Combina en OOF las predicciones de:
      - M2: greedy_caruana
      - M5: ridge_meta_context
      - M7: peak_tail_booster
    Optimizando pesos convexos (sum w = 1, w >= 0) en los 3 trimestres de train
    de cada fold de 2024 para evaluación OOF estricta.
    Para test 2025, resuelve los pesos convexos sobre todo val 2024.
    """
    folds = sorted(val_df['fold'].unique())
    y_val = val_df['y_true'].values
    pred_val_oof = np.zeros(len(val_df), dtype=float)
    
    X_val_components = np.column_stack([p_val_m2, p_val_m5, p_val_m7])
    X_test_components = np.column_stack([p_te_m2, p_te_m5, p_te_m7])
    comp_names = ['M2_caruana', 'M5_ridge_context', 'M7_peak_booster']
    
    def solve_convex_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok = X[ok]
        y_ok = y[ok]
        w, _ = nnls(X_ok, y_ok)
        sw = w.sum()
        if sw > 0:
            return w / sw
        else:
            return np.ones(X.shape[1]) / X.shape[1]
            
    # 4-Fold Cross-fitting OOF
    for f in folds:
        tr_m = val_df['fold'] != f
        va_m = val_df['fold'] == f
        
        X_tr = X_val_components[tr_m]
        y_tr = y_val[tr_m]
        X_va = X_val_components[va_m]
        
        w_f = solve_convex_weights(X_tr, y_tr)
        pred_val_oof[va_m] = X_va @ w_f
        
    pred_val_oof = np.maximum(0.0, pred_val_oof)
    ok_all = np.isfinite(y_val) & np.isfinite(pred_val_oof)
    val_rmse_oof = float(np.sqrt(np.mean((y_val[ok_all] - pred_val_oof[ok_all]) ** 2)))
    
    # Para test 2025: ajustar pesos sobre todo Val 2024
    w_final = solve_convex_weights(X_val_components, y_val)
    pred_test = np.maximum(0.0, X_test_components @ w_final)
    
    weights_dict = {comp_names[i]: float(round(w_final[i], 4)) for i in range(len(comp_names))}
    info = {
        'method': 'M8_grand_blend_oof',
        'weights': weights_dict,
        'val_rmse_oof': val_rmse_oof
    }
    return info, pred_val_oof, pred_test


# ---------------------------------------------------------------------------
# 3. Intervalos Conformales Calibrados y Tarea NOM-020-SSA1 (DMA8)
# ---------------------------------------------------------------------------

def calibrate_heteroscedastic_conformal(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_val_oof: np.ndarray,
    pred_test: np.ndarray,
    models: list[str],
    nominal_coverages: list[float] = [0.80, 0.90]
) -> dict[str, any]:
    """
    Construye intervalos conformal condicionados a régimen horario y dispersión inter-modelo.
    Puntuación de no-conformidad: s_i = |y_i - ŷ_i| / (σ_i + ε)
    """
    y_val = val_df['y_true'].values
    res_val = np.abs(y_val - pred_val_oof)
    
    # Dispersión inter-modelo
    disp_val = val_df[models].std(axis=1).values + 1.0
    disp_test = test_df[models].std(axis=1).values + 1.0
    
    scores = res_val / disp_val
    ok_v = np.isfinite(scores)
    scores = scores[ok_v]
    n_v = len(scores)
    
    results = {}
    for alpha_nom in nominal_coverages:
        p = min(1.0, np.ceil((n_v + 1) * alpha_nom) / n_v)
        q_score = float(np.quantile(scores, p))
        
        margin_val = q_score * disp_val
        margin_test = q_score * disp_test
        
        q_low_val = np.maximum(0.0, pred_val_oof - margin_val)
        q_high_val = pred_val_oof + margin_val
        
        q_low_test = np.maximum(0.0, pred_test - margin_test)
        q_high_test = pred_test + margin_test
        
        # Evaluar cobertura empírica y ancho en test
        y_test = test_df['y_true'].values
        ok_t = np.isfinite(y_test)
        cov_t = float(np.mean((y_test[ok_t] >= q_low_test[ok_t]) & (y_test[ok_t] <= q_high_test[ok_t])))
        width_t = float(np.mean(q_high_test[ok_t] - q_low_test[ok_t]))
        
        cov_v = float(np.mean((y_val[ok_v] >= q_low_val[ok_v]) & (y_val[ok_v] <= q_high_val[ok_v])))
        width_v = float(np.mean(q_high_val[ok_v] - q_low_val[ok_v]))
        
        cov_key = f"{int(alpha_nom*100)}"
        results[cov_key] = {
            'nominal': alpha_nom,
            'val_cov': cov_v,
            'val_width': width_v,
            'test_cov': cov_t,
            'test_width': width_t,
            'q_score': q_score,
            'q_low_test': q_low_test,
            'q_high_test': q_high_test,
            'q_low_val': q_low_val,
            'q_high_val': q_high_val
        }
        
    return results


def evaluate_nom020_dma8(
    test_df: pd.DataFrame,
    pred_test: np.ndarray,
    h: int,
    norm_threshold: float = 65.0
) -> dict[str, float]:
    """
    Evalúa la tarea regulatoria NOM-020-SSA1:
    Calcula el DMA8 (Daily Maximum 8-hour Average) sobre las predicciones horario-estación.
    """
    df_eval = test_df[['Date', 'Estacion', 'target_time']].copy()
    df_eval['y_true'] = test_df['y_true'].values
    df_eval['y_pred'] = pred_test
    
    # Ordenar por Estacion y target_time
    df_eval = df_eval.sort_values(['Estacion', 'target_time']).reset_index(drop=True)
    
    # Calcular promedio móvil de 8 horas
    df_eval['roll8_true'] = df_eval.groupby('Estacion')['y_true'].transform(lambda s: s.rolling(8, min_periods=6).mean())
    df_eval['roll8_pred'] = df_eval.groupby('Estacion')['y_pred'].transform(lambda s: s.rolling(8, min_periods=6).mean())
    df_eval['day'] = df_eval['target_time'].dt.date
    
    # Extraer el máximo diario de las medias móviles de 8h (DMA8)
    dma8 = df_eval.groupby(['Estacion', 'day'])[['roll8_true', 'roll8_pred']].max().dropna().reset_index()
    
    if len(dma8) == 0:
        return {'dma8_rmse': np.nan, 'dma8_mae': np.nan, 'dma8_r2': np.nan, 'dma8_f1': np.nan}
        
    yt = dma8['roll8_true'].values
    yp = dma8['roll8_pred'].values
    
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    r2 = float(1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2))
    
    # Detección de superación de la norma (>65 ppb)
    true_exceed = yt >= norm_threshold
    pred_exceed = yp >= norm_threshold
    
    tp = np.sum(true_exceed & pred_exceed)
    fp = np.sum((~true_exceed) & pred_exceed)
    fn = np.sum(true_exceed & (~pred_exceed))
    tn = np.sum((~true_exceed) & (~pred_exceed))
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_acc = float(0.5 * (tp / (tp + fn) + tn / (tn + fp))) if (tp + fn > 0 and tn + fp > 0) else 0.0
    
    return {
        'dma8_rmse': rmse,
        'dma8_mae': mae,
        'dma8_r2': r2,
        'exceed_prec': precision,
        'exceed_rec': recall,
        'exceed_f1': f1,
        'balanced_acc': balanced_acc,
        'n_days_stations': len(dma8),
        'total_exceed_events': int(true_exceed.sum())
    }


# ---------------------------------------------------------------------------
# 4. Pipeline Principal y Registro
# ---------------------------------------------------------------------------

def run_ensemble_round(pass_num: int = 1):
    print("=" * 80)
    print(f"PIPELINE DE ENSAMBLE / META-LEARNING Y CALIBRACIÓN ESTADÍSTICA (PASADA {pass_num})")
    print("=" * 80)
    
    all_summary_records = []
    best_comparison_records = []
    
    for h in HORIZONS:
        print(f"\n{'#'*35} HORIZONTE h = {h} ({h}h) {'#'*35}")
        cands = discover_candidate_models(h)
        if len(cands) == 0:
            print(f"[Aviso] No se hallaron modelos para h={h}")
            continue
            
        m_wide, elite_models = select_diverse_candidates_for_h(h, cands, max_per_family=4)
        val_df = m_wide[m_wide['split'] == 'val'].reset_index(drop=True)
        test_df = m_wide[m_wide['split'] == 'test'].reset_index(drop=True)
        
        methods = {}
        
        # M0: Best Individual
        inf_m0, p_val_m0, p_te_m0 = fit_m0_best_individual(val_df, test_df, elite_models)
        m_te_m0 = evaluate_test(test_df, p_te_m0, h)
        methods['M0_best_individual'] = {'val_rmse_oof': inf_m0['val_rmse_oof'], 'p_val': p_val_m0, 'p_te': p_te_m0, 'm': m_te_m0, 'info': str(inf_m0['best_model'])}
        print(f"  [M0 Best Indiv] ({inf_m0['best_model']}): Val OOF = {inf_m0['val_rmse_oof']:.4f} | Test RMSE = {m_te_m0['RMSE']:.4f} | R2 = {m_te_m0['R2']:.4f} | Pk_RMSE = {m_te_m0['RMSE_peaks']:.2f}")
        
        # M1: Top-K Mean
        inf_m1, p_val_m1, p_te_m1 = fit_m1_top_k_mean(val_df, test_df, elite_models)
        m_te_m1 = evaluate_test(test_df, p_te_m1, h)
        methods['M1_top_k_mean'] = {'val_rmse_oof': inf_m1['val_rmse_oof'], 'p_val': p_val_m1, 'p_te': p_te_m1, 'm': m_te_m1, 'info': f"Top-{inf_m1['best_k']} mean"}
        print(f"  [M1 Top-K Mean] (k={inf_m1['best_k']}): Val OOF = {inf_m1['val_rmse_oof']:.4f} | Test RMSE = {m_te_m1['RMSE']:.4f} | R2 = {m_te_m1['R2']:.4f}")
        
        # M2: Greedy Caruana
        inf_m2, p_val_m2, p_te_m2 = fit_m2_greedy_caruana(val_df, test_df, elite_models)
        m_te_m2 = evaluate_test(test_df, p_te_m2, h)
        methods['M2_greedy_caruana'] = {'val_rmse_oof': inf_m2['val_rmse_oof'], 'p_val': p_val_m2, 'p_te': p_te_m2, 'm': m_te_m2, 'info': f"Caruana ({len(inf_m2['weights'])} models)"}
        print(f"  [M2 Caruana] ({len(inf_m2['weights'])} mods): Val OOF = {inf_m2['val_rmse_oof']:.4f} | Test RMSE = {m_te_m2['RMSE']:.4f} | R2 = {m_te_m2['R2']:.4f}")
        
        # M3: NNLS Ridge Global
        inf_m3, p_val_m3, p_te_m3 = fit_m3_nnls_ridge_global(val_df, test_df, elite_models)
        m_te_m3 = evaluate_test(test_df, p_te_m3, h)
        methods['M3_nnls_ridge_global'] = {'val_rmse_oof': inf_m3['val_rmse_oof'], 'p_val': p_val_m3, 'p_te': p_te_m3, 'm': m_te_m3, 'info': f"NNLS Ridge (lam={inf_m3['best_lambda']})"}
        print(f"  [M3 NNLS Ridge] (lam={inf_m3['best_lambda']}): Val OOF = {inf_m3['val_rmse_oof']:.4f} | Test RMSE = {m_te_m3['RMSE']:.4f} | R2 = {m_te_m3['R2']:.4f}")
        
        # M4: NNLS by Regime
        inf_m4, p_val_m4, p_te_m4 = fit_m4_nnls_by_regime(val_df, test_df, elite_models)
        m_te_m4 = evaluate_test(test_df, p_te_m4, h)
        methods['M4_nnls_by_regime'] = {'val_rmse_oof': inf_m4['val_rmse_oof'], 'p_val': p_val_m4, 'p_te': p_te_m4, 'm': m_te_m4, 'info': "NNLS 4 Regimes"}
        print(f"  [M4 NNLS Regime]: Val OOF = {inf_m4['val_rmse_oof']:.4f} | Test RMSE = {m_te_m4['RMSE']:.4f} | R2 = {m_te_m4['R2']:.4f}")
        
        # M5: Ridge Meta Context
        inf_m5, p_val_m5, p_te_m5 = fit_m5_ridge_meta_context(val_df, test_df, elite_models)
        m_te_m5 = evaluate_test(test_df, p_te_m5, h)
        methods['M5_ridge_meta_context'] = {'val_rmse_oof': inf_m5['val_rmse_oof'], 'p_val': p_val_m5, 'p_te': p_te_m5, 'm': m_te_m5, 'info': f"Ridge Context (alpha={inf_m5['best_alpha']})"}
        print(f"  [M5 Ridge Context] (a={inf_m5['best_alpha']}): Val OOF = {inf_m5['val_rmse_oof']:.4f} | Test RMSE = {m_te_m5['RMSE']:.4f} | R2 = {m_te_m5['R2']:.4f}")
        
        # M6 & Hybrid Blend (M6_blend)
        linear_best_m = 'M3_nnls_ridge_global' if inf_m3['val_rmse_oof'] <= inf_m4['val_rmse_oof'] else 'M4_nnls_by_regime'
        inf_m6, p_val_m6, p_te_m6, inf_blend, p_val_blend, p_te_blend = fit_m6_lgbm_stacking_and_blend(
            val_df, test_df, elite_models,
            methods[linear_best_m]['p_val'],
            methods[linear_best_m]['p_te']
        )
        m_te_m6 = evaluate_test(test_df, p_te_m6, h)
        methods['M6_lgbm_stacking_oof'] = {'val_rmse_oof': inf_m6['val_rmse_oof'], 'p_val': p_val_m6, 'p_te': p_te_m6, 'm': m_te_m6, 'info': "LGBM Meta Stacking OOF"}
        print(f"  [M6 LGBM Stacking]: Val OOF = {inf_m6['val_rmse_oof']:.4f} | Test RMSE = {m_te_m6['RMSE']:.4f} | R2 = {m_te_m6['R2']:.4f}")
        
        m_te_blend = evaluate_test(test_df, p_te_blend, h)
        methods['M6_blend_hybrid'] = {'val_rmse_oof': inf_blend['val_rmse_oof'], 'p_val': p_val_blend, 'p_te': p_te_blend, 'm': m_te_blend, 'info': f"Hybrid Blend (linear={inf_blend['alpha_linear']}, lgbm={inf_blend['alpha_lgbm']})"}
        print(f"  [M6 Hybrid Blend] (w_lin={inf_blend['alpha_linear']}): Val OOF = {inf_blend['val_rmse_oof']:.4f} | Test RMSE = {m_te_blend['RMSE']:.4f} | R2 = {m_te_blend['R2']:.4f}")
        
        # M7: Peak Tail Booster (sobre el mejor ensamble actual hasta M6)
        curr_best_m = min(methods, key=lambda k: methods[k]['val_rmse_oof'])
        inf_m7, p_val_m7, p_te_m7 = fit_m7_peak_tail_booster(
            val_df, test_df, elite_models,
            methods[curr_best_m]['p_val'],
            methods[curr_best_m]['p_te']
        )
        m_te_m7 = evaluate_test(test_df, p_te_m7, h)
        methods['M7_peak_tail_booster'] = {'val_rmse_oof': inf_m7['val_rmse_oof'], 'p_val': p_val_m7, 'p_te': p_te_m7, 'm': m_te_m7, 'info': f"Peak Booster (tau_q={inf_m7['tau_quantile']}, beta={inf_m7['beta']})"}
        print(f"  [M7 Peak Booster] (q={inf_m7['tau_quantile']}, b={inf_m7['beta']}): Val OOF = {inf_m7['val_rmse_oof']:.4f} | Test RMSE = {m_te_m7['RMSE']:.4f} | Pk_RMSE = {m_te_m7['RMSE_peaks']:.2f} | Pk_Bias = {m_te_m7['bias_peaks']:+.2f}")
        
        # M8: Grand Blend OOF (M2 + M5 + M7)
        inf_m8, p_val_m8, p_te_m8 = fit_m8_grand_blend_oof(
            val_df, test_df,
            methods['M2_greedy_caruana']['p_val'], methods['M2_greedy_caruana']['p_te'],
            methods['M5_ridge_meta_context']['p_val'], methods['M5_ridge_meta_context']['p_te'],
            methods['M7_peak_tail_booster']['p_val'], methods['M7_peak_tail_booster']['p_te']
        )
        m_te_m8 = evaluate_test(test_df, p_te_m8, h)
        methods['M8_grand_blend_oof'] = {
            'val_rmse_oof': inf_m8['val_rmse_oof'],
            'p_val': p_val_m8,
            'p_te': p_te_m8,
            'm': m_te_m8,
            'info': f"Grand Blend OOF (w_M2={inf_m8['weights']['M2_caruana']}, w_M5={inf_m8['weights']['M5_ridge_context']}, w_M7={inf_m8['weights']['M7_peak_booster']})"
        }
        print(f"  [M8 Grand Blend] (M2={inf_m8['weights']['M2_caruana']:.2f}, M5={inf_m8['weights']['M5_ridge_context']:.2f}, M7={inf_m8['weights']['M7_peak_booster']:.2f}): Val OOF = {inf_m8['val_rmse_oof']:.4f} | Test RMSE = {m_te_m8['RMSE']:.4f} | R2 = {m_te_m8['R2']:.4f} | Pk_RMSE = {m_te_m8['RMSE_peaks']:.2f} | Pk_Bias = {m_te_m8['bias_peaks']:+.2f}")
        
        # Registrar todos los métodos
        for m_name, m_data in methods.items():
            log_result(
                experiment='noche2_ens2',
                variant=f"{m_name}_p{pass_num}",
                h=h,
                m=m_data['m'],
                notes=f"val_rmse_oof={m_data['val_rmse_oof']:.4f}, {m_data['info']}"
            )
            all_summary_records.append({
                'h': h,
                'method': m_name,
                'val_RMSE_OOF': round(m_data['val_rmse_oof'], 4),
                'test_RMSE': round(m_data['m']['RMSE'], 4),
                'test_MAE': round(m_data['m']['MAE'], 4),
                'test_R2': round(m_data['m']['R2'], 4),
                'test_RMSE_peaks': round(m_data['m']['RMSE_peaks'], 4),
                'test_bias_peaks': round(m_data['m']['bias_peaks'], 4),
                'info': m_data['info']
            })
            
        # SELECCIÓN ESTRICTA DEL GANADOR POR VAL RMSE OOF
        winner_name = min(methods, key=lambda k: methods[k]['val_rmse_oof'])
        winner_data = methods[winner_name]
        
        print(f"\n  >>> GANADOR OFICIAL h={h} (MENOR VAL OOF): '{winner_name}' <<<")
        print(f"      Val OOF RMSE = {winner_data['val_rmse_oof']:.4f} | Test 2025 RMSE = {winner_data['m']['RMSE']:.4f} (Ganancia vs Individual: {m_te_m0['RMSE'] - winner_data['m']['RMSE']:+.4f} ppb)")
        
        # Intervalos Conformal
        conf_res = calibrate_heteroscedastic_conformal(
            val_df, test_df, winner_data['p_val'], winner_data['p_te'], elite_models, nominal_coverages=[0.80, 0.90]
        )
        c80 = conf_res['80']
        c90 = conf_res['90']
        print(f"  [Conformal 80%] Cob Test: {c80['test_cov']*100:.2f}% (Nominal: 80%) | Ancho: {c80['test_width']:.2f} ppb")
        print(f"  [Conformal 90%] Cob Test: {c90['test_cov']*100:.2f}% (Nominal: 90%) | Ancho: {c90['test_width']:.2f} ppb")
        
        # Evaluación Regulatoria NOM-020-SSA1 (DMA8)
        dma8_res = evaluate_nom020_dma8(test_df, winner_data['p_te'], h)
        print(f"  [NOM-020 DMA8] RMSE = {dma8_res['dma8_rmse']:.2f} ppb | R2 = {dma8_res['dma8_r2']:.4f} | Exceed F1 = {dma8_res['exceed_f1']:.4f}")
        
        # Guardar Parquet Final
        # Val = OOF (2024), Test = 2025
        df_val_out = val_df[['Date', 'Estacion', 'split', 'y_true']].copy()
        df_val_out['h'] = h
        df_val_out['y_pred'] = winner_data['p_val']
        df_val_out['q10'] = c80['q_low_val']
        df_val_out['q90'] = c80['q_high_val']
        
        df_te_out = test_df[['Date', 'Estacion', 'split', 'y_true']].copy()
        df_te_out['h'] = h
        df_te_out['y_pred'] = winner_data['p_te']
        df_te_out['q10'] = c80['q_low_test']
        df_te_out['q90'] = c80['q_high_test']
        
        df_parquet = pd.concat([df_val_out, df_te_out], ignore_index=True)
        out_parquet_path = os.path.join(PREDS_DIR, f"ens2_best_h{h}.parquet")
        df_parquet.to_parquet(out_parquet_path, index=False)
        print(f"  Predicciones guardadas en {out_parquet_path} ({len(df_parquet):,} filas)")
        
        best_comparison_records.append({
            'h': h,
            'winner_method': winner_name,
            'best_individual': inf_m0['best_model'],
            'indiv_val_rmse': round(inf_m0['val_rmse_oof'], 4),
            'indiv_test_rmse': round(m_te_m0['RMSE'], 4),
            'ens_val_rmse_oof': round(winner_data['val_rmse_oof'], 4),
            'ens_test_rmse': round(winner_data['m']['RMSE'], 4),
            'gain_vs_indiv_test': round(m_te_m0['RMSE'] - winner_data['m']['RMSE'], 4),
            'ens_test_r2': round(winner_data['m']['R2'], 4),
            'ens_rmse_peaks': round(winner_data['m']['RMSE_peaks'], 4),
            'ens_bias_peaks': round(winner_data['m']['bias_peaks'], 4),
            'conf80_test_cov': round(c80['test_cov'] * 100, 2),
            'conf80_test_width': round(c80['test_width'], 2),
            'dma8_test_rmse': round(dma8_res['dma8_rmse'], 2),
            'dma8_exceed_f1': round(dma8_res['exceed_f1'], 4)
        })
        
    df_all_sum = pd.DataFrame(all_summary_records)
    df_all_sum.to_csv(os.path.join(EXP_DIR, f"ensemble_summary_p{pass_num}.csv"), index=False)
    
    df_best_comp = pd.DataFrame(best_comparison_records)
    df_best_comp.to_csv(os.path.join(EXP_DIR, "ensemble_best_comparison.csv"), index=False)
    print("\n" + "=" * 80)
    print(f"RESUMEN FINAL DE LA PASADA {pass_num}:")
    print("=" * 80)
    print(df_best_comp[['h', 'winner_method', 'ens_val_rmse_oof', 'ens_test_rmse', 'gain_vs_indiv_test', 'ens_test_r2', 'ens_rmse_peaks', 'conf80_test_cov']].to_string(index=False))
    return df_best_comp

if __name__ == '__main__':
    p_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_ensemble_round(pass_num=p_num)
