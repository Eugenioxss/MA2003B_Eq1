"""
experiments/noche/ensemble/ensemble.py
======================================
Pipeline de Ensamble y Stacking Reutilizable para la Corrida Nocturna SIMA Monterrey.

Características:
1. Escaneo Automático:
   - Descubre dinámicamente predicciones en `experiments/noche/*/preds/*_h<h>.parquet`
   - Excluye automáticamente variantes no honestas (nombres con '_pp' o 'pp_')
   - Excluye variantes sin split 'val' (2024)
2. Alineación Rigurosa:
   - Une por (Date, Estacion, split)
   - Extrae el target real observado
3. Validación Temporal Estricta OUT-OF-FOLD (OOF):
   - Para evitar sobreajuste, todos los métodos (top-k, NNLS, Stacking LGBM, calibración)
     generan predicciones de validación 2024 de forma estrictamente OUT-OF-FOLD (OOF)
     mediante validación cruzada temporal por trimestres (Q1..Q4) dentro de 2024.
   - La selección del mejor ensamble se realiza sobre el menor val_RMSE OOF.
   - Las predicciones guardadas en preds/ensemble_best_h<h>.parquet para split='val' son OOF.
4. Métodos de Ensamble Evaluados:
   (a) best_individual: Mejor modelo individual según val
   (b) simple_avg_topk: Promedio simple no ponderado de top-k (k in [2, 3, 5] seleccionado OOF)
   (c) nnls_global: Pesos no negativos normalizados (sum w = 1) globales por h
   (d) nnls_hourly_block: Pesos no negativos ajustados por h x bloque diurno
   (e) stacking_lgbm: Meta-aprendiz LightGBM con regularización y OOF cross-validation
5. Intervalos de Predicción Conformal (80% Cobertura Nominal):
   - Calibración por residuos absolutos condicionados a bloque diurno en val OOF
   - Cobertura empírica y ancho medio en test 2025
   - Guarda q10 y q90 en el parquet final
6. Calibración Post-hoc de Picos:
   - Calibración isotónica (IsotonicRegression) evaluada OOF en val y aplicada a test
7. Exportación y Registro:
   - Registra resultados en results_registry.csv (experiment='noche_ensemble')
   - Guarda predicciones val (OOF) + test en preds/ensemble_best_h<h>.parquet
   - Guarda tabla de resumen en ensemble_summary.csv
"""
from __future__ import annotations
import os
import sys
import glob
import re
import time
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import evaluate_test, log_result, HORIZONS, MODEL_STATIONS

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(EXP_DIR, 'preds')
FIGS_DIR = os.path.join(EXP_DIR, 'figures')
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Escaneo y Filtrado de Predicciones
# ---------------------------------------------------------------------------
def discover_prediction_files(base_dir: str = 'experiments/noche') -> dict[int, dict[str, str]]:
    """
    Escanea recursivamente los parquets de predicciones y clasifica por horizonte h.
    Excluye automáticamente modelos no honestos (_pp, pp_) y sin val.
    Devuelve: {h: {model_name: file_path}}
    """
    pattern = os.path.join(ROOT, base_dir, '*', 'preds', '*.parquet')
    all_files = glob.glob(pattern)
    
    models_by_h: dict[int, dict[str, str]] = {h: {} for h in HORIZONS}
    excluded_files = []
    
    for fpath in all_files:
        fname = os.path.basename(fpath)
        if 'ensemble_' in fname:
            continue
            
        m = re.match(r'(.+)_h(\d+)\.parquet', fname)
        if not m:
            continue
            
        model_name, h_str = m.groups()
        h = int(h_str)
        if h not in models_by_h:
            models_by_h[h] = {}
            
        # Regla 1: Excluir variantes no honestas (perfect prog)
        if '_pp' in model_name.lower() or 'pp_' in model_name.lower():
            excluded_files.append((fname, 'Contiene _pp o pp_ (No honesto)'))
            continue
            
        # Regla 2: Verificar existencia de split 'val'
        try:
            df_head = pd.read_parquet(fpath)
            if 'split' not in df_head.columns:
                excluded_files.append((fname, 'No tiene columna split'))
                continue
            splits = df_head['split'].unique()
            if 'val' not in splits:
                excluded_files.append((fname, 'Falta split val (2024)'))
                continue
            if 'test' not in splits:
                excluded_files.append((fname, 'Falta split test (2025)'))
                continue
                
            models_by_h[h][model_name] = fpath
        except Exception as e:
            excluded_files.append((fname, f'Error al leer: {e}'))
            
    print(f"Descubrimiento de modelos finalizado: {len(all_files)} archivos escaneados.", flush=True)
    if excluded_files:
        print(f"Modelos excluidos ({len(excluded_files)}):", flush=True)
        for fn, r in sorted(excluded_files)[:8]:
            print(f"  - {fn}: {r}", flush=True)
        if len(excluded_files) > 8:
            print(f"  ... y {len(excluded_files)-8} más.", flush=True)
            
    return models_by_h


def load_and_align_models_for_h(h: int, model_paths: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Carga todos los modelos honestos para el horizonte h y los une en una matriz ancha
    indexada por (Date, Estacion, split), garantizando que Date sea la fecha de emisión t.
    """
    dfs = {}
    y_true_series = None
    
    for mod_name, fpath in sorted(model_paths.items()):
        df = pd.read_parquet(fpath)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Estacion'] = df['Estacion'].astype(str)
        
        indexed = df.set_index(['Date', 'Estacion', 'split'])
        dfs[mod_name] = indexed['y_pred']
        
        if y_true_series is None and 'y_true' in indexed.columns:
            y_true_series = indexed['y_true']
            
    if not dfs:
        raise ValueError(f"No hay modelos honestos válidos disponibles para h={h}")
        
    m_wide = pd.DataFrame(dfs)
    if y_true_series is not None:
        m_wide['y_true'] = y_true_series
        m_wide[f'Target_{h}h'] = y_true_series
        
    m_wide = m_wide.reset_index()
    
    # Calendario basado en el momento objetivo (t + h)
    m_wide['target_time'] = m_wide['Date'] + pd.Timedelta(hours=h)
    m_wide['target_hour'] = m_wide['target_time'].dt.hour
    m_wide['target_month'] = m_wide['target_time'].dt.month
    m_wide['target_dayofweek'] = m_wide['target_time'].dt.dayofweek
    
    # Folds temporales dentro de 2024 por trimestre: Q1 (0), Q2 (1), Q3 (2), Q4 (3)
    m_wide['fold'] = (m_wide['target_month'] - 1) // 3
    
    # Bloque diurno objetivo:
    # 0: Noche (00-06h), 1: Mañana (07-11h), 2: Pico solar (12-17h), 3: Tarde/Noche (18-23h)
    def assign_block(hr):
        if hr <= 6: return 0
        elif hr <= 11: return 1
        elif hr <= 17: return 2
        else: return 3
    m_wide['target_block'] = m_wide['target_hour'].apply(assign_block)
    
    # Imputar NaNs residuales si faltara algún modelo en alguna hora
    mod_cols = list(dfs.keys())
    nan_counts = m_wide[mod_cols].isna().sum()
    if (nan_counts > 0).any():
        row_mean = m_wide[mod_cols].mean(axis=1)
        for c in mod_cols:
            m_wide[c] = m_wide[c].fillna(row_mean)
            
    return m_wide, mod_cols


# ---------------------------------------------------------------------------
# 2. Métodos de Ensamble con Out-Of-Fold (OOF) CV
# ---------------------------------------------------------------------------
def fit_best_individual_oof(val_df: pd.DataFrame, test_df: pd.DataFrame, models: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """(a) Mejor modelo individual por RMSE en val 2024."""
    y_val = val_df['y_true'].values
    rmses = {}
    for m in models:
        yp = val_df[m].values
        ok = np.isfinite(y_val) & np.isfinite(yp)
        rmses[m] = float(np.sqrt(np.mean((y_val[ok] - yp[ok]) ** 2)))
        
    best_m = min(rmses, key=rmses.get)
    pred_val_oof = val_df[best_m].values.copy()
    pred_test = test_df[best_m].values.copy()
    info = {'best_model': best_m, 'val_rmse_oof': rmses[best_m], 'all_rmses': rmses}
    return info, pred_val_oof, pred_test


def fit_simple_average_topk_oof(val_df: pd.DataFrame, test_df: pd.DataFrame, models: list[str], k_candidates: list[int] = [2, 3, 5]) -> tuple[dict, np.ndarray, np.ndarray]:
    """(b) Promedio simple de top-k modelos con validación cruzada temporal OOF dentro de 2024."""
    y_val = val_df['y_true'].values
    folds = sorted(val_df['fold'].unique())
    
    k_oof_scores = {}
    k_oof_preds = {}
    
    for k in k_candidates:
        if k > len(models):
            continue
        p_oof = np.zeros(len(val_df), dtype=float)
        
        for fold in folds:
            train_mask = val_df['fold'] != fold
            val_mask = val_df['fold'] == fold
            
            y_tr = val_df.loc[train_mask, 'y_true'].values
            rmses_fold = {}
            for m in models:
                yp = val_df.loc[train_mask, m].values
                ok = np.isfinite(y_tr) & np.isfinite(yp)
                rmses_fold[m] = np.mean((y_tr[ok] - yp[ok]) ** 2)
            top_k_fold = sorted(rmses_fold, key=rmses_fold.get)[:k]
            p_oof[val_mask] = val_df.loc[val_mask, top_k_fold].mean(axis=1).values
            
        ok_all = np.isfinite(y_val) & np.isfinite(p_oof)
        score_k = float(np.sqrt(np.mean((y_val[ok_all] - p_oof[ok_all]) ** 2)))
        k_oof_scores[k] = score_k
        k_oof_preds[k] = p_oof
        
    best_k = min(k_oof_scores, key=k_oof_scores.get)
    pred_val_oof = k_oof_preds[best_k]
    
    # Para test 2025: seleccionar top-k entrenados con todo 2024
    all_rmses = {}
    for m in models:
        yp = val_df[m].values
        ok = np.isfinite(y_val) & np.isfinite(yp)
        all_rmses[m] = np.mean((y_val[ok] - yp[ok]) ** 2)
    final_topk = sorted(all_rmses, key=all_rmses.get)[:best_k]
    pred_test = test_df[final_topk].mean(axis=1).values
    
    info = {'k': best_k, 'selected_models': final_topk, 'val_rmse_oof': k_oof_scores[best_k]}
    return info, pred_val_oof, pred_test


def fit_nnls_global_oof(val_df: pd.DataFrame, test_df: pd.DataFrame, models: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """(c) Pesos no negativos (NNLS) normalizados con validación cruzada temporal OOF."""
    folds = sorted(val_df['fold'].unique())
    pred_val_oof = np.zeros(len(val_df), dtype=float)
    y_val = val_df['y_true'].values
    
    # 1. Generar predicciones OOF fold a fold
    for fold in folds:
        train_mask = val_df['fold'] != fold
        val_mask = val_df['fold'] == fold
        
        X_tr = val_df.loc[train_mask, models].values
        y_tr = val_df.loc[train_mask, 'y_true'].values
        X_va = val_df.loc[val_mask, models].values
        
        ok_tr = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
        w_f, _ = nnls(X_tr[ok_tr], y_tr[ok_tr])
        sum_w = w_f.sum()
        w_norm = w_f / sum_w if sum_w > 0 else np.ones(len(models)) / len(models)
        
        pred_val_oof[val_mask] = X_va @ w_norm
        
    ok_all = np.isfinite(y_val) & np.isfinite(pred_val_oof)
    val_rmse_oof = float(np.sqrt(np.mean((y_val[ok_all] - pred_val_oof[ok_all]) ** 2)))
    
    # 2. Entrenar pesos finales sobre todo 2024 para test 2025
    X_val_all = val_df[models].values
    ok_all_tr = np.isfinite(y_val) & np.isfinite(X_val_all).all(axis=1)
    w_final, _ = nnls(X_val_all[ok_all_tr], y_val[ok_all_tr])
    sum_wf = w_final.sum()
    w_final_norm = w_final / sum_wf if sum_wf > 0 else np.ones(len(models)) / len(models)
    
    X_test = test_df[models].values
    pred_test = X_test @ w_final_norm
    
    weights_dict = {m: float(round(w_final_norm[i], 4)) for i, m in enumerate(models)}
    info = {'weights': weights_dict, 'val_rmse_oof': val_rmse_oof}
    return info, pred_val_oof, pred_test


def fit_nnls_hourly_block_oof(val_df: pd.DataFrame, test_df: pd.DataFrame, models: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """(d) Pesos no negativos por bloque diurno con validación cruzada temporal OOF."""
    folds = sorted(val_df['fold'].unique())
    pred_val_oof = np.zeros(len(val_df), dtype=float)
    y_val = val_df['y_true'].values
    blocks = [0, 1, 2, 3]
    
    # 1. OOF fold a fold por bloque
    for fold in folds:
        train_mask = val_df['fold'] != fold
        val_mask = val_df['fold'] == fold
        
        for b in blocks:
            mb_tr = train_mask & (val_df['target_block'] == b)
            mb_va = val_mask & (val_df['target_block'] == b)
            
            X_tr = val_df.loc[mb_tr, models].values
            y_tr = val_df.loc[mb_tr, 'y_true'].values
            X_va = val_df.loc[mb_va, models].values
            
            ok_tr = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
            if ok_tr.sum() > len(models):
                w_b, _ = nnls(X_tr[ok_tr], y_tr[ok_tr])
                sw = w_b.sum()
                w_norm = w_b / sw if sw > 0 else np.ones(len(models)) / len(models)
            else:
                w_norm = np.ones(len(models)) / len(models)
                
            pred_val_oof[mb_va] = X_va @ w_norm
            
    ok_all = np.isfinite(y_val) & np.isfinite(pred_val_oof)
    val_rmse_oof = float(np.sqrt(np.mean((y_val[ok_all] - pred_val_oof[ok_all]) ** 2)))
    
    # 2. Entrenar pesos finales por bloque sobre todo 2024 para test 2025
    pred_test = np.zeros(len(test_df), dtype=float)
    weights_by_block = {}
    for b in blocks:
        mb_all = val_df['target_block'] == b
        mb_te = test_df['target_block'] == b
        
        X_tr = val_df.loc[mb_all, models].values
        y_tr = val_df.loc[mb_all, 'y_true'].values
        X_te = test_df.loc[mb_te, models].values
        
        ok_tr = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
        w_b, _ = nnls(X_tr[ok_tr], y_tr[ok_tr])
        sw = w_b.sum()
        w_norm = w_b / sw if sw > 0 else np.ones(len(models)) / len(models)
        pred_test[mb_te] = X_te @ w_norm
        weights_by_block[f'block_{b}'] = {m: float(round(w_norm[i], 4)) for i, m in enumerate(models)}
        
    info = {'weights_by_block': weights_by_block, 'val_rmse_oof': val_rmse_oof}
    return info, pred_val_oof, pred_test


def fit_stacking_lightgbm_oof(val_df: pd.DataFrame, test_df: pd.DataFrame, models: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """(e) Stacking meta-learner LightGBM con validación cruzada temporal OOF dentro de 2024."""
    def prepare_meta(df: pd.DataFrame):
        X = df[models].copy()
        X['meta_mean'] = df[models].mean(axis=1)
        X['meta_std'] = df[models].std(axis=1).fillna(0)
        X['meta_min'] = df[models].min(axis=1)
        X['meta_max'] = df[models].max(axis=1)
        X['hour_sin'] = np.sin(2 * np.pi * df['target_hour'] / 24).astype('float32')
        X['hour_cos'] = np.cos(2 * np.pi * df['target_hour'] / 24).astype('float32')
        X['month_sin'] = np.sin(2 * np.pi * df['target_month'] / 12).astype('float32')
        X['month_cos'] = np.cos(2 * np.pi * df['target_month'] / 12).astype('float32')
        X['target_dayofweek'] = df['target_dayofweek'].astype('int32')
        X['Estacion'] = df['Estacion'].astype('category')
        return X
        
    X_meta_val = prepare_meta(val_df)
    X_meta_test = prepare_meta(test_df)
    y_val = val_df['y_true'].values
    
    folds = sorted(val_df['fold'].unique())
    pred_val_oof = np.zeros(len(val_df), dtype=float)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 15,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'num_threads': 3,
        'seed': 42,
        'verbose': -1
    }
    
    # 1. OOF fold a fold
    best_iters = []
    test_preds_folds = []
    
    for fold in folds:
        tr_mask = val_df['fold'] != fold
        va_mask = val_df['fold'] == fold
        
        dtr = lgb.Dataset(X_meta_val[tr_mask], y_val[tr_mask], categorical_feature=['Estacion'])
        dva = lgb.Dataset(X_meta_val[va_mask], y_val[va_mask], reference=dtr, categorical_feature=['Estacion'])
        
        bst_f = lgb.train(
            params,
            dtr,
            num_boost_round=600,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        pred_val_oof[va_mask] = bst_f.predict(X_meta_val[va_mask], num_iteration=bst_f.best_iteration)
        test_preds_folds.append(bst_f.predict(X_meta_test, num_iteration=bst_f.best_iteration))
        best_iters.append(int(bst_f.best_iteration))
        
    ok_all = np.isfinite(y_val) & np.isfinite(pred_val_oof)
    val_rmse_oof = float(np.sqrt(np.mean((y_val[ok_all] - pred_val_oof[ok_all]) ** 2)))
    
    # Predicción en test: ensamble de los modelos OOF (bagging de folds temporales)
    pred_test = np.mean(test_preds_folds, axis=0)
    
    info = {'val_rmse_oof': val_rmse_oof, 'mean_best_iter': int(np.mean(best_iters)), 'iters_by_fold': best_iters}
    return info, pred_val_oof, pred_test


# ---------------------------------------------------------------------------
# 3. Intervalos Conformal y Calibración Isotónica OOF
# ---------------------------------------------------------------------------
def calibrate_conformal_intervals_oof(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_val_oof: np.ndarray,
    pred_test: np.ndarray,
    coverage_target: float = 0.80
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Construye intervalos conformal condicionados a bloque diurno usando residuos OOF de val 2024."""
    y_val = val_df['y_true'].values
    res_abs_val = np.abs(y_val - pred_val_oof)
    
    q_bounds = {}
    blocks = [0, 1, 2, 3]
    for b in blocks:
        m_b = (val_df['target_block'] == b) & np.isfinite(res_abs_val)
        res_b = res_abs_val[m_b]
        n_b = len(res_b)
        alpha = 1.0 - coverage_target
        p = min(1.0, np.ceil((n_b + 1) * (1.0 - alpha)) / n_b)
        q_val = float(np.quantile(res_b, p))
        q_bounds[b] = q_val
        
    test_blocks = test_df['target_block'].values
    test_q = np.array([q_bounds[b] for b in test_blocks])
    
    q10_test = np.maximum(0.0, pred_test - test_q)
    q90_test = pred_test + test_q
    
    y_test = test_df['y_true'].values
    ok_t = np.isfinite(y_test)
    covered = (y_test[ok_t] >= q10_test[ok_t]) & (y_test[ok_t] <= q90_test[ok_t])
    empirical_coverage = float(np.mean(covered))
    mean_width = float(np.mean(q90_test[ok_t] - q10_test[ok_t]))
    
    conf_stats = {
        'target_coverage': coverage_target,
        'empirical_coverage': empirical_coverage,
        'mean_interval_width': mean_width,
        'q_by_block': q_bounds
    }
    return q10_test, q90_test, conf_stats


def evaluate_isotonic_calibration_oof(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_val_oof: np.ndarray,
    pred_test: np.ndarray,
    h: int
) -> tuple[np.ndarray, dict]:
    """Ajusta calibración isotónica de ŷ sobre val OOF y evalúa en test 2025."""
    y_val = val_df['y_true'].values
    ok_v = np.isfinite(y_val) & np.isfinite(pred_val_oof)
    
    iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
    iso.fit(pred_val_oof[ok_v], y_val[ok_v])
    
    pred_test_iso = iso.predict(pred_test)
    m_iso = evaluate_test(test_df, pred_test_iso, h)
    return pred_test_iso, m_iso


# ---------------------------------------------------------------------------
# 4. Pipeline Principal de Ensamble
# ---------------------------------------------------------------------------
def run_ensemble_pipeline():
    print("=" * 75, flush=True)
    print("PIPELINE DE ENSAMBLE / STACKING MULTI-MODELO (SIMA MONTERREY)", flush=True)
    print("Validación Cruzada Temporal Estricta: OUT-OF-FOLD (OOF) en 2024", flush=True)
    print("=" * 75, flush=True)
    
    t_start = time.time()
    models_by_h = discover_prediction_files()
    
    all_summary_records = []
    best_ensemble_metrics = []
    
    for h in HORIZONS:
        if not models_by_h[h]:
            print(f"\n[Aviso] No hay modelos honestos disponibles para h={h}. Saltando.", flush=True)
            continue
            
        print(f"\n{'='*35} HORIZONTE h = {h} ({h}h) {'='*35}", flush=True)
        m_wide, model_names = load_and_align_models_for_h(h, models_by_h[h])
        print(f"Modelos honestos incluidos ({len(model_names)}): {model_names}", flush=True)
        
        val_df = m_wide[m_wide['split'] == 'val'].reset_index(drop=True)
        test_df = m_wide[m_wide['split'] == 'test'].reset_index(drop=True)
        print(f"Filas val (2024): {len(val_df):,}, Filas test (2025): {len(test_df):,}", flush=True)
        
        methods = {}
        
        # (a) Mejor individual
        info_best, p_val_best, p_test_best = fit_best_individual_oof(val_df, test_df, model_names)
        m_test_best = evaluate_test(test_df, p_test_best, h)
        methods['best_individual'] = {
            'val_rmse_oof': info_best['val_rmse_oof'],
            'p_val_oof': p_val_best, 'p_test': p_test_best,
            'metrics': m_test_best, 'info': f"Best: {info_best['best_model']}"
        }
        print(f"  [a] Best Individual ({info_best['best_model']}): Val OOF RMSE = {info_best['val_rmse_oof']:.4f} | Test RMSE = {m_test_best['RMSE']:.4f} | R2 = {m_test_best['R2']:.4f}")
        
        # (b) Promedio simple top-k
        info_topk, p_val_topk, p_test_topk = fit_simple_average_topk_oof(val_df, test_df, model_names)
        m_test_topk = evaluate_test(test_df, p_test_topk, h)
        methods['simple_avg_topk'] = {
            'val_rmse_oof': info_topk['val_rmse_oof'],
            'p_val_oof': p_val_topk, 'p_test': p_test_topk,
            'metrics': m_test_topk, 'info': f"Top-{info_topk['k']} avg ({info_topk['selected_models']})"
        }
        print(f"  [b] Simple Avg Top-{info_topk['k']}: Val OOF RMSE = {info_topk['val_rmse_oof']:.4f} | Test RMSE = {m_test_topk['RMSE']:.4f} | R2 = {m_test_topk['R2']:.4f}")
        
        # (c) NNLS Global
        info_nnls, p_val_nnls, p_test_nnls = fit_nnls_global_oof(val_df, test_df, model_names)
        m_test_nnls = evaluate_test(test_df, p_test_nnls, h)
        methods['nnls_global'] = {
            'val_rmse_oof': info_nnls['val_rmse_oof'],
            'p_val_oof': p_val_nnls, 'p_test': p_test_nnls,
            'metrics': m_test_nnls, 'info': 'NNLS Global weights'
        }
        print(f"  [c] NNLS Global: Val OOF RMSE = {info_nnls['val_rmse_oof']:.4f} | Test RMSE = {m_test_nnls['RMSE']:.4f} | R2 = {m_test_nnls['R2']:.4f}")
        
        # (d) NNLS por Bloque Horario
        info_block, p_val_block, p_test_block = fit_nnls_hourly_block_oof(val_df, test_df, model_names)
        m_test_block = evaluate_test(test_df, p_test_block, h)
        methods['nnls_hourly_block'] = {
            'val_rmse_oof': info_block['val_rmse_oof'],
            'p_val_oof': p_val_block, 'p_test': p_test_block,
            'metrics': m_test_block, 'info': 'NNLS 4-Block weights'
        }
        print(f"  [d] NNLS Hourly Block: Val OOF RMSE = {info_block['val_rmse_oof']:.4f} | Test RMSE = {m_test_block['RMSE']:.4f} | R2 = {m_test_block['R2']:.4f}")
        
        # (e) Stacking LightGBM
        info_stack, p_val_stack, p_test_stack = fit_stacking_lightgbm_oof(val_df, test_df, model_names)
        m_test_stack = evaluate_test(test_df, p_test_stack, h)
        methods['stacking_lgbm'] = {
            'val_rmse_oof': info_stack['val_rmse_oof'],
            'p_val_oof': p_val_stack, 'p_test': p_test_stack,
            'metrics': m_test_stack, 'info': f"LGBM Stacking OOF (mean_iter={info_stack['mean_best_iter']})"
        }
        print(f"  [e] Stacking LightGBM: Val OOF RMSE = {info_stack['val_rmse_oof']:.4f} | Test RMSE = {m_test_stack['RMSE']:.4f} | R2 = {m_test_stack['R2']:.4f}")

        # Registrar todas las variantes de ensamble en results_registry.csv
        for m_name, m_data in methods.items():
            log_result(
                experiment='noche_ensemble',
                variant=f"{m_name}_oof",
                h=h,
                m=m_data['metrics'],
                notes=f"val_rmse_oof={m_data['val_rmse_oof']:.3f}, {m_data['info']}"
            )
            all_summary_records.append({
                'h': h,
                'method': m_name,
                'val_RMSE_OOF': round(m_data['val_rmse_oof'], 4),
                'test_RMSE': round(m_data['metrics']['RMSE'], 4),
                'test_MAE': round(m_data['metrics']['MAE'], 4),
                'test_R2': round(m_data['metrics']['R2'], 4),
                'test_RMSE_peaks': round(m_data['metrics']['RMSE_peaks'], 4),
                'test_bias_peaks': round(m_data['metrics']['bias_peaks'], 4),
                'test_skill': round(m_data['metrics'].get('skill_vs_clim', 0.0), 4),
                'info': m_data['info']
            })

        # 4. Seleccionar el mejor ensamble (Criterio Estricto: MENOR val_RMSE OOF en 2024)
        best_ens_method = min(methods, key=lambda k: methods[k]['val_rmse_oof'])
        best_data = methods[best_ens_method]
        print(f"\n  >>> MEJOR ENSAMBLE SELECCIONADO POR VAL OOF: '{best_ens_method}' <<<", flush=True)
        print(f"      Val OOF RMSE: {best_data['val_rmse_oof']:.4f} | Test RMSE: {best_data['metrics']['RMSE']:.4f} (Ganancia vs Individual: {m_test_best['RMSE'] - best_data['metrics']['RMSE']:+.4f} ppb)")
        
        # 5. Intervalos Conformal sobre el mejor ensamble usando residuos OOF
        q10_test, q90_test, conf_stats = calibrate_conformal_intervals_oof(
            val_df, test_df, best_data['p_val_oof'], best_data['p_test'], coverage_target=0.80
        )
        print(f"  [Conformal 80%] Cobertura Empírica en Test: {conf_stats['empirical_coverage']*100:.2f}% | Ancho Medio: {conf_stats['mean_interval_width']:.2f} ppb")
        
        # 6. Calibración Isotónica de Picos (Post-hoc) sobre OOF
        p_test_iso, m_test_iso = evaluate_isotonic_calibration_oof(
            val_df, test_df, best_data['p_val_oof'], best_data['p_test'], h
        )
        print(f"  [Post-hoc Isotónica] Test RMSE: {m_test_iso['RMSE']:.4f} | Pk_RMSE: {m_test_iso['RMSE_peaks']:.2f} | Bias_pk: {m_test_iso['bias_peaks']:+.2f}")

        # 7. Guardar predicciones finales obligatorias con val=OOF: preds/ensemble_best_h<h>.parquet
        df_pred_val = pd.DataFrame({
            'Date': val_df['Date'],
            'Estacion': val_df['Estacion'].astype(str),
            'h': h,
            'split': 'val',
            'y_true': val_df['y_true'].values,
            'y_pred': best_data['p_val_oof'], # ESTRICTAMENTE OUT-OF-FOLD
        })
        df_pred_test = pd.DataFrame({
            'Date': test_df['Date'],
            'Estacion': test_df['Estacion'].astype(str),
            'h': h,
            'split': 'test',
            'y_true': test_df['y_true'].values,
            'y_pred': best_data['p_test'],
            'q10': q10_test,
            'q90': q90_test,
        })
        df_final_preds = pd.concat([df_pred_val, df_pred_test], ignore_index=True)
        out_preds_file = os.path.join(PREDS_DIR, f"ensemble_best_h{h}.parquet")
        df_final_preds.to_parquet(out_preds_file, index=False)
        print(f"  Predicciones guardadas en {out_preds_file} ({len(df_final_preds)} filas con val=OOF)")

        best_ensemble_metrics.append({
            'h': h,
            'best_ens_method': best_ens_method,
            'best_individual': info_best['best_model'],
            'val_RMSE_OOF': round(best_data['val_rmse_oof'], 4),
            'indiv_test_RMSE': round(m_test_best['RMSE'], 4),
            'ens_test_RMSE': round(best_data['metrics']['RMSE'], 4),
            'delta_RMSE': round(best_data['metrics']['RMSE'] - m_test_best['RMSE'], 4),
            'indiv_test_R2': round(m_test_best['R2'], 4),
            'ens_test_R2': round(best_data['metrics']['R2'], 4),
            'indiv_bias_pk': round(m_test_best['bias_peaks'], 2),
            'ens_bias_pk': round(best_data['metrics']['bias_peaks'], 2),
            'conformal_coverage': round(conf_stats['empirical_coverage'], 4),
            'conformal_width': round(conf_stats['mean_interval_width'], 2),
            'iso_test_RMSE': round(m_test_iso['RMSE'], 4),
            'iso_bias_pk': round(m_test_iso['bias_peaks'], 2)
        })

    # Guardar resumen global en CSV
    df_all_summary = pd.DataFrame(all_summary_records)
    summary_csv = os.path.join(EXP_DIR, 'ensemble_summary.csv')
    df_all_summary.to_csv(summary_csv, index=False)
    
    df_best = pd.DataFrame(best_ensemble_metrics)
    best_csv = os.path.join(EXP_DIR, 'ensemble_best_comparison.csv')
    df_best.to_csv(best_csv, index=False)

    print("\n" + "=" * 75, flush=True)
    print("RESUMEN DE ENSAMBLE OOF: MEJOR ENSAMBLE VS MEJOR INDIVIDUAL POR HORIZONTE", flush=True)
    print("=" * 75, flush=True)
    print(df_best.to_string(index=False), flush=True)
    
    generate_ensemble_figures(df_best, df_all_summary)
    print(f"\nPipeline de ensamble OOF completado en {time.time()-t_start:.1f}s.", flush=True)
    return df_best, df_all_summary


def generate_ensemble_figures(df_best: pd.DataFrame, df_all: pd.DataFrame):
    """Genera gráficos de diagnóstico para el informe."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df_best))
    w = 0.35
    ax.bar(x - w/2, df_best['indiv_test_RMSE'], width=w, label='Mejor Modelo Individual', color='#4A90E2', alpha=0.9)
    ax.bar(x + w/2, df_best['ens_test_RMSE'], width=w, label='Mejor Ensamble (Seleccionado OOF)', color='#2ECC71', alpha=0.9)
    
    for i in range(len(df_best)):
        d = df_best.loc[i, 'delta_RMSE']
        ax.annotate(f"{d:+.2f}", (x[i], min(df_best.loc[i, 'indiv_test_RMSE'], df_best.loc[i, 'ens_test_RMSE']) - 0.4),
                    ha='center', fontsize=10, fontweight='bold', color='#D0021B' if d < 0 else '#4A4A4A')
                    
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}h" for h in df_best['h']], fontsize=11)
    ax.set_ylabel("RMSE en Test 2025 (ppb)", fontsize=11)
    ax.set_title("Ganancia de Ensamble OOF vs Mejor Modelo Individual (Test 2025)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'ensemble_vs_individual_rmse.png'), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    methods_order = ['best_individual', 'simple_avg_topk', 'nnls_global', 'nnls_hourly_block', 'stacking_lgbm']
    method_labels = {'best_individual': 'Mejor Individual', 'simple_avg_topk': 'Promedio Top-k',
                     'nnls_global': 'NNLS Global', 'nnls_hourly_block': 'NNLS Bloques Horarios',
                     'stacking_lgbm': 'Stacking LightGBM OOF'}
    colors = {'best_individual': '#9B9B9B', 'simple_avg_topk': '#F5A623', 'nnls_global': '#4A90E2',
              'nnls_hourly_block': '#9013FE', 'stacking_lgbm': '#2ECC71'}
              
    for m in methods_order:
        sub = df_all[df_all['method'] == m].sort_values('h')
        if len(sub) > 0:
            ax.plot(sub['h'], sub['test_RMSE'], marker='o', label=method_labels.get(m, m),
                    color=colors.get(m, '#000'), linewidth=2)
                    
    ax.set_xlabel("Horizonte de Pronóstico h (horas)", fontsize=11)
    ax.set_ylabel("RMSE en Test 2025 (ppb)", fontsize=11)
    ax.set_title("Comparativa de Métodos de Ensamble (Selección OOF) en Test 2025", fontsize=13, fontweight='bold')
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'ensemble_methods_comparison.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    run_ensemble_pipeline()
