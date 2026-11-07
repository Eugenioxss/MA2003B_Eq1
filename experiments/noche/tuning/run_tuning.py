"""
experiments/noche/tuning/run_tuning.py
======================================
Tuning de Hiperparámetros de LightGBM por Horizonte (h in [4, 8, 12, 24, 48])
sobre la variante honesta (meteo previous_day en val/test, seamless en train).

Características:
1. Utiliza el cache precomputado de v3_base: `experiments/noche/v3_base/cache/features_h<h>.parquet`
2. Espacio de Búsqueda Optuna (25 trials por h):
   - num_leaves, learning_rate, min_child_samples, colsample_bytree (feature_fraction),
     subsample (bagging), reg_lambda, reg_alpha, max_bin, y selección de features por importancia (top_features_pct).
   - Optimiza RMSE sobre val 2024 con early stopping.
   - Submuestreo estratificado temporal de train (35%) para velocidad (<15s por trial).
3. Re-entrenamiento final:
   - Entrena con 100% de train usando los mejores hiperparámetros y features seleccionadas.
   - Guarda predicciones val (2024) y test (2025) en `preds/tuned_honest_h<h>.parquet`.
   - Registra en `experiments/results_registry.csv` con experiment='noche_tuning', variant='tuned_honest'.
4. Bonus: XGBoost Diverso (xgb_honest):
   - Entrena XGBoost con tree_method='hist' y 3 hilos sobre las mismas features honestas.
   - Guarda predicciones en `preds/xgb_honest_h<h>.parquet`.
   - Registra con variant='xgb_honest'.
"""
from __future__ import annotations
import os
import sys
import time
import json
import warnings
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

# Desactivar logs ruidosos de Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import evaluate_test, log_result

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(EXP_DIR, 'preds')
CACHE_DIR = os.path.join(ROOT, 'experiments', 'noche', 'v3_base', 'cache')
os.makedirs(PREDS_DIR, exist_ok=True)

TUNING_HORIZONS = [4, 8, 12, 24, 48]
NUM_THREADS = 3
N_TRIALS = 25
TRAIN_SAMPLE_FRAC = 0.35


def get_feature_importance_ranking(X_tr: pd.DataFrame, y_tr: np.ndarray, cat_cols: list[str]) -> list[str]:
    """Entrena un modelo rápido para rankear las features por importancia de ganancia."""
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    params = {
        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.1,
        'num_leaves': 31, 'num_threads': NUM_THREADS, 'seed': 42, 'verbose': -1
    }
    bst = lgb.train(params, dtr, num_boost_round=100)
    gains = bst.feature_importance(importance_type='gain')
    feat_names = bst.feature_name()
    ranked = [f for _, f in sorted(zip(gains, feat_names), reverse=True)]
    return ranked


def tune_horizon(h: int) -> tuple[dict, dict, dict]:
    print("\n" + "=" * 70, flush=True)
    print(f"INICIANDO TUNING POR HORIZONTE: h = {h} ({h}h)", flush=True)
    print("=" * 70, flush=True)
    
    t0_h = time.time()
    cache_file = os.path.join(CACHE_DIR, f'features_h{h}.parquet')
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache no encontrado: {cache_file}")
        
    df = pd.read_parquet(cache_file)
    df['Estacion'] = df['Estacion'].astype('category')
    
    # Extraer columnas honestas
    chem_features = [c for c in df.columns if not c.startswith(('meteo_t0_', 'honest_meteo_', 'pp_meteo_')) and c not in ['Date', 'split', 'Target', f'Target_{h}h']]
    meteo_t0 = [c for c in df.columns if c.startswith('meteo_t0_')]
    honest_meteo = [c for c in df.columns if c.startswith('honest_meteo_')]
    all_honest_features = list(dict.fromkeys(chem_features + meteo_t0 + honest_meteo))
    
    # Splits temporales
    train_full = df[df['split'] == 'train'].dropna(subset=['Target']).reset_index(drop=True)
    val_full = df[df['split'] == 'val'].dropna(subset=['Target']).reset_index(drop=True)
    test_full = df[df['split'] == 'test'].dropna(subset=['Target']).reset_index(drop=True)
    
    print(f"[h={h}] Features honestas: {len(all_honest_features)} | Train: {len(train_full):,}, Val: {len(val_full):,}, Test: {len(test_full):,}", flush=True)
    
    # 1. Ranking inicial de importancia para feature selection
    print(f"[h={h}] Pre-computando ranking de importancia de features...", flush=True)
    sample_for_rank = train_full.sample(n=min(50000, len(train_full)), random_state=42)
    ranked_features = get_feature_importance_ranking(sample_for_rank[all_honest_features], sample_for_rank['Target'].values, cat_cols=['Estacion'])
    
    # Submuestreo para optimización rápida en Optuna
    train_sub = train_full.sample(frac=TRAIN_SAMPLE_FRAC, random_state=42).reset_index(drop=True)
    y_tr_sub = train_sub['Target'].values
    y_va = val_full['Target'].values
    
    # 2. Función Objetivo de Optuna
    def objective(trial: optuna.Trial) -> float:
        # Espacio de búsqueda
        num_leaves = trial.suggest_int('num_leaves', 31, 255)
        learning_rate = trial.suggest_float('learning_rate', 0.02, 0.08, log=True)
        min_child_samples = trial.suggest_int('min_child_samples', 30, 200)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.50, 0.95)
        subsample = trial.suggest_float('subsample', 0.65, 1.0)
        subsample_freq = 1 if subsample < 1.0 else 0
        reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
        max_bin = trial.suggest_categorical('max_bin', [127, 255, 511])
        top_features_pct = trial.suggest_float('top_features_pct', 0.70, 1.0, step=0.05)
        
        # Selección de features
        n_feats = max(30, int(len(ranked_features) * top_features_pct))
        selected_feats = ranked_features[:n_feats]
        if 'Estacion' not in selected_feats:
            selected_feats.append('Estacion')
            
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'colsample_bytree': colsample_bytree,
            'subsample': subsample,
            'subsample_freq': subsample_freq,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'max_bin': max_bin,
            'num_threads': NUM_THREADS,
            'seed': 42,
            'verbose': -1,
        }
        
        dtr = lgb.Dataset(train_sub[selected_feats], y_tr_sub, categorical_feature=['Estacion'], free_raw_data=False)
        dva = lgb.Dataset(val_full[selected_feats], y_va, reference=dtr, categorical_feature=['Estacion'], free_raw_data=False)
        
        bst = lgb.train(
            params,
            dtr,
            num_boost_round=600,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(35, verbose=False)]
        )
        
        pred_val = bst.predict(val_full[selected_feats], num_iteration=bst.best_iteration)
        rmse_val = float(np.sqrt(np.mean((y_va - pred_val) ** 2)))
        return rmse_val

    # 3. Ejecución de Optuna
    print(f"[h={h}] Iniciando {N_TRIALS} trials de Optuna...", flush=True)
    t0_opt = time.time()
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, timeout=900)
    print(f"[h={h}] Optuna completado en {time.time()-t0_opt:.1f}s. Mejor val RMSE: {study.best_value:.4f}", flush=True)
    
    best_params = study.best_params
    best_top_pct = best_params.pop('top_features_pct')
    n_feats_best = max(30, int(len(ranked_features) * best_top_pct))
    best_features = ranked_features[:n_feats_best]
    if 'Estacion' not in best_features:
        best_features.append('Estacion')
        
    print(f"[h={h}] Mejores hiperparámetros:", best_params, flush=True)
    print(f"[h={h}] Features seleccionadas: {len(best_features)} / {len(all_honest_features)} (top {best_top_pct*100:.0f}%)", flush=True)
    
    # Guardar mejores hiperparámetros
    best_params_record = {
        'horizon': h,
        'val_rmse_optuna': study.best_value,
        'best_params': best_params,
        'n_features': len(best_features),
        'top_features_pct': best_top_pct
    }
    with open(os.path.join(EXP_DIR, f'best_params_h{h}.json'), 'w') as f:
        json.dump(best_params_record, f, indent=2)

    # 4. Re-entrenamiento final sobre 100% de train (tuned_honest)
    print(f"[h={h}] Re-entrenando modelo final LightGBM sobre 100% train...", flush=True)
    final_params = {
        **best_params,
        'objective': 'regression',
        'metric': 'rmse',
        'subsample_freq': 1 if best_params.get('subsample', 1.0) < 1.0 else 0,
        'num_threads': NUM_THREADS,
        'seed': 42,
        'verbose': -1,
    }
    
    dtr_full = lgb.Dataset(train_full[best_features], train_full['Target'], categorical_feature=['Estacion'], free_raw_data=False)
    dva_full = lgb.Dataset(val_full[best_features], val_full['Target'], reference=dtr_full, categorical_feature=['Estacion'], free_raw_data=False)
    
    t0_fit = time.time()
    final_bst = lgb.train(
        final_params,
        dtr_full,
        num_boost_round=1500,
        valid_sets=[dva_full],
        callbacks=[lgb.early_stopping(60, verbose=False)]
    )
    fit_time = time.time() - t0_fit
    
    pred_val_lgb = final_bst.predict(val_full[best_features], num_iteration=final_bst.best_iteration)
    pred_test_lgb = final_bst.predict(test_full[best_features], num_iteration=final_bst.best_iteration)
    
    m_test_lgb = evaluate_test(test_full, pred_test_lgb, h)
    m_val_lgb = evaluate_test(val_full, pred_val_lgb, h)
    
    print(f"  [tuned_honest] h={h} -> Test RMSE: {m_test_lgb['RMSE']:.4f} | R2: {m_test_lgb['R2']:.4f} | Pk_RMSE: {m_test_lgb['RMSE_peaks']:.2f} | Val RMSE: {m_val_lgb['RMSE']:.4f} | Iter: {final_bst.best_iteration} ({fit_time:.1f}s)", flush=True)
    
    # Registrar en results_registry.csv
    log_result(
        experiment='noche_tuning',
        variant='tuned_honest',
        h=h,
        m=m_test_lgb,
        notes=f"optuna_tuned, n_feats={len(best_features)}, iters={final_bst.best_iteration}, val_rmse={m_val_lgb['RMSE']:.3f}"
    )
    
    # Guardar predicciones val+test en preds/tuned_honest_h<h>.parquet
    df_pred_val_lgb = pd.DataFrame({
        'Date': val_full['Date'],
        'Estacion': val_full['Estacion'].astype(str),
        'h': h,
        'split': 'val',
        'y_true': val_full['Target'].values,
        'y_pred': pred_val_lgb,
    })
    df_pred_test_lgb = pd.DataFrame({
        'Date': test_full['Date'],
        'Estacion': test_full['Estacion'].astype(str),
        'h': h,
        'split': 'test',
        'y_true': test_full['Target'].values,
        'y_pred': pred_test_lgb,
    })
    df_lgb_all = pd.concat([df_pred_val_lgb, df_pred_test_lgb], ignore_index=True)
    lgb_parquet = os.path.join(PREDS_DIR, f"tuned_honest_h{h}.parquet")
    df_lgb_all.to_parquet(lgb_parquet, index=False)

    # 5. BONUS: XGBoost Diverso (xgb_honest)
    print(f"[h={h}] Entrenando XGBoost diverso (xgb_honest)...", flush=True)
    t0_xgb = time.time()
    
    # Preparar matrices para XGBoost (codificar categorical Estacion si hace falta)
    xgb_features = [c for c in all_honest_features if c != 'Estacion']
    # Codificar estacion como int
    train_full['Estacion_code'] = train_full['Estacion'].cat.codes
    val_full['Estacion_code'] = val_full['Estacion'].cat.codes
    test_full['Estacion_code'] = test_full['Estacion'].cat.codes
    xgb_cols = ['Estacion_code'] + xgb_features
    
    dtr_xgb = xgb.DMatrix(train_full[xgb_cols], label=train_full['Target'], nthread=NUM_THREADS)
    dva_xgb = xgb.DMatrix(val_full[xgb_cols], label=val_full['Target'], nthread=NUM_THREADS)
    dte_xgb = xgb.DMatrix(test_full[xgb_cols], label=test_full['Target'], nthread=NUM_THREADS)
    
    xgb_params = {
        'tree_method': 'hist',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'nthread': NUM_THREADS,
        'seed': 42,
    }
    
    bst_xgb = xgb.train(
        xgb_params,
        dtr_xgb,
        num_boost_round=800,
        evals=[(dva_xgb, 'val')],
        early_stopping_rounds=40,
        verbose_eval=False
    )
    xgb_time = time.time() - t0_xgb
    
    pred_val_xgb = bst_xgb.predict(dva_xgb)
    pred_test_xgb = bst_xgb.predict(dte_xgb)
    
    m_test_xgb = evaluate_test(test_full, pred_test_xgb, h)
    m_val_xgb = evaluate_test(val_full, pred_val_xgb, h)
    
    print(f"  [xgb_honest] h={h} -> Test RMSE: {m_test_xgb['RMSE']:.4f} | R2: {m_test_xgb['R2']:.4f} | Pk_RMSE: {m_test_xgb['RMSE_peaks']:.2f} | Val RMSE: {m_val_xgb['RMSE']:.4f} | Iter: {bst_xgb.best_iteration} ({xgb_time:.1f}s)", flush=True)
    
    log_result(
        experiment='noche_tuning',
        variant='xgb_honest',
        h=h,
        m=m_test_xgb,
        notes=f"xgboost_hist, iters={bst_xgb.best_iteration}, val_rmse={m_val_xgb['RMSE']:.3f}"
    )
    
    # Guardar predicciones val+test en preds/xgb_honest_h<h>.parquet
    df_pred_val_xgb = pd.DataFrame({
        'Date': val_full['Date'],
        'Estacion': val_full['Estacion'].astype(str),
        'h': h,
        'split': 'val',
        'y_true': val_full['Target'].values,
        'y_pred': pred_val_xgb,
    })
    df_pred_test_xgb = pd.DataFrame({
        'Date': test_full['Date'],
        'Estacion': test_full['Estacion'].astype(str),
        'h': h,
        'split': 'test',
        'y_true': test_full['Target'].values,
        'y_pred': pred_test_xgb,
    })
    df_xgb_all = pd.concat([df_pred_val_xgb, df_pred_test_xgb], ignore_index=True)
    xgb_parquet = os.path.join(PREDS_DIR, f"xgb_honest_h{h}.parquet")
    df_xgb_all.to_parquet(xgb_parquet, index=False)

    res_lgb = {
        'h': h, 'variant': 'tuned_honest',
        'val_RMSE': round(m_val_lgb['RMSE'], 4), 'test_RMSE': round(m_test_lgb['RMSE'], 4),
        'test_MAE': round(m_test_lgb['MAE'], 4), 'test_R2': round(m_test_lgb['R2'], 4),
        'test_RMSE_peaks': round(m_test_lgb['RMSE_peaks'], 4), 'test_bias_peaks': round(m_test_lgb['bias_peaks'], 4),
        'test_skill': round(m_test_lgb.get('skill_vs_clim', 0.0), 4),
        'best_iter': final_bst.best_iteration, 'n_features': len(best_features),
        'fit_time_s': round(fit_time, 1)
    }
    res_xgb = {
        'h': h, 'variant': 'xgb_honest',
        'val_RMSE': round(m_val_xgb['RMSE'], 4), 'test_RMSE': round(m_test_xgb['RMSE'], 4),
        'test_MAE': round(m_test_xgb['MAE'], 4), 'test_R2': round(m_test_xgb['R2'], 4),
        'test_RMSE_peaks': round(m_test_xgb['RMSE_peaks'], 4), 'test_bias_peaks': round(m_test_xgb['bias_peaks'], 4),
        'test_skill': round(m_test_xgb.get('skill_vs_clim', 0.0), 4),
        'best_iter': bst_xgb.best_iteration, 'n_features': len(xgb_cols),
        'fit_time_s': round(xgb_time, 1)
    }
    
    print(f"[h={h}] Horizonte completado en {time.time()-t0_h:.1f}s.", flush=True)
    return res_lgb, res_xgb, best_params_record


def main():
    print("=" * 75, flush=True)
    print("EXPERIMENTO NOCHE: TUNING DE LIGHTGBM POR HORIZONTE + XGBOOST DIVERSO", flush=True)
    print(f"Horizontes: {TUNING_HORIZONS} | Hilos: {NUM_THREADS} | Trials por h: {N_TRIALS}", flush=True)
    print("=" * 75, flush=True)
    
    t_start = time.time()
    all_results = []
    
    for h in TUNING_HORIZONS:
        res_lgb, res_xgb, _ = tune_horizon(h)
        all_results.append(res_lgb)
        all_results.append(res_xgb)
        
    df_summary = pd.DataFrame(all_results)
    summary_path = os.path.join(EXP_DIR, 'tuning_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    
    print("\n" + "=" * 75, flush=True)
    print("RESUMEN DE TUNING Y MODELOS DIVERSOS (TEST 2025)", flush=True)
    print("=" * 75, flush=True)
    print(df_summary.to_string(index=False), flush=True)
    print(f"\nExperimento completo de tuning finalizado en {time.time()-t_start:.1f}s.", flush=True)


if __name__ == '__main__':
    main()
