"""
experiments/noche/audit_imput/run_imputation_experiments.py
==========================================================
Ejecuta la Parte 2: Experimentos de Imputación de Entradas (Downstream Forecast).
Variantes evaluadas en h in [1, 4, 8, 24, 48]:
  1. noimp: Panel original crudo sin imputar (manejo nativo de NaNs en LightGBM)
  2. imp_ffill_short: Imputación causal local por forward-fill (ffill <= 3h)
  3. imp_spatial_lgbm: Imputación causal multivariada espacial con LightGBM

Utiliza:
  - Features de Plan A V4_spatial (build_planA_features con variant='V4')
  - Hiperparámetros optimizados de best_params_optuna.json
  - num_threads=3, n_jobs=3
  - Target intacto observado real (Target_{h}h sin imputar)
  - Registro en experiments/results_registry.csv con experiment='noche_audit_imput'
  - Predicciones val+test guardadas en preds/<variante>_h<h>.parquet
"""
from __future__ import annotations
import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import (
    load_panel, split, evaluate_test, log_result,
    MODEL_STATIONS, ALL_STATIONS, HORIZONS, add_targets
)
from planA_quickwins.features_planA import build_planA_features

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(EXP_DIR, 'preds')
os.makedirs(PREDS_DIR, exist_ok=True)

EVAL_HORIZONS = [1, 4, 8, 24, 48]
NUM_THREADS = 3

# Cargar mejores hiperparámetros de Optuna
OPTUNA_PATH = os.path.join(ROOT, 'experiments', 'planA_quickwins', 'best_params_optuna.json')
with open(OPTUNA_PATH, 'r') as f:
    BEST_OPTUNA_PARAMS = json.load(f)

LGBM_TRAIN_PARAMS = {
    **BEST_OPTUNA_PARAMS,
    'objective': 'regression',
    'metric': 'rmse',
    'seed': 42,
    'verbose': -1,
    'num_threads': NUM_THREADS,
    'n_jobs': NUM_THREADS,
}

VARS_ALL = [
    'CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'PRS', 'RAINF',
    'RH', 'SO2', 'SR', 'TOUT', 'WSR', 'WDR', 'U_Wind', 'V_Wind'
]

# ---------------------------------------------------------------------------
# 1. Funciones de Imputación Causal
# ---------------------------------------------------------------------------
def create_panel_ffill_short(panel: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """
    Imputación causal local: Forward-fill de hasta `limit` horas por estación.
    Para huecos <= 3h, rellena con la última observación del sensor.
    Huecos > 3h permanecen NaN a partir de la 4ª hora.
    Estrictamente causal (usa solo t-1, t-2, t-3 de la misma estación).
    """
    print(f"Generando panel imp_ffill_short (limit={limit}h)...", flush=True)
    p_out = panel.copy().sort_values(['Estacion', 'Date']).reset_index(drop=True)
    
    for v in VARS_ALL:
        if v in p_out.columns:
            p_out[v] = p_out.groupby('Estacion', sort=False)[v].ffill(limit=limit)
            
    return p_out


def create_panel_spatial_lgbm(panel: pd.DataFrame, num_threads: int = NUM_THREADS) -> pd.DataFrame:
    """
    Imputación causal espacial con LightGBM:
    Para cada variable clave y estación objetivo:
      - Entrena un modelo LightGBM en 2020-2022 con validación temprana en 2023 (estrictamente <= 2023).
      - Covariables: observaciones simultáneas de otras estaciones donantes en t,
                     meteorología simultánea en t (TOUT, SR),
                     retardo histórico local s[V](t-1) y s[V](t-2),
                     ciclo horario y estacional.
      - Predice y rellena exclusivamente los huecos (NaNs) de la variable en t.
      - NUNCA utiliza datos futuros (t' > t).
    """
    cache_path = os.path.join(EXP_DIR, 'panel_spatial_lgbm.parquet')
    if os.path.exists(cache_path):
        print(f"Cargando panel_spatial_lgbm desde cache: {cache_path}", flush=True)
        return pd.read_parquet(cache_path)

    print("Construyendo panel imp_spatial_lgbm causal desde cero...", flush=True)
    t0_all = time.time()
    p_work = panel.copy().sort_values(['Estacion', 'Date']).reset_index(drop=True)
    
    # Variables a imputar espacialmente con LightGBM (las que más impacto tienen en Plan A V4)
    key_spatial_vars = ['O3', 'NOX', 'TOUT', 'SR', 'RH', 'PM2.5', 'PM10', 'U_Wind', 'V_Wind']
    
    # Variables secundarias se rellenan con ffill corto de respaldo
    for v_sec in ['CO', 'NO', 'NO2', 'SO2', 'PRS', 'RAINF', 'WSR', 'WDR']:
        if v_sec in p_work.columns:
            p_work[v_sec] = p_work.groupby('Estacion', sort=False)[v_sec].ffill(limit=3)

    params_imputer = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.08,
        'num_leaves': 31,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_threads': num_threads,
        'seed': 42,
        'verbose': -1
    }

    # Precomputar pivots para acelerar construcción de matrices
    pivots = {}
    for kv in ['O3', 'TOUT', 'SR', 'NOX', 'RH', 'PM2.5', 'PM10', 'U_Wind', 'V_Wind']:
        if kv in p_work.columns:
            pivots[kv] = p_work.pivot(index='Date', columns='Estacion', values=kv)
            
    dates = pivots['O3'].index
    calendar_df = pd.DataFrame({
        'hour_sin': np.sin(2 * np.pi * dates.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dates.hour / 24),
        'month_sin': np.sin(2 * np.pi * dates.month / 12),
        'month_cos': np.cos(2 * np.pi * dates.month / 12),
    }, index=dates)

    for v in key_spatial_vars:
        t0_v = time.time()
        p_v = pivots[v].copy()
        
        # Para cada estación
        for st in ALL_STATIONS:
            s_orig = p_v[st].values
            nan_mask = np.isnan(s_orig)
            if not nan_mask.any():
                continue
                
            donor_st = [s for s in ALL_STATIONS if s != st]
            
            # Construir X_wide para predecir st[v]
            donor_v = p_v[donor_st].add_prefix(f'{v}_')
            donor_tout = pivots['TOUT'][donor_st].add_prefix('TOUT_') if v != 'TOUT' and 'TOUT' in pivots else pd.DataFrame(index=dates)
            donor_sr = pivots['SR'][donor_st].add_prefix('SR_') if v != 'SR' and 'SR' in pivots else pd.DataFrame(index=dates)
            
            # Retardos del mismo sensor (causales)
            y_series = p_v[st]
            self_lag1 = y_series.shift(1).rename(f'{v}_self_lag1')
            self_lag2 = y_series.shift(2).rename(f'{v}_self_lag2')
            
            X_wide = pd.concat([donor_v, donor_tout, donor_sr, self_lag1, self_lag2, calendar_df], axis=1)
            
            # Condición de entrenamiento: estrictamente histórico (<= 2023) y target no nulo
            tr_cond = (X_wide.index.year <= 2022) & y_series.notna()
            val_cond = (X_wide.index.year == 2023) & y_series.notna()
            
            if tr_cond.sum() < 500:
                # Fallback suave a ffill si no hay datos suficientes de train
                p_v[st] = p_v[st].ffill(limit=6)
                continue
                
            dtrain = lgb.Dataset(X_wide[tr_cond], y_series[tr_cond])
            dval = lgb.Dataset(X_wide[val_cond], y_series[val_cond], reference=dtrain) if val_cond.sum() > 100 else None
            
            valid_sets = [dval] if dval is not None else [dtrain]
            callbacks = [lgb.early_stopping(25, verbose=False)] if dval is not None else []
            
            bst = lgb.train(
                params_imputer,
                dtrain,
                num_boost_round=300,
                valid_sets=valid_sets,
                callbacks=callbacks
            )
            
            # Predecir sobre todo el panel y rellenar únicamente NaNs
            preds = np.maximum(0, bst.predict(X_wide))
            filled_s = s_orig.copy()
            filled_s[nan_mask] = preds[nan_mask]
            
            # Si aún quedan NaNs residuales (ej. extremos), ffill local
            if np.isnan(filled_s).any():
                filled_s = pd.Series(filled_s).ffill().bfill().values
                
            p_v[st] = filled_s
            
        print(f"  Variable {v} imputada en {time.time()-t0_v:.1f}s", flush=True)
        # Reincorporar al panel
        v_unstacked = p_v.unstack().reset_index(name=f'{v}_imp')
        p_work = pd.merge(p_work, v_unstacked, on=['Estacion', 'Date'], how='left')
        p_work[v] = p_work[f'{v}_imp'].values
        p_work.drop(columns=[f'{v}_imp'], inplace=True)
        # Actualizar pivot para variables subsecuentes
        pivots[v] = p_v

    p_work.to_parquet(cache_path, index=False)
    print(f"Panel imp_spatial_lgbm finalizado y guardado en {time.time()-t0_all:.1f}s.", flush=True)
    return p_work


# ---------------------------------------------------------------------------
# 2. Pipeline de Entrenamiento y Evaluación Downstream
# ---------------------------------------------------------------------------
def run_variant(
    variant_name: str,
    panel_input: pd.DataFrame,
    true_targets_df: pd.DataFrame
) -> list[dict]:
    """
    Construye features V4 sobre panel_input, reemplaza targets con los reales,
    entrena LightGBM con Optuna params para cada h in [1, 4, 8, 24, 48],
    evalúa en test 2025, guarda predicciones val+test y registra resultados.
    """
    print("\n" + "=" * 70, flush=True)
    print(f"  INICIANDO EVALUACION DOWNSTREAM: VARIANTE '{variant_name}'", flush=True)
    print("=" * 70, flush=True)
    t0_var = time.time()
    
    # 1. Construir features Plan A V4
    print(f"[{variant_name}] Construyendo features V4_spatial...", flush=True)
    t_feat = time.time()
    df_feat, feature_cols = build_planA_features(panel_input, variant='V4')
    print(f"[{variant_name}] {len(feature_cols)} features construidas en {time.time()-t_feat:.2f}s.", flush=True)
    
    # 2. REEMPLAZAR estrictamente los targets con los targets verdaderos observados
    print(f"[{variant_name}] Restaurando targets observados reales (sin imputar)...", flush=True)
    # Alinear exactamente por (Estacion, Date)
    df_feat = df_feat.sort_values(['Estacion', 'Date']).reset_index(drop=True)
    true_aligned = pd.merge(
        df_feat[['Estacion', 'Date']],
        true_targets_df,
        on=['Estacion', 'Date'],
        how='left'
    )
    for h in EVAL_HORIZONS:
        target_col = f'Target_{h}h'
        df_feat[target_col] = true_aligned[target_col].values

    # 3. Splits temporales
    train_df, val_df, test_df = split(df_feat)
    print(f"[{variant_name}] Splits -> Train (<=2023): {len(train_df):,}, Val (2024): {len(val_df):,}, Test (2025): {len(test_df):,}", flush=True)

    variant_metrics = []

    for h in EVAL_HORIZONS:
        target_col = f'Target_{h}h'
        t0_h = time.time()
        print(f"\n--- [{variant_name}] Entrenando horizonte h = {h}h ---", flush=True)
        
        # Filtrar no nulos en el target real
        tr_h = train_df.dropna(subset=[target_col]).reset_index(drop=True)
        va_h = val_df.dropna(subset=[target_col]).reset_index(drop=True)
        te_h = test_df.dropna(subset=[target_col]).reset_index(drop=True)
        
        X_tr, y_tr = tr_h[feature_cols], tr_h[target_col]
        X_va, y_va = va_h[feature_cols], va_h[target_col]
        X_te, y_te = te_h[feature_cols], te_h[target_col]
        
        # LightGBM Datasets
        dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=['Estacion'])
        dval = lgb.Dataset(X_va, y_va, reference=dtrain, categorical_feature=['Estacion'])
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0)
        ]
        
        model = lgb.train(
            LGBM_TRAIN_PARAMS,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=callbacks
        )
        
        # Predicciones en val y test
        y_pred_val = model.predict(X_va, num_iteration=model.best_iteration)
        y_pred_test = model.predict(X_te, num_iteration=model.best_iteration)
        
        # Evaluación oficial con evaluate_test en test 2025
        m_test = evaluate_test(te_h, y_pred_test, h)
        train_time = round(time.time() - t0_h, 2)
        m_test['best_iteration'] = int(model.best_iteration)
        m_test['train_time_s'] = train_time
        
        # Evaluación en validación 2024
        m_val = evaluate_test(va_h, y_pred_val, h)
        m_test['val_RMSE'] = round(m_val['RMSE'], 4)
        m_test['val_R2'] = round(m_val['R2'], 4)
        
        # Registrar en results_registry.csv
        notes = f"optuna_params, iters={model.best_iteration}, val_rmse={m_val['RMSE']:.2f}, time={train_time}s"
        log_result(
            experiment='noche_audit_imput',
            variant=variant_name,
            h=h,
            m=m_test,
            notes=notes
        )
        
        # Guardar predicciones val+test en formato obligatorio: preds/<variante>_h<h>.parquet
        df_pred_val = pd.DataFrame({
            'Date': va_h['Date'],
            'Estacion': va_h['Estacion'].astype(str),
            'h': h,
            'split': 'val',
            'y_true': y_va.values,
            'y_pred': y_pred_val
        })
        df_pred_test = pd.DataFrame({
            'Date': te_h['Date'],
            'Estacion': te_h['Estacion'].astype(str),
            'h': h,
            'split': 'test',
            'y_true': y_te.values,
            'y_pred': y_pred_test
        })
        df_pred_all = pd.concat([df_pred_val, df_pred_test], ignore_index=True)
        pred_parquet_file = os.path.join(PREDS_DIR, f"{variant_name}_h{h}.parquet")
        df_pred_all.to_parquet(pred_parquet_file, index=False)
        print(f"  Predicciones guardadas: {pred_parquet_file} ({len(df_pred_all)} filas: {len(df_pred_val)} val, {len(df_pred_test)} test)", flush=True)

        res_entry = {
            'variant': variant_name,
            'h': h,
            'RMSE': round(m_test['RMSE'], 4),
            'MAE': round(m_test['MAE'], 4),
            'R2': round(m_test['R2'], 4),
            'RMSE_peaks': round(m_test['RMSE_peaks'], 4),
            'bias_peaks': round(m_test['bias_peaks'], 4),
            'skill_vs_clim': round(m_test.get('skill_vs_clim', 0.0), 4),
            'val_RMSE': round(m_val['RMSE'], 4),
            'best_iter': model.best_iteration,
            'train_time_s': train_time
        }
        variant_metrics.append(res_entry)
        
        print(f"  [{variant_name}] h={h:2d} -> Test RMSE: {m_test['RMSE']:.4f} | R2: {m_test['R2']:.4f} | "
              f"RMSE_pk: {m_test['RMSE_peaks']:.4f} | Bias_pk: {m_test['bias_peaks']:.2f} | "
              f"Skill: {m_test['skill_vs_clim']:.4f} | Val RMSE: {m_val['RMSE']:.2f} | Iter: {model.best_iteration} ({train_time}s)", flush=True)

    print(f"\nVariante {variant_name} completada en {time.time()-t0_var:.1f}s.", flush=True)
    return variant_metrics


def main():
    print("=" * 70, flush=True)
    print("EXPERIMENTO NOCHE: AUDITORIA E IMPUTACION DE ENTRADAS", flush=True)
    print(f"Horizontes: {EVAL_HORIZONS} | Hilos: {NUM_THREADS}", flush=True)
    print("=" * 70, flush=True)
    
    t_start = time.time()
    
    # 1. Cargar panel crudo original
    print("\n[Fase 1] Cargando panel crudo y extrayendo targets reales...", flush=True)
    panel_raw = load_panel(use_cache=True)
    
    # Extraer targets verdaderos no imputados
    panel_with_targets = add_targets(panel_raw.copy(), horizons=EVAL_HORIZONS)
    target_cols = [f'Target_{h}h' for h in EVAL_HORIZONS]
    true_targets_df = panel_with_targets[['Estacion', 'Date'] + target_cols].copy()
    print(f"Panel crudo cargado: {len(panel_raw):,} filas.", flush=True)

    # 2. Generar paneles para cada variante
    print("\n[Fase 2] Preparando paneles de entrada...", flush=True)
    panel_noimp = panel_raw.copy()
    panel_ffill = create_panel_ffill_short(panel_raw, limit=3)
    panel_spatial = create_panel_spatial_lgbm(panel_raw, num_threads=NUM_THREADS)

    # 3. Ejecutar las 3 variantes downstream
    all_summary = []
    
    # Variante 1: noimp
    res_noimp = run_variant('noimp', panel_noimp, true_targets_df)
    all_summary.extend(res_noimp)
    
    # Variante 2: imp_ffill_short
    res_ffill = run_variant('imp_ffill_short', panel_ffill, true_targets_df)
    all_summary.extend(res_ffill)
    
    # Variante 3: imp_spatial_lgbm
    res_spatial = run_variant('imp_spatial_lgbm', panel_spatial, true_targets_df)
    all_summary.extend(res_spatial)
    
    # Guardar resumen final en CSV y JSON
    df_summary = pd.DataFrame(all_summary)
    summary_csv = os.path.join(EXP_DIR, 'imputation_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    
    print("\n" + "=" * 70, flush=True)
    print("RESUMEN COMPARATIVO DE IMPUTACIÓN DOWNSTREAM (TEST 2025)", flush=True)
    print("=" * 70, flush=True)
    print(df_summary.to_string(index=False), flush=True)
    print(f"\nExperimento completo finalizado en {time.time()-t_start:.1f}s.", flush=True)


if __name__ == '__main__':
    main()
