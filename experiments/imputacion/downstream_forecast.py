"""
experiments/imputacion/downstream_forecast.py
=============================================
Prueba downstream: ¿La imputación ayuda al pronóstico de Ozono?
Compara:
  1. v2_native_nan: LightGBM v2 sobre panel crudo (manejo nativo de NaN)
  2. v2_tiered_imputed: LightGBM v2 sobre panel imputado por niveles (impute_tiered)
  3. v2_imputed_with_flags: v2 sobre panel imputado + indicadores de faltante (flags)

Evalúa sobre los targets reales observados de 2025 (conjunto estándar)
para horizontes h=1, 8, 24 y registra resultados con log_result().
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    load_panel, build_v2_features, split, evaluate_test, log_result,
    LGB_PARAMS_V2, MODEL_STATIONS, add_targets, HORIZONS
)
from imputers import impute_tiered, compute_distance_matrix, build_climatology_map

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(EXP_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

DOWNSTREAM_HORIZONS = [1, 8, 24]
KEY_FLAG_VARS = ['O3', 'NOX', 'PM2.5', 'TOUT', 'SR', 'U_Wind', 'V_Wind']

def run_downstream():
    print("=" * 70, flush=True)
    print("INICIANDO PRUEBA DOWNSTREAM DE PRONÓSTICO (O3)", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Cargar panel crudo original
    print("\n[1/5] Cargando panel crudo...", flush=True)
    panel_raw = load_panel()
    
    # Guardar máscara de faltantes original antes de imputar
    missing_flags = {}
    for v in KEY_FLAG_VARS:
        if v in panel_raw.columns:
            missing_flags[f'missing_{v}'] = panel_raw[v].isna().astype(np.int8)
    df_missing_flags = pd.DataFrame(missing_flags)
    df_missing_flags['Estacion'] = panel_raw['Estacion']
    df_missing_flags['Date'] = panel_raw['Date']
    
    # Extraer targets verdaderos no imputados
    print("  Calculando targets reales observados sobre panel crudo...", flush=True)
    panel_with_targets = add_targets(panel_raw.copy(), horizons=DOWNSTREAM_HORIZONS)
    target_cols = [f'Target_{h}h' for h in DOWNSTREAM_HORIZONS]
    true_targets = panel_with_targets[['Estacion', 'Date'] + target_cols].copy()
    
    # 2. Imputar panel con estrategia por niveles
    print("\n[2/5] Aplicando imputación por niveles (impute_tiered)...", flush=True)
    cache_imputed = os.path.join(EXP_DIR, 'panel_tiered_imputed.parquet')
    if os.path.exists(cache_imputed):
        print(f"  Cargando panel imputado desde cache: {cache_imputed}", flush=True)
        panel_imputed = pd.read_parquet(cache_imputed)
    else:
        panel_imputed = impute_tiered(panel_raw)
        panel_imputed.to_parquet(cache_imputed, index=False)
        print(f"  Panel imputado guardado en cache: {cache_imputed}", flush=True)
        
    # 3. Construir features para las 3 variantes
    print("\n[3/5] Construyendo features de v2...", flush=True)
    
    # Variante 1: Raw features (v2 nativo NaN)
    print("  Construyendo features v2 sobre panel crudo (Variante 1)...", flush=True)
    df_v2_raw, feats_raw = build_v2_features(panel_raw)
    
    # Variante 2: Imputed features (v2 sobre panel imputado)
    print("  Construyendo features v2 sobre panel imputado (Variante 2)...", flush=True)
    df_v2_imp, feats_imp = build_v2_features(panel_imputed)
    # Reemplazar targets en df_v2_imp con los targets reales crudos
    for h in DOWNSTREAM_HORIZONS:
        col_t = f'Target_{h}h'
        df_v2_imp[col_t] = df_v2_raw[col_t].values
        
    # Variante 3: Imputed features + Missingness flags
    print("  Construyendo features v2 con flags de faltante (Variante 3)...", flush=True)
    df_v2_flags = df_v2_imp.copy()
    # Filtrar flags a MODEL_STATIONS con mismo orden
    df_missing_flags = df_missing_flags[df_missing_flags['Estacion'].isin(MODEL_STATIONS)].sort_values(['Estacion', 'Date']).reset_index(drop=True)
    flag_feats = []
    for v in KEY_FLAG_VARS:
        flag_col = f'{v}_was_missing_lag1'
        flag_s = df_missing_flags.groupby('Estacion', sort=False)[f'missing_{v}'].shift(1)
        df_v2_flags[flag_col] = flag_s.fillna(0).astype(np.int8).values
        flag_feats.append(flag_col)
    feats_with_flags = feats_imp + flag_feats
    
    # 4. Entrenar y evaluar modelos para cada horizonte
    print("\n[4/5] Entrenando y evaluando LightGBM para h=1, 8, 24...", flush=True)
    lgb_params = LGB_PARAMS_V2.copy()
    lgb_params['num_threads'] = 3
    lgb_params['verbose'] = -1
    
    results = []
    
    variantes = [
        ('v2_native_nan', df_v2_raw, feats_raw, 'Baseline v2 nativo con NaNs'),
        ('v2_tiered_imputed', df_v2_imp, feats_imp, 'v2 con entradas imputadas por niveles'),
        ('v2_imputed_with_flags', df_v2_flags, feats_with_flags, 'v2 imputado + missingness indicators'),
    ]
    
    for var_name, df_data, feat_list, desc in variantes:
        print(f"\n--- Variante: {var_name} ({desc}) ---", flush=True)
        train_df, val_df, test_df = split(df_data)
        
        for h in DOWNSTREAM_HORIZONS:
            target_col = f'Target_{h}h'
            print(f"  Entrenando horizonte h={h}...", flush=True)
            
            # Filtrar donde target observado esté disponible (en train y val)
            tr = train_df.dropna(subset=[target_col])
            va = val_df.dropna(subset=[target_col])
            te = test_df.dropna(subset=[target_col])
            
            X_tr, y_tr = tr[feat_list], tr[target_col]
            X_va, y_va = va[feat_list], va[target_col]
            X_te = te[feat_list]
            
            dtrain = lgb.Dataset(X_tr, y_tr)
            dval = lgb.Dataset(X_va, y_va, reference=dtrain)
            
            bst = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=1500,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(60, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # Predicción sobre test
            y_pred = bst.predict(X_te, num_iteration=bst.best_iteration)
            
            # Evaluación estándar
            m = evaluate_test(te, y_pred, h)
            
            # Registrar resultado en registry global
            log_result('imputacion', var_name, h, m, notes=f'best_iter={bst.best_iteration}; {desc}')
            
            results.append({
                'variante': var_name,
                'h': h,
                'RMSE': m['RMSE'],
                'MAE': m['MAE'],
                'R2': m['R2'],
                'RMSE_peaks': m['RMSE_peaks'],
                'bias': m['bias'],
                'bias_peaks': m['bias_peaks'],
                'skill_vs_clim': m['skill_vs_clim'],
                'best_iter': bst.best_iteration,
                'n_test': m['n']
            })
            
            print(f"    h={h} -> RMSE={m['RMSE']:.3f}, R2={m['R2']:.4f}, MAE={m['MAE']:.3f}, "
                  f"RMSE_peaks={m['RMSE_peaks']:.3f}, Bias_peaks={m['bias_peaks']:+.2f}, "
                  f"Skill={m['skill_vs_clim']:.3f} (iter={bst.best_iteration})", flush=True)
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(EXP_DIR, 'downstream_results.csv'), index=False)
    print(f"\n[OK] Resultados downstream guardados en {os.path.join(EXP_DIR, 'downstream_results.csv')}", flush=True)
    
    # 5. Generar figuras comparativas
    print("\n[5/5] Generando figuras de downstream...", flush=True)
    sns.set_theme(style='whitegrid', font_scale=1.1)
    
    # Gráfica 1: Comparativa RMSE y R2 por horizonte
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df_res, x='h', y='RMSE', hue='variante', ax=axes[0], palette='Blues_r')
    axes[0].set_title('Impacto en RMSE (menor es mejor) — Test 2025', fontweight='bold')
    axes[0].set_ylabel('RMSE (ppb)')
    axes[0].set_xlabel('Horizonte (horas)')
    
    sns.barplot(data=df_res, x='h', y='R2', hue='variante', ax=axes[1], palette='Blues_r')
    axes[1].set_title('Impacto en R² (mayor es mejor) — Test 2025', fontweight='bold')
    axes[1].set_ylabel('R²')
    axes[1].set_xlabel('Horizonte (horas)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'downstream_rmse_r2.png'), dpi=200)
    plt.close()
    
    # Gráfica 2: RMSE picos
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_res, x='h', y='RMSE_peaks', hue='variante', ax=ax, palette='Oranges_r')
    ax.set_title('RMSE en Picos (Top 5% real ≥ p95) — Test 2025', fontweight='bold')
    ax.set_ylabel('RMSE Picos (ppb)')
    ax.set_xlabel('Horizonte (horas)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'downstream_peaks.png'), dpi=200)
    plt.close()
    
    print("[OK] Figuras de downstream generadas en experiments/imputacion/figures/", flush=True)

if __name__ == '__main__':
    run_downstream()
