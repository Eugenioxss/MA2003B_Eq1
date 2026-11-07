"""
experiments/planC_meteo/train_models.py
=======================================
Entrenamiento y evaluación multi-horizonte [1, 4, 8, 12, 24, 48] para:
1. v2_baseline: Baseline LightGBM con features v2 del SIMA.
2. perfect_prog: v2 + ERA5 reanálisis en t+h y t (cota superior teórica).
3. honest_forecast: v2 + pronóstico numérico histórico / previous runs en t+h.
4. meteo_only: Solo meteorología + calendario + estación (sin lags de contaminantes).

Registra resultados en results_registry.csv con log_result() y guarda tablas y figuras
en experiments/planC_meteo/.
"""
from __future__ import annotations
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
print = partial(print, flush=True)

# Raíz del proyecto
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    load_panel, build_v2_features, split, evaluate_test, log_result,
    LGB_PARAMS_V2, MODEL_STATIONS, HORIZONS
)
from meteo_features import (
    load_meteo_raw, build_meteo_features_for_horizon, merge_meteo_with_v2
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

# Parámetros LightGBM respetando num_threads=4 del contrato
PARAMS = {
    **LGB_PARAMS_V2,
    'num_threads': 4,
    'n_jobs': 4
}

CALENDAR_COLS = [
    'hour', 'month', 'dayofweek', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
]

EXACT_STATIONS = ['CE', 'NE', 'NE2', 'NO', 'NO2', 'NTE', 'NTE2', 'SE']


def train_and_eval_lgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    cat_cols: list[str] | None = None
) -> tuple[np.ndarray, lgb.Booster, dict]:
    """Entrena LightGBM con early stopping en val y predice en test."""
    train_mask = y_train.notna()
    val_mask = y_val.notna()

    dtrain = lgb.Dataset(X_train[train_mask], label=y_train[train_mask], categorical_feature=cat_cols or [])
    dval = lgb.Dataset(X_val[val_mask], label=y_val[val_mask], reference=dtrain, categorical_feature=cat_cols or [])

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)
    ]

    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Extraer importancia por ganancia
    gains = model.feature_importance(importance_type='gain')
    feat_names = model.feature_name()
    feat_imp = dict(zip(feat_names, [float(g) for g in gains]))

    return y_pred, model, feat_imp


def run_experiment():
    print("=" * 70)
    print("PLAN C: METEOROLOGÍA EXTERNA MULTI-HORIZONTE (SIMA + OPEN-METEO)")
    print("=" * 70)

    # 1. Cargar panel horario del SIMA y construir features v2
    print("\n[1/4] Cargando panel horario SIMA y construyendo features v2...")
    panel = load_panel(stations=MODEL_STATIONS)
    df_v2, v2_feature_names = build_v2_features(panel, stations=MODEL_STATIONS)
    df_v2['Estacion'] = df_v2['Estacion'].astype('category')
    print(f"Panel v2 construido: {len(df_v2)} filas, {len(v2_feature_names)} features.")

    # 2. Cargar datos meteorológicos crudos
    print("\n[2/4] Cargando datos meteorológicos crudos...")
    df_rean = load_meteo_raw(source='reanalysis', stations=MODEL_STATIONS)
    print(f"Reanálisis ERA5 cargado: {len(df_rean)} filas.")
    df_fcst = load_meteo_raw(source='forecast', stations=MODEL_STATIONS)
    print(f"Pronóstico histórico cargado: {len(df_fcst)} filas.")

    all_results = []
    all_results_exact = []
    all_importances = []

    # 3. Iterar por cada horizonte
    print("\n[3/4] Entrenando modelos por horizonte...")
    for h in HORIZONS:
        target_col = f'Target_{h}h'
        print(f"\n{'='*30} HORIZONTE h = {h} ({h}h) {'='*30}")

        # Generar features meteorológicas para h
        print(f"-> Construyendo features meteorológicas para h={h}...")
        rean_h, rean_cols = build_meteo_features_for_horizon(df_rean, h=h, source='reanalysis')
        fcst_h, fcst_cols = build_meteo_features_for_horizon(df_fcst, h=h, source='forecast', use_previous_runs=True)

        # Unir a v2
        df_rean_merged, _ = merge_meteo_with_v2(df_v2, rean_h)
        df_fcst_merged, _ = merge_meteo_with_v2(df_v2, fcst_h)
        df_rean_merged['Estacion'] = df_rean_merged['Estacion'].astype('category')
        df_fcst_merged['Estacion'] = df_fcst_merged['Estacion'].astype('category')

        # Splits temporales (train <= 2023, val == 2024, test == 2025)
        tr_v2, val_v2, te_v2 = split(df_v2)
        tr_re, val_re, te_re = split(df_rean_merged)
        tr_fc, val_fc, te_fc = split(df_fcst_merged)

        # --- VARIANTE 1: v2_baseline ---
        print(f"  [1] Entrenando v2_baseline (h={h})...")
        t0 = time.time()
        y_pred_v2, mod_v2, imp_v2 = train_and_eval_lgbm(
            tr_v2[v2_feature_names], tr_v2[target_col],
            val_v2[v2_feature_names], val_v2[target_col],
            te_v2[v2_feature_names], te_v2[target_col]
        )
        m_v2 = evaluate_test(te_v2, y_pred_v2, h)
        log_result('planC_meteo', 'v2_baseline', h, m_v2, f'Script 7 baseline reproduccion h={h}')
        all_results.append({'variant': 'v2_baseline', 'h': h, **m_v2, 'train_time': time.time() - t0})
        
        # Exact subset
        exact_mask_v2 = te_v2['Estacion'].isin(EXACT_STATIONS).values
        m_v2_ex = evaluate_test(te_v2[exact_mask_v2], y_pred_v2[exact_mask_v2], h)
        all_results_exact.append({'variant': 'v2_baseline', 'h': h, **m_v2_ex})
        print(f"      v2 RMSE: {m_v2['RMSE']:.2f} ppb | R2: {m_v2['R2']:.3f} | Skill: {m_v2.get('skill_vs_clim', 0)*100:.1f}%")

        # --- VARIANTE 2: perfect_prog (ERA5 reanálisis) ---
        print(f"  [2] Entrenando perfect_prog (ERA5 en t+h, h={h})...")
        re_features = v2_feature_names + rean_cols
        t0 = time.time()
        y_pred_re, mod_re, imp_re = train_and_eval_lgbm(
            tr_re[re_features], tr_re[target_col],
            val_re[re_features], val_re[target_col],
            te_re[re_features], te_re[target_col]
        )
        m_re = evaluate_test(te_re, y_pred_re, h)
        log_result('planC_meteo', 'perfect_prog', h, m_re, f'v2 + ERA5 reanalysis at t+h (cota superior) h={h}')
        all_results.append({'variant': 'perfect_prog', 'h': h, **m_re, 'train_time': time.time() - t0})
        
        exact_mask_re = te_re['Estacion'].isin(EXACT_STATIONS).values
        m_re_ex = evaluate_test(te_re[exact_mask_re], y_pred_re[exact_mask_re], h)
        all_results_exact.append({'variant': 'perfect_prog', 'h': h, **m_re_ex})
        print(f"      PerfectProg RMSE: {m_re['RMSE']:.2f} ppb | R2: {m_re['R2']:.3f} | Skill: {m_re.get('skill_vs_clim', 0)*100:.1f}%")

        for f_name, g_val in imp_re.items():
            all_importances.append({'horizon': h, 'variant': 'perfect_prog', 'feature': f_name, 'gain': g_val})

        # --- VARIANTE 3: honest_forecast (pronóstico histórico / previous runs) ---
        print(f"  [3] Entrenando honest_forecast (NWP operativo en t+h, h={h})...")
        fc_features = v2_feature_names + fcst_cols
        t0 = time.time()
        y_pred_fc, mod_fc, imp_fc = train_and_eval_lgbm(
            tr_fc[fc_features], tr_fc[target_col],
            val_fc[fc_features], val_fc[target_col],
            te_fc[fc_features], te_fc[target_col]
        )
        m_fc = evaluate_test(te_fc, y_pred_fc, h)
        note_fc = f'v2 + NWP forecast {"previous_day" if h>=24 else "seamless"} h={h}'
        log_result('planC_meteo', 'honest_forecast', h, m_fc, note_fc)
        all_results.append({'variant': 'honest_forecast', 'h': h, **m_fc, 'train_time': time.time() - t0})
        
        exact_mask_fc = te_fc['Estacion'].isin(EXACT_STATIONS).values
        m_fc_ex = evaluate_test(te_fc[exact_mask_fc], y_pred_fc[exact_mask_fc], h)
        all_results_exact.append({'variant': 'honest_forecast', 'h': h, **m_fc_ex})
        print(f"      HonestForecast RMSE: {m_fc['RMSE']:.2f} ppb | R2: {m_fc['R2']:.3f} | Skill: {m_fc.get('skill_vs_clim', 0)*100:.1f}%")

        for f_name, g_val in imp_fc.items():
            all_importances.append({'horizon': h, 'variant': 'honest_forecast', 'feature': f_name, 'gain': g_val})

        # --- VARIANTE 4: meteo_only (Sin lags químicos, solo meteorología + calendario + estación) ---
        print(f"  [4] Entrenando meteo_only (sin lags quimicos, h={h})...")
        meteo_only_features = ['Estacion'] + CALENDAR_COLS + fcst_cols
        t0 = time.time()
        y_pred_mo, mod_mo, imp_mo = train_and_eval_lgbm(
            tr_fc[meteo_only_features], tr_fc[target_col],
            val_fc[meteo_only_features], val_fc[target_col],
            te_fc[meteo_only_features], te_fc[target_col],
            cat_cols=['Estacion']
        )
        m_mo = evaluate_test(te_fc, y_pred_mo, h)
        log_result('planC_meteo', 'meteo_only', h, m_mo, f'Solo meteo NWP + calendario + estacion (sin lags) h={h}')
        all_results.append({'variant': 'meteo_only', 'h': h, **m_mo, 'train_time': time.time() - t0})
        
        exact_mask_mo = te_fc['Estacion'].isin(EXACT_STATIONS).values
        m_mo_ex = evaluate_test(te_fc[exact_mask_mo], y_pred_mo[exact_mask_mo], h)
        all_results_exact.append({'variant': 'meteo_only', 'h': h, **m_mo_ex})
        print(f"      MeteoOnly RMSE: {m_mo['RMSE']:.2f} ppb | R2: {m_mo['R2']:.3f} | Skill: {m_mo.get('skill_vs_clim', 0)*100:.1f}%")

    # 4. Guardar resúmenes y generar gráficos
    print("\n[4/4] Guardando resultados, tabla de métricas y gráficos...")
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics_summary.csv'), index=False)

    res_ex_df = pd.DataFrame(all_results_exact)
    res_ex_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics_8stations.csv'), index=False)

    imp_df = pd.DataFrame(all_importances)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

    # Gráfico 1: RMSE vs Horizonte
    plt.figure(figsize=(10, 6))
    for var, label, col, style in [
        ('v2_baseline', 'v2 Baseline (SIMA autoregresivo)', '#1f77b4', 'o--'),
        ('perfect_prog', 'Perfect Prog (v2 + ERA5 Reanalysis)', '#2ca02c', 's-'),
        ('honest_forecast', 'Honest Forecast (v2 + NWP Operativo)', '#ff7f0e', '^-'),
        ('meteo_only', 'Meteo Only (Sin lags quimicos)', '#d62728', 'd:')
    ]:
        sub = res_df[res_df['variant'] == var].sort_values('h')
        plt.plot(sub['h'], sub['RMSE'], style, label=label, color=col, linewidth=2, markersize=7)

    # Climatología de referencia
    clim_val = res_df[res_df['variant'] == 'v2_baseline'].sort_values('h')['RMSE_clim'].values
    plt.plot(HORIZONS, clim_val, 'k--', label='Climatologia (estacion x mes x hora)', alpha=0.6)

    plt.title('Comparación de Modelos por Horizonte: RMSE (ppb O3)', fontsize=13, fontweight='bold')
    plt.xlabel('Horizonte de Pronóstico h (horas)', fontsize=11)
    plt.ylabel('RMSE (ppb)', fontsize=11)
    plt.xticks(HORIZONS)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'rmse_vs_horizon.png'), dpi=200)
    plt.close()

    # Gráfico 2: Skill vs Climatología (%)
    plt.figure(figsize=(10, 6))
    for var, label, col, style in [
        ('v2_baseline', 'v2 Baseline', '#1f77b4', 'o--'),
        ('perfect_prog', 'Perfect Prog (ERA5)', '#2ca02c', 's-'),
        ('honest_forecast', 'Honest Forecast (NWP)', '#ff7f0e', '^-'),
        ('meteo_only', 'Meteo Only', '#d62728', 'd:')
    ]:
        sub = res_df[res_df['variant'] == var].sort_values('h')
        plt.plot(sub['h'], sub['skill_vs_clim'] * 100, style, label=label, color=col, linewidth=2, markersize=7)

    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.title('Skill Score vs Climatología [% = (1 - RMSE/RMSE_clim)*100]', fontsize=13, fontweight='bold')
    plt.xlabel('Horizonte de Pronóstico h (horas)', fontsize=11)
    plt.ylabel('Skill Score (%)', fontsize=11)
    plt.xticks(HORIZONS)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'skill_vs_horizon.png'), dpi=200)
    plt.close()

    # Gráfico 3: Importancia de features meteo a 24h y 48h
    for h_plot in [24, 48]:
        sub_imp = imp_df[(imp_df['horizon'] == h_plot) & (imp_df['variant'] == 'honest_forecast')]
        if len(sub_imp) > 0:
            top_feats = sub_imp.sort_values('gain', ascending=False).head(20)
            plt.figure(figsize=(10, 7))
            colors = ['#ff7f0e' if f.startswith('meteo_') else '#1f77b4' for f in top_feats['feature']]
            plt.barh(top_feats['feature'][::-1], top_feats['gain'][::-1], color=colors[::-1])
            plt.title(f'Top 20 Features por Ganancia (Honest Forecast h={h_plot}h)\n[Naranja = Meteorologia Externa | Azul = SIMA / Calendario]', fontsize=11, fontweight='bold')
            plt.xlabel('Gain Total en Árboles LightGBM', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGS_DIR, f'feature_importance_h{h_plot}.png'), dpi=200)
            plt.close()

    print("\n¡Experimento completado exitosamente!")
    print(f"Resumen guardado en: {os.path.join(OUTPUT_DIR, 'metrics_summary.csv')}")


if __name__ == '__main__':
    run_experiment()
