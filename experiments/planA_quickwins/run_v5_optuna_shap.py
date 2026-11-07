"""
experiments/planA_quickwins/run_v5_optuna_shap.py
================================================
Optuna Tuning ligero (25 trials para h=1, h=12, h=24 en val 2024),
entrenamiento de V5 final en 100% train con los mejores hiperparámetros
para todos los horizontes [1, 4, 8, 12, 24, 48],
análisis SHAP (TreeExplainer, 5k test rows) para h=1 y h=24,
y generación de figuras comparativas y de interpretabilidad.
"""
import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.abspath('experiments'))
from common import load_panel, split, evaluate_test, log_result, LGB_PARAMS_V2, HORIZONS, SEED
from planA_quickwins.features_planA import build_planA_features

OUT_DIR = os.path.join('experiments', 'planA_quickwins')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

def main():
    print("=" * 70)
    print("  EXPERIMENTO PLAN A: V5 TUNED (OPTUNA) + INTERPRETABILIDAD SHAP")
    print("=" * 70)

    # 1. Cargar panel y construir features V4 (set completo)
    t0 = time.time()
    panel = load_panel()
    df_feat, features = build_planA_features(panel, variant='V4')
    print(f"Features V4 construidos: {len(features)} variables en {time.time()-t0:.2f}s")

    # 2. Split temporal
    train_df, val_df, test_df = split(df_feat)
    
    # 3. Tuning con Optuna
    print("\n--- Optuna Tuning Ligero (25 trials en val 2024) ---")
    tuning_horizons = [1, 12, 24]
    
    # Pre-filtrar datos de validación y train submuestreado (stride 2) para agilidad en tuning
    tuning_data = {}
    for h in tuning_horizons:
        target = f'Target_{h}h'
        tr_h = train_df.dropna(subset=[target]).iloc[::2].reset_index(drop=True)
        va_h = val_df.dropna(subset=[target]).reset_index(drop=True)
        tuning_data[h] = (tr_h[features], tr_h[target], va_h[features], va_h[target], h)

    def objective(trial):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'seed': SEED,
            'verbose': -1,
            'num_threads': 4,
            'feature_pre_filter': False,
            'importance_type': 'gain',
            'num_leaves': trial.suggest_categorical('num_leaves', [31, 63, 127]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.03, 0.05, 0.08]),
            'min_child_samples': trial.suggest_categorical('min_child_samples', [20, 50, 100]),
            'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 1.0]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.8, 1.0]),
            'reg_alpha': trial.suggest_categorical('reg_alpha', [0.0, 0.1, 1.0]),
            'reg_lambda': trial.suggest_categorical('reg_lambda', [0.0, 0.1, 1.0]),
        }
        
        rmse_sum = 0.0
        for h in tuning_horizons:
            X_tr, y_tr, X_va, y_va, _ = tuning_data[h]
            lgb_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=['Estacion'], free_raw_data=False)
            lgb_va = lgb.Dataset(X_va, y_va, reference=lgb_tr, categorical_feature=['Estacion'], free_raw_data=False)
            model = lgb.train(
                param,
                lgb_tr,
                num_boost_round=1000,
                valid_sets=[lgb_va],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=40, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            y_pred = model.predict(X_va, num_iteration=model.best_iteration)
            rmse = float(np.sqrt(np.mean((y_va.values - y_pred) ** 2)))
            rmse_sum += rmse
            
        return rmse_sum / len(tuning_horizons)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    t_tune = time.time()
    study.optimize(objective, n_trials=25)
    best_params = study.best_params
    print(f"\nTuning finalizado en {time.time()-t_tune:.1f}s")
    print(f"Mejor score promedio val (h=1, 12, 24): {study.best_value:.4f}")
    print("Mejores parámetros encontrados:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Guardar mejores hiperparámetros
    with open(os.path.join(OUT_DIR, 'best_params_optuna.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    # 4. Entrenar V5 final en 100% train con los mejores params para todos los horizontes
    print("\n--- Entrenando V5 final en 100% de datos con early stopping 80 ---")
    v5_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'seed': SEED,
        'verbose': -1,
        'num_threads': 4,
        'feature_pre_filter': False,
        'importance_type': 'gain',
        **best_params
    }

    v5_metrics = []
    models_v5 = {}
    
    for h in HORIZONS:
        t_h = time.time()
        target_col = f'Target_{h}h'
        
        tr_h = train_df.dropna(subset=[target_col]).reset_index(drop=True)
        va_h = val_df.dropna(subset=[target_col]).reset_index(drop=True)
        ts_h = test_df.dropna(subset=[target_col]).reset_index(drop=True)

        X_tr, y_tr = tr_h[features], tr_h[target_col]
        X_va, y_va = va_h[features], va_h[target_col]
        X_ts, y_ts = ts_h[features], ts_h[target_col]

        lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=['Estacion'])
        lgb_val = lgb.Dataset(X_va, y_va, reference=lgb_train, categorical_feature=['Estacion'])

        model = lgb.train(
            v5_params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                lgb.log_evaluation(period=0),
            ]
        )

        y_pred = model.predict(X_ts, num_iteration=model.best_iteration)
        m = evaluate_test(ts_h, y_pred, h)
        m['best_iteration'] = int(model.best_iteration)
        m['train_time_s'] = round(time.time() - t_h, 2)
        m['n_features'] = len(features)
        
        notes = f"V5_tuned, iters={model.best_iteration}, num_leaves={best_params['num_leaves']}, lr={best_params['learning_rate']}"
        log_result('planA_quickwins', 'V5_tuned', h, m, notes=notes)

        # Guardar predicciones y modelo
        pred_df = ts_h[['Date', 'Estacion', target_col]].copy()
        pred_df['y_pred'] = y_pred
        pred_df.to_parquet(os.path.join(OUT_DIR, f"preds_V5_tuned_{h}h.parquet"), index=False)
        model.save_model(os.path.join(OUT_DIR, f"model_V5_tuned_{h}h.txt"))
        models_v5[h] = model

        res_entry = {
            'variant': 'V5_tuned',
            'h': h,
            'RMSE': round(m['RMSE'], 4),
            'MAE': round(m['MAE'], 4),
            'R2': round(m['R2'], 4),
            'RMSE_peaks': round(m['RMSE_peaks'], 4),
            'bias_peaks': round(m['bias_peaks'], 4),
            'RMSE_day': round(m.get('RMSE_day', 0.0), 4),
            'skill_vs_clim': round(m.get('skill_vs_clim', 0.0), 4),
            'best_iter': model.best_iteration,
            'train_time_s': m['train_time_s'],
            'n_features': len(features)
        }
        v5_metrics.append(res_entry)

        print(f"  [V5_tuned] h={h:2d} | RMSE: {m['RMSE']:.4f} | R2: {m['R2']:.4f} | "
              f"RMSE_pk: {m['RMSE_peaks']:.4f} | Bias_pk: {m['bias_peaks']:.2f} | "
              f"Skill: {m['skill_vs_clim']:.4f} | Iter: {model.best_iteration} ({m['train_time_s']:.1f}s)")

    # Actualizar ablation summary con V5
    abl_file = os.path.join(OUT_DIR, 'ablation_summary.csv')
    if os.path.exists(abl_file):
        df_prev = pd.read_csv(abl_file)
        df_all = pd.concat([df_prev, pd.DataFrame(v5_metrics)], ignore_index=True)
    else:
        df_all = pd.DataFrame(v5_metrics)
    df_all.to_csv(abl_file, index=False)

    # 5. Análisis SHAP para h=1 y h=24
    print("\n--- Calculando SHAP para h=1 y h=24 (muestra 5,000 filas test) ---")
    np.random.seed(SEED)
    sample_idx = np.random.choice(len(test_df), size=min(5000, len(test_df)), replace=False)
    test_sample = test_df.iloc[sample_idx][features].copy()

    for h_shap in [1, 24]:
        print(f"  Calculando SHAP para h={h_shap}h...")
        t_s = time.time()
        mod = models_v5[h_shap]
        explainer = shap.TreeExplainer(mod)
        shap_values = explainer.shap_values(test_sample)

        # Beeswarm plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, test_sample, max_display=20, show=False)
        plt.title(f"SHAP Summary (Top 20 Features) - V5 Tuned h={h_shap}h", fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_summary_h{h_shap}.png"), bbox_inches='tight')
        plt.close()

        # Bar plot de importancia global media |SHAP|
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, test_sample, plot_type='bar', max_display=20, show=False)
        plt.title(f"Top 20 Features por Importancia Media |SHAP| - h={h_shap}h", fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_bar_h{h_shap}.png"), bbox_inches='tight')
        plt.close()
        print(f"  Figuras SHAP para h={h_shap}h guardadas ({time.time()-t_s:.1f}s)")

    # 6. Generar Figuras Comparativas de Ablación
    print("\n--- Generando Figuras Comparativas de Ablación ---")
    df_all = pd.read_csv(abl_file)
    
    # Figura 1: RMSE por Horizonte y Variante
    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("Set2", n_colors=df_all['variant'].nunique())
    ax = sns.barplot(data=df_all, x='h', y='RMSE', hue='variant', palette=palette)
    plt.title("Comparación de RMSE en Test (2025) por Variante y Horizonte", fontsize=14, fontweight='bold')
    plt.xlabel("Horizonte de Pronóstico (horas)", fontsize=12)
    plt.ylabel("RMSE (ppb)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Variante", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablation_rmse_by_horizon.png"))
    plt.close()

    # Figura 2: R2 por Horizonte y Variante
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_all, x='h', y='R2', hue='variant', palette=palette)
    plt.title("Comparación de R² en Test (2025) por Variante y Horizonte", fontsize=14, fontweight='bold')
    plt.xlabel("Horizonte de Pronóstico (horas)", fontsize=12)
    plt.ylabel("R²", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Variante", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablation_r2_by_horizon.png"))
    plt.close()

    # Figura 3: Skill vs Climatología
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_all, x='h', y='skill_vs_clim', hue='variant', palette=palette)
    plt.title("Skill Score vs Climatología (1 - RMSE / RMSE_clim) en Test (2025)", fontsize=14, fontweight='bold')
    plt.xlabel("Horizonte de Pronóstico (horas)", fontsize=12)
    plt.ylabel("Skill Score", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Variante", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablation_skill_by_horizon.png"))
    plt.close()

    # Figura 4: Sesgo en Picos (y >= p95)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_all, x='h', y='bias_peaks', hue='variant', palette=palette)
    plt.title("Sesgo en Eventos Extremos (Picos >= p95) por Horizonte y Variante", fontsize=14, fontweight='bold')
    plt.xlabel("Horizonte de Pronóstico (horas)", fontsize=12)
    plt.ylabel("Sesgo Medio en Picos (ppb)", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Variante", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablation_peak_bias.png"))
    plt.close()

    print("Figuras de ablación guardadas exitosamente en experiments/planA_quickwins/")
    print(f"\nProceso V5 completado en {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
