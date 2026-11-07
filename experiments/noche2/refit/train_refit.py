"""
experiments/noche2/refit/train_refit.py
=======================================
Motor de entrenamiento para el experimento noche2_refit.
Hipótesis: entrenar con 2020-2023 + 2024 (pronósticos NWP reales) enseña al
modelo la distribución real del error del NWP y aporta datos más recientes.

Esquema de validación honesto out-of-fold (OOF):
- Cross-fitting por trimestres de 2024 (Q1, Q2, Q3, Q4).
- Para cada fold Q:
  - Val: trimestre Q de 2024.
  - Train: 2020-2023 (PP + ruido empírico calibrado SOLO con los otros 3 trimestres de 2024)
           + los otros 3 trimestres de 2024 (meteo honesta real).
- Evaluación val OOF: concatenación de las predicciones de los 4 trimestres de 2024.
- Selección de la mejor variante por horizonte h basada EXCLUSIVAMENTE en val OOF.
- Refit final: reentrenar con 2020-2023 + TODO 2024 (n_iter = int(median(best_iter) * 1.1))
  y predecir test 2025.
"""
from __future__ import annotations
import os
import sys
import time
import json
import gc
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import partial

print = partial(print, flush=True)

# Configurar rutas
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
EXP_DIR = os.path.join(ROOT, 'experiments')
sys.path.insert(0, EXP_DIR)

from common import evaluate_test, log_result, HORIZONS, SEED, metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(BASE_DIR, 'preds')
OOF_DIR = os.path.join(BASE_DIR, 'oof_preds')
CACHE_DIR = os.path.join(EXP_DIR, 'noche', 'v3_base', 'cache')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Hiperparámetros de best_params_optuna.json + contrato noche2
OPTUNA_PATH = os.path.join(EXP_DIR, 'planA_quickwins', 'best_params_optuna.json')
with open(OPTUNA_PATH, 'r') as f:
    BEST_OPTUNA = json.load(f)

BASE_LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'seed': SEED,
    'verbose': -1,
    'num_threads': 6,  # Contrato noche2: 6 hilos
    'feature_pre_filter': False,
    'importance_type': 'gain',
    **BEST_OPTUNA
}


def load_cache_for_h(h: int) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """Carga cache parquet para horizonte h, optimizando tipos."""
    fpath = os.path.join(CACHE_DIR, f'features_h{h}.parquet')
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No existe el archivo de cache: {fpath}")
    
    # Cargar dataframe
    df = pd.read_parquet(fpath)
    df['Estacion'] = df['Estacion'].astype('category')
    
    t0_cols = [c for c in df.columns if c.startswith('meteo_t0_')]
    honest_cols = [c for c in df.columns if c.startswith('honest_meteo_')]
    pp_cols = [c for c in df.columns if c.startswith('pp_meteo_')]
    exclude = {'Date', 'Estacion', 'split', 'Target', f'Target_{h}h'} | set(t0_cols) | set(honest_cols) | set(pp_cols)
    chem_cols = [c for c in df.columns if c not in exclude]

    # Convertir numéricas a float32 para ahorrar RAM
    for c in chem_cols + t0_cols + honest_cols + pp_cols:
        if df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')

    return df, chem_cols, t0_cols, honest_cols, pp_cols


def inject_meteo_noise(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    honest_cols: list[str],
    pp_cols: list[str],
    seed: int = 42
) -> pd.DataFrame:
    """
    Inyecta en train_df ruido empírico de NWP muestreado de calib_df (honest - pp),
    condicionado estrictamente por la hora del día.
    Para evitar data leakage, calib_df DEBE excluir el trimestre que se valida.
    """
    train_noisy = train_df.copy()
    rng = np.random.default_rng(seed)

    calib_hour = calib_df['Date'].dt.hour.values
    tr_hour = train_noisy['Date'].dt.hour.values

    for h_col in honest_cols:
        suffix = h_col.replace('honest_meteo_', '')
        pp_col = f'pp_meteo_{suffix}'
        if pp_col not in calib_df.columns:
            continue

        val_diff = (calib_df[h_col] - calib_df[pp_col]).values
        valid_mask = np.isfinite(val_diff)

        sampled_noise = np.zeros(len(train_noisy), dtype='float32')
        for hr in range(24):
            hr_calib_mask = valid_mask & (calib_hour == hr)
            hr_tr_mask = (tr_hour == hr)
            n_tr = int(hr_tr_mask.sum())
            if n_tr == 0:
                continue

            residuals_pool = val_diff[hr_calib_mask]
            if len(residuals_pool) > 0:
                sampled_noise[hr_tr_mask] = rng.choice(residuals_pool, size=n_tr, replace=True)
            else:
                sampled_noise[hr_tr_mask] = 0.0

        train_noisy[h_col] = (train_noisy[h_col] + sampled_noise).astype('float32')

        # Acotar físicamente
        if 'rh' in h_col or 'cloud' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, 100.0)
        elif 'rad' in h_col or 'ws' in h_col or 'precip' in h_col:
            train_noisy[h_col] = np.clip(train_noisy[h_col], 0.0, None)

    # Recalcular derivadas
    if 'honest_meteo_target_temp' in train_noisy.columns and 'meteo_t0_temp' in train_noisy.columns:
        train_noisy['honest_meteo_delta_temp'] = (train_noisy['honest_meteo_target_temp'] - train_noisy['meteo_t0_temp']).astype('float32')
    if 'honest_meteo_target_rh' in train_noisy.columns and 'meteo_t0_rh' in train_noisy.columns:
        train_noisy['honest_meteo_delta_rh'] = (train_noisy['honest_meteo_target_rh'] - train_noisy['meteo_t0_rh']).astype('float32')
    if 'honest_meteo_target_temp' in train_noisy.columns and 'honest_meteo_target_rad' in train_noisy.columns:
        train_noisy['honest_meteo_target_temp_x_rad'] = (train_noisy['honest_meteo_target_temp'] * train_noisy['honest_meteo_target_rad']).astype('float32')

    return train_noisy


def cross_fit_2024(
    h: int,
    df: pd.DataFrame,
    chem_cols: list[str],
    t0_cols: list[str],
    honest_cols: list[str],
    pp_cols: list[str],
    noise_injection: bool = True,
    is_nwp_feature: bool = False,
    weight_2024: float = 1.0,
    peak_weights: bool = False,
    leak_free_calib: bool = True,
    seed: int = SEED
) -> tuple[dict, pd.DataFrame, list[int]]:
    """
    Ejecuta el esquema de cross-fitting en 4 folds (trimestres de 2024).
    Retorna: (oof_metrics, oof_preds_df, best_iters_per_fold)
    """
    target_col = 'Target'
    cat_cols = ['Estacion']
    base_features = ['Estacion'] + chem_cols + t0_cols + honest_cols
    features = list(base_features)
    if is_nwp_feature:
        features.append('is_nwp')

    # Particiones
    train_mask = (df['split'] == 'train') & df[target_col].notna()
    val_mask = (df['split'] == 'val') & df[target_col].notna()
    
    tr_df_base = df[train_mask].copy().reset_index(drop=True)
    val_df_full = df[val_mask].copy().reset_index(drop=True)

    val_df_full['quarter'] = val_df_full['Date'].dt.quarter

    oof_preds = np.zeros(len(val_df_full), dtype='float32')
    best_iters = []
    fold_times = []

    print(f"    Iniciando 4-fold cross-fitting en 2024 (n_val={len(val_df_full)})...")

    for q in [1, 2, 3, 4]:
        t_fold_start = time.time()
        q_val_idx = (val_df_full['quarter'] == q).values
        q_tr_idx = (val_df_full['quarter'] != q).values

        # Calibración de ruido empírico
        if noise_injection:
            if leak_free_calib:
                calib_df = val_df_full[q_tr_idx]
            else:
                calib_df = val_df_full
            tr_part_a = inject_meteo_noise(tr_df_base, calib_df, honest_cols, pp_cols, seed=seed)
        else:
            tr_part_a = tr_df_base.copy()

        # Parte B: los otros 3 trimestres de 2024
        tr_part_b = val_df_full[q_tr_idx].copy()

        # Característica is_nwp
        if is_nwp_feature:
            tr_part_a['is_nwp'] = np.float32(0.0)
            tr_part_b['is_nwp'] = np.float32(1.0)
            val_df_full.loc[q_val_idx, 'is_nwp'] = np.float32(1.0)

        # Unir train para este fold
        tr_fold = pd.concat([tr_part_a, tr_part_b], ignore_index=True)

        # Calcular pesos de muestra
        weights = np.ones(len(tr_fold), dtype='float32')
        if weight_2024 != 1.0:
            # Las primeras len(tr_part_a) filas son 2020-2023; las siguientes son 2024
            weights[len(tr_part_a):] = np.float32(weight_2024)

        if peak_weights:
            y_fold = tr_fold[target_col].values
            p90 = np.percentile(y_fold, 90.0)
            pk_w = 1.0 + (y_fold / p90) ** 2
            weights = (weights * pk_w).astype('float32')

        X_tr = tr_fold[features]
        y_tr = tr_fold[target_col].values

        va_fold = val_df_full[q_val_idx]
        X_va = va_fold[features]
        y_va = va_fold[target_col].values

        lgb_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, weight=weights, free_raw_data=False)
        lgb_va = lgb.Dataset(X_va, y_va, reference=lgb_tr, categorical_feature=cat_cols, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0)
        ]

        params = dict(BASE_LGB_PARAMS)
        params['seed'] = seed

        model = lgb.train(
            params,
            lgb_tr,
            num_boost_round=2000,
            valid_sets=[lgb_va],
            callbacks=callbacks
        )

        b_iter = model.best_iteration
        best_iters.append(b_iter)
        pred_va = model.predict(X_va, num_iteration=b_iter)
        oof_preds[q_val_idx] = pred_va

        t_fold = time.time() - t_fold_start
        fold_times.append(t_fold)
        fold_rmse = float(np.sqrt(np.mean((y_va - pred_va) ** 2)))
        print(f"      Fold Q{q} | N={len(va_fold):5d} | BestIter={b_iter:4d} | RMSE={fold_rmse:6.3f} | Tiempo={t_fold:4.1f}s")

    # Métricas globales OOF en todo 2024
    y_true_all = val_df_full[target_col].values
    oof_m = metrics(y_true_all, oof_preds)
    oof_m['median_best_iter'] = int(np.median(best_iters))
    oof_m['mean_best_iter'] = float(np.mean(best_iters))
    oof_m['total_cv_time'] = float(np.sum(fold_times))

    preds_oof_df = pd.DataFrame({
        'Date': val_df_full['Date'],
        'Estacion': val_df_full['Estacion'],
        'h': h,
        'split': 'val',
        'y_true': y_true_all.astype('float32'),
        'y_pred': oof_preds.astype('float32')
    })

    return oof_m, preds_oof_df, best_iters


def refit_and_evaluate_test(
    h: int,
    df: pd.DataFrame,
    chem_cols: list[str],
    t0_cols: list[str],
    honest_cols: list[str],
    pp_cols: list[str],
    best_iters: list[int],
    noise_injection: bool = True,
    is_nwp_feature: bool = False,
    weight_2024: float = 1.0,
    peak_weights: bool = False,
    iter_mult: float = 1.1,
    seed: int = SEED,
    variant_name: str = 'refit_best'
) -> tuple[dict, pd.DataFrame]:
    """
    Reentrena con 2020-2023 + TODO 2024 con iteraciones fijadas = int(median(best_iters) * iter_mult)
    y predice test 2025.
    """
    target_col = 'Target'
    cat_cols = ['Estacion']
    base_features = ['Estacion'] + chem_cols + t0_cols + honest_cols
    features = list(base_features)
    if is_nwp_feature:
        features.append('is_nwp')

    train_mask = (df['split'] == 'train') & df[target_col].notna()
    val_mask = (df['split'] == 'val') & df[target_col].notna()
    test_mask = (df['split'] == 'test') & df[target_col].notna()

    tr_df_base = df[train_mask].copy().reset_index(drop=True)
    val_df_full = df[val_mask].copy().reset_index(drop=True)
    te_df = df[test_mask].copy().reset_index(drop=True)

    # Inyección de ruido calibrado en 2020-2023 con TODO 2024
    if noise_injection:
        tr_part_a = inject_meteo_noise(tr_df_base, val_df_full, honest_cols, pp_cols, seed=seed)
    else:
        tr_part_a = tr_df_base.copy()

    tr_part_b = val_df_full.copy()

    if is_nwp_feature:
        tr_part_a['is_nwp'] = np.float32(0.0)
        tr_part_b['is_nwp'] = np.float32(1.0)
        te_df['is_nwp'] = np.float32(1.0)

    tr_full = pd.concat([tr_part_a, tr_part_b], ignore_index=True)

    # Pesos
    weights = np.ones(len(tr_full), dtype='float32')
    if weight_2024 != 1.0:
        weights[len(tr_part_a):] = np.float32(weight_2024)

    if peak_weights:
        y_all = tr_full[target_col].values
        p90 = np.percentile(y_all, 90.0)
        pk_w = 1.0 + (y_all / p90) ** 2
        weights = (weights * pk_w).astype('float32')

    X_tr = tr_full[features]
    y_tr = tr_full[target_col].values
    X_te = te_df[features]
    y_te = te_df[target_col].values

    num_iter = int(round(np.median(best_iters) * iter_mult))
    print(f"    Reentrenando refit completo 2020-2024 con {num_iter} árboles (mediana={np.median(best_iters):.0f} * {iter_mult})...")

    lgb_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, weight=weights, free_raw_data=False)
    params = dict(BASE_LGB_PARAMS)
    params['seed'] = seed

    t_start = time.time()
    model = lgb.train(
        params,
        lgb_tr,
        num_boost_round=num_iter
    )
    t_refit = time.time() - t_start

    y_pred_te = model.predict(X_te, num_iteration=num_iter)

    # Evaluar métricas test con evaluate_test de common
    te_eval_df = te_df.copy()
    te_eval_df[f'Target_{h}h'] = te_eval_df[target_col]
    m_te = evaluate_test(te_eval_df, y_pred_te, h)
    m_te['refit_time'] = t_refit
    m_te['num_trees'] = num_iter

    preds_test_df = pd.DataFrame({
        'Date': te_df['Date'],
        'Estacion': te_df['Estacion'],
        'h': h,
        'split': 'test',
        'y_true': y_te.astype('float32'),
        'y_pred': y_pred_te.astype('float32')
    })

    return m_te, preds_test_df


VARIANTS = {
    'refit_base': {
        'noise_injection': True,
        'is_nwp_feature': False,
        'weight_2024': 1.0,
        'peak_weights': False,
        'leak_free_calib': True,
        'desc': 'Base: 2020-2023 ruido calibrado sin Q + 2024 w=1.0'
    },
    'refit_isnwp': {
        'noise_injection': True,
        'is_nwp_feature': True,
        'weight_2024': 1.0,
        'peak_weights': False,
        'leak_free_calib': True,
        'desc': 'Como base + feature is_nwp (0 en 2020-2023, 1 en 2024/25)'
    },
    'refit_isnwp_w2': {
        'noise_injection': True,
        'is_nwp_feature': True,
        'weight_2024': 2.0,
        'peak_weights': False,
        'leak_free_calib': True,
        'desc': 'is_nwp + peso extra x2 para filas 2024'
    },
    'refit_w2': {
        'noise_injection': True,
        'is_nwp_feature': False,
        'weight_2024': 2.0,
        'peak_weights': False,
        'leak_free_calib': True,
        'desc': 'Como base + peso extra x2 para filas 2024'
    },
    'refit_w4': {
        'noise_injection': True,
        'is_nwp_feature': False,
        'weight_2024': 4.0,
        'peak_weights': False,
        'leak_free_calib': True,
        'desc': 'Como base + peso extra x4 para filas 2024'
    },
    'refit_nonoise': {
        'noise_injection': False,
        'is_nwp_feature': False,
        'weight_2024': 1.0,
        'peak_weights': False,
        'leak_free_calib': True,
        'desc': '2020-2023 sin ruido + 2024 w=1.0'
    },
    'refit_peak_w': {
        'noise_injection': True,
        'is_nwp_feature': False,
        'weight_2024': 1.0,
        'peak_weights': True,
        'leak_free_calib': True,
        'desc': 'Como base + pesos picos w=1+(y/p90)^2'
    },
    'refit_leak_test': {
        'noise_injection': True,
        'is_nwp_feature': False,
        'weight_2024': 1.0,
        'peak_weights': False,
        'leak_free_calib': False,
        'desc': 'Como base pero ruido calibrado con TODO 2024 (test fuga)'
    }
}


def run_cv_exploration(horizons: list[int], variant_names: list[str]):
    """Corre cross-fitting de 2024 para las variantes y horizontes solicitados."""
    results_path = os.path.join(BASE_DIR, 'cv_results.csv')
    inv = pd.read_csv(os.path.join(ROOT, 'experiments', 'noche2', 'inventory.csv'))

    records = []
    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        records = existing_df.to_dict('records')

    for h in horizons:
        print(f"\n{'='*80}")
        print(f"  CARGANDO CACHE HORIZONTE h={h}")
        print(f"{'='*80}")
        df, chem, t0, honest, pp = load_cache_for_h(h)

        # Baseline D_honest_noise de referencia
        base_d_row = inv[(inv['exp'] == 'v3_base') & (inv['var'] == 'D_honest_noise') & (inv['h'] == h)]
        base_d_val = base_d_row['val_rmse'].values[0] if len(base_d_row) > 0 else np.nan
        base_d_test = base_d_row['test_rmse'].values[0] if len(base_d_row) > 0 else np.nan
        print(f"  Referencia D_honest_noise h={h:2d} -> Val RMSE: {base_d_val:6.4f} | Test RMSE: {base_d_test:6.4f}")

        for var_name in variant_names:
            if var_name not in VARIANTS:
                print(f"  Variante desconocida: {var_name}")
                continue

            cfg = VARIANTS[var_name]
            print(f"\n  --- Evaluando {var_name} (h={h}) ---")
            print(f"      Descripcion: {cfg['desc']}")

            oof_m, preds_oof_df, best_iters = cross_fit_2024(
                h=h,
                df=df,
                chem_cols=chem,
                t0_cols=t0,
                honest_cols=honest,
                pp_cols=pp,
                noise_injection=cfg['noise_injection'],
                is_nwp_feature=cfg['is_nwp_feature'],
                weight_2024=cfg['weight_2024'],
                peak_weights=cfg['peak_weights'],
                leak_free_calib=cfg['leak_free_calib']
            )

            # Guardar predicciones OOF intermedias
            oof_p = os.path.join(OOF_DIR, f'oof_{var_name}_h{h}.parquet')
            preds_oof_df.to_parquet(oof_p, index=False)

            diff_d = oof_m['RMSE'] - base_d_val
            pct_diff = (diff_d / base_d_val) * 100.0

            rec = {
                'horizon': h,
                'variant': var_name,
                'val_OOF_RMSE': round(oof_m['RMSE'], 5),
                'base_D_val_RMSE': round(base_d_val, 5),
                'diff_vs_D': round(diff_d, 5),
                'pct_diff_vs_D': round(pct_diff, 3),
                'val_MAE': round(oof_m['MAE'], 5),
                'val_R2': round(oof_m['R2'], 5),
                'val_RMSE_peaks': round(oof_m['RMSE_peaks'], 5),
                'val_bias_peaks': round(oof_m['bias_peaks'], 5),
                'median_best_iter': oof_m['median_best_iter'],
                'mean_best_iter': round(oof_m['mean_best_iter'], 1),
                'best_iters': str(best_iters),
                'cv_time_s': round(oof_m['total_cv_time'], 1),
                'desc': cfg['desc']
            }
            records.append(rec)

            # Actualizar CSV en cada variante
            df_out = pd.DataFrame(records)
            df_out = df_out.drop_duplicates(subset=['horizon', 'variant'], keep='last').reset_index(drop=True)
            df_out.to_csv(results_path, index=False)

            sign = "-" if diff_d < 0 else "+"
            status = "MEJORA" if diff_d < 0 else "PEOR/IGUAL"
            print(f"  [RESULTADO] h={h:2d} | {var_name:15s} | Val OOF RMSE: {oof_m['RMSE']:6.4f} (vs Base D {base_d_val:6.4f} -> {sign}{abs(diff_d):.4f}, {sign}{abs(pct_diff):.2f}%) [{status}] | Pk RMSE: {oof_m['RMSE_peaks']:6.2f} | Time: {oof_m['total_cv_time']:.1f}s")

        del df
        gc.collect()

    print("\n" + "="*80)
    print("  EXPLORACION COMPLETADA")
    print("="*80)


def run_full_pipeline_for_h(h: int, best_variant: str, iter_mult: float = 1.1, num_seeds: int = 1):
    """
    Toma la mejor variante para el horizonte h (elegida SOLO con val OOF),
    carga el cache, recupera las mejores iteraciones del CV (o corre CV si no existe),
    reentrena el modelo final con 2020-2023 + TODO 2024, predice test 2025,
    concatena val (OOF) + test (refit) y guarda el parquet oficial según el contrato.
    """
    print(f"\n{'='*80}")
    print(f"  PIPELINE COMPLETO HORIZONTE h={h} | VARIANTE: {best_variant}")
    print(f"{'='*80}")

    df, chem, t0, honest, pp = load_cache_for_h(h)
    cfg = VARIANTS[best_variant]

    # Verificar si ya tenemos los best_iters de CV
    results_path = os.path.join(BASE_DIR, 'cv_results.csv')
    best_iters = None
    if os.path.exists(results_path):
        res_df = pd.read_csv(results_path)
        m = res_df[(res_df['horizon'] == h) & (res_df['variant'] == best_variant)]
        if len(m) > 0:
            best_iters = json.loads(m['best_iters'].values[0])
            print(f"  Recuperados best_iters de CV previo: {best_iters}")

    if best_iters is None:
        print("  Ejecutando CV para obtener best_iters y OOF preds...")
        oof_m, preds_oof_df, best_iters = cross_fit_2024(
            h=h, df=df, chem_cols=chem, t0_cols=t0, honest_cols=honest, pp_cols=pp,
            noise_injection=cfg['noise_injection'],
            is_nwp_feature=cfg['is_nwp_feature'],
            weight_2024=cfg['weight_2024'],
            peak_weights=cfg['peak_weights'],
            leak_free_calib=cfg['leak_free_calib']
        )
    else:
        # Cargar predicciones OOF ya guardadas
        oof_p = os.path.join(OOF_DIR, f'oof_{best_variant}_h{h}.parquet')
        if os.path.exists(oof_p):
            preds_oof_df = pd.read_parquet(oof_p)
        else:
            print("  Recalculando OOF preds...")
            _, preds_oof_df, _ = cross_fit_2024(
                h=h, df=df, chem_cols=chem, t0_cols=t0, honest_cols=honest, pp_cols=pp,
                noise_injection=cfg['noise_injection'],
                is_nwp_feature=cfg['is_nwp_feature'],
                weight_2024=cfg['weight_2024'],
                peak_weights=cfg['peak_weights'],
                leak_free_calib=cfg['leak_free_calib']
            )

    # Reentrenar modelo final
    if num_seeds == 1:
        m_te, preds_test_df = refit_and_evaluate_test(
            h=h, df=df, chem_cols=chem, t0_cols=t0, honest_cols=honest, pp_cols=pp,
            best_iters=best_iters,
            noise_injection=cfg['noise_injection'],
            is_nwp_feature=cfg['is_nwp_feature'],
            weight_2024=cfg['weight_2024'],
            peak_weights=cfg['peak_weights'],
            iter_mult=iter_mult,
            seed=SEED,
            variant_name=best_variant
        )
    else:
        # Bagging multi-semillas
        print(f"  Ejecutando bagging con {num_seeds} semillas (42, 43, 44)...")
        seeds = [42, 43, 44][:num_seeds]
        test_preds_list = []
        for s in seeds:
            print(f"    Seed {s}...")
            m_s, p_s = refit_and_evaluate_test(
                h=h, df=df, chem_cols=chem, t0_cols=t0, honest_cols=honest, pp_cols=pp,
                best_iters=best_iters,
                noise_injection=cfg['noise_injection'],
                is_nwp_feature=cfg['is_nwp_feature'],
                weight_2024=cfg['weight_2024'],
                peak_weights=cfg['peak_weights'],
                iter_mult=iter_mult,
                seed=s,
                variant_name=best_variant
            )
            test_preds_list.append(p_s['y_pred'].values)
        
        avg_pred = np.mean(test_preds_list, axis=0).astype('float32')
        te_eval_df = df[df['split'] == 'test'].copy().reset_index(drop=True)
        te_eval_df = te_eval_df[te_eval_df['Target'].notna()].reset_index(drop=True)
        te_eval_df[f'Target_{h}h'] = te_eval_df['Target']
        m_te = evaluate_test(te_eval_df, avg_pred, h)
        m_te['num_seeds'] = num_seeds
        preds_test_df = pd.DataFrame({
            'Date': te_eval_df['Date'],
            'Estacion': te_eval_df['Estacion'],
            'h': h,
            'split': 'test',
            'y_true': te_eval_df['Target'].astype('float32'),
            'y_pred': avg_pred
        })

    # Concatenar val (OOF) y test (refit)
    preds_all = pd.concat([preds_oof_df, preds_test_df], ignore_index=True)
    out_parquet = os.path.join(PREDS_DIR, f'{best_variant}_h{h}.parquet')
    preds_all.to_parquet(out_parquet, index=False)
    print(f"  Predicciones guardadas en: {out_parquet} (total filas: {len(preds_all)})")

    # Registrar en results_registry.csv
    val_m = metrics(preds_oof_df['y_true'], preds_oof_df['y_pred'])
    notes = f"Refit 2020-2024 con {best_variant} ({cfg['desc']}). Val=OOF 4-folds 2024, Test=refit {m_te.get('num_trees', 0)} trees."
    log_result(
        experiment='noche2_refit',
        variant=best_variant,
        h=h,
        m=m_te,
        notes=notes
    )

    print(f"\n  [FINAL h={h}] Val OOF RMSE: {val_m['RMSE']:6.4f} | Test RMSE: {m_te['RMSE']:6.4f} | Test R2: {m_te['R2']:.4f} | Pk RMSE: {m_te['RMSE_peaks']:6.2f} | Pk Bias: {m_te['bias_peaks']:6.2f}")

    del df
    gc.collect()
    return val_m, m_te


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experimentos noche2_refit")
    parser.add_argument('--action', type=str, default='explore', choices=['explore', 'refit', 'both'])
    parser.add_argument('--horizons', type=str, default='24,4', help="Horizontes separados por comas, e.g. '24,4'")
    parser.add_argument('--variants', type=str, default='all', help="Variantes separadas por comas, o 'all'")
    parser.add_argument('--iter_mult', type=float, default=1.1, help="Multiplicador de iteraciones sobre mediana de CV")
    parser.add_argument('--seeds', type=int, default=1, help="Numero de semillas para refit final (1..3)")
    args = parser.parse_args()

    h_list = [int(x.strip()) for x in args.horizons.split(',')]

    if args.variants == 'all':
        var_list = ['refit_base', 'refit_isnwp', 'refit_w2', 'refit_w4', 'refit_nonoise', 'refit_peak_w', 'refit_leak_test']
    else:
        var_list = [x.strip() for x in args.variants.split(',')]

    if args.action in ['explore', 'both']:
        run_cv_exploration(h_list, var_list)

