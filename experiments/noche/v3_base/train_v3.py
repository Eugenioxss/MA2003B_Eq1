"""
experiments/noche/v3_base/train_v3.py
=====================================
Entrena y evalúa las variantes del modelo base v3 para h in [1, 4, 8, 12, 24, 48]:
  1. C_honest: Plan A V4 química/espacial + meteo honesta (PP-train / NWP-test)
  2. A_chem: Solo química/espacial Plan A V4 (sanity check h1 ~ 5.09)
  3. B_pp: Química + meteo seamless en t+h también en test (cota superior)
  4. D_honest_noise: Como C pero con inyección en train de ruido empírico de NWP 2024
  5. E_honest_w: Mejor de C/D según RMSE val 2024 + pesos w = 1 + (y/p90)^2

Uso:
  python train_v3.py --variant C_honest
  python train_v3.py --variant all
"""
from __future__ import annotations
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import partial
print = partial(print, flush=True)

EXP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, EXP_DIR)

from common import evaluate_test, log_result, HORIZONS, SEED, metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
PREDS_DIR = os.path.join(BASE_DIR, 'preds')
os.makedirs(PREDS_DIR, exist_ok=True)

# Hiperparámetros de best_params_optuna.json + contrato
OPTUNA_PATH = os.path.join(EXP_DIR, 'planA_quickwins', 'best_params_optuna.json')
with open(OPTUNA_PATH, 'r') as f:
    BEST_OPTUNA = json.load(f)

BASE_LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'seed': SEED,
    'verbose': -1,
    'num_threads': 4,
    'feature_pre_filter': False,
    'importance_type': 'gain',
    **BEST_OPTUNA
}


def load_cache_for_h(h: int) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """Carga dataset parquet del cache para el horizonte h y devuelve columnas agrupadas."""
    fpath = os.path.join(CACHE_DIR, f'features_h{h}.parquet')
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No existe el archivo de cache: {fpath}")
    df = pd.read_parquet(fpath)
    df['Estacion'] = df['Estacion'].astype('category')

    # Identificar grupos de columnas
    t0_cols = [c for c in df.columns if c.startswith('meteo_t0_')]
    honest_cols = [c for c in df.columns if c.startswith('honest_meteo_')]
    pp_cols = [c for c in df.columns if c.startswith('pp_meteo_')]
    exclude = {'Date', 'Estacion', 'split', 'Target', f'Target_{h}h'} | set(t0_cols) | set(honest_cols) | set(pp_cols)
    chem_cols = [c for c in df.columns if c not in exclude]

    return df, chem_cols, t0_cols, honest_cols, pp_cols


def inject_meteo_noise(train_df: pd.DataFrame, val_df: pd.DataFrame, honest_cols: list[str], pp_cols: list[str], t0_cols: list[str], seed: int = 42) -> pd.DataFrame:
    """
    Agrega a train_df en las columnas honest_meteo ruido empírico muestreado
    de (val[honest_meteo] - val[pp_meteo]) en 2024, condicionado por hora del día.
    """
    train_noisy = train_df.copy()
    rng = np.random.default_rng(seed)

    val_hour = val_df['Date'].dt.hour.values
    tr_hour = train_noisy['Date'].dt.hour.values

    # Mapeo de columnas correspondientes
    for h_col in honest_cols:
        suffix = h_col.replace('honest_meteo_', '')
        pp_col = f'pp_meteo_{suffix}'
        if pp_col not in val_df.columns:
            continue

        # Residuo empírico en val 2024
        val_diff = (val_df[h_col] - val_df[pp_col]).values
        valid_mask = np.isfinite(val_diff)

        # Muestreo por hora
        sampled_noise = np.zeros(len(train_noisy), dtype='float32')
        for hr in range(24):
            hr_val_mask = valid_mask & (val_hour == hr)
            hr_tr_mask = (tr_hour == hr)
            n_tr = int(hr_tr_mask.sum())
            if n_tr == 0:
                continue

            residuals_pool = val_diff[hr_val_mask]
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

    # Recalcular derivadas si aplica
    if 'honest_meteo_target_temp' in train_noisy.columns and 'meteo_t0_temp' in train_noisy.columns:
        train_noisy['honest_meteo_delta_temp'] = train_noisy['honest_meteo_target_temp'] - train_noisy['meteo_t0_temp']
    if 'honest_meteo_target_rh' in train_noisy.columns and 'meteo_t0_rh' in train_noisy.columns:
        train_noisy['honest_meteo_delta_rh'] = train_noisy['honest_meteo_target_rh'] - train_noisy['meteo_t0_rh']
    if 'honest_meteo_target_temp' in train_noisy.columns and 'honest_meteo_target_rad' in train_noisy.columns:
        train_noisy['honest_meteo_target_temp_x_rad'] = train_noisy['honest_meteo_target_temp'] * train_noisy['honest_meteo_target_rad']

    return train_noisy


def train_and_eval_variant(
    variant: str,
    h: int,
    df: pd.DataFrame,
    features: list[str],
    sample_weight: np.ndarray | None = None,
    noise_injection: bool = False,
    honest_cols: list[str] | None = None,
    pp_cols: list[str] | None = None,
    t0_cols: list[str] | None = None,
) -> tuple[dict, dict, pd.DataFrame]:
    """Entrena LightGBM para un horizonte y variante específicos."""
    target_col = 'Target'
    cat_cols = ['Estacion']

    # Filtrar por split
    train_mask = (df['split'] == 'train') & df[target_col].notna()
    val_mask = (df['split'] == 'val') & df[target_col].notna()
    test_mask = (df['split'] == 'test') & df[target_col].notna()

    tr_df = df[train_mask].copy().reset_index(drop=True)
    va_df = df[val_mask].copy().reset_index(drop=True)
    te_df = df[test_mask].copy().reset_index(drop=True)

    # Si se requiere inyección de ruido empírico en train (variante D)
    if noise_injection and honest_cols and pp_cols and t0_cols:
        tr_df = inject_meteo_noise(tr_df, va_df, honest_cols, pp_cols, t0_cols, seed=SEED)

    X_tr, y_tr = tr_df[features], tr_df[target_col].values
    X_va, y_va = va_df[features], va_df[target_col].values
    X_te, y_te = te_df[features], te_df[target_col].values

    # Datasets LightGBM
    lgb_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, weight=sample_weight, free_raw_data=False)
    lgb_va = lgb.Dataset(X_va, y_va, reference=lgb_tr, categorical_feature=cat_cols, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=80, verbose=False),
        lgb.log_evaluation(period=0)
    ]

    t0 = time.time()
    model = lgb.train(
        BASE_LGB_PARAMS,
        lgb_tr,
        num_boost_round=2000,
        valid_sets=[lgb_va],
        callbacks=callbacks
    )
    train_time = time.time() - t0

    # Predicciones
    y_pred_va = model.predict(X_va, num_iteration=model.best_iteration)
    y_pred_te = model.predict(X_te, num_iteration=model.best_iteration)

    # Métricas
    m_va = metrics(y_va, y_pred_va)
    m_te = evaluate_test(te_df, y_pred_te, h)

    # Guardar dataframe de predicciones val + test
    preds_val = pd.DataFrame({
        'Date': va_df['Date'],
        'Estacion': va_df['Estacion'],
        'h': h,
        'split': 'val',
        'y_true': y_va.astype('float32'),
        'y_pred': y_pred_va.astype('float32')
    })
    preds_test = pd.DataFrame({
        'Date': te_df['Date'],
        'Estacion': te_df['Estacion'],
        'h': h,
        'split': 'test',
        'y_true': y_te.astype('float32'),
        'y_pred': y_pred_te.astype('float32')
    })
    preds_all = pd.concat([preds_val, preds_test], ignore_index=True)

    preds_path = os.path.join(PREDS_DIR, f'{variant}_h{h}.parquet')
    preds_all.to_parquet(preds_path, index=False)

    m_te['train_time'] = train_time
    m_te['best_iter'] = model.best_iteration
    m_te['n_features'] = len(features)
    m_va['best_iter'] = model.best_iteration

    return m_va, m_te, preds_all


def run_experiment(variants_to_run: list[str]):
    print("=" * 75)
    print("  EXPERIMENTO: NOCHE v3_base (Plan A + Plan C honesto)")
    print(f"  Variantes solicitadas: {variants_to_run}")
    print(f"  LightGBM params: num_threads=4, num_leaves={BASE_LGB_PARAMS['num_leaves']}, lr={BASE_LGB_PARAMS['learning_rate']}")
    print("=" * 75)

    all_val_metrics = {}
    all_test_metrics = {}

    # Cargar cache por horizonte bajo demanda
    cache_data = {}
    print("\nCargando datos de cache...")
    for h in HORIZONS:
        df, chem, t0, honest, pp = load_cache_for_h(h)
        cache_data[h] = (df, chem, t0, honest, pp)
        print(f"  Cache h={h}: {len(df)} filas, {len(chem)} chem, {len(t0)} t0, {len(honest)} honest, {len(pp)} pp")

    # Función auxiliar para correr una variante
    def execute_variant(var_name: str):
        print(f"\n{'='*75}")
        print(f"  INICIANDO VARIANTE: {var_name}")
        print(f"{'='*75}")
        var_test_results = []

        for h in HORIZONS:
            t_h = time.time()
            df, chem, t0, honest, pp = cache_data[h]

            if var_name == 'A_chem':
                features = ['Estacion'] + chem
                noise = False
                weights = None
                notes = 'Solo Plan A V4 features (quimica+espacial)'

            elif var_name == 'B_pp':
                features = ['Estacion'] + chem + t0 + pp
                noise = False
                weights = None
                notes = 'Chem + meteo seamless t+h en test (cota superior, NO operativa)'

            elif var_name == 'C_honest':
                features = ['Estacion'] + chem + t0 + honest
                noise = False
                weights = None
                notes = 'Protocolo honesto PP-train / NWP-test'

            elif var_name == 'D_honest_noise':
                features = ['Estacion'] + chem + t0 + honest
                noise = True
                weights = None
                notes = 'Honest con ruido empirico NWP 2024 en train'

            elif var_name == 'E_honest_w':
                # Seleccionar mejor de C o D según val RMSE
                val_c = all_val_metrics.get(('C_honest', h), {}).get('RMSE', None)
                if val_c is None:
                    p_c = os.path.join(PREDS_DIR, f'C_honest_h{h}.parquet')
                    if os.path.exists(p_c):
                        df_p = pd.read_parquet(p_c)
                        df_pv = df_p[df_p['split'] == 'val']
                        val_c = float(np.sqrt(np.mean((df_pv['y_true'] - df_pv['y_pred']) ** 2)))
                    else:
                        val_c = 999.0

                val_d = all_val_metrics.get(('D_honest_noise', h), {}).get('RMSE', None)
                if val_d is None:
                    p_d = os.path.join(PREDS_DIR, f'D_honest_noise_h{h}.parquet')
                    if os.path.exists(p_d):
                        df_p = pd.read_parquet(p_d)
                        df_pv = df_p[df_p['split'] == 'val']
                        val_d = float(np.sqrt(np.mean((df_pv['y_true'] - df_pv['y_pred']) ** 2)))
                    else:
                        val_d = 999.0

                winner = 'C_honest' if val_c <= val_d else 'D_honest_noise'
                noise = (winner == 'D_honest_noise')
                features = ['Estacion'] + chem + t0 + honest

                # Pesos de muestra w = 1 + (y/p90)^2
                tr_mask = (df['split'] == 'train') & df['Target'].notna()
                y_tr = df.loc[tr_mask, 'Target'].values
                p90 = np.percentile(y_tr, 90.0)
                weights = 1.0 + (y_tr / p90) ** 2
                notes = f'Mejor de C/D ({winner}, val RMSE={min(val_c, val_d):.2f}) + pesos w=1+(y/p90)^2 (p90={p90:.1f})'

            else:
                raise ValueError(f"Variante desconocida: {var_name}")

            # Entrenar y evaluar
            m_va, m_te, _ = train_and_eval_variant(
                variant=var_name,
                h=h,
                df=df,
                features=features,
                sample_weight=weights,
                noise_injection=noise,
                honest_cols=honest,
                pp_cols=pp,
                t0_cols=t0
            )

            all_val_metrics[(var_name, h)] = m_va
            all_test_metrics[(var_name, h)] = m_te

            # Registrar en registry oficial
            log_result(
                experiment='noche_v3_base',
                variant=var_name,
                h=h,
                m=m_te,
                notes=notes
            )

            var_test_results.append({
                'variant': var_name, 'h': h,
                'val_RMSE': m_va['RMSE'], 'test_RMSE': m_te['RMSE'],
                'test_R2': m_te['R2'], 'test_RMSE_peaks': m_te['RMSE_peaks'],
                'test_bias_peaks': m_te['bias_peaks'],
                'skill': m_te.get('skill_vs_clim', 0.0) * 100,
                'time_s': m_te['train_time'], 'best_iter': m_te['best_iter']
            })

            print(f"  h={h:2d} | Val RMSE: {m_va['RMSE']:6.3f} | Test RMSE: {m_te['RMSE']:6.3f} | R2: {m_te['R2']:.3f} | Pk RMSE: {m_te['RMSE_peaks']:6.2f} | Pk Bias: {m_te['bias_peaks']:6.2f} | Time: {m_te['train_time']:5.1f}s (iter={m_te['best_iter']})")

        return pd.DataFrame(var_test_results)

    # Ejecutar en orden de prioridad:
    # Si se pide 'all', se corre en el orden estricto de la tarea:
    # C_honest (prioridad absoluta), luego A_chem, B_pp, D_honest_noise, E_honest_w
    order = ['C_honest', 'A_chem', 'B_pp', 'D_honest_noise', 'E_honest_w']
    selected = [v for v in order if v in variants_to_run]

    summary_tables = []
    for v in selected:
        df_res = execute_variant(v)
        summary_tables.append(df_res)

    if summary_tables:
        new_summary = pd.concat(summary_tables, ignore_index=True)
        summary_path = os.path.join(BASE_DIR, 'metrics_summary.csv')
        if os.path.exists(summary_path):
            old_summary = pd.read_csv(summary_path)
            combined = pd.concat([old_summary, new_summary], ignore_index=True)
            final_summary = combined.drop_duplicates(subset=['variant', 'h'], keep='last').reset_index(drop=True)
        else:
            final_summary = new_summary
        final_summary.to_csv(summary_path, index=False)
        print(f"\nResumen consolidado guardado en: {summary_path}")
        print("\n" + final_summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenamiento v3_base")
    parser.add_argument('--variant', type=str, default='remaining',
                        help="Variante(s) a correr ('all', 'remaining', o separadas por comas p.ej. 'A_chem,B_pp')")
    args = parser.parse_args()

    if args.variant == 'all':
        variants = ['C_honest', 'A_chem', 'B_pp', 'D_honest_noise', 'E_honest_w']
    elif args.variant == 'remaining':
        variants = ['A_chem', 'B_pp', 'D_honest_noise', 'E_honest_w']
    elif ',' in args.variant:
        variants = [v.strip() for v in args.variant.split(',')]
    else:
        variants = [args.variant]

    run_experiment(variants)
