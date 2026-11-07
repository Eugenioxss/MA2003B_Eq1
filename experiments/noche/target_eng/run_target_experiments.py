"""
experiments/noche/target_eng/run_target_experiments.py
======================================================
Ejecución sistemática de las variantes de formulación del target para LightGBM:
  1. direct_base (referencia V4_spatial + optuna params)
  2. resid_clim (residual de climatologia estacional/diurna 2020-2023)
  3. resid_persist (residual de persistencia contemporanea / diaria)
  4. transformaciones de target: log1p (con correccion de varianza), sqrt (con correccion),
     y tweedie (p=1.2, p=1.5)
  5. per_hour (features explicitas de la hora objetivo t+h, delta_clim e interacciones)
  6. multi_h_global (modelo unico global con todas las h apiladas y h como feature)

Cumple con:
  - num_threads=3
  - Seleccion de variantes SOLO con val 2024
  - Predicciones val + test guardadas en preds/<variante>_h<h>.parquet
  - log_result(experiment='noche_target_eng')
"""
import sys
import os
import time
import json
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXP_DIR = os.path.join(ROOT, 'experiments')
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from common import split, evaluate_test, log_result, HORIZONS, MODEL_STATIONS, climatology_pred, metrics
from noche.target_eng.data_prep import get_v4_data

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(CUR_DIR, 'preds')
os.makedirs(PREDS_DIR, exist_ok=True)
LOG_FILE = os.path.join(CUR_DIR, 'run.log')
METRICS_CSV = os.path.join(CUR_DIR, 'metrics_summary.csv')

# Setup logger
logger = logging.getLogger('target_eng')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
fh.setFormatter(formatter)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

def get_base_params():
    params_path = os.path.join(EXP_DIR, 'planA_quickwins', 'best_params_optuna.json')
    with open(params_path, 'r') as f:
        p = json.load(f)
    p.update({
        'objective': 'regression',
        'metric': 'rmse',
        'num_threads': 3,
        'seed': 42,
        'verbose': -1,
    })
    return p

def save_predictions(variant: str, h: int, va_df: pd.DataFrame, y_pred_val: np.ndarray,
                     ts_df: pd.DataFrame, y_pred_test: np.ndarray):
    """Guarda predicciones val+test en formato obligatorio."""
    target_col = f'Target_{h}h'
    df_val = pd.DataFrame({
        'Date': va_df['Date'].values,
        'Estacion': va_df['Estacion'].values,
        'h': h,
        'split': 'val',
        'y_true': va_df[target_col].values,
        'y_pred': y_pred_val
    })
    df_test = pd.DataFrame({
        'Date': ts_df['Date'].values,
        'Estacion': ts_df['Estacion'].values,
        'h': h,
        'split': 'test',
        'y_true': ts_df[target_col].values,
        'y_pred': y_pred_test
    })
    combined = pd.concat([df_val, df_test], ignore_index=True)
    out_path = os.path.join(PREDS_DIR, f"{variant}_h{h}.parquet")
    combined.to_parquet(out_path, index=False)
    logger.info(f"  -> Predicciones guardadas: {out_path} ({len(combined):,} filas)")

def clean_metrics_for_registry(m: dict) -> dict:
    """Extrae unicamente las claves estandar para no corromper el CSV del registry."""
    std_keys = ['n', 'coverage', 'RMSE', 'MAE', 'R2', 'bias', 'RMSE_peaks', 'bias_peaks', 'RMSE_day', 'RMSE_clim', 'skill_vs_clim']
    return {k: m[k] for k in std_keys if k in m}

def evaluate_val(val_df: pd.DataFrame, y_pred, h: int) -> dict:
    """Calcula metricas en conjunto de validacion 2024."""
    tt = val_df['Date'] + pd.Timedelta(hours=h)
    clim = climatology_pred(val_df['Estacion'], tt)
    return metrics(val_df[f'Target_{h}h'].values, y_pred, clim=clim, hours=tt.dt.hour.values)

def record_result(variant: str, h: int, val_m: dict, test_m: dict, notes: str, all_records: list):
    """Registra en CSV local y en registry global."""
    clean_m = clean_metrics_for_registry(test_m)
    log_result('noche_target_eng', variant, h, clean_m, notes=notes)
    
    row = {
        'variant': variant,
        'h': h,
        'val_RMSE': round(val_m['RMSE'], 4),
        'val_MAE': round(val_m['MAE'], 4),
        'val_R2': round(val_m['R2'], 4),
        'val_RMSE_peaks': round(val_m['RMSE_peaks'], 4),
        'val_bias_peaks': round(val_m['bias_peaks'], 4),
        'test_RMSE': round(test_m['RMSE'], 4),
        'test_MAE': round(test_m['MAE'], 4),
        'test_R2': round(test_m['R2'], 4),
        'test_RMSE_peaks': round(test_m['RMSE_peaks'], 4),
        'test_bias_peaks': round(test_m['bias_peaks'], 4),
        'test_skill': round(test_m.get('skill_vs_clim', 0.0), 4),
        'notes': notes
    }
    all_records.append(row)
    pd.DataFrame(all_records).to_csv(METRICS_CSV, index=False)
    logger.info(f"[{variant}] h={h:2d} | Val RMSE: {val_m['RMSE']:.4f} | Test RMSE: {test_m['RMSE']:.4f} | Test R2: {test_m['R2']:.4f} | Test Skill: {test_m.get('skill_vs_clim', 0.0):.4f}")


# ===========================================================================
# 1. DIRECT BASELINE
# ===========================================================================
def run_direct_base(df, feats, all_records):
    logger.info("\n" + "="*70)
    logger.info("  INICIANDO VARIANTE: direct_base")
    logger.info("="*70)
    tr_df, va_df, ts_df = split(df)
    params = get_base_params()

    for h in HORIZONS:
        target_col = f'Target_{h}h'
        tr_h = tr_df.dropna(subset=[target_col]).reset_index(drop=True)
        va_h = va_df.dropna(subset=[target_col]).reset_index(drop=True)
        ts_h = ts_df.dropna(subset=[target_col]).reset_index(drop=True)

        ds_tr = lgb.Dataset(tr_h[feats], tr_h[target_col], categorical_feature=['Estacion'])
        ds_va = lgb.Dataset(va_h[feats], va_h[target_col], reference=ds_tr, categorical_feature=['Estacion'])

        t0 = time.time()
        model = lgb.train(
            params,
            ds_tr,
            num_boost_round=1500,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(60, verbose=False)]
        )
        t_sec = round(time.time() - t0, 1)

        y_pred_val = np.maximum(0.0, model.predict(va_h[feats]))
        y_pred_test = np.maximum(0.0, model.predict(ts_h[feats]))

        val_m = evaluate_val(va_h, y_pred_val, h)
        test_m = evaluate_test(ts_h, y_pred_test, h)

        save_predictions('direct_base', h, va_h, y_pred_val, ts_h, y_pred_test)
        record_result('direct_base', h, val_m, test_m, f"iters={model.best_iteration}, time={t_sec}s", all_records)


# ===========================================================================
# 2. RESIDUAL CLIMATOLOGY (resid_clim)
# ===========================================================================
def run_resid_clim(df, feats, all_records):
    logger.info("\n" + "="*70)
    logger.info("  INICIANDO VARIANTE: resid_clim")
    logger.info("="*70)
    tr_df, va_df, ts_df = split(df)
    params = get_base_params()

    for h in HORIZONS:
        target_col = f'Target_{h}h'
        tr_h = tr_df.dropna(subset=[target_col]).reset_index(drop=True)
        va_h = va_df.dropna(subset=[target_col]).reset_index(drop=True)
        ts_h = ts_df.dropna(subset=[target_col]).reset_index(drop=True)

        # Precomputar climatologia en t+h
        c_tr = climatology_pred(tr_h['Estacion'], tr_h['Date'] + pd.Timedelta(hours=h))
        c_va = climatology_pred(va_h['Estacion'], va_h['Date'] + pd.Timedelta(hours=h))
        c_ts = climatology_pred(ts_h['Estacion'], ts_h['Date'] + pd.Timedelta(hours=h))

        y_tr_res = tr_h[target_col].values - c_tr
        y_va_res = va_h[target_col].values - c_va

        ds_tr = lgb.Dataset(tr_h[feats], y_tr_res, categorical_feature=['Estacion'])
        ds_va = lgb.Dataset(va_h[feats], y_va_res, reference=ds_tr, categorical_feature=['Estacion'])

        t0 = time.time()
        model = lgb.train(
            params,
            ds_tr,
            num_boost_round=1500,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(60, verbose=False)]
        )
        t_sec = round(time.time() - t0, 1)

        r_pred_val = model.predict(va_h[feats])
        r_pred_test = model.predict(ts_h[feats])

        y_pred_val = np.maximum(0.0, c_va + r_pred_val)
        y_pred_test = np.maximum(0.0, c_ts + r_pred_test)

        val_m = evaluate_val(va_h, y_pred_val, h)
        test_m = evaluate_test(ts_h, y_pred_test, h)

        save_predictions('resid_clim', h, va_h, y_pred_val, ts_h, y_pred_test)
        record_result('resid_clim', h, val_m, test_m, f"iters={model.best_iteration}, time={t_sec}s", all_records)


# ===========================================================================
# 3. RESIDUAL PERSISTENCE (resid_persist)
# ===========================================================================
def run_resid_persist(df, feats, all_records):
    logger.info("\n" + "="*70)
    logger.info("  INICIANDO VARIANTE: resid_persist")
    logger.info("="*70)
    tr_df, va_df, ts_df = split(df)
    params = get_base_params()

    for h in HORIZONS:
        target_col = f'Target_{h}h'
        tr_h = tr_df.dropna(subset=[target_col]).reset_index(drop=True)
        va_h = va_df.dropna(subset=[target_col]).reset_index(drop=True)
        ts_h = ts_df.dropna(subset=[target_col]).reset_index(drop=True)

        # Seleccion de persistencia honesta:
        # Para h=1, 4, 8: O3(t) es lag 0 contemporaneo.
        # Para h=24: O3(t) es simultaneamente contemporaneo y same-hour diario (t+24-24 = t).
        # Para h=48: O3(t) es same-hour de hace 2 dias (t+48-48 = t).
        # Para h=12: O3(t) vs O3(t-12). O3(t) suele ser muy fuerte con V4 features; evaluamos O3(t).
        def get_p(sub_df):
            p = sub_df['O3'].to_numpy(copy=True, dtype=np.float64)
            # Fallback a O3_lag_1 si O3(t) es NaN, luego a climatologia
            nan_mask = ~np.isfinite(p)
            if nan_mask.any():
                lag1 = sub_df['O3_lag_1'].to_numpy(copy=True, dtype=np.float64)
                p[nan_mask] = lag1[nan_mask]
                nan_mask2 = ~np.isfinite(p)
                if nan_mask2.any():
                    c = climatology_pred(sub_df['Estacion'], sub_df['Date'])
                    p[nan_mask2] = c[nan_mask2]
            return p

        p_tr = get_p(tr_h)
        p_va = get_p(va_h)
        p_ts = get_p(ts_h)

        y_tr_diff = tr_h[target_col].values - p_tr
        y_va_diff = va_h[target_col].values - p_va

        ds_tr = lgb.Dataset(tr_h[feats], y_tr_diff, categorical_feature=['Estacion'])
        ds_va = lgb.Dataset(va_h[feats], y_va_diff, reference=ds_tr, categorical_feature=['Estacion'])

        t0 = time.time()
        model = lgb.train(
            params,
            ds_tr,
            num_boost_round=1500,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(60, verbose=False)]
        )
        t_sec = round(time.time() - t0, 1)

        diff_pred_val = model.predict(va_h[feats])
        diff_pred_test = model.predict(ts_h[feats])

        y_pred_val = np.maximum(0.0, p_va + diff_pred_val)
        y_pred_test = np.maximum(0.0, p_ts + diff_pred_test)

        val_m = evaluate_val(va_h, y_pred_val, h)
        test_m = evaluate_test(ts_h, y_pred_test, h)

        save_predictions('resid_persist', h, va_h, y_pred_val, ts_h, y_pred_test)
        record_result('resid_persist', h, val_m, test_m, f"iters={model.best_iteration}, time={t_sec}s", all_records)


# ===========================================================================
# 4. TARGET TRANSFORMS (sqrt, log1p, tweedie)
# ===========================================================================
def run_target_transforms(df, feats, all_records):
    logger.info("\n" + "="*70)
    logger.info("  INICIANDO VARIANTE: transformaciones de target (sqrt, log1p, tweedie)")
    logger.info("="*70)
    tr_df, va_df, ts_df = split(df)

    # Probamos las 4 transformaciones en horizontes clave {1, 12, 24} para seleccionar la mejor con val 2024
    transforms = ['sqrt', 'log1p', 'tweedie_p12', 'tweedie_p15']
    val_scores = {t: [] for t in transforms}

    logger.info("Screening de transformaciones en val 2024 (h in [1, 12, 24])...")
    for h in [1, 12, 24]:
        target_col = f'Target_{h}h'
        tr_h = tr_df.dropna(subset=[target_col]).reset_index(drop=True)
        va_h = va_df.dropna(subset=[target_col]).reset_index(drop=True)

        for trans in transforms:
            params = get_base_params()
            if trans == 'sqrt':
                y_tr = np.sqrt(np.maximum(0.0, tr_h[target_col].values))
                y_va = np.sqrt(np.maximum(0.0, va_h[target_col].values))
                ds_tr = lgb.Dataset(tr_h[feats], y_tr, categorical_feature=['Estacion'])
                ds_va = lgb.Dataset(va_h[feats], y_va, reference=ds_tr, categorical_feature=['Estacion'])
                model = lgb.train(params, ds_tr, num_boost_round=1200, valid_sets=[ds_va], callbacks=[lgb.early_stopping(60, verbose=False)])
                z_tr_pred = model.predict(tr_h[feats])
                sigma2 = np.var(y_tr - z_tr_pred)
                z_va_pred = np.maximum(0.0, model.predict(va_h[feats]))
                y_va_pred = np.maximum(0.0, z_va_pred**2 + sigma2)

            elif trans == 'log1p':
                y_tr = np.log1p(np.maximum(0.0, tr_h[target_col].values))
                y_va = np.log1p(np.maximum(0.0, va_h[target_col].values))
                ds_tr = lgb.Dataset(tr_h[feats], y_tr, categorical_feature=['Estacion'])
                ds_va = lgb.Dataset(va_h[feats], y_va, reference=ds_tr, categorical_feature=['Estacion'])
                model = lgb.train(params, ds_tr, num_boost_round=1200, valid_sets=[ds_va], callbacks=[lgb.early_stopping(60, verbose=False)])
                z_tr_pred = model.predict(tr_h[feats])
                sigma2 = np.var(y_tr - z_tr_pred)
                z_va_pred = model.predict(va_h[feats])
                y_va_pred = np.maximum(0.0, np.expm1(z_va_pred + 0.5 * sigma2))

            elif trans == 'tweedie_p12':
                params.update({'objective': 'tweedie', 'tweedie_variance_power': 1.2})
                y_tr = np.maximum(0.0, tr_h[target_col].values)
                y_va = np.maximum(0.0, va_h[target_col].values)
                ds_tr = lgb.Dataset(tr_h[feats], y_tr, categorical_feature=['Estacion'])
                ds_va = lgb.Dataset(va_h[feats], y_va, reference=ds_tr, categorical_feature=['Estacion'])
                model = lgb.train(params, ds_tr, num_boost_round=1200, valid_sets=[ds_va], callbacks=[lgb.early_stopping(60, verbose=False)])
                y_va_pred = np.maximum(0.0, model.predict(va_h[feats]))

            elif trans == 'tweedie_p15':
                params.update({'objective': 'tweedie', 'tweedie_variance_power': 1.5})
                y_tr = np.maximum(0.0, tr_h[target_col].values)
                y_va = np.maximum(0.0, va_h[target_col].values)
                ds_tr = lgb.Dataset(tr_h[feats], y_tr, categorical_feature=['Estacion'])
                ds_va = lgb.Dataset(va_h[feats], y_va, reference=ds_tr, categorical_feature=['Estacion'])
                model = lgb.train(params, ds_tr, num_boost_round=1200, valid_sets=[ds_va], callbacks=[lgb.early_stopping(60, verbose=False)])
                y_va_pred = np.maximum(0.0, model.predict(va_h[feats]))

            vm = evaluate_val(va_h, y_va_pred, h)
            val_scores[trans].append(vm['RMSE'])
            logger.info(f"  [Screening] {trans:12s} h={h:2d} | Val RMSE = {vm['RMSE']:.4f}")

    # Promedio en validacion para elegir el mejor transform
    mean_val = {t: np.mean(val_scores[t]) for t in transforms}
    best_trans = min(mean_val, key=mean_val.get)
    logger.info(f"==> Ganador de transformacion en Val 2024: '{best_trans}' (RMSE prom: {mean_val[best_trans]:.4f}) vs {mean_val}")

    # Ejecutar todos los 6 horizontes para el mejor transform
    logger.info(f"\nEjecutando '{best_trans}' en los 6 horizontes completos...")
    for h in HORIZONS:
        target_col = f'Target_{h}h'
        tr_h = tr_df.dropna(subset=[target_col]).reset_index(drop=True)
        va_h = va_df.dropna(subset=[target_col]).reset_index(drop=True)
        ts_h = ts_df.dropna(subset=[target_col]).reset_index(drop=True)

        params = get_base_params()
        t0 = time.time()
        if best_trans == 'sqrt':
            y_tr = np.sqrt(np.maximum(0.0, tr_h[target_col].values))
            y_va = np.sqrt(np.maximum(0.0, va_h[target_col].values))
            ds_tr = lgb.Dataset(tr_h[feats], y_tr, categorical_feature=['Estacion'])
            ds_va = lgb.Dataset(va_h[feats], y_va, reference=ds_tr, categorical_feature=['Estacion'])
            model = lgb.train(params, ds_tr, num_boost_round=1500, valid_sets=[ds_va], callbacks=[lgb.early_stopping(60, verbose=False)])
            sigma2 = np.var(y_tr - model.predict(tr_h[feats]))
            y_pred_val = np.maximum(0.0, np.maximum(0.0, model.predict(va_h[feats]))**2 + sigma2)
            y_pred_test = np.maximum(0.0, np.maximum(0.0, model.predict(ts_h[feats]))**2 + sigma2)

        elif best_trans == 'log1p':
            y_tr = np.log1p(np.maximum(0.0, tr_h[target_col].values))
            y_va = np.log1p(np.maximum(0.0, va_h[target_col].values))
            ds_tr = lgb.Dataset(tr_h[feats], y_tr, categorical_feature=['Estacion'])
            ds_va = lgb.Dataset(va_h[feats], y_va, reference=ds_tr, categorical_feature=['Estacion'])
            model = lgb.train(params, ds_tr, num_boost_round=1500, valid_sets=[ds_va], callbacks=[lgb.early_stopping(60, verbose=False)])
            sigma2 = np.var(y_tr - model.predict(tr_h[feats]))
            y_pred_val = np.maximum(0.0, np.expm1(model.predict(va_h[feats]) + 0.5 * sigma2))
            y_pred_test = np.maximum(0.0, np.expm1(model.predict(ts_h[feats]) + 0.5 * sigma2))

        else: # tweedie
            vp = 1.2 if best_trans == 'tweedie_p12' else 1.5
            params.update({'objective': 'tweedie', 'tweedie_variance_power': vp})
            y_tr = np.maximum(0.0, tr_h[target_col].values)
            y_va = np.maximum(0.0, va_h[target_col].values)
            ds_tr = lgb.Dataset(tr_h[feats], y_tr, categorical_feature=['Estacion'])
            ds_va = lgb.Dataset(va_h[feats], y_va, reference=ds_tr, categorical_feature=['Estacion'])
            model = lgb.train(params, ds_tr, num_boost_round=1500, valid_sets=[ds_va], callbacks=[lgb.early_stopping(60, verbose=False)])
            y_pred_val = np.maximum(0.0, model.predict(va_h[feats]))
            y_pred_test = np.maximum(0.0, model.predict(ts_h[feats]))

        t_sec = round(time.time() - t0, 1)
        val_m = evaluate_val(va_h, y_pred_val, h)
        test_m = evaluate_test(ts_h, y_pred_test, h)

        save_predictions(best_trans, h, va_h, y_pred_val, ts_h, y_pred_test)
        record_result(best_trans, h, val_m, test_m, f"best_transform_by_val, iters={model.best_iteration}, time={t_sec}s", all_records)


# ===========================================================================
# 5. PER HOUR (features explicitas de hora objetivo t+h + interaccion)
# ===========================================================================
def build_per_hour_features(df_sub, h, base_feats):
    """Agrega features explicitas de la hora objetivo t+h y delta climatologico."""
    df_out = df_sub.copy()
    target_dt = df_out['Date'] + pd.Timedelta(hours=h)
    target_hr = target_dt.dt.hour
    
    df_out['target_hour'] = target_hr.astype('float32')
    df_out['target_hour_sin'] = np.sin(2 * np.pi * target_hr / 24).astype('float32')
    df_out['target_hour_cos'] = np.cos(2 * np.pi * target_hr / 24).astype('float32')
    
    clim_target = climatology_pred(df_out['Estacion'], target_dt)
    clim_now = climatology_pred(df_out['Estacion'], df_out['Date'])
    df_out['target_clim'] = clim_target.astype('float32')
    df_out['delta_clim'] = (clim_target - clim_now).astype('float32')
    
    df_out['is_target_daylight'] = ((target_hr >= 10) & (target_hr <= 18)).astype('float32')
    
    # Interacciones diurnas fotocataliticas
    o3_now = df_out['O3'].fillna(df_out['O3_lag_1']).fillna(df_out['target_clim'])
    df_out['O3_x_target_sin'] = (o3_now * df_out['target_hour_sin']).astype('float32')
    df_out['O3_x_target_cos'] = (o3_now * df_out['target_hour_cos']).astype('float32')
    df_out['SR_x_target_sin'] = (df_out['SR_lag_0'].fillna(0) * df_out['target_hour_sin']).astype('float32')
    df_out['O3_x_delta_clim'] = (o3_now * df_out['delta_clim'] / 50.0).astype('float32')
    
    extra_cols = [
        'target_hour', 'target_hour_sin', 'target_hour_cos', 'target_clim', 'delta_clim',
        'is_target_daylight', 'O3_x_target_sin', 'O3_x_target_cos', 'SR_x_target_sin', 'O3_x_delta_clim'
    ]
    return df_out, base_feats + extra_cols

def run_per_hour(df, feats, all_records):
    logger.info("\n" + "="*70)
    logger.info("  INICIANDO VARIANTE: per_hour (features de t+h explicitas + delta_clim)")
    logger.info("="*70)
    tr_df, va_df, ts_df = split(df)
    params = get_base_params()

    for h in HORIZONS:
        target_col = f'Target_{h}h'
        tr_h = tr_df.dropna(subset=[target_col]).reset_index(drop=True)
        va_h = va_df.dropna(subset=[target_col]).reset_index(drop=True)
        ts_h = ts_df.dropna(subset=[target_col]).reset_index(drop=True)

        tr_h_ph, feats_ph = build_per_hour_features(tr_h, h, feats)
        va_h_ph, _ = build_per_hour_features(va_h, h, feats)
        ts_h_ph, _ = build_per_hour_features(ts_h, h, feats)

        ds_tr = lgb.Dataset(tr_h_ph[feats_ph], tr_h_ph[target_col], categorical_feature=['Estacion'])
        ds_va = lgb.Dataset(va_h_ph[feats_ph], va_h_ph[target_col], reference=ds_tr, categorical_feature=['Estacion'])

        t0 = time.time()
        model = lgb.train(
            params,
            ds_tr,
            num_boost_round=1500,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(60, verbose=False)]
        )
        t_sec = round(time.time() - t0, 1)

        y_pred_val = np.maximum(0.0, model.predict(va_h_ph[feats_ph]))
        y_pred_test = np.maximum(0.0, model.predict(ts_h_ph[feats_ph]))

        val_m = evaluate_val(va_h, y_pred_val, h)
        test_m = evaluate_test(ts_h, y_pred_test, h)

        save_predictions('per_hour', h, va_h, y_pred_val, ts_h, y_pred_test)
        record_result('per_hour', h, val_m, test_m, f"iters={model.best_iteration}, n_feat={len(feats_ph)}, time={t_sec}s", all_records)


# ===========================================================================
# 6. MULTI-HORIZON GLOBAL MODEL (multi_h_global)
# ===========================================================================
def run_multi_h_global(df, feats, all_records):
    logger.info("\n" + "="*70)
    logger.info("  INICIANDO VARIANTE: multi_h_global (un solo modelo con todas las h)")
    logger.info("="*70)
    tr_df, va_df, ts_df = split(df)
    params = get_base_params()

    # Construir datasets apilados
    logger.info("Construyendo datasets apilados con h in [1, 4, 8, 12, 24, 48]...")
    tr_list = []
    # Submuestreo estrided en train para no saturar memoria/tiempo (cada 2 horas)
    tr_sub = tr_df.iloc[::2].copy()
    for h in HORIZONS:
        target_col = f'Target_{h}h'
        sub = tr_sub.dropna(subset=[target_col]).copy()
        target_dt = sub['Date'] + pd.Timedelta(hours=h)
        target_hr = target_dt.dt.hour
        sub['horizon'] = float(h)
        sub['log_horizon'] = float(np.log(h))
        sub['target_hour'] = target_hr.astype('float32')
        sub['target_hour_sin'] = np.sin(2 * np.pi * target_hr / 24).astype('float32')
        sub['target_hour_cos'] = np.cos(2 * np.pi * target_hr / 24).astype('float32')
        clim_target = climatology_pred(sub['Estacion'], target_dt)
        clim_now = climatology_pred(sub['Estacion'], sub['Date'])
        sub['target_clim'] = clim_target.astype('float32')
        sub['delta_clim'] = (clim_target - clim_now).astype('float32')
        sub['global_target'] = sub[target_col].values
        tr_list.append(sub)

    stacked_tr = pd.concat(tr_list, ignore_index=True)

    va_list = []
    for h in HORIZONS:
        target_col = f'Target_{h}h'
        sub = va_df.dropna(subset=[target_col]).copy()
        target_dt = sub['Date'] + pd.Timedelta(hours=h)
        target_hr = target_dt.dt.hour
        sub['horizon'] = float(h)
        sub['log_horizon'] = float(np.log(h))
        sub['target_hour'] = target_hr.astype('float32')
        sub['target_hour_sin'] = np.sin(2 * np.pi * target_hr / 24).astype('float32')
        sub['target_hour_cos'] = np.cos(2 * np.pi * target_hr / 24).astype('float32')
        clim_target = climatology_pred(sub['Estacion'], target_dt)
        clim_now = climatology_pred(sub['Estacion'], sub['Date'])
        sub['target_clim'] = clim_target.astype('float32')
        sub['delta_clim'] = (clim_target - clim_now).astype('float32')
        sub['global_target'] = sub[target_col].values
        va_list.append(sub)

    stacked_va = pd.concat(va_list, ignore_index=True)

    global_feats = feats + ['horizon', 'log_horizon', 'target_hour', 'target_hour_sin', 'target_hour_cos', 'target_clim', 'delta_clim']
    logger.info(f"Dataset global creado: Train = {len(stacked_tr):,} filas, Val = {len(stacked_va):,} filas, {len(global_feats)} features.")

    ds_tr = lgb.Dataset(stacked_tr[global_feats], stacked_tr['global_target'], categorical_feature=['Estacion'])
    ds_va = lgb.Dataset(stacked_va[global_feats], stacked_va['global_target'], reference=ds_tr, categorical_feature=['Estacion'])

    t0 = time.time()
    logger.info("Entrenando modelo global unico...")
    model = lgb.train(
        params,
        ds_tr,
        num_boost_round=1500,
        valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(60, verbose=False)]
    )
    t_sec = round(time.time() - t0, 1)
    logger.info(f"Modelo global entrenado en {t_sec}s (best_iter={model.best_iteration}).")

    # Evaluar por cada horizonte h
    for h in HORIZONS:
        target_col = f'Target_{h}h'
        va_h = va_df.dropna(subset=[target_col]).copy()
        ts_h = ts_df.dropna(subset=[target_col]).copy()

        # Construir features para val y test con este h
        for sub in [va_h, ts_h]:
            target_dt = sub['Date'] + pd.Timedelta(hours=h)
            target_hr = target_dt.dt.hour
            sub['horizon'] = float(h)
            sub['log_horizon'] = float(np.log(h))
            sub['target_hour'] = target_hr.astype('float32')
            sub['target_hour_sin'] = np.sin(2 * np.pi * target_hr / 24).astype('float32')
            sub['target_hour_cos'] = np.cos(2 * np.pi * target_hr / 24).astype('float32')
            clim_target = climatology_pred(sub['Estacion'], target_dt)
            clim_now = climatology_pred(sub['Estacion'], sub['Date'])
            sub['target_clim'] = clim_target.astype('float32')
            sub['delta_clim'] = (clim_target - clim_now).astype('float32')

        y_pred_val = np.maximum(0.0, model.predict(va_h[global_feats]))
        y_pred_test = np.maximum(0.0, model.predict(ts_h[global_feats]))

        val_m = evaluate_val(va_h, y_pred_val, h)
        test_m = evaluate_test(ts_h, y_pred_test, h)

        save_predictions('multi_h_global', h, va_h, y_pred_val, ts_h, y_pred_test)
        record_result('multi_h_global', h, val_m, test_m, f"single_global_model, iters={model.best_iteration}, time={t_sec}s", all_records)


def main():
    logger.info("=====================================================================")
    logger.info("  NOCHE TARGET ENGINEERING - CORRIDA COMPLETA DE VARIANTES (1 - 5)")
    logger.info("=====================================================================")
    t_global = time.time()

    # Cargar datos
    df, feats = get_v4_data(use_cache=True)
    all_records = []
    if os.path.exists(METRICS_CSV):
        all_records = pd.read_csv(METRICS_CSV).to_dict(orient='records')

    completed_variants = {r['variant'] for r in all_records if r.get('h') == 48}
    logger.info(f"Variantes ya completadas (h=48): {completed_variants}")

    # 1. Direct baseline
    if 'direct_base' not in completed_variants:
        run_direct_base(df, feats, all_records)
    else:
        logger.info("direct_base ya completado con éxito, continuando con la siguiente variante.")

    # 2. Residual climatology
    if 'resid_clim' not in completed_variants:
        run_resid_clim(df, feats, all_records)
    else:
        logger.info("resid_clim ya completado con éxito, continuando con la siguiente variante.")

    # 3. Residual persistence
    run_resid_persist(df, feats, all_records)

    # 4. Target transformations & tweedie
    run_target_transforms(df, feats, all_records)

    # 5. Per-hour explicit features
    run_per_hour(df, feats, all_records)

    # 6. Multi-h global model
    run_multi_h_global(df, feats, all_records)

    logger.info("\n" + "="*70)
    logger.info(f"  TODAS LAS VARIANTES 1-5 FINALIZADAS CON EXITO EN {time.time()-t_global:.1f}s")
    logger.info("=====================================================================")

if __name__ == '__main__':
    main()
