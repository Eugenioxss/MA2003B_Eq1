"""
experiments/noche/target_eng/run_honest_meteo.py
================================================
Fase 2: Repeticion de las mejores formulaciones de target utilizando
el cache de features v3 con meteorologia honesta (PP-train / NWP-test)
proveniente de experiments/noche/v3_base/cache/.

Registra con sufijo *_honest en results_registry.csv (experiment='noche_target_eng')
y guarda predicciones en preds/<variante>_honest_h<h>.parquet.
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

from common import evaluate_test, log_result, HORIZONS, climatology_pred, metrics

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(CUR_DIR, 'preds')
CACHE_V3_DIR = os.path.join(EXP_DIR, 'noche', 'v3_base', 'cache')
LOG_FILE = os.path.join(CUR_DIR, 'run_honest.log')
METRICS_CSV = os.path.join(CUR_DIR, 'metrics_honest_summary.csv')

logger = logging.getLogger('honest_meteo')
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

def clean_metrics_for_registry(m: dict) -> dict:
    std_keys = ['n', 'coverage', 'RMSE', 'MAE', 'R2', 'bias', 'RMSE_peaks', 'bias_peaks', 'RMSE_day', 'RMSE_clim', 'skill_vs_clim']
    return {k: m[k] for k in std_keys if k in m}

def evaluate_val(val_df: pd.DataFrame, y_pred, h: int) -> dict:
    tt = val_df['Date'] + pd.Timedelta(hours=h)
    clim = climatology_pred(val_df['Estacion'], tt)
    return metrics(val_df['Target'].values, y_pred, clim=clim, hours=tt.dt.hour.values)

def save_predictions_honest(variant: str, h: int, va_df: pd.DataFrame, y_pred_val: np.ndarray,
                            ts_df: pd.DataFrame, y_pred_test: np.ndarray):
    df_val = pd.DataFrame({
        'Date': va_df['Date'].values,
        'Estacion': va_df['Estacion'].values,
        'h': h,
        'split': 'val',
        'y_true': va_df['Target'].values,
        'y_pred': y_pred_val
    })
    df_test = pd.DataFrame({
        'Date': ts_df['Date'].values,
        'Estacion': ts_df['Estacion'].values,
        'h': h,
        'split': 'test',
        'y_true': ts_df['Target'].values,
        'y_pred': y_pred_test
    })
    combined = pd.concat([df_val, df_test], ignore_index=True)
    out_path = os.path.join(PREDS_DIR, f"{variant}_h{h}.parquet")
    combined.to_parquet(out_path, index=False)
    logger.info(f"  -> Predicciones guardadas: {out_path} ({len(combined):,} filas)")

def run_honest_experiments():
    logger.info("=====================================================================")
    logger.info("  FASE 2: CORRIDA CON METEOROLOGIA HONESTA (v3_base cache)")
    logger.info("=====================================================================")
    
    params = get_base_params()
    records = []
    
    # Probamos direct_honest, resid_clim_honest, y per_hour_honest
    variants_to_run = ['direct_base_honest', 'resid_clim_honest', 'per_hour_honest']
    
    for h in HORIZONS:
        cache_file = os.path.join(CACHE_V3_DIR, f"features_h{h}.parquet")
        logger.info(f"\nCargando cache v3_base para h={h} ({cache_file})...")
        t0 = time.time()
        df = pd.read_parquet(cache_file)
        df['Estacion'] = df['Estacion'].astype('category')
        
        # Extraer features honestas
        chem_features = [c for c in df.columns if not c.startswith(('meteo_t0_', 'honest_meteo_', 'pp_meteo_')) and c not in ['Date', 'split', 'Target', f'Target_{h}h']]
        meteo_t0 = [c for c in df.columns if c.startswith('meteo_t0_')]
        honest_meteo = [c for c in df.columns if c.startswith('honest_meteo_')]
        features_honest = chem_features + meteo_t0 + honest_meteo
        logger.info(f"Cache cargado en {time.time()-t0:.2f}s | {len(features_honest)} features honestas.")
        
        # Splits
        tr_h = df[(df['split'] == 'train') & df['Target'].notnull()].reset_index(drop=True)
        va_h = df[(df['split'] == 'val') & df['Target'].notnull()].reset_index(drop=True)
        ts_h = df[(df['split'] == 'test') & df['Target'].notnull()].reset_index(drop=True)
        
        # 1. direct_base_honest
        logger.info(f"--- [h={h}] direct_base_honest ---")
        t_tr = time.time()
        ds_tr = lgb.Dataset(tr_h[features_honest], tr_h['Target'], categorical_feature=['Estacion'])
        ds_va = lgb.Dataset(va_h[features_honest], va_h['Target'], reference=ds_tr, categorical_feature=['Estacion'])
        m_direct = lgb.train(params, ds_tr, num_boost_round=1500, valid_sets=[ds_va], callbacks=[lgb.early_stopping(60, verbose=False)])
        yp_va_dir = np.maximum(0.0, m_direct.predict(va_h[features_honest]))
        yp_ts_dir = np.maximum(0.0, m_direct.predict(ts_h[features_honest]))
        val_m_dir = evaluate_val(va_h, yp_va_dir, h)
        test_m_dir = evaluate_test(ts_h, yp_ts_dir, h)
        save_predictions_honest('direct_base_honest', h, va_h, yp_va_dir, ts_h, yp_ts_dir)
        log_result('noche_target_eng', 'direct_base_honest', h, clean_metrics_for_registry(test_m_dir), notes=f"honest_meteo, iters={m_direct.best_iteration}")
        records.append({'variant': 'direct_base_honest', 'h': h, 'val_RMSE': val_m_dir['RMSE'], 'test_RMSE': test_m_dir['RMSE'], 'test_skill': test_m_dir.get('skill_vs_clim', 0.0), 'iters': m_direct.best_iteration})
        logger.info(f"  [direct_base_honest] h={h:2d} | Val RMSE: {val_m_dir['RMSE']:.4f} | Test RMSE: {test_m_dir['RMSE']:.4f} | Test Skill: {test_m_dir.get('skill_vs_clim', 0.0):.4f}")

        # 2. resid_persist_honest
        logger.info(f"--- [h={h}] resid_persist_honest ---")
        def get_p(sub_df):
            p = sub_df['O3_lag_0'].to_numpy(copy=True, dtype=np.float64)
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
        y_tr_diff = tr_h['Target'].values - p_tr
        y_va_diff = va_h['Target'].values - p_va
        ds_tr_p = lgb.Dataset(tr_h[features_honest], y_tr_diff, categorical_feature=['Estacion'])
        ds_va_p = lgb.Dataset(va_h[features_honest], y_va_diff, reference=ds_tr_p, categorical_feature=['Estacion'])
        m_persist = lgb.train(params, ds_tr_p, num_boost_round=1500, valid_sets=[ds_va_p], callbacks=[lgb.early_stopping(60, verbose=False)])
        yp_va_p = np.maximum(0.0, p_va + m_persist.predict(va_h[features_honest]))
        yp_ts_p = np.maximum(0.0, p_ts + m_persist.predict(ts_h[features_honest]))
        val_m_p = evaluate_val(va_h, yp_va_p, h)
        test_m_p = evaluate_test(ts_h, yp_ts_p, h)
        save_predictions_honest('resid_persist_honest', h, va_h, yp_va_p, ts_h, yp_ts_p)
        log_result('noche_target_eng', 'resid_persist_honest', h, clean_metrics_for_registry(test_m_p), notes=f"honest_meteo, iters={m_persist.best_iteration}")
        records.append({'variant': 'resid_persist_honest', 'h': h, 'val_RMSE': val_m_p['RMSE'], 'test_RMSE': test_m_p['RMSE'], 'test_skill': test_m_p.get('skill_vs_clim', 0.0), 'iters': m_persist.best_iteration})
        logger.info(f"  [resid_persist_honest] h={h:2d} | Val RMSE: {val_m_p['RMSE']:.4f} | Test RMSE: {test_m_p['RMSE']:.4f} | Test Skill: {test_m_p.get('skill_vs_clim', 0.0):.4f}")

        # 3. resid_clim_honest
        logger.info(f"--- [h={h}] resid_clim_honest ---")
        c_tr = climatology_pred(tr_h['Estacion'], tr_h['Date'] + pd.Timedelta(hours=h))
        c_va = climatology_pred(va_h['Estacion'], va_h['Date'] + pd.Timedelta(hours=h))
        c_ts = climatology_pred(ts_h['Estacion'], ts_h['Date'] + pd.Timedelta(hours=h))
        y_tr_res = tr_h['Target'].values - c_tr
        y_va_res = va_h['Target'].values - c_va
        ds_tr_c = lgb.Dataset(tr_h[features_honest], y_tr_res, categorical_feature=['Estacion'])
        ds_va_c = lgb.Dataset(va_h[features_honest], y_va_res, reference=ds_tr_c, categorical_feature=['Estacion'])
        m_clim = lgb.train(params, ds_tr_c, num_boost_round=1500, valid_sets=[ds_va_c], callbacks=[lgb.early_stopping(60, verbose=False)])
        yp_va_clim = np.maximum(0.0, c_va + m_clim.predict(va_h[features_honest]))
        yp_ts_clim = np.maximum(0.0, c_ts + m_clim.predict(ts_h[features_honest]))
        val_m_clim = evaluate_val(va_h, yp_va_clim, h)
        test_m_clim = evaluate_test(ts_h, yp_ts_clim, h)
        save_predictions_honest('resid_clim_honest', h, va_h, yp_va_clim, ts_h, yp_ts_clim)
        log_result('noche_target_eng', 'resid_clim_honest', h, clean_metrics_for_registry(test_m_clim), notes=f"honest_meteo, iters={m_clim.best_iteration}")
        records.append({'variant': 'resid_clim_honest', 'h': h, 'val_RMSE': val_m_clim['RMSE'], 'test_RMSE': test_m_clim['RMSE'], 'test_skill': test_m_clim.get('skill_vs_clim', 0.0), 'iters': m_clim.best_iteration})
        logger.info(f"  [resid_clim_honest] h={h:2d} | Val RMSE: {val_m_clim['RMSE']:.4f} | Test RMSE: {test_m_clim['RMSE']:.4f} | Test Skill: {test_m_clim.get('skill_vs_clim', 0.0):.4f}")

        # 3. per_hour_honest
        logger.info(f"--- [h={h}] per_hour_honest ---")
        def add_ph(sub_df):
            sub = sub_df.copy()
            target_dt = sub['Date'] + pd.Timedelta(hours=h)
            target_hr = target_dt.dt.hour
            sub['target_hour'] = target_hr.astype('float32')
            sub['target_hour_sin'] = np.sin(2 * np.pi * target_hr / 24).astype('float32')
            sub['target_hour_cos'] = np.cos(2 * np.pi * target_hr / 24).astype('float32')
            clim_tgt = climatology_pred(sub['Estacion'], target_dt)
            clim_cur = climatology_pred(sub['Estacion'], sub['Date'])
            sub['target_clim'] = clim_tgt.astype('float32')
            sub['delta_clim'] = (clim_tgt - clim_cur).astype('float32')
            sub['is_target_daylight'] = ((target_hr >= 10) & (target_hr <= 18)).astype('float32')
            o3_now = sub['O3_lag_0'].fillna(sub['O3_lag_1']).fillna(sub['target_clim'])
            sub['O3_x_target_sin'] = (o3_now * sub['target_hour_sin']).astype('float32')
            sub['O3_x_target_cos'] = (o3_now * sub['target_hour_cos']).astype('float32')
            sub['O3_x_delta_clim'] = (o3_now * sub['delta_clim'] / 50.0).astype('float32')
            return sub

        ph_extra = ['target_hour', 'target_hour_sin', 'target_hour_cos', 'target_clim', 'delta_clim', 'is_target_daylight', 'O3_x_target_sin', 'O3_x_target_cos', 'O3_x_delta_clim']
        tr_ph = add_ph(tr_h)
        va_ph = add_ph(va_h)
        ts_ph = add_ph(ts_h)
        feats_ph = features_honest + ph_extra

        ds_tr_ph = lgb.Dataset(tr_ph[feats_ph], tr_ph['Target'], categorical_feature=['Estacion'])
        ds_va_ph = lgb.Dataset(va_ph[feats_ph], va_ph['Target'], reference=ds_tr_ph, categorical_feature=['Estacion'])
        m_ph = lgb.train(params, ds_tr_ph, num_boost_round=1500, valid_sets=[ds_va_ph], callbacks=[lgb.early_stopping(60, verbose=False)])
        yp_va_ph = np.maximum(0.0, m_ph.predict(va_ph[feats_ph]))
        yp_ts_ph = np.maximum(0.0, m_ph.predict(ts_ph[feats_ph]))
        val_m_ph = evaluate_val(va_ph, yp_va_ph, h)
        test_m_ph = evaluate_test(ts_ph, yp_ts_ph, h)
        save_predictions_honest('per_hour_honest', h, va_ph, yp_va_ph, ts_ph, yp_ts_ph)
        log_result('noche_target_eng', 'per_hour_honest', h, clean_metrics_for_registry(test_m_ph), notes=f"honest_meteo, iters={m_ph.best_iteration}")
        records.append({'variant': 'per_hour_honest', 'h': h, 'val_RMSE': val_m_ph['RMSE'], 'test_RMSE': test_m_ph['RMSE'], 'test_skill': test_m_ph.get('skill_vs_clim', 0.0), 'iters': m_ph.best_iteration})
        logger.info(f"  [per_hour_honest] h={h:2d} | Val RMSE: {val_m_ph['RMSE']:.4f} | Test RMSE: {test_m_ph['RMSE']:.4f} | Test Skill: {test_m_ph.get('skill_vs_clim', 0.0):.4f}")

        # Guardar checkpoint intermedio de Fase 2
        pd.DataFrame(records).to_csv(METRICS_CSV, index=False)

    logger.info("=====================================================================")
    logger.info("  FASE 2 FINALIZADA CON EXITO")
    logger.info("=====================================================================")

if __name__ == '__main__':
    run_honest_experiments()
