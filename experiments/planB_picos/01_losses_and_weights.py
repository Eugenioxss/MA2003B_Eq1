"""
experiments/planB_picos/01_losses_and_weights.py
================================================
Compara pérdidas alternativas, transformaciones del target y ponderaciones
de muestras contra el baseline v2 (objective='regression', MSE) para mitigar
la subestimación sistemática de picos de O3.

Variantes evaluadas:
 1. v2_baseline: MSE estándar (referencia Script 7)
 2. loss_huber: Pérdida Huber (alpha=0.9)
 3. loss_tweedie_p12: Tweedie con variance_power=1.2
 4. loss_tweedie_p15: Tweedie con variance_power=1.5
 5. loss_quantile_p50: Regresión mediana (alpha=0.5, pérdida pinball/MAE)
 6. target_sqrt: Target transformado con sqrt(y), inversa z^2
 7. target_log1p: Target transformado con log1p(y), inversa expm1(z)
 8. weight_step_p90: Ponderación w=2.0 para y > p90(train), w=1.0 resto
 9. weight_quad_p90: Ponderación cuadrática w = 1 + min(9, (y/p90)^2)
 10. weight_step_p95: Ponderación w=2.5 para y > p95(train), w=1.0 resto

Horizontes prioritarios: [1, 8, 24, 48].
Registra en experiments/results_registry.csv y guarda CSV en planB_picos/metrics_losses.csv.
"""

import os, sys, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import load_panel, build_v2_features, split, evaluate_test, log_result, LGB_PARAMS_V2, HORIZONS

EXP_NAME = 'planB_picos'
OUT_DIR = os.path.join(ROOT, 'experiments', EXP_NAME)
PRED_DIR = os.path.join(OUT_DIR, 'preds')
os.makedirs(PRED_DIR, exist_ok=True)
METRICS_CSV = os.path.join(OUT_DIR, 'metrics_losses.csv')

def get_base_params():
    p = dict(LGB_PARAMS_V2)
    p['num_threads'] = 4
    p['learning_rate'] = 0.08
    p['num_leaves'] = 45
    p['min_child_samples'] = 50
    p['subsample'] = 0.8
    p['colsample_bytree'] = 0.8
    p['verbose'] = -1
    return p

def run_experiments():
    print("=" * 70, flush=True)
    print("INICIANDO EVALUACIÓN DE PÉRDIDAS, TRANSFORMACIONES Y PESOS", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    panel = load_panel()
    df, feats = build_v2_features(panel)
    train_df, val_df, test_df = split(df)
    train_sub = train_df.iloc[::2].copy()
    print(f"Features preparados: {len(feats)} vars | Train: {len(train_sub):,} | Val: {len(val_df):,} | Test: {len(test_df):,}", flush=True)

    all_metrics = []

    # -------------------------------------------------------------
    # 1. EVALUAR Y REGISTRAR BASELINE V2 EN TODOS LOS HORIZONTES
    # -------------------------------------------------------------
    print("\n--- Evaluando baseline v2 en todos los horizontes ---", flush=True)
    for h in HORIZONS:
        v2_pred_file = os.path.join(ROOT, 'results', 'multihorizonte_v2', f'pred_test_{h}h.parquet')
        if os.path.exists(v2_pred_file):
            p_df = pd.read_parquet(v2_pred_file)
            te_h = test_df.dropna(subset=[f'Target_{h}h'])
            yp_v2 = p_df['y_pred'].values
            m = evaluate_test(te_h, yp_v2, h)
            m_record = dict(m)
            m_record['variant'] = 'v2_baseline'
            m_record['h'] = h
            m_record['best_iter'] = np.nan
            m_record['train_sec'] = np.nan
            all_metrics.append(m_record)
            log_result(EXP_NAME, 'v2_baseline', h, m, notes="Baseline Script 7 (objective=regression, lr=0.05)")
            print(f"  v2_baseline h={h:2d} -> RMSE={m['RMSE']:.3f}, R2={m['R2']:.3f}, Peak Bias={m['bias_peaks']:+6.2f} ppb, Peak RMSE={m['RMSE_peaks']:.2f} ppb", flush=True)

            out_p = te_h[['Date', 'Estacion', f'Target_{h}h']].copy()
            out_p['y_pred'] = yp_v2
            out_p.to_parquet(os.path.join(PRED_DIR, f'pred_v2_baseline_{h}h.parquet'), index=False)

    # -------------------------------------------------------------
    # 2. DEFINIR VARIANTES DE PÉRDIDAS / PESOS
    # -------------------------------------------------------------
    eval_horizons = [1, 8, 24, 48]

    variant_configs = [
        {
            'name': 'loss_huber',
            'obj': 'huber',
            'metric': 'rmse',
            'extra_params': {'alpha': 0.9},
            'transform': None,
            'weight_func': None,
            'notes': 'Huber loss (alpha=0.9)'
        },
        {
            'name': 'loss_tweedie_p12',
            'obj': 'tweedie',
            'metric': 'rmse',
            'extra_params': {'tweedie_variance_power': 1.2},
            'transform': None,
            'weight_func': None,
            'notes': 'Tweedie variance_power=1.2'
        },
        {
            'name': 'loss_tweedie_p15',
            'obj': 'tweedie',
            'metric': 'rmse',
            'extra_params': {'tweedie_variance_power': 1.5},
            'transform': None,
            'weight_func': None,
            'notes': 'Tweedie variance_power=1.5'
        },
        {
            'name': 'loss_quantile_p50',
            'obj': 'quantile',
            'metric': 'mae',
            'extra_params': {'alpha': 0.5},
            'transform': None,
            'weight_func': None,
            'notes': 'Regresión cuantílica mediana alpha=0.5'
        },
        {
            'name': 'target_sqrt',
            'obj': 'regression',
            'metric': 'rmse',
            'extra_params': {},
            'transform': 'sqrt',
            'weight_func': None,
            'notes': 'Transformación target sqrt(y), inversa z^2'
        },
        {
            'name': 'target_log1p',
            'obj': 'regression',
            'metric': 'rmse',
            'extra_params': {},
            'transform': 'log1p',
            'weight_func': None,
            'notes': 'Transformación target log1p(y), inversa expm1(z)'
        },
        {
            'name': 'weight_step_p90',
            'obj': 'regression',
            'metric': 'rmse',
            'extra_params': {},
            'transform': None,
            'weight_func': lambda y, p90, p95: np.where(y > p90, 2.0, 1.0),
            'notes': 'Muestras y > p90 con peso w=2.0'
        },
        {
            'name': 'weight_quad_p90',
            'obj': 'regression',
            'metric': 'rmse',
            'extra_params': {},
            'transform': None,
            'weight_func': lambda y, p90, p95: 1.0 + np.minimum(9.0, (np.maximum(0.0, y) / p90) ** 2),
            'notes': 'Ponderación cuadrática w = 1 + min(9, (y/p90)^2)'
        },
        {
            'name': 'weight_step_p95',
            'obj': 'regression',
            'metric': 'rmse',
            'extra_params': {},
            'transform': None,
            'weight_func': lambda y, p90, p95: np.where(y > p95, 2.5, 1.0),
            'notes': 'Muestras y > p95 con peso w=2.5'
        },
    ]

    for cfg in variant_configs:
        v_name = cfg['name']
        print(f"\n=======================================================", flush=True)
        print(f"ENTRENANDO VARIANTE: {v_name}", flush=True)
        print(f"Descripción: {cfg['notes']}", flush=True)
        print(f"=======================================================", flush=True)

        for h in eval_horizons:
            tc = f'Target_{h}h'
            tr_h = train_sub.dropna(subset=[tc])
            va_h = val_df.dropna(subset=[tc])
            te_h = test_df.dropna(subset=[tc])

            y_tr_raw = tr_h[tc].values
            y_va_raw = va_h[tc].values
            y_te_raw = te_h[tc].values

            p90_tr = float(np.percentile(y_tr_raw, 90))
            p95_tr = float(np.percentile(y_tr_raw, 95))

            if cfg['transform'] == 'sqrt':
                y_tr = np.sqrt(np.maximum(0.0, y_tr_raw))
                y_va = np.sqrt(np.maximum(0.0, y_va_raw))
            elif cfg['transform'] == 'log1p':
                y_tr = np.log1p(np.maximum(0.0, y_tr_raw))
                y_va = np.log1p(np.maximum(0.0, y_va_raw))
            else:
                y_tr = y_tr_raw
                y_va = y_va_raw

            if cfg['weight_func'] is not None:
                w_tr = cfg['weight_func'](y_tr_raw, p90_tr, p95_tr)
                w_va = cfg['weight_func'](y_va_raw, p90_tr, p95_tr)
            else:
                w_tr = None
                w_va = None

            dtrain = lgb.Dataset(tr_h[feats].values, y_tr, weight=w_tr)
            dval = lgb.Dataset(va_h[feats].values, y_va, weight=w_va, reference=dtrain)

            params = get_base_params()
            params['objective'] = cfg['obj']
            params['metric'] = cfg['metric']
            params.update(cfg['extra_params'])

            t_start = time.time()
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=800,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            t_train = time.time() - t_start

            raw_pred = model.predict(te_h[feats].values)
            if cfg['transform'] == 'sqrt':
                y_pred = np.square(np.maximum(0.0, raw_pred))
            elif cfg['transform'] == 'log1p':
                y_pred = np.expm1(np.maximum(0.0, raw_pred))
            else:
                y_pred = np.maximum(0.0, raw_pred)

            m = evaluate_test(te_h, y_pred, h)
            m_record = dict(m)
            m_record['variant'] = v_name
            m_record['h'] = h
            m_record['best_iter'] = model.best_iteration
            m_record['train_sec'] = round(t_train, 1)
            all_metrics.append(m_record)

            log_result(EXP_NAME, v_name, h, m, notes=f"{cfg['notes']} (iter={model.best_iteration})")

            out_p = te_h[['Date', 'Estacion', tc]].copy()
            out_p['y_pred'] = y_pred
            out_p.to_parquet(os.path.join(PRED_DIR, f'pred_{v_name}_{h}h.parquet'), index=False)

            print(f"  h={h:2d} ({t_train:4.1f}s, iter={model.best_iteration:3d}) -> "
                  f"RMSE={m['RMSE']:.3f}, R2={m['R2']:.3f}, "
                  f"Peak Bias={m['bias_peaks']:+6.2f} ppb, Peak RMSE={m['RMSE_peaks']:.2f} ppb", flush=True)

            del dtrain, dval, model
            gc.collect()

        df_out = pd.DataFrame(all_metrics)
        df_out.to_csv(METRICS_CSV, index=False)

    print("\n" + "=" * 70, flush=True)
    print("EVALUACIÓN DE PÉRDIDAS Y PESOS FINALIZADA EXITOSAMENTE", flush=True)
    print(f"Resultados guardados en {METRICS_CSV}", flush=True)
    print("=" * 70, flush=True)

if __name__ == '__main__':
    run_experiments()
