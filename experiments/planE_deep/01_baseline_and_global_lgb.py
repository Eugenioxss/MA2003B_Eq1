"""
experiments/planE_deep/01_baseline_and_global_lgb.py
---------------------------------------------------
Task 1:
- Train the 6 direct LightGBM models (v2 baseline)
- Train 1 GLOBAL multi-horizon LightGBM model with 'h' as a feature
- Evaluate both on 2025 test set using evaluate_test()
- Log metrics to results_registry.csv
- Save predictions for downstream ensembling
"""
import sys, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import (
    load_panel, build_v2_features, split, evaluate_test, log_result,
    LGB_PARAMS_V2, HORIZONS, SEED
)

OUTPUT_DIR = os.path.join(ROOT, 'experiments', 'planE_deep')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run():
    print("=== [Step 1] Loading panel and building v2 features ===")
    t0 = time.time()
    panel = load_panel()
    df, features = build_v2_features(panel)
    train_df, val_df, test_df = split(df)
    print(f"Data ready in {time.time()-t0:.1f}s. Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    print(f"Base feature count: {len(features)}")

    params_direct = {
        **LGB_PARAMS_V2,
        'num_threads': 3,
        'verbose': -1,
    }

    # ----------------------------------------------------
    # 1. Direct v2 models (one per horizon)
    # ----------------------------------------------------
    print("\n--- Training 6 Direct LightGBM Models (v2) ---")
    direct_preds = test_df[['Date', 'Estacion']].copy()
    direct_metrics = {}

    for h in HORIZONS:
        target_col = f'Target_{h}h'
        t_start = time.time()
        
        tr_h = train_df.dropna(subset=[target_col])
        va_h = val_df.dropna(subset=[target_col])
        te_h = test_df.dropna(subset=[target_col])

        X_tr, y_tr = tr_h[features], tr_h[target_col]
        X_va, y_va = va_h[features], va_h[target_col]
        X_te = test_df[features]

        ds_tr = lgb.Dataset(X_tr, y_tr)
        ds_va = lgb.Dataset(X_va, y_va, reference=ds_tr)

        model = lgb.train(
            params_direct,
            ds_tr,
            num_boost_round=2000,
            valid_sets=[ds_va],
            callbacks=[
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                lgb.log_evaluation(period=0),
            ]
        )
        elapsed = time.time() - t_start
        pred_te = model.predict(X_te, num_iteration=model.best_iteration)
        direct_preds[f'pred_v2_{h}h'] = pred_te

        m = evaluate_test(test_df, pred_te, h)
        direct_metrics[h] = m
        log_result('planE_deep', 'v2_direct', h, m, f"Direct v2 LightGBM, {elapsed:.1f}s, best_iter={model.best_iteration}")
        print(f"  h={h:2d}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}, RMSE_peaks={m['RMSE_peaks']:.4f}, best_iter={model.best_iteration} ({elapsed:.1f}s)")

    direct_preds.to_parquet(os.path.join(OUTPUT_DIR, 'preds_v2_direct.parquet'), index=False)

    # ----------------------------------------------------
    # 2. Global Multi-Horizon LightGBM Model
    # ----------------------------------------------------
    print("\n--- Training 1 Global Multi-Horizon LightGBM Model ---")
    t_global_start = time.time()

    # Stack datasets across all 6 horizons
    train_stacks = []
    val_stacks = []

    for h in HORIZONS:
        target_col = f'Target_{h}h'
        
        # Subsample train by 50% if needed to keep memory & runtime optimal across 6 horizons
        tr_sub = train_df.dropna(subset=[target_col]).copy()
        # With 400k rows * 6 = 2.4M rows. 50% subsampling gives 1.2M rows which is very fast and robust.
        tr_sub = tr_sub.sample(frac=0.5, random_state=SEED + h)
        tr_sub['horizon'] = h
        tr_sub['target_y'] = tr_sub[target_col]
        train_stacks.append(tr_sub[features + ['horizon', 'target_y']])

        va_sub = val_df.dropna(subset=[target_col]).copy()
        va_sub['horizon'] = h
        va_sub['target_y'] = va_sub[target_col]
        val_stacks.append(va_sub[features + ['horizon', 'target_y']])

    df_train_global = pd.concat(train_stacks, ignore_index=True)
    df_val_global = pd.concat(val_stacks, ignore_index=True)

    global_features = features + ['horizon']
    print(f"Global train shape: {df_train_global.shape}, features: {len(global_features)}")

    params_global = {
        **LGB_PARAMS_V2,
        'num_threads': 3,
        'verbose': -1,
    }

    ds_tr_glob = lgb.Dataset(df_train_global[global_features], df_train_global['target_y'], categorical_feature=['horizon'])
    ds_va_glob = lgb.Dataset(df_val_global[global_features], df_val_global['target_y'], reference=ds_tr_glob, categorical_feature=['horizon'])

    global_model = lgb.train(
        params_global,
        ds_tr_glob,
        num_boost_round=2500,
        valid_sets=[ds_va_glob],
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=100),
        ]
    )
    t_global_train = time.time() - t_global_start
    print(f"Global model trained in {t_global_train:.1f}s, best_iteration={global_model.best_iteration}")

    # Predict test for each horizon
    global_preds = test_df[['Date', 'Estacion']].copy()
    global_metrics = {}

    for h in HORIZONS:
        test_h = test_df[features].copy()
        test_h['horizon'] = h
        pred_te = global_model.predict(test_h[global_features], num_iteration=global_model.best_iteration)
        global_preds[f'pred_global_{h}h'] = pred_te

        m = evaluate_test(test_df, pred_te, h)
        global_metrics[h] = m
        log_result('planE_deep', 'lgb_global', h, m, f"Global LightGBM (horizon feature), best_iter={global_model.best_iteration}")
        print(f"  Global h={h:2d}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}, RMSE_peaks={m['RMSE_peaks']:.4f}")

    global_preds.to_parquet(os.path.join(OUTPUT_DIR, 'preds_lgb_global.parquet'), index=False)

    # Save summary table
    print("\n=== Comparison: Direct v2 vs Global ===")
    print(f"{'h':<4} | {'v2 RMSE':<10} {'Global RMSE':<12} {'Diff':<8} | {'v2 R2':<8} {'Global R2':<10}")
    print("-" * 55)
    for h in HORIZONS:
        d_rmse = direct_metrics[h]['RMSE']
        g_rmse = global_metrics[h]['RMSE']
        diff = g_rmse - d_rmse
        d_r2 = direct_metrics[h]['R2']
        g_r2 = global_metrics[h]['R2']
        print(f"{h:<4} | {d_rmse:<10.4f} {g_rmse:<12.4f} {diff:+8.4f} | {d_r2:<8.4f} {g_r2:<10.4f}")

if __name__ == '__main__':
    run()
