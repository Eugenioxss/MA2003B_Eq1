"""
experiments/planE_deep/02_chronos_zeroshot.py
---------------------------------------------
Task 2: Zero-shot foundation model Chronos-Bolt.
- Loads amazon/chronos-bolt-small on CUDA
- Extracts 512h O3 context up to t-1 for each test instance (Date, Estacion) in 2025
- Fills short NaNs (< 6h) with linear interpolation; tracks NaN ratio (> 20% flagged)
- Predicts 49 steps ahead on GPU in batches
- Extracts quantiles [0.1, 0.5, 0.9] for horizons [1, 4, 8, 12, 24, 48]
- Evaluates median forecasts with evaluate_test() and computes empirical 80% coverage
- Logs to results_registry.csv and saves test predictions to preds_chronos.parquet
"""
import sys, os, time
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import (
    load_panel, add_targets, split, evaluate_test, log_result,
    HORIZONS, MODEL_STATIONS
)

OUTPUT_DIR = os.path.join(ROOT, 'experiments', 'planE_deep')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run():
    print("=== [Step 2] Zero-Shot Forecasting with Chronos-Bolt ===")
    t0 = time.time()
    
    # 1. Load Chronos Pipeline
    from chronos import BaseChronosPipeline
    print("Loading amazon/chronos-bolt-small on CUDA...")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map="cuda",
        torch_dtype=torch.float32,
    )
    print("Pipeline ready.")

    # 2. Load complete hourly panel for context extraction
    panel = load_panel(stations=MODEL_STATIONS)
    panel = add_targets(panel)
    panel = panel.sort_values(['Estacion', 'Date']).reset_index(drop=True)
    
    # Prepare station series dictionary for fast O(1) slicing
    # Indexing by Date
    station_series = {}
    for est, grp in panel.groupby('Estacion'):
        s = grp.set_index('Date')['O3']
        station_series[est] = s

    # 3. Target test set (year 2025)
    train_df, val_df, test_df = split(panel)
    test_dates_stations = test_df[['Date', 'Estacion']].drop_duplicates().sort_values(['Date', 'Estacion']).reset_index(drop=True)
    N_test = len(test_dates_stations)
    print(f"Total test instances to forecast: {N_test:,} (13 stations x 8760 hours)")

    CONTEXT_LEN = 512
    PRED_LEN = 49  # step 0 is t, step 1 is t+1, ..., step 48 is t+48
    BATCH_SIZE = 256

    # Build context tensors
    print("Extracting 512h contexts up to t-1 for all test points...")
    contexts = []
    high_nan_flags = []
    
    for _, row in test_dates_stations.iterrows():
        est = row['Estacion']
        t = row['Date']
        s = station_series[est]
        # Context ends at t-1 hour
        end_time = t - pd.Timedelta(hours=1)
        start_time = end_time - pd.Timedelta(hours=CONTEXT_LEN - 1)
        
        ctx = s.loc[start_time:end_time].to_numpy(dtype=np.float32)
        if len(ctx) < CONTEXT_LEN:
            # Pad with nan if earlier than start
            pad = np.full(CONTEXT_LEN - len(ctx), np.nan, dtype=np.float32)
            ctx = np.concatenate([pad, ctx])
            
        nan_ratio = np.isnan(ctx).mean()
        high_nan_flags.append(nan_ratio > 0.20)
        
        # Interpolate short NaNs
        s_ctx = pd.Series(ctx)
        s_interp = s_ctx.interpolate(method='linear', limit=6).bfill().ffill()
        # If all NaN or still has NaN, fill with station mean or 0
        if s_interp.isna().any():
            s_interp = s_interp.fillna(0.0)
            
        contexts.append(s_interp.to_numpy(dtype=np.float32))

    contexts = np.array(contexts)
    high_nan_flags = np.array(high_nan_flags)
    print(f"Contexts built. High NaN (>20%) cases: {high_nan_flags.sum():,} / {N_test:,} ({high_nan_flags.mean()*100:.2f}%)")

    # 4. Predict in GPU batches
    print(f"Generating predictions in batches of {BATCH_SIZE} on GPU...")
    all_preds_q10 = []
    all_preds_q50 = []
    all_preds_q90 = []
    
    pipeline.model.eval()
    num_batches = int(np.ceil(N_test / BATCH_SIZE))
    t_inf_start = time.time()

    for b in range(num_batches):
        b_start = b * BATCH_SIZE
        b_end = min(b_start + BATCH_SIZE, N_test)
        b_ctx = torch.from_numpy(contexts[b_start:b_end]).to(device="cuda", dtype=torch.float32)

        with torch.no_grad():
            # quantiles tensor shape: (batch_size, n_quantiles, prediction_length)
            # Default quantiles in bolt: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            preds = pipeline.predict(b_ctx, prediction_length=PRED_LEN)
            # Bolt quantiles: index 0 = 0.1, index 4 = 0.5, index 8 = 0.9
            q10 = preds[:, 0, :].cpu().numpy()
            q50 = preds[:, 4, :].cpu().numpy()
            q90 = preds[:, 8, :].cpu().numpy()

        all_preds_q10.append(q10)
        all_preds_q50.append(q50)
        all_preds_q90.append(q90)

        if (b + 1) % 100 == 0 or (b + 1) == num_batches:
            torch.cuda.empty_cache()
            print(f"  Batch {b+1}/{num_batches} done ({time.time() - t_inf_start:.1f}s)")

    preds_q10 = np.concatenate(all_preds_q10, axis=0)
    preds_q50 = np.concatenate(all_preds_q50, axis=0)
    preds_q90 = np.concatenate(all_preds_q90, axis=0)
    total_inf_time = time.time() - t_inf_start
    print(f"Total inference time for {N_test:,} series x 49 horizons: {total_inf_time:.1f}s ({N_test/total_inf_time:.1f} series/s)")

    # 5. Map forecasts to test dataframe
    # Add columns to test_dates_stations
    # Note: horizon h is at step index h (since step 0 is t, step 1 is t+1, ..., step 48 is t+48)
    for h in HORIZONS:
        test_dates_stations[f'chronos_pred_{h}h'] = preds_q50[:, h]
        test_dates_stations[f'chronos_q10_{h}h'] = preds_q10[:, h]
        test_dates_stations[f'chronos_q90_{h}h'] = preds_q90[:, h]
    test_dates_stations['context_high_nan'] = high_nan_flags

    # Merge predictions into test_df for standard evaluate_test
    eval_test = pd.merge(test_df, test_dates_stations, on=['Date', 'Estacion'], how='left')

    # 6. Evaluate each horizon
    print("\n=== Chronos-Bolt Evaluation Results (Test 2025) ===")
    chronos_results = {}
    for h in HORIZONS:
        pred_med = eval_test[f'chronos_pred_{h}h'].values
        # Target_h
        y_true = eval_test[f'Target_{h}h'].values
        q10 = eval_test[f'chronos_q10_{h}h'].values
        q90 = eval_test[f'chronos_q90_{h}h'].values

        valid = np.isfinite(y_true) & np.isfinite(pred_med)
        cov_80 = float(np.mean((y_true[valid] >= q10[valid]) & (y_true[valid] <= q90[valid])))

        m = evaluate_test(eval_test, pred_med, h)
        m['coverage_80'] = cov_80
        chronos_results[h] = m

        notes = f"Chronos-bolt-small zero-shot 512h context. 80% coverage: {cov_80*100:.1f}%. Inf time: {total_inf_time:.1f}s"
        log_result('planE_deep', 'chronos_bolt_small', h, m, notes)
        print(f"  h={h:2d}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2']:.4f}, RMSE_peaks={m['RMSE_peaks']:.4f}, Cov80%={cov_80*100:.1f}%")

    # Save predictions
    save_cols = ['Date', 'Estacion', 'context_high_nan'] + [f'chronos_pred_{h}h' for h in HORIZONS] + [f'chronos_q10_{h}h' for h in HORIZONS] + [f'chronos_q90_{h}h' for h in HORIZONS]
    eval_test[save_cols].to_parquet(os.path.join(OUTPUT_DIR, 'preds_chronos.parquet'), index=False)
    print(f"\nChronos predictions saved to {os.path.join(OUTPUT_DIR, 'preds_chronos.parquet')}")

if __name__ == '__main__':
    run()
