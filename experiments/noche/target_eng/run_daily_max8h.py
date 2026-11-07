"""
experiments/noche/target_eng/run_daily_max8h.py
===============================================
Pronostico regulatorio NOM-020-SSA1:
Predecir el maximo diario de la media movil de 8 horas de O3 (DMA8) del dia siguiente
con emision operativa a las 18:00 del dia actual.

Compara:
  1. Climatologia (Estacion x Mes, 2020-2023)
  2. Persistencia (DMA8 observado hoy hasta las 18:00)
  3. LightGBM (features meteorologicos y quimicos a las 18:00)

Reporta:
  - RMSE, MAE, R2
  - Deteccion de excedencias > 70 ppb (norma NOM-020 proteccion a la salud)
  - Deteccion de excedencias > 90 ppb (precontingencia ambiental)
"""
import sys
import os
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXP_DIR = os.path.join(ROOT, 'experiments')
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from common import load_panel, MODEL_STATIONS

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(CUR_DIR, 'daily_max8h_summary.csv')
PREDS_FILE = os.path.join(CUR_DIR, 'preds_daily_max8h.parquet')

def compute_dma8_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    """Calcula la media movil de 8h y construye el dataset diario con emision a las 18:00."""
    print("Calculando media movil de 8h de O3 por estacion...")
    df = panel[panel['Estacion'].isin(MODEL_STATIONS)].copy()
    df = df.sort_values(['Estacion', 'Date']).reset_index(drop=True)
    
    # Media movil de 8h (min 6 periodos = 75% completitud segun norma)
    df['O3_8h'] = df.groupby('Estacion')['O3'].transform(lambda s: s.rolling(8, min_periods=6).mean())
    df['Date_only'] = df['Date'].dt.normalize()
    df['Hour'] = df['Date'].dt.hour
    
    # 1. DMA8 del dia completo (target para D+1)
    # Exigir al menos 18 horas validas de O3_8h para considerar dia representativo
    daily_stats = df.groupby(['Estacion', 'Date_only']).agg(
        DMA8_true=('O3_8h', 'max'),
        valid_8h_count=('O3_8h', 'count')
    ).reset_index()
    daily_stats.loc[daily_stats['valid_8h_count'] < 18, 'DMA8_true'] = np.nan
    
    # 2. Features del dia actual hasta las 18:00
    df_18h = df[df['Hour'] <= 18].copy()
    feats_daily = df_18h.groupby(['Estacion', 'Date_only']).agg(
        DMA8_today_so_far=('O3_8h', 'max'),
        O3_max_1h_today=('O3', 'max'),
        O3_mean_today=('O3', 'mean'),
        TOUT_max_today=('TOUT', 'max'),
        TOUT_mean_today=('TOUT', 'mean'),
        SR_sum_today=('SR', 'sum'),
        NOX_mean_today=('NOX', 'mean'),
        PM25_mean_today=('PM2.5', 'mean'),
        PM10_mean_today=('PM10', 'mean')
    ).reset_index()
    
    # Instantaneos a las 18:00
    df_at_18 = df[df['Hour'] == 18][['Estacion', 'Date_only', 'O3', 'TOUT', 'RH', 'WSR', 'WDR', 'NOX']].copy()
    df_at_18.columns = ['Estacion', 'Date_only', 'O3_at_18', 'TOUT_at_18', 'RH_at_18', 'WSR_at_18', 'WDR_at_18', 'NOX_at_18']
    
    # Merge de features del dia D
    dataset = pd.merge(feats_daily, df_at_18, on=['Estacion', 'Date_only'], how='left')
    
    # Target: DMA8 del dia D+1 (tomorrow)
    daily_stats_target = daily_stats[['Estacion', 'Date_only', 'DMA8_true']].copy()
    daily_stats_target['Date_only'] = daily_stats_target['Date_only'] - pd.Timedelta(days=1)
    daily_stats_target.rename(columns={'DMA8_true': 'Target_DMA8_next_day'}, inplace=True)
    
    dataset = pd.merge(dataset, daily_stats_target, on=['Estacion', 'Date_only'], how='inner')
    
    # Lag 1 de DMA8 (yesterday)
    daily_stats_yesterday = daily_stats[['Estacion', 'Date_only', 'DMA8_true']].copy()
    daily_stats_yesterday['Date_only'] = daily_stats_yesterday['Date_only'] + pd.Timedelta(days=1)
    daily_stats_yesterday.rename(columns={'DMA8_true': 'DMA8_yesterday'}, inplace=True)
    dataset = pd.merge(dataset, daily_stats_yesterday, on=['Estacion', 'Date_only'], how='left')
    
    # Fecha y calendario
    dataset['Date'] = dataset['Date_only'] + pd.Timedelta(hours=18)
    dataset['month'] = dataset['Date'].dt.month
    dataset['dayofweek'] = dataset['Date'].dt.dayofweek
    dataset['is_weekend'] = (dataset['dayofweek'] >= 5).astype(int)
    dataset['month_sin'] = np.sin(2 * np.pi * dataset['month'] / 12)
    dataset['month_cos'] = np.cos(2 * np.pi * dataset['month'] / 12)
    
    # Features espaciales de la red a las 18:00 (promedio y max regional de DMA8_today)
    reg_stats = dataset.groupby('Date_only')['DMA8_today_so_far'].agg(
        reg_DMA8_mean='mean',
        reg_DMA8_max='max'
    ).reset_index()
    dataset = pd.merge(dataset, reg_stats, on='Date_only', how='left')
    
    dataset['Estacion'] = dataset['Estacion'].astype('category')
    dataset = dataset.sort_values(['Date_only', 'Estacion']).reset_index(drop=True)
    print(f"Dataset DMA8 listo: {len(dataset):,} filas diarias.")
    return dataset

def calc_classif_metrics(y_true, y_pred, threshold):
    yt_bin = (y_true >= threshold).astype(int)
    yp_bin = (y_pred >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(yt_bin, yp_bin, labels=[0, 1]).ravel()
    acc = accuracy_score(yt_bin, yp_bin)
    prec = precision_score(yt_bin, yp_bin, zero_division=0)
    rec = recall_score(yt_bin, yp_bin, zero_division=0)
    f1 = f1_score(yt_bin, yp_bin, zero_division=0)
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    return {
        'threshold': threshold,
        'exceedances_true': int(yt_bin.sum()),
        'exceedances_pred': int(yp_bin.sum()),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'CSI_threat_score': round(csi, 4)
    }

def main():
    print("=" * 70)
    print("  EXPERIMENTO: PRONOSTICO DEL MAXIMO DIARIO 8H (NOM-020-SSA1)")
    print("  Emision a las 18:00 para predecir DMA8 del dia siguiente")
    print("=" * 70)
    
    panel = load_panel()
    ds = compute_dma8_dataset(panel)
    
    # Filtrar solo dias con target valido
    ds = ds.dropna(subset=['Target_DMA8_next_day']).reset_index(drop=True)
    
    # Split
    tr = ds[ds['Date'].dt.year <= 2023].copy()
    va = ds[ds['Date'].dt.year == 2024].copy()
    ts = ds[ds['Date'].dt.year == 2025].copy()
    
    print(f"Split DMA8 -> Train: {len(tr):,}, Val: {len(va):,}, Test: {len(ts):,}")
    
    feature_cols = [
        'Estacion', 'DMA8_today_so_far', 'O3_max_1h_today', 'O3_mean_today',
        'TOUT_max_today', 'TOUT_mean_today', 'SR_sum_today', 'NOX_mean_today',
        'PM25_mean_today', 'PM10_mean_today', 'O3_at_18', 'TOUT_at_18', 'RH_at_18',
        'WSR_at_18', 'WDR_at_18', 'NOX_at_18', 'DMA8_yesterday',
        'month', 'dayofweek', 'is_weekend', 'month_sin', 'month_cos',
        'reg_DMA8_mean', 'reg_DMA8_max'
    ]
    
    # -------------------------------------------------------------
    # 1. Baseline: Climatologia (estacion x mes)
    # -------------------------------------------------------------
    clim_map = tr.groupby(['Estacion', 'month'])['Target_DMA8_next_day'].mean()
    idx_ts = pd.MultiIndex.from_arrays([ts['Estacion'], ts['month']])
    pred_clim = clim_map.reindex(idx_ts).values
    
    # -------------------------------------------------------------
    # 2. Baseline: Persistencia (DMA8_today_so_far)
    # -------------------------------------------------------------
    pred_persist = np.where(ts['DMA8_today_so_far'].notnull(), ts['DMA8_today_so_far'].values, pred_clim)
    
    # -------------------------------------------------------------
    # 3. LightGBM
    # -------------------------------------------------------------
    params_path = os.path.join(EXP_DIR, 'planA_quickwins', 'best_params_optuna.json')
    with open(params_path, 'r') as f:
        params = json.load(f)
    params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'num_threads': 3,
        'seed': 42,
        'verbose': -1
    })
    
    ds_tr = lgb.Dataset(tr[feature_cols], tr['Target_DMA8_next_day'], categorical_feature=['Estacion'])
    ds_va = lgb.Dataset(va[feature_cols], va['Target_DMA8_next_day'], reference=ds_tr, categorical_feature=['Estacion'])
    
    model = lgb.train(
        params,
        ds_tr,
        num_boost_round=1000,
        valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    pred_lgb_val = np.maximum(0.0, model.predict(va[feature_cols]))
    pred_lgb_test = np.maximum(0.0, model.predict(ts[feature_cols]))
    
    y_test = ts['Target_DMA8_next_day'].values
    
    # Evaluacion de los 3 modelos
    models = {
        'Climatologia (Mes x Est)': pred_clim,
        'Persistencia (DMA8 hoy)': pred_persist,
        'LightGBM 18:00': pred_lgb_test
    }
    
    results = []
    for name, yp in models.items():
        ok = np.isfinite(y_test) & np.isfinite(yp)
        yt, yp_ok = y_test[ok], yp[ok]
        rmse = float(np.sqrt(np.mean((yt - yp_ok)**2)))
        mae = float(np.mean(np.abs(yt - yp_ok)))
        r2 = float(1 - np.sum((yt - yp_ok)**2) / np.sum((yt - yt.mean())**2))
        bias = float(np.mean(yp_ok - yt))
        
        c70 = calc_classif_metrics(yt, yp_ok, 70.0)
        c90 = calc_classif_metrics(yt, yp_ok, 90.0)
        
        row = {
            'Modelo': name,
            'RMSE': round(rmse, 3),
            'MAE': round(mae, 3),
            'R2': round(r2, 3),
            'Bias': round(bias, 3),
            'Acc_70ppb': c70['Accuracy'],
            'Prec_70ppb': c70['Precision'],
            'Rec_70ppb': c70['Recall'],
            'F1_70ppb': c70['F1'],
            'CSI_70ppb': c70['CSI_threat_score'],
            'Acc_90ppb': c90['Accuracy'],
            'Prec_90ppb': c90['Precision'],
            'Rec_90ppb': c90['Recall'],
            'F1_90ppb': c90['F1'],
            'CSI_90ppb': c90['CSI_threat_score']
        }
        results.append(row)
        print(f"\n--- {name} ---")
        print(f"  RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f} | Bias: {bias:.3f}")
        print(f"  >70 ppb: F1={c70['F1']:.3f}, Recall={c70['Recall']:.3f}, Prec={c70['Precision']:.3f}, CSI={c70['CSI_threat_score']:.3f} (TP={c70['TP']}, FP={c70['FP']}, FN={c70['FN']})")
        print(f"  >90 ppb: F1={c90['F1']:.3f}, Recall={c90['Recall']:.3f}, Prec={c90['Precision']:.3f}, CSI={c90['CSI_threat_score']:.3f} (TP={c90['TP']}, FP={c90['FP']}, FN={c90['FN']})")

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_CSV, index=False)
    print(f"\nResumen DMA8 guardado en {OUT_CSV}")
    
    # Guardar predicciones de test
    pred_df = ts[['Date', 'Estacion', 'Target_DMA8_next_day']].copy()
    pred_df['pred_clim'] = pred_clim
    pred_df['pred_persist'] = pred_persist
    pred_df['pred_lgb'] = pred_lgb_test
    pred_df.to_parquet(PREDS_FILE, index=False)
    print(f"Predicciones guardadas en {PREDS_FILE}")

if __name__ == '__main__':
    main()
