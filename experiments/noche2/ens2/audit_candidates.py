"""
experiments/noche2/ens2/audit_candidates.py
===========================================
Auditoría Exhaustiva de Modelos Honestos y Matriz de Diversidad Inter-Familia.

Objetivos:
1. Escanear todos los parquets honestos en noche y noche2.
2. Calcular métricas completas en Val 2024 (RMSE, MAE, R2, RMSE_peaks, bias_peaks, RMSE_day).
3. Calcular correlación de residuos entre modelos y familias.
4. Generar reportes CSV y figuras de calor (heatmaps).
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import re
import glob
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
ENS2_DIR = os.path.abspath(os.path.dirname(__file__))
PREDS_DIR = os.path.join(ENS2_DIR, 'preds')
FIGS_DIR = os.path.join(ENS2_DIR, 'figures')
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

HORIZONS = [1, 4, 8, 12, 24, 48]

def classify_family(exp: str, var: str) -> str:
    """Clasifica un modelo en su familia algorítmica."""
    v_lower = var.lower()
    e_lower = exp.lower()
    
    if 'xgb' in v_lower:
        return 'XGBoost'
    if 'et_' in v_lower or 'extratrees' in v_lower:
        return 'ExtraTrees'
    if 'deep_imput2' in v_lower:
        return 'Deep_Imput2_Resid'
    if 'imput2' in e_lower or 'audit_imput' in e_lower or 'imput' in v_lower:
        return 'Imputation_Spatial'
    if 'deep2' in e_lower or 'deep' in e_lower or 'resnet' in v_lower or 'gated' in v_lower or 'swiglu' in v_lower or 'gru' in v_lower or 'seq' in v_lower:
        if 'resnet' in v_lower:
            return 'Deep_TabResNet'
        elif 'gated' in v_lower or 'swiglu' in v_lower:
            return 'Deep_GatedSwiGLU'
        elif 'residual' in v_lower:
            return 'Deep_LGBM_Residual'
        elif 'gru' in v_lower or 'seq' in v_lower:
            return 'Deep_Seq_GRU'
        else:
            return 'Deep_Other'
    if 'refit' in e_lower or 'refit' in v_lower:
        return 'LightGBM_Refit'
    if 'nwp' in e_lower or 'nwp' in v_lower:
        return 'LightGBM_NWP'
    if 'target_eng' in e_lower or 'eng' in v_lower:
        return 'Target_Eng'
    if 'tuning' in e_lower:
        return 'LGBM_Tuning'
    if 'v3_base' in e_lower:
        return 'LGBM_v3_Base'
    return 'Other'

def discover_candidate_files() -> dict[int, list[dict]]:
    """Descubre recursivamente todos los parquets válidos y honestos."""
    patterns = [
        os.path.join(ROOT, 'experiments', 'noche', '*', 'preds', '*.parquet'),
        os.path.join(ROOT, 'experiments', 'noche2', '*', 'preds', '*.parquet')
    ]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat))
        
    discovered: dict[int, list[dict]] = {h: [] for h in HORIZONS}
    excluded = []
    
    for fpath in all_files:
        fname = os.path.basename(fpath)
        m = re.match(r'(.+)_h(\d+)\.parquet$', fname)
        if not m:
            continue
        var, h_str = m.group(1), m.group(2)
        h = int(h_str)
        if h not in HORIZONS:
            continue
            
        parts = fpath.replace('\\', '/').split('/')
        # exp_folder = noche/something or noche2/something
        exp = parts[-3]
        
        # Filtros de honestidad estricta
        if any(bad in var.lower() for bad in ['_pp', 'pp_', 'leak_test', 'ensemble_', 'ens2_']):
            excluded.append((fpath, "Variante excluida por honestidad (_pp, ensemble, etc.)"))
            continue
            
        # Comprobar columnas y splits requeridos
        try:
            head = pd.read_parquet(fpath, columns=['split', 'y_true', 'y_pred'])
            splits = head['split'].unique()
            if 'val' not in splits or 'test' not in splits:
                excluded.append((fpath, f"Faltan splits (splits={splits})"))
                continue
                
            val_cov = head[head['split'] == 'val']['y_pred'].notna().mean()
            test_cov = head[head['split'] == 'test']['y_pred'].notna().mean()
            if val_cov < 0.99 or test_cov < 0.99:
                excluded.append((fpath, f"Cobertura incompleta (val={val_cov:.3f}, test={test_cov:.3f})"))
                continue
                
            family = classify_family(exp, var)
            unique_name = f"{exp}__{var}"
            discovered[h].append({
                'exp': exp,
                'var': var,
                'model_name': unique_name,
                'path': fpath,
                'family': family,
                'h': h
            })
        except Exception as e:
            excluded.append((fpath, f"Error leyendo parquet: {e}"))
            
    print(f"Descubrimiento completado: {sum(len(v) for v in discovered.values())} modelos válidos en total.")
    print(f"Excluidos por honestidad o formato: {len(excluded)}")
    return discovered

def audit_horizon(h: int, candidates: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga y audita modelos para un horizonte h."""
    print(f"\n--- Auditando Horizonte h={h} ({len(candidates)} candidatos) ---")
    
    # Cargar y alinear todos los modelos
    val_preds = {}
    test_preds = {}
    y_true_val = None
    y_true_test = None
    meta_val = None
    meta_test = None
    
    for c in candidates:
        name = c['model_name']
        df = pd.read_parquet(c['path'])
        df['Date'] = pd.to_datetime(df['Date'])
        
        df_val = df[df['split'] == 'val'].sort_values(['Date', 'Estacion']).reset_index(drop=True)
        df_test = df[df['split'] == 'test'].sort_values(['Date', 'Estacion']).reset_index(drop=True)
        
        if y_true_val is None:
            y_true_val = df_val['y_true'].values
            meta_val = df_val[['Date', 'Estacion']].copy()
            meta_val['target_time'] = meta_val['Date'] + pd.Timedelta(hours=h)
            meta_val['target_hour'] = meta_val['target_time'].dt.hour
            meta_val['target_month'] = meta_val['target_time'].dt.month
            meta_val['fold'] = (meta_val['target_month'] - 1) // 3
            
        if y_true_test is None:
            y_true_test = df_test['y_true'].values
            meta_test = df_test[['Date', 'Estacion']].copy()
            meta_test['target_time'] = meta_test['Date'] + pd.Timedelta(hours=h)
            meta_test['target_hour'] = meta_test['target_time'].dt.hour
            
        val_preds[name] = df_val['y_pred'].values
        test_preds[name] = df_test['y_pred'].values
        
    df_val_preds = pd.DataFrame(val_preds)
    df_test_preds = pd.DataFrame(test_preds)
    
    # Imputar si hubiera algún NaN residual aislado con media de la fila
    if df_val_preds.isna().any().any():
        row_mean = df_val_preds.mean(axis=1)
        df_val_preds = df_val_preds.apply(lambda col: col.fillna(row_mean))
    if df_test_preds.isna().any().any():
        row_mean = df_test_preds.mean(axis=1)
        df_test_preds = df_test_preds.apply(lambda col: col.fillna(row_mean))
        
    # Calcular métricas individuales en Val 2024 y Test 2025
    ok_val = np.isfinite(y_true_val)
    thr_val_pk = np.quantile(y_true_val[ok_val], 0.95)
    pk_val = ok_val & (y_true_val >= thr_val_pk)
    day_val = ok_val & (meta_val['target_hour'] >= 10) & (meta_val['target_hour'] <= 18)
    
    ok_test = np.isfinite(y_true_test)
    thr_test_pk = np.quantile(y_true_test[ok_test], 0.95)
    pk_test = ok_test & (y_true_test >= thr_test_pk)
    day_test = ok_test & (meta_test['target_hour'] >= 10) & (meta_test['target_hour'] <= 18)
    
    metrics_records = []
    residuals_val = {}
    
    for c in candidates:
        name = c['model_name']
        yp_v = df_val_preds[name].values
        yp_t = df_test_preds[name].values
        
        # Residuos en val: e = y - ŷ
        res_v = y_true_val - yp_v
        residuals_val[name] = res_v
        
        # Métricas Val
        val_rmse = np.sqrt(np.mean((y_true_val[ok_val] - yp_v[ok_val]) ** 2))
        val_mae = np.mean(np.abs(y_true_val[ok_val] - yp_v[ok_val]))
        val_r2 = 1.0 - np.sum((y_true_val[ok_val] - yp_v[ok_val]) ** 2) / np.sum((y_true_val[ok_val] - y_true_val[ok_val].mean()) ** 2)
        val_pk_rmse = np.sqrt(np.mean((y_true_val[pk_val] - yp_v[pk_val]) ** 2))
        val_pk_bias = np.mean(yp_v[pk_val] - y_true_val[pk_val])
        val_day_rmse = np.sqrt(np.mean((y_true_val[day_val] - yp_v[day_val]) ** 2))
        
        # Métricas Test
        test_rmse = np.sqrt(np.mean((y_true_test[ok_test] - yp_t[ok_test]) ** 2))
        test_mae = np.mean(np.abs(y_true_test[ok_test] - yp_t[ok_test]))
        test_r2 = 1.0 - np.sum((y_true_test[ok_test] - yp_t[ok_test]) ** 2) / np.sum((y_true_test[ok_test] - y_true_test[ok_test].mean()) ** 2)
        test_pk_rmse = np.sqrt(np.mean((y_true_test[pk_test] - yp_t[pk_test]) ** 2))
        test_pk_bias = np.mean(yp_t[pk_test] - y_true_test[pk_test])
        test_day_rmse = np.sqrt(np.mean((y_true_test[day_test] - yp_t[day_test]) ** 2))
        
        metrics_records.append({
            'h': h,
            'model_name': name,
            'exp': c['exp'],
            'var': c['var'],
            'family': c['family'],
            'val_RMSE': round(val_rmse, 4),
            'val_MAE': round(val_mae, 4),
            'val_R2': round(val_r2, 4),
            'val_RMSE_peaks': round(val_pk_rmse, 4),
            'val_bias_peaks': round(val_pk_bias, 4),
            'val_RMSE_day': round(val_day_rmse, 4),
            'test_RMSE': round(test_rmse, 4),
            'test_MAE': round(test_mae, 4),
            'test_R2': round(test_r2, 4),
            'test_RMSE_peaks': round(test_pk_rmse, 4),
            'test_bias_peaks': round(test_pk_bias, 4),
            'test_RMSE_day': round(test_day_rmse, 4)
        })
        
    df_metrics = pd.DataFrame(metrics_records).sort_values('val_RMSE').reset_index(drop=True)
    df_res = pd.DataFrame(residuals_val)
    
    # Matriz de correlación de residuos
    corr_matrix = df_res.corr()
    
    # Guardar heatmap de los top modelos
    top_models = df_metrics.head(15)['model_name'].tolist()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix.loc[top_models, top_models], annot=True, fmt=".2f", cmap="coolwarm", vmin=0.3, vmax=1.0)
    plt.title(f"Horizonte h={h}: Correlación de Residuos (Val 2024) entre Top Modelos", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f"corr_residuals_top_h{h}.png"), dpi=200)
    plt.close()
    
    # Correlación agregada por familia
    family_res = {}
    for fam in df_metrics['family'].unique():
        fam_mods = df_metrics[df_metrics['family'] == fam]['model_name'].tolist()
        family_res[fam] = df_res[fam_mods].mean(axis=1)
    df_fam_res = pd.DataFrame(family_res)
    fam_corr = df_fam_res.corr()
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(fam_corr, annot=True, fmt=".3f", cmap="vlag", vmin=-0.2, vmax=1.0)
    plt.title(f"Horizonte h={h}: Correlación de Residuos entre Familias (Val 2024)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f"corr_families_h{h}.png"), dpi=200)
    plt.close()
    
    return df_metrics, corr_matrix, fam_corr

def run_audit():
    print("=" * 75)
    print("AUDITORÍA DE CANDIDATOS Y MATRIZ DE DIVERSIDAD (NOCHE + NOCHE2)")
    print("=" * 75)
    
    discovered = discover_candidate_files()
    all_metrics = []
    
    for h in HORIZONS:
        cands = discovered[h]
        if not cands:
            print(f"No hay candidatos para h={h}")
            continue
        df_m, corr_m, fam_corr = audit_horizon(h, cands)
        all_metrics.append(df_m)
        corr_m.to_csv(os.path.join(ENS2_DIR, f"corr_residuals_h{h}.csv"))
        fam_corr.to_csv(os.path.join(ENS2_DIR, f"corr_families_h{h}.csv"))
        
        print(f"\nTop 5 Modelos en Val 2024 para h={h}:")
        print(df_m[['model_name', 'family', 'val_RMSE', 'val_RMSE_peaks', 'val_bias_peaks', 'test_RMSE']].head(5).to_string(index=False))
        
    full_audit = pd.concat(all_metrics, ignore_index=True)
    full_audit.to_csv(os.path.join(ENS2_DIR, 'audit_models_table.csv'), index=False)
    print(f"\nAuditoría guardada en {os.path.join(ENS2_DIR, 'audit_models_table.csv')}")

if __name__ == '__main__':
    run_audit()
