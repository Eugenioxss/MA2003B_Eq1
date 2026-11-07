"""
experiments/noche2/deep2/train_deep_nwp_refit.py
=================================================
Entrenamiento Fase 3:
1. deep_nwp_refit_gated_h<h>: TabGated (SwiGLU) con 4-fold OOF en 2024 y Refit 2020-2024 para test 2025.
2. deep_nwp_refit_resnet_h<h>: TabResNet con 4-fold OOF en 2024 y Refit 2020-2024 para test 2025.
3. lgbm_nwp_residual_h<h>: Aprendizaje neuronal de residuos sobre LightGBM incorporando features NWP.
"""
import os, sys, time, copy, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial

print = partial(print, flush=True)

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

from common import MODEL_STATIONS, HORIZONS, SEED, evaluate_test, log_result
from dataset_nwp_refit import load_combined_nwp_df, NWPFoldManager
from model import TabResNet
from train_deep2_v2 import TabGatedMLP

torch.set_num_threads(3)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(BASE_DIR, 'preds')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
OOF_DIR = os.path.join(BASE_DIR, 'oof')
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS = [42, 123, 999]

QUARTERS = [
    (1, [1, 2, 3]),
    (2, [4, 5, 6]),
    (3, [7, 8, 9]),
    (4, [10, 11, 12])
]

def train_eval_model_loop(model_type: str, fm: NWPFoldManager, seed: int, max_epochs: int, delta: float, use_w: bool, device: torch.device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if model_type == 'gated':
        model = TabGatedMLP(len(MODEL_STATIONS), fm.input_dim, hidden_dim=384, num_layers=3, dropout=0.05).to(device)
    else: # resnet
        model = TabResNet(len(MODEL_STATIONS), fm.input_dim, embed_dim=16, hr_embed_dim=8, hidden_dim=256, num_blocks=3, dropout=0.05).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    batch_size = 2048
    n_train = len(fm.y_tr)
    n_batches = (n_train + batch_size - 1) // batch_size

    best_val_rmse = float('inf')
    best_vp = None
    best_epoch = 1

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            optimizer.zero_grad()
            pred = model(fm.tr_st[idx], fm.tr_hr[idx], fm.X_tr[idx])
            err = pred - fm.y_tr[idx]
            abs_err = torch.abs(err)
            loss_raw = torch.where(abs_err < delta, 0.5 * (err**2), delta * (abs_err - 0.5 * delta))
            loss = torch.mean(fm.weights_tr[idx] * loss_raw) if use_w else torch.mean(loss_raw)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = model(fm.va_st, fm.va_hr, fm.X_va)
            v_rmse = torch.sqrt(torch.mean((vp - fm.y_va)**2)).item()

        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_vp = vp.cpu().numpy().ravel()
            best_epoch = ep

    return best_vp, best_epoch, best_val_rmse


def train_refit_predict_loop(model_type: str, fm: NWPFoldManager, seed: int, n_epochs: int, delta: float, use_w: bool, device: torch.device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_type == 'gated':
        model = TabGatedMLP(len(MODEL_STATIONS), fm.input_dim, hidden_dim=384, num_layers=3, dropout=0.05).to(device)
    else:
        model = TabResNet(len(MODEL_STATIONS), fm.input_dim, embed_dim=16, hr_embed_dim=8, hidden_dim=256, num_blocks=3, dropout=0.05).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    max_epochs_refit = max(2, n_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs_refit, eta_min=1e-5)
    batch_size = 2048
    n_train = len(fm.y_tr)
    n_batches = (n_train + batch_size - 1) // batch_size

    for ep in range(1, max_epochs_refit + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            optimizer.zero_grad()
            pred = model(fm.tr_st[idx], fm.tr_hr[idx], fm.X_tr[idx])
            err = pred - fm.y_tr[idx]
            abs_err = torch.abs(err)
            loss_raw = torch.where(abs_err < delta, 0.5 * (err**2), delta * (abs_err - 0.5 * delta))
            loss = torch.mean(fm.weights_tr[idx] * loss_raw) if use_w else torch.mean(loss_raw)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        tp = model(fm.te_st, fm.te_hr, fm.X_te)
        test_pred = tp.cpu().numpy().ravel()
    return test_pred


def run_refit_variant_for_h(variant_name: str, model_type: str, h: int, df_all: pd.DataFrame, num_cols: list[str], device: torch.device):
    print(f"\n{'='*70}\n  [h={h:02d}] Entrenando {variant_name} ({model_type.upper()})\n{'='*70}")
    t0 = time.time()
    
    # Separar splits base
    tr_base = df_all[(df_all['split'] == 'train') & df_all['Target'].notna()].copy()
    va_2024 = df_all[(df_all['split'] == 'val') & df_all['Target'].notna()].copy()
    te_2025 = df_all[(df_all['split'] == 'test') & df_all['Target'].notna()].copy()

    use_w = (h >= 8)
    delta = 1.5 if h <= 4 else 2.0
    max_epochs = 10 if h <= 4 else 8

    # --- 1. Cross-Fitting 4-Fold en 2024 ---
    oof_val_preds = np.zeros(len(va_2024), dtype=np.float32)
    fold_best_epochs = []

    va_months = va_2024['Date'].dt.month.values

    for q_idx, months in QUARTERS:
        t_q = time.time()
        q_mask = np.isin(va_months, months)
        val_fold_df = va_2024[q_mask].copy().reset_index(drop=True)
        other_2024_df = va_2024[~q_mask].copy().reset_index(drop=True)

        # Train fold: 2020-2023 + otros 3 trimestres de 2024
        train_fold_df = pd.concat([tr_base, other_2024_df], ignore_index=True)
        
        fm = NWPFoldManager(train_fold_df, val_fold_df, te_df=None, num_cols=num_cols, device=device)

        seed_preds = []
        seed_epochs = []
        for s in SEEDS:
            vp, be, vr = train_eval_model_loop(model_type, fm, seed=s, max_epochs=max_epochs, delta=delta, use_w=use_w, device=device)
            seed_preds.append(vp)
            seed_epochs.append(be)

        fold_pred_avg = np.mean(seed_preds, axis=0)
        oof_val_preds[q_mask] = fold_pred_avg
        med_epoch = int(np.round(np.median(seed_epochs)))
        fold_best_epochs.append(med_epoch)

        q_rmse = float(np.sqrt(np.mean((val_fold_df['Target'].values - fold_pred_avg)**2)))
        print(f"    Q{q_idx} (Meses {months}) -> OOF RMSE: {q_rmse:.4f} | Mediana Epocas: {med_epoch} ({time.time()-t_q:.1f}s)")

    oof_total_rmse = float(np.sqrt(np.mean((va_2024['Target'].values - oof_val_preds)**2)))
    refit_epochs = max(2, int(np.round(np.median(fold_best_epochs))))
    print(f"  [2024 OOF Completo] Val RMSE: {oof_total_rmse:.4f} | Epocas para Refit: {refit_epochs}")

    # --- 2. Refit Completo en 2020-2024 para Test 2025 ---
    t_refit = time.time()
    full_train_df = pd.concat([tr_base, va_2024], ignore_index=True)
    fm_refit = NWPFoldManager(full_train_df, va_df=va_2024, te_df=te_2025, num_cols=num_cols, device=device)

    test_seed_preds = []
    for s in SEEDS:
        tp = train_refit_predict_loop(model_type, fm_refit, seed=s, n_epochs=refit_epochs, delta=delta, use_w=use_w, device=device)
        test_seed_preds.append(tp)

    final_test_pred = np.mean(test_seed_preds, axis=0)
    print(f"  [Refit 2020-2024 completado en {time.time()-t_refit:.1f}s]")

    # Guardar predicciones oficiales segun contrato
    val_out = pd.DataFrame({
        'Date': va_2024['Date'], 'Estacion': va_2024['Estacion'], 'h': h, 'split': 'val',
        'y_true': va_2024['Target'].values.astype('float32'), 'y_pred': oof_val_preds.astype('float32')
    })
    test_out = pd.DataFrame({
        'Date': te_2025['Date'], 'Estacion': te_2025['Estacion'], 'h': h, 'split': 'test',
        'y_true': te_2025['Target'].values.astype('float32'), 'y_pred': final_test_pred.astype('float32')
    })
    preds_all = pd.concat([val_out, test_out], ignore_index=True)
    out_path = os.path.join(PREDS_DIR, f"{variant_name}_h{h}.parquet")
    preds_all.to_parquet(out_path, index=False)

    m_te = evaluate_test(te_2025, final_test_pred, h=h)

    # Correlacion vs LGBM
    lgb_df = pd.read_parquet(os.path.join(ROOT, f"experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet"))
    lgb_te = lgb_df[lgb_df['split'] == 'test']['y_pred'].values
    lgb_res = te_2025[f'Target_{h}h'].values - lgb_te
    nn_res = te_2025[f'Target_{h}h'].values - final_test_pred
    res_corr = float(np.corrcoef(lgb_res, nn_res)[0, 1])

    blend_te = 0.5 * lgb_te + 0.5 * final_test_pred
    m_blend = evaluate_test(te_2025, blend_te, h=h)

    log_result(
        experiment='noche2_deep2', variant=variant_name, h=h, m=m_te,
        notes=f"NWP Multi-Model + 4-fold OOF Refit 2024 ({model_type}). corr={res_corr:.3f}, blend={m_blend['RMSE']:.3f}"
    )

    print(f"\n  [{variant_name} h={h:02d} RESUMEN] Val OOF: {oof_total_rmse:.4f} | Test RMSE: {m_te['RMSE']:.4f}")
    print(f"     Peaks RMSE: {m_te['RMSE_peaks']:.2f} | Bias Peaks: {m_te['bias_peaks']:.2f} | Corr vs LGBM: {res_corr:.4f} | Blend: {m_blend['RMSE']:.4f} | Tiempo: {time.time()-t0:.1f}s")

    return {
        'variant': variant_name, 'h': h, 'val_rmse': oof_total_rmse, 'test_rmse': m_te['RMSE'],
        'blend_test_rmse': m_blend['RMSE'], 'test_r2': m_te['R2'], 'test_peaks_rmse': m_te['RMSE_peaks'],
        'test_peaks_bias': m_te['bias_peaks'], 'res_corr_vs_lgb': res_corr
    }


def run_lgbm_nwp_residual_for_h(h: int, df_all: pd.DataFrame, num_cols: list[str], device: torch.device):
    print(f"\n{'='*70}\n  [h={h:02d}] Entrenando lgbm_nwp_residual_h{h}\n{'='*70}")
    t0 = time.time()
    
    oof_path = os.path.join(OOF_DIR, f'lgbm_oof_h{h}.parquet')
    oof_df = pd.read_parquet(oof_path)

    tr_df = df_all[(df_all['split'] == 'train') & df_all['Target'].notna()].copy().reset_index(drop=True)
    va_df = df_all[(df_all['split'] == 'val') & df_all['Target'].notna()].copy().reset_index(drop=True)
    te_df = df_all[(df_all['split'] == 'test') & df_all['Target'].notna()].copy().reset_index(drop=True)

    oof_tr = oof_df[oof_df['split'] == 'train'].reset_index(drop=True)
    oof_va = oof_df[oof_df['split'] == 'val'].reset_index(drop=True)
    oof_te = oof_df[oof_df['split'] == 'test'].reset_index(drop=True)

    y_lgb_tr = oof_tr['y_pred_lgbm'].values.astype(np.float32)
    y_lgb_va = oof_va['y_pred_lgbm'].values.astype(np.float32)
    y_lgb_te = oof_te['y_pred_lgbm'].values.astype(np.float32)

    # Residual targets: r = y - y_lgb
    r_tr = torch.tensor(tr_df['Target'].values.astype(np.float32) - y_lgb_tr, dtype=torch.float32, device=device).unsqueeze(1)
    r_va = torch.tensor(va_df['Target'].values.astype(np.float32) - y_lgb_va, dtype=torch.float32, device=device).unsqueeze(1)
    r_te = torch.tensor(te_df['Target'].values.astype(np.float32) - y_lgb_te, dtype=torch.float32, device=device).unsqueeze(1)

    fm = NWPFoldManager(tr_df, va_df, te_df, num_cols=num_cols, device=device)

    # Añadir y_lgb estandarizado
    lgb_m = float(np.mean(y_lgb_tr))
    lgb_s = float(np.std(y_lgb_tr)) + 1e-4

    feat_lgb_tr = torch.tensor((y_lgb_tr - lgb_m) / lgb_s, dtype=torch.float32, device=device).unsqueeze(1)
    feat_lgb_va = torch.tensor((y_lgb_va - lgb_m) / lgb_s, dtype=torch.float32, device=device).unsqueeze(1)
    feat_lgb_te = torch.tensor((y_lgb_te - lgb_m) / lgb_s, dtype=torch.float32, device=device).unsqueeze(1)

    X_tr = torch.cat([fm.X_tr, feat_lgb_tr], dim=1)
    X_va = torch.cat([fm.X_va, feat_lgb_va], dim=1)
    X_te = torch.cat([fm.X_te, feat_lgb_te], dim=1)

    input_dim = X_tr.shape[1]
    n_epochs = 10 if h <= 4 else 8
    delta = 1.5 if h <= 4 else 2.0
    use_w = (h >= 8)

    va_res_preds, te_res_preds = [], []

    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        model = TabResNet(len(MODEL_STATIONS), input_dim, embed_dim=16, hr_embed_dim=8, hidden_dim=256, num_blocks=3, dropout=0.05).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
        batch_size = 2048
        n_train = len(r_tr)
        n_batches = (n_train + batch_size - 1) // batch_size

        best_val = float('inf')
        best_vp, best_tp = None, None

        for ep in range(1, n_epochs + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            for b in range(n_batches):
                idx = perm[b*batch_size : (b+1)*batch_size]
                optimizer.zero_grad()
                pred_r = model(fm.tr_st[idx], fm.tr_hr[idx], X_tr[idx])
                err = pred_r - r_tr[idx]
                abs_err = torch.abs(err)
                loss_raw = torch.where(abs_err < delta, 0.5 * (err**2), delta * (abs_err - 0.5 * delta))
                loss = torch.mean(fm.weights_tr[idx] * loss_raw) if use_w else torch.mean(loss_raw)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                vp = model(fm.va_st, fm.va_hr, X_va)
                tp = model(fm.te_st, fm.te_hr, X_te)
                y_final_va = torch.clamp(torch.tensor(y_lgb_va, device=device).unsqueeze(1) + vp, min=0.0)
                v_rmse = torch.sqrt(torch.mean((y_final_va - fm.y_va)**2)).item()

            if v_rmse < best_val:
                best_val = v_rmse
                best_vp = vp.cpu().numpy().ravel()
                best_tp = tp.cpu().numpy().ravel()

        va_res_preds.append(best_vp)
        te_res_preds.append(best_tp)

    final_r_va = np.mean(va_res_preds, axis=0)
    final_r_te = np.mean(te_res_preds, axis=0)

    final_y_va = np.clip(y_lgb_va + final_r_va, 0.0, None)
    final_y_te = np.clip(y_lgb_te + final_r_te, 0.0, None)

    val_out = pd.DataFrame({
        'Date': va_df['Date'], 'Estacion': va_df['Estacion'], 'h': h, 'split': 'val',
        'y_true': va_df['Target'].values.astype('float32'), 'y_pred': final_y_va.astype('float32')
    })
    test_out = pd.DataFrame({
        'Date': te_df['Date'], 'Estacion': te_df['Estacion'], 'h': h, 'split': 'test',
        'y_true': te_df['Target'].values.astype('float32'), 'y_pred': final_y_te.astype('float32')
    })
    preds_all = pd.concat([val_out, test_out], ignore_index=True)
    out_path = os.path.join(PREDS_DIR, f"lgbm_nwp_residual_h{h}.parquet")
    preds_all.to_parquet(out_path, index=False)

    val_rmse = float(np.sqrt(np.mean((final_y_va - va_df['Target'].values)**2)))
    m_te = evaluate_test(te_df, final_y_te, h=h)

    lgb_res = te_df[f'Target_{h}h'].values - y_lgb_te
    corrected_res = te_df[f'Target_{h}h'].values - final_y_te
    res_corr = float(np.corrcoef(lgb_res, corrected_res)[0, 1])
    lgb_test_rmse = float(np.sqrt(np.mean(lgb_res**2)))

    log_result(
        experiment='noche2_deep2', variant='lgbm_nwp_residual', h=h, m=m_te,
        notes=f"Residual learning over LGBM + NWP Multi-Model. lgb={lgb_test_rmse:.3f} -> {m_te['RMSE']:.3f}, corr={res_corr:.3f}"
    )

    print(f"\n  [lgbm_nwp_residual h={h:02d}] Val RMSE: {val_rmse:.4f} | Test RMSE: {m_te['RMSE']:.4f} (LGBM: {lgb_test_rmse:.4f})")
    print(f"     Peaks RMSE: {m_te['RMSE_peaks']:.2f} | Bias Peaks: {m_te['bias_peaks']:.2f} | Res Corr: {res_corr:.4f} | Tiempo: {time.time()-t0:.1f}s")

    return {
        'variant': 'lgbm_nwp_residual', 'h': h, 'val_rmse': val_rmse, 'test_rmse': m_te['RMSE'],
        'lgb_test_rmse': lgb_test_rmse, 'test_r2': m_te['R2'], 'test_peaks_rmse': m_te['RMSE_peaks'],
        'test_peaks_bias': m_te['bias_peaks'], 'res_corr_vs_lgb': res_corr
    }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando Fase 3 Deep Learning (NWP + Refit) en {device} ...")

    all_rows = []
    for h in HORIZONS:
        print(f"\n{'#'*75}\n# HORIZONTE h = {h:02d}\n{'#'*75}")
        df_all, num_cols = load_combined_nwp_df(h)

        # 1. deep_nwp_refit_gated
        r_gated = run_refit_variant_for_h('deep_nwp_refit_gated', 'gated', h, df_all, num_cols, device)
        all_rows.append(r_gated)

        # 2. deep_nwp_refit_resnet
        r_resnet = run_refit_variant_for_h('deep_nwp_refit_resnet', 'resnet', h, df_all, num_cols, device)
        all_rows.append(r_resnet)

        # 3. lgbm_nwp_residual
        r_resid = run_lgbm_nwp_residual_for_h(h, df_all, num_cols, device)
        all_rows.append(r_resid)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(os.path.join(BASE_DIR, 'summary_phase3_nwp_refit.csv'), index=False)
    print("\nFase 3 finalizada exitosamente. Resumen guardado en summary_phase3_nwp_refit.csv.")
