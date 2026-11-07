"""
experiments/noche2/deep2/train_deep_imput2_resid_refit.py
=========================================================
Entrenamiento del ensamble neuronal sobre el pipeline de características de imput2:
- Base química imputada causalmente (audit_imput)
- Memoria diaria D-1 y dinámicas de cuenca regional
- Consenso meteorológico multi-modelo NWP + MOS (ECMWF, GFS, ICON, CAPE, VPD)
- Aprendizaje del residual de persistencia: Delta = y - O3(t0)
- Ensamble 50/50 TabResNet + TabGated (SwiGLU)
- Esquema 4-fold OOF en 2024 y Refit completo 2020-2024 para test 2025
"""
import os, sys, time, copy, gc
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

from common import (
    load_panel, evaluate_test, log_result, MODEL_STATIONS, HORIZONS, SEED
)
from planA_quickwins.features_planA import build_planA_features
from experiments.noche2.imput2.features_pipeline import (
    compute_missing_features, compute_yesterday_stats, compute_regional_trends
)
from experiments.noche2.imput2.train_phase2_refit import build_horizon_dataset, IMPUTED_PANEL_PATH
from model import TabResNet
from train_deep2_v2 import TabGatedMLP

torch.set_num_threads(3)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(BASE_DIR, 'preds')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

QUARTERS = [
    (1, [1, 2, 3]),
    (2, [4, 5, 6]),
    (3, [7, 8, 9]),
    (4, [10, 11, 12])
]


class Imput2FoldManager:
    """Gestiona tensores en memoria GPU para features imput2."""
    def __init__(self, tr_df, va_df, te_df, num_cols, o3_col, delta_col, device):
        self.device = device
        st_map = {s: i for i, s in enumerate(MODEL_STATIONS)}

        # Embeddings
        self.tr_st = torch.tensor([st_map[s] for s in tr_df['Estacion']], dtype=torch.long, device=device)
        self.va_st = torch.tensor([st_map[s] for s in va_df['Estacion']], dtype=torch.long, device=device)
        self.te_st = torch.tensor([st_map[s] for s in te_df['Estacion']], dtype=torch.long, device=device) if te_df is not None else None

        self.tr_hr = torch.tensor(tr_df['Date'].dt.hour.values, dtype=torch.long, device=device)
        self.va_hr = torch.tensor(va_df['Date'].dt.hour.values, dtype=torch.long, device=device)
        self.te_hr = torch.tensor(te_df['Date'].dt.hour.values, dtype=torch.long, device=device) if te_df is not None else None

        # Residual target Delta = y - O3(t0)
        self.delta_tr = torch.tensor(tr_df[delta_col].values.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)
        self.delta_va = torch.tensor(va_df[delta_col].values.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)
        self.delta_te = torch.tensor(te_df[delta_col].values.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1) if te_df is not None else None

        # O3(t0) base
        self.o3_va = va_df[o3_col].values.astype(np.float32)
        self.o3_te = te_df[o3_col].values.astype(np.float32) if te_df is not None else None

        # Pesos opcionales de picos sobre target real y
        y_tr_vals = tr_df['Target'].values.astype(np.float32)
        p90 = np.percentile(y_tr_vals, 90.0)
        self.weights_tr = torch.tensor(1.0 + (y_tr_vals / p90)**2, dtype=torch.float32, device=device).unsqueeze(1)

        # Preprocesamiento ajustado en train
        X_tr_raw = tr_df[num_cols].values.astype(np.float32)
        X_va_raw = va_df[num_cols].values.astype(np.float32)
        X_te_raw = te_df[num_cols].values.astype(np.float32) if te_df is not None else None

        means = np.nanmean(X_tr_raw, axis=0)
        stds = np.nanstd(X_tr_raw, axis=0)
        stds[stds < 1e-4] = 1.0

        null_in_tr = np.isnan(X_tr_raw).any(axis=0)
        null_col_indices = np.where(null_in_tr)[0]

        def transform(X_raw):
            if X_raw is None:
                return None
            mask = np.isnan(X_raw[:, null_col_indices]).astype(np.float32)
            X_norm = (X_raw - means) / stds
            X_norm = np.nan_to_num(X_norm, nan=0.0)
            X_norm = np.clip(X_norm, -10.0, 10.0)
            return np.hstack([X_norm, mask])

        self.X_tr = torch.tensor(transform(X_tr_raw), dtype=torch.float32, device=device)
        self.X_va = torch.tensor(transform(X_va_raw), dtype=torch.float32, device=device)
        self.X_te = torch.tensor(transform(X_te_raw), dtype=torch.float32, device=device) if X_te_raw is not None else None

        self.input_dim = self.X_tr.shape[1]


def train_eval_delta_model(model_type, fm, seed, max_epochs, delta_loss, use_w, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_type == 'gated':
        model = TabGatedMLP(len(MODEL_STATIONS), fm.input_dim, hidden_dim=384, num_layers=3, dropout=0.05).to(device)
    else:
        model = TabResNet(len(MODEL_STATIONS), fm.input_dim, embed_dim=16, hr_embed_dim=8, hidden_dim=256, num_blocks=3, dropout=0.05).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    batch_size = 2048
    n_train = len(fm.delta_tr)
    n_batches = (n_train + batch_size - 1) // batch_size

    best_val_rmse = float('inf')
    best_vp = None
    best_ep = 1

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            optimizer.zero_grad()
            pred_delta = model(fm.tr_st[idx], fm.tr_hr[idx], fm.X_tr[idx])
            err = pred_delta - fm.delta_tr[idx]
            abs_err = torch.abs(err)
            loss_raw = torch.where(abs_err < delta_loss, 0.5 * (err**2), delta_loss * (abs_err - 0.5 * delta_loss))
            loss = torch.mean(fm.weights_tr[idx] * loss_raw) if use_w else torch.mean(loss_raw)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = model(fm.va_st, fm.va_hr, fm.X_va)
            # Evaluar RMSE en espacio real y = ReLU(O3(t0) + delta)
            y_recon = torch.clamp(torch.tensor(fm.o3_va, device=device).unsqueeze(1) + vp, min=0.0)
            y_true_va = torch.tensor(fm.o3_va, device=device).unsqueeze(1) + fm.delta_va
            v_rmse = torch.sqrt(torch.mean((y_recon - y_true_va)**2)).item()

        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_vp = vp.cpu().numpy().ravel()
            best_ep = ep

    return best_vp, best_ep, best_val_rmse


def train_refit_delta_model(model_type, fm, seed, n_epochs, delta_loss, use_w, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_type == 'gated':
        model = TabGatedMLP(len(MODEL_STATIONS), fm.input_dim, hidden_dim=384, num_layers=3, dropout=0.05).to(device)
    else:
        model = TabResNet(len(MODEL_STATIONS), fm.input_dim, embed_dim=16, hr_embed_dim=8, hidden_dim=256, num_blocks=3, dropout=0.05).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    refit_eps = max(2, n_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=refit_eps, eta_min=1e-5)
    batch_size = 2048
    n_train = len(fm.delta_tr)
    n_batches = (n_train + batch_size - 1) // batch_size

    for ep in range(1, refit_eps + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            optimizer.zero_grad()
            pred_delta = model(fm.tr_st[idx], fm.tr_hr[idx], fm.X_tr[idx])
            err = pred_delta - fm.delta_tr[idx]
            abs_err = torch.abs(err)
            loss_raw = torch.where(abs_err < delta_loss, 0.5 * (err**2), delta_loss * (abs_err - 0.5 * delta_loss))
            loss = torch.mean(fm.weights_tr[idx] * loss_raw) if use_w else torch.mean(loss_raw)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        tp = model(fm.te_st, fm.te_hr, fm.X_te)
        test_delta = tp.cpu().numpy().ravel()
    return test_delta


def run_deep_imput2_resid_refit(device):
    print("\n" + "="*80)
    print("  INICIANDO: deep_imput2_resid_refit (Ensamble Neuronal Imput2 Residual)")
    print(f"  Dispositivo: {device}")
    print("="*80)
    t0_all = time.time()

    # Precomputar base estatica
    print("\n1. Cargando paneles y precomputando base estatica imputada causalmente...")
    raw_panel = load_panel()
    imputed_panel = pd.read_parquet(IMPUTED_PANEL_PATH)

    df_planA_imp, chem_imp_cols_raw = build_planA_features(imputed_panel, variant='V4')
    target_cols_to_drop = [c for c in df_planA_imp.columns if c.startswith('Target')]
    df_planA_imp.drop(columns=target_cols_to_drop, inplace=True)
    chem_imp_cols = [c for c in chem_imp_cols_raw if c != 'Estacion' and not c.startswith('Target')]

    yesterday_df = compute_yesterday_stats(imputed_panel)
    regional_df = compute_regional_trends(imputed_panel)
    missing_df = compute_missing_features(raw_panel)

    base_static = pd.merge(df_planA_imp, yesterday_df, on=['Estacion', 'Date'], how='left')
    base_static = pd.merge(base_static, regional_df, on=['Estacion', 'Date'], how='left')
    base_static = pd.merge(base_static, missing_df, on=['Estacion', 'Date'], how='left')

    yesterday_cols = [c for c in yesterday_df.columns if c not in ['Date', 'Estacion']]
    regional_cols = [c for c in regional_df.columns if c not in ['Date', 'Estacion']]
    missing_cols = [c for c in missing_df.columns if c not in ['Date', 'Estacion']]

    del imputed_panel, raw_panel, df_planA_imp, yesterday_df, regional_df
    gc.collect()

    summary_rows = []

    for h in HORIZONS:
        t0_h = time.time()
        out_path = os.path.join(PREDS_DIR, f"deep_imput2_resid_refit_h{h}.parquet")
        if os.path.exists(out_path):
            print(f"\n  [SKIP] {out_path} ya existe; cargando métricas existentes para h={h:02d}...")
            ex_df = pd.read_parquet(out_path)
            va_ex = ex_df[ex_df['split'] == 'val']
            te_ex = ex_df[ex_df['split'] == 'test'].copy()
            te_ex[f'Target_{h}h'] = te_ex['y_true'].values
            te_ex['Target'] = te_ex['y_true'].values
            oof_val_rmse = float(np.sqrt(np.mean((va_ex['y_true'].values - va_ex['y_pred'].values)**2)))
            final_test_y = te_ex['y_pred'].values
            m_te = evaluate_test(te_ex, final_test_y, h=h)
            lgb_df = pd.read_parquet(os.path.join(ROOT, f"experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet"))
            lgb_te = lgb_df[lgb_df['split'] == 'test']['y_pred'].values
            lgb_res = te_ex['y_true'].values - lgb_te
            nn_res = te_ex['y_true'].values - final_test_y
            res_corr = float(np.corrcoef(lgb_res, nn_res)[0, 1])
            blend_te = 0.5 * lgb_te + 0.5 * final_test_y
            m_blend = evaluate_test(te_ex, blend_te, h=h)
            summary_rows.append({
                'h': h, 'val_rmse': oof_val_rmse, 'test_rmse': m_te['RMSE'], 'blend_test_rmse': m_blend['RMSE'],
                'test_r2': m_te['R2'], 'test_peaks_rmse': m_te['RMSE_peaks'], 'test_peaks_bias': m_te['bias_peaks'],
                'res_corr_vs_lgb': res_corr
            })
            continue

        print(f"\n{'='*75}\n  HORIZONTE h={h:02d}\n{'='*75}")
        df_h, feature_cols, honest_cols, pp_cols = build_horizon_dataset(
            h=h,
            base_static=base_static,
            chem_imp_cols=chem_imp_cols,
            yesterday_cols=yesterday_cols,
            regional_cols=regional_cols,
            missing_cols=missing_cols
        )

        o3_col = 'O3_lag_0' if 'O3_lag_0' in df_h.columns else 'O3'
        delta_col = 'Delta_O3'
        df_h[delta_col] = (df_h['Target'] - df_h[o3_col]).astype('float32')

        # Features numericas para la red
        num_cols = [c for c in feature_cols if c != 'Estacion']

        # Limpiar centinelas
        for c in num_cols:
            bad = (df_h[c] < -500) | (df_h[c] > 9000)
            if bad.any():
                df_h.loc[bad, c] = np.nan

        tr_base = df_h[(df_h['split'] == 'train') & df_h['Target'].notna()].copy().reset_index(drop=True)
        va_2024 = df_h[(df_h['split'] == 'val') & df_h['Target'].notna()].copy().reset_index(drop=True)
        te_2025 = df_h[(df_h['split'] == 'test') & df_h['Target'].notna()].copy().reset_index(drop=True)

        use_w = (h >= 8)
        delta_loss = 1.5 if h <= 4 else 2.0
        max_eps = 10 if h <= 4 else 8

        # --- 1. 4-Fold Cross-Fitting en 2024 ---
        oof_recon_y = np.zeros(len(va_2024), dtype=np.float32)
        best_eps_list = []
        va_months = va_2024['Date'].dt.month.values

        for q_idx, months in QUARTERS:
            t_q = time.time()
            q_mask = np.isin(va_months, months)
            val_fold_df = va_2024[q_mask].copy().reset_index(drop=True)
            other_2024_df = va_2024[~q_mask].copy().reset_index(drop=True)

            train_fold_df = pd.concat([tr_base, other_2024_df], ignore_index=True)
            fm = Imput2FoldManager(train_fold_df, val_fold_df, te_df=None, num_cols=num_cols, o3_col=o3_col, delta_col=delta_col, device=device)

            # Entrenar TabResNet y TabGated
            resnet_deltas, gated_deltas = [], []
            for s in [42, 123]:
                rd, be_r, _ = train_eval_delta_model('resnet', fm, s, max_eps, delta_loss, use_w, device)
                gd, be_g, _ = train_eval_delta_model('gated', fm, s, max_eps, delta_loss, use_w, device)
                resnet_deltas.append(rd)
                gated_deltas.append(gd)
                best_eps_list.extend([be_r, be_g])

            # Ensamble 50/50 de deltas
            avg_delta = 0.5 * np.mean(resnet_deltas, axis=0) + 0.5 * np.mean(gated_deltas, axis=0)
            recon_y = np.maximum(0.0, val_fold_df[o3_col].values + avg_delta)
            oof_recon_y[q_mask] = recon_y

            q_rmse = float(np.sqrt(np.mean((val_fold_df['Target'].values - recon_y)**2)))
            print(f"    Q{q_idx} (Meses {months}) -> OOF RMSE: {q_rmse:.4f} ({time.time()-t_q:.1f}s)")

        oof_val_rmse = float(np.sqrt(np.mean((va_2024['Target'].values - oof_recon_y)**2)))
        med_refit_eps = max(2, int(np.round(np.median(best_eps_list))))
        print(f"  [2024 OOF Completo] Val RMSE: {oof_val_rmse:.4f} | Mediana Epocas Refit: {med_refit_eps}")

        # --- 2. Refit Completo 2020-2024 para Test 2025 ---
        t_refit = time.time()
        full_train_df = pd.concat([tr_base, va_2024], ignore_index=True)
        fm_refit = Imput2FoldManager(full_train_df, va_df=va_2024, te_df=te_2025, num_cols=num_cols, o3_col=o3_col, delta_col=delta_col, device=device)

        test_resnet_deltas, test_gated_deltas = [], []
        for s in [42, 123]:
            trd = train_refit_delta_model('resnet', fm_refit, s, med_refit_eps, delta_loss, use_w, device)
            tgd = train_refit_delta_model('gated', fm_refit, s, med_refit_eps, delta_loss, use_w, device)
            test_resnet_deltas.append(trd)
            test_gated_deltas.append(tgd)

        avg_test_delta = 0.5 * np.mean(test_resnet_deltas, axis=0) + 0.5 * np.mean(test_gated_deltas, axis=0)
        final_test_y = np.maximum(0.0, te_2025[o3_col].values + avg_test_delta)
        print(f"  [Refit completado en {time.time()-t_refit:.1f}s]")

        # Guardar parquet oficial segun contrato
        val_out = pd.DataFrame({
            'Date': va_2024['Date'], 'Estacion': va_2024['Estacion'], 'h': h, 'split': 'val',
            'y_true': va_2024['Target'].values.astype('float32'), 'y_pred': oof_recon_y.astype('float32')
        })
        test_out = pd.DataFrame({
            'Date': te_2025['Date'], 'Estacion': te_2025['Estacion'], 'h': h, 'split': 'test',
            'y_true': te_2025['Target'].values.astype('float32'), 'y_pred': final_test_y.astype('float32')
        })
        preds_all = pd.concat([val_out, test_out], ignore_index=True)
        out_path = os.path.join(PREDS_DIR, f"deep_imput2_resid_refit_h{h}.parquet")
        preds_all.to_parquet(out_path, index=False)

        m_te = evaluate_test(te_2025, final_test_y, h=h)

        # Correlacion vs LGBM
        lgb_df = pd.read_parquet(os.path.join(ROOT, f"experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet"))
        lgb_te = lgb_df[lgb_df['split'] == 'test']['y_pred'].values
        lgb_res = te_2025[f'Target_{h}h'].values - lgb_te
        nn_res = te_2025[f'Target_{h}h'].values - final_test_y
        res_corr = float(np.corrcoef(lgb_res, nn_res)[0, 1])

        blend_te = 0.5 * lgb_te + 0.5 * final_test_y
        m_blend = evaluate_test(te_2025, blend_te, h=h)

        log_result(
            experiment='noche2_deep2', variant='deep_imput2_resid_refit', h=h, m=m_te,
            notes=f"Deep Imput2 Persistence Residual Refit (TabResNet+TabGated). corr={res_corr:.3f}, blend={m_blend['RMSE']:.3f}"
        )

        print(f"\n  [deep_imput2_resid_refit h={h:02d} RESUMEN] Val OOF: {oof_val_rmse:.4f} | Test RMSE: {m_te['RMSE']:.4f}")
        print(f"     Peaks RMSE: {m_te['RMSE_peaks']:.2f} | Bias Peaks: {m_te['bias_peaks']:.2f} | Corr vs LGBM: {res_corr:.4f} | Blend: {m_blend['RMSE']:.4f} | Tiempo: {time.time()-t0_h:.1f}s")

        summary_rows.append({
            'h': h, 'val_rmse': oof_val_rmse, 'test_rmse': m_te['RMSE'], 'blend_test_rmse': m_blend['RMSE'],
            'test_r2': m_te['R2'], 'test_peaks_rmse': m_te['RMSE_peaks'], 'test_peaks_bias': m_te['bias_peaks'],
            'res_corr_vs_lgb': res_corr
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(os.path.join(BASE_DIR, 'summary_deep_imput2_resid_refit.csv'), index=False)
    print(f"\ndeep_imput2_resid_refit finalizado exitosamente en {time.time()-t0_all:.1f}s.")
    return sum_df


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_deep_imput2_resid_refit(device)
