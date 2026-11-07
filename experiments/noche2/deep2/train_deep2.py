"""
experiments/noche2/deep2/train_deep2.py
=======================================
Pipeline maestro de entrenamiento y evaluación para modelos neuronales tabulares en noche2.
Variantes soportadas:
  1. tab_resnet_honest: Tabular ResNet con Huber loss + NWP noise + ensamble de 3 semillas.
  2. tab_resnet_quantile: Tabular ResNet con pérdida monotónica pinball para cuantiles [q10, q90].

Uso:
  python train_deep2.py --variant tab_resnet_honest
  python train_deep2.py --variant tab_resnet_quantile
  python train_deep2.py --variant all
"""
import os, sys, time, argparse, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial

# Redirigir print para asegurar flush
print = partial(print, flush=True)

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

from common import (
    MODEL_STATIONS, HORIZONS, SEED, evaluate_test, log_result, metrics
)
from dataset import TabularDatasetManager
from model import TabResNet, TabResNetQuantile, pinball_loss

# Restricción estricta de cómputo
torch.set_num_threads(3)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(BASE_DIR, 'preds')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS = [42, 123, 999]

def train_single_seed_honest(
    dm: TabularDatasetManager,
    h: int,
    seed: int,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    delta: float = 1.5,
    use_weights: bool = False
) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TabResNet(
        num_stations=len(MODEL_STATIONS),
        num_features=dm.input_dim,
        embed_dim=16,
        hr_embed_dim=8,
        hidden_dim=256,
        num_blocks=3,
        dropout=0.05
    ).to(dm.device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_train = len(dm.y_tr)
    n_batches = (n_train + batch_size - 1) // batch_size

    best_val_rmse = float('inf')
    best_va_preds = None
    best_te_preds = None
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=dm.device)
        loss_acc = 0.0
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            optimizer.zero_grad()
            pred = model(dm.tr_st[idx], dm.tr_hr[idx], dm.X_tr[idx])
            err = pred - dm.y_tr[idx]
            abs_err = torch.abs(err)
            loss_raw = torch.where(abs_err < delta, 0.5 * (err**2), delta * (abs_err - 0.5 * delta))
            if use_weights:
                loss = torch.mean(dm.weights_tr[idx] * loss_raw)
            else:
                loss = torch.mean(loss_raw)
            loss.backward()
            optimizer.step()
            loss_acc += loss.item() * len(idx)
        scheduler.step()

        # Validación
        model.eval()
        with torch.no_grad():
            vp = model(dm.va_st, dm.va_hr, dm.X_va)
            tp = model(dm.te_st, dm.te_hr, dm.X_te)
            v_rmse = torch.sqrt(torch.mean((vp - dm.y_va)**2)).item()
            t_rmse = torch.sqrt(torch.mean((tp - dm.y_te)**2)).item()

        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_va_preds = vp.cpu().numpy().ravel()
            best_te_preds = tp.cpu().numpy().ravel()
            best_state = copy.deepcopy(model.state_dict())

        print(f"    Seed {seed} Ep {ep:02d} | Val RMSE: {v_rmse:.4f} | Test RMSE: {t_rmse:.4f} | Best Val: {best_val_rmse:.4f}")

    return best_va_preds, best_te_preds, best_state


def train_single_seed_quantile(
    dm: TabularDatasetManager,
    h: int,
    seed: int,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    delta: float = 1.5,
    use_weights: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TabResNetQuantile(
        num_stations=len(MODEL_STATIONS),
        num_features=dm.input_dim,
        embed_dim=16,
        hr_embed_dim=8,
        hidden_dim=256,
        num_blocks=3,
        dropout=0.05
    ).to(dm.device)

    huber = nn.HuberLoss(delta=delta)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_train = len(dm.y_tr)
    n_batches = (n_train + batch_size - 1) // batch_size

    best_val_rmse = float('inf')
    best_va = None
    best_te = None
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=dm.device)
        loss_acc = 0.0
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            optimizer.zero_grad()
            pred, q10, q90 = model(dm.tr_st[idx], dm.tr_hr[idx], dm.X_tr[idx])
            l_pred = huber(pred, dm.y_tr[idx])
            l_q10 = pinball_loss(q10, dm.y_tr[idx], 0.10)
            l_q90 = pinball_loss(q90, dm.y_tr[idx], 0.90)
            loss = l_pred + 0.5 * (l_q10 + l_q90)
            loss.backward()
            optimizer.step()
            loss_acc += loss.item() * len(idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp, v10, v90 = model(dm.va_st, dm.va_hr, dm.X_va)
            tp, t10, t90 = model(dm.te_st, dm.te_hr, dm.X_te)
            v_rmse = torch.sqrt(torch.mean((vp - dm.y_va)**2)).item()
            t_rmse = torch.sqrt(torch.mean((tp - dm.y_te)**2)).item()

        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_va = (vp.cpu().numpy().ravel(), v10.cpu().numpy().ravel(), v90.cpu().numpy().ravel())
            best_te = (tp.cpu().numpy().ravel(), t10.cpu().numpy().ravel(), t90.cpu().numpy().ravel())
            best_state = copy.deepcopy(model.state_dict())

        print(f"    Seed {seed} Ep {ep:02d} | Val RMSE: {v_rmse:.4f} | Test RMSE: {t_rmse:.4f} | Best Val: {best_val_rmse:.4f}")

    return best_va[0], best_va[1], best_va[2], best_te[0], best_te[1], best_te[2], best_state


def run_variant(variant: str, device: torch.device):
    print(f"\n{'='*75}")
    print(f"  INICIANDO VARIANTE: {variant}")
    print(f"  Dispositivo: {device} | Semillas: {SEEDS}")
    print(f"{'='*75}")

    summary_rows = []

    for h in HORIZONS:
        t0 = time.time()
        print(f"\n>>> Procesando h = {h:02d} ...")
        dm = TabularDatasetManager(h=h, device=device, seed=SEED)

        # Configuración por horizonte
        use_w = (h >= 8)
        delta = 1.5 if h <= 4 else 2.0
        n_epochs = 10 if h <= 4 else 8

        if variant == 'tab_resnet_honest':
            va_preds_list = []
            te_preds_list = []
            for s in SEEDS:
                print(f"  [h={h:02d}] Entrenando semilla {s} ...")
                va_p, te_p, state = train_single_seed_honest(
                    dm, h=h, seed=s, epochs=n_epochs, delta=delta, use_weights=use_w
                )
                va_preds_list.append(va_p)
                te_preds_list.append(te_p)
                # Guardar checkpoint de la primera semilla
                if s == SEEDS[0]:
                    torch.save(state, os.path.join(CKPT_DIR, f"{variant}_h{h}_s{s}.pt"))

            final_va_pred = np.mean(va_preds_list, axis=0)
            final_te_pred = np.mean(te_preds_list, axis=0)

            # Dataframes de predicciones según contrato
            val_df_out = pd.DataFrame({
                'Date': dm.va_df['Date'],
                'Estacion': dm.va_df['Estacion'],
                'h': h,
                'split': 'val',
                'y_true': dm.va_df['Target'].values.astype('float32'),
                'y_pred': final_va_pred.astype('float32')
            })
            test_df_out = pd.DataFrame({
                'Date': dm.te_df['Date'],
                'Estacion': dm.te_df['Estacion'],
                'h': h,
                'split': 'test',
                'y_true': dm.te_df['Target'].values.astype('float32'),
                'y_pred': final_te_pred.astype('float32')
            })
            preds_all = pd.concat([val_df_out, test_df_out], ignore_index=True)
            preds_path = os.path.join(PREDS_DIR, f"{variant}_h{h}.parquet")
            preds_all.to_parquet(preds_path, index=False)

        elif variant == 'tab_resnet_quantile':
            va_p_list, va_10_list, va_90_list = [], [], []
            te_p_list, te_10_list, te_90_list = [], [], []
            for s in SEEDS:
                print(f"  [h={h:02d}] Entrenando semilla cuantílica {s} ...")
                va_p, va_10, va_90, te_p, te_10, te_90, state = train_single_seed_quantile(
                    dm, h=h, seed=s, epochs=n_epochs, delta=delta, use_weights=use_w
                )
                va_p_list.append(va_p); va_10_list.append(va_10); va_90_list.append(va_90)
                te_p_list.append(te_p); te_10_list.append(te_10); te_90_list.append(te_90)
                if s == SEEDS[0]:
                    torch.save(state, os.path.join(CKPT_DIR, f"{variant}_h{h}_s{s}.pt"))

            final_va_pred = np.mean(va_p_list, axis=0)
            final_va_10 = np.mean(va_10_list, axis=0)
            final_va_90 = np.mean(va_90_list, axis=0)
            final_te_pred = np.mean(te_p_list, axis=0)
            final_te_10 = np.mean(te_10_list, axis=0)
            final_te_90 = np.mean(te_90_list, axis=0)

            val_df_out = pd.DataFrame({
                'Date': dm.va_df['Date'],
                'Estacion': dm.va_df['Estacion'],
                'h': h,
                'split': 'val',
                'y_true': dm.va_df['Target'].values.astype('float32'),
                'y_pred': final_va_pred.astype('float32'),
                'q10': final_va_10.astype('float32'),
                'q90': final_va_90.astype('float32')
            })
            test_df_out = pd.DataFrame({
                'Date': dm.te_df['Date'],
                'Estacion': dm.te_df['Estacion'],
                'h': h,
                'split': 'test',
                'y_true': dm.te_df['Target'].values.astype('float32'),
                'y_pred': final_te_pred.astype('float32'),
                'q10': final_te_10.astype('float32'),
                'q90': final_te_90.astype('float32')
            })
            preds_all = pd.concat([val_df_out, test_df_out], ignore_index=True)
            preds_path = os.path.join(PREDS_DIR, f"{variant}_h{h}.parquet")
            preds_all.to_parquet(preds_path, index=False)

        # Evaluación oficial
        val_rmse = float(np.sqrt(np.mean((final_va_pred - dm.va_df['Target'].values)**2)))
        m_te = evaluate_test(dm.te_df, final_te_pred, h=h)
        total_time = time.time() - t0

        # Cargar LightGBM de referencia para correlación de residuos
        lgb_path = os.path.join(ROOT, f"experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet")
        if os.path.exists(lgb_path):
            lgb_df = pd.read_parquet(lgb_path)
            lgb_te = lgb_df[lgb_df['split'] == 'test']['y_pred'].values
            lgb_res = dm.te_df[f'Target_{h}h'].values - lgb_te
            nn_res = dm.te_df[f'Target_{h}h'].values - final_te_pred
            res_corr = float(np.corrcoef(lgb_res, nn_res)[0, 1])

            # Evaluar blend 50/50
            blend_te = 0.5 * lgb_te + 0.5 * final_te_pred
            blend_m = evaluate_test(dm.te_df, blend_te, h=h)
            blend_rmse = blend_m['RMSE']
            lgb_rmse = float(np.sqrt(np.mean(lgb_res**2)))
        else:
            res_corr = float('nan')
            blend_rmse = float('nan')
            lgb_rmse = float('nan')

        # Registrar métricas en results_registry.csv
        notes = f"TabResNet 3-seeds, peak_w={use_w}, corr_vs_lgb={res_corr:.3f}, blend_rmse={blend_rmse:.3f}"
        log_result(experiment='noche2_deep2', variant=variant, h=h, m=m_te, notes=notes)

        print(f"\n  [h={h:02d} RESUMEN] Val RMSE: {val_rmse:.4f} | Test RMSE: {m_te['RMSE']:.4f} | R2: {m_te['R2']:.4f}")
        print(f"             Peaks RMSE: {m_te['RMSE_peaks']:.2f} | Bias Peaks: {m_te['bias_peaks']:.2f} | Tiempo: {total_time:.1f}s")
        print(f"             Corr vs D_honest_noise: {res_corr:.4f} | LGBM: {lgb_rmse:.4f} -> Blend: {blend_rmse:.4f}")

        summary_rows.append({
            'variant': variant,
            'h': h,
            'val_rmse': val_rmse,
            'test_rmse': m_te['RMSE'],
            'test_r2': m_te['R2'],
            'test_rmse_peaks': m_te['RMSE_peaks'],
            'test_bias_peaks': m_te['bias_peaks'],
            'res_corr_vs_lgb': res_corr,
            'lgb_test_rmse': lgb_rmse,
            'blend_test_rmse': blend_rmse,
            'elapsed_sec': total_time
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(BASE_DIR, f"summary_{variant}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nResumen guardado en {summary_path}")
    return summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='all', choices=['tab_resnet_honest', 'tab_resnet_quantile', 'all'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando ejecucion train_deep2.py en dispositivo: {device}")

    variants = ['tab_resnet_honest', 'tab_resnet_quantile'] if args.variant == 'all' else [args.variant]
    for v in variants:
        run_variant(v, device)

    print("\nEntrenamiento completo finalizado exitosamente.")
