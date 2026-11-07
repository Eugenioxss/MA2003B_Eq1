import os, sys, time, pandas as pd, numpy as np, torch

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'noche2', 'deep2'))

from common import evaluate_test, log_result, MODEL_STATIONS, HORIZONS
from experiments.noche.deep.dataset import MultiHorizonDataBundle, FastDataLoader
from model_seq_corrected import SeqCorrectedModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bundle = MultiHorizonDataBundle(seq_len=48, future_len=48)

o3_mean, o3_std = bundle.var_stats['O3']
EVAL_HORIZONS = [1, 4, 8, 12, 24, 48]
EVAL_INDICES = [h - 1 for h in EVAL_HORIZONS]

model = SeqCorrectedModel(
    past_channels=36, past_len=48, future_channels=18, future_len=48,
    spatial_dim=37, num_stations=len(MODEL_STATIONS), station_emb_dim=16,
    gru_hidden=128, future_proj_dim=32, hidden_dim=256, dropout=0.10
).to(device)

ckpt_path = os.path.join(ROOT, 'experiments/noche2/deep2/checkpoints/seq_corrected_honest.pt')
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

def predict_split(loader):
    all_preds = []
    with torch.no_grad():
        for past, fut, spat, st, clim, tgt, msk in loader:
            past = past.to(device)
            fut = fut.to(device)
            spat = spat.to(device)
            st = st.to(device)
            base_o3 = (past[:, -1, 0:1] * o3_std + o3_mean).clamp(min=0.0)
            yp = model(past, fut, spat, st, base_o3)
            all_preds.append(yp.cpu().numpy())
    concat_preds = np.concatenate(all_preds, axis=0) # (N, 48)
    pairs = loader.pairs
    st_list = [bundle.stations[s] for s in pairs[:, 0]]
    dt_list = [bundle.dates[t] for t in pairs[:, 1]]
    df = pd.DataFrame({'Date': dt_list, 'Estacion': st_list})
    for h, h_idx in zip(EVAL_HORIZONS, EVAL_INDICES):
        df[f'pred_h{h}'] = concat_preds[:, h_idx].astype(np.float32)
    return df

val_loader = FastDataLoader(bundle, t_start=35064, t_end=43848, batch_size=512, mode='honest', shuffle=False, add_noise=False, allow_clamp=True)
test_loader = FastDataLoader(bundle, t_start=43848, t_end=52608, batch_size=512, mode='honest', shuffle=False, add_noise=False, allow_clamp=True)

df_val_preds = predict_split(val_loader)
df_test_preds = predict_split(test_loader)

print("Alignment fixed. Evaluating metrics with evaluate_test...")
PREDS_DIR = os.path.join(ROOT, 'experiments/noche2/deep2/preds')

rows = []
for h in EVAL_HORIZONS:
    fpath_h = os.path.join(ROOT, f"experiments/noche/v3_base/cache/features_h{h}.parquet")
    df_cache = pd.read_parquet(fpath_h)
    
    va_df = df_cache[(df_cache['split'] == 'val') & df_cache['Target'].notna()].copy()
    te_df = df_cache[(df_cache['split'] == 'test') & df_cache['Target'].notna()].copy()

    # Merge by Date and Estacion to align
    va_merged = pd.merge(va_df[['Date', 'Estacion', 'Target']], df_val_preds[['Date', 'Estacion', f'pred_h{h}']], on=['Date', 'Estacion'], how='inner')
    te_merged = pd.merge(te_df[['Date', 'Estacion', f'Target_{h}h']], df_test_preds[['Date', 'Estacion', f'pred_h{h}']], on=['Date', 'Estacion'], how='inner')

    va_p = va_merged[f'pred_h{h}'].values
    va_t = va_merged['Target'].values
    val_rmse = float(np.sqrt(np.mean((va_p - va_t)**2)))

    te_p = te_merged[f'pred_h{h}'].values
    te_t = te_merged[f'Target_{h}h'].values
    m_te = evaluate_test(te_merged, te_p, h=h)

    # Correlacion vs LGBM
    lgb_df = pd.read_parquet(os.path.join(ROOT, f"experiments/noche/v3_base/preds/D_honest_noise_h{h}.parquet"))
    lgb_te = lgb_df[lgb_df['split'] == 'test'].reset_index(drop=True)
    te_with_lgb = pd.merge(te_merged, lgb_te[['Date', 'Estacion', 'y_pred']], on=['Date', 'Estacion'], how='inner')
    
    lgb_res = te_with_lgb[f'Target_{h}h'].values - te_with_lgb['y_pred'].values
    seq_res = te_with_lgb[f'Target_{h}h'].values - te_with_lgb[f'pred_h{h}'].values
    res_corr = float(np.corrcoef(lgb_res, seq_res)[0, 1])

    blend_pred = 0.5 * te_with_lgb['y_pred'].values + 0.5 * te_with_lgb[f'pred_h{h}'].values
    m_blend = evaluate_test(te_with_lgb, blend_pred, h=h)
    lgb_rmse = float(np.sqrt(np.mean(lgb_res**2)))

    # Guardar preds parquet oficiales segun contrato
    va_out = pd.DataFrame({
        'Date': va_merged['Date'], 'Estacion': va_merged['Estacion'], 'h': h, 'split': 'val',
        'y_true': va_t.astype('float32'), 'y_pred': va_p.astype('float32')
    })
    te_out = pd.DataFrame({
        'Date': te_merged['Date'], 'Estacion': te_merged['Estacion'], 'h': h, 'split': 'test',
        'y_true': te_t.astype('float32'), 'y_pred': te_p.astype('float32')
    })
    pd.concat([va_out, te_out], ignore_index=True).to_parquet(
        os.path.join(PREDS_DIR, f"seq_corrected_honest_h{h}.parquet"), index=False
    )

    log_result(
        experiment='noche2_deep2', variant='seq_corrected_honest', h=h, m=m_te,
        notes=f"Sequential GRU + Persistence Anchor + Variance Normalization. corr={res_corr:.3f}, blend={m_blend['RMSE']:.3f}"
    )

    print(f"h={h:02d} | Val RMSE: {val_rmse:.4f} | Test RMSE: {m_te['RMSE']:.4f} | LGBM: {lgb_rmse:.4f} -> Blend: {m_blend['RMSE']:.4f} | Res Corr: {res_corr:.4f}")
    rows.append({
        'h': h, 'val_rmse': val_rmse, 'test_rmse': m_te['RMSE'], 'lgb_test_rmse': lgb_rmse,
        'blend_test_rmse': m_blend['RMSE'], 'test_r2': m_te['R2'], 'test_peaks_rmse': m_te['RMSE_peaks'],
        'test_peaks_bias': m_te['bias_peaks'], 'res_corr_vs_lgb': res_corr
    })

res_df = pd.DataFrame(rows)
res_df.to_csv(os.path.join(ROOT, 'experiments/noche2/deep2/summary_seq_corrected_honest.csv'), index=False)
