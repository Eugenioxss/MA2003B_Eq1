"""
experiments/noche/deep/plot_results.py
--------------------------------------
Generates comparison plots and summary tables for noche_deep.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = r"D:\Codingggg\Notas\Multivariados\MA2003B_Eq1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

FIG_DIR = os.path.join(ROOT, 'experiments', 'noche', 'deep', 'figures')
PRED_DIR = os.path.join(ROOT, 'experiments', 'noche', 'deep', 'preds')
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Baseline references from README_noche.md
# Horizons: 1, 4, 8, 12, 24, 48
horizons = [1, 4, 8, 12, 24, 48]

v2_rmse = [8.0509, 10.9066, 11.9672, 12.2051, 12.7609, 13.9494]
v2_r2   = [0.8537, 0.7317, 0.6771, 0.6642, 0.6332, 0.5624]
v2_peaks_rmse = [19.25, 27.90, 31.12, 31.65, 33.34, 37.57]

planA_rmse = [5.09, 9.69, 11.28, 11.73, 12.39, 13.76]
planC_rmse = [7.55, 9.38, 9.82, 9.80, 10.79, 11.58]

# Load results from results_registry.csv
reg_path = os.path.join(ROOT, 'experiments', 'results_registry.csv')
df_reg = pd.read_csv(reg_path, on_bad_lines='skip')
deep_df = df_reg[df_reg['experiment'] == 'noche_deep'].drop_duplicates(['variant', 'h'], keep='last')

variants = ['tide_honest', 'tide_pp', 'tide_quantile_honest', 'tide_quantile_pp', 'gru_honest', 'ensemble_honest', 'ensemble_pp']
var_data = {}
for v in variants:
    sub = deep_df[deep_df['variant'] == v].sort_values('h')
    var_data[v] = {
        'RMSE': sub['RMSE'].values,
        'R2': sub['R2'].values,
        'RMSE_peaks': sub['RMSE_peaks'].values,
        'bias_peaks': sub['bias_peaks'].values,
    }

# -------------------------------------------------------------
# Figure 1: RMSE across horizons (Test 2025)
# -------------------------------------------------------------
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(horizons, v2_rmse, 'k--o', label='Baseline v2 (Script 7, tabular)', linewidth=1.8, markersize=6)
plt.plot(horizons, planA_rmse, 'gray', linestyle=':', marker='s', label='Plan A V4 (Spatial + lag 0, no meteo)', linewidth=1.5)
plt.plot(horizons, var_data['tide_honest']['RMSE'], 'b-o', label='TiDE-MLP (Honest NWP)', linewidth=2.0, markersize=6)
plt.plot(horizons, var_data['gru_honest']['RMSE'], 'g-^', label='GRU-TiDE (Honest NWP)', linewidth=1.8, markersize=6)
plt.plot(horizons, var_data['ensemble_honest']['RMSE'], 'm-D', label='Deep Ensemble (Honest NWP - Operativo)', linewidth=2.5, markersize=7)
plt.plot(horizons, var_data['ensemble_pp']['RMSE'], 'r-.v', label='Deep Ensemble (PP - Cota Superior no operativa)', linewidth=2.0, markersize=6)

plt.title('Comparación de RMSE por Horizonte (Conjunto de Prueba 2025, 13 Estaciones SIMA)', fontsize=12, fontweight='bold')
plt.xlabel('Horizonte de Pronóstico $h$ (horas)', fontsize=11)
plt.ylabel('RMSE (ppb)', fontsize=11)
plt.xticks(horizons)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, fontsize=9.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'rmse_by_horizon.png'))
plt.close()

# -------------------------------------------------------------
# Figure 2: Peaks RMSE (>P95) Comparison
# -------------------------------------------------------------
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(horizons, v2_peaks_rmse, 'k--o', label='Baseline v2 (Script 7)', linewidth=1.8, markersize=6)
plt.plot(horizons, var_data['tide_honest']['RMSE_peaks'], 'b-o', label='TiDE-MLP (Honest)', linewidth=2.0, markersize=6)
plt.plot(horizons, var_data['tide_quantile_honest']['RMSE_peaks'], 'c-s', label='TiDE Quantile (Honest)', linewidth=1.8, markersize=6)
plt.plot(horizons, var_data['gru_honest']['RMSE_peaks'], 'g-^', label='GRU-TiDE (Honest)', linewidth=1.8, markersize=6)
plt.plot(horizons, var_data['ensemble_honest']['RMSE_peaks'], 'm-D', label='Deep Ensemble (Honest)', linewidth=2.5, markersize=7)
plt.plot(horizons, var_data['ensemble_pp']['RMSE_peaks'], 'r-.v', label='Deep Ensemble (PP Cota Superior)', linewidth=2.0, markersize=6)

plt.title('RMSE en Eventos Extremos de Ozono (>P95) en Test 2025', fontsize=12, fontweight='bold')
plt.xlabel('Horizonte de Pronóstico $h$ (horas)', fontsize=11)
plt.ylabel('RMSE en Picos (ppb)', fontsize=11)
plt.xticks(horizons)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, fontsize=9.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'peaks_rmse_by_horizon.png'))
plt.close()

# -------------------------------------------------------------
# Figure 3: Honest vs PP Gap (NWP uncertainty penalty)
# -------------------------------------------------------------
plt.figure(figsize=(9, 5), dpi=150)
tide_gap = var_data['tide_honest']['RMSE'] - var_data['tide_pp']['RMSE']
ens_gap = var_data['ensemble_honest']['RMSE'] - var_data['ensemble_pp']['RMSE']

x = np.arange(len(horizons))
width = 0.35
plt.bar(x - width/2, tide_gap, width, label='Brecha TiDE-MLP (Honest - PP)', color='royalblue')
plt.bar(x + width/2, ens_gap, width, label='Brecha Deep Ensemble (Honest - PP)', color='mediumorchid')

for i in range(len(horizons)):
    plt.text(x[i] - width/2, tide_gap[i] + 0.03, f"{tide_gap[i]:.2f}", ha='center', va='bottom', fontsize=9)
    plt.text(x[i] + width/2, ens_gap[i] + 0.03, f"{ens_gap[i]:.2f}", ha='center', va='bottom', fontsize=9)

plt.title('Penalización por Incertidumbre de Pronóstico Meteorológico Numérico (Δ RMSE)', fontsize=12, fontweight='bold')
plt.xlabel('Horizonte $h$ (horas)', fontsize=11)
plt.ylabel('Δ RMSE (ppb) = Honest - PP', fontsize=11)
plt.xticks(x, [f'h={h}' for h in horizons])
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.legend(frameon=True)
plt.ylim(0, 1.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'honest_vs_pp_gap.png'))
plt.close()

# -------------------------------------------------------------
# Figure 4: Sample Prediction Interval (Quantile q10, q50, q90)
# -------------------------------------------------------------
p_sample = os.path.join(PRED_DIR, 'ensemble_honest_h24.parquet')
df_sample = pd.read_parquet(p_sample)
df_ce = df_sample[(df_sample['Estacion'] == 'CE') & (df_sample['split'] == 'test')].sort_values('Date').reset_index(drop=True)

# Select a 14-day high-ozone episode in spring (May 2025)
df_ce['Date'] = pd.to_datetime(df_ce['Date'])
window = df_ce[(df_ce['Date'] >= '2025-05-01') & (df_ce['Date'] <= '2025-05-14')]

plt.figure(figsize=(12, 5), dpi=150)
plt.plot(window['Date'], window['y_true'], 'k-', label='O3 Observado (t+24h)', linewidth=1.5)
plt.plot(window['Date'], window['y_pred'], 'b-', label='Pronóstico Mediana (q50)', linewidth=1.8)
if 'q10' in window.columns and 'q90' in window.columns:
    plt.fill_between(window['Date'], window['q10'], window['q90'], color='cyan', alpha=0.3, label='Intervalo de Predicción 80% [q10, q90]')

plt.title('Pronóstico Multi-Horizonte a 24 horas (h=24) con Intervalo de Confianza del 80% — Estación Centro (CE)', fontsize=12, fontweight='bold')
plt.xlabel('Fecha de Emisión t', fontsize=11)
plt.ylabel('Concentración de O3 (ppb)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sample_prediction_interval.png'))
plt.close()

print("Figures generated successfully in", FIG_DIR)

# -------------------------------------------------------------
# Summary CSV comparison
# -------------------------------------------------------------
summary_rows = []
for i, h in enumerate(horizons):
    row = {
        'h': h,
        'v2_RMSE': v2_rmse[i],
        'v2_R2': v2_r2[i],
        'planA_RMSE': planA_rmse[i],
        'tide_honest_RMSE': var_data['tide_honest']['RMSE'][i],
        'tide_honest_R2': var_data['tide_honest']['R2'][i],
        'tide_quantile_honest_RMSE': var_data['tide_quantile_honest']['RMSE'][i],
        'gru_honest_RMSE': var_data['gru_honest']['RMSE'][i],
        'ensemble_honest_RMSE': var_data['ensemble_honest']['RMSE'][i],
        'ensemble_honest_R2': var_data['ensemble_honest']['R2'][i],
        'delta_RMSE_vs_v2': var_data['ensemble_honest']['RMSE'][i] - v2_rmse[i],
        'ensemble_pp_RMSE': var_data['ensemble_pp']['RMSE'][i],
        'pp_honest_gap': var_data['ensemble_honest']['RMSE'][i] - var_data['ensemble_pp']['RMSE'][i],
        'peaks_RMSE_v2': v2_peaks_rmse[i],
        'peaks_RMSE_ensemble_honest': var_data['ensemble_honest']['RMSE_peaks'][i],
        'delta_peaks_RMSE': var_data['ensemble_honest']['RMSE_peaks'][i] - v2_peaks_rmse[i]
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(ROOT, 'experiments', 'noche', 'deep', 'summary_deep_comparison.csv')
summary_df.to_csv(summary_path, index=False)
print("Summary CSV saved to", summary_path)
print(summary_df[['h', 'v2_RMSE', 'planA_RMSE', 'tide_honest_RMSE', 'ensemble_honest_RMSE', 'ensemble_pp_RMSE', 'delta_RMSE_vs_v2']])
