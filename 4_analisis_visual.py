import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.tsa.stattools import ccf

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

print("=======================================================")
print(" SCRIPT 4: ANÁLISIS VISUAL (INTERDEPENDENCIA Y CCF)")
print("=======================================================")

df_ml = pd.read_parquet('data/ml_ready/dataset_ozono_predictivo.parquet')

# --- 1. DISTRIBUCIONES FÍSICO-ESTADÍSTICAS ---
print("Generando Gráficos de Ajuste de Distribuciones (floc=0)...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

pm25_valid = df_ml['PM2.5'].dropna()
pm25_valid = pm25_valid[pm25_valid > 0] # Lognormal estrictamente requiere valores > 0
shape_log, loc_log, scale_log = stats.lognorm.fit(pm25_valid, floc=0)
x_pm25 = np.linspace(pm25_valid.min(), pm25_valid.max(), 100)
axes[0].hist(pm25_valid, bins=50, density=True, alpha=0.5, color='orange')
axes[0].plot(x_pm25, stats.lognorm.pdf(x_pm25, shape_log, loc_log, scale_log), 'r-', lw=2, label='Lognormal Fit')
axes[0].set_title('Distribución Asimétrica de $PM_{2.5}$')
axes[0].set_xlabel('Concentración ($\mu g/m^3$)')
axes[0].legend()
axes[0].set_xlim(0, np.percentile(pm25_valid, 99.5)) # Recortar cola larga visualmente

pm10_valid = df_ml['PM10'].dropna()
pm10_valid = pm10_valid[pm10_valid > 0]
shape_gam, loc_gam, scale_gam = stats.gamma.fit(pm10_valid, floc=0)
x_pm10 = np.linspace(pm10_valid.min(), pm10_valid.max(), 100)
axes[1].hist(pm10_valid, bins=50, density=True, alpha=0.5, color='gray')
axes[1].plot(x_pm10, stats.gamma.pdf(x_pm10, shape_gam, loc_gam, scale_gam), 'b-', lw=2, label='Gamma Fit')
axes[1].set_title('Distribución de Eventos Extremos $PM_{10}$')
axes[1].set_xlabel('Concentración ($\mu g/m^3$)')
axes[1].legend()
axes[1].set_xlim(0, np.percentile(pm10_valid, 99.5))

plt.tight_layout()
plt.savefig('results/figures/Etapa3_01_Distribuciones.png')
plt.close()

# --- 2. CORRELACIÓN CRUZADA (CCF) ---
print("Ejecutando Correlación Cruzada (CCF) agrupada por estación...")
lags_to_plot = 48
ccf_pm25_est, ccf_sr_est = [], []

for est, group in df_ml.groupby('Estacion'):
    if len(group) > 50:
        # El Orden correcto: Target va primero (para que el Predictor lidere al Target)
        ccf_pm = ccf(group['O3_8h'], group['PM2.5_12h'], adjusted=False)[:lags_to_plot]
        ccf_sr = ccf(group['O3_8h'], group['SR'], adjusted=False)[:lags_to_plot]
        ccf_pm25_est.append(ccf_pm)
        ccf_sr_est.append(ccf_sr)

ccf_pm25_o3 = np.nanmean(np.array(ccf_pm25_est), axis=0)
ccf_sr_o3 = np.nanmean(np.array(ccf_sr_est), axis=0)

# IMPRESIÓN TRANSPARENTE EN CONSOLA
min_lag_pm = np.argmin(ccf_pm25_o3)
max_lag_pm = np.argmax(ccf_pm25_o3)
max_lag_sr = np.argmax(ccf_sr_o3)
print(f"-> O3 liderado por PM2.5 (argmin): Lag {min_lag_pm} horas (r={ccf_pm25_o3[min_lag_pm]:.4f}).")
print(f"-> O3 liderado por PM2.5 (argmax): Lag {max_lag_pm} horas (r={ccf_pm25_o3[max_lag_pm]:.4f}).")
print(f"-> O3 liderado por Radiación Solar (argmax): Lag {max_lag_sr} horas (r={ccf_sr_o3[max_lag_sr]:.4f}).")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
lags = np.arange(0, lags_to_plot)

# Banda de significancia +- 1.96 / sqrt(N)
conf_bound = 1.96 / np.sqrt(len(df_ml))

axes[0].vlines(lags, [0], ccf_pm25_o3, color='crimson', lw=3)
axes[0].axhline(0, color='black', lw=1)
axes[0].axhline(conf_bound, color='blue', linestyle='--', alpha=0.5)
axes[0].axhline(-conf_bound, color='blue', linestyle='--', alpha=0.5)
axes[0].set_title('Correlación Cruzada: $PM_{2.5}$ (NowCast) liderando a $O_3$ (8h)')
axes[0].set_xlabel('Lag (horas)')
axes[0].set_ylabel('Correlación')

axes[1].vlines(lags, [0], ccf_sr_o3, color='orange', lw=3)
axes[1].axhline(0, color='black', lw=1)
axes[1].axhline(conf_bound, color='blue', linestyle='--', alpha=0.5)
axes[1].axhline(-conf_bound, color='blue', linestyle='--', alpha=0.5)
axes[1].set_title('Correlación Cruzada: Radiación Solar liderando a $O_3$ (8h)')
axes[1].set_xlabel('Lag (horas)')
axes[1].set_ylabel('Correlación')

plt.tight_layout()
plt.savefig('results/figures/Etapa3_02_CCF.png')
plt.close()
print("✅ ¡Análisis visual completado y exportado!")
