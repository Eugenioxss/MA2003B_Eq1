"""
=============================================================================
  SCRIPT 7: MULTI-HORIZONTE CON FEATURE ENGINEERING AVANZADO
  Mismo pipeline que Script 6, pero con features adicionales:
    - Interacciones fisicoquimicas (NOX*SR, TOUT*SR, O3/NOX ratio, wind_speed)
    - Tasas de cambio (diffs de 1h y 24h)
    - Rolling stats (mean, std, max, min en ventanas de 6h y 24h)
    - Anomalia vs ayer (ratios y diferencias dia anterior)
  Los resultados se guardan en carpeta separada para comparar vs Script 6.
=============================================================================
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime

# -- Configuracion ----------------------------------------------------------
HORIZONTES = [1, 4, 8, 12, 24, 48]
VARIABLES_BASE = ['O3', 'NOX', 'SR', 'TOUT', 'WSR', 'WDR', 'PM10', 'PM2.5']
ESTACIONES_EXCLUIDAS = ['NE3', 'NO3']
SEED = 42

OUT_DIR = 'results/multihorizonte_v2'
FIG_DIR = f'{OUT_DIR}/figures'
MODEL_DIR = f'{OUT_DIR}/models'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
PALETTE = sns.color_palette('viridis', len(HORIZONTES))

# ===========================================================================
#  ETAPA 1: CARGA DE DATOS
# ===========================================================================
print("=" * 65)
print("  MULTI-HORIZONTE v2: CON FEATURE ENGINEERING")
print(f"  Horizontes: {HORIZONTES} horas")
print("=" * 65)

print("\n[1/6] Cargando y uniendo variables base...")
dfs = []
for var in VARIABLES_BASE:
    path = f"data/processed/variables/{var}_clean.parquet"
    if os.path.exists(path):
        dfs.append(pd.read_parquet(path))

df_ml = reduce(
    lambda left, right: pd.merge(left, right, on=['Date', 'Estacion'], how='outer'),
    dfs,
)
df_ml.sort_values(by=['Estacion', 'Date'], inplace=True)
df_ml.reset_index(drop=True, inplace=True)
df_ml = df_ml[~df_ml['Estacion'].isin(ESTACIONES_EXCLUIDAS)]

# Viento U y V
radianes = df_ml['WDR'] * (np.pi / 180)
df_ml['U_Wind'] = -df_ml['WSR'] * np.sin(radianes)
df_ml['V_Wind'] = -df_ml['WSR'] * np.cos(radianes)
df_ml.drop(columns=['WDR', 'WSR'], inplace=True)

# ===========================================================================
#  ETAPA 2: LAGS BASE + FEATURE ENGINEERING
# ===========================================================================
print("[2/6] Generando lags base + features de ingenieria...")

start_date = df_ml['Date'].min()
end_date = df_ml['Date'].max()
full_idx = pd.date_range(start=start_date, end=end_date, freq='h')


def process_station(group):
    group = group.set_index('Date')
    group = group[~group.index.duplicated(keep='first')]
    group = group.reindex(full_idx)

    # ================================================================
    # A) LAGS BASE (mismos que Script 6 para comparacion justa)
    # ================================================================
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        group[f'O3_lag_{lag}'] = group['O3'].shift(lag)

    for lag in [1, 2, 4, 12, 24, 48]:
        for feat in ['NOX', 'SR', 'TOUT', 'PM10', 'PM2.5', 'U_Wind', 'V_Wind']:
            group[f'{feat}_lag_{lag}'] = group[feat].shift(lag)

    # Calendario base
    group['hour'] = group.index.hour
    group['month'] = group.index.month
    group['dayofweek'] = group.index.dayofweek

    # ================================================================
    # B) FEATURES NUEVOS: Interacciones Fisicoquimicas
    # ================================================================
    # Potencial fotoquimico: NOX + Radiacion Solar = produccion de O3
    group['NOX_x_SR'] = group['NOX'].shift(1) * group['SR'].shift(1)
    # Intensidad termica + solar combinada
    group['TOUT_x_SR'] = group['TOUT'].shift(1) * group['SR'].shift(1)
    # Regimen quimico: ratio O3/NOX (alto = regimen VOC-limitado)
    group['O3_NOX_ratio'] = group['O3'].shift(1) / (group['NOX'].shift(1) + 1)
    # Velocidad real del viento (magnitud del vector)
    group['wind_speed'] = np.sqrt(
        group['U_Wind'].shift(1)**2 + group['V_Wind'].shift(1)**2
    )

    # ================================================================
    # C) FEATURES NUEVOS: Tasas de Cambio (Derivadas)
    # ================================================================
    # Velocidad de cambio de O3 (subiendo o bajando?)
    group['O3_diff_1h'] = group['O3'].shift(1) - group['O3'].shift(2)
    # Aceleracion de O3 (se esta acelerando el cambio?)
    group['O3_accel'] = (
        (group['O3'].shift(1) - group['O3'].shift(2)) -
        (group['O3'].shift(2) - group['O3'].shift(3))
    )
    # Anomalia de O3 vs ayer a esta misma hora
    group['O3_diff_24h'] = group['O3'].shift(1) - group['O3'].shift(25)
    # Tasa de cambio de temperatura (se esta calentando rapido?)
    group['TOUT_diff_3h'] = group['TOUT'].shift(1) - group['TOUT'].shift(4)
    # Tasa de cambio de NOX (pico de trafico reciente?)
    group['NOX_diff_1h'] = group['NOX'].shift(1) - group['NOX'].shift(2)
    # Cambio en radiacion solar
    group['SR_diff_1h'] = group['SR'].shift(1) - group['SR'].shift(2)

    # ================================================================
    # D) FEATURES NUEVOS: Rolling Stats (Resumenes de ventana)
    # ================================================================
    # --- Ventana de 6 horas (corto plazo) ---
    o3_shifted = group['O3'].shift(1)  # shift para no usar dato actual
    group['O3_mean_6h'] = o3_shifted.rolling(6, min_periods=2).mean()
    group['O3_std_6h'] = o3_shifted.rolling(6, min_periods=2).std()

    tout_shifted = group['TOUT'].shift(1)
    group['TOUT_mean_6h'] = tout_shifted.rolling(6, min_periods=2).mean()

    nox_shifted = group['NOX'].shift(1)
    group['NOX_mean_6h'] = nox_shifted.rolling(6, min_periods=2).mean()

    sr_shifted = group['SR'].shift(1)
    group['SR_sum_6h'] = sr_shifted.rolling(6, min_periods=2).sum()

    # --- Ventana de 24 horas (ciclo diario completo) ---
    group['O3_max_24h'] = o3_shifted.rolling(24, min_periods=6).max()
    group['O3_min_24h'] = o3_shifted.rolling(24, min_periods=6).min()
    group['O3_range_24h'] = group['O3_max_24h'] - group['O3_min_24h']
    group['TOUT_max_24h'] = tout_shifted.rolling(24, min_periods=6).max()
    group['SR_sum_12h'] = sr_shifted.rolling(12, min_periods=3).sum()

    # ================================================================
    # E) FEATURES NUEVOS: Anomalia vs Patron Diario
    # ================================================================
    # Ratio hoy/ayer a la misma hora (>1 = peor que ayer)
    group['O3_vs_yesterday'] = group['O3'].shift(1) / (group['O3'].shift(25) + 1)
    # Diferencia de temperatura vs ayer
    group['TOUT_vs_yesterday'] = group['TOUT'].shift(1) - group['TOUT'].shift(25)

    # ================================================================
    # F) FEATURES NUEVOS: Contexto Urbano
    # ================================================================
    group['is_weekend'] = (group.index.dayofweek >= 5).astype(int)
    group['is_rush_hour'] = group.index.hour.isin([7, 8, 9, 17, 18, 19]).astype(int)

    # ================================================================
    # G) FEATURES NUEVOS: Codificacion Ciclica
    # ================================================================
    group['hour_sin'] = np.sin(2 * np.pi * group.index.hour / 24)
    group['hour_cos'] = np.cos(2 * np.pi * group.index.hour / 24)
    group['month_sin'] = np.sin(2 * np.pi * group.index.month / 12)
    group['month_cos'] = np.cos(2 * np.pi * group.index.month / 12)

    # ================================================================
    # TARGETS
    # ================================================================
    for h in HORIZONTES:
        group[f'Target_{h}h'] = group['O3'].shift(-h)

    return group


df_full = df_ml.groupby('Estacion').apply(process_station).reset_index()
if 'Date' not in df_full.columns:
    df_full.rename(columns={df_full.columns[1]: 'Date'}, inplace=True)

# Features: todos los lags + todos los features nuevos + calendario
EXCLUIR = {'Date', 'Estacion', 'O3', 'NOX', 'SR', 'TOUT', 'PM10', 'PM2.5',
           'U_Wind', 'V_Wind', 'year'} | {f'Target_{h}h' for h in HORIZONTES}
features = [c for c in df_full.columns if c not in EXCLUIR and not c.startswith('level_')]

print(f"  -> {len(features)} features totales")
print(f"     (vs 52 del Script 6 sin feature engineering)")

# Contar features nuevos
feat_base = [c for c in features if 'lag_' in c or c in ['hour', 'month', 'dayofweek']]
feat_nuevos = [c for c in features if c not in feat_base]
print(f"     {len(feat_base)} features base (lags + calendario)")
print(f"     {len(feat_nuevos)} features de ingenieria NUEVOS:")
for f in feat_nuevos:
    print(f"       + {f}")

# ===========================================================================
#  ETAPA 3: ENTRENAMIENTO
# ===========================================================================
print("\n[3/6] Entrenando modelos independientes por horizonte...\n")

df_full['year'] = df_full['Date'].dt.year

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': SEED,
    'verbose': -1,
}

resultados = []
modelos = {}
importancias = {}

for h in HORIZONTES:
    target_col = f'Target_{h}h'
    print(f"  -- Horizonte +{h}h --")

    df_h = df_full.dropna(subset=[target_col]).copy()

    train = df_h[df_h['year'] <= 2023]
    val   = df_h[df_h['year'] == 2024]
    test  = df_h[df_h['year'] == 2025]

    X_train, y_train = train[features], train[target_col]
    X_val,   y_val   = val[features],   val[target_col]
    X_test,  y_test  = test[features],  test[target_col]

    print(f"     Train: {len(X_train):>9,}  |  Val: {len(X_val):>8,}  |  Test: {len(X_test):>8,}")

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    mae   = mean_absolute_error(y_test, y_pred)
    r2    = r2_score(y_test, y_pred)

    print(f"     RMSE: {rmse:.4f} ppb  |  MAE: {mae:.4f}  |  R2: {r2:.4f}")

    resultados.append({
        'Horizonte (h)': h,
        'RMSE (ppb)': round(rmse, 4),
        'MAE (ppb)': round(mae, 4),
        'R2': round(r2, 4),
        'Best Iteration': model.best_iteration,
        'Cobertura Test': len(y_test),
    })

    model.save_model(f'{MODEL_DIR}/lgbm_v2_{h}h.txt')
    modelos[h] = model
    importancias[h] = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importance(),
    }).sort_values('Importance', ascending=False)

    test_out = test[['Date', 'Estacion']].copy()
    test_out['y_real'] = y_test.values
    test_out['y_pred'] = y_pred
    test_out.to_parquet(f'{OUT_DIR}/pred_test_{h}h.parquet', index=False)
    print()

df_resultados = pd.DataFrame(resultados)

# ===========================================================================
#  ETAPA 4: COMPARACION vs SCRIPT 6 (v1)
# ===========================================================================
print("[4/6] Comparando con resultados del Script 6 (sin feature engineering)...\n")

# Resultados v1 (Script 6) - hardcodeados del run anterior
v1_data = {
    'Horizonte (h)': [1, 4, 8, 12, 24, 48],
    'RMSE_v1': [8.2391, 11.4649, 12.4179, 12.2860, 12.7518, 13.9266],
    'R2_v1':   [0.8468, 0.7035, 0.6523, 0.6597, 0.6338, 0.5638],
    'MAE_v1':  [5.6921, 8.1063, 8.8670, 8.8004, 9.1455, 10.0149],
}
df_v1 = pd.DataFrame(v1_data)

df_comp = pd.merge(df_v1, df_resultados, on='Horizonte (h)')
df_comp['Delta_RMSE'] = df_comp['RMSE (ppb)'] - df_comp['RMSE_v1']
df_comp['Delta_R2'] = df_comp['R2'] - df_comp['R2_v1']
df_comp['Mejora_RMSE_%'] = ((df_comp['RMSE_v1'] - df_comp['RMSE (ppb)']) / df_comp['RMSE_v1'] * 100)

print("  COMPARATIVA v1 (lags) vs v2 (lags + feature eng.)")
print("  " + "-" * 60)
for _, row in df_comp.iterrows():
    h = int(row['Horizonte (h)'])
    signo_rmse = "+" if row['Delta_RMSE'] > 0 else ""
    signo_r2 = "+" if row['Delta_R2'] > 0 else ""
    print(f"  +{h:>2}h | RMSE: {row['RMSE_v1']:.2f} -> {row['RMSE (ppb)']:.2f}  ({signo_rmse}{row['Delta_RMSE']:.2f})"
          f"  |  R2: {row['R2_v1']:.3f} -> {row['R2']:.3f}  ({signo_r2}{row['Delta_R2']:.3f})")
print()

# ===========================================================================
#  ETAPA 5: GRAFICAS
# ===========================================================================
print("[5/6] Generando graficas...\n")

# -- 5.1  Comparativa RMSE v1 vs v2 ----------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
x_pos = np.arange(len(HORIZONTES))
width = 0.35
b1 = ax.bar(x_pos - width/2, df_comp['RMSE_v1'],    width, label='v1: Solo Lags', color='#8ecae6')
b2 = ax.bar(x_pos + width/2, df_comp['RMSE (ppb)'], width, label='v2: Lags + Feature Eng.', color='#023047')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'+{h}h' for h in HORIZONTES])
ax.set_xlabel('Horizonte de Prediccion')
ax.set_ylabel('RMSE (ppb)')
ax.set_title('RMSE: Solo Lags vs Lags + Feature Engineering', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/01_comparativa_rmse_v1_vs_v2.png')
plt.close(fig)
print("  [ok] 01_comparativa_rmse_v1_vs_v2.png")

# -- 5.2  Comparativa R2 v1 vs v2 ------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot([f'+{h}h' for h in HORIZONTES], df_comp['R2_v1'], marker='s', markersize=9,
        linewidth=2.2, color='#8ecae6', label='v1: Solo Lags', markeredgecolor='#457b9d', markeredgewidth=1.5)
ax.plot([f'+{h}h' for h in HORIZONTES], df_comp['R2'], marker='o', markersize=9,
        linewidth=2.2, color='#023047', label='v2: Lags + Feature Eng.', markeredgecolor='white', markeredgewidth=1.5)
for i, h in enumerate(HORIZONTES):
    ax.annotate(f'{df_comp.iloc[i]["R2_v1"]:.3f}', (i, df_comp.iloc[i]['R2_v1']),
                textcoords='offset points', xytext=(-20, 10), fontsize=9, color='#457b9d')
    ax.annotate(f'{df_comp.iloc[i]["R2"]:.3f}', (i, df_comp.iloc[i]['R2']),
                textcoords='offset points', xytext=(5, -15), fontsize=9, color='#023047', fontweight='bold')
ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.4, label='Umbral aceptable (0.70)')
ax.set_xlabel('Horizonte de Prediccion')
ax.set_ylabel('R2')
ax.set_title('R2: Solo Lags vs Lags + Feature Engineering', fontsize=14, fontweight='bold')
ax.set_ylim(0.4, 0.95)
ax.legend(fontsize=10, loc='lower left')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/02_comparativa_r2_v1_vs_v2.png')
plt.close(fig)
print("  [ok] 02_comparativa_r2_v1_vs_v2.png")

# -- 5.3  Feature importance panel ------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for idx, h in enumerate(HORIZONTES):
    ax = axes[idx]
    top = importancias[h].head(15)
    # Colorear features nuevos vs base
    colors = []
    for f in top['Feature']:
        if f in feat_nuevos:
            colors.append('#e76f51')  # Naranja = nuevo
        else:
            colors.append('#2a9d8f')  # Verde = base
    ax.barh(range(len(top)-1, -1, -1), top['Importance'].values, color=colors)
    ax.set_yticks(range(len(top)-1, -1, -1))
    ax.set_yticklabels(top['Feature'].values)
    ax.set_title(f'Top 15 Features -- +{h}h', fontweight='bold')
    ax.set_xlabel('Importancia')
# Leyenda manual
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2a9d8f', label='Feature base (lag)'),
                   Patch(facecolor='#e76f51', label='Feature nuevo (ingenieria)')]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12,
           bbox_to_anchor=(0.5, 1.02))
plt.suptitle('Top 15 Variables por Horizonte (v2)', fontsize=15, fontweight='bold', y=1.04)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/03_feature_importance_v2_panel.png', bbox_inches='tight')
plt.close(fig)
print("  [ok] 03_feature_importance_v2_panel.png")

# -- 5.4  Heatmap de features nuevos por horizonte -------------------------
nuevos_heat = pd.DataFrame(index=feat_nuevos, columns=[f'+{h}h' for h in HORIZONTES])
for h in HORIZONTES:
    imp_h = importancias[h].set_index('Feature')['Importance']
    for f in feat_nuevos:
        nuevos_heat.loc[f, f'+{h}h'] = imp_h.get(f, 0)
nuevos_heat = nuevos_heat.astype(float)
nuevos_heat['total'] = nuevos_heat.sum(axis=1)
nuevos_heat = nuevos_heat.sort_values('total', ascending=False).drop(columns='total')

fig, ax = plt.subplots(figsize=(10, max(7, len(nuevos_heat) * 0.38)))
sns.heatmap(nuevos_heat, annot=True, fmt='.0f', cmap='OrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Importancia'})
ax.set_title('Importancia de Features NUEVOS por Horizonte', fontsize=14, fontweight='bold')
ax.set_xlabel('Horizonte')
ax.set_ylabel('')
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/04_heatmap_features_nuevos.png')
plt.close(fig)
print("  [ok] 04_heatmap_features_nuevos.png")

# -- 5.5  Series real vs predicho (semana junio 2025) -----------------------
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()
for idx, h in enumerate(HORIZONTES):
    ax = axes[idx]
    pred_df = pd.read_parquet(f'{OUT_DIR}/pred_test_{h}h.parquet')
    sample_est = pred_df['Estacion'].value_counts().index[0]
    sample = pred_df[pred_df['Estacion'] == sample_est].copy()
    sample['Date'] = pd.to_datetime(sample['Date'])
    sample = sample.sort_values('Date')
    start = pd.Timestamp('2025-06-01')
    end = pd.Timestamp('2025-06-08')
    week = sample[(sample['Date'] >= start) & (sample['Date'] < end)]
    if len(week) > 0:
        ax.plot(week['Date'], week['y_real'], label='Real', color='#264653', linewidth=1.5)
        ax.plot(week['Date'], week['y_pred'], label='Predicho v2', color='#e76f51',
                linewidth=1.5, linestyle='--', alpha=0.85)
        ax.fill_between(week['Date'], week['y_real'], week['y_pred'],
                        alpha=0.12, color='#e76f51')
    ax.set_title(f'+{h}h  |  {sample_est}  |  1-8 Jun 2025', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.set_ylabel('O3 (ppb)')
    ax.spines[['top', 'right']].set_visible(False)
plt.suptitle('Prediccion v2 vs Realidad -- Semana de ejemplo (Jun 2025)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/05_serie_real_vs_pred_v2.png', bbox_inches='tight')
plt.close(fig)
print("  [ok] 05_serie_real_vs_pred_v2.png")

# -- 5.6  Scatter real vs predicho -----------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for idx, h in enumerate(HORIZONTES):
    ax = axes[idx]
    pred_df = pd.read_parquet(f'{OUT_DIR}/pred_test_{h}h.parquet')
    sample = pred_df.sample(n=min(8000, len(pred_df)), random_state=SEED)
    ax.scatter(sample['y_real'], sample['y_pred'], alpha=0.15, s=6, color=PALETTE[idx])
    lims = [0, max(sample['y_real'].max(), sample['y_pred'].max()) * 1.05]
    ax.plot(lims, lims, '--', color='gray', linewidth=1, label='Ideal (y=x)')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('O3 Real (ppb)')
    ax.set_ylabel('O3 Predicho (ppb)')
    r2_h = df_resultados[df_resultados['Horizonte (h)'] == h]['R2'].values[0]
    ax.set_title(f'+{h}h  |  R2 = {r2_h:.3f}', fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_aspect('equal', 'box')
plt.suptitle('Dispersion Real vs Predicho v2 (Test 2025)', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/06_scatter_v2.png', bbox_inches='tight')
plt.close(fig)
print("  [ok] 06_scatter_v2.png")

# -- 5.7  Mejora porcentual de RMSE por horizonte --------------------------
fig, ax = plt.subplots(figsize=(9, 5))
colores_mejora = ['#2a9d8f' if x > 0 else '#e76f51' for x in df_comp['Mejora_RMSE_%']]
bars = ax.bar([f'+{h}h' for h in HORIZONTES], df_comp['Mejora_RMSE_%'],
              color=colores_mejora, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, df_comp['Mejora_RMSE_%']):
    offset = 0.15 if val >= 0 else -0.4
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
            f'{val:+.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Horizonte de Prediccion')
ax.set_ylabel('Mejora en RMSE (%)')
ax.set_title('Mejora Porcentual del Feature Engineering sobre Solo Lags',
             fontsize=13, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/07_mejora_porcentual.png')
plt.close(fig)
print("  [ok] 07_mejora_porcentual.png")

# ===========================================================================
#  ETAPA 6: REPORTE MARKDOWN
# ===========================================================================
print("\n[6/6] Generando reporte comparativo...\n")

ts = datetime.now().strftime('%Y-%m-%d %H:%M')
r = []
r.append(f"# Reporte Multi-Horizonte v2 -- Feature Engineering")
r.append(f"**Generado:** {ts}\n")

r.append("## Que hay de nuevo en v2?")
r.append("Se conservaron los mismos lags base del Script 6 y se agregaron "
         f"**{len(feat_nuevos)} features de ingenieria** en las siguientes categorias:\n")
r.append("| Categoria | Features | Descripcion |")
r.append("|-----------|----------|-------------|")
r.append("| Interacciones fisicoquimicas | `NOX_x_SR`, `TOUT_x_SR`, `O3_NOX_ratio`, `wind_speed` | Codifican la reaccion quimica del O3 |")
r.append("| Tasas de cambio | `O3_diff_1h`, `O3_accel`, `O3_diff_24h`, `TOUT_diff_3h`, `NOX_diff_1h`, `SR_diff_1h` | Direccion y velocidad del cambio |")
r.append("| Rolling stats | `O3_mean_6h`, `O3_std_6h`, `TOUT_mean_6h`, `NOX_mean_6h`, `SR_sum_6h`, `O3_max_24h`, `O3_min_24h`, `O3_range_24h`, `TOUT_max_24h`, `SR_sum_12h` | Resumenes estadisticos de ventana |")
r.append("| Anomalia vs ayer | `O3_vs_yesterday`, `TOUT_vs_yesterday` | Desviacion del patron diario |")
r.append("| Contexto urbano | `is_weekend`, `is_rush_hour` | Patrones de trafico |")
r.append("| Codificacion ciclica | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` | Continuidad temporal |\n")

r.append(f"**Total features:** {len(features)} (vs 52 en v1)\n")

r.append("---\n")
r.append("## Comparativa v1 vs v2\n")

comp_table = df_comp[['Horizonte (h)', 'RMSE_v1', 'RMSE (ppb)', 'Delta_RMSE',
                       'R2_v1', 'R2', 'Delta_R2', 'Mejora_RMSE_%']].copy()
comp_table.columns = ['Horizonte', 'RMSE v1', 'RMSE v2', 'Delta RMSE',
                       'R2 v1', 'R2 v2', 'Delta R2', 'Mejora RMSE %']
r.append(comp_table.to_markdown(index=False))
r.append("")

r.append("\n---\n")
r.append("## Graficas Generadas\n")
graficas = [
    ("01_comparativa_rmse_v1_vs_v2.png", "RMSE comparado: v1 vs v2"),
    ("02_comparativa_r2_v1_vs_v2.png", "R2 comparado: v1 vs v2"),
    ("03_feature_importance_v2_panel.png", "Top 15 features por horizonte (base=verde, nuevo=naranja)"),
    ("04_heatmap_features_nuevos.png", "Heatmap de solo los features nuevos"),
    ("05_serie_real_vs_pred_v2.png", "Series de tiempo: prediccion v2 vs real"),
    ("06_scatter_v2.png", "Dispersion real vs predicho v2"),
    ("07_mejora_porcentual.png", "Mejora porcentual RMSE por horizonte"),
]
for fname, desc in graficas:
    r.append(f"- **`{fname}`** -- {desc}")
r.append("")

r.append("\n---\n")
r.append("## Top 5 Features Nuevos Mas Utiles por Horizonte\n")
for h in HORIZONTES:
    top_new = importancias[h][importancias[h]['Feature'].isin(feat_nuevos)].head(5)
    r.append(f"### +{h}h")
    r.append(top_new.to_markdown(index=False))
    r.append("")

report_path = f'{OUT_DIR}/reporte_v2_feature_engineering.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(r))

print("=" * 65)
print("  PIPELINE v2 COMPLETADO!")
print("=" * 65)
print(f"\n  Graficas:  {FIG_DIR}/")
print(f"  Modelos:   {MODEL_DIR}/")
print(f"  Reporte:   {report_path}")
print()
