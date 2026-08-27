"""
Etapa 2 CRISP-DM — Análisis Descriptivo de Calidad del Aire (SIMA)
Equipo 1 · MA2003B · Tecnológico de Monterrey
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight', 'figure.figsize': (12, 7)})

BASE = 'data/processed/variables'
OUT  = 'results/figures'

# ══════════════════════════════════════════════════════════════════
# PASO 1 — Carga y unión de datos
# ══════════════════════════════════════════════════════════════════
archivos = {
    'O3':   f'{BASE}/O3_clean.parquet',
    'PM2.5': f'{BASE}/PM2.5_clean.parquet',
    'PM10': f'{BASE}/PM10_clean.parquet',
    'SR':   f'{BASE}/SR_clean.parquet',
    'TOUT': f'{BASE}/TOUT_clean.parquet',
}

dfs = {}
print("="*60)
print("PASO 1: Carga de archivos parquet")
print("="*60)

for nombre, ruta in archivos.items():
    df = pd.read_parquet(ruta)
    dfs[nombre] = df
    print(f"\n--- {nombre} ({ruta}) ---")
    print(f"  Filas: {len(df):,}")
    print(f"  Columnas: {list(df.columns)}")
    print(f"  Tipos: {dict(df.dtypes)}")
    print(f"  Estaciones únicas ({df['Estacion'].nunique()}): {sorted(df['Estacion'].unique())}")

# Verificar consistencia de tipos Date y Estacion
print("\n--- Verificación de tipos ---")
for nombre, df in dfs.items():
    print(f"  {nombre}: Date={df['Date'].dtype}, Estacion={df['Estacion'].dtype}")

# Detectar inconsistencias en nombres de estación
todas_estaciones = set()
for nombre, df in dfs.items():
    todas_estaciones.update(df['Estacion'].unique())
print(f"\nEstaciones únicas globales ({len(todas_estaciones)}): {sorted(todas_estaciones)}")

# Inner join secuencial
print("\n--- Inner Join ---")
merged = dfs['O3'].copy()
for nombre in ['PM2.5', 'PM10', 'SR', 'TOUT']:
    merged = merged.merge(dfs[nombre], on=['Date', 'Estacion'], how='inner')
    print(f"  Después de join con {nombre}: {len(merged):,} filas")

# Limpieza final
antes_drop = len(merged)
merged = merged.dropna()
print(f"  Después de dropna: {len(merged):,} (removidas: {antes_drop - len(merged)})")

antes_dedup = len(merged)
merged = merged.drop_duplicates(subset=['Date', 'Estacion'])
print(f"  Después de drop_duplicates: {len(merged):,} (removidas: {antes_dedup - len(merged)})")

# Reportar pérdida
min_filas = min(len(df) for df in dfs.values())
max_filas = max(len(df) for df in dfs.values())
print(f"\n  Archivo más restrictivo: {min_filas:,} filas")
print(f"  Dataset final: {len(merged):,} filas")
print(f"  % pérdida vs más restrictivo: {100*(1 - len(merged)/min_filas):.2f}%")
print(f"  % pérdida vs más grande: {100*(1 - len(merged)/max_filas):.2f}%")

# Guardar para referencia
print(f"\nColumnas finales: {list(merged.columns)}")
print(f"Dimensión final: {merged.shape}")
print(f"Tipos finales:\n{merged.dtypes}")

# Identificar columnas numéricas reales
num_cols = [c for c in merged.columns if c not in ['Date', 'Estacion']]
print(f"Variables numéricas: {num_cols}")
print(merged[num_cols].head(10))

# ══════════════════════════════════════════════════════════════════
# PASO 2 — Análisis Estadístico
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PASO 2: Análisis Estadístico")
print("="*60)

print(f"\nDimensión del dataset: {merged.shape[0]} filas × {merged.shape[1]} columnas")

# --- Tendencia central, dispersión, posición ---
stats_list = []
for col in num_cols:
    s = merged[col]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((s < lower) | (s > upper)).sum()
    pct_outliers = 100 * n_outliers / len(s)
    
    # Moda: redondear a 1 decimal para variables continuas
    moda_val = s.round(1).mode()
    moda_str = moda_val.iloc[0] if len(moda_val) > 0 else np.nan
    
    stats_list.append({
        'Variable': col,
        'N': len(s),
        'Promedio': round(s.mean(), 4),
        'Mediana': round(s.median(), 4),
        'Moda (1 dec)': moda_str,
        'Rango': round(s.max() - s.min(), 4),
        'Varianza': round(s.var(), 4),
        'Desv. Estándar': round(s.std(), 4),
        'Q1': round(q1, 4),
        'Q3': round(q3, 4),
        'IQR': round(iqr, 4),
        'Outliers (n)': n_outliers,
        'Outliers (%)': round(pct_outliers, 2),
    })

stats_df = pd.DataFrame(stats_list)
print("\n--- Estadísticas descriptivas ---")
print(stats_df.to_string(index=False))

# --- Estacion: frecuencia ---
freq = merged['Estacion'].value_counts().reset_index()
freq.columns = ['Estacion', 'Conteo']
freq['% Relativo'] = round(100 * freq['Conteo'] / freq['Conteo'].sum(), 2)
freq = freq.sort_values('Estacion')
print("\n--- Distribución de frecuencia por Estación ---")
print(freq.to_string(index=False))

# --- Correlación Pearson ---
corr_pearson = merged[num_cols].corr(method='pearson')
print("\n--- Matriz de Correlación (Pearson) ---")
print(corr_pearson.round(4).to_string())

# --- Correlación Spearman ---
corr_spearman = merged[num_cols].corr(method='spearman')
print("\n--- Matriz de Correlación (Spearman) ---")
print(corr_spearman.round(4).to_string())

# --- Correlación O3-PM desglosada por Estacion ---
print("\n--- Correlación Pearson O3 vs PM por Estación ---")
corr_by_est = []
for est in sorted(merged['Estacion'].unique()):
    sub = merged[merged['Estacion'] == est]
    if len(sub) > 10:
        r_pm25 = sub['O3'].corr(sub['PM2.5'])
        r_pm10 = sub['O3'].corr(sub['PM10'])
        corr_by_est.append({'Estacion': est, 'O3 vs PM2.5': round(r_pm25, 4), 
                            'O3 vs PM10': round(r_pm10, 4), 'n': len(sub)})
corr_est_df = pd.DataFrame(corr_by_est)
print(corr_est_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════
# PASO 3 — Visualizaciones
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PASO 3: Generación de Visualizaciones")
print("="*60)

# 1. Boxplots por estación
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, var in zip(axes, ['O3', 'PM2.5', 'PM10']):
    merged.boxplot(column=var, by='Estacion', ax=ax, grid=False,
                   boxprops=dict(color='steelblue'), medianprops=dict(color='red'))
    ax.set_title(f'{var} por Estación', fontsize=12)
    ax.set_xlabel('Estación')
    ax.set_ylabel(f'{var}')
    ax.tick_params(axis='x', rotation=45)
fig.suptitle('Distribución de contaminantes por estación de monitoreo', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/boxplot_estaciones.png')
plt.close()
print("  [OK] boxplot_estaciones.png")

# 2. Histogramas con KDE
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, var, color in zip(axes, ['O3', 'PM2.5', 'PM10'], ['#2196F3', '#FF9800', '#4CAF50']):
    data = merged[var].dropna()
    skew_val = data.skew()
    ax.hist(data, bins=50, density=True, alpha=0.6, color=color, edgecolor='white')
    data.plot.kde(ax=ax, color='black', linewidth=2)
    ax.axvline(data.mean(), color='red', linestyle='--', label=f'Media={data.mean():.1f}')
    ax.axvline(data.median(), color='blue', linestyle='--', label=f'Mediana={data.median():.1f}')
    ax.set_title(f'{var} (sesgo = {skew_val:.2f})', fontsize=12)
    ax.set_xlabel(var)
    ax.set_ylabel('Densidad')
    ax.legend(fontsize=8)
fig.suptitle('Distribución de contaminantes — Histograma + KDE', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/histogramas_distribucion.png')
plt.close()
print("  [OK] histogramas_distribucion.png")

# 3. Heatmap de correlación Pearson
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_pearson, dtype=bool), k=1)
sns.heatmap(corr_pearson, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
            mask=mask, cbar_kws={'label': 'Correlación de Pearson'})
ax.set_title('Matriz de Correlación de Pearson', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/heatmap_correlacion.png')
plt.close()
print("  [OK] heatmap_correlacion.png")

# 4. Distribución de registros por estación (barras)
fig, ax = plt.subplots(figsize=(10, 5))
freq_sorted = freq.sort_values('Conteo', ascending=True)
bars = ax.barh(freq_sorted['Estacion'], freq_sorted['Conteo'], color='steelblue', edgecolor='white')
for bar, pct in zip(bars, freq_sorted['% Relativo']):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
            f'{pct:.1f}%', va='center', fontsize=8)
ax.set_xlabel('Número de registros')
ax.set_title('Distribución de registros válidos por estación de monitoreo')
plt.tight_layout()
plt.savefig(f'{OUT}/piechart_estaciones.png')
plt.close()
print("  [OK] piechart_estaciones.png")

# 5. Scatter PM2.5 vs O3 coloreado por SR
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(merged['PM2.5'], merged['O3'], c=merged['SR'], 
                cmap='YlOrRd', alpha=0.4, s=8, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax, label='Radiación Solar (SR)')
ax.set_xlabel('PM2.5 (μg/m³)')
ax.set_ylabel('O3 (ppb)')
ax.set_title('PM2.5 vs O3, coloreado por Radiación Solar')

# Añadir línea de tendencia
z = np.polyfit(merged['PM2.5'], merged['O3'], 1)
p = np.poly1d(z)
x_line = np.linspace(merged['PM2.5'].min(), merged['PM2.5'].min() + (merged['PM2.5'].max() - merged['PM2.5'].min()) * 0.95, 100)
ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.5, 
        label=f'Tendencia lineal (r={merged["PM2.5"].corr(merged["O3"]):.3f})')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/scatter_pm_o3_sr.png')
plt.close()
print("  [OK] scatter_pm_o3_sr.png")

print("\n" + "="*60)
print("TODAS LAS VISUALIZACIONES GENERADAS")
print("="*60)


# ══════════════════════════════════════════════════════════════════
# EXPORTAR TABLAS EN MARKDOWN
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TABLAS EN FORMATO MARKDOWN")
print("="*60)

# Tabla 1: Estadísticas descriptivas (split en 2 para que quepa)
print("\n### Tabla 1a: Tendencia central y dispersión\n")
t1a = stats_df[['Variable', 'N', 'Promedio', 'Mediana', 'Moda (1 dec)', 'Rango', 'Varianza', 'Desv. Estándar']]
print(t1a.to_markdown(index=False))

print("\n### Tabla 1b: Posición no central y outliers\n")
t1b = stats_df[['Variable', 'Q1', 'Q3', 'IQR', 'Outliers (n)', 'Outliers (%)']]
print(t1b.to_markdown(index=False))

print("\n### Tabla 2: Frecuencia por estación\n")
print(freq.to_markdown(index=False))

print("\n### Tabla 3: Correlación Pearson\n")
print(corr_pearson.round(4).to_markdown())

print("\n### Tabla 4: Correlación Spearman\n")
print(corr_spearman.round(4).to_markdown())

print("\n### Tabla 5: Correlación O3 vs PM por estación\n")
print(corr_est_df.to_markdown(index=False))

# Resumen extra de skewness
print("\n### Sesgo (skewness) por variable\n")
for col in num_cols:
    print(f"  {col}: {merged[col].skew():.4f}")

print("\n\n=== SCRIPT COMPLETADO EXITOSAMENTE ===")
