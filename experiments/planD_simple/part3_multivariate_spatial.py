"""
experiments/planD_simple/part3_multivariate_spatial.py
=====================================================
Parte 3 del experimento Plan D (Estadística Multivariada):
Análisis multivariado de las 15 estaciones de monitoreo del SIMA Monterrey.

Objetivos:
1. PCA sobre la matriz temporal (tiempo x 15 estaciones) de O3:
   - % de varianza del primer componente ("Ozono Regional / Fondo Común")
   - Loadings (pesos e interpretaciones espaciales de cada estación)
   - PCA sobre perfiles diurnos medios (24 horas x 15 estaciones)
2. Clustering jerárquico y k-means de estaciones:
   - Construcción de perfiles multivariados por estación (diurno O3, estacional O3, diurno NOX, diurno TOUT)
   - Comparación con las 4 regiones definidas a priori:
     * Poniente/Sierra: NO2, NO3, SO, SO2
     * Centro/Norte: CE, NO, NTE, NTE2
     * Noreste: NE, NE2, NE3
     * Sur: SE, SE2, SE3, SUR
   - Métricas: ARI (Adjusted Rand Index) y Coeficiente de Silueta.
3. Matriz de correlación vs distancia espacial:
   - Correlación de Pearson de O3 entre los 105 pares de estaciones.
   - Distancia geográfica Haversine (km) a partir de STATION_COORDS.
   - Ajuste de modelo de decaimiento espacial exponencial: r(d) = r0 * exp(-d / d0).
   - Estimación de la escala de correlación regional (e-folding distance).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import load_panel, ALL_STATIONS, STATION_COORDS, SEED

OUT_DIR = os.path.join(ROOT, 'experiments', 'planD_simple')
FIGURES_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Regiones manuales de referencia (definidas a mano)
# ---------------------------------------------------------------------------
MANUAL_REGIONS = {
    'NO2': 'Poniente/Sierra', 'NO3': 'Poniente/Sierra', 'SO': 'Poniente/Sierra', 'SO2': 'Poniente/Sierra',
    'CE': 'Centro/Norte', 'NO': 'Centro/Norte', 'NTE': 'Centro/Norte', 'NTE2': 'Centro/Norte',
    'NE': 'Noreste', 'NE2': 'Noreste', 'NE3': 'Noreste',
    'SE': 'Sur', 'SE2': 'Sur', 'SE3': 'Sur', 'SUR': 'Sur'
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radio de la Tierra en km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ---------------------------------------------------------------------------
# 1. PCA Temporal y Diurno
# ---------------------------------------------------------------------------
def run_pca_analysis(panel: pd.DataFrame):
    print("\n--- Ejecutando PCA Multivariado de Estaciones ---")
    
    # 1.1 Matriz temporal (Date x 15 Estaciones) de O3
    piv = panel.pivot(index='Date', columns='Estacion', values='O3')[ALL_STATIONS]
    print(f"Dimensiones de matriz temporal de O3: {piv.shape}")
    
    # Filas con cobertura suficiente (al menos 12 de 15 estaciones con dato)
    valid_rows = piv.count(axis=1) >= 12
    piv_sub = piv[valid_rows].copy()
    
    # Imputar faltantes residuales con la media por estación para PCA
    piv_imputed = piv_sub.fillna(piv_sub.mean())
    print(f"Filas seleccionadas para PCA temporal (>=12 estaciones): {len(piv_imputed)} ({len(piv_imputed)/len(piv)*100:.1f}%)")
    
    # Estandarización de las estaciones
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(piv_imputed)
    
    pca_time = PCA()
    pca_time.fit(X_scaled)
    
    var_exp = pca_time.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_exp)
    
    print("\nPCA Temporal (Tiempo x 15 Estaciones):")
    for i in range(min(5, len(var_exp))):
        print(f"  PC{i+1}: {var_exp[i]:.2f}% varianza (acumulada: {cum_var[i]:.2f}%)")
        
    # Loadings (autovectores) de PC1 y PC2
    loadings_time = pd.DataFrame(
        pca_time.components_[:3].T,
        index=ALL_STATIONS,
        columns=['Load_PC1_Regional', 'Load_PC2', 'Load_PC3']
    )
    loadings_time['Manual_Region'] = loadings_time.index.map(MANUAL_REGIONS)
    loadings_time.to_csv(os.path.join(OUT_DIR, 'pca_time_loadings.csv'))
    
    # 1.2 PCA sobre perfiles diurnos medios (24 horas x 15 estaciones)
    panel['hour'] = panel['Date'].dt.hour
    diurnal_mean = panel.groupby(['Estacion', 'hour'])['O3'].mean().unstack(level='Estacion')[ALL_STATIONS]
    # Matriz: 24 horas x 15 estaciones
    scaler_diurnal = StandardScaler()
    X_diurnal_sc = scaler_diurnal.fit_transform(diurnal_mean)
    
    pca_diurnal = PCA()
    pca_diurnal.fit(X_diurnal_sc)
    
    var_exp_d = pca_diurnal.explained_variance_ratio_ * 100
    cum_var_d = np.cumsum(var_exp_d)
    
    print("\nPCA sobre Perfiles Diurnos Medios (24h x 15 Estaciones):")
    for i in range(min(4, len(var_exp_d))):
        print(f"  PC{i+1}_diurno: {var_exp_d[i]:.2f}% varianza (acumulada: {cum_var_d[i]:.2f}%)")
        
    loadings_diurnal = pd.DataFrame(
        pca_diurnal.components_[:2].T,
        index=ALL_STATIONS,
        columns=['Load_Diurnal_PC1', 'Load_Diurnal_PC2']
    )
    loadings_diurnal.to_csv(os.path.join(OUT_DIR, 'pca_diurnal_loadings.csv'))
    
    # 1.3 Gráficas de PCA
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    axes[0].bar(range(1, 16), var_exp, color='tab:blue', alpha=0.7, label='% Varianza Individual')
    axes[0].plot(range(1, 16), cum_var, color='tab:red', marker='o', lw=2, label='% Varianza Acumulada')
    axes[0].axhline(var_exp[0], color='tab:blue', linestyle=':', alpha=0.7)
    axes[0].set_xlabel('Componente Principal')
    axes[0].set_ylabel('% Varianza Explicada')
    axes[0].set_title(f'Scree Plot O3 Temporal (PC1 "Ozono Regional" = {var_exp[0]:.1f}%)')
    axes[0].set_xticks(range(1, 16))
    axes[0].grid(True, linestyle=':')
    axes[0].legend()
    
    # Loadings PC1 por estación
    colors_reg = {'Poniente/Sierra': 'tab:green', 'Centro/Norte': 'tab:blue', 'Noreste': 'tab:orange', 'Sur': 'tab:purple'}
    bar_cols = [colors_reg[MANUAL_REGIONS[s]] for s in ALL_STATIONS]
    
    axes[1].bar(ALL_STATIONS, loadings_time['Load_PC1_Regional'], color=bar_cols, alpha=0.85)
    axes[1].set_xlabel('Estación')
    axes[1].set_ylabel('Loading en PC1 (Ozono Regional)')
    axes[1].set_title('Loadings en PC1: Contribución al Fondo Regional de Ozono')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, linestyle=':')
    
    # Leyenda de regiones
    from matplotlib.patches import Patch
    leg_patches = [Patch(color=c, label=r) for r, c in colors_reg.items()]
    axes[1].legend(handles=leg_patches, title='Región Manual', loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_pca_ozono_regional.png'), dpi=300)
    plt.close()
    print("Gráfica fig_pca_ozono_regional.png guardada.")
    
    return var_exp, loadings_time, diurnal_mean

# ---------------------------------------------------------------------------
# 2. Clustering Jerárquico y K-Means vs Regiones Manuales
# ---------------------------------------------------------------------------
def run_clustering_analysis(panel: pd.DataFrame, diurnal_mean: pd.DataFrame):
    print("\n--- Ejecutando Clustering de Estaciones ---")
    
    # Construcción del vector de características multivariadas por estación:
    # 1. Perfil diurno medio de O3 (24 valores, normalizado a suma=1 o estandarizado)
    # 2. Perfil estacional medio de O3 (12 meses)
    # 3. Perfil diurno medio de NOX (24 valores)
    # 4. Perfil diurno medio de TOUT (24 valores)
    panel['month'] = panel['Date'].dt.month
    panel['hour'] = panel['Date'].dt.hour
    
    o3_diurnal = panel.groupby(['Estacion', 'hour'])['O3'].mean().unstack(level='hour')[range(24)]
    o3_seasonal = panel.groupby(['Estacion', 'month'])['O3'].mean().unstack(level='month')[range(1, 13)]
    nox_diurnal = panel.groupby(['Estacion', 'hour'])['NOX'].mean().unstack(level='hour')[range(24)]
    tout_diurnal = panel.groupby(['Estacion', 'hour'])['TOUT'].mean().unstack(level='hour')[range(24)]
    
    o3_diurnal.columns = [f'o3_h_{c}' for c in o3_diurnal.columns]
    o3_seasonal.columns = [f'o3_m_{c}' for c in o3_seasonal.columns]
    nox_diurnal.columns = [f'nox_h_{c}' for c in nox_diurnal.columns]
    tout_diurnal.columns = [f'tout_h_{c}' for c in tout_diurnal.columns]
    
    # Estandarización de cada bloque para igual ponderación
    def std_block(df_b):
        return (df_b - df_b.mean().mean()) / (df_b.std().mean() + 1e-6)
        
    X_features = pd.concat([
        std_block(o3_diurnal),
        std_block(o3_seasonal),
        std_block(nox_diurnal),
        std_block(tout_diurnal)
    ], axis=1).loc[ALL_STATIONS]
    
    print(f"Matriz de perfiles de estaciones: {X_features.shape} (15 estaciones x {X_features.shape[1]} variables)")
    
    # 2.1 Clustering Jerárquico (Ward)
    linkage_matrix = linkage(X_features, method='ward')
    agg_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')
    labels_hier = agg_cluster.fit_predict(X_features)
    
    # 2.2 K-Means (k=4)
    kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=50)
    labels_km = kmeans.fit_predict(X_features)
    
    # Mapeo de etiquetas manuales a enteros
    reg_unique = list(sorted(set(MANUAL_REGIONS.values())))
    reg_to_id = {r: i for i, r in enumerate(reg_unique)}
    labels_manual = np.array([reg_to_id[MANUAL_REGIONS[s]] for s in ALL_STATIONS])
    
    # Métricas de evaluación
    ari_hier = adjusted_rand_score(labels_manual, labels_hier)
    ari_km = adjusted_rand_score(labels_manual, labels_km)
    
    sil_hier = silhouette_score(X_features, labels_hier)
    sil_km = silhouette_score(X_features, labels_km)
    sil_manual = silhouette_score(X_features, labels_manual)
    
    print("\nResultados de Comparación de Agrupamientos:")
    print(f"  Regiones Manuales a priori: Silueta = {sil_manual:.3f}")
    print(f"  Clustering Jerárquico (Ward, k=4): ARI = {ari_hier:.3f}, Silueta = {sil_hier:.3f}")
    print(f"  K-Means (k=4): ARI = {ari_km:.3f}, Silueta = {sil_km:.3f}")
    
    df_clust = pd.DataFrame({
        'Estacion': ALL_STATIONS,
        'Region_Manual': [MANUAL_REGIONS[s] for s in ALL_STATIONS],
        'Cluster_Jerarquico': labels_hier,
        'Cluster_KMeans': labels_km
    })
    df_clust.to_csv(os.path.join(OUT_DIR, 'clustering_stations_comparison.csv'), index=False)
    
    # 2.3 Visualizaciones
    # Dendrograma
    plt.figure(figsize=(10, 6))
    dendrogram(
        linkage_matrix,
        labels=ALL_STATIONS,
        leaf_rotation=0,
        leaf_font_size=11,
        color_threshold=linkage_matrix[-3, 2] # Umbral para 4 clusters
    )
    plt.title('Dendrograma de Clustering Jerárquico (Ward) — 15 Estaciones SIMA', fontsize=12, fontweight='bold')
    plt.xlabel('Estación')
    plt.ylabel('Distancia Euclidiana / Disimilitud (Varianza dentro de grupo)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_dendrograma_estaciones.png'), dpi=300)
    plt.close()
    print("Gráfica fig_dendrograma_estaciones.png guardada.")
    
    return ari_hier, sil_hier, ari_km, sil_km, sil_manual

# ---------------------------------------------------------------------------
# 3. Matriz de Correlación vs Distancia Espacial
# ---------------------------------------------------------------------------
def run_spatial_correlation_analysis(panel: pd.DataFrame):
    print("\n--- Ejecutando Análisis de Autocorrelación Espacial vs Distancia ---")
    
    piv = panel.pivot(index='Date', columns='Estacion', values='O3')[ALL_STATIONS]
    corr_matrix = piv.corr(method='pearson')
    
    pairs = []
    for i in range(len(ALL_STATIONS)):
        for j in range(i + 1, len(ALL_STATIONS)):
            s1 = ALL_STATIONS[i]
            s2 = ALL_STATIONS[j]
            r = corr_matrix.loc[s1, s2]
            
            lat1, lon1, elev1 = STATION_COORDS[s1]
            lat2, lon2, elev2 = STATION_COORDS[s2]
            dist_km = haversine(lat1, lon1, lat2, lon2)
            delev = abs(elev1 - elev2)
            
            same_reg = (MANUAL_REGIONS[s1] == MANUAL_REGIONS[s2])
            
            pairs.append({
                'station_1': s1, 'station_2': s2,
                'pair': f"{s1}-{s2}",
                'correlation': r,
                'distance_km': dist_km,
                'delta_elev_m': delev,
                'same_manual_region': same_reg
            })
            
    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(os.path.join(OUT_DIR, 'station_distance_correlation.csv'), index=False)
    
    # Ajuste de modelo de decaimiento exponencial: r(d) = r0 * exp(-d / d0)
    def exp_decay(d, r0, d0):
        return r0 * np.exp(-d / d0)
        
    popt, pcov = curve_fit(exp_decay, df_pairs['distance_km'], df_pairs['correlation'], p0=[0.9, 30.0])
    r0_fit, d0_fit = popt
    r0_err, d0_err = np.sqrt(np.diag(pcov))
    
    print(f"Modelo de Decaimiento Espacial Ajustado: r(d) = r0 * exp(-d / d0)")
    print(f"  r0 (correlación en origen / pepita): {r0_fit:.3f} ± {r0_err:.3f}")
    print(f"  d0 (escala espacial de e-folding): {d0_fit:.1f} ± {d0_err:.1f} km")
    
    # Gráfica: Correlación vs Distancia
    plt.figure(figsize=(9, 6))
    
    # Puntos intra-región vs inter-región
    intra = df_pairs[df_pairs['same_manual_region']]
    inter = df_pairs[~df_pairs['same_manual_region']]
    
    plt.scatter(inter['distance_km'], inter['correlation'], color='tab:blue', alpha=0.6, s=50, label='Pares Inter-Región')
    plt.scatter(intra['distance_km'], intra['correlation'], color='tab:red', alpha=0.8, s=60, marker='^', label='Pares Intra-Región')
    
    # Curva ajustada
    d_grid = np.linspace(0, df_pairs['distance_km'].max() * 1.05, 100)
    plt.plot(d_grid, exp_decay(d_grid, r0_fit, d0_fit), 'k-', lw=2.5,
             label=f'Ajuste Exponencial: $r(d) = {r0_fit:.2f} e^{{-d / {d0_fit:.1f}}}$')
    
    # Línea de media distancia de decorrelación
    plt.axvline(d0_fit, color='gray', linestyle='--', alpha=0.7, label=f'Escala $d_0$ = {d0_fit:.1f} km')
    
    plt.xlabel('Distancia Geográfica (km, Haversine)')
    plt.ylabel('Correlación de Pearson en Concentración de Ozono Horario ($r$)')
    plt.title('Decaimiento Espacial de la Correlación de Ozono entre Estaciones (SIMA)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_correlacion_vs_distancia.png'), dpi=300)
    plt.close()
    print("Gráfica fig_correlacion_vs_distancia.png guardada.")
    
    # Gráfica adicional: Matriz de Correlación ordenada geográficamente
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='viridis', vmin=0.4, vmax=1.0)
    plt.title('Matriz de Correlación de Pearson de O3 entre las 15 Estaciones SIMA')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_matriz_correlacion_estaciones.png'), dpi=300)
    plt.close()
    print("Gráfica fig_matriz_correlacion_estaciones.png guardada.")

def run_part3():
    panel = load_panel()
    var_exp, loadings_time, diurnal_mean = run_pca_analysis(panel)
    ari_h, sil_h, ari_k, sil_k, sil_m = run_clustering_analysis(panel, diurnal_mean)
    run_spatial_correlation_analysis(panel)

if __name__ == '__main__':
    run_part3()
