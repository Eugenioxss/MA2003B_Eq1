"""
experiments/planD_simple/part1_hourly_parsimonious.py
====================================================
Parte 1 del experimento Plan D (Estadística Multivariada):
Modelos horarios parsimoniosos e interpretables para h = 1, 8, 24, 48.

Comparación de:
1. Climatología (referencia no paramétrica)
2. Regresión Ridge (~13 features físicos/temporales + dummies de estación)
3. GAM (pygam: splines univariados + cyclic spline de hora + tensor product de viento + factor estación)
4. LightGBM Parsimonioso (las mismas 13 features, n_jobs=2)
5. LightGBM v2 (referencia del contrato)

Manejo riguroso de valores faltantes (NaN):
- Reporte de cobertura para cada modelo en el conjunto estándar (2025).
- Evaluación comparativa en la intersección estricta de casos completos.
- Registro en results_registry.csv con log_result().
- Generación de gráficas:
  a) Complejidad vs Desempeño (RMSE vs nº parámetros/features)
  b) Efectos parciales e interpretables del GAM
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from pygam import LinearGAM, s, f, te
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import (
    load_panel, add_targets, evaluate_test, log_result,
    MODEL_STATIONS, HORIZONS, SEED, metrics, climatology_pred
)

OUT_DIR = os.path.join(ROOT, 'experiments', 'planD_simple')
os.makedirs(OUT_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Construcción de Features Parsimoniosos (~13 features)
# ---------------------------------------------------------------------------
def build_parsimonious_features(panel: pd.DataFrame, stations=MODEL_STATIONS) -> tuple[pd.DataFrame, list[str]]:
    print("Construyendo features parsimoniosos...")
    t0 = time.time()
    df = panel[panel['Estacion'].isin(stations)].copy()
    df = df.sort_values(['Estacion', 'Date']).reset_index(drop=True)
    out = []
    
    for est, g in df.groupby('Estacion', sort=False):
        g = g.copy()
        s_ = lambda c, k: g[c].shift(k)
        
        # 1. Memoria temporal de O3
        g['O3_lag_1'] = s_('O3', 1)
        g['O3_lag_24'] = s_('O3', 24)
        o3s = s_('O3', 1)
        g['O3_max_24h'] = o3s.rolling(24, min_periods=6).max()
        
        # 2. Variables meteorológicas y fotoquímicas clave
        g['TOUT_lag_1'] = s_('TOUT', 1)
        g['SR_lag_1'] = s_('SR', 1)
        g['NOX_lag_1'] = s_('NOX', 1)
        g['U_Wind_lag_1'] = s_('U_Wind', 1)
        g['V_Wind_lag_1'] = s_('V_Wind', 1)
        
        # 3. Componentes cíclicos armónicos
        g['hour'] = g['Date'].dt.hour
        g['month'] = g['Date'].dt.month
        g['hour_sin'] = np.sin(2 * np.pi * g['hour'] / 24).astype('float32')
        g['hour_cos'] = np.cos(2 * np.pi * g['hour'] / 24).astype('float32')
        g['month_sin'] = np.sin(2 * np.pi * g['month'] / 12).astype('float32')
        g['month_cos'] = np.cos(2 * np.pi * g['month'] / 12).astype('float32')
        
        out.append(g)
        
    df = pd.concat(out, ignore_index=True)
    df = add_targets(df, horizons=[1, 8, 24, 48])
    
    # Codificación de estación para modelos lineales y GAM
    st_map = {s: i for i, s in enumerate(stations)}
    df['st_idx'] = df['Estacion'].map(st_map).astype(int)
    
    numeric_feats = [
        'O3_lag_1', 'O3_lag_24', 'O3_max_24h',
        'TOUT_lag_1', 'SR_lag_1', 'NOX_lag_1',
        'U_Wind_lag_1', 'V_Wind_lag_1',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ]
    print(f"Features generados en {time.time()-t0:.2f}s. Features numéricos: {len(numeric_feats)}")
    return df, numeric_feats

# ---------------------------------------------------------------------------
# 2. Entrenamiento y Evaluación
# ---------------------------------------------------------------------------
def run_part1():
    panel = load_panel(MODEL_STATIONS)
    df, num_feats = build_parsimonious_features(panel, MODEL_STATIONS)
    
    train_mask = df['Date'].dt.year <= 2023
    val_mask = df['Date'].dt.year == 2024
    test_mask = df['Date'].dt.year == 2025
    
    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")
    
    # One-hot encoding de estación para Ridge
    st_dummies_tr = pd.get_dummies(train_df['Estacion'], drop_first=True, dtype=float)
    st_dummies_te = pd.get_dummies(test_df['Estacion'], drop_first=True, dtype=float)
    # Asegurar que las columnas coincidan exactamente
    for col in st_dummies_tr.columns:
        if col not in st_dummies_te.columns:
            st_dummies_te[col] = 0.0
    st_dummies_te = st_dummies_te[st_dummies_tr.columns]
    
    X_tr_ridge_df = pd.concat([train_df[num_feats], st_dummies_tr], axis=1)
    X_te_ridge_df = pd.concat([test_df[num_feats], st_dummies_te], axis=1)
    
    # GAM feature order:
    # 0: O3_lag_1, 1: TOUT_lag_1, 2: hour, 3: U_Wind_lag_1, 4: V_Wind_lag_1,
    # 5: st_idx, 6: O3_lag_24, 7: SR_lag_1, 8: NOX_lag_1, 9: month
    gam_cols = [
        'O3_lag_1', 'TOUT_lag_1', 'hour', 'U_Wind_lag_1', 'V_Wind_lag_1',
        'st_idx', 'O3_lag_24', 'SR_lag_1', 'NOX_lag_1', 'month'
    ]
    
    horizons = [1, 8, 24, 48]
    
    # Dataframe para guardar comparaciones
    records_std = []
    records_inter = []
    gam_models = {}
    
    for h in horizons:
        tcol = f'Target_{h}h'
        print(f"\n==========================================")
        print(f"PROCESANDO HORIZONTE h = {h}")
        print(f"==========================================")
        
        # 1. Climatología
        tt_test = test_df['Date'] + pd.Timedelta(hours=h)
        pred_clim = climatology_pred(test_df['Estacion'], tt_test)
        m_clim = evaluate_test(test_df, pred_clim, h)
        log_result('planD_simple', 'climatology', h, m_clim, 'Baseline Climatología (estacion x mes x hora)')
        records_std.append({'horizon': h, 'model': 'Climatología', 'coverage': m_clim['coverage'], **m_clim})
        print(f"[h={h}] Climatología: RMSE = {m_clim['RMSE']:.2f}, R2 = {m_clim['R2']:.3f}, coverage = {m_clim['coverage']:.3f}")
        
        # 2. Ridge Regression
        # Filtrar entrenamiento con filas completas
        train_h = train_df.dropna(subset=num_feats + [tcol])
        X_tr_ridge = X_tr_ridge_df.loc[train_h.index].values
        y_tr_ridge = train_h[tcol].values
        
        # Estandarización de features para Ridge
        mean_r = X_tr_ridge.mean(axis=0)
        std_r = X_tr_ridge.std(axis=0)
        std_r[std_r == 0] = 1.0
        X_tr_ridge_sc = (X_tr_ridge - mean_r) / std_r
        
        ridge = RidgeCV(alphas=np.logspace(-1, 5, 10))
        ridge.fit(X_tr_ridge_sc, y_tr_ridge)
        print(f"[h={h}] Ridge entrenado con alpha={ridge.alpha_:.2f} en N={len(X_tr_ridge)}")
        
        # Predicción en test
        test_valid_mask_ridge = test_df[num_feats].notna().all(axis=1)
        pred_ridge = np.full(len(test_df), np.nan, dtype=float)
        X_te_ridge = X_te_ridge_df.loc[test_valid_mask_ridge].values
        X_te_ridge_sc = (X_te_ridge - mean_r) / std_r
        pred_ridge[test_valid_mask_ridge.values] = ridge.predict(X_te_ridge_sc)
        
        m_ridge = evaluate_test(test_df, pred_ridge, h)
        log_result('planD_simple', 'ridge_parsimonious', h, m_ridge, f'Ridge 13 feats + dummies (cov {m_ridge["coverage"]:.3f})')
        records_std.append({'horizon': h, 'model': 'Ridge (13 feats)', 'coverage': m_ridge['coverage'], **m_ridge})
        print(f"[h={h}] Ridge: RMSE = {m_ridge['RMSE']:.2f}, R2 = {m_ridge['R2']:.3f}, coverage = {m_ridge['coverage']:.3f}")
        
        # 3. GAM (pygam)
        # Submuestreo balanceado de entrenamiento completo para ajuste rápido y numéricamente estable
        train_h_gam = train_df.dropna(subset=gam_cols + [tcol])
        n_gam_sample = min(60000, len(train_h_gam))
        sample_idx = train_h_gam.sample(n=n_gam_sample, random_state=SEED).index
        X_tr_gam = train_df.loc[sample_idx, gam_cols].values
        y_tr_gam = train_df.loc[sample_idx, tcol].values
        
        gam_terms = (
            s(0, n_splines=12) +                                # O3_lag_1
            s(1, n_splines=12) +                                # TOUT_lag_1
            s(2, basis='cp', edge_knots=[0, 24], n_splines=10) +# hour cíclico
            te(3, 4, n_splines=8) +                             # U_Wind x V_Wind
            f(5) +                                              # estación factor
            s(6, n_splines=12) +                                # O3_lag_24
            s(7, n_splines=10) +                                # SR_lag_1
            s(8, n_splines=10) +                                # NOX_lag_1
            s(9, basis='cp', edge_knots=[1, 13], n_splines=8)   # month cíclico
        )
        gam = LinearGAM(gam_terms).fit(X_tr_gam, y_tr_gam)
        gam_models[h] = gam
        
        # Predicción en test
        test_valid_mask_gam = test_df[gam_cols].notna().all(axis=1)
        pred_gam = np.full(len(test_df), np.nan, dtype=float)
        X_te_gam = test_df.loc[test_valid_mask_gam, gam_cols].values
        pred_gam[test_valid_mask_gam.values] = gam.predict(X_te_gam)
        
        m_gam = evaluate_test(test_df, pred_gam, h)
        log_result('planD_simple', 'gam_parsimonious', h, m_gam, f'GAM pygam (cov {m_gam["coverage"]:.3f})')
        records_std.append({'horizon': h, 'model': 'GAM (Splines)', 'coverage': m_gam['coverage'], **m_gam})
        print(f"[h={h}] GAM: RMSE = {m_gam['RMSE']:.2f}, R2 = {m_gam['R2']:.3f}, coverage = {m_gam['coverage']:.3f}")
        
        # 4. LightGBM Parsimonioso
        # Mismas 13 features numéricas + categoría de estación
        lgb_cols = num_feats + ['Estacion']
        tr_lgb = train_df[lgb_cols].copy()
        tr_lgb['Estacion'] = tr_lgb['Estacion'].astype('category')
        y_tr_lgb = train_df[tcol]
        
        val_lgb = val_df[lgb_cols].copy()
        val_lgb['Estacion'] = val_lgb['Estacion'].astype('category')
        y_val_lgb = val_df[tcol]
        
        te_lgb = test_df[lgb_cols].copy()
        te_lgb['Estacion'] = te_lgb['Estacion'].astype('category')
        
        valid_tr_target = y_tr_lgb.notna()
        valid_val_target = y_val_lgb.notna()
        
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=2,
            random_state=SEED,
            verbose=-1
        )
        lgb_model.fit(
            tr_lgb[valid_tr_target], y_tr_lgb[valid_tr_target],
            eval_set=[(val_lgb[valid_val_target], y_val_lgb[valid_val_target])],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        pred_lgb = lgb_model.predict(te_lgb)
        m_lgb = evaluate_test(test_df, pred_lgb, h)
        log_result('planD_simple', 'lgb_parsimonious', h, m_lgb, 'LightGBM 13 feats (n_jobs=2)')
        records_std.append({'horizon': h, 'model': 'LightGBM (13 feats)', 'coverage': m_lgb['coverage'], **m_lgb})
        print(f"[h={h}] LightGBM Parsimonioso: RMSE = {m_lgb['RMSE']:.2f}, R2 = {m_lgb['R2']:.3f}, coverage = {m_lgb['coverage']:.3f}")
        
        # 5. Evaluación en la INTERSECCIÓN ESTRICTA de filas completas
        inter_mask = (
            test_df[tcol].notna() &
            test_valid_mask_ridge &
            test_valid_mask_gam &
            np.isfinite(pred_lgb)
        )
        print(f"[h={h}] Filas en intersección estricta: {inter_mask.sum()} de {test_df[tcol].notna().sum()} válidas ({inter_mask.mean():.3f})")
        
        yt_inter = test_df.loc[inter_mask, tcol].values
        clim_inter = pred_clim[inter_mask]
        hours_inter = tt_test[inter_mask].dt.hour.values
        
        m_clim_i = metrics(yt_inter, pred_clim[inter_mask], clim=clim_inter, hours=hours_inter)
        m_ridge_i = metrics(yt_inter, pred_ridge[inter_mask], clim=clim_inter, hours=hours_inter)
        m_gam_i = metrics(yt_inter, pred_gam[inter_mask], clim=clim_inter, hours=hours_inter)
        m_lgb_i = metrics(yt_inter, pred_lgb[inter_mask], clim=clim_inter, hours=hours_inter)
        
        for name, m_i in [
            ('Climatología', m_clim_i),
            ('Ridge (13 feats)', m_ridge_i),
            ('GAM (Splines)', m_gam_i),
            ('LightGBM (13 feats)', m_lgb_i)
        ]:
            records_inter.append({'horizon': h, 'model': name, 'n_eval': int(inter_mask.sum()), **m_i})
            
    # Guardar tablas de resultados en CSV
    df_std = pd.DataFrame(records_std)
    df_inter = pd.DataFrame(records_inter)
    df_std.to_csv(os.path.join(OUT_DIR, 'hourly_standard_results.csv'), index=False)
    df_inter.to_csv(os.path.join(OUT_DIR, 'hourly_intersection_results.csv'), index=False)
    print("\nResultados horarios guardados exitosamente.")
    
    # -----------------------------------------------------------------------
    # 3. Gráficas de Parte 1
    # -----------------------------------------------------------------------
    # Gráfica 1: Complejidad vs Desempeño
    # Complejidad estimada (nº parámetros/grados de libertad aproximados):
    # Climatología: 13 * 12 * 24 = 3744 celdas empíricas
    # Ridge: 25 parámetros lineales
    # GAM: ~120 grados de libertad efectivos (splines)
    # LightGBM: ~31 leaves * ~150 trees = ~4650 parámetros
    # LightGBM v2 (referencia README): ~63 leaves * 300 trees = ~18900
    
    plt.figure(figsize=(10, 6))
    complexity_map = {
        'Ridge (13 feats)': 25,
        'GAM (Splines)': 120,
        'LightGBM (13 feats)': 4650,
        'Climatología': 3744
    }
    
    colors = {'Climatología': 'gray', 'Ridge (13 feats)': 'tab:blue', 'GAM (Splines)': 'tab:green', 'LightGBM (13 feats)': 'tab:red'}
    markers = {'Climatología': 's', 'Ridge (13 feats)': 'o', 'GAM (Splines)': '^', 'LightGBM (13 feats)': 'D'}
    
    # Graficar por horizonte
    for h in horizons:
        sub = df_inter[df_inter['horizon'] == h]
        for _, row in sub.iterrows():
            m_name = row['model']
            comp = complexity_map[m_name]
            plt.scatter(comp, row['RMSE'], color=colors[m_name], marker=markers[m_name], s=100,
                        label=m_name if h == 1 else "")
            plt.text(comp * 1.08, row['RMSE'], f"h={h}", fontsize=9, verticalalignment='center')
            
    # Líneas conectoras por horizonte
    for h in horizons:
        sub = df_inter[df_inter['horizon'] == h].copy()
        sub['comp'] = sub['model'].map(complexity_map)
        sub = sub.sort_values('comp')
        plt.plot(sub['comp'], sub['RMSE'], linestyle='--', alpha=0.4, label=f'Trayectoria h={h}' if h==1 else "")
        
    plt.xscale('log')
    plt.xlabel('Complejidad del Modelo (Nº aproximado de parámetros / DoF, escala log)')
    plt.ylabel('RMSE en Test 2025 (ppb) — Intersección común')
    plt.title('Compromiso Complejidad vs Desempeño (Trade-off Parsimonia)')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=10, label='Ridge (25 params)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='tab:green', markersize=10, label='GAM Splines (~120 DoF)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='tab:red', markersize=10, label='LightGBM 13 feats (~4.6k DoF)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Climatología (3.7k celdas)'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_complejidad_vs_desempeno.png'), dpi=300)
    plt.close()
    print("Gráfica fig_complejidad_vs_desempeno.png guardada.")
    
    # Gráfica 2: Efectos parciales del GAM (para h=1)
    gam1 = gam_models[1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    # 1. Efecto O3_lag_1 (term 0)
    XX_0 = gam1.generate_X_grid(term=0)
    pdep_0, confi_0 = gam1.partial_dependence(term=0, X=XX_0, width=0.95)
    axes[0].plot(XX_0[:, 0], pdep_0, color='tab:blue', lw=2)
    axes[0].fill_between(XX_0[:, 0], confi_0[:, 0], confi_0[:, 1], color='tab:blue', alpha=0.2)
    axes[0].set_title('Efecto parcial: $O_3(t-1)$')
    axes[0].set_xlabel('O3 lag 1 (ppb)')
    axes[0].set_ylabel('Efecto en Target 1h (ppb)')
    axes[0].grid(True, linestyle=':')
    
    # 2. Efecto TOUT_lag_1 (term 1)
    XX_1 = gam1.generate_X_grid(term=1)
    pdep_1, confi_1 = gam1.partial_dependence(term=1, X=XX_1, width=0.95)
    axes[1].plot(XX_1[:, 1], pdep_1, color='tab:red', lw=2)
    axes[1].fill_between(XX_1[:, 1], confi_1[:, 0], confi_1[:, 1], color='tab:red', alpha=0.2)
    axes[1].set_title('Efecto parcial: Temperatura ($TOUT_{t-1}$)')
    axes[1].set_xlabel('Temperatura (°C)')
    axes[1].set_ylabel('Efecto en Target 1h (ppb)')
    axes[1].grid(True, linestyle=':')
    
    # 3. Efecto Cíclico de la Hora (term 2)
    XX_2 = gam1.generate_X_grid(term=2)
    pdep_2, confi_2 = gam1.partial_dependence(term=2, X=XX_2, width=0.95)
    axes[2].plot(XX_2[:, 2], pdep_2, color='tab:orange', lw=2)
    axes[2].fill_between(XX_2[:, 2], confi_2[:, 0], confi_2[:, 1], color='tab:orange', alpha=0.2)
    axes[2].set_title('Efecto Diurno Cíclico (Hora del día)')
    axes[2].set_xlabel('Hora (0 a 23 h)')
    axes[2].set_ylabel('Efecto en Target 1h (ppb)')
    axes[2].grid(True, linestyle=':')
    
    # 4. Efecto de Estación (Factor term 5)
    st_names = MODEL_STATIONS
    XX_5 = np.zeros((len(st_names), len(gam_cols)))
    XX_5[:, 5] = np.arange(len(st_names))
    pdep_5, confi_5 = gam1.partial_dependence(term=4, X=XX_5, width=0.95) # Note: term indexing in pygam
    # Check term index for factor
    term_idx_st = None
    for idx_t, term in enumerate(gam1.terms):
        if term.isintercept:
            continue
        if hasattr(term, 'feature') and term.feature == 5:
            term_idx_st = idx_t
            break
    if term_idx_st is not None:
        pdep_5, confi_5 = gam1.partial_dependence(term=term_idx_st, X=XX_5, width=0.95)
        axes[3].bar(st_names, pdep_5, yerr=np.abs(confi_5[:, 1] - pdep_5), capsize=4, color='tab:purple', alpha=0.7)
        axes[3].set_title('Efecto de Sitio (Estación SIMA)')
        axes[3].set_xlabel('Estación')
        axes[3].set_ylabel('Efecto relativo (ppb)')
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].grid(True, linestyle=':')
    
    # 5. Efecto Radiación Solar (term 7)
    term_idx_sr = None
    for idx_t, term in enumerate(gam1.terms):
        if hasattr(term, 'feature') and term.feature == 7:
            term_idx_sr = idx_t
            break
    if term_idx_sr is not None:
        XX_7 = gam1.generate_X_grid(term=term_idx_sr)
        pdep_7, confi_7 = gam1.partial_dependence(term=term_idx_sr, X=XX_7, width=0.95)
        axes[4].plot(XX_7[:, 7], pdep_7, color='gold', lw=2)
        axes[4].fill_between(XX_7[:, 7], confi_7[:, 0], confi_7[:, 1], color='gold', alpha=0.3)
        axes[4].set_title('Efecto parcial: Radiación Solar ($SR_{t-1}$)')
        axes[4].set_xlabel('Radiación Solar ($kW/m^2$)')
        axes[4].set_ylabel('Efecto en Target 1h (ppb)')
        axes[4].grid(True, linestyle=':')
        
    # 6. Efecto NOX_lag_1 (term 8)
    term_idx_nox = None
    for idx_t, term in enumerate(gam1.terms):
        if hasattr(term, 'feature') and term.feature == 8:
            term_idx_nox = idx_t
            break
    if term_idx_nox is not None:
        XX_8 = gam1.generate_X_grid(term=term_idx_nox)
        pdep_8, confi_8 = gam1.partial_dependence(term=term_idx_nox, X=XX_8, width=0.95)
        axes[5].plot(XX_8[:, 8], pdep_8, color='tab:brown', lw=2)
        axes[5].fill_between(XX_8[:, 8], confi_8[:, 0], confi_8[:, 1], color='tab:brown', alpha=0.2)
        axes[5].set_title('Efecto parcial: Precursores $NO_x$')
        axes[5].set_xlabel('NOx (ppb)')
        axes[5].set_ylabel('Efecto en Target 1h (ppb)')
        axes[5].grid(True, linestyle=':')
        
    plt.suptitle('Funciones de Efectos Parciales Suaves del Modelo GAM (h=1)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_gam_efectos_parciales.png'), dpi=300)
    plt.close()
    print("Gráfica fig_gam_efectos_parciales.png guardada.")
    
    # Gráfica adicional: Mapa de interacción 2D tensor product te(U, V)
    term_idx_te = None
    for idx_t, term in enumerate(gam1.terms):
        if term.istensor:
            term_idx_te = idx_t
            break
            
    if term_idx_te is not None:
        plt.figure(figsize=(7, 6))
        u_vals = np.linspace(-6, 6, 50)
        v_vals = np.linspace(-6, 6, 50)
        UU, VV = np.meshgrid(u_vals, v_vals)
        grid_2d = np.zeros((len(u_vals)*len(v_vals), len(gam_cols)))
        grid_2d[:, 3] = UU.flatten()
        grid_2d[:, 4] = VV.flatten()
        
        pdep_te = gam1.partial_dependence(term=term_idx_te, X=grid_2d)
        ZZ = pdep_te.reshape(len(v_vals), len(u_vals))
        
        cnt = plt.contourf(UU, VV, ZZ, levels=20, cmap='RdBu_r')
        plt.colorbar(cnt, label='Efecto aditivo en O3 (ppb)')
        plt.xlabel('Viento Zonal U (m/s, >0 hacia el Este)')
        plt.ylabel('Viento Meridional V (m/s, >0 hacia el Norte)')
        plt.title('Superficie de Interacción del Viento te(U, V) en GAM')
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig_gam_viento_tensor.png'), dpi=300)
        plt.close()
        print("Gráfica fig_gam_viento_tensor.png guardada.")

if __name__ == '__main__':
    run_part1()
