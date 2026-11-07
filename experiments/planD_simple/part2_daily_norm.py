"""
experiments/planD_simple/part2_daily_norm.py
===========================================
Parte 2 del experimento Plan D (Estadística Multivariada):
Modelado enfocado en el marco normativo (NOM-020-SSA1-2021).

Objetivos:
1. Target relevante para la salud pública y regulación:
   - O3 MÁXIMO DIARIO por estación
   - Máximo diario del promedio móvil de 8h (O3_8h, min 6 horas válidas; día válido con >=18 h)
2. Predictores disponibles al cierre del día D-1:
   - O3_max, O3_8h_max, TOUT max/min/mean, SR_sum, NOX matutino (6-9h), viento medio, RH medio.
   - Calendario: dayofweek, month, estación.
3. Horizontes:
   - Día D (horizonte ~24h)
   - Día D+1 (horizonte ~48h)
4. Modelos de regresión:
   - Persistencia (máximo de ayer)
   - Climatología diaria (estación x mes en train 2020-2023)
   - Ridge / Lasso
   - GAM (pygam: splines de precursores + temperatura + estación)
   - LightGBM pequeño (n_jobs=2)
5. Modelos de clasificación de excedencias normativas:
   - Excedencia 8h: O3_8h max > 65 ppb (NOM-020-SSA1-2021: 0.065 ppm)
   - Excedencia 1h: O3 max > 90 ppb (NOM-020-SSA1-2021: 0.090 ppm)
   - Clasificadores: Regresión Logística vs LightGBM pequeño
   - Métricas: POD (Recall), FAR, CSI (Threat Score), AUC-ROC
6. Guardar en CSV propios (no en results_registry.csv) y generar figuras.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV, LassoCV, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from pygam import LinearGAM, s, f
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import load_panel, MODEL_STATIONS, SEED

OUT_DIR = os.path.join(ROOT, 'experiments', 'planD_simple')
FIGURES_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Agregación Diaria y Cumplimiento de Criterios de la Norma
# ---------------------------------------------------------------------------
def build_daily_dataset(panel: pd.DataFrame, stations=MODEL_STATIONS) -> pd.DataFrame:
    print("Construyendo panel diario normativo...")
    t0 = time.time()
    df = panel[panel['Estacion'].isin(stations)].copy()
    df = df.sort_values(['Estacion', 'Date']).reset_index(drop=True)
    
    # 1. Calcular promedio móvil de 8h de O3 (min 6 horas válidas)
    df_out = []
    for est, g in df.groupby('Estacion', sort=False):
        g = g.copy()
        g['O3_8h'] = g['O3'].rolling(8, min_periods=6).mean()
        df_out.append(g)
    df = pd.concat(df_out, ignore_index=True)
    
    # 2. Agregar por estación y fecha calendario vectorialmente
    df['date_day'] = pd.to_datetime(df['Date'].dt.date)
    df['hour'] = df['Date'].dt.hour
    df['wind_spd'] = np.sqrt(df['U_Wind']**2 + df['V_Wind']**2)
    
    # Agrupación base
    g = df.groupby(['Estacion', 'date_day'])
    
    agg_main = g.agg(
        n_o3=('O3', 'count'),
        o3_max=('O3', 'max'),
        o3_8h_max=('O3_8h', 'max'),
        n_tout=('TOUT', 'count'),
        tout_max=('TOUT', 'max'),
        tout_min=('TOUT', 'min'),
        tout_mean=('TOUT', 'mean'),
        n_sr=('SR', 'count'),
        sr_sum=('SR', 'sum'),
        n_wind=('wind_spd', 'count'),
        wind_mean=('wind_spd', 'mean'),
        n_rh=('RH', 'count'),
        rh_mean=('RH', 'mean')
    ).reset_index()
    
    # NOX matutino (06:00 a 09:00 horas)
    morn_df = df[df['hour'].isin([6, 7, 8, 9])]
    agg_nox = morn_df.groupby(['Estacion', 'date_day'])['NOX'].agg(
        n_nox_morn='count',
        nox_morn='mean'
    ).reset_index()
    
    daily = pd.merge(agg_main, agg_nox, on=['Estacion', 'date_day'], how='left')
    
    # Aplicar criterios normativos de suficiencia de datos (NOM-020)
    invalid_o3 = daily['n_o3'] < 18
    daily.loc[invalid_o3, 'o3_max'] = np.nan
    daily.loc[invalid_o3, 'o3_8h_max'] = np.nan
    
    invalid_tout = daily['n_tout'] < 18
    daily.loc[invalid_tout, ['tout_max', 'tout_min', 'tout_mean']] = np.nan
    
    daily.loc[daily['n_sr'] < 12, 'sr_sum'] = np.nan
    daily.loc[daily['n_nox_morn'].isna() | (daily['n_nox_morn'] < 2), 'nox_morn'] = np.nan
    daily.loc[daily['n_wind'] < 18, 'wind_mean'] = np.nan
    daily.loc[daily['n_rh'] < 18, 'rh_mean'] = np.nan
    
    daily = daily.rename(columns={
        'o3_max': 'O3_max',
        'o3_8h_max': 'O3_8h_max',
        'tout_max': 'TOUT_max',
        'tout_min': 'TOUT_min',
        'tout_mean': 'TOUT_mean',
        'sr_sum': 'SR_sum',
        'nox_morn': 'NOX_morn',
        'rh_mean': 'RH_mean',
    })
    daily['valid_day'] = ~invalid_o3
    daily = daily.sort_values(['Estacion', 'date_day']).reset_index(drop=True)
    
    # 3. Construir predictores disponibles al día D-1 para pronosticar día D (h24) y D+1 (h48)
    daily_records = []
    for est, g in daily.groupby('Estacion', sort=False):
        g = g.copy()
        s_ = lambda c, k: g[c].shift(k)
        
        # Predictores en D-1
        g['O3_max_D1'] = s_('O3_max', 1)
        g['O3_8h_max_D1'] = s_('O3_8h_max', 1)
        g['TOUT_max_D1'] = s_('TOUT_max', 1)
        g['TOUT_min_D1'] = s_('TOUT_min', 1)
        g['TOUT_mean_D1'] = s_('TOUT_mean', 1)
        g['SR_sum_D1'] = s_('SR_sum', 1)
        g['NOX_morn_D1'] = s_('NOX_morn', 1)
        g['wind_mean_D1'] = s_('wind_mean', 1)
        g['RH_mean_D1'] = s_('RH_mean', 1)
        
        # Targets: Día D (lead 1 día)
        g['target_D_O3_max'] = g['O3_max']
        g['target_D_O3_8h_max'] = g['O3_8h_max']
        
        # Targets: Día D+1 (lead 2 días)
        g['target_Dp1_O3_max'] = g['O3_max'].shift(-1)
        g['target_Dp1_O3_8h_max'] = g['O3_8h_max'].shift(-1)
        
        daily_records.append(g)
        
    res = pd.concat(daily_records, ignore_index=True)
    res['year'] = res['date_day'].dt.year
    res['month'] = res['date_day'].dt.month
    res['dayofweek'] = res['date_day'].dt.dayofweek
    
    print(f"Panel diario construido en {time.time()-t0:.2f}s. Filas totales: {len(res)}")
    return res

# ---------------------------------------------------------------------------
# 2. Funciones de Métricas para Días
# ---------------------------------------------------------------------------
def eval_daily_reg(y_true, y_pred):
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    rmse = float(np.sqrt(np.mean((yt - yp)**2)))
    mae = float(np.mean(np.abs(yt - yp)))
    r2 = float(1 - np.sum((yt - yp)**2) / np.sum((yt - yt.mean())**2))
    
    # Sesgo en días altos (top 10% observados)
    q90 = np.quantile(yt, 0.90)
    high_mask = yt >= q90
    bias_high = float(np.mean(yp[high_mask] - yt[high_mask]))
    rmse_high = float(np.sqrt(np.mean((yt[high_mask] - yp[high_mask])**2)))
    
    return {
        'n': int(ok.sum()), 'RMSE': rmse, 'MAE': mae, 'R2': r2,
        'bias_high': bias_high, 'RMSE_high': rmse_high
    }

def eval_contingency(y_true_binary, prob_pred, threshold=0.5):
    ok = np.isfinite(y_true_binary) & np.isfinite(prob_pred)
    yt = y_true_binary[ok].astype(int)
    yp = (prob_pred[ok] >= threshold).astype(int)
    
    auc = float(roc_auc_score(yt, prob_pred[ok])) if len(np.unique(yt)) > 1 else np.nan
    
    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    
    # POD = Probability of Detection (Recall / Hit Rate) = TP / (TP + FN)
    pod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    # FAR = False Alarm Ratio = FP / (TP + FP)
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    
    # CSI = Critical Success Index (Threat Score) = TP / (TP + FP + FN)
    csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    
    return {
        'n': int(ok.sum()), 'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'POD': pod, 'FAR': far, 'CSI': csi, 'AUC': auc
    }

# ---------------------------------------------------------------------------
# 3. Entrenamiento y Evaluación Diaria
# ---------------------------------------------------------------------------
def run_part2():
    panel = load_panel(MODEL_STATIONS)
    daily = build_daily_dataset(panel, MODEL_STATIONS)
    
    predictors = [
        'O3_max_D1', 'O3_8h_max_D1', 'TOUT_max_D1', 'TOUT_min_D1', 'TOUT_mean_D1',
        'SR_sum_D1', 'NOX_morn_D1', 'wind_mean_D1', 'RH_mean_D1', 'month', 'dayofweek'
    ]
    
    train_mask = daily['year'] <= 2023
    val_mask = daily['year'] == 2024
    test_mask = daily['year'] == 2025
    
    test_df = daily[test_mask].copy().reset_index(drop=True)
    
    # One-hot encoding de estación
    st_dummies = pd.get_dummies(daily['Estacion'], drop_first=True, dtype=float)
    X_all_linear = pd.concat([daily[predictors], st_dummies], axis=1)
    
    # Códigos de estación para GAM y LightGBM
    st_map = {s: i for i, s in enumerate(MODEL_STATIONS)}
    daily['st_idx'] = daily['Estacion'].map(st_map).astype(int)
    gam_features = [
        'O3_max_D1', 'O3_8h_max_D1', 'TOUT_max_D1', 'SR_sum_D1',
        'NOX_morn_D1', 'wind_mean_D1', 'RH_mean_D1', 'month', 'st_idx'
    ]
    
    # LightGBM dataset con categoría de estación
    df_lgb = daily[predictors + ['Estacion']].copy()
    df_lgb['Estacion'] = df_lgb['Estacion'].astype('category')
    
    reg_records = []
    cls_records = []
    roc_data = {}
    
    targets_info = [
        ('target_D_O3_8h_max', 'D (24h)', 'O3_8h_max', 65.0, 'O3_8h_max_D1'),
        ('target_D_O3_max', 'D (24h)', 'O3_max_1h', 90.0, 'O3_max_D1'),
        ('target_Dp1_O3_8h_max', 'D+1 (48h)', 'O3_8h_max', 65.0, 'O3_8h_max_D1'),
        ('target_Dp1_O3_max', 'D+1 (48h)', 'O3_max_1h', 90.0, 'O3_max_D1'),
    ]
    
    for tcol, horizon_label, target_name, exceed_thr, persist_col in targets_info:
        print(f"\n=======================================================")
        print(f"TARGET: {target_name} en Horizonte {horizon_label} (Umbral {exceed_thr} ppb)")
        print(f"=======================================================")
        
        y_test = test_df[tcol].values
        
        # 1. Persistencia (valor de D-1)
        pred_persist = test_df[persist_col].values
        m_persist = eval_daily_reg(y_test, pred_persist)
        reg_records.append({'target': target_name, 'horizon': horizon_label, 'model': 'Persistencia', **m_persist})
        print(f"Persistencia: RMSE={m_persist['RMSE']:.2f}, R2={m_persist['R2']:.3f}, BiasHigh={m_persist['bias_high']:.2f}")
        
        # 2. Climatología diaria (promedio por estación x mes en train 2020-2023)
        clim_map = daily[train_mask].groupby(['Estacion', 'month'])[tcol].mean()
        clim_idx = pd.MultiIndex.from_arrays([test_df['Estacion'], test_df['month']])
        pred_clim = clim_map.reindex(clim_idx).values
        m_clim = eval_daily_reg(y_test, pred_clim)
        reg_records.append({'target': target_name, 'horizon': horizon_label, 'model': 'Climatología', **m_clim})
        print(f"Climatología: RMSE={m_clim['RMSE']:.2f}, R2={m_clim['R2']:.3f}")
        
        # 3. Ridge Regression
        train_valid_lin = train_mask & daily[tcol].notna() & X_all_linear.notna().all(axis=1)
        X_tr_lin = X_all_linear[train_valid_lin].values
        y_tr_lin = daily.loc[train_valid_lin, tcol].values
        
        mean_lin = X_tr_lin.mean(axis=0)
        std_lin = X_tr_lin.std(axis=0)
        std_lin[std_lin == 0] = 1.0
        X_tr_lin_sc = (X_tr_lin - mean_lin) / std_lin
        
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 15)).fit(X_tr_lin_sc, y_tr_lin)
        
        test_valid_lin = X_all_linear[test_mask].notna().all(axis=1)
        pred_ridge = np.full(len(test_df), np.nan)
        X_te_lin = X_all_linear[test_mask][test_valid_lin].values
        pred_ridge[test_valid_lin.values] = ridge.predict((X_te_lin - mean_lin) / std_lin)
        
        m_ridge = eval_daily_reg(y_test, pred_ridge)
        reg_records.append({'target': target_name, 'horizon': horizon_label, 'model': 'Ridge', **m_ridge})
        print(f"Ridge: RMSE={m_ridge['RMSE']:.2f}, R2={m_ridge['R2']:.3f}, BiasHigh={m_ridge['bias_high']:.2f}")
        
        # 4. GAM (pygam)
        train_valid_gam = train_mask & daily[tcol].notna() & daily[gam_features].notna().all(axis=1)
        X_tr_gam = daily.loc[train_valid_gam, gam_features].values
        y_tr_gam = daily.loc[train_valid_gam, tcol].values
        
        # Términos GAM
        gam_terms = (
            s(0, n_splines=10) + # O3_max_D1
            s(1, n_splines=10) + # O3_8h_max_D1
            s(2, n_splines=10) + # TOUT_max_D1
            s(3, n_splines=8) +  # SR_sum_D1
            s(4, n_splines=8) +  # NOX_morn_D1
            s(5, n_splines=8) +  # wind_mean_D1
            s(6, n_splines=8) +  # RH_mean_D1
            s(7, basis='cp', edge_knots=[1, 13], n_splines=6) + # month
            f(8) # estación
        )
        gam = LinearGAM(gam_terms).fit(X_tr_gam, y_tr_gam)
        
        test_valid_gam = daily.loc[test_mask, gam_features].notna().all(axis=1)
        pred_gam = np.full(len(test_df), np.nan)
        pred_gam[test_valid_gam.values] = gam.predict(daily.loc[test_mask][test_valid_gam][gam_features].values)
        m_gam = eval_daily_reg(y_test, pred_gam)
        reg_records.append({'target': target_name, 'horizon': horizon_label, 'model': 'GAM (Splines)', **m_gam})
        print(f"GAM: RMSE={m_gam['RMSE']:.2f}, R2={m_gam['R2']:.3f}, BiasHigh={m_gam['bias_high']:.2f}")
        
        # 5. LightGBM pequeño
        tr_lgb = df_lgb[train_mask]
        y_tr_lgb = daily.loc[train_mask, tcol]
        val_lgb = df_lgb[val_mask]
        y_val_lgb = daily.loc[val_mask, tcol]
        
        ok_tr = y_tr_lgb.notna()
        ok_val = y_val_lgb.notna()
        
        lgb_reg = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=150,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=2,
            random_state=SEED,
            verbose=-1
        )
        lgb_reg.fit(
            tr_lgb[ok_tr], y_tr_lgb[ok_tr],
            eval_set=[(val_lgb[ok_val], y_val_lgb[ok_val])],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        pred_lgb = lgb_reg.predict(df_lgb[test_mask])
        m_lgb = eval_daily_reg(y_test, pred_lgb)
        reg_records.append({'target': target_name, 'horizon': horizon_label, 'model': 'LightGBM Pequeño', **m_lgb})
        print(f"LightGBM: RMSE={m_lgb['RMSE']:.2f}, R2={m_lgb['R2']:.3f}, BiasHigh={m_lgb['bias_high']:.2f}")
        
        # -------------------------------------------------------------------
        # CLASIFICACIÓN DE DÍAS DE EXCEDENCIA
        # -------------------------------------------------------------------
        y_binary_train = (daily.loc[train_mask, tcol] > exceed_thr).astype(int)
        y_binary_val = (daily.loc[val_mask, tcol] > exceed_thr).astype(int)
        y_binary_test = (test_df[tcol] > exceed_thr).astype(int)
        
        # 1. Regresión Logística
        # Imputación simple por mediana de entrenamiento para asegurar predicción completa
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        imputer = SimpleImputer(strategy='median')
        X_tr_imp = imputer.fit_transform(X_all_linear[train_mask])
        X_te_imp = imputer.transform(X_all_linear[test_mask])
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_imp)
        X_te_sc = scaler.transform(X_te_imp)
        
        logit = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)
        logit.fit(X_tr_sc[ok_tr.values], y_binary_train.values[ok_tr.values])
        prob_logit = logit.predict_proba(X_te_sc)[:, 1]
        
        m_logit = eval_contingency(y_binary_test.values, prob_logit, threshold=0.5)
        cls_records.append({
            'target': target_name, 'horizon': horizon_label, 'threshold_ppb': exceed_thr,
            'model': 'Regresión Logística', **m_logit
        })
        print(f"Logit: POD={m_logit['POD']:.3f}, FAR={m_logit['FAR']:.3f}, CSI={m_logit['CSI']:.3f}, AUC={m_logit['AUC']:.3f}")
        
        # 2. LightGBM Classifier
        lgb_clf = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=150,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            is_unbalance=True,
            n_jobs=2,
            random_state=SEED,
            verbose=-1
        )
        lgb_clf.fit(
            tr_lgb[ok_tr], y_binary_train.loc[ok_tr],
            eval_set=[(val_lgb[ok_val], y_binary_val.loc[ok_val])],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        prob_lgb = lgb_clf.predict_proba(df_lgb[test_mask])[:, 1]
        
        m_lgb_clf = eval_contingency(y_binary_test.values, prob_lgb, threshold=0.5)
        cls_records.append({
            'target': target_name, 'horizon': horizon_label, 'threshold_ppb': exceed_thr,
            'model': 'LightGBM Classifier', **m_lgb_clf
        })
        print(f"LGBM Clf: POD={m_lgb_clf['POD']:.3f}, FAR={m_lgb_clf['FAR']:.3f}, CSI={m_lgb_clf['CSI']:.3f}, AUC={m_lgb_clf['AUC']:.3f}")
        
        # Guardar curvas ROC para graficar
        key = f"{tcol}_{horizon_label}"
        roc_data[key] = {
            'y_true': y_binary_test.values,
            'prob_logit': prob_logit,
            'prob_lgb': prob_lgb
        }
        
        # Guardar predicciones de D (24h) O3_8h_max para dispersión
        if target_name == 'O3_8h_max' and horizon_label == 'D (24h)':
            disp_pred_ridge = pred_ridge.copy()
            disp_pred_lgb = pred_lgb.copy()
        
    # Guardar CSVs
    df_reg = pd.DataFrame(reg_records)
    df_cls = pd.DataFrame(cls_records)
    
    df_reg.to_csv(os.path.join(OUT_DIR, 'daily_regression_metrics.csv'), index=False)
    df_cls.to_csv(os.path.join(OUT_DIR, 'daily_classification_metrics.csv'), index=False)
    print("\nResultados diarios de regresión y clasificación guardados en CSV.")
    
    # -----------------------------------------------------------------------
    # 4. Gráficas de Parte 2
    # -----------------------------------------------------------------------
    # Gráfica 1: Curvas ROC de Clasificación de Excedencias
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    titles = [
        ('target_D_O3_8h_max_D (24h)', '$O_{3, 8h}$ Máx Diario > 65 ppb (D, ~24h)'),
        ('target_D_O3_max_D (24h)', '$O_3$ Horario Máx > 90 ppb (D, ~24h)'),
        ('target_Dp1_O3_8h_max_D+1 (48h)', '$O_{3, 8h}$ Máx Diario > 65 ppb (D+1, ~48h)'),
        ('target_Dp1_O3_max_D+1 (48h)', '$O_3$ Horario Máx > 90 ppb (D+1, ~48h)'),
    ]
    
    for i, (k, title) in enumerate(titles):
        ax = axes[i]
        d = roc_data[k]
        yt = d['y_true']
        ok = np.isfinite(yt) & np.isfinite(d['prob_logit']) & np.isfinite(d['prob_lgb'])
        yt = yt[ok]
        
        # Logit ROC
        fpr_log, tpr_log, _ = roc_curve(yt, d['prob_logit'][ok])
        auc_log = roc_auc_score(yt, d['prob_logit'][ok])
        ax.plot(fpr_log, tpr_log, label=f'Logit (AUC = {auc_log:.3f})', color='tab:blue', lw=2)
        
        # LGB ROC
        fpr_lgb, tpr_lgb, _ = roc_curve(yt, d['prob_lgb'][ok])
        auc_lgb = roc_auc_score(yt, d['prob_lgb'][ok])
        ax.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC = {auc_lgb:.3f})', color='tab:red', lw=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Azar (AUC=0.50)')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
        ax.set_ylabel('Tasa de Verdaderos Positivos (POD / Sensibilidad)')
        ax.grid(True, linestyle=':')
        ax.legend(loc='lower right')
        
    plt.suptitle('Curvas ROC para Predicción de Excedencias Normativas (NOM-020-SSA1-2021)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_roc_excedencias_norma.png'), dpi=300)
    plt.close()
    print("Gráfica fig_roc_excedencias_norma.png guardada.")
    
    # Gráfica 2: Dispersión Observado vs Predicho en 2025 (O3_8h max, D ~24h)
    plt.figure(figsize=(8, 7))
    mask_ok = np.isfinite(test_df['target_D_O3_8h_max']) & np.isfinite(disp_pred_ridge) & np.isfinite(disp_pred_lgb)
    
    plt.scatter(test_df.loc[mask_ok, 'target_D_O3_8h_max'], disp_pred_ridge[mask_ok],
                alpha=0.25, color='tab:blue', s=20, label='Ridge Lineal')
    plt.scatter(test_df.loc[mask_ok, 'target_D_O3_8h_max'], disp_pred_lgb[mask_ok],
                alpha=0.25, color='tab:red', s=20, label='LightGBM Pequeño')
    
    lims = [0, 140]
    plt.plot(lims, lims, 'k--', lw=2, label='1:1 Ideal')
    plt.axvline(65, color='orange', linestyle=':', lw=2, label='Umbral NOM-020 (65 ppb)')
    plt.axhline(65, color='orange', linestyle=':', lw=2)
    
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('Observado $O_{3, 8h}$ Máx Diario en 2025 (ppb)')
    plt.ylabel('Predicho $O_{3, 8h}$ Máx Diario (ppb)')
    plt.title('Calibración y Sesgo en Días Críticos de Ozono (D ~24h)')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_dispersion_dias_altos.png'), dpi=300)
    plt.close()
    print("Gráfica fig_dispersion_dias_altos.png guardada.")

if __name__ == '__main__':
    run_part2()
