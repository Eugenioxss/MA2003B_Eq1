"""
experiments/planB_picos/05_generate_figures_and_report.py
=========================================================
Genera todas las figuras diagnósticas y gráficos comparativos para el
experimento planB_picos:
 1. 01_tradeoff_losses_bias_vs_rmse.png: Trade-off entre RMSE global y Sesgo en picos
 2. 02_summer_week_prediction_intervals.png: Semana de verano 2025 con bandas de incertidumbre al 80%
 3. 03_exceedance_calibration_and_roc.png: Curvas de confiabilidad y curvas ROC/PR para excedencias (>90 ppb)
 4. 04_csi_and_pod_comparison.png: Comparación de CSI, POD y FAR por horizonte
 5. 05_interval_width_vs_coverage.png: Comparación de ancho vs cobertura en picos de los intervalos
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

EXP_DIR = os.path.join(ROOT, 'experiments', 'planB_picos')
FIG_DIR = os.path.join(EXP_DIR, 'figures')
PRED_DIR = os.path.join(EXP_DIR, 'preds')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

def generate_figure_1_tradeoff():
    """Figura 1: Trade-off RMSE Global vs Sesgo en Picos (top 5%)."""
    csv_path = os.path.join(EXP_DIR, 'metrics_losses.csv')
    if not os.path.exists(csv_path):
        print("metrics_losses.csv no encontrado, saltando Fig 1")
        return
    df = pd.read_csv(csv_path)

    horizons = [1, 8, 24, 48]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    colors = {
        'v2_baseline': '#1f77b4',
        'loss_huber': '#aec7e8',
        'loss_tweedie_p12': '#ff7f0e',
        'loss_tweedie_p15': '#d62728',
        'loss_quantile_p50': '#9467bd',
        'target_sqrt': '#8c564b',
        'target_log1p': '#e377c2',
        'weight_step_p90': '#2ca02c',
        'weight_quad_p90': '#17becf',
        'weight_step_p95': '#bcbd22',
    }

    markers = {
        'v2_baseline': 'X',
        'loss_huber': 'o',
        'loss_tweedie_p12': 's',
        'loss_tweedie_p15': 's',
        'loss_quantile_p50': 'D',
        'target_sqrt': '^',
        'target_log1p': 'v',
        'weight_step_p90': '*',
        'weight_quad_p90': 'p',
        'weight_step_p95': 'h',
    }

    for idx, h in enumerate(horizons):
        ax = axes[idx]
        df_h = df[df['h'] == h]
        if df_h.empty:
            continue

        base_row = df_h[df_h['variant'] == 'v2_baseline']
        base_rmse = base_row['RMSE'].values[0] if not base_row.empty else None
        base_bias = base_row['bias_peaks'].values[0] if not base_row.empty else None

        for _, row in df_h.iterrows():
            v = row['variant']
            c = colors.get(v, '#333333')
            m = markers.get(v, 'o')
            size = 180 if v in ['v2_baseline', 'weight_step_p90', 'weight_quad_p90'] else 100
            ax.scatter(row['RMSE'], row['bias_peaks'], color=c, marker=m, s=size, edgecolors='black', linewidth=0.7, zorder=4, label=v if idx == 0 else "")

            # Etiqueta compacta
            offset_x = 0.05
            offset_y = 0.3
            label_text = v.replace('loss_', '').replace('weight_', 'w_').replace('target_', 't_')
            ax.text(row['RMSE'] + offset_x, row['bias_peaks'] + offset_y, label_text, fontsize=7.5, alpha=0.85)

        if base_bias is not None and base_rmse is not None:
            ax.axhline(0, color='red', linestyle=':', linewidth=1.0, alpha=0.7, label='Sesgo Cero (Ideal)' if idx == 0 else "")
            ax.axvline(base_rmse, color='#1f77b4', linestyle='--', linewidth=0.8, alpha=0.5)

        ax.set_title(f'Horizonte +{h}h (O3 a t+{h})', fontsize=12, fontweight='bold')
        ax.set_xlabel('RMSE Global (ppb, menor es mejor)', fontsize=10)
        ax.set_ylabel('Sesgo en Picos top 5% (ppb, 0 es ideal)', fontsize=10)
        ax.grid(True, alpha=0.6)

    fig.suptitle('Trade-off: RMSE Global vs Sesgo en Picos (Top 5% Test 2025)', fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(FIG_DIR, '01_tradeoff_losses_bias_vs_rmse.png')
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Guardada Fig 1 en {out_file}")

def generate_figure_2_intervals():
    """Figura 2: Semana de Verano 2025 con Bandas de Incertidumbre (80%)."""
    pred_path = os.path.join(PRED_DIR, 'intervals_1h.parquet')
    if not os.path.exists(pred_path):
        print("intervals_1h.parquet no encontrado, saltando Fig 2")
        return
    df = pd.read_parquet(pred_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Filtrar una estación representativa de altos picos (p.ej. SE, SO o CE)
    # y una semana calurosa de verano: 2025-06-20 a 2025-06-27
    station = 'SE'
    df_st = df[df['Estacion'] == station].copy()
    if df_st.empty:
        station = df['Estacion'].iloc[0]
        df_st = df[df['Estacion'] == station].copy()

    mask = (df_st['Date'] >= '2025-06-20') & (df_st['Date'] <= '2025-06-27')
    df_week = df_st[mask].sort_values('Date')

    if df_week.empty or len(df_week) < 24:
        mask = (df_st['Date'] >= '2025-05-10') & (df_st['Date'] <= '2025-05-17')
        df_week = df_st[mask].sort_values('Date')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    dates = df_week['Date']
    y_real = df_week['y_real']
    y_point = df_week['y_point_v2']

    # Subplot 1: Split Conformal Global vs Conformal Condicional
    ax1.plot(dates, y_real, color='black', linewidth=1.8, label='O3 Observado (Real)', zorder=5)
    ax1.plot(dates, y_point, color='#023047', linewidth=1.2, linestyle='--', label='Predicción v2 (Puntual)', zorder=4)

    ax1.fill_between(dates, df_week['lo_conf_global'], df_week['up_conf_global'],
                     color='#8ecae6', alpha=0.45, label='Split Conformal Global (80%)', zorder=2)
    ax1.plot(dates, df_week['up_conf_hour'], color='#fb8500', linewidth=1.3, linestyle=':', label='Límite Sup. Conformal Condicional (Hora)', zorder=3)
    ax1.plot(dates, df_week['lo_conf_hour'], color='#fb8500', linewidth=1.3, linestyle=':', zorder=3)

    ax1.axhline(90, color='red', linestyle='--', linewidth=1.0, alpha=0.8, label='Umbral NOM-020 (90 ppb)')
    ax1.set_title(f'Estación {station} — Comparación Split Conformal Global vs Condicional por Hora (+1h)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Ozono O3 (ppb)', fontsize=11)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.5)

    # Subplot 2: Conformalized Quantile Regression (CQR)
    ax2.plot(dates, y_real, color='black', linewidth=1.8, label='O3 Observado (Real)', zorder=5)
    ax2.plot(dates, y_point, color='#023047', linewidth=1.2, linestyle='--', label='Predicción v2 (Puntual)', zorder=4)

    ax2.fill_between(dates, df_week['lo_cqr'], df_week['up_cqr'],
                     color='#2a9d8f', alpha=0.45, label='CQR - Conformalized Quantile Reg. (80%)', zorder=2)
    ax2.plot(dates, df_week['up_qr'], color='#e76f51', linewidth=1.2, linestyle='-.', label='QR Raw Quantile 90%', zorder=3)
    ax2.plot(dates, df_week['lo_qr'], color='#e76f51', linewidth=1.2, linestyle='-.', label='QR Raw Quantile 10%', zorder=3)

    ax2.axhline(90, color='red', linestyle='--', linewidth=1.0, alpha=0.8, label='Umbral NOM-020 (90 ppb)')
    ax2.set_title(f'Estación {station} — Bandas Adaptativas Asimétricas CQR vs Quantile Regression Raw (+1h)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Fecha y Hora (Verano 2025)', fontsize=11)
    ax2.set_ylabel('Ozono O3 (ppb)', fontsize=11)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.5)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%a %d-%b\n%H:00'))
    fig.suptitle('Intervalos de Predicción al 80% en Episodio de Altos Picos (SIMA Verano 2025)', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_file = os.path.join(FIG_DIR, '02_summer_week_prediction_intervals.png')
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Guardada Fig 2 en {out_file}")

def generate_figure_3_calibration():
    """Figura 3: Curvas de Calibración / Diagrama de Confiabilidad y Curvas ROC/PR para NOM-020."""
    calib_path = os.path.join(EXP_DIR, 'calibration_data.parquet')
    if not os.path.exists(calib_path):
        print("calibration_data.parquet no encontrado, saltando Fig 3")
        return
    df = pd.read_parquet(calib_path)

    horizons = [1, 8, 24]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, h in enumerate(horizons):
        ax = axes[idx]
        df_h = df[df['h'] == h]
        if df_h.empty:
            continue

        yt = df_h['y_true_bin'].values
        p_clf = df_h['prob_clf'].values
        p_v2 = df_h['prob_v2_gauss'].dropna().values
        yt_v2 = df_h.loc[df_h['prob_v2_gauss'].notna(), 'y_true_bin'].values

        # Curva de calibración LGBMClassifier
        prob_true_c, prob_pred_c = calibration_curve(yt, p_clf, n_bins=10, strategy='quantile')
        ax.plot(prob_pred_c, prob_true_c, marker='s', linewidth=2, color='#2ca02c', label='LGBMClassifier (Dedicado)')

        # Curva calibración Gaussiana v2
        if len(yt_v2) > 0:
            prob_true_g, prob_pred_g = calibration_curve(yt_v2, p_v2, n_bins=10, strategy='quantile')
            ax.plot(prob_pred_g, prob_true_g, marker='o', linewidth=1.5, linestyle='--', color='#1f77b4', label='v2 Regresión (Cola Gaussiana)')

        ax.plot([0, 1], [0, 1], linestyle=':', color='gray', label='Perfectamente Calibrado')
        ax.set_title(f'Horizonte +{h}h (Excedencia > 90 ppb)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Probabilidad Pronosticada', fontsize=10)
        ax.set_ylabel('Frecuencia Empírica Observada', fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.5)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Diagramas de Confiabilidad (Curvas de Calibración) — Alerta NOM-020 (O3 > 90 ppb)', fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_file = os.path.join(FIG_DIR, '03_exceedance_calibration_and_roc.png')
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Guardada Fig 3 en {out_file}")

def generate_figure_4_csi_pod():
    """Figura 4: Comparación de Desempeño Atmosférico CSI, POD y FAR por Horizonte."""
    csv_path = os.path.join(EXP_DIR, 'metrics_classification.csv')
    if not os.path.exists(csv_path):
        print("metrics_classification.csv no encontrado, saltando Fig 4")
        return
    df = pd.read_csv(csv_path)

    # Filtrar solo umbral NOM_90ppb
    df_nom = df[df['threshold_mode'] == 'NOM_90ppb'].copy()
    if df_nom.empty:
        return

    models_to_compare = [
        'Naive Regression v2 (y_hat >= thr)',
        'Regression v2 + Gaussian Tail Prob',
        'LGBMClassifier (tuned p*)'
    ]

    labels = ['Regresión v2 Ingenua', 'Regresión v2 Probabilística', 'LGBMClassifier (p* Óptimo)']
    palette = ['#e63946', '#457b9d', '#2a9d8f']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_plot = [('CSI', 'Critical Success Index (CSI / Threat Score)', True),
                    ('POD', 'Probability of Detection (POD / Recall)', True),
                    ('FAR', 'False Alarm Ratio (FAR)', False)]

    horizons = sorted(df_nom['h'].unique())
    x = np.arange(len(horizons))
    width = 0.25

    for ax_idx, (met, title, higher_better) in enumerate(metrics_plot):
        ax = axes[ax_idx]
        for m_idx, (m_name, m_lbl, col) in enumerate(zip(models_to_compare, labels, palette)):
            sub = df_nom[df_nom['model'] == m_name].sort_values('h')
            if not sub.empty:
                vals = sub[met].values
                bars = ax.bar(x + (m_idx - 1) * width, vals, width, label=m_lbl if ax_idx == 0 else "", color=col, alpha=0.9)
                for b in bars:
                    h_val = b.get_height()
                    ax.text(b.get_x() + b.get_width()/2, h_val + 0.01, f'{h_val:.2f}', ha='center', va='bottom', fontsize=7.5)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'+{h}h' for h in horizons])
        ax.set_xlabel('Horizonte de Pronóstico', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.5, axis='y')

    axes[0].legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    fig.suptitle('Desempeño de Detección de Excedencias NOM-020 (> 90 ppb) en Test 2025', fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_file = os.path.join(FIG_DIR, '04_csi_and_pod_comparison.png')
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Guardada Fig 4 en {out_file}")

def generate_figure_5_interval_tradeoff():
    """Figura 5: Comparación de Ancho de Banda vs Cobertura en Picos."""
    csv_path = os.path.join(EXP_DIR, 'metrics_intervals.csv')
    if not os.path.exists(csv_path):
        print("metrics_intervals.csv no encontrado, saltando Fig 5")
        return
    df = pd.read_csv(csv_path)

    methods = df['method'].unique()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {
        'Split Conformal (Global)': '#1f77b4',
        'Conformal Condicional (Hora)': '#ff7f0e',
        'Quantile Regression (LightGBM)': '#d62728',
        'CQR (Conformalized QR)': '#2ca02c',
    }

    horizons = sorted(df['h'].unique())
    for met in methods:
        sub = df[df['method'] == met].sort_values('h')
        c = colors.get(met, '#333333')
        ax1.plot(sub['h'], sub['coverage_peaks'] * 100, marker='o', linewidth=2, label=met, color=c)
        ax2.plot(sub['h'], sub['width_mean'], marker='s', linewidth=2, label=met, color=c)

    ax1.axhline(80, color='gray', linestyle=':', label='Nominal 80%')
    ax1.set_title('Cobertura Específica en Picos (Top 5% Test)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Horizonte (horas)', fontsize=10)
    ax1.set_ylabel('Cobertura Empírica (%)', fontsize=10)
    ax1.set_xticks(horizons)
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc='lower left', fontsize=8.5)

    ax2.set_title('Ancho Medio del Intervalo (ppb)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Horizonte (horas)', fontsize=10)
    ax2.set_ylabel('Ancho Medio (ppb, menor es más nítido)', fontsize=10)
    ax2.set_xticks(horizons)
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc='upper left', fontsize=8.5)

    fig.suptitle('Evaluación de Intervalos de Incertidumbre: Cobertura en Picos vs Nitidez (Test 2025)', fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_file = os.path.join(FIG_DIR, '05_interval_width_vs_coverage.png')
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Guardada Fig 5 en {out_file}")

def main():
    print("Generando figuras diagnósticas...")
    generate_figure_1_tradeoff()
    generate_figure_2_intervals()
    generate_figure_3_calibration()
    generate_figure_4_csi_pod()
    generate_figure_5_interval_tradeoff()
    print("Todas las figuras disponibles han sido generadas.")

if __name__ == '__main__':
    main()
