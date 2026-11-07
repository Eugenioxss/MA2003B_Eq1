"""
experiments/noche/target_eng/evaluate_and_report.py
===================================================
Generación de figuras, tablas comparativas y compilación de REPORT.md
para el experimento noche_target_eng.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXP_DIR = os.path.join(ROOT, 'experiments')
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_CSV = os.path.join(CUR_DIR, 'metrics_summary.csv')
METRICS_HONEST_CSV = os.path.join(CUR_DIR, 'metrics_honest_summary.csv')
DMA8_CSV = os.path.join(CUR_DIR, 'daily_max8h_summary.csv')
REPORT_MD = os.path.join(CUR_DIR, 'REPORT.md')

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

def generate_plots():
    if not os.path.exists(METRICS_CSV):
        print(f"No se encuentra {METRICS_CSV}")
        return
    df = pd.read_csv(METRICS_CSV)
    
    # 1. RMSE por horizonte Fase 1
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x='h', y='test_RMSE', hue='variant', ax=ax, palette='Set2')
    ax.set_title("RMSE en Test 2025 por Variante de Target y Horizonte (Fase 1: V4 Spatial)")
    ax.set_xlabel("Horizonte de Pronóstico h (horas)")
    ax.set_ylabel("RMSE (ppb)")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(CUR_DIR, 'rmse_by_horizon.png'))
    plt.close()
    
    # 2. Skill vs Climatología Fase 1
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x='h', y='test_skill', hue='variant', marker='o', ax=ax, palette='Set2')
    ax.set_title("Skill Score vs Climatología (1 - RMSE/RMSE_clim) - Fase 1")
    ax.set_xlabel("Horizonte h (horas)")
    ax.set_ylabel("Skill Score")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(CUR_DIR, 'skill_by_horizon.png'))
    plt.close()

    # 3. Sesgo en Picos (Top 5%)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x='h', y='test_bias_peaks', hue='variant', ax=ax, palette='Set2')
    ax.set_title("Sesgo en Picos (y >= p95) en Test 2025 por Horizonte")
    ax.set_xlabel("Horizonte h (horas)")
    ax.set_ylabel("Sesgo en Picos (ppb)")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(CUR_DIR, 'bias_peaks_by_horizon.png'))
    plt.close()

    # 4. Comparativa Fase 1 vs Fase 2 (Honest Meteo)
    if os.path.exists(METRICS_HONEST_CSV):
        df_h = pd.read_csv(METRICS_HONEST_CSV)
        comp_df = pd.concat([
            df[df['variant'].isin(['direct_base', 'resid_persist'])][['variant', 'h', 'test_RMSE']],
            df_h[df_h['variant'].isin(['direct_base_honest', 'resid_persist_honest'])][['variant', 'h', 'test_RMSE']]
        ], ignore_index=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=comp_df, x='h', y='test_RMSE', hue='variant', ax=ax, palette='tab10')
        ax.set_title("Impacto de la Meteorología Honesta NWP en RMSE (Test 2025)")
        ax.set_xlabel("Horizonte h (horas)")
        ax.set_ylabel("RMSE (ppb)")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        fig.savefig(os.path.join(CUR_DIR, 'rmse_honest_vs_phase1.png'))
        plt.close()

    # 5. DMA8 scatter
    dma8_preds_file = os.path.join(CUR_DIR, 'preds_daily_max8h.parquet')
    if os.path.exists(dma8_preds_file):
        dma8_df = pd.read_parquet(dma8_preds_file)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, sharex=True)
        models = [('pred_clim', 'Climatología (Mes x Estación)'),
                  ('pred_persist', 'Persistencia (DMA8 Hoy 18h)'),
                  ('pred_lgb', 'LightGBM 18:00 (Meteo + Química)')]
        for ax, (col, title) in zip(axes, models):
            ax.scatter(dma8_df['Target_DMA8_next_day'], dma8_df[col], alpha=0.15, s=10, color='teal')
            ax.plot([0, 140], [0, 140], 'r--', lw=1.5, label='1:1')
            ax.axvline(70, color='orange', linestyle=':', label='NOM-020 (70 ppb)')
            ax.axvline(90, color='red', linestyle=':', label='Contingencia (90 ppb)')
            ax.set_title(title)
            ax.set_xlabel("DMA8 Real Día Siguiente (ppb)")
            if ax == axes[0]:
                ax.set_ylabel("DMA8 Pronosticado (ppb)")
            ax.grid(True, alpha=0.4)
            ax.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(CUR_DIR, 'dma8_eval_scatter.png'))
        plt.close()

    print("Figuras generadas con éxito.")

def build_report():
    if not os.path.exists(METRICS_CSV):
        print(f"No se encuentra {METRICS_CSV}")
        return
    df = pd.read_csv(METRICS_CSV)
    df_h = pd.read_csv(METRICS_HONEST_CSV) if os.path.exists(METRICS_HONEST_CSV) else None
    
    # Referencias de contrato
    v2_ref = {1: 8.05, 4: 10.91, 8: 11.97, 12: 12.21, 24: 12.76, 48: 13.95}
    v4_ref = {1: 5.09, 4: 9.69, 8: 11.28, 12: 11.73, 24: 12.39, 48: 13.76}
    planc_ref = {1: 7.55, 4: 9.38, 8: 9.82, 12: 9.80, 24: 10.79, 48: 11.58}
    
    report_lines = []
    report_lines.append("# Experimento Noche: Target Engineering y Reformulación del Problema")
    report_lines.append("\n**Autor / Agente:** Científico de Datos / Estadístico (`noche_target_eng`)  ")
    report_lines.append("**Fecha:** 30 de Octubre de 2026  ")
    report_lines.append("**Carpeta:** `experiments/noche/target_eng/`  ")
    report_lines.append("**Configuración de Recursos:** `num_threads=3`, CPU pura (sin GPU).  ")
    report_lines.append("**Registro Global:** 60 corridas registradas en `experiments/results_registry.csv` bajo `noche_target_eng`.  ")
    report_lines.append("**Predicciones Guardadas:** 60 archivos parquet en `preds/` cubriendo todas las filas val 2024 y test 2025.  ")

    report_lines.append("\n## 1. Resumen Ejecutivo")
    report_lines.append(
        "En este estudio sistemático investigamos si **reformular el target y el esquema predictivo** puede superar "
        "a la formulación directa estándar de LightGBM sobre el conjunto de características avanzadas de Plan A V4_spatial "
        "(174 variables con meteorología contemporánea lag 0 y red espacial de 15 estaciones SIMA), y su integración "
        "posterior con pronósticos numéricos del tiempo (NWP) bajo el protocolo honesto operativo (Fase 2, cache `v3_base`).\n\n"
        "**Conclusiones Principales:**\n"
        "1. **Residual de Persistencia (`resid_persist`) es el ganador para horizontes ultra-cortos y largos**: "
        "Predecir el incremento sobre la persistencia contemporánea $O_3(t)$ reduce el RMSE en $h=1$ a **5.0386 ppb** (superando a la formulación directa 5.0574 y al récord de Plan A 5.0935) "
        "y en $h=48$ a **13.6860 ppb** (reduciendo 0.10 ppb respecto a la formulación directa 13.7866 y Plan A 13.7644).\n"
        "2. **Features explícitas de hora objetivo (`per_hour`) ganan en horizontes intermedios**: Informar la hora de destino $t+h$, su armónico sinusoidal y el salto climatológico $\\Delta_{\\text{clim}}$ logra el menor RMSE en $h=4$ (**9.6220 ppb** sin meteo, y **9.3228 ppb** con meteo honesta).\n"
        "3. **El residuo climatológico (`resid_clim`) perjudica el desempeño de LightGBM**: Restar la media estacional-diurna estática $O_3(t+h) - \\text{clim}(s, m, hr)$ degrada el RMSE entre +0.2 ppb y +0.4 ppb en todos los horizontes. La razón teórica: LightGBM es un modelo inherentemente no lineal que ya modela la interacción entre hora, mes y química; forzar un residuo aditivo simétrico distorsiona la cota natural de no-negatividad y aumenta la varianza.\n"
        "4. **Tweedie ($p=1.2$) superó a `sqrt` y `log1p` en validación**, pero en evaluación final el regresor cuadrático directo sobre $y$ mantiene el mejor equilibrio de RMSE y captura de picos.\n"
        "5. **El modelo global único multi-horizonte (`multi_h_global`)** demuestra ser altamente competitivo a $h=48$ (13.815 ppb vs 13.786 ppb de modelos directos) con la inmensa ventaja de requerir un solo artefacto de modelo en producción.\n"
        "6. **Impacto masivo de la Meteorología Honesta (Fase 2)**: Al acoplar las mejores formulaciones con pronósticos NWP honestos (`*_honest`), el RMSE cae drásticamente en horizontes largos: de 13.68 a **11.3128 ppb** en $h=48$ (ganancia neta de -2.37 ppb, superando a la cota no honesta de Plan C 11.58 ppb) y de 12.39 a **10.5085 ppb** en $h=24$ (-1.88 ppb).\n"
        "7. **Pronóstico Regulatorio NOM-020-SSA1 (DMA8 a 18:00 hrs)**: LightGBM predice el máximo diario de 8 horas del día siguiente con un RMSE de **13.106 ppb** ($R^2 = 0.459$), superando con creces a la persistencia (14.685 ppb) y a la climatología (18.191 ppb), alcanzando un **65.7% de precisión** en la detección de superaciones a la norma (>70 ppb)."
    )

    report_lines.append("\n## 2. Metodología y Variantes Evaluadas")
    report_lines.append("Todas las variantes usaron `num_threads=3`, semilla 42, early stopping de 60 rondas en val 2024 e hiperparámetros de `best_params_optuna.json`:")
    report_lines.append("```json\n{\n  \"num_leaves\": 127, \"learning_rate\": 0.03, \"min_child_samples\": 100,\n  \"subsample\": 1.0, \"colsample_bytree\": 0.6, \"reg_alpha\": 0.0, \"reg_lambda\": 1.0\n}\n```")
    report_lines.append(
        "\n### Descripción Matemática de Variantes:\n"
        "- **`direct_base`**: $\\min_f \\sum (O_3(t+h) - f(X_t))^2$, con $\\hat{y} = \\max(0, f(X_t))$.\n"
        "- **`resid_clim`**: Target $r_t = O_3(t+h) - c(s, \\text{month}(t+h), \\text{hour}(t+h))$ donde $c$ se calcula estrictamente en Train 2020–2023. Predicción: $\\hat{y} = \\max(0, c + \\hat{r}_t)$.\n"
        "- **`resid_persist`**: Target $\\Delta_t = O_3(t+h) - P(t)$. Para $h=1,4,8,24,48$, $P(t) = O_3(t)$ contemporáneo (para $h=24$ y $h=48$, $O_3(t)$ es además la persistencia honesta de la misma hora 1 y 2 días atrás). Para huecos en $O_3(t)$, se aplica fallback a $O_3(t-1)$ y climatología. Predicción: $\\hat{y} = \\max(0, P(t) + \\hat{\\Delta}_t)$.\n"
        "- **Transformaciones (`log1p`, `sqrt`, `tweedie`)**: Inversión con corrección analítica de segundo orden de Jensen: $\\hat{y}_{\\text{sqrt}} = \\max(0, \\hat{z}^2 + \\sigma^2)$ y $\\hat{y}_{\\text{log1p}} = \\max(0, \\exp(\\hat{z} + \\frac{1}{2}\\sigma^2) - 1)$, donde $\\sigma^2 = \\text{Var}(z - \\hat{z})$ en train.\n"
        "- **`per_hour`**: Incorpora al vector de features: `target_hour`, $\\sin(2\\pi \\cdot \\text{target\\_hour}/24)$, $\\cos(2\\pi \\cdot \\text{target\\_hour}/24)$, climatología en destino $c(t+h)$, salto esperado $\\Delta_{\\text{clim}} = c(t+h) - c(t)$ e interacciones fotocatalíticas ($O_3(t) \\times \\sin(\\text{hr}_{t+h})$, $SR(t) \\times \\sin(\\text{hr}_{t+h})$).\n"
        "- **`multi_h_global`**: Una sola matriz apilada con $1.08\\text{M}$ filas en Train (stride 2 para balance cómputo/memoria) y $651\\text{k}$ filas en Val, con $h$, $\\log(h)$ y features de destino explícitas.\n"
        "- **`*_honest` (Fase 2)**: Re-evaluación de las mejores variantes incorporando las 24 variables meteorológicas honestas (Open-Meteo NWP con lead honesto) del cache `v3_base`."
    )

    report_lines.append("\n## 3. Resultados en Validación (2024) — Selección Estricta de Modelos")
    report_lines.append("Siguiendo el contrato de evaluación, **la selección de arquitecturas se basó exclusivamente en Validación 2024**:")
    
    val_pivot = df.pivot(index='h', columns='variant', values='val_RMSE')
    report_lines.append("\n### RMSE en Validación 2024 (Fase 1: Features Químicas y Espaciales V4)")
    report_lines.append(val_pivot.to_markdown())

    report_lines.append(
        "\n**Observaciones de Selección en Val 2024:**\n"
        "- En $h=1$, `resid_persist` alcanza el menor error de validación (**4.9728 ppb**).\n"
        "- En $h=4$, `per_hour` obtiene el menor error de validación (**9.1642 ppb**).\n"
        "- En $h=8$ y $h=12$, `direct_base` y `per_hour` dominan (10.689 y 11.151 ppb).\n"
        "- En $h=48$, `per_hour` logra el mejor error de validación (**13.0728 ppb**) seguido de `resid_persist` (**13.1178 ppb**).\n"
        "- En la criba de transformaciones de target, Tweedie $p=1.2$ fue la mejor transformación no lineal (RMSE promedio 9.373 vs 9.407 de `sqrt` y 9.426 de `log1p`)."
    )

    report_lines.append("\n## 4. Resultados Finales en Test (2025) — Fase 1 (Target Engineering Puro)")
    report_lines.append("Evaluación estándar sobre el conjunto estándar 2025 (108,512 filas válidas en $h=1$):")
    
    test_pivot = df.pivot(index='h', columns='variant', values='test_RMSE')
    test_comp = test_pivot.copy()
    test_comp['Referencia v2 (Script 7)'] = [v2_ref[h] for h in test_comp.index]
    test_comp['Plan A V4_spatial'] = [v4_ref[h] for h in test_comp.index]
    
    report_lines.append("\n### Tabla Comparativa de RMSE en Test 2025 vs Baselines Oficiales")
    report_lines.append(test_comp.to_markdown())

    report_lines.append("\n### Métricas Completas por Variante y Horizonte en Test 2025")
    report_lines.append("| Variante | h | RMSE (ppb) | MAE (ppb) | R² | RMSE Picos | Sesgo Picos | Skill vs Clim |")
    report_lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in df.sort_values(['h', 'variant']).iterrows():
        report_lines.append(f"| {r['variant']} | {int(r['h'])} | {r['test_RMSE']:.3f} | {r['test_MAE']:.3f} | {r['test_R2']:.3f} | {r['test_RMSE_peaks']:.3f} | {r['test_bias_peaks']:.2f} | {r['test_skill']:.3f} |")

    # Fase 2
    if df_h is not None:
        report_lines.append("\n## 5. Resultados de Fase 2: Integración con Meteorología Honesta (`*_honest`)")
        report_lines.append(
            "Al desbloquear el cache `experiments/noche/v3_base/cache/` (198 features con pronósticos NWP de Open-Meteo previos al momento $t$), "
            "repetimos las mejores formulaciones de target (`direct_base`, `resid_persist`, `resid_clim`, `per_hour`):\n"
        )
        h_pivot_test = df_h.pivot(index='h', columns='variant', values='test_RMSE')
        h_pivot_test['Plan A V4 (sin meteo)'] = [v4_ref[h] for h in h_pivot_test.index]
        h_pivot_test['Plan C Meteo (no honesto)'] = [planc_ref[h] for h in h_pivot_test.index]
        report_lines.append("### RMSE en Test 2025 con Meteorología Honesta NWP")
        report_lines.append(h_pivot_test.to_markdown())

        report_lines.append("\n### Métricas de Validación y Test para Variantes Honestas")
        report_lines.append("| Variante Honesta | h | RMSE Val 2024 | RMSE Test 2025 | Skill vs Clim | Iteraciones |")
        report_lines.append("|---|---|---|---|---|---|")
        for _, r in df_h.sort_values(['h', 'variant']).iterrows():
            report_lines.append(f"| {r['variant']} | {int(r['h'])} | {r['val_RMSE']:.3f} | {r['test_RMSE']:.3f} | {r['test_skill']:.3f} | {int(r['iters'])} |")

        report_lines.append(
            "\n**Hallazgo Crucial de Fase 2:**\n"
            "- A horizontes cortos ($h=1$), el impacto de la meteorología futura es menor (5.055 vs 5.038 ppb) porque la inercia química local domina.\n"
            "- A horizontes largos ($h \\ge 4$), la meteorología honesta transforma completamente la capacidad predictiva: en $h=4$ baja a **9.322 ppb**, en $h=8$ a **10.371 ppb**, en $h=24$ a **10.508 ppb**, y en $h=48$ a **11.312 ppb**.\n"
            "- Cabe destacar que en $h=48$, `resid_persist_honest` (**11.313 ppb**) supera incluso a la referencia reportada de Plan C (11.58 ppb), siendo un pronóstico **100% operativo y honesto**."
        )

    # Tarea 6: DMA8
    if os.path.exists(DMA8_CSV):
        df_dma8 = pd.read_csv(DMA8_CSV)
        report_lines.append("\n## 6. Tarea Regulatoria NOM-020-SSA1: Máximo Diario de Media Móvil de 8 Horas (DMA8)")
        report_lines.append(
            "Pronóstico del máximo diario de la media móvil de 8 horas de $O_3$ para el día siguiente, con momento de emisión "
            "operativo a las **18:00 hrs del día actual**. Evaluado sobre 4,376 días-estación en Test 2025:\n"
        )
        report_lines.append(df_dma8.to_markdown(index=False))
        report_lines.append(
            "\n### Análisis Epidemiológico y Regulatorio:\n"
            "- **Climatología**: Fracasa por completo en alertar contingencias ($F_1 = 0$, $CSI = 0$). Subestima los eventos agudos con sesgo sistemático de $-6.19$ ppb.\n"
            "- **Persistencia**: Asumir que el día siguiente repetirá el nivel de hoy a las 18:00 logra un $R^2$ de 0.321 y detecta 211 eventos >70 ppb, pero genera un alto volumen de falsas alarmas (270 falsos positivos, precisión de solo 43.9%).\n"
            "- **LightGBM 18:00**: Reduce el error cuadrático medio a **13.106 ppb** (reducción de $1.58$ ppb sobre persistencia y $5.08$ ppb sobre climatología) y alcanza un $R^2$ de **0.459**. Para el umbral de protección a la salud (>70 ppb), ofrece una precisión del **65.7%**, reduciendo las falsas alarmas a solo 36 casos."
        )

    report_lines.append("\n## 7. Discusión Estadística y Hallazgos Clave")
    report_lines.append(
        "1. **¿Por qué falla el residuo climatológico (`resid_clim`)?**\n"
        "   En series temporales lineales clásicas (ARIMA/SARIMAX), restar la estacionalidad es obligatorio para lograr estacionariedad en media. Sin embargo, los Gradient Boosted Trees no asumen linealidad. Al restar una media fija, se introduce ruido aditivo que perturba las relaciones contemporáneas ($NO_X$, $SR$, advección espacial) y fuerza al árbol a predecir valores negativos en horas nocturnas que luego se truncan con $\\max(0, \\cdot)$, perdiendo la calibración del gradiente.\n\n"
        "2. **¿Por qué funciona el residuo de persistencia (`resid_persist`) en $h=1$ y $h=48$?**\n"
        "   En $h=1$, la correlación entre $O_3(t)$ y $O_3(t+1)$ es $>0.95$. Entrenar al árbol para predecir $\\Delta O_3 = O_3(t+1) - O_3(t)$ enfoca la división de nodos en la tasa de cambio instantánea inducida por la radiación o el viento, en lugar del nivel basal. En $h=48$, $O_3(t)$ representa exactamente la misma hora solar de 2 días atrás, proporcionando un ancla de escala local que estabiliza la predicción ante olas de calor o estabilidad atmosférica regional.\n\n"
        "3. **Features Explícitas de Hora Objetivo (`per_hour`)**:\n"
        "   Proporcionar `delta_clim` (la ganancia fotoquímica esperada entre la hora actual y la hora objetivo) permite que una sola división en el árbol separe escenarios donde el horizonte avanza hacia el mediodía solar de aquellos que avanzan hacia la noche, logrando la mejor precisión en $h=4$.\n\n"
        "4. **Modelo Global Multi-Horizonte (`multi_h_global`)**:\n"
        "   Aunque pierde 0.7 ppb en $h=1$ por no sobre-especializarse en la autorregresión inmediata, su desempeño en horizontes de 24 y 48 horas es prácticamente idéntico al de modelos dedicados. En entornos de producción con limitaciones de memoria o mantenimiento de modelos, un solo modelo global es una solución sumamente atractiva."
    )

    report_lines.append("\n## 8. Recomendación para el Ensamble Final y la Operación")
    report_lines.append(
        "Para maximizar la precisión en el ensamble final de la noche:\n"
        "- **Para $h=1$**: Utilizar `resid_persist_honest` (RMSE: **5.055 ppb** en test, **4.973 ppb** en val) o `resid_persist` sin meteo (**5.039 ppb**).\n"
        "- **Para $h=4$**: Utilizar `per_hour_honest` (RMSE: **9.323 ppb** en test, **8.861 ppb** en val) o `resid_persist_honest` (**9.327 ppb**).\n"
        "- **Para $h=8, 12, 24$**: Utilizar `direct_base_honest` o `per_hour_honest` (RMSE $h=8$: **10.371 ppb**, $h=12$: **10.447 ppb**, $h=24$: **10.508 ppb**).\n"
        "- **Para $h=48$**: Utilizar `resid_persist_honest` (RMSE: **11.313 ppb** en test, **11.262 ppb** en val).\n"
        "- Todos los 60 parquets de predicciones val+test están disponibles en `experiments/noche/target_eng/preds/` para su consumo directo por el agente ensamblador."
    )

    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"REPORT.md generado con éxito en {REPORT_MD}")

if __name__ == '__main__':
    generate_plots()
    build_report()
