#!/usr/bin/env python3
"""
PASO 6: Comparación reconstruido vs. real — el punto crítico.

Para cada escenario (A = 1 mes, B = 55 días):
  1. RMSE y R² generales
  2. RMSE sobre el 5% superior de observaciones reales (picos)
  3. Gráfica overlay: reconstruido vs real con picos marcados
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

EVAL_DIR = "/home/emv/Multivariados/eval_bloques"
IMPUTED = os.path.join(EVAL_DIR, "outputs", "sima_imputado.csv")
GT_A = os.path.join(EVAL_DIR, "ground_truth_A.csv")
GT_B = os.path.join(EVAL_DIR, "ground_truth_B.csv")
OUT_DIR = os.path.join(EVAL_DIR, "resultados")
os.makedirs(OUT_DIR, exist_ok=True)

VARS = ["O3", "PM2.5"]

def load_and_align(gt_path, imputed_df):
    """Load ground truth, merge with imputed values."""
    gt = pd.read_csv(gt_path, parse_dates=["time"])
    merged = gt.merge(
        imputed_df[["time", "ID"] + VARS],
        on=["time", "ID"],
        suffixes=("_real", "_imputed"),
        how="left",
    )
    return merged

def compute_metrics(real, imputed, label):
    """Compute RMSE, R², and RMSE on top 5% peaks."""
    # Remove rows where real is NaN (original gaps before our masking)
    valid = np.isfinite(real) & np.isfinite(imputed)
    r = real[valid]
    p = imputed[valid]

    if len(r) == 0:
        return {"label": label, "n": 0, "rmse": np.nan, "r2": np.nan,
                "rmse_peaks": np.nan, "n_peaks": 0}

    rmse = np.sqrt(mean_squared_error(r, p))
    r2 = r2_score(r, p)

    # Top 5% peaks
    threshold = np.percentile(r, 95)
    peak_mask = r >= threshold
    n_peaks = peak_mask.sum()
    if n_peaks > 0:
        rmse_peaks = np.sqrt(mean_squared_error(r[peak_mask], p[peak_mask]))
    else:
        rmse_peaks = np.nan

    return {
        "label": label,
        "n": len(r),
        "rmse": rmse,
        "r2": r2,
        "rmse_peaks": rmse_peaks,
        "n_peaks": n_peaks,
        "threshold_95": threshold,
        "mean_real": r.mean(),
        "mean_imputed": p.mean(),
    }

def plot_overlay(merged, var, scenario_name, out_path):
    """Plot real vs imputed overlay with peaks highlighted."""
    real_col = f"{var}_real"
    imp_col = f"{var}_imputed"

    # Filter valid rows
    valid = merged[real_col].notna() & merged[imp_col].notna()
    d = merged[valid].copy()
    if len(d) == 0:
        print(f"  ⚠️  No valid data for {var} in {scenario_name}")
        return

    real = d[real_col].values
    imputed = d[imp_col].values
    times = d["time"].values

    # Find top 5% peaks
    threshold = np.percentile(real, 95)
    peaks = real >= threshold

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(times, real, label="Real (oculto)", color="black", alpha=0.8, linewidth=0.8)
    ax.plot(times, imputed, label="Reconstruido", color="tab:blue", alpha=0.7, linewidth=0.8)

    # Highlight peaks
    ax.scatter(times[peaks], real[peaks], color="red", s=25, zorder=5,
               label=f"Picos reales (top 5%, ≥{threshold:.1f})", edgecolors="darkred", linewidths=0.5)

    # RMSE and R²
    rmse = np.sqrt(mean_squared_error(real, imputed))
    r2 = r2_score(real, imputed)
    rmse_p = np.sqrt(mean_squared_error(real[peaks], imputed[peaks])) if peaks.any() else np.nan

    ax.set_title(f"{scenario_name} — {var}\nRMSE={rmse:.2f}  R²={r2:.4f}  RMSE_picos={rmse_p:.2f}",
                 fontsize=12)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(var)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Gráfica guardada: {out_path}")

# =============================
# MAIN
# =============================
print("Cargando dataset imputado...")
imp_df = pd.read_csv(IMPUTED, parse_dates=["time"])
print(f"  Shape: {imp_df.shape}")

all_metrics = []

for scenario, gt_path, name in [
    ("A", GT_A, "Escenario A (1 mes)"),
    ("B", GT_B, "Escenario B (55 días / 1333h)"),
]:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print('='*60)

    merged = load_and_align(gt_path, imp_df)
    print(f"  Filas merged: {len(merged)}")

    for var in VARS:
        real = merged[f"{var}_real"].values
        imputed = merged[f"{var}_imputed"].values
        m = compute_metrics(real, imputed, f"{name} — {var}")
        all_metrics.append(m)

        print(f"\n  {var}:")
        print(f"    N={m['n']}, RMSE={m['rmse']:.3f}, R²={m['r2']:.4f}")
        print(f"    RMSE picos (top 5%): {m['rmse_peaks']:.3f} (N={m['n_peaks']}, threshold={m.get('threshold_95', 'N/A')})")
        print(f"    Media real={m['mean_real']:.2f}, Media imputed={m['mean_imputed']:.2f}")

        # Plot
        fig_path = os.path.join(OUT_DIR, f"overlay_{scenario}_{var.replace('.', '')}.png")
        plot_overlay(merged, var, name, fig_path)

# Summary table
print(f"\n{'='*60}")
print("RESUMEN DE MÉTRICAS")
print('='*60)
metrics_df = pd.DataFrame(all_metrics)
print(metrics_df[["label", "n", "rmse", "r2", "rmse_peaks", "n_peaks"]].to_string(index=False))

# Save metrics
metrics_path = os.path.join(OUT_DIR, "metricas_bloques.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\nMétricas guardadas: {metrics_path}")

# === GENERAR REPORTE MARKDOWN ===
report_path = os.path.join(EVAL_DIR, "evaluacion_imputacion_bloques.md")

# Determine recommendation
r2_a_o3 = [m for m in all_metrics if "A" in m["label"] and "O3" in m["label"]][0]["r2"]
r2_b_o3 = [m for m in all_metrics if "B" in m["label"] and "O3" in m["label"]][0]["r2"]
r2_a_pm = [m for m in all_metrics if "A" in m["label"] and "PM2.5" in m["label"]][0]["r2"]
r2_b_pm = [m for m in all_metrics if "B" in m["label"] and "PM2.5" in m["label"]][0]["r2"]
rmse_peaks_b_o3 = [m for m in all_metrics if "B" in m["label"] and "O3" in m["label"]][0]["rmse_peaks"]
rmse_peaks_b_pm = [m for m in all_metrics if "B" in m["label"] and "PM2.5" in m["label"]][0]["rmse_peaks"]

# Decision logic
if r2_b_o3 >= 0.85 and r2_b_pm >= 0.85:
    recommendation = f"La imputación es confiable incluso para apagones de ~55 días (R² escenario B: O3={r2_b_o3:.3f}, PM2.5={r2_b_pm:.3f})."
elif r2_a_o3 >= 0.85 and r2_a_pm >= 0.85 and (r2_b_o3 < 0.85 or r2_b_pm < 0.85):
    recommendation = f"La imputación es confiable solo para huecos cortos (~1 mes), falla en el escenario B de 55 días (R² escenario B: O3={r2_b_o3:.3f}, PM2.5={r2_b_pm:.3f})."
else:
    recommendation = f"La imputación no es confiable ni siquiera para huecos de 1 mes (R² escenario A: O3={r2_a_o3:.3f}, PM2.5={r2_a_pm:.3f})."

with open(report_path, "w") as f:
    f.write("# Evaluación de Imputación por Bloques — SAITS+CSDI (MIMA-v6)\n\n")
    f.write("## Objetivo\n\n")
    f.write("Determinar empíricamente si el modelo híbrido SAITS+CSDI del repo MA2003B-Equipo-6-MIMA-v6 "
            "reconstruye de forma confiable apagones largos de sensores, o solo funciona para huecos "
            "puntuales dispersos (MCAR).\n\n")

    f.write("## Protocolo\n\n")
    f.write("1. Se seleccionó la estación con mayor cobertura simultánea de O3 y PM2.5 (2022-2024).\n")
    f.write("2. Se enmascararon dos bloques como NaN (sin alterar el resto del dataset):\n")
    f.write("   - **Escenario A (moderado):** 1 mes completo (~720h) de O3 y PM2.5.\n")
    f.write("   - **Escenario B (peor caso real):** ~1,333 horas consecutivas (55 días).\n")
    f.write("3. Se entrenó el modelo con los hiperparámetros por defecto (50 épocas, seq_len=168, etc.).\n")
    f.write("4. Se imputó el dataset enmascarado y se extrajeron las reconstrucciones.\n")
    f.write("5. Se compararon las reconstrucciones contra los valores reales guardados (ground truth).\n\n")

    f.write("## Métricas\n\n")
    f.write("| Escenario | Variable | N | RMSE | R² | RMSE picos (top 5%) | N picos |\n")
    f.write("|-----------|----------|---|------|----|--------------------|----------|\n")
    for m in all_metrics:
        f.write(f"| {m['label'].split('—')[0].strip()} | {m['label'].split('—')[1].strip()} "
                f"| {m['n']} | {m['rmse']:.3f} | {m['r2']:.4f} "
                f"| {m['rmse_peaks']:.3f} | {m['n_peaks']} |\n")

    f.write("\n## Gráficas Overlay\n\n")
    f.write("### Escenario A — O3\n")
    f.write(f"![Overlay A O3]({os.path.join(OUT_DIR, 'overlay_A_O3.png')})\n\n")
    f.write("### Escenario A — PM2.5\n")
    f.write(f"![Overlay A PM25]({os.path.join(OUT_DIR, 'overlay_A_PM25.png')})\n\n")
    f.write("### Escenario B — O3\n")
    f.write(f"![Overlay B O3]({os.path.join(OUT_DIR, 'overlay_B_O3.png')})\n\n")
    f.write("### Escenario B — PM2.5\n")
    f.write(f"![Overlay B PM25]({os.path.join(OUT_DIR, 'overlay_B_PM25.png')})\n\n")

    f.write("## Recomendación\n\n")
    f.write(f"**{recommendation}**\n")

print(f"\n✅ Reporte guardado: {report_path}")

# Copy to MA2003B_Eq1/results/
results_dest = "/home/emv/Multivariados/MA2003B_Eq1/results/"
os.makedirs(results_dest, exist_ok=True)
import shutil
shutil.copy2(report_path, os.path.join(results_dest, "evaluacion_imputacion_bloques.md"))
# Also copy figures
for fig in os.listdir(OUT_DIR):
    if fig.endswith(".png"):
        shutil.copy2(os.path.join(OUT_DIR, fig), os.path.join(results_dest, fig))
print(f"✅ Copiado a {results_dest}")
