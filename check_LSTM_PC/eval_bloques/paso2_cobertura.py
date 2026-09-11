#!/usr/bin/env python3
"""
PASO 2: Selección verificada del bloque a ocultar.

Calcula la cobertura SIMULTÁNEA de O3 y PM2.5 por estación y mes,
para el periodo 2022-2024. Identifica la estación/año con cobertura
más cercana al 100% en ambas variables al mismo tiempo.
"""
import pandas as pd
import numpy as np

CSV = "/home/emv/Multivariados/eval_bloques/sima_input.csv"

print("Cargando dataset...")
df = pd.read_csv(CSV, parse_dates=["time"], usecols=["time", "ID", "O3", "PM2.5"])

# Filtrar 2022-2024
df = df[(df["time"].dt.year >= 2022) & (df["time"].dt.year <= 2024)]
print(f"Filas 2022-2024: {len(df):,}")

# Crear columna año-mes
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month

# Para cada estación/mes/año: calcular horas esperadas y cobertura simultánea
print("\n=== COBERTURA SIMULTÁNEA O3 + PM2.5 POR ESTACIÓN/MES ===\n")

results = []
for (station, year, month), grp in df.groupby(["ID", "year", "month"]):
    # Horas esperadas en ese mes
    days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month
    expected_hours = days_in_month * 24

    # Contar filas donde AMBAS variables están presentes
    both_present = grp["O3"].notna() & grp["PM2.5"].notna()
    n_both = both_present.sum()
    coverage = n_both / expected_hours * 100

    results.append({
        "station": station,
        "year": year,
        "month": month,
        "expected_hours": expected_hours,
        "both_present": n_both,
        "coverage_pct": coverage,
    })

results_df = pd.DataFrame(results)

# Cobertura anual promedio por estación (promedio de meses)
print("--- Cobertura ANUAL promedio (media de 12 meses) ---")
annual = results_df.groupby(["station", "year"]).agg(
    mean_coverage=("coverage_pct", "mean"),
    min_coverage=("coverage_pct", "min"),
    months_above_95=("coverage_pct", lambda x: (x >= 95).sum()),
).reset_index()

annual = annual.sort_values("mean_coverage", ascending=False)
print(annual.to_string(index=False))

# El mejor: estación/año con mayor cobertura mínima mensual
print("\n--- TOP 10 estación/año por cobertura MÍNIMA mensual ---")
top = annual.sort_values("min_coverage", ascending=False).head(10)
print(top.to_string(index=False))

# Desglose mensual del top 1
best = top.iloc[0]
best_station = best["station"]
best_year = int(best["year"])
print(f"\n=== DESGLOSE MENSUAL: Estación {best_station}, Año {best_year} ===")
detail = results_df[(results_df["station"] == best_station) & (results_df["year"] == best_year)]
detail = detail.sort_values("month")
print(detail[["month", "expected_hours", "both_present", "coverage_pct"]].to_string(index=False))

# También mostrar CE y SE3 como referencia
for s in ["CE", "SE3"]:
    for y in [2023]:
        print(f"\n=== Referencia: Estación {s}, Año {y} ===")
        d = results_df[(results_df["station"] == s) & (results_df["year"] == y)]
        d = d.sort_values("month")
        print(d[["month", "expected_hours", "both_present", "coverage_pct"]].to_string(index=False))

print(f"\n✅ Recomendación: usar estación '{best_station}' año {best_year}")
print(f"   Cobertura mínima mensual: {best['min_coverage']:.1f}%")
print(f"   Cobertura media anual: {best['mean_coverage']:.1f}%")
