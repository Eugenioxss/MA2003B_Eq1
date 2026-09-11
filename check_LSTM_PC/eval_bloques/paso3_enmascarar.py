#!/usr/bin/env python3
"""
PASO 3: Enmascarar DOS escenarios sobre la estación/periodo seleccionado.

IMPORTANTE: Este script lee la salida del Paso 2 para saber qué estación/año usar.
Si el Paso 2 indicó una estación diferente a CE/SE3, actualizar STATION y YEAR abajo.

Escenarios:
  A (moderado): ocultar 1 mes completo (~720h) de O3 y PM2.5
  B (peor caso): ocultar ~1333h consecutivas (55 días) de O3 y PM2.5

Guarda:
  - ground_truth_A.csv y ground_truth_B.csv (valores reales ocultos)
  - sima_input_masked.csv (dataset con los bloques como NaN)
"""
import sys
import pandas as pd
import numpy as np

CSV_IN = "/home/emv/Multivariados/eval_bloques/sima_input.csv"
CSV_OUT = "/home/emv/Multivariados/eval_bloques/sima_input_masked.csv"
GT_A = "/home/emv/Multivariados/eval_bloques/ground_truth_A.csv"
GT_B = "/home/emv/Multivariados/eval_bloques/ground_truth_B.csv"

# ======================================================
# CONFIGURACIÓN — ACTUALIZAR CON RESULTADO DEL PASO 2
# ======================================================
STATION = sys.argv[1] if len(sys.argv) > 1 else "SE3"  # Se puede pasar como argumento
YEAR = int(sys.argv[2]) if len(sys.argv) > 2 else 2023

# Escenario A: mes a ocultar (1-12)
MONTH_A = 6  # Junio — centro del año, buena cobertura típica

# Escenario B: 55 días = 1320 horas (redondeamos a 1333 del prompt)
BLOCK_B_HOURS = 1333
# Empezar desde julio 1 para no solaparse con Escenario A (junio)
BLOCK_B_START_MONTH = 7
BLOCK_B_START_DAY = 1

VARS_TO_MASK = ["O3", "PM2.5"]
# ======================================================

print(f"Configuración:")
print(f"  Estación: {STATION}")
print(f"  Año: {YEAR}")
print(f"  Escenario A: mes {MONTH_A} completo")
print(f"  Escenario B: {BLOCK_B_HOURS}h desde {YEAR}-{BLOCK_B_START_MONTH:02d}-{BLOCK_B_START_DAY:02d}")
print()

# Cargar
print("Cargando dataset...")
df = pd.read_csv(CSV_IN, parse_dates=["time"])
print(f"  Shape original: {df.shape}")

# Filtrar estación para verificar
station_df = df[df["ID"] == STATION].copy()
station_df = station_df.sort_values("time").reset_index(drop=True)
print(f"  Filas estación {STATION}: {len(station_df):,}")

# === ESCENARIO A: mes completo ===
mask_a = (
    (df["ID"] == STATION) &
    (df["time"].dt.year == YEAR) &
    (df["time"].dt.month == MONTH_A)
)
rows_a = df.loc[mask_a].copy()
print(f"\nEscenario A: {mask_a.sum()} filas en {STATION}/{YEAR}/{MONTH_A:02d}")

# Verificar cobertura en el bloque A antes de enmascarar
for var in VARS_TO_MASK:
    n_valid = rows_a[var].notna().sum()
    pct = n_valid / len(rows_a) * 100 if len(rows_a) > 0 else 0
    print(f"  {var} disponible: {n_valid}/{len(rows_a)} ({pct:.1f}%)")

# Guardar ground truth A
gt_a = rows_a[["time", "ID"] + VARS_TO_MASK].copy()
gt_a.to_csv(GT_A, index=False)
print(f"  Ground truth A guardado: {GT_A}")

# === ESCENARIO B: bloque de ~1333 horas ===
start_b = pd.Timestamp(year=YEAR, month=BLOCK_B_START_MONTH, day=BLOCK_B_START_DAY)
end_b = start_b + pd.Timedelta(hours=BLOCK_B_HOURS - 1)
print(f"\nEscenario B: {start_b} → {end_b} ({BLOCK_B_HOURS}h)")

mask_b = (
    (df["ID"] == STATION) &
    (df["time"] >= start_b) &
    (df["time"] <= end_b)
)
rows_b = df.loc[mask_b].copy()
print(f"  {mask_b.sum()} filas en el bloque")

for var in VARS_TO_MASK:
    n_valid = rows_b[var].notna().sum()
    pct = n_valid / len(rows_b) * 100 if len(rows_b) > 0 else 0
    print(f"  {var} disponible: {n_valid}/{len(rows_b)} ({pct:.1f}%)")

# Guardar ground truth B
gt_b = rows_b[["time", "ID"] + VARS_TO_MASK].copy()
gt_b.to_csv(GT_B, index=False)
print(f"  Ground truth B guardado: {GT_B}")

# === ENMASCARAR (poner NaN) ===
print("\nEnmascarando...")
df_masked = df.copy()

# Escenario A
for var in VARS_TO_MASK:
    df_masked.loc[mask_a, var] = np.nan

# Escenario B
for var in VARS_TO_MASK:
    df_masked.loc[mask_b, var] = np.nan

# Verificar que no se solapan (si MONTH_A < BLOCK_B_START_MONTH, OK)
overlap = mask_a & mask_b
if overlap.any():
    print(f"  ⚠️  HAY SOLAPAMIENTO: {overlap.sum()} filas")
else:
    print(f"  ✅ Sin solapamiento entre escenarios A y B")

# Verificar enmascaramiento
total_masked_a = mask_a.sum()
total_masked_b = mask_b.sum()
print(f"  Total filas enmascaradas: A={total_masked_a}, B={total_masked_b}")

# Guardar dataset enmascarado
print(f"\nGuardando dataset enmascarado: {CSV_OUT}")
df_masked.to_csv(CSV_OUT, index=False)

import os
size_mb = os.path.getsize(CSV_OUT) / 1e6
print(f"  Tamaño: {size_mb:.1f} MB")
print("\n✅ Paso 3 completo")
