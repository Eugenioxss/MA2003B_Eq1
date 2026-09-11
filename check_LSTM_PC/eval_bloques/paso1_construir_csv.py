#!/usr/bin/env python3
"""
PASO 1: Construir el CSV en el esquema que MIMA/SIMA espera.

Esquema requerido por sima.py (AirQualityDataProcessor):
  Columnas: ID, time, CO, NO, NO2, NOX, O3, PM10, PM2.5, PRS, RAINF, RH, SO2, SR, TOUT, WSR, WDR

Nuestros parquets (MA2003B_Eq1/data/processed/variables/*_clean.parquet):
  Columnas: Date, Estacion, <Variable>
  → Renombrar: Date→time, Estacion→ID
  → Mergear los 15 archivos en un solo DataFrame pivotado

NOTA: PM2.5_clean.parquet y PM25_clean.parquet son idénticos.
      Usamos PM2.5_clean.parquet (que ya tiene la columna 'PM2.5').
"""
import os
import time
import pandas as pd

BASE = "/home/emv/Multivariados/MA2003B_Eq1/data/processed/variables"
OUT = "/home/emv/Multivariados/eval_bloques/sima_input.csv"

# Las 15 variables que MIMA espera, en el orden de FEATURE_COLUMNS
FEATURES = [
    "CO", "NO", "NO2", "NOX", "O3", "PM10", "PM2.5",
    "PRS", "RAINF", "RH", "SO2", "SR", "TOUT", "WSR", "WDR",
]

# Mapeo archivo → nombre de variable
FILE_MAP = {
    "CO_clean.parquet": "CO",
    "NO_clean.parquet": "NO",
    "NO2_clean.parquet": "NO2",
    "NOX_clean.parquet": "NOX",
    "O3_clean.parquet": "O3",
    "PM10_clean.parquet": "PM10",
    "PM2.5_clean.parquet": "PM2.5",  # Usamos este, no PM25
    "PRS_clean.parquet": "PRS",
    "RAINF_clean.parquet": "RAINF",
    "RH_clean.parquet": "RH",
    "SO2_clean.parquet": "SO2",
    "SR_clean.parquet": "SR",
    "TOUT_clean.parquet": "TOUT",
    "WSR_clean.parquet": "WSR",
    "WDR_clean.parquet": "WDR",
}

t0 = time.time()

# Leer y combinar todos los parquets
print("Leyendo parquets...")
dfs = {}
for filename, var_name in FILE_MAP.items():
    path = os.path.join(BASE, filename)
    df = pd.read_parquet(path)
    # Renombrar columnas: Date→time, Estacion→ID
    df = df.rename(columns={"Date": "time", "Estacion": "ID"})
    # Seleccionar solo las columnas necesarias
    df = df[["time", "ID", var_name]]
    dfs[var_name] = df
    print(f"  {filename}: {len(df):,} filas")

# Merge progresivo: empezamos con la primera variable y hacemos outer join
print("\nMerging 15 variables...")
merged = None
for i, var in enumerate(FEATURES):
    df = dfs[var]
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on=["time", "ID"], how="outer")
    print(f"  [{i+1}/15] Después de merge con {var}: {len(merged):,} filas")

# Asegurar orden de columnas: ID, time, <15 features>
merged = merged[["ID", "time"] + FEATURES]

# Ordenar por estación y fecha
merged = merged.sort_values(["ID", "time"]).reset_index(drop=True)

# Reporte
print(f"\n=== DATASET FINAL ===")
print(f"Shape: {merged.shape}")
print(f"Estaciones: {sorted(merged['ID'].unique())}")
print(f"Rango temporal: {merged['time'].min()} → {merged['time'].max()}")
print(f"\nCobertura por variable (% no-nulos):")
for var in FEATURES:
    pct = merged[var].notna().mean() * 100
    print(f"  {var:>6s}: {pct:.1f}%")

# Guardar
print(f"\nGuardando en {OUT}...")
merged.to_csv(OUT, index=False)
size_mb = os.path.getsize(OUT) / 1e6
elapsed = time.time() - t0
print(f"Listo: {size_mb:.1f} MB, {elapsed:.1f}s")
