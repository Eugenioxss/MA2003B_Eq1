"""
experiments/noche2/nwp/download_nwp_data.py
============================================
Descarga datos multi-modelo y de estabilidad atmosferica de Open-Meteo
Previous Runs API (2024-01-01 a 2025-12-31):
- Modelos: ECMWF IFS 0.25, GFS Seamless, ICON Seamless
- Variables: T2m, Shortwave Rad, Direct Rad, Wind Speed, VPD, Surface Pressure, CAPE, Wind Gusts
  en leads honestos: previous_day1 (h <= 24) y previous_day2 (h = 48).
"""
import os
import sys
import time
import requests
import pandas as pd

sys.path.insert(0, '.')
from experiments.common import STATION_COORDS, MODEL_STATIONS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

START_DATE = '2024-01-01'
END_DATE = '2025-12-31'
TIMEZONE = 'America/Monterrey'

# Agrupaciones principales
# CE: cubre CE, SE, SE2, SE3, SO, SO2, SUR
# NE: cubre NE, NTE2
# NO: NO
# NO2: NO2
# NTE: NTE
# NE2: NE2
PRIMARY_STATIONS = ['CE', 'NE', 'NO', 'NO2', 'NTE', 'NE2', 'SO', 'SE2', 'SUR']
CLUSTER_MAP = {
    'SE': 'CE',
    'SE3': 'SE2',
    'SO2': 'SO',
    'NTE2': 'NE',
}

VARS_TO_FETCH = [
    "temperature_2m_previous_day1", "temperature_2m_previous_day2",
    "shortwave_radiation_previous_day1", "shortwave_radiation_previous_day2",
    "direct_radiation_previous_day1", "direct_radiation_previous_day2",
    "wind_speed_10m_previous_day1", "wind_speed_10m_previous_day2",
    "vapour_pressure_deficit_previous_day1", "vapour_pressure_deficit_previous_day2",
    "surface_pressure_previous_day1", "surface_pressure_previous_day2",
    "cape_previous_day1", "cape_previous_day2",
    "wind_gusts_10m_previous_day1", "wind_gusts_10m_previous_day2"
]


def fetch_with_retry(url, params, max_retries=6, backoff=5):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = backoff * (2 ** attempt)
                print(f"  [429 Rate Limit] Esperando {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  [HTTP {r.status_code}] Intento {attempt+1}/{max_retries}: {r.text[:120]}", flush=True)
                time.sleep(backoff * (attempt + 1))
        except Exception as e:
            print(f"  [Error] {e}. Reintentando...", flush=True)
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Fallo persistente en API Open-Meteo: {url}")


def download_station(station: str, lat: float, lon: float):
    target_path = os.path.join(DATA_DIR, f'nwp_multimodel_{station}.parquet')
    if os.path.exists(target_path):
        print(f"[{station}] Ya existe en cache.", flush=True)
        return

    print(f"[{station}] Descargando 2024-2025 (lat={lat:.4f}, lon={lon:.4f})...", flush=True)
    t0 = time.time()
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "models": "ecmwf_ifs025,gfs_seamless,icon_seamless",
        "hourly": ",".join(VARS_TO_FETCH),
        "timezone": TIMEZONE
    }

    data = fetch_with_retry(url, params)
    df = pd.DataFrame(data['hourly'])
    df['Date'] = pd.to_datetime(df['time'])
    df['Estacion'] = station
    df = df.drop(columns=['time'])

    # Reordenar columnas: Date, Estacion al inicio
    cols = ['Date', 'Estacion'] + [c for c in df.columns if c not in ['Date', 'Estacion']]
    df = df[cols]
    df.to_parquet(target_path, index=False)
    elapsed = time.time() - t0
    print(f"[{station}] Guardado en {elapsed:.1f}s ({len(df)} filas, {len(df.columns)} columnas).", flush=True)


def main():
    print("=" * 65)
    print("  DESCARGA DE PRONOSTICOS HISTORICOS HONESTOS (NWP MULTI-MODELO)")
    print(f"  Periodo: {START_DATE} a {END_DATE}")
    print("=" * 65)

    # 1. Descargar estaciones primarias
    for st in PRIMARY_STATIONS:
        lat, lon, _ = STATION_COORDS[st]
        download_station(st, lat, lon)
        time.sleep(2.0)  # Cortesía con la API

    # 2. Asignar/duplicar para estaciones secundarias del cluster
    for sec_st, rep_st in CLUSTER_MAP.items():
        sec_path = os.path.join(DATA_DIR, f'nwp_multimodel_{sec_st}.parquet')
        rep_path = os.path.join(DATA_DIR, f'nwp_multimodel_{rep_st}.parquet')
        if not os.path.exists(sec_path) and os.path.exists(rep_path):
            print(f"[{sec_st}] Copiando celda desde gemelo espacial {rep_st}...", flush=True)
            df = pd.read_parquet(rep_path)
            df['Estacion'] = sec_st
            df.to_parquet(sec_path, index=False)

    print("\nVerificando cobertura de las 13 estaciones MODEL_STATIONS:")
    all_ok = True
    for st in MODEL_STATIONS:
        p = os.path.join(DATA_DIR, f'nwp_multimodel_{st}.parquet')
        if os.path.exists(p):
            df = pd.read_parquet(p)
            print(f"  - {st:<6}: OK ({len(df)} filas)")
        else:
            print(f"  - {st:<6}: FALTA!")
            all_ok = False

    if all_ok:
        print("\nTodas las estaciones listas y verificadas con exito.")


if __name__ == '__main__':
    main()
