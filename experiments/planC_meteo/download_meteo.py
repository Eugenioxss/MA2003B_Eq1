"""
experiments/planC_meteo/download_meteo.py
=========================================
Descarga de datos meteorológicos horarios (2020-01-01 a 2025-12-31) desde Open-Meteo:
1. Reanálisis (ERA5): https://archive-api.open-meteo.com/v1/archive
2. Pronóstico histórico / Previous runs: https://historical-forecast-api.open-meteo.com/v1/forecast

Deduplica celdas de cuadrícula coincidentes:
- SE comparte celda exacta con CE
- NTE2 comparte celda exacta con NE
- SO2 comparte celda exacta con SO

Se guarda en formato Parquet en experiments/planC_meteo/data/ por estación:
- reanalysis_{estacion}.parquet
- forecast_{estacion}.parquet
"""
import os
import sys
import time
import requests
import pandas as pd
import numpy as np

# Raíz del proyecto
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from common import STATION_COORDS, MODEL_STATIONS

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

START_DATE = '2020-01-01'
END_DATE = '2025-12-31'
TIMEZONE = 'America/Monterrey'

# Deduplicación de celdas idénticas en la malla Open-Meteo ERA5 / NWP
GRID_TWINS = {
    'SE': 'CE',
    'NTE2': 'NE',
    'SO2': 'SO'
}

REANALYSIS_VARS = [
    'temperature_2m',
    'relative_humidity_2m',
    'shortwave_radiation',
    'cloud_cover',
    'wind_speed_10m',
    'wind_direction_10m',
    'surface_pressure',
    'precipitation',
    'boundary_layer_height',
    'dew_point_2m'
]

FORECAST_BASE_VARS = [
    'temperature_2m',
    'relative_humidity_2m',
    'shortwave_radiation',
    'cloud_cover',
    'wind_speed_10m',
    'wind_direction_10m',
    'surface_pressure',
    'precipitation',
    'dew_point_2m',
    'boundary_layer_height'
]

FORECAST_PREV_VARS = [
    'temperature_2m_previous_day1', 'temperature_2m_previous_day2',
    'shortwave_radiation_previous_day1', 'shortwave_radiation_previous_day2',
    'relative_humidity_2m_previous_day1', 'relative_humidity_2m_previous_day2',
    'wind_speed_10m_previous_day1', 'wind_speed_10m_previous_day2',
    'wind_direction_10m_previous_day1', 'wind_direction_10m_previous_day2'
]

ALL_FORECAST_VARS = FORECAST_BASE_VARS + FORECAST_PREV_VARS


def fetch_with_retry(url, params, max_retries=5, backoff=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=180)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = backoff * (2 ** attempt)
                print(f"[HTTP 429 Rate Limit] Esperando {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"[HTTP {r.status_code}] Intento {attempt+1}/{max_retries}: {r.text[:150]}", flush=True)
                time.sleep(backoff * (attempt + 1))
        except Exception as e:
            print(f"[Error de conexión] Intento {attempt+1}/{max_retries}: {e}", flush=True)
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Fallo al descargar tras {max_retries} intentos: {url}")


def download_reanalysis(station, lat, lon):
    target_path = os.path.join(DATA_DIR, f'reanalysis_{station}.parquet')
    if os.path.exists(target_path):
        print(f"[Reanálisis] {station} ya existe en caché.", flush=True)
        return pd.read_parquet(target_path)

    # Revisar si se puede deduplicar de gemelo de cuadrícula
    if station in GRID_TWINS:
        twin = GRID_TWINS[station]
        twin_path = os.path.join(DATA_DIR, f'reanalysis_{twin}.parquet')
        if os.path.exists(twin_path):
            print(f"[Reanálisis] {station} comparte celda exacta con {twin}. Deduplicando...", flush=True)
            df = pd.read_parquet(twin_path)
            df['Estacion'] = station
            df.to_parquet(target_path, index=False)
            return df

    print(f"[Reanálisis] Descargando {station} (lat={lat:.4f}, lon={lon:.4f})...", flush=True)
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'hourly': ','.join(REANALYSIS_VARS),
        'timezone': TIMEZONE
    }
    t0 = time.time()
    data = fetch_with_retry('https://archive-api.open-meteo.com/v1/archive', params)
    df = pd.DataFrame(data['hourly'])
    df['Date'] = pd.to_datetime(df['time'])
    df['Estacion'] = station
    df = df.drop(columns=['time'])
    cols = ['Date', 'Estacion'] + [c for c in df.columns if c not in ['Date', 'Estacion']]
    df = df[cols]
    df.to_parquet(target_path, index=False)
    print(f"[Reanálisis] {station} completado en {round(time.time() - t0, 1)}s ({len(df)} filas).", flush=True)
    return df


def download_forecast(station, lat, lon):
    target_path = os.path.join(DATA_DIR, f'forecast_{station}.parquet')
    if os.path.exists(target_path):
        print(f"[Pronóstico] {station} ya existe en caché.", flush=True)
        return pd.read_parquet(target_path)

    # Revisar si se puede deduplicar de gemelo de cuadrícula
    if station in GRID_TWINS:
        twin = GRID_TWINS[station]
        twin_path = os.path.join(DATA_DIR, f'forecast_{twin}.parquet')
        if os.path.exists(twin_path):
            print(f"[Pronóstico] {station} comparte celda exacta con {twin}. Deduplicando...", flush=True)
            df = pd.read_parquet(twin_path)
            df['Estacion'] = station
            df.to_parquet(target_path, index=False)
            return df

    print(f"[Pronóstico] Descargando {station} (lat={lat:.4f}, lon={lon:.4f})...", flush=True)
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'hourly': ','.join(ALL_FORECAST_VARS),
        'timezone': TIMEZONE
    }
    t0 = time.time()
    data = fetch_with_retry('https://historical-forecast-api.open-meteo.com/v1/forecast', params)
    df = pd.DataFrame(data['hourly'])
    df['Date'] = pd.to_datetime(df['time'])
    df['Estacion'] = station
    df = df.drop(columns=['time'])
    cols = ['Date', 'Estacion'] + [c for c in df.columns if c not in ['Date', 'Estacion']]
    df = df[cols]
    df.to_parquet(target_path, index=False)
    print(f"[Pronóstico] {station} completado en {round(time.time() - t0, 1)}s ({len(df)} filas).", flush=True)
    return df


def main():
    print(f"Iniciando descarga/deduplicación para {len(MODEL_STATIONS)} estaciones...", flush=True)
    t_start = time.time()

    for i, station in enumerate(MODEL_STATIONS, 1):
        lat, lon, elev = STATION_COORDS[station]
        print(f"\n--- Estación {i}/{len(MODEL_STATIONS)}: {station} ---", flush=True)

        # 1. Reanálisis
        download_reanalysis(station, lat, lon)
        time.sleep(0.5)

        # 2. Pronóstico histórico
        download_forecast(station, lat, lon)
        time.sleep(1.0)

    print(f"\nProceso finalizado con éxito en {round(time.time() - t_start, 1)} segundos.", flush=True)


if __name__ == '__main__':
    main()
