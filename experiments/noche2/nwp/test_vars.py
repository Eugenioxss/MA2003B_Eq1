import requests
import pandas as pd

lat, lon = 25.6758, -100.3383

# 1. Test when BLH is available in historical forecast or previous-runs
dates_to_test = [
    "2023-06-01",
    "2024-01-01",
    "2024-06-01",
    "2024-09-01",
    "2024-11-01",
    "2025-01-01",
    "2025-06-01"
]

print("Checking BLH across dates...")
for d in dates_to_test:
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": d,
        "end_date": d,
        "hourly": "temperature_2m,boundary_layer_height,cape,uv_index",
        "timezone": "America/Monterrey"
    }
    r = requests.get(url, params=params, timeout=10)
    if r.status_code == 200:
        data = r.json().get('hourly', {})
        blh = [x for x in data.get('boundary_layer_height', []) if x is not None]
        cape = [x for x in data.get('cape', []) if x is not None]
        uv = [x for x in data.get('uv_index', []) if x is not None]
        print(f"Date {d}: BLH non-null count={len(blh)} (vals={blh[:3] if blh else None}), CAPE count={len(cape)}, UV count={len(uv)}")
    else:
        print(f"Date {d}: status {r.status_code}")

print("\nChecking previous_day variables available...")
test_vars = [
    "cape_previous_day1", "cape_previous_day2",
    "uv_index_previous_day1", "uv_index_previous_day2",
    "direct_radiation_previous_day1", "direct_radiation_previous_day2",
    "cloud_cover_low_previous_day1", "cloud_cover_low_previous_day2",
    "temperature_850hPa_previous_day1", "temperature_850hPa_previous_day2",
    "surface_pressure_previous_day1", "surface_pressure_previous_day2",
    "dew_point_2m_previous_day1", "dew_point_2m_previous_day2"
]
url = "https://previous-runs-api.open-meteo.com/v1/forecast"
params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": "2024-06-01",
    "end_date": "2024-06-02",
    "hourly": ",".join(test_vars),
    "timezone": "America/Monterrey"
}
r = requests.get(url, params=params, timeout=10)
print(f"Test extra previous_day vars: {r.status_code}")
if r.status_code == 200:
    data = r.json().get('hourly', {})
    for v in test_vars:
        vals = [x for x in data.get(v, []) if x is not None]
        print(f"  {v}: non-null count={len(vals)}, sample={vals[:3] if vals else None}")
else:
    print("Error:", r.text[:300])
