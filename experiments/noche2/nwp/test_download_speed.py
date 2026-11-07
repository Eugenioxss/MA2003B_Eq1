import requests
import time

lat, lon = 25.6758, -100.3383

url = "https://previous-runs-api.open-meteo.com/v1/forecast"
vars_to_fetch = [
    "temperature_2m_previous_day1", "temperature_2m_previous_day2",
    "shortwave_radiation_previous_day1", "shortwave_radiation_previous_day2",
    "direct_radiation_previous_day1", "direct_radiation_previous_day2",
    "wind_speed_10m_previous_day1", "wind_speed_10m_previous_day2",
    "wind_gusts_10m_previous_day1", "wind_gusts_10m_previous_day2",
    "cape_previous_day1", "cape_previous_day2",
    "vapour_pressure_deficit_previous_day1", "vapour_pressure_deficit_previous_day2",
    "surface_pressure_previous_day1", "surface_pressure_previous_day2"
]

params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": "2024-01-01",
    "end_date": "2025-12-31",
    "models": "ecmwf_ifs025,gfs_seamless,icon_seamless",
    "hourly": ",".join(vars_to_fetch),
    "timezone": "America/Monterrey"
}

print("Fetching 2024-2025 CE data...")
t0 = time.time()
r = requests.get(url, params=params, timeout=120)
t_elapsed = time.time() - t0
print(f"Status: {r.status_code}, time: {t_elapsed:.2f}s")
if r.status_code == 200:
    data = r.json().get('hourly', {})
    print(f"Number of series: {len(data)}, length: {len(data['time'])}")
else:
    print("Error:", r.text[:300])
