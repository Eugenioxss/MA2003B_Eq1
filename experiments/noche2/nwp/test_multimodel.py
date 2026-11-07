import requests
import json

lat, lon = 25.6758, -100.3383

# Test models with previous_day1 and previous_day2
models = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]

print("Testing multi-model support on previous-runs-api...")
url = "https://previous-runs-api.open-meteo.com/v1/forecast"
params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": "2024-06-01",
    "end_date": "2024-06-02",
    "models": ",".join(models),
    "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2,shortwave_radiation_previous_day1,shortwave_radiation_previous_day2,wind_speed_10m_previous_day1,wind_speed_10m_previous_day2",
    "timezone": "America/Monterrey"
}

r = requests.get(url, params=params, timeout=15)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json().get('hourly', {})
    for k, v in data.items():
        if k != 'time':
            non_null = sum(1 for x in v if x is not None)
            sample = [round(x, 1) for x in v[:5] if x is not None]
            print(f"  {k}: non-null={non_null}/{len(v)}, sample={sample}")
else:
    print(f"Error: {r.text[:300]}")

print("\nTesting historical availability of multi-model on previous-runs-api...")
for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    p = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-05-01",
        "end_date": f"{year}-05-02",
        "models": "ecmwf_ifs025,gfs_seamless,icon_seamless",
        "hourly": "temperature_2m_previous_day1,shortwave_radiation_previous_day1",
        "timezone": "America/Monterrey"
    }
    r = requests.get(url, params=p, timeout=15)
    if r.status_code == 200:
        data = r.json().get('hourly', {})
        t_ec = sum(1 for x in data.get('temperature_2m_previous_day1_ecmwf_ifs025', []) if x is not None)
        t_gfs = sum(1 for x in data.get('temperature_2m_previous_day1_gfs_seamless', []) if x is not None)
        t_icon = sum(1 for x in data.get('temperature_2m_previous_day1_icon_seamless', []) if x is not None)
        r_ec = sum(1 for x in data.get('shortwave_radiation_previous_day1_ecmwf_ifs025', []) if x is not None)
        r_gfs = sum(1 for x in data.get('shortwave_radiation_previous_day1_gfs_seamless', []) if x is not None)
        r_icon = sum(1 for x in data.get('shortwave_radiation_previous_day1_icon_seamless', []) if x is not None)
        print(f"Year {year}: ECMWF(T={t_ec}, Rad={r_ec}), GFS(T={t_gfs}, Rad={r_gfs}), ICON(T={t_icon}, Rad={r_icon})")
    else:
        print(f"Year {year}: status {r.status_code}")
