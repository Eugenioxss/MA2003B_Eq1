import requests

lat, lon = 25.6758, -100.3383

url = "https://previous-runs-api.open-meteo.com/v1/forecast"
for year in [2021, 2022, 2023, 2024, 2025]:
    p = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-06-01",
        "end_date": f"{year}-06-02",
        "hourly": "cape_previous_day1,direct_radiation_previous_day1,vapour_pressure_deficit_previous_day1,wind_gusts_10m_previous_day1",
        "timezone": "America/Monterrey"
    }
    r = requests.get(url, params=p, timeout=15)
    if r.status_code == 200:
        data = r.json().get('hourly', {})
        c = sum(1 for x in data.get('cape_previous_day1', []) if x is not None)
        dr = sum(1 for x in data.get('direct_radiation_previous_day1', []) if x is not None)
        vpd = sum(1 for x in data.get('vapour_pressure_deficit_previous_day1', []) if x is not None)
        wg = sum(1 for x in data.get('wind_gusts_10m_previous_day1', []) if x is not None)
        print(f"Year {year}: CAPE={c}, DirRad={dr}, VPD={vpd}, WindGusts={wg}")
    else:
        print(f"Year {year}: status {r.status_code}")
