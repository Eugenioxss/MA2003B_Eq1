import requests

lat, lon = 25.6758, -100.3383

url = "https://previous-runs-api.open-meteo.com/v1/forecast"
vars_to_test = [
    "cape_previous_day1", "cape_previous_day2",
    "wind_gusts_10m_previous_day1", "wind_gusts_10m_previous_day2"
]

p = {
    "latitude": lat,
    "longitude": lon,
    "start_date": "2024-06-01",
    "end_date": "2024-06-02",
    "models": "gfs_seamless,icon_seamless",
    "hourly": ",".join(vars_to_test),
    "timezone": "America/Monterrey"
}

r = requests.get(url, params=p, timeout=15)
if r.status_code == 200:
    data = r.json().get('hourly', {})
    for k, v in data.items():
        if k != 'time':
            non_null = sum(1 for x in v if x is not None)
            print(f"  {k}: non-null={non_null}/{len(v)}")
else:
    print("Error:", r.text[:300])
