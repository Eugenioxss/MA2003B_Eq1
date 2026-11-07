import requests

lat, lon = 25.6758, -100.3383

url = "https://previous-runs-api.open-meteo.com/v1/forecast"
vars_to_test = [
    "temperature_2m_previous_day1", "temperature_2m_previous_day2",
    "shortwave_radiation_previous_day1", "shortwave_radiation_previous_day2",
    "direct_radiation_previous_day1", "direct_radiation_previous_day2",
    "wind_speed_10m_previous_day1", "wind_speed_10m_previous_day2",
    "wind_gusts_10m_previous_day1", "wind_gusts_10m_previous_day2",
    "cape_previous_day1", "cape_previous_day2",
    "vapour_pressure_deficit_previous_day1", "vapour_pressure_deficit_previous_day2",
    "surface_pressure_previous_day1", "surface_pressure_previous_day2"
]

p = {
    "latitude": lat,
    "longitude": lon,
    "start_date": "2024-06-01",
    "end_date": "2024-06-02",
    "models": "ecmwf_ifs025,gfs_seamless,icon_seamless",
    "hourly": ",".join(vars_to_test),
    "timezone": "America/Monterrey"
}

r = requests.get(url, params=p, timeout=15)
print("Status:", r.status_code)
if r.status_code == 200:
    data = r.json().get('hourly', {})
    print(f"Total series returned: {len(data) - 1}")
    for k in list(data.keys())[:15]:
        if k != 'time':
            v = data[k]
            non_null = sum(1 for x in v if x is not None)
            print(f"  {k}: non-null={non_null}/{len(v)}")
else:
    print("Error:", r.text[:300])
