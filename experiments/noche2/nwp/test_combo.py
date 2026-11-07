import requests

lat, lon = 25.6758, -100.3383

url = "https://previous-runs-api.open-meteo.com/v1/forecast"
p = {
    "latitude": lat,
    "longitude": lon,
    "start_date": "2024-06-01",
    "end_date": "2024-06-02",
    "models": "ecmwf_ifs025,gfs_seamless,icon_seamless",
    "hourly": "temperature_2m_previous_day1,shortwave_radiation_previous_day1,cape_previous_day1",
    "timezone": "America/Monterrey"
}

r = requests.get(url, params=p, timeout=10)
print("Status:", r.status_code)
if r.status_code == 200:
    data = r.json().get('hourly', {})
    print("Keys returned:", list(data.keys()))
else:
    print("Error:", r.text[:300])
