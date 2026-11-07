import requests

lat, lon = 25.6758, -100.3383

url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    p = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-05-01",
        "end_date": f"{year}-05-02",
        "models": "ecmwf_ifs025,gfs_seamless,icon_seamless",
        "hourly": "temperature_2m,shortwave_radiation,wind_speed_10m",
        "timezone": "America/Monterrey"
    }
    r = requests.get(url, params=p, timeout=15)
    if r.status_code == 200:
        data = r.json().get('hourly', {})
        t_ec = sum(1 for x in data.get('temperature_2m_ecmwf_ifs025', []) if x is not None)
        t_gfs = sum(1 for x in data.get('temperature_2m_gfs_seamless', []) if x is not None)
        t_icon = sum(1 for x in data.get('temperature_2m_icon_seamless', []) if x is not None)
        print(f"Historical Forecast Year {year}: ECMWF(T={t_ec}), GFS(T={t_gfs}), ICON(T={t_icon})")
    else:
        print(f"Historical Forecast Year {year}: status {r.status_code}")
