import sys
import requests
import json

lat, lon = 25.6758, -100.3383 # CE approx

print("Testing Open-Meteo APIs...")

def test_endpoint(name, url, params):
    print(f"\n--- Testing {name} ---")
    print(f"URL: {url}")
    print(f"Params: {params}")
    try:
        r = requests.get(url, params=params, timeout=15)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if 'hourly' in data:
                print("Hourly keys:", list(data['hourly'].keys()))
                for k in list(data['hourly'].keys())[:5]:
                    vals = data['hourly'][k][:5]
                    print(f"  {k} (first 5): {vals}")
            else:
                print("Keys:", list(data.keys()))
        else:
            print("Response:", r.text[:300])
    except Exception as e:
        print("Error:", e)

# 1. Previous runs API
test_endpoint(
    "Previous Runs API (base)",
    "https://previous-runs-api.open-meteo.com/v1/forecast",
    {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-05-01",
        "end_date": "2024-05-02",
        "hourly": "temperature_2m,temperature_2m_previous_day1,temperature_2m_previous_day2,boundary_layer_height,boundary_layer_height_previous_day1",
        "timezone": "America/Monterrey"
    }
)

# 2. Historical forecast with models
test_endpoint(
    "Historical Forecast API with models",
    "https://historical-forecast-api.open-meteo.com/v1/forecast",
    {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-05-01",
        "end_date": "2024-05-02",
        "models": "ecmwf_ifs025,gfs_seamless,icon_seamless",
        "hourly": "temperature_2m,surface_pressure",
        "timezone": "America/Monterrey"
    }
)

# 3. Check extra variables on historical forecast
test_endpoint(
    "Historical Forecast API extra variables",
    "https://historical-forecast-api.open-meteo.com/v1/forecast",
    {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-05-01",
        "end_date": "2024-05-02",
        "hourly": "boundary_layer_height,cape,uv_index,direct_radiation,cloud_cover_low,temperature_850hPa,boundary_layer_height_previous_day1",
        "timezone": "America/Monterrey"
    }
)

# 4. Check date availability: how far back does historical forecast go?
test_endpoint(
    "Historical Forecast API in 2021",
    "https://historical-forecast-api.open-meteo.com/v1/forecast",
    {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2021-06-01",
        "end_date": "2021-06-02",
        "hourly": "temperature_2m,temperature_2m_previous_day1,cape,uv_index",
        "timezone": "America/Monterrey"
    }
)
