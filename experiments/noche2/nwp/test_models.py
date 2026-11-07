import requests

lat, lon = 25.6758, -100.3383

# Check models and BLH
models = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless", "meteofrance_seamless"]
print("Testing BLH per model...")
for m in models:
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-08-01",
        "end_date": "2024-08-02",
        "models": m,
        "hourly": "temperature_2m,boundary_layer_height,cape",
        "timezone": "America/Monterrey"
    }
    r = requests.get(url, params=params, timeout=10)
    if r.status_code == 200:
        data = r.json().get('hourly', {})
        blh_key = [k for k in data.keys() if 'boundary_layer_height' in k]
        if blh_key:
            blh_vals = data[blh_key[0]][10:16]
            print(f"Model {m}: key {blh_key[0]}, sample vals: {blh_vals}")
        else:
            print(f"Model {m}: no BLH key in {list(data.keys())}")
    else:
        print(f"Model {m}: failed {r.status_code} - {r.text[:100]}")

# Also test previous_day for models
print("\nTesting previous_day with specific models...")
url = "https://previous-runs-api.open-meteo.com/v1/forecast"
params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": "2024-08-01",
    "end_date": "2024-08-02",
    "models": "ecmwf_ifs025,gfs_seamless,icon_seamless",
    "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
    "timezone": "America/Monterrey"
}
r = requests.get(url, params=params, timeout=10)
print(f"Previous runs with models: {r.status_code}")
if r.status_code == 200:
    data = r.json().get('hourly', {})
    print("Keys:", list(data.keys())[:10])
else:
    print("Error:", r.text[:200])
