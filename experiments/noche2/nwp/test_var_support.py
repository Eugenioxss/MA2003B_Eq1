import requests

lat, lon = 25.6758, -100.3383

candidate_vars = [
    # standard vars
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_gusts_10m",
    "soil_temperature_0cm",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "global_tilted_irradiance",
    "terrestrial_radiation",
    "shortwave_radiation_instant",
    "direct_radiation_instant",
    "diffuse_radiation_instant",
    "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant",
    "terrestrial_radiation_instant",
    "uv_index",
    "uv_index_clear_sky",
    "cape",
    "freezing_level_height",
    "sunshine_duration",
    "boundary_layer_height",
    "temperature_850hPa",
    "relative_humidity_850hPa",
    "wind_speed_850hPa",
    "wind_direction_850hPa"
]

print("1. Checking individual variables with _previous_day1 on previous-runs-api:")
supported_previous = []
unsupported_previous = []

for v in candidate_vars:
    var_prev = f"{v}_previous_day1"
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-11-01",
        "end_date": "2024-11-02",
        "hourly": var_prev,
        "timezone": "America/Monterrey"
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json().get('hourly', {})
            vals = data.get(var_prev, [])
            valid_cnt = sum(1 for x in vals if x is not None)
            print(f"  [OK] {var_prev} (valid={valid_cnt}/{len(vals)})")
            supported_previous.append(v)
        else:
            unsupported_previous.append(v)
    except Exception as e:
        print(f"  [ERR] {var_prev}: {e}")

print(f"\nSupported with _previous_day: {supported_previous}")
print(f"Unsupported with _previous_day: {unsupported_previous}")
