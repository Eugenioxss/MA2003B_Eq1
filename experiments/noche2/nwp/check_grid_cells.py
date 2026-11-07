import sys
import requests
sys.path.insert(0, '.')
from experiments.common import STATION_COORDS, MODEL_STATIONS

print("Querying snapped grid coordinates for each station...")
grid_cells = {}
for st in MODEL_STATIONS:
    lat, lon, elev = STATION_COORDS[st]
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    p = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-06-01",
        "end_date": "2024-06-01",
        "hourly": "temperature_2m",
        "timezone": "America/Monterrey"
    }
    r = requests.get(url, params=p, timeout=10)
    if r.status_code == 200:
        data = r.json()
        snap_lat = data.get('latitude')
        snap_lon = data.get('longitude')
        grid_key = (round(snap_lat, 2), round(snap_lon, 2))
        grid_cells.setdefault(grid_key, []).append(st)
        print(f"Station {st:<5} ({lat:.4f}, {lon:.4f}) -> Grid snapped: ({snap_lat:.4f}, {snap_lon:.4f})")
    else:
        print(f"Station {st:<5} -> Error {r.status_code}")

print(f"\nTotal unique grid cells: {len(grid_cells)}")
for grid, st_list in grid_cells.items():
    print(f"  Grid {grid}: {st_list}")
