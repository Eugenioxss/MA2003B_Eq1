"""
experiments/noche/v3_base/prepare_cache.py
=========================================
Construye y guarda en cache las matrices de features completas para cada horizonte
h in [1, 4, 8, 12, 24, 48]:
- Features químicas, espaciales y de calendario (Plan A V4)
- Features meteorológicas contemporáneas t0 (Open-Meteo)
- Features meteorológicas futuras honestas (PP-train / NWP-test previous_day1/2)
- Features meteorológicas futuras PP (seamless en t+h, cota superior)
- Metadata: Date, Estacion, split ('train', 'val', 'test'), Target, Target_{h}h

Guarda en experiments/noche/v3_base/cache/features_h{h}.parquet
y genera cache/FEATURES.md con la documentación técnica.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import pandas as pd

# Directorios
EXP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ROOT = os.path.dirname(EXP_DIR)
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.join(EXP_DIR, 'planC_meteo'))

from common import load_panel, MODEL_STATIONS, HORIZONS
from planA_quickwins.features_planA import build_planA_features
from meteo_features import load_meteo_raw

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def compute_meteo_for_horizon(df_fc: pd.DataFrame, h: int) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Calcula features meteo para el horizonte h:
    - t0: contemporáneas al momento de emisión (seamless t0, válidas en todos los splits)
    - pp_meteo: seamless en t+h (cota superior)
    - honest_meteo: PP-train (seamless en t+h para <=2023), NWP-test (previous_day1 para h<=24, previous_day2 para h=48 en >=2024)
    """
    suf = '_previous_day1' if h <= 24 else '_previous_day2'
    df_sorted = df_fc.sort_values(['Estacion', 'Date']).copy()

    out = []
    for est, g in df_sorted.groupby('Estacion', sort=False):
        g = g.copy()
        is_val_or_test = (g['Date'].dt.year >= 2024).values

        # 1. Features contemporáneas t0
        g['meteo_t0_temp'] = g['temperature_2m'].astype('float32')
        g['meteo_t0_rh'] = g['relative_humidity_2m'].astype('float32')
        g['meteo_t0_rad'] = g['shortwave_radiation'].astype('float32')
        g['meteo_t0_ws'] = g['wind_speed_10m'].astype('float32')
        g['meteo_t0_pres'] = g['surface_pressure'].astype('float32')
        g['meteo_t0_u'] = g['U_Wind_10m'].astype('float32')
        g['meteo_t0_v'] = g['V_Wind_10m'].astype('float32')

        # 2. PP target vars (seamless)
        pp_temp = g['temperature_2m'].astype('float32')
        pp_rh = g['relative_humidity_2m'].astype('float32')
        pp_rad = g['shortwave_radiation'].astype('float32')
        pp_cloud = g['cloud_cover'].astype('float32')
        pp_ws = g['wind_speed_10m'].astype('float32')
        pp_pres = g['surface_pressure'].astype('float32')
        pp_precip = g['precipitation'].astype('float32')
        pp_dew = g['dew_point_2m'].astype('float32')
        pp_u = g['U_Wind_10m'].astype('float32')
        pp_v = g['V_Wind_10m'].astype('float32')

        # 3. Honest target vars (seamless en train <=2023, previous_day en val/test >=2024)
        def make_honest(base_col):
            col_prev = f'{base_col}{suf}'
            if col_prev in g.columns:
                prev_val = g[col_prev].fillna(g[base_col]).astype('float32')
                honest_s = g[base_col].astype('float32').copy()
                honest_s.loc[is_val_or_test] = prev_val.loc[is_val_or_test]
                return honest_s
            return g[base_col].astype('float32')

        h_temp = make_honest('temperature_2m')
        h_rh = make_honest('relative_humidity_2m')
        h_rad = make_honest('shortwave_radiation')
        h_cloud = make_honest('cloud_cover')
        h_ws = make_honest('wind_speed_10m')
        h_pres = make_honest('surface_pressure')
        h_precip = make_honest('precipitation')
        h_dew = make_honest('dew_point_2m')
        h_u = make_honest('U_Wind_10m')
        h_v = make_honest('V_Wind_10m')

        # Derivadas PP en t+h
        g['pp_meteo_target_temp'] = pp_temp.shift(-h)
        g['pp_meteo_target_rh'] = pp_rh.shift(-h)
        g['pp_meteo_target_rad'] = pp_rad.shift(-h)
        g['pp_meteo_target_cloud'] = pp_cloud.shift(-h)
        g['pp_meteo_target_ws'] = pp_ws.shift(-h)
        g['pp_meteo_target_pres'] = pp_pres.shift(-h)
        g['pp_meteo_target_precip'] = pp_precip.shift(-h)
        g['pp_meteo_target_dew'] = pp_dew.shift(-h)
        g['pp_meteo_target_u'] = pp_u.shift(-h)
        g['pp_meteo_target_v'] = pp_v.shift(-h)
        g['pp_meteo_target_temp_max_3h'] = pp_temp.rolling(4, min_periods=2).max().shift(-h)
        g['pp_meteo_target_rad_sum_3h'] = pp_rad.rolling(4, min_periods=2).sum().shift(-h)
        g['pp_meteo_target_rh_mean_3h'] = pp_rh.rolling(4, min_periods=2).mean().shift(-h)
        g['pp_meteo_target_ws_mean_3h'] = pp_ws.rolling(4, min_periods=2).mean().shift(-h)
        g['pp_meteo_delta_temp'] = g['pp_meteo_target_temp'] - g['meteo_t0_temp']
        g['pp_meteo_delta_rh'] = g['pp_meteo_target_rh'] - g['meteo_t0_rh']
        g['pp_meteo_target_temp_x_rad'] = g['pp_meteo_target_temp'] * g['pp_meteo_target_rad']

        # Derivadas Honest en t+h
        g['honest_meteo_target_temp'] = h_temp.shift(-h)
        g['honest_meteo_target_rh'] = h_rh.shift(-h)
        g['honest_meteo_target_rad'] = h_rad.shift(-h)
        g['honest_meteo_target_cloud'] = h_cloud.shift(-h)
        g['honest_meteo_target_ws'] = h_ws.shift(-h)
        g['honest_meteo_target_pres'] = h_pres.shift(-h)
        g['honest_meteo_target_precip'] = h_precip.shift(-h)
        g['honest_meteo_target_dew'] = h_dew.shift(-h)
        g['honest_meteo_target_u'] = h_u.shift(-h)
        g['honest_meteo_target_v'] = h_v.shift(-h)
        g['honest_meteo_target_temp_max_3h'] = h_temp.rolling(4, min_periods=2).max().shift(-h)
        g['honest_meteo_target_rad_sum_3h'] = h_rad.rolling(4, min_periods=2).sum().shift(-h)
        g['honest_meteo_target_rh_mean_3h'] = h_rh.rolling(4, min_periods=2).mean().shift(-h)
        g['honest_meteo_target_ws_mean_3h'] = h_ws.rolling(4, min_periods=2).mean().shift(-h)
        g['honest_meteo_delta_temp'] = g['honest_meteo_target_temp'] - g['meteo_t0_temp']
        g['honest_meteo_delta_rh'] = g['honest_meteo_target_rh'] - g['meteo_t0_rh']
        g['honest_meteo_target_temp_x_rad'] = g['honest_meteo_target_temp'] * g['honest_meteo_target_rad']

        keep_cols = ['Date', 'Estacion'] + [
            c for c in g.columns if c.startswith('meteo_t0_') or c.startswith('pp_meteo_') or c.startswith('honest_meteo_')
        ]
        out.append(g[keep_cols])

    res = pd.concat(out, ignore_index=True)
    t0_cols = [c for c in res.columns if c.startswith('meteo_t0_')]
    honest_cols = [c for c in res.columns if c.startswith('honest_meteo_')]
    pp_cols = [c for c in res.columns if c.startswith('pp_meteo_')]
    return res, t0_cols, honest_cols, pp_cols


def build_and_save_cache():
    print("=" * 70)
    print("  CREACIÓN DE CACHE DE FEATURES v3_base")
    print("=" * 70)
    t_start = time.time()

    # 1. Cargar panel SIMA completo y construir features V4 de Plan A
    print("\n[1/3] Cargando panel horario SIMA y calculando features V4...")
    panel = load_panel(stations=None)  # Necesita todas las 15 estaciones para features espaciales
    df_chem, feat_chem = build_planA_features(panel, variant='V4', stations=MODEL_STATIONS)
    print(f"Features químicos V4 construidos: {len(feat_chem)} variables, shape={df_chem.shape}")

    # Asignar split temporal
    y = df_chem['Date'].dt.year
    split_series = pd.Series('train', index=df_chem.index)
    split_series.loc[y == 2024] = 'val'
    split_series.loc[y == 2025] = 'test'
    df_chem['split'] = split_series.astype('category')

    # 2. Cargar pronósticos meteorológicos de Open-Meteo
    print("\n[2/3] Cargando meteorología raw (Open-Meteo forecast)...")
    df_fc = load_meteo_raw('forecast', stations=MODEL_STATIONS)
    print(f"Meteorología raw cargada: {df_fc.shape}")

    # 3. Iterar por cada horizonte h y guardar cache
    print("\n[3/3] Generando datasets por horizonte h in [1, 4, 8, 12, 24, 48]...")
    features_doc = {}

    for h in HORIZONS:
        t_h = time.time()
        print(f"\n  -> Procesando horizonte h = {h}h...")
        meteo_h, t0_cols, honest_cols, pp_cols = compute_meteo_for_horizon(df_fc, h)

        # Merge de features químicas con meteorológicas
        df_merged = pd.merge(df_chem, meteo_h, on=['Date', 'Estacion'], how='left')

        # Target para este h
        target_col = f'Target_{h}h'
        df_merged['Target'] = df_merged[target_col].astype('float32')

        # Lista de columnas ordenadas:
        # Metadatos: Date, Estacion, split, Target, Target_{h}h
        # Químicas/espaciales: feat_chem (Estacion ya incluida al inicio o como categórica)
        # Meteo t0: t0_cols
        # Meteo honestas: honest_cols
        # Meteo PP: pp_cols
        chem_cols_no_est = [c for c in feat_chem if c != 'Estacion']
        final_cols = ['Date', 'Estacion', 'split', 'Target', target_col] + chem_cols_no_est + t0_cols + honest_cols + pp_cols
        df_out = df_merged[final_cols].copy()

        # Optimizar memoria a float32 para columnas numéricas float64
        for c in df_out.select_dtypes(include=['float64']).columns:
            df_out[c] = df_out[c].astype('float32')

        # Guardar parquet en CACHE_DIR
        out_path = os.path.join(CACHE_DIR, f'features_h{h}.parquet')
        df_out.to_parquet(out_path, index=False, compression='snappy')
        file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"     Guardado: {out_path} ({file_size_mb:.1f} MB, {len(df_out)} filas, {len(df_out.columns)} cols) en {time.time()-t_h:.1f}s")

        features_doc[h] = {
            'total_columns': len(df_out.columns),
            'chem_features': len(chem_cols_no_est) + 1,  # +1 por Estacion
            'meteo_t0': len(t0_cols),
            'honest_meteo': len(honest_cols),
            'pp_meteo': len(pp_cols),
            'size_mb': round(file_size_mb, 1)
        }

    # Escribir FEATURES.md
    write_features_md(feat_chem, t0_cols, honest_cols, pp_cols, features_doc)
    print(f"\nCache completado con éxito en {time.time()-t_start:.1f}s.")


def write_features_md(feat_chem, t0_cols, honest_cols, pp_cols, features_doc):
    doc_path = os.path.join(CACHE_DIR, 'FEATURES.md')
    content = f"""# Cache de Características v3_base (`experiments/noche/v3_base/cache/`)

Este directorio contiene las matrices de entrenamiento, validación y prueba completas
para los horizontes $h \\in \\{{1, 4, 8, 12, 24, 48\\}}$, combinando lo mejor de **Plan A (química + espacial)**
y **Plan C (meteorología externa Open-Meteo)** bajo un estricto **protocolo honesto operativo**.

## Archivos generados
- `features_h1.parquet` ({features_doc[1]['size_mb']} MB)
- `features_h4.parquet` ({features_doc[4]['size_mb']} MB)
- `features_h8.parquet` ({features_doc[8]['size_mb']} MB)
- `features_h12.parquet` ({features_doc[12]['size_mb']} MB)
- `features_h24.parquet` ({features_doc[24]['size_mb']} MB)
- `features_h48.parquet` ({features_doc[48]['size_mb']} MB)

## Estructura de Columnas (Total: {features_doc[1]['total_columns']} columnas)

### 1. Metadatos y Objetivos
- `Date`: Timestamp horario de emisión del pronóstico ($t$).
- `Estacion`: Categórica con las 13 estaciones de monitoreo SIMA (`CE`, `NE`, `NE2`, `NO`, `NO2`, `NTE`, `NTE2`, `SE`, `SE2`, `SE3`, `SO`, `SO2`, `SUR`).
- `split`: Categórica con valores:
  - `'train'`: 2020 a 2023 (entrenamiento).
  - `'val'`: 2024 (validación / early stopping / tuning).
  - `'test'`: 2025 (evaluación final estándar).
- `Target`: Concentración observada de $O_3$ en $t+h$ (equivalente a `Target_{{h}}h`).
- `Target_{{h}}h`: Alias del objetivo para consistencia con scripts v2.

### 2. Características Químicas, Espaciales y de Calendario ({len(feat_chem)} features)
Reutilizadas de `experiments/planA_quickwins/features_planA.py` (variante V4_spatial):
- `Estacion`: Variable categórica de la estación.
- Lags contemporáneos ($t=0$) y pasados ($t-1, t-2, \\dots$) de contaminantes: $O_3$, $NO_X$, $NO$, $NO_2$, $O_x = O_3 + NO_2$, $CO$, $SO_2$, $PM_{10}$, $PM_{2.5}$.
- Variables meteorológicas SIMA en $t \\le 0$: $SR$, $TOUT$, $RH$, $PRS$, $RAINF$, $U\\_Wind$, $V\\_Wind$, `wind_speed`.
- Estadísticas en ventanas móviles terminando en $t0$: medias 6h, máximos 24h, mínimos 24h, sumas de lluvia/radiación.
- Diffs y aceleraciones: $O_3(t) - O_3(t-1)$, $O_3(t) - O_3(t-24)$, aceleración $O_3$.
- Ratios e interacciones fotoquímicas: $NO_X \\times SR$, $TOUT \\times SR$, $O_3 / (NO_X + 1)$.
- Variables de calendario: hora, mes, día de semana, fin de semana, rush hour, codificación sen/cos.
- **Red Espacial Regional (15 estaciones)**: medias regionales leave-one-out de $O_3$, $O_x$, $TOUT$; diferencia respecto a la media regional; valores en la estación más cercana y distancia; concentración de $O_3$ contra el viento (upwind advection).

### 3. Meteorología en el Momento de Emisión $t$ (`meteo_t0_*`, {len(t0_cols)} features)
Variables de Open-Meteo correspondientes a la emisión $t \\le 0$ (idénticas y honestas en todos los splits):
- `meteo_t0_temp`, `meteo_t0_rh`, `meteo_t0_rad`, `meteo_t0_ws`, `meteo_t0_pres`, `meteo_t0_u`, `meteo_t0_v`

### 4. Meteorología Futura Honesta (`honest_meteo_*`, {len(honest_cols)} features)
Aplica el **protocolo honesto operativo (PP-train / NWP-test)**:
- En `train` (2020–2023): Se entrena con corrida seamless en $t+h$ (modelo Perfect-Prog / reanálisis).
- En `val` (2024) y `test` (2025): Se sustituyen estrictamente por las salidas de pronósticos numéricos del tiempo (NWP) disponibles antes de $t$:
  - Para $h \\le 24$: `*_previous_day1` (lead 24–47h).
  - Para $h = 48$: `*_previous_day2` (lead 48–71h).
- **Importante**: No incluye Boundary Layer Height (BLH) porque no existe pronóstico NWP previo para 2024.
Variables:
- Instantáneas en $t+h$: `honest_meteo_target_temp`, `honest_meteo_target_rh`, `honest_meteo_target_rad`, `honest_meteo_target_cloud`, `honest_meteo_target_ws`, `honest_meteo_target_pres`, `honest_meteo_target_precip`, `honest_meteo_target_dew`, `honest_meteo_target_u`, `honest_meteo_target_v`
- Ventanas móviles 4h terminando en $t+h$: `honest_meteo_target_temp_max_3h`, `honest_meteo_target_rad_sum_3h`, `honest_meteo_target_rh_mean_3h`, `honest_meteo_target_ws_mean_3h`
- Deltas e interacciones: `honest_meteo_delta_temp`, `honest_meteo_delta_rh`, `honest_meteo_target_temp_x_rad`

### 5. Meteorología Futura Perfect-Prog (`pp_meteo_*`, {len(pp_cols)} features)
Cota superior teórica (no operativa): utiliza datos seamless en $t+h$ para todos los splits (`train`, `val`, `test`).
Mismas 17 variables que las honestas con prefijo `pp_meteo_`.

---

## Cómo usar el Cache en Python

```python
import pandas as pd

# Cargar horizonte h=1
df = pd.read_parquet('experiments/noche/v3_base/cache/features_h1.parquet')

# Definir conjuntos de variables
chem_features = [c for c in df.columns if not c.startswith(('meteo_t0_', 'honest_meteo_', 'pp_meteo_')) and c not in ['Date', 'split', 'Target', 'Target_1h']]
meteo_t0 = [c for c in df.columns if c.startswith('meteo_t0_')]
honest_meteo = [c for c in df.columns if c.startswith('honest_meteo_')]
pp_meteo = [c for c in df.columns if c.startswith('pp_meteo_')]

# Modelo Honesto Operativo (198 features)
features_honest = ['Estacion'] + chem_features + meteo_t0 + honest_meteo

# Separar por split
train = df[df['split'] == 'train']
val   = df[df['split'] == 'val']
test  = df[df['split'] == 'test']
```
"""
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Documentación escrita en: {doc_path}")


if __name__ == '__main__':
    build_and_save_cache()
