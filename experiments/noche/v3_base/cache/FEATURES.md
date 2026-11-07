# Cache de Características v3_base (`experiments/noche/v3_base/cache/`)

Este directorio contiene las matrices de entrenamiento, validación y prueba completas
para los horizontes $h \in \{1, 4, 8, 12, 24, 48\}$, combinando lo mejor de **Plan A (química + espacial)**
y **Plan C (meteorología externa Open-Meteo)** bajo un estricto **protocolo honesto operativo**.

## Archivos generados
- `features_h1.parquet` (187.0 MB)
- `features_h4.parquet` (187.2 MB)
- `features_h8.parquet` (187.1 MB)
- `features_h12.parquet` (187.1 MB)
- `features_h24.parquet` (187.2 MB)
- `features_h48.parquet` (187.3 MB)

## Estructura de Columnas (Total: 219 columnas)

### 1. Metadatos y Objetivos
- `Date`: Timestamp horario de emisión del pronóstico ($t$).
- `Estacion`: Categórica con las 13 estaciones de monitoreo SIMA (`CE`, `NE`, `NE2`, `NO`, `NO2`, `NTE`, `NTE2`, `SE`, `SE2`, `SE3`, `SO`, `SO2`, `SUR`).
- `split`: Categórica con valores:
  - `'train'`: 2020 a 2023 (entrenamiento).
  - `'val'`: 2024 (validación / early stopping / tuning).
  - `'test'`: 2025 (evaluación final estándar).
- `Target`: Concentración observada de $O_3$ en $t+h$ (equivalente a `Target_{h}h`).
- `Target_{h}h`: Alias del objetivo para consistencia con scripts v2.

### 2. Características Químicas, Espaciales y de Calendario (174 features)
Reutilizadas de `experiments/planA_quickwins/features_planA.py` (variante V4_spatial):
- `Estacion`: Variable categórica de la estación.
- Lags contemporáneos ($t=0$) y pasados ($t-1, t-2, \dots$) de contaminantes: $O_3$, $NO_X$, $NO$, $NO_2$, $O_x = O_3 + NO_2$, $CO$, $SO_2$, $PM_10$, $PM_2.5$.
- Variables meteorológicas SIMA en $t \le 0$: $SR$, $TOUT$, $RH$, $PRS$, $RAINF$, $U\_Wind$, $V\_Wind$, `wind_speed`.
- Estadísticas en ventanas móviles terminando en $t0$: medias 6h, máximos 24h, mínimos 24h, sumas de lluvia/radiación.
- Diffs y aceleraciones: $O_3(t) - O_3(t-1)$, $O_3(t) - O_3(t-24)$, aceleración $O_3$.
- Ratios e interacciones fotoquímicas: $NO_X \times SR$, $TOUT \times SR$, $O_3 / (NO_X + 1)$.
- Variables de calendario: hora, mes, día de semana, fin de semana, rush hour, codificación sen/cos.
- **Red Espacial Regional (15 estaciones)**: medias regionales leave-one-out de $O_3$, $O_x$, $TOUT$; diferencia respecto a la media regional; valores en la estación más cercana y distancia; concentración de $O_3$ contra el viento (upwind advection).

### 3. Meteorología en el Momento de Emisión $t$ (`meteo_t0_*`, 7 features)
Variables de Open-Meteo correspondientes a la emisión $t \le 0$ (idénticas y honestas en todos los splits):
- `meteo_t0_temp`, `meteo_t0_rh`, `meteo_t0_rad`, `meteo_t0_ws`, `meteo_t0_pres`, `meteo_t0_u`, `meteo_t0_v`

### 4. Meteorología Futura Honesta (`honest_meteo_*`, 17 features)
Aplica el **protocolo honesto operativo (PP-train / NWP-test)**:
- En `train` (2020–2023): Se entrena con corrida seamless en $t+h$ (modelo Perfect-Prog / reanálisis).
- En `val` (2024) y `test` (2025): Se sustituyen estrictamente por las salidas de pronósticos numéricos del tiempo (NWP) disponibles antes de $t$:
  - Para $h \le 24$: `*_previous_day1` (lead 24–47h).
  - Para $h = 48$: `*_previous_day2` (lead 48–71h).
- **Importante**: No incluye Boundary Layer Height (BLH) porque no existe pronóstico NWP previo para 2024.
Variables:
- Instantáneas en $t+h$: `honest_meteo_target_temp`, `honest_meteo_target_rh`, `honest_meteo_target_rad`, `honest_meteo_target_cloud`, `honest_meteo_target_ws`, `honest_meteo_target_pres`, `honest_meteo_target_precip`, `honest_meteo_target_dew`, `honest_meteo_target_u`, `honest_meteo_target_v`
- Ventanas móviles 4h terminando en $t+h$: `honest_meteo_target_temp_max_3h`, `honest_meteo_target_rad_sum_3h`, `honest_meteo_target_rh_mean_3h`, `honest_meteo_target_ws_mean_3h`
- Deltas e interacciones: `honest_meteo_delta_temp`, `honest_meteo_delta_rh`, `honest_meteo_target_temp_x_rad`

### 5. Meteorología Futura Perfect-Prog (`pp_meteo_*`, 17 features)
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
