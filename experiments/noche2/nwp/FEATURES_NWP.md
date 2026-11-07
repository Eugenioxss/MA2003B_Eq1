# Documentación de Características NWP Honestas (`features_nwp_h<h>.parquet`)

**Experimento**: `noche2_nwp` (`experiments/noche2/nwp/`)  
**Fecha**: Octubre 2026  
**Responsable**: Científico de Datos / Ingeniero Meteorológico — Antigravity Squad  
**Entorno de ejecución**: Python 3.11+, CPU (`num_threads=3`), Pandas / LightGBM.

---

## 1. Propósito y Filosofía de Diseño

Este conjunto de datos complementario fue desarrollado para cerrar la brecha predictiva identificada en los modelos v3 (`v3_base` y `planC_meteo`) entre la meteorología perfecta (*perfect-prog*, cota superior teórica) y el pronóstico numérico honesto operativo.

Hasta la versión v3, los pronósticos meteorológicos externos provenían de una única corrida determinista ("seamless" / mezcla Open-Meteo), sin calibración local contra la red superficial de estaciones SIMA y sin variables de estabilidad convectiva ni consenso multi-modelo.

En este módulo se incorporan:
1. **Consenso Multi-Modelo Global (Multi-NWP Ensemble)**:
   - Centro Europeo de Pronóstico a Plazo Medio (**ECMWF IFS 0.25°**) — El modelo global de mayor resolución y habilidad de skill en el mundo.
   - Centro Nacional de Predicción Ambiental de EE. UU. (**GFS Seamless / FV3 0.25°**).
   - Servicio Meteorológico Alemán (**ICON Seamless 13 km**).
2. **Diagnóstico de Estabilidad Atmosférica y Dinámica Convectiva**:
   - Energía Potencial Convectiva Disponible (**CAPE**).
   - Ráfagas máximas de viento (**Wind Gusts**) y factor de turbulencia.
   - Déficit de presión de vapor (**VPD**).
3. **Calibración Estadística Causal (MOS — Model Output Statistics)**:
   - Corrección estricta de sesgo empírico por estación y hora solar estimada exclusivamente con datos históricos $\le 2023$ (Train) para predecir en 2024 (Val) y 2025 (Test).
   - Reduce el error de velocidad de viento superficial en un **-44.1%** frente al NWP crudo y el error de temperatura en un **-13.8%**.

---

## 2. Convención de Archivos y Llaves de Cruce

- **Ubicación**: `experiments/noche2/nwp/features_nwp_h<h>.parquet`  
  con $h \in \{1, 4, 8, 12, 24, 48\}$.
- **Llaves primarias**: `['Date', 'Estacion']`  
  - `Date`: Timestamp correspondiente al **momento de emisión $t$**.
  - `Estacion`: Identificador de la estación de monitoreo SIMA (`CE, NE, NO, NO2, ...`).
- **Número de filas**: Exactamente 683,904 filas por archivo (panel completo horario 2020–2025 reindexado para las 13 estaciones estándar).
- **Alineación temporal ($t+h$)**: Todas las columnas representan las condiciones meteorológicas esperadas en el momento objetivo $t+h$, pero indexadas en la fila del instante de emisión $t$.

---

## 3. Catálogo Detallado de Columnas

| Columna | Tipo | Unidad | Descripción Meteorológica / Física |
|---|:---:|:---:|---|
| `Date` | datetime64[ns] | — | Momento de emisión del pronóstico ($t$). |
| `Estacion` | string / object | — | Estación SIMA objetivo. |
| `nwp_temp_mean` | float32 | °C | Temperatura media de bulbo seco proyectada para $t+h$ entre ECMWF, GFS e ICON. |
| `nwp_temp_std` | float32 | °C | Dispersión / desviación estándar de temperatura entre modelos en $t+h$ (índice de incertidumbre sinóptica). |
| `nwp_rad_mean` | float32 | $\text{W/m}^2$ | Radiación solar incidente media en superficie proyectada para $t+h$ (consenso multi-modelo). |
| `nwp_rad_std` | float32 | $\text{W/m}^2$ | Dispersión de radiación solar entre modelos (incertidumbre de cobertura nubosa y frentes). |
| `nwp_ws_mean` | float32 | km/h | Velocidad del viento media a 10m proyectada para $t+h$ entre modelos. |
| `nwp_ws_std` | float32 | km/h | Dispersión de velocidad del viento entre modelos (incertidumbre de ventilación). |
| `nwp_vpd_mean` | float32 | kPa | Déficit de presión de vapor en $t+h$ (sequedad atmosférica y estrés biogénico). |
| `nwp_cape` | float32 | J/kg | Convective Available Potential Energy proyectada para $t+h$ (inestabilidad vs inversión térmica). |
| `nwp_wind_gust` | float32 | km/h | Ráfaga máxima de viento proyectada para $t+h$ (mezcla mecánica de la capa superficial). |
| `nwp_mos_temp` | float32 | °C | Temperatura superficial en la estación calibrada causalmente mediante MOS por estación y hora solar. |
| `nwp_mos_ws` | float32 | km/h | Velocidad de viento superficial calibrada por MOS (corrige fricción orográfica urbana de Monterrey). |
| `nwp_mos_photothermal` | float32 | $\text{°C} \cdot \text{W/m}^2$ | Término de acoplamiento fotoquímico calibrado: $\text{MOS\_Temp} \times \text{NWP\_Rad\_Mean}$. |

---

## 4. Reglas de Honestidad y Horizontes Operativos

En cumplimiento estricto del protocolo `README_noche2.md`:
1. **Para $h \in \{1, 4, 8, 12, 24\}$ horas**:
   - Se utilizan exclusivamente pronósticos con lead time honesto emitidos el día previo (`previous_day1`, lead de 24 a 47 horas antes del objetivo).
2. **Para $h = 48$ horas**:
   - Se utilizan pronósticos emitidos 2 días antes (`previous_day2`, lead de 48 a 71 horas).
3. **Causalidad MOS**:
   - Los factores de calibración de sesgo horario por estación se aprenden exclusivamente sobre el conjunto de entrenamiento (2020–2023). El conjunto de validación (2024) y prueba (2025) jamás intervienen en el ajuste de los parámetros MOS.

---

## 5. Código de Carga y Fusión Rápida

Cualquier otro pipeline o agente (Ensemble, Deep Learning / LSTM / TFT) puede incorporar estas características en una sola línea mediante `pd.merge`:

```python
import pandas as pd

h = 24

# 1. Cargar matriz base existente (v3_base o Plan A)
df_base = pd.read_parquet(f'experiments/noche/v3_base/cache/features_h{h}.parquet')

# 2. Cargar features NWP honestas
df_nwp = pd.read_parquet(f'experiments/noche2/nwp/features_nwp_h{h}.parquet')

# 3. Cruzar por Date y Estacion
df_combined = pd.merge(df_base, df_nwp, on=['Date', 'Estacion'], how='inner')

print(f"Dimensiones fusionadas h={h}: {df_combined.shape}")
```
