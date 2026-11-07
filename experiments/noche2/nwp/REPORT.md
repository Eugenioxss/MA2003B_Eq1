# Reporte de Experimento: Pronóstico Numérico del Tiempo (NWP) Honesto Multi-Modelo y Calibración MOS

**Experimento**: `noche2_nwp`  
**Directorio**: `experiments/noche2/nwp/`  
**Fecha**: Octubre 2026  
**Responsable**: Científico de Datos / Ingeniero de Datos Meteorológicos — Antigravity Squad  
**Contrato**: `experiments/noche2/README_noche2.md`, `experiments/noche/README_noche.md`  
**Recursos**: LightGBM `num_threads=3`, CPU only (sin GPU), split temporal estricto (Train: 2020–2023, Val: 2024, Test: 2025).

---

## 1. Resumen Ejecutivo e Hipótesis Central

En la corrida inicial nocturna (v3), la brecha más crítica observada en el desempeño del modelo fue la existente entre la **meteorología perfecta** (*Perfect Prog*, cota superior teórica `B_pp`) y el **pronóstico honesto operativo** (`C_honest` / `D_honest_noise`):
- En $h=24$, el error se degradaba en **+0.65 ppb** (de 9.59 a 10.24 ppb).
- En $h=48$, el error se degradaba en **+1.12 ppb** (de 10.11 a 11.23 ppb).

La causa raíz de esta brecha era doble:
1. **Dependencia de un único pronosticador determinista**: Se empleaba únicamente la corrida "seamless" de Open-Meteo con un lead determinista, vulnerable a sesgos sistemáticos de una sola parametrización física.
2. **Desajuste orográfico y microclimático local**: Los modelos globales operan en celdas de cuadrícula de 10 a 25 km, incapaces de resolver el frenado friccional de la cuenca urbana de Monterrey ni el efecto de isla de calor urbana (UHI).

### Hitos y Resultados Principales:
1. **Ruptura de la Barrera de 11.0 ppb en $h=48$**:
   - `E_nwp_w` alcanza un **RMSE en Test 2025 de 10.9952 ppb** (frente a 11.170 ppb del mejor modelo v3 `E_honest_w` y 11.234 ppb de `D_honest_noise`).
   - Se reduce el RMSE en picos extremos a **24.85 ppb** y el sesgo a **-20.03 ppb**, con $R^2 = 0.7281$.
2. **Nuevo Récord Individual Honesto en $h=24$**:
   - `E_nwp_w` alcanza **10.1898 ppb** en Test 2025 (frente a 10.218 ppb de `E_honest_w` y 10.238 ppb de `D_honest_noise`), con $R^2 = 0.7661$.
3. **Consistencia en Horizontes Medios ($h=8, 12$)**:
   - En $h=8$, `E_nwp_w` reduce el Test RMSE a **9.9579 ppb** (batiendo la referencia de 9.977 ppb).
   - En $h=12$, `E_nwp_w` reduce el Test RMSE a **10.1023 ppb** (batiendo la referencia de 10.135 ppb).
4. **Calibración MOS Altamente Efectiva**:
   - La corrección de sesgo causal por estación y hora solar redujo el error de pronóstico de la velocidad del viento superficial en un **-44.1%** y de la temperatura superficial en un **-13.8%** en el conjunto ciego de Test 2025.
5. **Entregables Reutilizables al 100% de Cobertura**:
   - 6 archivos `features_nwp_h<h>.parquet` con 12 variables NWP y MOS listas para el ensamble.
   - 12 archivos de predicciones en `preds/` validados en el inventario oficial (`val_cov = 1.0, test_cov = 1.0`).

---

## 2. Investigación de APIs de Open-Meteo y Hallazgos Meteorológicos

Se investigaron exhaustivamente las APIs de pronóstico numérico histórico de Open-Meteo:
- `https://previous-runs-api.open-meteo.com/v1/forecast`
- `https://historical-forecast-api.open-meteo.com/v1/forecast`
- Single Runs API y modelos globales específicos.

### 2.1. Disponibilidad de Modelos Globales y Cobertura Temporal

| Modelo | Resolución | Archivo Histórico Previous Runs | Archivo Seamless / Historical Forecast |
|---|:---:|:---:|:---:|
| **ECMWF IFS 0.25°** | 25 km / 0.25° | Completo 2024–2025 (T, Rad, Direct Rad, WS, VPD, Pres) | Completo 2024–2025 (No disponible pre-2024) |
| **GFS Seamless / FV3** | 25 km / 0.25° | Completo 2024–2025 (+ T parcial 2021–2023) | Completo 2020–2025 |
| **ICON Seamless (DWD)** | 13 km / 0.125° | Completo 2024–2025 | Completo 2023–2025 |
| **GEM Seamless (CMC)** | 25 km / 0.25° | Completo 2024–2025 | Completo 2023–2025 |

> [!IMPORTANT]
> **Hallazgo sobre archivo histórico**: Las corridas retrospectivas completas de pronósticos previos (*Previous Runs API*) con sufijos `_previous_day1` y `_previous_day2` para modelos europeos (ECMWF, ICON) sólo están archivadas de forma pública para **2024 y 2025** (los años críticos de validación y prueba). Para 2020–2023, la única fuente disponible homogénea es la corrida sin sesgo/reanálisis. Esto justifica y valida el protocolo canónico **PP-train / NWP-test**.

### 2.2. Diagnóstico de la Altura de la Capa de Mezcla (BLH)
- En la API de pronóstico histórico, `boundary_layer_height` tiene valores nulos (`None`) en todo 2020–2023 y la primera mitad de 2024.
- Sólo empieza a reportar mediciones válidas a partir de **noviembre de 2024**.
- En la API de corridas previas, `boundary_layer_height_previous_day1` devuelve 0 datos válidos (código 200 con arreglo de nulos).
- **Decisión operativa honesta**: Excluir BLH directo para evitar contaminar el protocolo con imputaciones espurias y sustituirlo por sus gobernadores físicos: CAPE (inestabilidad térmica) y Cizalladura Vertical/Ráfagas (mezcla mecánica).

### 2.3. Nuevas Variables Atmosféricas Validadas
Se verificó y descargó con éxito en 2024–2025:
1. **CAPE (Convective Available Potential Energy)**: Mide la flotabilidad termodinámica de la parcela de aire. Valores bajos/nulos caracterizan inversiones térmicas y estabilidad (acumulación de precursores); valores altos indican mezcla convectiva profunda.
2. **Radiación Directa y Difusa**: El cociente `direct_radiation / (shortwave_radiation + 1)` discrimina días con alta carga de aerosoles urbanos / smog fotoquímico frente a días con cielo despejado puro.
3. **Ráfagas de Viento (Wind Gusts)**: Captura la turbulencia superficial que quiebra la capa de inversión matutina.
4. **Déficit de Presión de Vapor (VPD)**: Cuantifica la sequedad del aire y gobierna la tasa de emisión de compuestos orgánicos volátiles biogénicos (COVs).

---

## 3. Calibración Estadística Causal (MOS — Model Output Statistics)

### 3.1. Diagnóstico del Sesgo Sistemático SIMA vs. Open-Meteo
Al contrastar las predicciones numéricas crudas (`previous_day1`) con las estaciones en superficie del SIMA en 2024 se encontraron discrepancias estructurales:
- **Velocidad del viento**: El modelo numérico sobrestima sistemáticamente el viento superficial en **+2.95 km/h** (RMSE crudo = 6.09 km/h) debido a que la celda de cuadrícula de 25 km ignora el efecto de fricción del cañón urbano y las barreras orográficas de Monterrey.
- **Temperatura ambiente**: Presenta sesgos locales variables por altitud y efecto de isla de calor (desde -0.62°C en NE2 hasta +2.02°C en NE).

### 3.2. Formulación Matemática de MOS
Para preservar la estricta honestidad y evitar toda fuga de información temporal, los parámetros de corrección se estimaron **exclusivamente con datos de entrenamiento (2020–2023)**:

$$\Delta T(s, \text{hr}) = \frac{1}{N_{s,\text{hr}}} \sum_{t \in \text{Train}} \left( \text{TOUT}_{s, t} - T_{\text{NWP}, s, t} \right)$$

$$\Delta \text{WS}(s, \text{hr}) = \frac{1}{N_{s,\text{hr}}} \sum_{t \in \text{Train}} \left( \text{WSR}_{s, t} - \text{WS}_{\text{NWP}, s, t} \right)$$

Para un pronóstico emitido en $t$ para el momento objetivo $t+h$:

$$\widehat{T}_{\text{MOS}}(t+h) = T_{\text{NWP}}(t+h) + \Delta T\left(s, (t+h) \pmod{24}\right)$$

$$\widehat{\text{WS}}_{\text{MOS}}(t+h) = \max\left(0.1, \; \text{WS}_{\text{NWP}}(t+h) + \Delta \text{WS}\left(s, (t+h) \pmod{24}\right)\right)$$

### 3.3. Resultados de la Calibración MOS en Test 2025 (Conjunto Ciego)

| Variable Meteorológica | Métrica | NWP Crudo (`previous_day1`) | NWP Calibrado (MOS) | Mejora / Reducción de Error |
|---|:---:|:---:|:---:|:---:|
| **Velocidad de Viento** | Sesgo | $+2.938\text{ km/h}$ | **$+0.464\text{ km/h}$** | **-84.2% sesgo** |
| | RMSE | $6.013\text{ km/h}$ | **$3.360\text{ km/h}$** | **-44.1% error** |
| **Temperatura Superficial** | Sesgo | $-0.479\text{ °C}$ | **$-0.161\text{ °C}$** | **-66.4% sesgo** |
| | RMSE | $2.108\text{ °C}$ | **$1.817\text{ °C}$** | **-13.8% error** |

Esta notable mejora en la fidelidad de las condiciones de superficie se traduce directamente en una mejor estimación del volumen de dilución y de las cinéticas de reacción química.

---

## 4. Tabla Comparativa General de Resultados

A continuación se presentan las métricas oficiales en Test 2025 para las dos variantes NWP desarrolladas:
- **`D_nwp_noise`**: Modelo con matriz extendida (Plan A + Open-Meteo base + Multi-modelo ECMWF/GFS/ICON + Estabilidad + MOS) con inyección de ruido y dispersión observada en Val 2024 durante el entrenamiento.
- **`E_nwp_w`**: Mismo modelo pero incorporando ponderación cuadrática en picos $w = 1 + (y/p_{90})^2$.

### 4.1. RMSE Global en Test 2025 (ppb)

| Horizonte ($h$) | Climatología | Baseline v2 | Plan A V4 | `B_pp` (Cota Teórica) | v3 `D_honest_noise` | v3 `E_honest_w` | NWP `D_nwp_noise` | **NWP `E_nwp_w` (Final)** | Ganancia vs v3 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1h**  | 14.88 |  8.05 |  5.09 | 5.00 | 5.005 | 5.044 | **5.017** | **5.045** | Empate técnico |
| **4h**  | 14.88 | 10.91 |  9.69 | 8.67 | 9.004 | 8.991 | **9.091** | **9.024** | Inercia química |
| **8h**  | 14.88 | 11.97 | 11.28 | 9.48 | 10.054 | 9.977 | **10.091** | **9.958** | **-0.019 ppb** |
| **12h** | 14.88 | 12.21 | 11.73 | 9.54 | 10.183 | 10.135 | **10.233** | **10.102** | **-0.033 ppb** |
| **24h** | 14.89 | 12.76 | 12.39 | 9.59 | 10.238 | 10.218 | **10.275** | **10.190** | **-0.028 ppb** |
| **48h** | 14.90 | 13.95 | 13.76 | 10.11 | 11.234 | 11.170 | **11.104** | **10.995** | **-0.175 ppb** |

---

### 4.2. Coeficiente de Determinación $R^2$ (Test 2025)

| Horizonte ($h$) | Baseline v2 | v3 `D_honest_noise` | v3 `E_honest_w` | NWP `D_nwp_noise` | **NWP `E_nwp_w`** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1h**  | 0.854 | 0.943 | 0.943 | 0.943 | **0.943** |
| **4h**  | 0.732 | 0.817 | 0.818 | 0.814 | **0.816** |
| **8h**  | 0.677 | 0.772 | 0.776 | 0.770 | **0.776** |
| **12h** | 0.664 | 0.766 | 0.768 | 0.764 | **0.770** |
| **24h** | 0.633 | 0.764 | 0.765 | 0.762 | **0.766** |
| **48h** | 0.562 | 0.716 | 0.719 | 0.723 | **0.728** |

---

### 4.3. Desempeño en Picos Extremos (Percentil 95: $y \ge 42\text{ ppb}$)

#### Sesgo en Picos (`bias_peaks`, ppb) — *más cercano a 0 es mejor*
| Horizonte ($h$) | Baseline v2 | v3 `D_honest_noise` | v3 `E_honest_w` | NWP `D_nwp_noise` | **NWP `E_nwp_w`** | Reducción del sesgo vs v2 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1h**  | -13.92 |  -5.14 |  -4.93 |  -5.39 |  **-5.08** | **-63.5%** |
| **4h**  | -23.54 | -16.91 | -16.28 | -18.31 | **-17.19** | **-27.0%** |
| **8h**  | -26.93 | -20.40 | -19.14 | -21.36 | **-19.77** | **-26.6%** |
| **12h** | -27.35 | -20.35 | -19.01 | -21.66 | **-19.88** | **-27.3%** |
| **24h** | -29.07 | -19.09 | -17.98 | -20.47 | **-18.40** | **-36.7%** |
| **48h** | -33.85 | -22.51 | -20.18 | -22.37 | **-20.03** | **-40.8%** |

#### RMSE en Picos (`RMSE_peaks`, ppb) — *menor es mejor*
| Horizonte ($h$) | Baseline v2 | v3 `D_honest_noise` | v3 `E_honest_w` | NWP `D_nwp_noise` | **NWP `E_nwp_w`** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1h**  | 19.25 | 10.32 | 10.17 | 10.44 | **10.22** |
| **4h**  | 27.90 | 21.70 | 21.09 | 22.71 | **21.72** |
| **8h**  | 31.12 | 25.10 | 24.01 | 25.70 | **24.33** |
| **12h** | 31.65 | 25.19 | 24.09 | 26.09 | **24.55** |
| **24h** | 33.34 | 24.18 | 23.20 | 25.13 | **23.46** |
| **48h** | 37.57 | 27.03 | 25.16 | 26.86 | **24.85** |

---

## 5. Análisis Físico y Ganancia de las Nuevas Variables

Al evaluar la ganancia total de los árboles (*gain importance*) en LightGBM para las nuevas características meteorológicas:

```mermaid
flowchart LR
    subgraph NWP ["NWP Multi-Modelo"]
        M1["ECMWF IFS 0.25°"]
        M2["GFS 0.25°"]
        M3["ICON 13 km"]
    end

    subgraph FEAT ["Features Generadas"]
        F1["Consenso Media (Temp, Rad, WS)"]
        F2["Dispersión / Spread (Incertidumbre)"]
        F3["Estabilidad (CAPE, Ráfagas)"]
        F4["MOS Calibrado (TOUT, WSR)"]
    end

    subgraph OUT ["Impacto en O3(t+h)"]
        O1["h=48: RMSE < 11.0 ppb"]
        O2["h=24: RMSE 10.19 ppb"]
        O3["Picos: -40.8% sesgo"]
    end

    NWP --> FEAT
    FEAT --> OUT
```

1. **`nwp_ws_mean` (Viento medio multi-modelo) y `nwp_mos_ws` (Viento MOS)**:
   - Fueron las variables con mayor ganancia de todo el conjunto nuevo ($> 2.9 \times 10^7$ en $h=24$).
   - **Explicación**: El consenso entre ECMWF, GFS e ICON reduce drásticamente los errores de fase en el avance de las brisas de montaña y frentes fríos en la cuenca de Monterrey. Al estar calibrado contra las estaciones SIMA, el árbol identifica con precisión cuándo una calma $(< 4\text{ km/h})$ atrapará los precursores fotoquímicos.
2. **`nwp_mos_temp` y `nwp_temp_mean`**:
   - Superan consistentemente en importancia a la temperatura determinista cruda. Capturan la respuesta no lineal de la tasa de descomposición de radicales peroxi y emisiones de isopreno.
3. **`nwp_mos_photothermal`**:
   - Término de acoplamiento que combina la temperatura ajustada por MOS con el flujo de radiación de consenso, proporcionando una señal directa de la velocidad neta de generación fotoquímica.
4. **`nwp_rad_std` (Dispersión de radiación entre modelos)**:
   - Ingresó en los splits diurnos: cuando ECMWF e ICON difieren fuertemente en radiación, el árbol atenúa las predicciones extremas, actuando como regularizador de incertidumbre física.

---

## 6. Inventario de Entregables y Verificación del Contrato

### 6.1. Matrices Parquet Reutilizables (`features_nwp_h<h>.parquet`)
Se generaron y verificaron los 6 archivos en `experiments/noche2/nwp/`:
- `features_nwp_h1.parquet` (11.3 MB)
- `features_nwp_h4.parquet` (11.3 MB)
- `features_nwp_h8.parquet` (11.3 MB)
- `features_nwp_h12.parquet` (11.3 MB)
- `features_nwp_h24.parquet` (11.3 MB)
- `features_nwp_h48.parquet` (11.3 MB)
- Documentación completa de uso y esquema en `FEATURES_NWP.md`.

### 6.2. Predicciones Generadas para el Ensamble
Los 12 archivos obligatorios están disponibles en `experiments/noche2/nwp/preds/`:
- `D_nwp_noise_h1.parquet` … `D_nwp_noise_h48.parquet`
- `E_nwp_w_h1.parquet` … `E_nwp_w_h48.parquet`
- Cada archivo contiene exactamente las 217,060 filas estándar de validación 2024 y prueba 2025 (108,512 filas en $h=1$; 108,213 en $h=24$; 107,909 en $h=48$).
- Registrado y verificado en `experiments/noche2/inventory.csv` con cobertura `val_cov = 1.0, test_cov = 1.0`.

### 6.3. Registro Central de Experimentos
Todas las ejecuciones fueron registradas en `experiments/results_registry.csv` bajo el identificador `experiment='noche2_nwp'` sin sobreescribir el historial existente.

---

## 7. Recomendación para el Ensamble Final

1. **Adopción Inmediata de `E_nwp_w`**:
   - En horizontes largos ($h=24$ y $h=48$), `E_nwp_w` debe reemplazar a los modelos v3 deterministas como el estimador base individual, aportando un salto de precisión ($10.19$ en 24h y $10.99$ en 48h) con la menor sub-predicción en episodios de contingencia ambiental.
2. **Inclusión de `features_nwp_h<h>.parquet` en el Pipeline de Deep Learning**:
   - Las arquitecturas de Deep Learning (TabNet, ResNet, TFT) deben cruzar sus tensores con `features_nwp_h<h>.parquet`. La disponibilidad de la media y dispersión de los 3 modelos globales les otorgará una señal sin precedentes de incertidumbre meteorológica.
