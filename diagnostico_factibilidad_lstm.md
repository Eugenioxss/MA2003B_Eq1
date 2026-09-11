# Diagnóstico de Factibilidad LSTM

## Tabla de Cobertura
| Estación | Partición | L (horas) | Ventanas Válidas | Máx. Teórico | % Cobertura |
|----------|-----------|-----------|------------------|--------------|-------------|
| CE | train | 12 | 16184 | 35053 | 46.17% |
| CE | train | 24 | 11141 | 35041 | 31.79% |
| CE | train | 48 | 6903 | 35017 | 19.71% |
| CE | val | 12 | 4590 | 8773 | 52.32% |
| CE | val | 24 | 3232 | 8761 | 36.89% |
| CE | val | 48 | 2181 | 8737 | 24.96% |
| CE | test | 12 | 4330 | 8749 | 49.49% |
| CE | test | 24 | 2924 | 8737 | 33.47% |
| CE | test | 48 | 1614 | 8713 | 18.52% |
| NE | train | 12 | 13036 | 35053 | 37.19% |
| NE | train | 24 | 8203 | 35041 | 23.41% |
| NE | train | 48 | 4614 | 35017 | 13.18% |
| NE | val | 12 | 2055 | 8773 | 23.42% |
| NE | val | 24 | 1145 | 8761 | 13.07% |
| NE | val | 48 | 548 | 8737 | 6.27% |
| NE | test | 12 | 376 | 8749 | 4.30% |
| NE | test | 24 | 272 | 8737 | 3.11% |
| NE | test | 48 | 189 | 8713 | 2.17% |
| NE2 | train | 12 | 9259 | 35053 | 26.41% |
| NE2 | train | 24 | 4969 | 35041 | 14.18% |
| NE2 | train | 48 | 2534 | 35017 | 7.24% |
| NE2 | val | 12 | 1569 | 8773 | 17.88% |
| NE2 | val | 24 | 539 | 8761 | 6.15% |
| NE2 | val | 48 | 175 | 8737 | 2.00% |
| NE2 | test | 12 | 145 | 8749 | 1.66% |
| NE2 | test | 24 | 64 | 8737 | 0.73% |
| NE2 | test | 48 | 0 | 8713 | 0.00% |
| NO | train | 12 | 5000 | 35053 | 14.26% |
| NO | train | 24 | 2329 | 35041 | 6.65% |
| NO | train | 48 | 976 | 35017 | 2.79% |
| NO | val | 12 | 122 | 8773 | 1.39% |
| NO | val | 24 | 0 | 8761 | 0.00% |
| NO | val | 48 | 0 | 8737 | 0.00% |
| NO | test | 12 | 151 | 8749 | 1.73% |
| NO | test | 24 | 80 | 8737 | 0.92% |
| NO | test | 48 | 27 | 8713 | 0.31% |
| NO2 | train | 12 | 10622 | 35053 | 30.30% |
| NO2 | train | 24 | 6203 | 35041 | 17.70% |
| NO2 | train | 48 | 3002 | 35017 | 8.57% |
| NO2 | val | 12 | 1206 | 8773 | 13.75% |
| NO2 | val | 24 | 554 | 8761 | 6.32% |
| NO2 | val | 48 | 170 | 8737 | 1.95% |
| NO2 | test | 12 | 2797 | 8749 | 31.97% |
| NO2 | test | 24 | 1899 | 8737 | 21.74% |
| NO2 | test | 48 | 1067 | 8713 | 12.25% |
| NTE | train | 12 | 5920 | 35053 | 16.89% |
| NTE | train | 24 | 2150 | 35041 | 6.14% |
| NTE | train | 48 | 607 | 35017 | 1.73% |
| NTE | val | 12 | 2500 | 8773 | 28.50% |
| NTE | val | 24 | 914 | 8761 | 10.43% |
| NTE | val | 48 | 199 | 8737 | 2.28% |
| NTE | test | 12 | 2820 | 8749 | 32.23% |
| NTE | test | 24 | 1613 | 8737 | 18.46% |
| NTE | test | 48 | 786 | 8713 | 9.02% |
| NTE2 | train | 12 | 10765 | 35053 | 30.71% |
| NTE2 | train | 24 | 5383 | 35041 | 15.36% |
| NTE2 | train | 48 | 2280 | 35017 | 6.51% |
| NTE2 | val | 12 | 3761 | 8773 | 42.87% |
| NTE2 | val | 24 | 2000 | 8761 | 22.83% |
| NTE2 | val | 48 | 798 | 8737 | 9.13% |
| NTE2 | test | 12 | 3860 | 8749 | 44.12% |
| NTE2 | test | 24 | 2180 | 8737 | 24.95% |
| NTE2 | test | 48 | 894 | 8713 | 10.26% |
| SE | train | 12 | 9951 | 35053 | 28.39% |
| SE | train | 24 | 5520 | 35041 | 15.75% |
| SE | train | 48 | 2606 | 35017 | 7.44% |
| SE | val | 12 | 3868 | 8773 | 44.09% |
| SE | val | 24 | 2458 | 8761 | 28.06% |
| SE | val | 48 | 1425 | 8737 | 16.31% |
| SE | test | 12 | 4558 | 8749 | 52.10% |
| SE | test | 24 | 3039 | 8737 | 34.78% |
| SE | test | 48 | 1652 | 8713 | 18.96% |
| SE2 | train | 12 | 8368 | 35053 | 23.87% |
| SE2 | train | 24 | 3908 | 35041 | 11.15% |
| SE2 | train | 48 | 1221 | 35017 | 3.49% |
| SE2 | val | 12 | 3233 | 8773 | 36.85% |
| SE2 | val | 24 | 1562 | 8761 | 17.83% |
| SE2 | val | 48 | 451 | 8737 | 5.16% |
| SE2 | test | 12 | 3587 | 8749 | 41.00% |
| SE2 | test | 24 | 1707 | 8737 | 19.54% |
| SE2 | test | 48 | 505 | 8713 | 5.80% |
| SE3 | train | 12 | 16259 | 35053 | 46.38% |
| SE3 | train | 24 | 10162 | 35041 | 29.00% |
| SE3 | train | 48 | 5102 | 35017 | 14.57% |
| SE3 | val | 12 | 3665 | 8773 | 41.78% |
| SE3 | val | 24 | 1798 | 8761 | 20.52% |
| SE3 | val | 48 | 601 | 8737 | 6.88% |
| SE3 | test | 12 | 2316 | 8749 | 26.47% |
| SE3 | test | 24 | 1022 | 8737 | 11.70% |
| SE3 | test | 48 | 328 | 8713 | 3.76% |
| SO | train | 12 | 11608 | 35053 | 33.12% |
| SO | train | 24 | 7224 | 35041 | 20.62% |
| SO | train | 48 | 4426 | 35017 | 12.64% |
| SO | val | 12 | 3453 | 8773 | 39.36% |
| SO | val | 24 | 1948 | 8761 | 22.23% |
| SO | val | 48 | 1052 | 8737 | 12.04% |
| SO | test | 12 | 1681 | 8749 | 19.21% |
| SO | test | 24 | 661 | 8737 | 7.57% |
| SO | test | 48 | 278 | 8713 | 3.19% |
| SO2 | train | 12 | 11600 | 35053 | 33.09% |
| SO2 | train | 24 | 6934 | 35041 | 19.79% |
| SO2 | train | 48 | 3461 | 35017 | 9.88% |
| SO2 | val | 12 | 3498 | 8773 | 39.87% |
| SO2 | val | 24 | 1730 | 8761 | 19.75% |
| SO2 | val | 48 | 635 | 8737 | 7.27% |
| SO2 | test | 12 | 1819 | 8749 | 20.79% |
| SO2 | test | 24 | 566 | 8737 | 6.48% |
| SO2 | test | 48 | 169 | 8713 | 1.94% |
| SUR | train | 12 | 7053 | 35053 | 20.12% |
| SUR | train | 24 | 3200 | 35041 | 9.13% |
| SUR | train | 48 | 1159 | 35017 | 3.31% |
| SUR | val | 12 | 3456 | 8773 | 39.39% |
| SUR | val | 24 | 2019 | 8761 | 23.05% |
| SUR | val | 48 | 837 | 8737 | 9.58% |
| SUR | test | 12 | 2613 | 8749 | 29.87% |
| SUR | test | 24 | 1451 | 8737 | 16.61% |
| SUR | test | 48 | 701 | 8713 | 8.05% |


## Criterios de Decisión

### 1. Viabilidad de embedding propio (Ventanas válidas en Train con L=48)
Las estaciones **NO y NTE** tienen < 1,000 ventanas válidas (976 y 607 respectivamente) en train para L=48.
Son **insuficientes para embedding propio confiable**. Se propone excluirlas de la LSTM o agruparlas con una vecina de su macro-región en vez de tener vector individual.

### 2. Comparativa L=48 vs L=24 (Evidencia de rachas largas vs huecos dispersos)
La cobertura en L=48 es **dramáticamente menor** que en L=24 en múltiples estaciones, demostrando que los datos sufren de huecos dispersos y no solo de grandes ausencias estacionales. En promedio la cobertura de ventanas sanas baja casi un 50% al doblar la ventana: el promedio en Train para L=24 es 16.97%, y al pasar a L=48 cae drásticamente hasta **8.52%**. La gran mayoría de las estaciones quedan con un dígito de porcentaje de cobertura.

### 3. Confiabilidad de la partición de Test (2025)
La cobertura en la partición de Test (2025) es **notablemente peor** que en Train para múltiples estaciones clave.
En estaciones como NE (23.41% a 3.11%), NE2 (14.18% a 0.73%), SO (20.62% a 7.57%), SO2 (19.79% a 6.48%) y SE3 (29.00% a 11.70%), el porcentaje de cobertura en Test L=24 colapsa respecto a Train. Evaluaciones fuera de muestra sobre particiones de prueba tan fragmentadas no serán confiables, sin importar qué tan bien entrene el modelo.

---

### Recomendación Final
**ninguna ventana es razonable, recomendamos gradient boosting sobre lags puntuales en su lugar** — con apenas un promedio de 16.97% de cobertura para L=24 en Train y colapsos dramáticos en la partición de Test (bajando a menos de 10% en muchas estaciones), el dataset no soporta ventanas continuas.
