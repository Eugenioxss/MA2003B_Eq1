# 🌪️ AirQuality Analytics (MA2003B_Eq1)

Sistema de análisis y preprocesamiento de datos históricos de calidad del aire.

## 📖 Acerca del Proyecto

Este repositorio contiene el código fuente para el procesamiento, limpieza y estructuración de bases de datos de calidad del aire. Actualmente incluye el pipeline de datos que toma los archivos en crudo, los unifica, valida rangos físicos y separa por variable en formato `.parquet` optimizado.

## 🚀 Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto localmente:

1. **Clonar el repositorio:**
   ```bash
   git clone <url-del-repo>
   cd MA2003B_Eq1
   ```

2. **Crear un entorno virtual (recomendado):**
   ```bash
   python -m venv .venv
   ```

3. **Activar el entorno virtual:**
   * En Windows:
     ```bash
     .venv\Scripts\activate
     ```
   * En macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## 🗂️ Estructura Principal
* `1_join_raw.ipynb`: Une las bases de datos separadas en una sola historia completa.
* `1.5_xray_lostData.ipynb`: Funciona como una radiografía de los datos, analizando la información faltante o perdida.
* `2_clean_split.ipynb`: Limpia los datos, aplica límites físicos y separa por variables.
* `3_features.ipynb`: Último código de esta etapa para crear un "Dataset Inteligente" enfocado en predecir el Ozono ($O_3$). Toma las tablas limpias del Script 2, las unifica y calcula la trigonometría y la "memoria" matemática (lags).
* `data/`: Directorio donde se almacenan los datos.
