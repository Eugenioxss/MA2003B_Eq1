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
* `2_clean_split.ipynb`: Limpia los datos, aplica límites físicos y separa por variables.
* `data/`: Directorio donde se almacenan los datos (no versionado en git si se ignoran los .parquet).
