# 🌪️ AirQuality DEC Analytics
*Sistema de Análisis Histórico y Clustering de Calidad del Aire*

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)
![Firebase](https://img.shields.io/badge/Firebase-039BE5?logo=Firebase&logoColor=white)

## 📖 Acerca del Proyecto
Este proyecto procesa y analiza datos históricos de calidad del aire utilizando técnicas de Deep Learning. Mediante un modelo **DEC (Deep Embedding Clustering)** desarrollado en PyTorch, el sistema analiza ventanas temporales para identificar agrupaciones de comportamiento ambiental y detectar alertas de desfase entre distintos contaminantes (como $PM_{10}$ y $O_3$).

## 🚀 Arquitectura y Flujo de Trabajo (Batch Processing)
Para optimizar costos y complejidad, el proyecto implementa un enfoque de **procesamiento por lotes (batch)**. Esto elimina la necesidad de mantener servidores costosos corriendo 24/7 y garantiza una carga instantánea en el cliente.

El flujo es el siguiente:
1. **Extracción:** Lectura de los archivos `.csv` históricos (2024/2025).
2. **Inferencia Local:** Un script de Python procesa los datos y ejecuta el modelo localmente.
3. **Consolidación:** Se genera un archivo `.json` estático con los resultados (clústeres y alertas de desfase).
4. **Despliegue:** El JSON resultante se sube directamente a **Firebase**.
5. **Consumo:** La aplicación web lee este archivo de Firebase y renderiza los datos. Rápido, barato y sin latencia.

## 🧠 Preprocesamiento y Entrenamiento

El pipeline de Machine Learning tiene directrices estrictas para evitar el colapso del modelo DEC en PyTorch:

*   **Retención Total (V0):** En esta primera iteración no se descartan registros.
*   **Limpieza Estricta de NaNs:** Dado que los tensores de PyTorch no toleran valores nulos (`NaN`) o ceros generados por fallas eléctricas de las estaciones, el script de limpieza aplica un **forward-fill** (repetición del último valor válido) o una **interpolación lineal básica** para reparar los huecos en la serie de tiempo antes del *forward pass*.
*   **Computación Local:** El procesamiento de ventanas temporales y el entrenamiento se ejecutan en **hardware local de alto rendimiento**. Esto evita los límites de RAM y los *timeouts* de la nube. 
*   > *Nota: Google Colab queda estrictamente reservado como Plan B en caso de requerir pruebas simultáneas por otros miembros del equipo.*

## 📊 Producto Mínimo Viable (MVP) y Fases
La interfaz gráfica es una PWA construida con la triada **React + Firebase**, pensada para ser ágil y responsiva.

### 🎯 Fase 1 - Prioridad (Dashboard Analítico)
- **Gráfica Interactiva de 48 hrs:** Visualización central del comportamiento temporal.
- **Selector de Contexto Histórico:** Permite seleccionar fechas específicas (ej. "Día de Año Nuevo").
- **Correlación de Variables:** Análisis visual de la caída de $PM_{10}$ vs. la subida de $O_3$ (con su característico desfase de ~3 horas), complementado con sombreado dinámico indicando los niveles de temperatura.

### 🗺️ Fase 2 - Deseable (Inteligencia Espacial)
- **Mapa Geoespacial:** Un mapa interactivo utilizando las coordenadas exactas de las **15 estaciones** de monitoreo para visualizar la huella geográfica de los clústeres.

---
*Hecho con ❤️ por el equipo de Data Science y Desarrollo.*
