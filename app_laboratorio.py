import streamlit as st
import pandas as pd
import lightgbm as lgb
import os
import math

st.set_page_config(page_title="Laboratorio IA", layout="wide", page_icon="🧪")

st.title("🧪 Laboratorio Local: 'El Cerebro del Modelo'")
st.markdown("Configura el contexto de fecha/hora y mueve el clima para engañar al modelo y ver cómo reacciona.")
st.markdown("---")

model_path = 'results/multihorizonte_v2/models/lgbm_v2_1h.txt'
if not os.path.exists(model_path):
    st.error(f"No se encontró el modelo en {model_path}.")
    st.stop()

@st.cache_resource
def cargar_modelo():
    return lgb.Booster(model_file=model_path)

bst = cargar_modelo()
features = bst.feature_name()

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("1. Contexto Temporal y Espacial")
    c1, c2, c3 = st.columns(3)
    estacion = c1.selectbox("📍 Estación", ["CE", "SE", "SE2", "SE3", "SUR", "NE", "NE2", "NO", "NO2", "NTE", "NTE2", "SO", "SO2"])
    fecha = c2.date_input("📅 Fecha a simular")
    hora = c3.slider("🕒 Hora del día", 0, 23, 15)

    st.subheader("2. Condiciones Atmosféricas (Inputs Básicos)")
    c4, c5 = st.columns(2)
    temp_hoy = c4.slider("🌡️ Temperatura HOY (°C)", 5.0, 45.0, 37.0, 0.5)
    temp_ayer = c5.slider("🌡️ Temperatura AYER a esta hora (°C)", 5.0, 45.0, 38.0, 0.5)
    
    c6, c7, c8 = st.columns(3)
    nox = c6.slider("🚗 Contaminación NOₓ (ppb)", 0.0, 150.0, 23.0, 1.0)
    sol = c7.slider("☀️ Radiación Solar (kW/m²)", 0.0, 1.2, 1.0, 0.05)
    o3_actual = c8.slider("☁️ Ozono Actual (ppb)", 0.0, 150.0, 60.0, 1.0)

# Traducir inputs humanos a los 80 features del modelo
base_data = {}
for f in features:
    # 1. Variables de tiempo calculadas automáticamente
    if f == 'hour': base_data[f] = float(hora)
    elif f == 'month': base_data[f] = float(fecha.month)
    elif f == 'dayofweek': base_data[f] = float(fecha.weekday())
    elif f == 'is_weekend': base_data[f] = 1.0 if fecha.weekday() >= 5 else 0.0
    elif f == 'hour_sin': base_data[f] = math.sin(2 * math.pi * hora / 24.0)
    elif f == 'hour_cos': base_data[f] = math.cos(2 * math.pi * hora / 24.0)
    elif f == 'month_sin': base_data[f] = math.sin(2 * math.pi * fecha.month / 12.0)
    elif f == 'month_cos': base_data[f] = math.cos(2 * math.pi * fecha.month / 12.0)
    
    # 2. Variables de clima (Ayer vs Hoy)
    elif 'lag_24' in f and 'TOUT' in f: base_data[f] = temp_ayer
    elif 'TOUT_vs_yesterday' in f: base_data[f] = temp_hoy - temp_ayer
    elif 'TOUT' in f: base_data[f] = temp_hoy
    
    # 3. Otros precursores
    elif 'NOX' in f: base_data[f] = nox
    elif 'SR' in f: base_data[f] = sol
    
    # Inercia del ozono
    elif 'O3' in f: base_data[f] = o3_actual
    
    # Llenado neutral para el resto
    else: base_data[f] = 5.0

df_input = pd.DataFrame([base_data])
pred = bst.predict(df_input)[0]

with col2:
    st.subheader("🔮 Ozono Predicho (+1h)")
    color = "#E74C3C" if pred > 70 else "#27AE60"
    st.markdown(f"""
    <div style="background-color: #F8F9FA; padding: 40px; border-radius: 15px; text-align: center; border: 3px solid {color}; margin-top: 20px;">
        <h1 style='color:{color}; font-size: 80px; margin: 0;'>{pred:.1f}</h1>
        <h3 style='color:#7F8C8D; margin: 0;'>ppb O₃</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info(f"**¿Qué hace el código por detrás?**\n\nAl seleccionar **{hora}:00 hrs** y dar una Temp Hoy de **{temp_hoy}°C** vs Ayer de **{temp_ayer}°C**, la app calcula automáticamente derivadas matemáticas (como `TOUT_vs_yesterday`) y transformaciones trigonométricas de la hora para construir las 80 variables exactas que exige tu modelo de Inteligencia Artificial.")
