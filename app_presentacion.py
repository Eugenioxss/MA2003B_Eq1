import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import os

# Configuración de la página
st.set_page_config(page_title="SIMA NowCast | Eq1", page_icon="🌬️", layout="wide")

st.title("🌬️ SIMA NowCast: Sistema de Alerta Temprana")
st.markdown("**Pronóstico Multi-Horizonte de Ozono Troposférico usando Gradient Boosting (LightGBM)**")

@st.cache_data
def cargar_datos():
    horizontes = [1, 4, 8, 12, 24, 48]
    df_completo = []

    for h in horizontes:
        ruta = f"results/multihorizonte_v2/pred_test_{h}h.parquet"
        if os.path.exists(ruta):
            df = pd.read_parquet(ruta).reset_index()

            # Identificar columnas dinámicamente por seguridad
            col_fecha = 'Date' if 'Date' in df.columns else df.columns[0]
            col_est = 'Estacion' if 'Estacion' in df.columns else 'estacion'
            cols_numericas = df.select_dtypes(include=['float64', 'float32']).columns
            col_real = [c for c in cols_numericas if 'real' in c.lower() or c == 'O3' or c == 'Real'][0]
            col_pred = [c for c in cols_numericas if 'pred' in c.lower()][0]

            df = df[[col_fecha, col_est, col_real, col_pred]].copy()
            df.columns = ['Fecha', 'Estacion', 'Real', 'Prediccion']
            df['Horizonte'] = h
            df_completo.append(df)

    if not df_completo:
        return None

    df_final = pd.concat(df_completo, ignore_index=True)
    df_final['Fecha'] = pd.to_datetime(df_final['Fecha'])
    return df_final

df_all = cargar_datos()

if df_all is None:
    st.error("❌ No se encontraron los datos de predicción. Asegúrate de ejecutar el script 7_multihorizonte_feature_eng.py primero.")
    st.stop()

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header("⚙️ Configuración del Pronóstico")
estaciones = sorted(df_all['Estacion'].unique())
estacion_sel = st.sidebar.selectbox("📍 Selecciona una Estación:", estaciones)

df_est = df_all[df_all['Estacion'] == estacion_sel]
fechas_disponibles = df_est['Fecha'].dt.date.unique()

fecha_sel = st.sidebar.selectbox("📅 Selecciona un Día (Test 2025):", sorted(fechas_disponibles, reverse=True))

df_dia = df_est[df_est['Fecha'].dt.date == fecha_sel]
horas_disponibles = df_dia['Fecha'].dt.time.unique()
hora_sel = st.sidebar.selectbox("🕒 Hora de emisión del pronóstico:", sorted(horas_disponibles))

# T0 es el momento exacto en el que estamos parados simulando hacer la predicción
t0 = pd.to_datetime(f"{fecha_sel} {hora_sel}")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip para la presentación:** Juega con la fecha y hora hasta encontrar un día donde la línea azul clara (lo que realmente pasó en el futuro) suba por encima de la línea roja punteada (70 ppb). Así demostrarás cómo tu línea roja punteada (el pronóstico) logró anticiparlo.")

# --- PROCESAMIENTO DE DATOS PARA LA GRÁFICA ---
df_real = df_est[df_est['Horizonte'] == 1].copy()

# 1. Historial (las últimas 24h hasta T0)
historial = df_real[(df_real['Fecha'] > t0 - timedelta(hours=24)) & (df_real['Fecha'] <= t0)].copy()

# 2. Puntos futuros predichos
puntos_pred = []
for h in [1, 4, 8, 12, 24, 48]:
    target_time = t0 + timedelta(hours=h)
    match = df_est[(df_est['Horizonte'] == h) & (df_est['Fecha'] == target_time)]
    if not match.empty:
        puntos_pred.append({
            'Fecha': target_time,
            'Prediccion': match['Prediccion'].values[0],
            'Horizonte': f"+{h}h"
        })

df_futuro = pd.DataFrame(puntos_pred)

# --- PANEL DE MÉTRICAS Y ALERTAS ---
st.subheader(f"📍 Estación: {estacion_sel} | 🕒 Pronóstico emitido: {t0.strftime('%Y-%m-%d %H:%M')}")

if not df_futuro.empty:
    max_pred = df_futuro['Prediccion'].max()
    col1, col2, col3 = st.columns(3)
    
    val_actual = historial['Real'].iloc[-1] if not historial.empty else 0
    col1.metric("O₃ Actual (T=0)", f"{val_actual:.1f} ppb")
    col2.metric("O₃ Máx Pronosticado", f"{max_pred:.1f} ppb", delta=f"{max_pred - 70:.1f} ppb vs NOM" if max_pred > 70 else "Dentro de norma", delta_color="inverse")

    if max_pred > 70:
        st.error("⚠️ **ALERTA AMBIENTAL:** El pronóstico indica que se superará el límite de 70 ppb (NOM-020) en las próximas 48 horas.")
    else:
        st.success("✅ **CALIDAD DEL AIRE ACEPTABLE:** No se prevén rebasos de la NOM-020 en el horizonte pronosticado.")
else:
    st.warning("No hay predicciones completas disponibles para esta hora específica.")

# --- GRÁFICA INTERACTIVA PLOTLY ---
fig = go.Figure()

# Línea histórica (Azul sólido)
if not historial.empty:
    fig.add_trace(go.Scatter(
        x=historial['Fecha'], y=historial['Real'],
        mode='lines+markers', name='O₃ Histórico (Real)',
        line=dict(color='#2E86C1', width=3), marker=dict(size=6)
    ))

# Pronóstico (Rojo punteado)
if not df_futuro.empty:
    if not historial.empty:
        punto_conexion = pd.DataFrame([{'Fecha': t0, 'Prediccion': historial['Real'].iloc[-1]}])
        df_plot_futuro = pd.concat([punto_conexion, df_futuro[['Fecha', 'Prediccion']]])
    else:
        df_plot_futuro = df_futuro

    fig.add_trace(go.Scatter(
        x=df_plot_futuro['Fecha'], y=df_plot_futuro['Prediccion'],
        mode='lines+markers', name='O₃ Pronóstico',
        line=dict(color='#E74C3C', width=3, dash='dash'), marker=dict(size=8, symbol='diamond')
    ))

    # Realidad futura (Azul claro/transparente) para comparar qué tan bueno fue el modelo
    real_futuro = df_real[(df_real['Fecha'] > t0) & (df_real['Fecha'] <= t0 + timedelta(hours=48))]
    if not real_futuro.empty:
        fig.add_trace(go.Scatter(
            x=real_futuro['Fecha'], y=real_futuro['Real'],
            mode='lines', name='O₃ Real (Lo que realmente ocurrió)',
            line=dict(color='rgba(46, 134, 193, 0.4)', width=2)
        ))

# Límite NOM-020
fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Límite NOM-020 (70 ppb)", annotation_position="top left")
# Línea vertical indicando "AHORA"
fig.add_vline(x=t0, line_dash="dash", line_color="gray", annotation_text="Momento Actual", annotation_position="top right")

fig.update_layout(
    title="Simulador Operativo: Monitoreo y Pronóstico Multi-Horizonte",
    xaxis_title="Fecha y Hora", yaxis_title="Concentración O₃ (ppb)",
    hovermode="x unified", template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("*Tecnológico de Monterrey | Proyecto Final MA2003B - Equipo 1 | Datos: SIMA Nuevo León | Motor predictivo: LightGBM*")
