import os
from datetime import timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SIMA NowCast & Forecast | Eq1",
    page_icon="🌬️",
    layout="wide",
)

HORIZONTES = [1, 4, 8, 12, 24, 48]

VERSIONES = {
    "v3_final": {
        "nombre": "v3.0 Final Hoy — Meta-Ensemble Pasada 5 (NWP + Imput2 + Deep2 + Refit)",
        "corto": "v3.0 Final (Pasada 5)",
        "color": "#E74C3C",
        "dash": "solid",
        "symbol": "diamond",
        "rutas": [
            "experiments/noche2/ens2/preds/ens2_best_h{h}.parquet",
            "app_deploy/results/noche2_ens2/pred_test_{h}h.parquet",
            "results/noche2_ens2/pred_test_{h}h.parquet",
        ],
    },
    "v2_noche1": {
        "nombre": "v2.0 Noche 1 — Ensemble Tabular + TargetEng + Deep (Sin NWP)",
        "corto": "v2.0 Noche 1",
        "color": "#8E44AD",
        "dash": "dashdot",
        "symbol": "circle",
        "rutas": [
            "experiments/noche/ensemble/preds/ensemble_best_h{h}.parquet",
            "app_deploy/results/noche_ens1/pred_test_{h}h.parquet",
            "results/noche_ens1/pred_test_{h}h.parquet",
        ],
    },
    "v1_lag0": {
        "nombre": "v1.0 Ronda 1 — LightGBM con lag_0 corregido (t0 observado)",
        "corto": "v1.0 Ronda 1 (lag_0)",
        "color": "#F39C12",
        "dash": "dot",
        "symbol": "square",
        "rutas": [
            "experiments/planA_quickwins/preds_V1_lag0_{h}h.parquet",
            "app_deploy/results/ronda1_lag0/pred_test_{h}h.parquet",
            "results/ronda1_lag0/pred_test_{h}h.parquet",
        ],
    },
    "v0_beta": {
        "nombre": "v0.0 Beta (Fin de Materia) — Baseline LightGBM v2 (Script 7)",
        "corto": "v0.0 Beta (Fin de Materia)",
        "color": "#7F8C8D",
        "dash": "dash",
        "symbol": "x",
        "rutas": [
            "results/multihorizonte_v2/pred_test_{h}h.parquet",
            "app_deploy/results/multihorizonte_v2/pred_test_{h}h.parquet",
        ],
    },
}


def _resolver_ruta(plantillas, h):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for tpl in plantillas:
        rel = tpl.format(h=h)
        for cand in [rel, os.path.join(base_dir, rel), os.path.join(base_dir, "..", rel)]:
            if os.path.exists(cand):
                return cand
    return None


@st.cache_data(show_spinner="Cargando predicciones multi-horizonte (Beta -> Final Hoy)...")
def cargar_todas_las_versiones():
    datos = {}
    for ver_id, meta in VERSIONES.items():
        partes = []
        for h in HORIZONTES:
            ruta = _resolver_ruta(meta["rutas"], h)
            if ruta is None:
                continue
            df = pd.read_parquet(ruta).reset_index(drop=True)
            if "split" in df.columns:
                df = df[df["split"] == "test"].copy()

            col_fecha = "Date" if "Date" in df.columns else df.columns[0]
            col_est = "Estacion" if "Estacion" in df.columns else "estacion"

            if "y_true" in df.columns:
                col_real = "y_true"
            elif "y_real" in df.columns:
                col_real = "y_real"
            elif f"Target_{h}h" in df.columns:
                col_real = f"Target_{h}h"
            else:
                cols_num = df.select_dtypes(include=["float64", "float32"]).columns
                col_real = [c for c in cols_num if "real" in c.lower() or "true" in c.lower() or "target" in c.lower()][0]

            col_pred = "y_pred" if "y_pred" in df.columns else [c for c in df.columns if "pred" in c.lower()][0]

            out = pd.DataFrame({
                "Fecha_Emision": pd.to_datetime(df[col_fecha]),
                "Estacion": df[col_est].astype(str),
                "Real": df[col_real].astype(float),
                "Prediccion": df[col_pred].astype(float),
                "Horizonte": int(h),
            })
            # Fecha_Emision es t0; el valor Real y Prediccion corresponden exactamente a t0 + h horas
            out["Fecha_Objetivo"] = out["Fecha_Emision"] + pd.to_timedelta(h, unit="h")
            out["q10"] = df["q10"].astype(float) if "q10" in df.columns else np.nan
            out["q90"] = df["q90"].astype(float) if "q90" in df.columns else np.nan
            partes.append(out)

        if partes:
            datos[ver_id] = pd.concat(partes, ignore_index=True)
    return datos


@st.cache_data
def construir_serie_real_observada(df_version):
    """
    Construye la serie temporal física exacta O3(t) sin desfase horario.
    En h=1, la fila con Fecha_Emision = t - 1h tiene Real = O3(t), es decir, Fecha_Objetivo = t.
    """
    df_h1 = df_version[df_version["Horizonte"] == 1][["Estacion", "Fecha_Objetivo", "Real"]].copy()
    df_h1 = df_h1.rename(columns={"Fecha_Objetivo": "Fecha"}).drop_duplicates(subset=["Estacion", "Fecha"])
    return df_h1.sort_values(["Estacion", "Fecha"]).reset_index(drop=True)


@st.cache_data
def calcular_tabla_evolucion(datos_versiones, estacion_filtro=None):
    filas = []
    for ver_id, meta in VERSIONES.items():
        if ver_id not in datos_versiones:
            continue
        df_v = datos_versiones[ver_id]
        if estacion_filtro and estacion_filtro != "TODAS (13 estaciones)":
            df_v = df_v[df_v["Estacion"] == estacion_filtro]

        for h in HORIZONTES:
            sub = df_v[df_v["Horizonte"] == h].dropna(subset=["Real", "Prediccion"])
            if sub.empty:
                continue
            err = sub["Prediccion"] - sub["Real"]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((sub["Real"] - sub["Real"].mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            picos = sub[sub["Real"] >= 70.0]
            rmse_picos = float(np.sqrt(np.mean((picos["Prediccion"] - picos["Real"]) ** 2))) if len(picos) > 0 else np.nan

            cov80 = np.nan
            if sub["q10"].notna().any() and sub["q90"].notna().any():
                dentro = (sub["Real"] >= sub["q10"]) & (sub["Real"] <= sub["q90"])
                cov80 = float(dentro.mean() * 100.0)

            filas.append({
                "ver_id": ver_id,
                "Versión": meta["corto"],
                "Horizonte": f"+{h}h",
                "h_num": h,
                "RMSE (ppb)": rmse,
                "MAE (ppb)": mae,
                "R²": r2,
                "RMSE NOM >70 (ppb)": rmse_picos,
                "Cobertura 80% (%)": cov80,
                "N muestras": len(sub),
            })
    return pd.DataFrame(filas)


datos_versiones = cargar_todas_las_versiones()
if not datos_versiones:
    st.error("❌ No se encontraron archivos de predicción en `experiments/` ni en `results/`.")
    st.stop()

ver_base_ref = "v3_final" if "v3_final" in datos_versiones else list(datos_versiones.keys())[0]
df_serie_real = construir_serie_real_observada(datos_versiones[ver_base_ref])

# --- ENCABEZADO ---
st.title("🌬️ SIMA NowCast & Forecast: Evolución Beta ➔ Meta-Ensemble Final")
st.markdown(
    "**Pronóstico Multi-Horizonte de Ozono Troposférico ($+1\\text{ h}$ a $+48\\text{ h}$) en el Área Metropolitana de Monterrey (Test 2025)**"
)

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Control de Versiones y Simulación")

versiones_disp = [k for k in VERSIONES.keys() if k in datos_versiones]
ver_principal = st.sidebar.selectbox(
    "🚀 Versión Principal (Activa):",
    options=versiones_disp,
    index=0,
    format_func=lambda k: VERSIONES[k]["corto"],
)

comparar_con = st.sidebar.multiselect(
    "📊 Superponer para comparar mejora:",
    options=[k for k in versiones_disp if k != ver_principal],
    default=[k for k in ["v0_beta", "v2_noche1"] if k in versiones_disp and k != ver_principal][:1],
    format_func=lambda k: VERSIONES[k]["corto"],
)

mostrar_banda = st.sidebar.checkbox(
    "🎯 Mostrar Banda Conformal 80% (q10–q90)",
    value=True,
    help="Intervalo de predicción conformal calibrado (disponible en v3.0 Final y v2.0 Noche 1).",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Selección Espacial y Temporal")

estaciones = sorted(df_serie_real["Estacion"].unique())
estacion_sel = st.sidebar.selectbox("Estación SIMA:", estaciones, index=estaciones.index("CE") if "CE" in estaciones else 0)

df_real_est = df_serie_real[df_serie_real["Estacion"] == estacion_sel].copy()

# Buscador rápido de episodios críticos (>70 ppb NOM-020)
df_picos_est = df_real_est[df_real_est["Real"] > 70.0].sort_values("Real", ascending=False)
usar_episodio = st.sidebar.checkbox("🔥 Ir a un episodio crítico (>70 ppb NOM-020)", value=False)

if usar_episodio and not df_picos_est.empty:
    opciones_pico = []
    fechas_vistas = set()
    for _, row in df_picos_est.iterrows():
        # Emitir 4 horas antes del pico para apreciar la anticipación a +4h/+8h/+24h
        t_pico = row["Fecha"]
        if t_pico.date() not in fechas_vistas:
            fechas_vistas.add(t_pico.date())
            t_emision_sug = t_pico - timedelta(hours=4)
            opciones_pico.append((t_emision_sug, f"{t_pico.strftime('%Y-%m-%d %H:%M')} (Pico: {row['Real']:.0f} ppb -> emitir -4h)"))
        if len(opciones_pico) >= 15:
            break

    sel_pico = st.sidebar.selectbox(
        "Episodios de contingencia detectados:",
        options=opciones_pico,
        format_func=lambda x: x[1],
    )
    t0 = sel_pico[0]
else:
    fechas_disponibles = sorted(df_real_est["Fecha"].dt.date.unique(), reverse=True)
    fecha_sel = st.sidebar.selectbox("📅 Día de evaluación (Test 2025):", fechas_disponibles, index=min(120, len(fechas_disponibles) - 1))
    df_dia = df_real_est[df_real_est["Fecha"].dt.date == fecha_sel]
    horas_disponibles = sorted(df_dia["Fecha"].dt.time.unique())
    hora_def = next((i for i, h in enumerate(horas_disponibles) if h.hour == 10), 0)
    hora_sel = st.sidebar.selectbox("🕒 Hora de emisión (T0):", horas_disponibles, index=hora_def)
    t0 = pd.to_datetime(f"{fecha_sel} {hora_sel}")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Nota metodológica (Alineación exacta):**\n"
    "- Cada pronóstico emitido en $T_0$ evalúa `Fecha_Emision == T0` y se grafica en su instante futuro real $T_0 + h\\text{ h}$.\n"
    "- La serie observada $\\text{O}_3(t)$ está alineada sin desfase horario."
)


def obtener_abanico_t0(ver_id, estacion, t_emision):
    df_v = datos_versiones[ver_id]
    sub = df_v[(df_v["Estacion"] == estacion) & (df_v["Fecha_Emision"] == t_emision)].sort_values("Horizonte").copy()
    return sub


# --- PESTAÑAS PRINCIPALES ---
tab_sim, tab_serie, tab_metricas = st.tabs([
    "📡 1. Simulador Operativo (T0 ➔ +48h)",
    "📈 2. Serie Continua por Horizonte (+h fijo)",
    "🏆 3. Evolución de Métricas (Beta vs. Final Hoy)",
])

# ==============================================================================
# PESTAÑA 1: SIMULADOR OPERATIVO (T0 -> +48H)
# ==============================================================================
with tab_sim:
    historial = df_real_est[(df_real_est["Fecha"] >= t0 - timedelta(hours=24)) & (df_real_est["Fecha"] <= t0)].copy()
    real_futuro = df_real_est[(df_real_est["Fecha"] > t0) & (df_real_est["Fecha"] <= t0 + timedelta(hours=48))].copy()

    abanico_prin = obtener_abanico_t0(ver_principal, estacion_sel, t0)
    abanico_beta = obtener_abanico_t0("v0_beta", estacion_sel, t0) if "v0_beta" in datos_versiones else pd.DataFrame()

    st.subheader(f"📍 Estación {estacion_sel} | 🕒 Emisión $T_0$: {t0.strftime('%Y-%m-%d %H:%M hrs')}")

    if not abanico_prin.empty:
        val_t0 = historial["Real"].iloc[-1] if not historial.empty else np.nan
        max_pred = abanico_prin["Prediccion"].max()
        max_q90 = abanico_prin["q90"].max() if abanico_prin["q90"].notna().any() else np.nan
        max_real_48h = abanico_prin["Real"].max()

        rmse_t0_prin = float(np.sqrt(np.mean((abanico_prin["Prediccion"] - abanico_prin["Real"]) ** 2)))
        rmse_t0_beta = (
            float(np.sqrt(np.mean((abanico_beta["Prediccion"] - abanico_beta["Real"]) ** 2)))
            if not abanico_beta.empty
            else np.nan
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("O₃ Observado en T0", f"{val_t0:.1f} ppb" if pd.notna(val_t0) else "--")
        c2.metric(
            f"O₃ Máx Predicho ({VERSIONES[ver_principal]['corto']})",
            f"{max_pred:.1f} ppb",
            delta=f"Real máx: {max_real_48h:.1f} ppb",
            delta_color="off",
        )
        c3.metric(
            "Cota Superior Conformal (q90 máx)",
            f"{max_q90:.1f} ppb" if pd.notna(max_q90) else "N/A (Sin banda)",
            delta="Riesgo >70 ppb (NOM-020)" if pd.notna(max_q90) and max_q90 > 70 else "Bajo umbral extremo",
            delta_color="inverse" if pd.notna(max_q90) and max_q90 > 70 else "normal",
        )
        delta_rmse_txt = (
            f"{rmse_t0_prin - rmse_t0_beta:+.2f} ppb vs Beta ({rmse_t0_beta:.2f})"
            if pd.notna(rmse_t0_beta)
            else None
        )
        c4.metric(
            "RMSE del Abanico (+1h a +48h)",
            f"{rmse_t0_prin:.2f} ppb",
            delta=delta_rmse_txt,
            delta_color="inverse",
        )

        if max_pred > 70:
            st.error(
                f"⚠️ **ALERTA DIRECTA NOM-020 (>70 ppb):** El pronóstico puntual ({max_pred:.1f} ppb) anticipa superación de norma en las próximas 48 horas."
            )
        elif pd.notna(max_q90) and max_q90 > 70:
            st.warning(
                f"🟡 **PRE-ALERTA PREVENTIVA (q90 = {max_q90:.1f} ppb > 70 ppb):** El pronóstico central es {max_pred:.1f} ppb, pero la banda superior de incertidumbre conformal (p90) rebasa los 70 ppb."
            )
        else:
            st.success("✅ **CALIDAD DEL AIRE DENTRO DE NORMA:** Ni el pronóstico puntual ni la banda superior q90 rebasan los 70 ppb.")

    # --- GRÁFICA PLOTLY ---
    fig = go.Figure()

    # 1. Historial observado (-24h a T0)
    if not historial.empty:
        fig.add_trace(
            go.Scatter(
                x=historial["Fecha"],
                y=historial["Real"],
                mode="lines+markers",
                name="O₃ Observado Histórico (-24h a T0)",
                line=dict(color="#1F618D", width=3),
                marker=dict(size=5),
            )
        )

    # 2. Realidad futura (+1h a +48h)
    if not real_futuro.empty:
        if not historial.empty:
            real_plot = pd.concat([historial.tail(1), real_futuro], ignore_index=True)
        else:
            real_plot = real_futuro
        fig.add_trace(
            go.Scatter(
                x=real_plot["Fecha"],
                y=real_plot["Real"],
                mode="lines",
                name="O₃ Real Futuro (Observado en 2025)",
                line=dict(color="rgba(41, 128, 185, 0.45)", width=2.5),
            )
        )

    # 3. Banda conformal q10-q90 del modelo principal
    if mostrar_banda and not abanico_prin.empty and abanico_prin["q10"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=pd.concat([abanico_prin["Fecha_Objetivo"], abanico_prin["Fecha_Objetivo"][::-1]]),
                y=pd.concat([abanico_prin["q90"], abanico_prin["q10"][::-1]]),
                fill="toself",
                fillcolor="rgba(231, 76, 60, 0.16)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name=f"Banda Conformal 80% [q10–q90] ({VERSIONES[ver_principal]['corto']})",
            )
        )

    # 4. Modelos secundarios de comparación (p. ej. Beta)
    for ver_comp in comparar_con:
        ab_comp = obtener_abanico_t0(ver_comp, estacion_sel, t0)
        if not ab_comp.empty:
            meta_c = VERSIONES[ver_comp]
            if not historial.empty:
                pto_t0 = pd.DataFrame([{"Fecha_Objetivo": t0, "Prediccion": historial["Real"].iloc[-1]}])
                df_c_plot = pd.concat([pto_t0, ab_comp[["Fecha_Objetivo", "Prediccion"]]], ignore_index=True)
            else:
                df_c_plot = ab_comp
            fig.add_trace(
                go.Scatter(
                    x=df_c_plot["Fecha_Objetivo"],
                    y=df_c_plot["Prediccion"],
                    mode="lines+markers",
                    name=f"Pronóstico {meta_c['corto']}",
                    line=dict(color=meta_c["color"], width=2.5, dash=meta_c["dash"]),
                    marker=dict(size=8, symbol=meta_c["symbol"]),
                )
            )

    # 5. Modelo principal activo
    if not abanico_prin.empty:
        meta_p = VERSIONES[ver_principal]
        if not historial.empty:
            pto_t0 = pd.DataFrame([{"Fecha_Objetivo": t0, "Prediccion": historial["Real"].iloc[-1]}])
            df_p_plot = pd.concat([pto_t0, abanico_prin[["Fecha_Objetivo", "Prediccion"]]], ignore_index=True)
        else:
            df_p_plot = abanico_prin
        fig.add_trace(
            go.Scatter(
                x=df_p_plot["Fecha_Objetivo"],
                y=df_p_plot["Prediccion"],
                mode="lines+markers",
                name=f"Pronóstico {meta_p['corto']}",
                line=dict(color=meta_p["color"], width=3.5, dash=meta_p["dash"]),
                marker=dict(size=10, symbol=meta_p["symbol"]),
            )
        )

    fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="#C0392B",
        annotation_text="Umbral NOM-020 (70 ppb)",
        annotation_position="top left",
    )
    fig.add_vline(
        x=t0,
        line_dash="dash",
        line_color="#566573",
        annotation_text="Emisión T0",
        annotation_position="top right",
    )

    fig.update_layout(
        height=480,
        title=f"Trayectoria Observada vs. Abanico Multi-Horizonte Emitido en T0 ({estacion_sel})",
        xaxis_title="Fecha y Hora (Tiempo Físico Objetivo)",
        yaxis_title="Concentración de O₃ (ppb)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- TABLA DETALLADA DEL ABANICO EN T0 ---
    if not abanico_prin.empty:
        st.markdown("#### 🔍 Desglose Puntual del Pronóstico Emitido en $T_0$ por Horizonte")
        filas_det = []
        for _, r in abanico_prin.iterrows():
            h = int(r["Horizonte"])
            real_val = r["Real"]
            pred_prin = r["Prediccion"]
            err_prin = abs(pred_prin - real_val)

            pred_beta = np.nan
            err_beta = np.nan
            if not abanico_beta.empty:
                m_b = abanico_beta[abanico_beta["Horizonte"] == h]
                if not m_b.empty:
                    pred_beta = float(m_b["Prediccion"].iloc[0])
                    err_beta = abs(pred_beta - real_val)

            filas_det.append({
                "Horizonte": f"+{h}h",
                "Hora Objetivo": r["Fecha_Objetivo"].strftime("%Y-%m-%d %H:%M"),
                "O₃ Real (ppb)": round(real_val, 2),
                f"Pred {VERSIONES[ver_principal]['corto']}": round(pred_prin, 2),
                "Banda 80% [q10, q90]": (
                    f"[{r['q10']:.1f}, {r['q90']:.1f}]" if pd.notna(r["q10"]) and pd.notna(r["q90"]) else "N/A"
                ),
                "Pred v0.0 Beta": round(pred_beta, 2) if pd.notna(pred_beta) else np.nan,
                f"|Error| {VERSIONES[ver_principal]['corto']}": round(err_prin, 2),
                "|Error| v0.0 Beta": round(err_beta, 2) if pd.notna(err_beta) else np.nan,
                "Mejora |Error| vs Beta (ppb)": round(err_beta - err_prin, 2) if pd.notna(err_beta) else np.nan,
            })
        st.dataframe(pd.DataFrame(filas_det), use_container_width=True, hide_index=True)

# ==============================================================================
# PESTAÑA 2: SERIE CONTINUA A HORIZONTE FIJO (+H)
# ==============================================================================
with tab_serie:
    st.subheader(f"📈 Desempeño Continuo a Horizonte Fijo — Estación {estacion_sel}")
    col_h, col_d = st.columns([1, 2])
    h_fijo = col_h.selectbox("Selecciona Horizonte Fijo:", HORIZONTES, index=4, format_func=lambda x: f"+{x} horas")
    dias_ventana = col_d.slider("Ventana de visualización alrededor de T0 (días):", min_value=2, max_value=14, value=5)

    t_ini = t0 - timedelta(days=dias_ventana // 2)
    t_fin = t0 + timedelta(days=dias_ventana)

    df_p_h = datos_versiones[ver_principal]
    sub_p = df_p_h[
        (df_p_h["Estacion"] == estacion_sel)
        & (df_p_h["Horizonte"] == h_fijo)
        & (df_p_h["Fecha_Objetivo"] >= t_ini)
        & (df_p_h["Fecha_Objetivo"] <= t_fin)
    ].sort_values("Fecha_Objetivo")

    fig_s = go.Figure()

    if not sub_p.empty:
        fig_s.add_trace(
            go.Scatter(
                x=sub_p["Fecha_Objetivo"],
                y=sub_p["Real"],
                mode="lines",
                name="O₃ Real Observado",
                line=dict(color="#1F618D", width=2.5),
            )
        )

        if mostrar_banda and sub_p["q10"].notna().any():
            fig_s.add_trace(
                go.Scatter(
                    x=pd.concat([sub_p["Fecha_Objetivo"], sub_p["Fecha_Objetivo"][::-1]]),
                    y=pd.concat([sub_p["q90"], sub_p["q10"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(231, 76, 60, 0.14)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name=f"Banda Conformal 80% (+{h_fijo}h)",
                )
            )

        for ver_comp in comparar_con:
            df_c_h = datos_versiones[ver_comp]
            sub_c = df_c_h[
                (df_c_h["Estacion"] == estacion_sel)
                & (df_c_h["Horizonte"] == h_fijo)
                & (df_c_h["Fecha_Objetivo"] >= t_ini)
                & (df_c_h["Fecha_Objetivo"] <= t_fin)
            ].sort_values("Fecha_Objetivo")
            if not sub_c.empty:
                meta_c = VERSIONES[ver_comp]
                rmse_v_c = float(np.sqrt(np.mean((sub_c["Prediccion"] - sub_c["Real"]) ** 2)))
                fig_s.add_trace(
                    go.Scatter(
                        x=sub_c["Fecha_Objetivo"],
                        y=sub_c["Prediccion"],
                        mode="lines",
                        name=f"{meta_c['corto']} (RMSE ventana: {rmse_v_c:.2f} ppb)",
                        line=dict(color=meta_c["color"], width=2, dash=meta_c["dash"]),
                    )
                )

        rmse_v_p = float(np.sqrt(np.mean((sub_p["Prediccion"] - sub_p["Real"]) ** 2)))
        meta_p = VERSIONES[ver_principal]
        fig_s.add_trace(
            go.Scatter(
                x=sub_p["Fecha_Objetivo"],
                y=sub_p["Prediccion"],
                mode="lines",
                name=f"{meta_p['corto']} (RMSE ventana: {rmse_v_p:.2f} ppb)",
                line=dict(color=meta_p["color"], width=2.8),
            )
        )

    fig_s.add_hline(y=70, line_dash="dot", line_color="#C0392B", annotation_text="NOM-020 (70 ppb)")
    fig_s.update_layout(
        height=460,
        title=f"Comparativa Continua de Pronósticos Emitidos con {h_fijo}h de Anticipación ({estacion_sel})",
        xaxis_title="Fecha y Hora Objetivo",
        yaxis_title="Concentración de O₃ (ppb)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_s, use_container_width=True)

# ==============================================================================
# PESTAÑA 3: EVOLUCIÓN DE MÉTRICAS (BETA -> FINAL HOY)
# ==============================================================================
with tab_metricas:
    st.subheader("🏆 Auditoría de Mejora Acumulada en Test 2025 (Desde Beta hasta Meta-Ensemble Pasada 5)")
    alcance = st.radio(
        "Alcance de evaluación:",
        ["TODAS (13 estaciones)", estacion_sel],
        horizontal=True,
    )
    df_ev = calcular_tabla_evolucion(datos_versiones, alcance)

    if not df_ev.empty:
        # Pivote de RMSE por versión y horizonte
        piv_rmse = df_ev.pivot(index="h_num", columns="Versión", values="RMSE (ppb)").reset_index()
        piv_rmse["Horizonte"] = piv_rmse["h_num"].apply(lambda h: f"+{h}h")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_bar = go.Figure()
            for ver_id in versiones_disp:
                meta = VERSIONES[ver_id]
                sub_v = df_ev[df_ev["ver_id"] == ver_id].sort_values("h_num")
                fig_bar.add_trace(
                    go.Bar(
                        x=sub_v["Horizonte"],
                        y=sub_v["RMSE (ppb)"],
                        name=meta["corto"],
                        marker_color=meta["color"],
                        text=sub_v["RMSE (ppb)"].round(2),
                        textposition="outside",
                    )
                )
            fig_bar.update_layout(
                barmode="group",
                height=420,
                title=f"RMSE Global en Test 2025 por Horizonte ({alcance})",
                xaxis_title="Horizonte de Pronóstico",
                yaxis_title="RMSE (ppb, menor es mejor)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_g2:
            fig_picos = go.Figure()
            for ver_id in versiones_disp:
                meta = VERSIONES[ver_id]
                sub_v = df_ev[df_ev["ver_id"] == ver_id].sort_values("h_num")
                fig_picos.add_trace(
                    go.Scatter(
                        x=sub_v["Horizonte"],
                        y=sub_v["RMSE NOM >70 (ppb)"],
                        mode="lines+markers",
                        name=meta["corto"],
                        line=dict(color=meta["color"], width=3, dash=meta["dash"]),
                        marker=dict(size=9, symbol=meta["symbol"]),
                    )
                )
            fig_picos.update_layout(
                height=420,
                title=f"RMSE en Episodios Críticos (>70 ppb NOM-020) ({alcance})",
                xaxis_title="Horizonte de Pronóstico",
                yaxis_title="RMSE en Picos >70 ppb (menor es mejor)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_picos, use_container_width=True)

        # Tabla resumen comparativa con % de mejora respecto a Beta y respecto a Ronda 1 (lag_0)
        st.markdown("#### 📋 Tabla Comparativa Completa por Versión y Horizonte (Test 2025)")
        df_mostrar = df_ev.drop(columns=["ver_id", "h_num"]).copy()
        df_mostrar["RMSE (ppb)"] = df_mostrar["RMSE (ppb)"].round(4)
        df_mostrar["MAE (ppb)"] = df_mostrar["MAE (ppb)"].round(4)
        df_mostrar["R²"] = df_mostrar["R²"].round(4)
        df_mostrar["RMSE NOM >70 (ppb)"] = df_mostrar["RMSE NOM >70 (ppb)"].round(3)
        df_mostrar["Cobertura 80% (%)"] = df_mostrar["Cobertura 80% (%)"].round(2)
        st.dataframe(df_mostrar, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    "*Tecnológico de Monterrey | Proyecto MA2003B - Equipo 1 | Datos: SIMA Nuevo León + Open-Meteo NWP | Meta-Ensemble Pasada 5 (LightGBM + CatBoost + TabResNet + GatedMLP + Conformal)*"
)
