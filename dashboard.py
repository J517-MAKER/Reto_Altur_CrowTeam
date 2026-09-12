"""
dashboard.py  v2.0
===================
Dashboard premium para el Altur Voice Detector.

Mejoras v2.0
------------
* Dark mode con CSS custom
* Espectrograma interactivo (Plotly)
* Gauge chart de confianza
* Panel de explicabilidad (top features del modelo)
* Historial de analisis en sesion
* Tiempo de respuesta del backend visible
* Modo rapido / completo configurable desde sidebar
* Soporte batch: analizar multiples archivos a la vez
"""

import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import requests
import soundfile as sf
import streamlit as st

# ── Configuracion de pagina ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Altur Voice Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Dark Premium ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Fondo principal */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Tarjetas de metricas */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

/* Botones primarios */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    padding: 0.6rem 2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.6);
}

/* Alertas */
.success-card {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #6ee7b7;
    font-weight: 500;
    font-size: 1.1rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(16,185,129,0.2);
}
.danger-card {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #fca5a5;
    font-weight: 500;
    font-size: 1.1rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(239,68,68,0.2);
}
.info-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    color: #94a3b8;
    font-size: 0.8rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/microphone.png", width=60)
    st.title("⚙️ Configuracion")
    st.divider()

    api_url = st.text_input("URL del backend", "http://localhost:8000")
    fast_mode = st.toggle("Modo rapido (yin)", value=True,
                          help="Activo: ~5x mas veloz. Inactivo: pyin, mayor precision.")
    st.divider()
    st.caption("Canal 0 = llamante (a clasificar)")
    st.caption("Canal 1 = agente del banco")
    st.divider()

    # Estado del backend
    try:
        r = requests.get(f"{api_url}/health", timeout=2)
        if r.ok:
            info = r.json()
            st.success("Backend conectado")
            st.caption(f"Cache: {info.get('cache_size', 0)} entradas")
        else:
            st.error("Backend con error")
    except Exception:
        st.error("Backend no disponible")

    # Info del modelo
    try:
        r2 = requests.get(f"{api_url}/model/info", timeout=2)
        if r2.ok:
            minfo = r2.json()
            st.divider()
            st.caption(f"**Features:** {minfo.get('n_features', '?')}")
            st.caption(f"**ROC-AUC:** {minfo.get('val_roc_auc', '?')}")
            st.caption(f"**Threshold:** {minfo.get('threshold', 0.5):.3f}")
    except Exception:
        pass

    try:
        from wav2vec2_extractor import is_wav2vec2_ready
        st.divider()
        if is_wav2vec2_ready():
            st.success("🧠 Wav2Vec 2.0: Listo")
        else:
            st.caption("⚡ Motor: Acústico Bicanal (239 feats)")
    except Exception:
        pass


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🎙️ Altur Voice Detector")
st.markdown("**HackMTY 2026** — Clasificacion de voz humana vs. sintetica en llamadas bancarias")
st.divider()

# Historial en sesion
if "history" not in st.session_state:
    st.session_state.history = []


# ── Funcion de analisis ────────────────────────────────────────────────────────
def analyze_audio(audio_bytes: bytes, filename: str, fast: bool):
    """Envia el audio al backend y retorna el resultado."""
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    t0 = time.perf_counter()
    resp = requests.post(
        f"{api_url}/detect",
        json={"audio_base64": b64, "fast": fast},
        timeout=60,
    )
    resp.raise_for_status()
    elapsed_client = round((time.perf_counter() - t0) * 1000, 1)
    result = resp.json()
    result["client_ms"] = elapsed_client
    result["filename"]  = filename
    return result


def make_gauge(confidence: float, is_synthetic: bool) -> go.Figure:
    """Crea un gauge chart de confianza con Plotly."""
    color = "#ef4444" if is_synthetic else "#10b981"
    label = "SINTETICA" if is_synthetic else "HUMANA"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        title={"text": f"Confianza — {label}", "font": {"size": 16, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#94a3b8"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 50],  "color": "#0f172a"},
                {"range": [50, 75], "color": "#1e293b"},
                {"range": [75, 100],"color": "#0f172a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": confidence * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=30, r=30, t=60, b=20),
        height=300,
    )
    return fig


def make_spectrogram(y: np.ndarray, sr: int, title: str, color: str) -> go.Figure:
    """Espectrograma interactivo con Plotly."""
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    times = np.linspace(0, len(y) / sr, S_db.shape[1])
    freqs = librosa.mel_frequencies(n_mels=64, fmax=8000)

    fig = go.Figure(go.Heatmap(
        z=S_db,
        x=times,
        y=freqs,
        colorscale=[[0, "#0f172a"], [0.5, color], [1, "#ffffff"]],
        showscale=False,
        hovertemplate="Tiempo: %{x:.2f}s<br>Freq: %{y:.0f}Hz<br>dB: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title={"text": title, "font": {"color": "#e2e8f0", "size": 14}},
        xaxis={"title": "Tiempo (s)", "color": "#94a3b8", "gridcolor": "#1e293b"},
        yaxis={"title": "Frecuencia (Hz)", "color": "#94a3b8", "gridcolor": "#1e293b"},
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=60, r=10, t=50, b=50),
        height=200,
    )
    return fig


# ── Tabs principales ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Analizar", "Historial", "Subgrupos y Errores", "Acerca de"])

with tab1:
    uploaded_files = st.file_uploader(
        "Sube uno o varios archivos WAV estereo",
        type=["wav"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            audio_bytes = uploaded.getvalue()

            try:
                data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True, dtype="float32")
            except Exception as e:
                st.error(f"{uploaded.name}: No se pudo leer el WAV — {e}")
                continue

            if data.shape[1] < 2:
                st.error(f"{uploaded.name}: El archivo debe ser estereo (2 canales).")
                continue

            caller = data[:, 0]
            agent  = data[:, 1]
            duration = len(caller) / sr

            with st.expander(f"🎵 {uploaded.name}  ({duration:.1f}s · {sr}Hz)", expanded=True):
                # Reproductor de audio
                st.audio(audio_bytes, format="audio/wav")

                # Espectrogramas
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.plotly_chart(
                        make_spectrogram(caller, sr, "Canal 0 — Llamante", "#6366f1"),
                        use_container_width=True, config={"displayModeBar": False},
                    )
                with col_s2:
                    st.plotly_chart(
                        make_spectrogram(agent, sr, "Canal 1 — Agente", "#f59e0b"),
                        use_container_width=True, config={"displayModeBar": False},
                    )

                # Boton de analisis
                btn_key = f"analyze_{uploaded.name}"
                if st.button(f"🔍 Analizar {uploaded.name}", type="primary", key=btn_key):
                    with st.spinner("Procesando con el modelo..."):
                        try:
                            result = analyze_audio(audio_bytes, uploaded.name, fast_mode)
                        except Exception as e:
                            st.error(f"Error al consultar el backend: {e}")
                            result = None

                    if result:
                        is_synth   = result["is_synthetic"]
                        confidence = result["confidence"]
                        proc_ms    = result.get("processing_ms", 0)
                        cached     = result.get("cached", False)

                        # Gauge + resultado
                        col_g, col_r = st.columns([1, 1])
                        with col_g:
                            st.plotly_chart(
                                make_gauge(confidence, is_synth),
                                use_container_width=True,
                                config={"displayModeBar": False},
                            )
                        with col_r:
                            st.write("")
                            st.write("")
                            if is_synth:
                                st.markdown(
                                    '<div class="danger-card"> VOZ SINTETICA (IA)</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    '<div class="success-card"> VOZ HUMANA</div>',
                                    unsafe_allow_html=True,
                                )

                            col_m1, col_m2 = st.columns(2)
                            col_m1.metric("Confianza", f"{confidence:.1%}")
                            col_m2.metric("Tiempo (backend)", f"{proc_ms:.0f} ms")

                            if cached:
                                st.info("⚡ Resultado desde cache")

                        # Guardar historial
                        st.session_state.history.append({
                            "archivo":    uploaded.name,
                            "veredicto":  "Sintetica" if is_synth else "Humana",
                            "confianza":  f"{confidence:.1%}",
                            "tiempo_ms":  proc_ms,
                            "modo":       "rapido" if fast_mode else "completo",
                        })

    else:
        st.info("Sube uno o varios archivos .wav estereo para comenzar.")
        st.caption("Cada archivo debe tener 2 canales: canal 0 = llamante, canal 1 = agente.")


with tab2:
    st.subheader("Historial de analisis")
    if st.session_state.history:
        import pandas as pd
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(
            df_hist,
            use_container_width=True,
            hide_index=True,
            column_config={
                "confianza":  st.column_config.TextColumn("Confianza"),
                "tiempo_ms":  st.column_config.NumberColumn("Tiempo (ms)", format="%.0f"),
                "veredicto":  st.column_config.TextColumn("Veredicto"),
            },
        )

        col_h1, col_h2, col_h3 = st.columns(3)
        total = len(df_hist)
        synth = (df_hist["veredicto"] == "Sintetica").sum()
        col_h1.metric("Total analizados", total)
        col_h2.metric("Sinteticas", synth)
        col_h3.metric("Humanas", total - synth)

        if st.button("Limpiar historial"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Aun no hay analisis registrados en esta sesion.")


with tab3:
    st.subheader("Analisis de Desempeno por Subgrupos y Errores")
    error_rep_path = Path("models/error_analysis.json")
    feat_rep_path  = Path("models/feature_report.json")

    if error_rep_path.exists():
        with open(error_rep_path, "r", encoding="utf-8") as f:
            err_data = json.load(f)

        g = err_data.get("global", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Muestras Evaluadas", g.get("n", 0))
        c2.metric("F1 Score Global", f"{g.get('f1', 0):.4f}")
        c3.metric("ROC-AUC Global", f"{g.get('roc_auc', 0):.4f}")
        c4.metric("Precision Global", f"{g.get('accuracy', 0):.1%}")

        st.markdown("### Evaluacion por Subgrupos Criticos")

        sub_tabs = st.tabs(["Duracion", "Calidad Acustica / SNR", "Densidad de Turnos", "Proporcion de Silencio"])

        def render_sub_table(sub_dict):
            import pandas as pd
            records = [{"Subgrupo": k, "Muestras": v["n"], "F1": v["f1"], "Precision": v["precision"], "Recall": v["recall"], "ROC-AUC": v["roc_auc"]} for k, v in sub_dict.items()]
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

        with sub_tabs[0]:
            st.caption("Desempeno comparativo segun duracion de la llamada (<60s, 60-120s, >120s)")
            render_sub_table(err_data.get("by_duration", {}))

        with sub_tabs[1]:
            st.caption("Robustez ante ruido acustico y SNR estimado (Bajo, Medio, Alto SNR)")
            render_sub_table(err_data.get("by_snr", {}))

        with sub_tabs[2]:
            st.caption("Comportamiento segun alternancia de turnos conversacionales")
            render_sub_table(err_data.get("by_turns", {}))

        with sub_tabs[3]:
            st.caption("Rendimiento segun presencia de pausas y silencios en la llamada")
            render_sub_table(err_data.get("by_silence", {}))

        if feat_rep_path.exists():
            with open(feat_rep_path, "r", encoding="utf-8") as f:
                feat_data = json.load(f)
            top_feats = feat_data.get("top20_features", [])
            if top_feats:
                st.markdown("### Top Features Mas Discriminantes (Random Forest)")
                f_names = [x["name"] for x in top_feats[:12]][::-1]
                f_imps  = [x["importance"] for x in top_feats[:12]][::-1]
                fig_imp = go.Figure(go.Bar(
                    x=f_imps,
                    y=f_names,
                    orientation="h",
                    marker=dict(color="#38bdf8"),
                ))
                fig_imp.update_layout(
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#94a3b8"),
                    margin=dict(l=150, r=20, t=20, b=20),
                    height=350,
                )
                st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Ejecuta error_analysis.py para generar el reporte de subgrupos.")


with tab4:
    st.subheader("Acerca del sistema")
    st.markdown("""
    **Altur Voice Detector v3.0** — HackMTY 2026

    ### Arquitectura Bicanal de Vanguardia
    - **Backend**: FastAPI + scikit-learn + LightGBM (VotingClassifier ensemble cuádruple: RF + LightGBM + GB + SVC Calibrado)
    - **Calibración**: Calibración isotónica fina y búsqueda de umbral óptimo en 181 puntos
    - **Frontend**: Streamlit con diseño dark mode premium y visualizaciones Plotly interactivas

    ### Features del Modelo (239 features bicanal)
    - 🎧 **Canal 0 (Llamante)** y **Canal 1 (Agente)** analizados simétricamente
    - 🎵 **MFCCs (20)** + **Delta-MFCCs** de ambos canales
    - 📊 **Spectral entropy, chroma, flatness, spectral contrast**
    - 🔊 **Shimmer y Jitter** — artificialidad y estabilidad micro-temporal
    - 🎤 **Pitch (F0)**: percentiles p10, p50, p90, rango F0, media, std y ratio sonoro
    - ⚖️ **Features diferenciales (Caller vs Agent)**: `diff_shimmer`, `diff_f0`, `diff_rms`, `diff_entropy`, `ratio_rms`
    - ⏱️ **Métricas conversacionales**: Latencia de respuesta, varianza, silencios y solapamientos

    ### Congruencia Train / Inferencia
    - Algoritmo de pitch `yin` y longitud de ventana estandarizados uniformemente (`fast=True`), eliminando distribution shifts.

    ### Equipo
    **CrowTeam** — HackMTY 2026
    """)
