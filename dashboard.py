"""
dashboard.py  v3.1
===================
Dashboard premium para el Altur Voice Detector.
Visual redesign: animaciones, pantalla de carga, alertas premium.
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

# ── CSS Dark Premium + Animaciones ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Fondo principal con gradiente sutil ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── Animaciones globales ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    50%       { box-shadow: 0 0 0 14px rgba(16,185,129,0); }
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
    50%       { box-shadow: 0 0 0 14px rgba(239,68,68,0); }
}
@keyframes glow-purple {
    0%, 100% { box-shadow: 0 0 8px rgba(99,102,241,0.4); }
    50%       { box-shadow: 0 0 28px rgba(99,102,241,0.9); }
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes wave {
    0%, 100% { transform: scaleY(0.4); }
    50%       { transform: scaleY(1.0); }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes dot-bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40%           { transform: translateY(-10px); }
}
@keyframes slide-in-right {
    from { opacity: 0; transform: translateX(40px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes badge-pop {
    0%   { transform: scale(0.7); opacity: 0; }
    70%  { transform: scale(1.1); }
    100% { transform: scale(1); opacity: 1; }
}

/* ── Header hero ── */
.hero-header {
    animation: fadeInUp 0.7s ease both;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #6366f1, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    line-height: 1.15;
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    margin-top: 0.25rem;
    animation: fadeIn 1s ease 0.3s both;
}

/* ── Tarjetas de metricas ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeInUp 0.5s ease both;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.2);
}
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

/* ── Botón primario ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.65rem 2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
    letter-spacing: 0.3px;
    animation: glow-purple 3s ease infinite;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 28px rgba(99,102,241,0.7);
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0) scale(0.98);
}

/* ── Alertas de resultado ── */
.success-card {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 1.5px solid #10b981;
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    color: #6ee7b7;
    font-weight: 700;
    font-size: 1.25rem;
    text-align: center;
    margin: 0.8rem 0;
    animation: pulse-green 2.5s ease infinite, fadeInUp 0.4s ease both;
    letter-spacing: 0.5px;
}
.success-card .result-icon { font-size: 2.2rem; display: block; margin-bottom: 0.4rem; }
.success-card .result-label { font-size: 0.8rem; color: #34d399; letter-spacing: 2px; text-transform: uppercase; }

.danger-card {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1.5px solid #ef4444;
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    color: #fca5a5;
    font-weight: 700;
    font-size: 1.25rem;
    text-align: center;
    margin: 0.8rem 0;
    animation: pulse-red 2.5s ease infinite, fadeInUp 0.4s ease both;
    letter-spacing: 0.5px;
}
.danger-card .result-icon { font-size: 2.2rem; display: block; margin-bottom: 0.4rem; }
.danger-card .result-label { font-size: 0.8rem; color: #f87171; letter-spacing: 2px; text-transform: uppercase; }

/* ── Loader animado ── */
.loader-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2.5rem 1rem;
    animation: fadeIn 0.3s ease;
}
.loader-title {
    color: #a78bfa;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1.4rem;
    letter-spacing: 0.5px;
}
.loader-sub {
    color: #475569;
    font-size: 0.82rem;
    margin-top: 1rem;
    letter-spacing: 1px;
}
.wave-bars {
    display: flex;
    align-items: flex-end;
    gap: 5px;
    height: 44px;
}
.wave-bar {
    width: 6px;
    border-radius: 3px;
    animation: wave 1.1s ease-in-out infinite;
}
.wave-bar:nth-child(1) { background:#6366f1; animation-delay: 0s;    height: 100%; }
.wave-bar:nth-child(2) { background:#818cf8; animation-delay: 0.1s;  height: 80%; }
.wave-bar:nth-child(3) { background:#a78bfa; animation-delay: 0.2s;  height: 60%; }
.wave-bar:nth-child(4) { background:#c4b5fd; animation-delay: 0.3s;  height: 100%; }
.wave-bar:nth-child(5) { background:#a78bfa; animation-delay: 0.4s;  height: 75%; }
.wave-bar:nth-child(6) { background:#818cf8; animation-delay: 0.5s;  height: 55%; }
.wave-bar:nth-child(7) { background:#6366f1; animation-delay: 0.6s;  height: 90%; }
.wave-bar:nth-child(8) { background:#4f46e5; animation-delay: 0.7s;  height: 65%; }

/* ── Alerta de espera (evaluando) ── */
.eval-alert {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border: 1px solid #4f46e5;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0.8rem 0;
    animation: fadeInUp 0.4s ease;
}
.eval-spinner {
    width: 22px;
    height: 22px;
    border: 3px solid #312e81;
    border-top: 3px solid #818cf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
}
.eval-text { color: #c4b5fd; font-size: 0.95rem; font-weight: 500; }
.eval-text small { display: block; color: #6366f1; font-size: 0.78rem; margin-top: 2px; }

/* ── Badge de status ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    animation: badge-pop 0.5s ease both;
}
.badge-ok   { background: #052e16; border: 1px solid #10b981; color: #34d399; }
.badge-warn { background: #431407; border: 1px solid #f97316; color: #fb923c; }
.badge-err  { background: #450a0a; border: 1px solid #ef4444; color: #f87171; }
.badge-dot  { width: 7px; height: 7px; border-radius: 50%; }
.dot-green  { background: #10b981; box-shadow: 0 0 6px #10b981; animation: pulse-green 2s infinite; }
.dot-red    { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
.dot-orange { background: #f97316; box-shadow: 0 0 6px #f97316; }

/* ── Expander estilizado ── */
[data-testid="stExpander"] {
    border: 1px solid #1e293b;
    border-radius: 14px;
    background: linear-gradient(180deg, #0f172a 0%, #0a0f1e 100%);
    margin-bottom: 1.2rem;
    transition: border-color 0.3s ease;
    animation: fadeInUp 0.5s ease both;
}
[data-testid="stExpander"]:hover {
    border-color: #334155;
}

/* ── Info pill ── */
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

/* ── Divider override ── */
hr { border-color: #1e293b !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0f172a 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600;
    font-size: 0.88rem;
    color: #64748b;
    letter-spacing: 0.3px;
    transition: color 0.2s ease;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #a78bfa;
    border-bottom-color: #7c3aed !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    animation: fadeIn 0.5s ease;
}

/* ── Shimmer skeleton (placeholder) ── */
.shimmer-box {
    height: 80px;
    border-radius: 12px;
    background: linear-gradient(90deg, #1e293b 25%, #2d3748 50%, #1e293b 75%);
    background-size: 400px 100%;
    animation: shimmer 1.5s infinite;
    margin-bottom: 0.8rem;
}

/* ── Result animado con slide ── */
.result-slide {
    animation: slide-in-right 0.5s ease both;
}

/* ── Cached pill ── */
.cached-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #172554;
    border: 1px solid #1d4ed8;
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    color: #93c5fd;
    font-size: 0.78rem;
    font-weight: 600;
    animation: fadeIn 0.5s ease;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: loader HTML ────────────────────────────────────────────────────────
def loader_html(msg: str = "Evaluando audio con modelo entrenado...", sub: str = "Extrayendo 239 features bicanal") -> str:
    return f"""
    <div class="loader-container">
        <div class="loader-title">🎙️ {msg}</div>
        <div class="wave-bars">
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
        </div>
        <div class="loader-sub">⏳ {sub}</div>
    </div>
    """

def eval_alert_html(msg: str, sub: str = "") -> str:
    sub_html = f"<small>{sub}</small>" if sub else ""
    return f"""
    <div class="eval-alert">
        <div class="eval-spinner"></div>
        <div class="eval-text">{msg}{sub_html}</div>
    </div>
    """

def badge_html(label: str, kind: str = "ok") -> str:
    dot_class = {"ok": "dot-green", "err": "dot-red", "warn": "dot-orange"}.get(kind, "dot-green")
    badge_class = {"ok": "badge-ok", "err": "badge-err", "warn": "badge-warn"}.get(kind, "badge-ok")
    return f'<span class="status-badge {badge_class}"><span class="badge-dot {dot_class}"></span>{label}</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/microphone.png", width=60)
    st.markdown("## ⚙️ Configuracion")
    st.divider()

    api_url = st.text_input("URL del backend", "http://localhost:8000")
    fast_mode = st.toggle("Modo rapido (yin)", value=True,
                          help="Activo: ~5x mas veloz. Inactivo: pyin, mayor precision.")
    st.divider()
    st.caption("Canal 0 = llamante (a clasificar)")
    st.caption("Canal 1 = agente del banco")
    st.divider()

    # Estado del backend con badge animado
    try:
        r = requests.get(f"{api_url}/health", timeout=2)
        if r.ok:
            info = r.json()
            st.markdown(badge_html("Backend conectado", "ok"), unsafe_allow_html=True)
            st.caption(f"Cache: {info.get('cache_size', 0)} entradas")
        else:
            st.markdown(badge_html("Backend con error", "warn"), unsafe_allow_html=True)
    except Exception:
        st.markdown(badge_html("Backend no disponible", "err"), unsafe_allow_html=True)

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
            st.markdown(badge_html("🧠 Wav2Vec 2.0: Activo", "ok"), unsafe_allow_html=True)
        else:
            st.markdown(badge_html("⚡ Motor: Acústico Bicanal", "warn"), unsafe_allow_html=True)
    except Exception:
        pass


# ── Header Hero ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🎙️ Altur Voice Detector</div>
    <div class="hero-sub">HackMTY 2026 — Clasificación de voz humana vs. sintética en llamadas bancarias</div>
</div>
""", unsafe_allow_html=True)
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
    label = "SINTÉTICA" if is_synthetic else "HUMANA"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "color": color, "family": "Inter"}},
        title={"text": f"Confianza — {label}", "font": {"size": 15, "color": "#94a3b8", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#94a3b8"}},
            "bar": {"color": color, "thickness": 0.32},
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
        font=dict(family="Inter"),
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
        title={"text": title, "font": {"color": "#e2e8f0", "size": 13, "family": "Inter"}},
        xaxis={"title": "Tiempo (s)", "color": "#94a3b8", "gridcolor": "#1e293b"},
        yaxis={"title": "Frecuencia (Hz)", "color": "#94a3b8", "gridcolor": "#1e293b"},
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=60, r=10, t=50, b=50),
        height=200,
        font=dict(family="Inter"),
    )
    return fig


# ── Tabs principales ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Analizar", "📋 Historial", "📊 Subgrupos y Errores", "ℹ️ Acerca de"])

with tab1:
    uploaded_files = st.file_uploader(
        "Sube uno o varios archivos WAV estéreo",
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
                st.error(f"{uploaded.name}: El archivo debe ser estéreo (2 canales).")
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

                    # ── Pantalla de carga animada ──
                    loader_slot = st.empty()
                    alert_slot  = st.empty()

                    loader_slot.markdown(loader_html(), unsafe_allow_html=True)
                    alert_slot.markdown(
                        eval_alert_html(
                            "Espera un momento...",
                            "El modelo está procesando los canales de audio"
                        ),
                        unsafe_allow_html=True
                    )

                    result = None
                    try:
                        result = analyze_audio(audio_bytes, uploaded.name, fast_mode)
                    except Exception as e:
                        loader_slot.empty()
                        alert_slot.empty()
                        st.error(f"Error al consultar el backend: {e}")

                    if result:
                        loader_slot.empty()
                        alert_slot.empty()

                        is_synth   = result["is_synthetic"]
                        confidence = result["confidence"]
                        proc_ms    = result.get("processing_ms", 0)
                        cached     = result.get("cached", False)

                        # ── Gauge + resultado animado ──
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
                                    """
                                    <div class="danger-card result-slide">
                                        <span class="result-icon">🤖</span>
                                        VOZ SINTÉTICA (IA)
                                        <span class="result-label">Alerta — Posible fraude</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    """
                                    <div class="success-card result-slide">
                                        <span class="result-icon">✅</span>
                                        VOZ HUMANA
                                        <span class="result-label">Verificado — Llamante real</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            col_m1, col_m2 = st.columns(2)
                            col_m1.metric("Confianza", f"{confidence:.1%}")
                            col_m2.metric("Tiempo (backend)", f"{proc_ms:.0f} ms")

                            if cached:
                                st.markdown(
                                    '<div class="cached-pill">⚡ Resultado desde caché</div>',
                                    unsafe_allow_html=True,
                                )

                        # Guardar historial
                        st.session_state.history.append({
                            "archivo":    uploaded.name,
                            "veredicto":  "Sintetica" if is_synth else "Humana",
                            "confianza":  f"{confidence:.1%}",
                            "tiempo_ms":  proc_ms,
                            "modo":       "rapido" if fast_mode else "completo",
                        })

    else:
        # Estado vacío con instruccion animada
        st.markdown("""
        <div style="
            text-align:center;
            padding: 3.5rem 2rem;
            border: 1.5px dashed #1e293b;
            border-radius: 20px;
            background: linear-gradient(135deg, #0a0f1e, #0f172a);
            animation: fadeIn 0.8s ease;
            margin-top: 1rem;
        ">
            <div style="font-size:3.5rem; margin-bottom:1rem;">🎙️</div>
            <div style="color:#a78bfa; font-size:1.15rem; font-weight:700; margin-bottom:0.4rem;">
                Sube un archivo WAV estéreo para comenzar
            </div>
            <div style="color:#475569; font-size:0.88rem;">
                Canal 0 = llamante &nbsp;·&nbsp; Canal 1 = agente del banco
            </div>
        </div>
        """, unsafe_allow_html=True)


with tab2:
    st.subheader("Historial de análisis")
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
        col_h2.metric("Sintéticas", synth)
        col_h3.metric("Humanas", total - synth)

        if st.button("Limpiar historial"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("""
        <div style="
            text-align:center; padding:2.5rem;
            border: 1px dashed #1e293b; border-radius:16px;
            color:#475569; animation: fadeIn 0.6s ease;
        ">
            📋 Aún no hay análisis registrados en esta sesión.
        </div>
        """, unsafe_allow_html=True)


with tab3:
    st.subheader("Análisis de Desempeño por Subgrupos y Errores")
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
        c4.metric("Precisión Global", f"{g.get('accuracy', 0):.1%}")

        st.markdown("### Evaluación por Subgrupos Críticos")

        sub_tabs = st.tabs(["Duración", "Calidad Acústica / SNR", "Densidad de Turnos", "Proporción de Silencio"])

        def render_sub_table(sub_dict):
            import pandas as pd
            records = [{"Subgrupo": k, "Muestras": v["n"], "F1": v["f1"], "Precisión": v["precision"], "Recall": v["recall"], "ROC-AUC": v["roc_auc"]} for k, v in sub_dict.items()]
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

        with sub_tabs[0]:
            st.caption("Desempeño comparativo según duración de la llamada (<60s, 60-120s, >120s)")
            render_sub_table(err_data.get("by_duration", {}))

        with sub_tabs[1]:
            st.caption("Robustez ante ruido acústico y SNR estimado (Bajo, Medio, Alto SNR)")
            render_sub_table(err_data.get("by_snr", {}))

        with sub_tabs[2]:
            st.caption("Comportamiento según alternancia de turnos conversacionales")
            render_sub_table(err_data.get("by_turns", {}))

        with sub_tabs[3]:
            st.caption("Rendimiento según presencia de pausas y silencios en la llamada")
            render_sub_table(err_data.get("by_silence", {}))

        if feat_rep_path.exists():
            with open(feat_rep_path, "r", encoding="utf-8") as f:
                feat_data = json.load(f)
            top_feats = feat_data.get("top20_features", [])
            if top_feats:
                st.markdown("### Top Features más Discriminantes (Random Forest)")
                f_names = [x["name"] for x in top_feats[:12]][::-1]
                f_imps  = [x["importance"] for x in top_feats[:12]][::-1]
                fig_imp = go.Figure(go.Bar(
                    x=f_imps,
                    y=f_names,
                    orientation="h",
                    marker=dict(
                        color=f_imps,
                        colorscale=[[0, "#1d4ed8"], [0.5, "#6366f1"], [1, "#a78bfa"]],
                        showscale=False,
                    ),
                ))
                fig_imp.update_layout(
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#94a3b8", family="Inter"),
                    margin=dict(l=150, r=20, t=20, b=20),
                    height=350,
                )
                st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown("""
        <div style="
            text-align:center; padding:2rem;
            border: 1px dashed #1e293b; border-radius:16px;
            color:#475569; animation: fadeIn 0.6s ease;
        ">
            📊 Ejecuta <code>error_analysis.py</code> para generar el reporte de subgrupos.
        </div>
        """, unsafe_allow_html=True)


with tab4:
    st.subheader("Acerca del sistema")
    st.markdown("""
    **Altur Voice Detector v3.0** — HackMTY 2026

    ### Arquitectura Bicanal de Vanguardia
    - **Backend**: FastAPI + scikit-learn + LightGBM (VotingClassifier ensemble cuádruple: RF + LightGBM + GB + SVC Calibrado)
    - **Calibración**: Calibración isotónica fina y búsqueda de umbral óptimo en 181 puntos
    - **Frontend**: Streamlit con diseño dark mode premium y visualizaciones Plotly interactivas

    ### Features del Modelo (239 features bicanal)
    - **Canal 0 (Llamante)** y **Canal 1 (Agente)** analizados simétricamente
    - **MFCCs (20)** + **Delta-MFCCs** de ambos canales
    - **Spectral entropy, chroma, flatness, spectral contrast**
    - **Shimmer y Jitter** — artificialidad y estabilidad micro-temporal
    - **Pitch (F0)**: percentiles p10, p50, p90, rango F0, media, std y ratio sonoro
    - **Features diferenciales (Caller vs Agent)**: `diff_shimmer`, `diff_f0`, `diff_rms`, `diff_entropy`, `ratio_rms`
    - **Métricas conversacionales**: Latencia de respuesta, varianza, silencios y solapamientos

    ### Congruencia Train / Inferencia
    - Algoritmo de pitch `yin` y longitud de ventana estandarizados uniformemente (`fast=True`), eliminando distribution shifts.

    ### Equipo
    **CrowTeam** — HackMTY 2026
    """)
