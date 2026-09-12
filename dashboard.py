"""
dashboard.py
============
Dashboard de Streamlit para:
  1. Subir un WAV estéreo de prueba.
  2. Visualizar la forma de onda de ambos canales (llamante / agente).
  3. Enviarlo al backend FastAPI (/detect) y mostrar el resultado.

Ejecutar con:
    streamlit run dashboard.py
(con el backend corriendo en paralelo: uvicorn main:app --port 8000)
"""

import base64
import io

import matplotlib.pyplot as plt
import requests
import soundfile as sf
import streamlit as st

st.set_page_config(page_title="Altur Voice Detector", page_icon="🎙️", layout="centered")

st.title("🎙️ Altur Voice Detector")
st.caption("HackMTY 2026 — Clasificación humano vs. voz sintética en llamadas bancarias")

api_url = st.sidebar.text_input("URL del endpoint", "http://localhost:8000/detect")
st.sidebar.markdown("---")
st.sidebar.write("Canal 0 = llamante a clasificar")
st.sidebar.write("Canal 1 = agente del banco")

uploaded = st.file_uploader("Sube un archivo WAV estéreo", type=["wav"])

if uploaded is not None:
    audio_bytes = uploaded.read()

    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True)
    except Exception as e:
        st.error(f"No se pudo leer el WAV: {e}")
        st.stop()

    if data.shape[1] < 2:
        st.error("El archivo debe ser estéreo (2 canales).")
        st.stop()

    caller, agent = data[:, 0], data[:, 1]

    st.audio(audio_bytes, format="audio/wav")
    st.write(f"Duración: {len(caller) / sr:.2f} s — Sample rate: {sr} Hz")

    st.subheader("Forma de onda")
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    axes[0].plot(caller, linewidth=0.4)
    axes[0].set_title("Canal 0 — Llamante")
    axes[1].plot(agent, linewidth=0.4, color="darkorange")
    axes[1].set_title("Canal 1 — Agente")
    axes[1].set_xlabel("Muestras")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Clasificación")
    if st.button("Analizar llamada", type="primary"):
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        with st.spinner("Consultando el modelo..."):
            try:
                resp = requests.post(api_url, json={"audio_base64": b64_audio}, timeout=30)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                st.error(f"Error al consultar el backend ({api_url}): {e}")
                result = None

        if result is not None:
            is_synth = result["is_synthetic"]
            confidence = result["confidence"]

            col1, col2 = st.columns(2)
            col1.metric("Veredicto", "🤖 Sintética" if is_synth else "🧑 Humana")
            col2.metric("Confianza", f"{confidence:.1%}")

            if is_synth:
                st.error("Se detectaron patrones consistentes con una voz generada por IA.")
            else:
                st.success("Se detectaron patrones consistentes con habla humana natural.")

            st.progress(confidence)
else:
    st.info("Sube un archivo .wav estéreo para comenzar.")
