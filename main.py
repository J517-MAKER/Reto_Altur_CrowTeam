"""
main.py
=======
API FastAPI que carga el modelo entrenado y expone:

    POST /detect
    body:  {"audio_base64": "<wav estéreo codificado en base64>"}
    resp:  {"is_synthetic": true|false, "confidence": 0.87}

El audio se decodifica, se separan los canales (0=llamante, 1=agente),
se extraen las MISMAS features que en entrenamiento (ver features.py)
y se pasa el vector resultante por el pipeline guardado.

Ejecutar con:
    uvicorn main:app --reload --port 8000
"""

import base64
import io
import logging

import joblib
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import extract_features_from_audio

MODEL_PATH = "models/model.joblib"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("altur-voice-detector")

app = FastAPI(
    title="Altur Voice Detector — HackMTY 2026",
    description="Clasifica llamadas telefónicas como humanas o generadas por IA.",
    version="0.1.0",
)

# Permite que el dashboard de Streamlit (u otro cliente local) consuma la API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_bundle = None


@app.on_event("startup")
def load_model():
    global _model_bundle
    try:
        _model_bundle = joblib.load(MODEL_PATH)
        logger.info(f"Modelo cargado desde {MODEL_PATH} "
                    f"({len(_model_bundle['feature_names'])} features).")
    except FileNotFoundError:
        logger.warning(
            f"No se encontró {MODEL_PATH}. Corre train_model.py antes de usar /detect."
        )
        _model_bundle = None


class DetectRequest(BaseModel):
    audio_base64: str = Field(..., description="WAV estéreo (8kHz, 16-bit PCM) en base64")


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_bundle is not None}


@app.post("/detect", response_model=DetectResponse)
def detect(payload: DetectRequest):
    if _model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Ejecuta train_model.py y reinicia el servidor.",
        )

    # 1) Decodificar el audio
    try:
        audio_bytes = base64.b64decode(payload.audio_base64)
        data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True, dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo decodificar el audio: {e}")

    if data.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail="Se requiere audio estéreo con 2 canales (0=llamante, 1=agente).",
        )

    caller = np.ascontiguousarray(data[:, 0])
    agent = np.ascontiguousarray(data[:, 1])

    # 2) Extraer features (sin turns.json -> se estiman con VAD, ver features.py)
    try:
        feats = extract_features_from_audio(caller, agent, sr, turns=None)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error extrayendo features: {e}")

    # 3) Alinear el vector de features al orden esperado por el modelo
    feature_names = _model_bundle["feature_names"]
    x = np.array([[feats.get(name, np.nan) for name in feature_names]], dtype=np.float64)

    pipeline = _model_bundle["pipeline"]
    proba = pipeline.predict_proba(x)[0]  # [P(human), P(synthetic)]
    pred_class = int(np.argmax(proba))

    return DetectResponse(
        is_synthetic=bool(pred_class == 1),
        confidence=round(float(proba[pred_class]), 4),
    )
