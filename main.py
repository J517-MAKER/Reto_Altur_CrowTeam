"""
main.py  v2.1
==============
API FastAPI optimizada para el detector de voz sintetica.

Mejoras v2.1
------------
* Contrato del juez: acepta call_id, sample_rate y channels en POST /detect
* call_id logueado para trazabilidad durante la evaluacion en vivo
* sample_rate del request usado en el log (el WAV es la fuente de verdad)

Mejoras v2.0
------------
* Modo rapido (fast=true): usa yin en lugar de pyin (~10x mas veloz)
* Cache LRU por hash MD5 del audio (evita re-procesar el mismo WAV)
* Tiempo de procesamiento incluido en la respuesta (processing_ms)
* Threshold optimizado cargado del modelo (en lugar del fijo 0.5)
* Endpoint /detect/batch para multiples audios en paralelo
* Endpoint /model/info con metricas y features del modelo
* Manejo de errores mejorado

Ejecutar con:
    uvicorn main:app --reload --port 8000

Contrato del juez (POST /detect)
---------------------------------
Request:
    {"call_id": "...", "audio_base64": "<base64 WAV>", "sample_rate": 8000, "channels": 2}
Response (HTTP 200):
    {"is_synthetic": true|false, "confidence": 0.0-1.0}
"""

import base64
import hashlib
import io
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
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
    description="Clasifica llamadas telefonicas como humanas o generadas por IA.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_bundle = None

# Cache LRU simple: {md5_hex: DetectResponse_dict}
_CACHE_MAX = 50
_cache: OrderedDict = OrderedDict()


def _cache_get(key: str):
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_set(key: str, value: dict):
    _cache[key] = value
    _cache.move_to_end(key)
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


@app.on_event("startup")
def load_model():
    global _model_bundle
    try:
        _model_bundle = joblib.load(MODEL_PATH)
        logger.info(
            f"Modelo cargado: {len(_model_bundle['feature_names'])} features, "
            f"threshold={_model_bundle.get('threshold', 0.5):.3f}, "
            f"val_roc_auc={_model_bundle.get('val_roc_auc', 'N/A')}"
        )
    except FileNotFoundError:
        logger.warning(f"No se encontro {MODEL_PATH}. Corre train_model.py.")
        _model_bundle = None


# ── Modelos Pydantic ───────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    # Campos del contrato oficial del juez
    call_id: Optional[str] = Field(None, description="Identificador de la llamada (enviado por el juez)")
    audio_base64: str = Field(..., description="WAV estereo en base64 (canal 0=llamante, canal 1=agente)")
    sample_rate: Optional[int] = Field(None, description="Tasa de muestreo declarada (referencia; la fuente real es el WAV)")
    channels: Optional[int] = Field(None, description="Numero de canales declarado (referencia; la fuente real es el WAV)")
    # Campo interno del equipo
    fast: bool = Field(True, description="Modo rapido (yin). False = pyin, mas preciso pero ~10x mas lento")


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float
    processing_ms: float = Field(0.0, description="Tiempo de procesamiento en ms (campo interno, ignorado por el juez)")
    cached: bool = Field(False, description="True si la respuesta vino del cache LRU")


class BatchDetectRequest(BaseModel):
    audios: List[DetectRequest]


class BatchDetectResponse(BaseModel):
    results: List[Optional[DetectResponse]]
    total_ms: float


# ── Funcion interna de deteccion ───────────────────────────────────────────────
def _run_detect(audio_base64: str, fast: bool) -> dict:
    """Ejecuta la deteccion. Retorna dict compatible con DetectResponse."""
    t0 = time.perf_counter()

    # Cache
    cache_key = hashlib.md5(audio_base64.encode()).hexdigest() + ("_fast" if fast else "_full")
    cached = _cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    # Decodificar audio
    try:
        audio_bytes = base64.b64decode(audio_base64)
        data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True, dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo decodificar el audio: {e}")

    if data.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail="Se requiere audio estereo con 2 canales (0=llamante, 1=agente).",
        )

    caller = np.ascontiguousarray(data[:, 0])
    agent  = np.ascontiguousarray(data[:, 1])

    # Extraer features
    try:
        feats = extract_features_from_audio(caller, agent, sr, turns=None, fast=fast)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error extrayendo features: {e}")

    # Vector de features alineado al modelo con nombres de columnas
    feature_names = _model_bundle["feature_names"]
    x = pd.DataFrame([[feats.get(name, np.nan) for name in feature_names]], columns=feature_names)

    pipeline  = _model_bundle["pipeline"]
    threshold = _model_bundle.get("threshold", 0.5)

    proba      = pipeline.predict_proba(x)[0]  # [P(human), P(synthetic)]
    prob_synth = float(proba[1])
    is_synth   = prob_synth >= threshold
    confidence = prob_synth if is_synth else float(proba[0])

    processing_ms = round((time.perf_counter() - t0) * 1000, 1)

    result = {
        "is_synthetic":   is_synth,
        "confidence":     round(confidence, 4),
        "processing_ms":  processing_ms,
        "cached":         False,
    }
    _cache_set(cache_key, {k: v for k, v in result.items() if k != "cached"})
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model_bundle is not None,
        "cache_size": len(_cache),
    }


@app.get("/model/info")
def model_info():
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    return {
        "n_features":        len(_model_bundle["feature_names"]),
        "threshold":         _model_bundle.get("threshold", 0.5),
        "val_roc_auc":       _model_bundle.get("val_roc_auc"),
        "val_f1":            _model_bundle.get("val_f1"),
        "top_features":      _model_bundle["feature_names"][:10],
    }


@app.post("/detect", response_model=DetectResponse)
def detect(payload: DetectRequest):
    if _model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Ejecuta train_model.py y reinicia el servidor.",
        )
    call_tag = f"[call_id={payload.call_id}] " if payload.call_id else ""
    logger.info(
        f"{call_tag}Recibida peticion — "
        f"sample_rate={payload.sample_rate}, channels={payload.channels}, "
        f"fast={payload.fast}"
    )
    result = _run_detect(payload.audio_base64, payload.fast)
    logger.info(
        f"{call_tag}is_synthetic={result['is_synthetic']}, "
        f"confidence={result['confidence']}, "
        f"processing_ms={result['processing_ms']}"
    )
    return DetectResponse(**result)


@app.post("/detect/batch", response_model=BatchDetectResponse)
def detect_batch(payload: BatchDetectRequest):
    """Procesa multiples audios en paralelo."""
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    if len(payload.audios) > 10:
        raise HTTPException(status_code=400, detail="Maximo 10 audios por batch.")

    t0 = time.perf_counter()
    results = [None] * len(payload.audios)

    with ThreadPoolExecutor(max_workers=min(len(payload.audios), 4)) as ex:
        future_map = {
            ex.submit(_run_detect, req.audio_base64, req.fast): i
            for i, req in enumerate(payload.audios)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = DetectResponse(**future.result())
            except Exception as e:
                logger.error(f"Error en batch[{idx}]: {e}")
                results[idx] = None

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    return BatchDetectResponse(results=results, total_ms=total_ms)
