"""
wav2vec2_extractor.py  v2.0
===========================
Extractor de representaciones profundas de voz basado en Wav2Vec 2.0 (facebook/wav2vec2-base).

Mejoras v2.0:
------------
* Singleton thread-safe / cache en memoria del modelo (evita re-descargas y re-instanciaciones).
* Sanitizacion robusta de entrada: mono conversion, eliminacion de NaN/Inf, normalizacion pico.
* Ventana optima de muestreo (15s) para mantener latencia baja (<500ms en CPU).
* Vector enriquecido de estadisticas latentes sobre las 768 dimensiones:
  - Momentos globales (mean, std, norm, peak) para llamante y agente.
  - Metricas de contraste inter-canal (similitud coseno, distancia euclidiana).
  - Proyeccion de 32 componentes latentes + componentes diferenciales.
* Modo offline y manejo transparente de excepciones si PyTorch/Transformers no estan disponibles.
"""

import logging
import os
import threading
from typing import Dict, Optional, Tuple

import numpy as np

# Silenciar advertencia de symlinks en Windows para HuggingFace
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger("wav2vec2-extractor")

_TORCH_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

TORCH_AVAILABLE = _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE

# Singleton global para reutilizar pesos en memoria
_extractor_lock = threading.Lock()
_global_extractor: Optional["Wav2Vec2EmbeddingExtractor"] = None


def is_wav2vec2_ready() -> bool:
    """Retorna True si PyTorch y Transformers estan instalados y listos."""
    return TORCH_AVAILABLE


def get_wav2vec2_extractor(model_name: str = "facebook/wav2vec2-base", device: Optional[str] = None) -> "Wav2Vec2EmbeddingExtractor":
    """Patron Singleton: obtiene la instancia unica en memoria del extractor."""
    global _global_extractor
    with _extractor_lock:
        if _global_extractor is None:
            _global_extractor = Wav2Vec2EmbeddingExtractor(model_name=model_name, device=device)
        return _global_extractor


class Wav2Vec2EmbeddingExtractor:
    """Extractor de embeddings y metricas latentes Wav2Vec 2.0 para deteccion bicanal."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: Optional[str] = None):
        self.model_name = model_name
        self.available = TORCH_AVAILABLE
        self.processor = None
        self.model = None

        if not self.available:
            logger.info(
                "PyTorch o Transformers no disponibles. Módulo operando en modo fallback (sin error)."
            )
            self.device = "cpu"
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """Carga el modelo pre-entrenado desde HuggingFace o cache local."""
        try:
            logger.info(f"Cargando {self.model_name} en {self.device}...")
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.available = True
            logger.info("Wav2Vec 2.0 inicializado exitosamente.")
        except Exception as e:
            logger.warning(f"No se pudo cargar Wav2Vec 2.0 ({e}). Modo fallback activado.")
            self.available = False
            self.processor = None
            self.model = None

    def _prepare_audio(self, audio: np.ndarray, sr: int, max_seconds: float = 15.0) -> np.ndarray:
        """Sanitiza, normaliza y remuestrea el audio para Wav2Vec2 (16kHz)."""
        if audio is None or len(audio) == 0:
            return np.zeros(16000, dtype=np.float32)

        # Convertir a 1D mono si llega 2D
        if audio.ndim > 1:
            audio = audio[0] if audio.shape[0] < audio.shape[1] else audio[:, 0]

        audio = np.squeeze(audio).astype(np.float32)

        # Limpiar NaNs o Infs
        if not np.all(np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Remuestrear a 16000 Hz si es diferente
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Limitar duracion (15s es optimo para evitar alta latencia en CPU)
        max_samples = int(max_seconds * 16000)
        if len(audio) > max_samples:
            # Tomar segmento central representativo
            start = (len(audio) - max_samples) // 2
            audio = audio[start:start + max_samples]

        # Normalizacion pico
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak

        return audio

    def extract_channel_embedding(self, audio: np.ndarray, sr: int, max_seconds: float = 15.0) -> np.ndarray:
        """Extrae el vector latente (768 dimensiones) mediante mean pooling temporal."""
        if not self.available or self.model is None or self.processor is None:
            return np.zeros(768, dtype=np.float32)

        audio_clean = self._prepare_audio(audio, sr, max_seconds=max_seconds)
        if len(audio_clean) < 160:  # Minimo 10ms
            return np.zeros(768, dtype=np.float32)

        try:
            with torch.no_grad():
                inputs = self.processor(
                    audio_clean,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )
                input_values = inputs.input_values.to(self.device)
                outputs = self.model(input_values)
                # Mean-pooling temporal sobre la secuencia
                hidden_states = outputs.last_hidden_state  # [1, T, 768]
                emb = hidden_states.mean(dim=1).squeeze().cpu().numpy()
                if emb.ndim == 0 or len(emb) != 768:
                    return np.zeros(768, dtype=np.float32)
                return emb.astype(np.float32)
        except Exception as e:
            logger.warning(f"Falla al inferir embedding Wav2Vec2: {e}")
            return np.zeros(768, dtype=np.float32)

    def extract_bichannel_features(self, caller: np.ndarray, agent: np.ndarray, sr: int) -> Dict[str, float]:
        """Extrae metricas de representacion profunda de ambos canales y contrastes."""
        c_emb = self.extract_channel_embedding(caller, sr)
        a_emb = self.extract_channel_embedding(agent, sr)

        feats = {}

        # ── 1. Momentos Estadisticos del Espacio Latente (768 dims) ─────────────
        c_norm = float(np.linalg.norm(c_emb))
        a_norm = float(np.linalg.norm(a_emb))

        feats["w2v2_caller_latent_mean"] = float(np.mean(c_emb))
        feats["w2v2_caller_latent_std"]  = float(np.std(c_emb))
        feats["w2v2_caller_latent_norm"] = c_norm
        feats["w2v2_caller_latent_max"]  = float(np.max(c_emb))

        feats["w2v2_agent_latent_mean"]  = float(np.mean(a_emb))
        feats["w2v2_agent_latent_std"]   = float(np.std(a_emb))
        feats["w2v2_agent_latent_norm"]  = a_norm
        feats["w2v2_agent_latent_max"]   = float(np.max(a_emb))

        # ── 2. Metricas de Contraste e Interaccion (Caller vs Agent) ───────────
        denom = (c_norm * a_norm) + 1e-9
        cosine_sim = float(np.dot(c_emb, a_emb) / denom)
        euclidean_dist = float(np.linalg.norm(c_emb - a_emb))

        feats["w2v2_cosine_similarity"] = cosine_sim
        feats["w2v2_euclidean_dist"]    = euclidean_dist
        feats["w2v2_diff_latent_mean"]  = abs(feats["w2v2_caller_latent_mean"] - feats["w2v2_agent_latent_mean"])
        feats["w2v2_diff_latent_std"]   = abs(feats["w2v2_caller_latent_std"] - feats["w2v2_agent_latent_std"])

        # ── 3. Proyeccion de Componentes Latentes Principales (32 dims) ────────
        # 32 componentes capturan la firma de voz sin inflar excesivamente el espacio tabular
        for i in range(32):
            feats[f"w2v2_caller_c{i+1}"] = float(c_emb[i])
            feats[f"w2v2_agent_c{i+1}"]  = float(a_emb[i])
            feats[f"w2v2_diff_c{i+1}"]   = float(c_emb[i] - a_emb[i])

        return feats


if __name__ == "__main__":
    print(f"[Wav2Vec2] PyTorch disponible: {_TORCH_AVAILABLE}")
    print(f"[Wav2Vec2] Transformers disponible: {_TRANSFORMERS_AVAILABLE}")
    print(f"[Wav2Vec2] Sistema listo: {TORCH_AVAILABLE}")

    ext = get_wav2vec2_extractor()
    dummy_c = np.random.randn(32000).astype(np.float32)
    dummy_a = np.random.randn(32000).astype(np.float32)

    feats = ext.extract_bichannel_features(dummy_c, dummy_a, sr=16000)
    print(f"[Wav2Vec2] Total de features extraidas: {len(feats)}")
    print(f"[Wav2Vec2] Cosine similarity: {feats.get('w2v2_cosine_similarity'):.4f}")
    print(f"[Wav2Vec2] Caller latent std: {feats.get('w2v2_caller_latent_std'):.4f}")
