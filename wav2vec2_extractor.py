"""
wav2vec2_extractor.py
=====================
Extractor modular de embeddings profundos con Wav2Vec 2.0 (facebook/wav2vec2-base).

Permite extraer representaciones latentes densas (768 dimensiones) para
capturar patrones sinteticos profundos directamente de la forma de onda pura,
complementando o sustituyendo las caracteristicas acusticas tradicionales.

Requisitos:
    pip install torch transformers
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("wav2vec2-extractor")

try:
    import torch
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Wav2Vec2EmbeddingExtractor:
    """Extractor de embeddings Wav2Vec 2.0 para deteccion de voz sintetica."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: Optional[str] = None):
        self.model_name = model_name
        self.available = TORCH_AVAILABLE

        if not self.available:
            logger.warning(
                "PyTorch o Transformers no estan instalados en el entorno actual. "
                "Para habilitar embeddings de Wav2Vec2 ejecuta: pip install torch transformers"
            )
            self.processor = None
            self.model = None
            self.device = "cpu"
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Cargando modelo {model_name} en {self.device}...")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info("Modelo Wav2Vec2 cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error cargando Wav2Vec2: {e}")
            self.available = False
            self.processor = None
            self.model = None

    def extract_channel_embedding(self, audio: np.ndarray, sr: int, max_seconds: float = 30.0) -> np.ndarray:
        """Extrae un vector de embedding (768 dimensiones) mediante mean-pooling temporal."""
        if not self.available or self.model is None:
            # Fallback en caso de no contar con PyTorch
            return np.zeros(768, dtype=np.float32)

        # Limitar duracion para no saturar memoria RAM/GPU en inferencia
        max_samples = int(max_seconds * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Resamplear a 16000 Hz si es necesario (requisito de Wav2Vec 2.0)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        with torch.no_grad():
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)
            outputs = self.model(input_values)
            # Pool sobre dimension temporal -> vector de 768
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return embeddings

    def extract_bichannel_embeddings(self, caller: np.ndarray, agent: np.ndarray, sr: int) -> dict:
        """Extrae embeddings de ambos canales y calcula similitud coseno de interaccion."""
        c_emb = self.extract_channel_embedding(caller, sr)
        a_emb = self.extract_channel_embedding(agent, sr)

        feats = {}
        for i, val in enumerate(c_emb[:64]):  # Top 64 componentes principales para tabular
            feats[f"w2v2_caller_dim_{i}"] = float(val)
        for i, val in enumerate(a_emb[:64]):
            feats[f"w2v2_agent_dim_{i}"] = float(val)

        # Similitud de representaciones latentes
        norm_c = np.linalg.norm(c_emb) + 1e-9
        norm_a = np.linalg.norm(a_emb) + 1e-9
        cosine_sim = float(np.dot(c_emb, a_emb) / (norm_c * norm_a))
        feats["w2v2_cosine_similarity"] = cosine_sim

        return feats


if __name__ == "__main__":
    print(f"Disponibilidad de PyTorch y Wav2Vec2: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        extractor = Wav2Vec2EmbeddingExtractor()
        dummy_audio = np.random.randn(16000).astype(np.float32)
        emb = extractor.extract_channel_embedding(dummy_audio, 16000)
        print(f"Embedding extraido con forma: {emb.shape}")
    else:
        print("Módulo listo. Instalar torch y transformers para activar inferencia profunda.")
