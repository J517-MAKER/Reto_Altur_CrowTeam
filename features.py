"""
features.py
============
Extracción de características para clasificar llamadas como HUMANO vs
VOZ SINTÉTICA (IA), a partir de audio estéreo (canal 0 = llamante a
clasificar, canal 1 = agente del banco) y, opcionalmente, los turnos
de habla anotados en turns/<anon_id>.json.

DECISIÓN DE ARQUITECTURA IMPORTANTE
------------------------------------
La función `extract_features_from_audio()` es la ÚNICA fuente de verdad
para generar el vector de características. Se usa:
  - En entrenamiento (build_dataset): con los turnos REALES del JSON.
  - En inferencia (main.py /detect): SIN turnos, porque en producción
    solo llega el WAV en base64. En ese caso, los turnos se estiman
    automáticamente con un VAD simple (estimate_turns_vad).

Esto genera un pequeño "distribution shift" entre train/inferencia que
es aceptable para un prototipo de hackathon, pero es importante que el
equipo lo tenga presente: si en producción también van a tener turnos
reales (p.ej. de un diarizador), lo ideal sería usarlos también aquí.

Esquema asumido para turns/<anon_id>.json (AJUSTAR SI DIFIERE):
{
  "channel_0": [{"start": 0.52, "end": 3.21}, ...],
  "channel_1": [{"start": 3.40, "end": 5.00}, ...]
}
`load_turns()` también acepta las claves alternativas "caller"/"agent"
y "ch0"/"ch1". Si tu formato real es distinto, ajusta esa función.
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13

NON_FEATURE_COLS = ["anon_id", "label", "split"]


# ----------------------------------------------------------------------
# Carga de audio
# ----------------------------------------------------------------------
def load_audio(path):
    """Carga un WAV estéreo sin remuestrear ni mezclar a mono.

    Retorna (caller, agent, sr) donde caller=canal 0, agent=canal 1.
    """
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        raise ValueError(f"Se esperaba audio estéreo (2 canales) en {path}, llegó mono.")
    if y.shape[0] < 2:
        raise ValueError(f"Audio con menos de 2 canales en {path}.")
    caller, agent = y[0], y[1]
    return caller, agent, sr


# ----------------------------------------------------------------------
# Turnos: carga desde JSON o estimación por VAD
# ----------------------------------------------------------------------
def load_turns(json_path):
    """Lee turns/<anon_id>.json y devuelve (caller_turns, agent_turns)
    como listas de tuplas (start_seg, end_seg) en segundos.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "turns" in raw and isinstance(raw["turns"], list):
        caller_turns = [(float(t["start"]), float(t["end"])) for t in raw["turns"] if t.get("channel") in (0, "0", "channel_0", "caller")]
        agent_turns = [(float(t["start"]), float(t["end"])) for t in raw["turns"] if t.get("channel") in (1, "1", "channel_1", "agent")]
        return caller_turns, agent_turns

    def _extract(keys):
        for k in keys:
            if k in raw:
                return [(float(seg["start"]), float(seg["end"])) for seg in raw[k]]
        return None

    caller_turns = _extract(["channel_0", "caller", "ch0", "llamante"])
    agent_turns = _extract(["channel_1", "agent", "ch1", "agente"])

    if caller_turns is None or agent_turns is None:
        raise ValueError(
            f"No se reconoce el esquema de {json_path}. "
            "Ajusta las claves esperadas en load_turns()."
        )
    return caller_turns, agent_turns


def estimate_turns_vad(y, sr, top_db=30, min_turn_dur=0.2, merge_gap=0.15):
    """Estima segmentos de habla con detección de silencios (VAD simple),
    usada como fallback cuando no hay turns/*.json (p.ej. en inferencia
    vía API, donde solo se recibe el WAV).
    """
    if len(y) == 0:
        return []
    intervals = librosa.effects.split(y, top_db=top_db)
    turns = []
    for start, end in intervals:
        t_start, t_end = start / sr, end / sr
        if turns and t_start - turns[-1][1] <= merge_gap:
            turns[-1] = (turns[-1][0], t_end)
        else:
            turns.append((t_start, t_end))
    return [(s, e) for s, e in turns if e - s >= min_turn_dur]


def compute_turn_features(caller_turns, agent_turns, duration):
    """Métricas de toma de turnos: duración de intervenciones, latencia
    de respuesta del llamante tras hablar el agente, solapamientos, etc.
    Estas métricas son clave: los bots/voces sintéticas suelen tener
    latencias de respuesta muy regulares (poca varianza) comparado con
    humanos.
    """
    feats = {}
    caller_durs = [e - s for s, e in caller_turns]
    agent_durs = [e - s for s, e in agent_turns]

    feats["caller_num_turns"] = len(caller_turns)
    feats["caller_turn_dur_mean"] = float(np.mean(caller_durs)) if caller_durs else 0.0
    feats["caller_turn_dur_std"] = float(np.std(caller_durs)) if caller_durs else 0.0
    feats["caller_speech_ratio"] = float(sum(caller_durs) / duration) if duration > 0 else 0.0

    feats["agent_num_turns"] = len(agent_turns)
    feats["agent_turn_dur_mean"] = float(np.mean(agent_durs)) if agent_durs else 0.0

    # Latencia de respuesta: fin de turno del agente -> inicio del
    # siguiente turno del llamante.
    latencies = []
    for a_start, a_end in agent_turns:
        candidates = [c_start - a_end for c_start, c_end in caller_turns if c_start >= a_end]
        if candidates:
            latencies.append(min(candidates))

    feats["response_latency_mean"] = float(np.mean(latencies)) if latencies else np.nan
    feats["response_latency_std"] = float(np.std(latencies)) if latencies else np.nan
    feats["response_latency_min"] = float(np.min(latencies)) if latencies else np.nan

    # Solapamiento (interrupciones / habla simultánea)
    overlap = 0.0
    for c_s, c_e in caller_turns:
        for a_s, a_e in agent_turns:
            overlap += max(0.0, min(c_e, a_e) - max(c_s, a_s))
    feats["overlap_ratio"] = float(overlap / duration) if duration > 0 else 0.0

    total_speech = sum(caller_durs) + sum(agent_durs) - overlap
    feats["silence_ratio"] = float(max(0.0, 1 - total_speech / duration)) if duration > 0 else 0.0

    return feats


# ----------------------------------------------------------------------
# Características acústicas del canal del llamante
# ----------------------------------------------------------------------
def extract_signal_features(y, sr):
    feats = {}

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats["rms_mean"], feats["rms_std"] = float(np.mean(rms)), float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats["zcr_mean"], feats["zcr_std"] = float(np.mean(zcr)), float(np.std(zcr))

    # Planitud espectral: voces sintéticas tienden a un espectro más
    # "plano"/regular que la voz humana natural.
    flat = librosa.feature.spectral_flatness(y=y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats["flatness_mean"], feats["flatness_std"] = float(np.mean(flat)), float(np.std(flat))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats["centroid_mean"], feats["centroid_std"] = float(np.mean(centroid)), float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats["bandwidth_mean"], feats["bandwidth_std"] = float(np.mean(bandwidth)), float(np.std(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    feats["rolloff_mean"], feats["rolloff_std"] = float(np.mean(rolloff)), float(np.std(rolloff))

    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, n_bands=3)
        feats["contrast_mean"] = float(np.mean(contrast))
    except Exception:
        feats["contrast_mean"] = 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    return feats


def extract_pitch_features(y, sr):
    """F0 (pitch) y una aproximación de "jitter". La voz sintética suele
    mostrar menor variabilidad natural de pitch que la voz humana."""
    feats = {}
    try:
        max_pitch = min(float(librosa.note_to_hz("C7")), float(sr / 2 - 10)) if sr else float(librosa.note_to_hz("C7"))
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=max_pitch,
            sr=sr,
        )
        f0_voiced = f0[~np.isnan(f0)]
        feats["f0_mean"] = float(np.mean(f0_voiced)) if len(f0_voiced) else 0.0
        feats["f0_std"] = float(np.std(f0_voiced)) if len(f0_voiced) else 0.0
        feats["f0_voiced_ratio"] = float(np.mean(voiced_flag)) if len(voiced_flag) else 0.0
        feats["f0_jitter"] = (
            float(np.mean(np.abs(np.diff(f0_voiced)))) if len(f0_voiced) > 1 else 0.0
        )
    except Exception:
        feats.update({"f0_mean": 0.0, "f0_std": 0.0, "f0_voiced_ratio": 0.0, "f0_jitter": 0.0})
    return feats


# ----------------------------------------------------------------------
# Función principal: combina todo en un solo vector de features
# ----------------------------------------------------------------------
def extract_features_from_audio(caller, agent, sr, turns=None):
    """Punto de entrada único usado por train_model.py (con turns reales)
    y por main.py /detect (turns=None -> se estiman con VAD)."""
    duration = len(caller) / sr if sr else 0.0

    feats = {}
    feats.update(extract_signal_features(caller, sr))
    feats.update(extract_pitch_features(caller, sr))

    if turns is not None:
        caller_turns, agent_turns = turns
    else:
        caller_turns = estimate_turns_vad(caller, sr)
        agent_turns = estimate_turns_vad(agent, sr)

    feats.update(compute_turn_features(caller_turns, agent_turns, duration))
    return feats


from concurrent.futures import ThreadPoolExecutor, as_completed


def _process_item_tuple(item):
    anon_id, label, split, audio_dir, turns_dir = item
    wav_path = Path(audio_dir) / f"{anon_id}.wav"
    turns_path = Path(turns_dir) / f"{anon_id}.json"

    if not wav_path.exists():
        print(f"[WARN] No existe {wav_path}, se omite.")
        return None

    try:
        caller, agent, sr = load_audio(wav_path)
    except Exception as e:
        print(f"[WARN] Error leyendo {wav_path}: {e}")
        return None

    turns = None
    if turns_path.exists():
        try:
            turns = load_turns(turns_path)
        except Exception as e:
            print(f"[WARN] turns inválidos para {anon_id} ({e}); usando VAD como fallback.")

    feats = extract_features_from_audio(caller, agent, sr, turns=turns)
    feats["anon_id"] = anon_id
    feats["label"] = label
    feats["split"] = split
    print(f"[OK] Procesado {anon_id} ({label}/{split})")
    return feats


# ----------------------------------------------------------------------
# Construcción del dataset completo a partir del manifest.csv
# ----------------------------------------------------------------------
def build_dataset(manifest_path, audio_dir, turns_dir):
    manifest = pd.read_csv(manifest_path)
    rows = []

    items = [
        (row["anon_id"], row["label"], row["split"], str(audio_dir), str(turns_dir))
        for _, row in manifest.iterrows()
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process_item_tuple, item) for item in items]
        for future in as_completed(futures):
            try:
                res = future.result()
                if res is not None:
                    rows.append(res)
            except Exception as e:
                print(f"[ERROR] Error procesando item: {e}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extrae features del dataset de llamadas.")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--audio-dir", default="data/audio")
    parser.add_argument("--turns-dir", default="data/turns")
    parser.add_argument("--out", default="data/features.csv")
    args = parser.parse_args()

    df = build_dataset(args.manifest, args.audio_dir, args.turns_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nGuardado dataset de features en {args.out} ({len(df)} filas, {df.shape[1]} columnas).")


if __name__ == "__main__":
    main()
