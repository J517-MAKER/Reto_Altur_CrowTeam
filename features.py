"""
features.py  v2.0
==================
Extraccion de caracteristicas para clasificar llamadas como HUMANO vs
VOZ SINTETICA (IA). v2.0 agrega:

Novedades
---------
* N_MFCC ampliado a 20 (vs 13 anteriores)
* Delta-MFCCs (derivada temporal) — voces TTS tienen transiciones mas suaves
* Spectral entropy — TTS tiene distribucion espectral mas uniforme
* Chroma features — captura armonia/tonalidad
* Shimmer — variacion de amplitud; voces sinteticas son mas estables
* Percentiles de F0 (p10, p50, p90, rango) — distribucion de pitch
* Percentiles de duracion de turnos del llamante
* Varianza de latencia de respuesta — bots tienen latencia muy regular
* Silencios por minuto
* Modo fast=True: usa yin en lugar de pyin (~10x mas rapido en inferencia)
* Extraccion paralela senal + pitch con ThreadPoolExecutor
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# Constantes
FRAME_LENGTH      = 2048   # entrenamiento (calidad alta)
FRAME_LENGTH_FAST = 1024   # inferencia rapida
HOP_LENGTH        = 512
N_MFCC            = 20     # ampliado de 13 a 20

NON_FEATURE_COLS = ["anon_id", "label", "split"]


# ── Carga de audio ─────────────────────────────────────────────────────────────
def load_audio(path):
    """Carga un WAV estereo sin remuestrear ni mezclar a mono.
    Retorna (caller, agent, sr) donde caller=canal 0, agent=canal 1.
    """
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        raise ValueError(f"Se esperaba audio estereo (2 canales) en {path}, llego mono.")
    if y.shape[0] < 2:
        raise ValueError(f"Audio con menos de 2 canales en {path}.")
    return y[0], y[1], sr


# ── Turnos ─────────────────────────────────────────────────────────────────────
def load_turns(json_path):
    """Lee turns/<anon_id>.json y devuelve (caller_turns, agent_turns)."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "turns" in raw and isinstance(raw["turns"], list):
        caller_turns = [(float(t["start"]), float(t["end"])) for t in raw["turns"]
                        if t.get("channel") in (0, "0", "channel_0", "caller")]
        agent_turns  = [(float(t["start"]), float(t["end"])) for t in raw["turns"]
                        if t.get("channel") in (1, "1", "channel_1", "agent")]
        return caller_turns, agent_turns

    def _extract(keys):
        for k in keys:
            if k in raw:
                return [(float(seg["start"]), float(seg["end"])) for seg in raw[k]]
        return None

    caller_turns = _extract(["channel_0", "caller", "ch0", "llamante"])
    agent_turns  = _extract(["channel_1", "agent",  "ch1", "agente"])

    if caller_turns is None or agent_turns is None:
        raise ValueError(f"No se reconoce el esquema de {json_path}.")
    return caller_turns, agent_turns


def estimate_turns_vad(y, sr, top_db=30, min_turn_dur=0.2, merge_gap=0.15):
    """Estima segmentos de habla con VAD simple (fallback sin JSON)."""
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
    """Metricas de toma de turnos: latencia, solapamientos, silencios, etc."""
    feats = {}
    caller_durs = [e - s for s, e in caller_turns]
    agent_durs  = [e - s for s, e in agent_turns]

    feats["caller_num_turns"]     = len(caller_turns)
    feats["caller_turn_dur_mean"] = float(np.mean(caller_durs)) if caller_durs else 0.0
    feats["caller_turn_dur_std"]  = float(np.std(caller_durs))  if caller_durs else 0.0
    feats["caller_speech_ratio"]  = float(sum(caller_durs) / duration) if duration > 0 else 0.0

    # Percentiles de duracion de turnos (NUEVO)
    if caller_durs:
        feats["caller_turn_dur_p10"] = float(np.percentile(caller_durs, 10))
        feats["caller_turn_dur_p90"] = float(np.percentile(caller_durs, 90))
    else:
        feats["caller_turn_dur_p10"] = 0.0
        feats["caller_turn_dur_p90"] = 0.0

    feats["agent_num_turns"]     = len(agent_turns)
    feats["agent_turn_dur_mean"] = float(np.mean(agent_durs)) if agent_durs else 0.0

    latencies = []
    for a_start, a_end in agent_turns:
        candidates = [c_start - a_end for c_start, _ in caller_turns if c_start >= a_end]
        if candidates:
            latencies.append(min(candidates))

    feats["response_latency_mean"] = float(np.mean(latencies)) if latencies else np.nan
    feats["response_latency_std"]  = float(np.std(latencies))  if latencies else np.nan
    feats["response_latency_min"]  = float(np.min(latencies))  if latencies else np.nan
    # Varianza: bots tienen latencia muy regular (var cercana a 0) (NUEVO)
    feats["response_latency_var"]  = float(np.var(latencies))  if latencies else np.nan

    overlap = 0.0
    for c_s, c_e in caller_turns:
        for a_s, a_e in agent_turns:
            overlap += max(0.0, min(c_e, a_e) - max(c_s, a_s))
    feats["overlap_ratio"] = float(overlap / duration) if duration > 0 else 0.0

    total_speech = sum(caller_durs) + sum(agent_durs) - overlap
    feats["silence_ratio"] = float(max(0.0, 1 - total_speech / duration)) if duration > 0 else 0.0

    # Silencios por minuto (NUEVO)
    num_silences = max(0, len(caller_turns) - 1)
    feats["silence_count_per_min"] = float(num_silences / (duration / 60)) if duration > 0 else 0.0

    return feats


# ── Caracteristicas acusticas ──────────────────────────────────────────────────
def extract_signal_features(y, sr, fast=False):
    """Features espectrales + MFCCs + delta-MFCCs + chroma + entropy + shimmer."""
    feats = {}
    fl = FRAME_LENGTH_FAST if fast else FRAME_LENGTH

    rms = librosa.feature.rms(y=y, frame_length=fl, hop_length=HOP_LENGTH)[0]
    feats["rms_mean"], feats["rms_std"] = float(np.mean(rms)), float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=fl, hop_length=HOP_LENGTH)[0]
    feats["zcr_mean"], feats["zcr_std"] = float(np.mean(zcr)), float(np.std(zcr))

    flat = librosa.feature.spectral_flatness(y=y, n_fft=fl, hop_length=HOP_LENGTH)[0]
    feats["flatness_mean"], feats["flatness_std"] = float(np.mean(flat)), float(np.std(flat))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=fl, hop_length=HOP_LENGTH)[0]
    feats["centroid_mean"], feats["centroid_std"] = float(np.mean(centroid)), float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=fl, hop_length=HOP_LENGTH)[0]
    feats["bandwidth_mean"], feats["bandwidth_std"] = float(np.mean(bandwidth)), float(np.std(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=fl, hop_length=HOP_LENGTH)[0]
    feats["rolloff_mean"], feats["rolloff_std"] = float(np.mean(rolloff)), float(np.std(rolloff))

    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=fl, hop_length=HOP_LENGTH, n_bands=3)
        feats["contrast_mean"] = float(np.mean(contrast))
        feats["contrast_std"]  = float(np.std(contrast))   # NUEVO
    except Exception:
        feats["contrast_mean"] = 0.0
        feats["contrast_std"]  = 0.0

    # MFCCs (20 coefs) + delta MFCCs ──────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=fl, hop_length=HOP_LENGTH)
    delta_mfcc = librosa.feature.delta(mfcc)
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"]       = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"]        = float(np.std(mfcc[i]))
        feats[f"delta_mfcc_{i+1}_mean"] = float(np.mean(delta_mfcc[i]))  # NUEVO
        feats[f"delta_mfcc_{i+1}_std"]  = float(np.std(delta_mfcc[i]))   # NUEVO

    # Chroma (armonia / tonalidad) ──────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=fl, hop_length=HOP_LENGTH)
    feats["chroma_mean"] = float(np.mean(chroma))  # NUEVO
    feats["chroma_std"]  = float(np.std(chroma))   # NUEVO

    # Spectral entropy (TTS tiene distribucion espectral mas uniforme) ──────────
    S = np.abs(librosa.stft(y, n_fft=fl, hop_length=HOP_LENGTH)) + 1e-9
    S_norm = S / S.sum(axis=0, keepdims=True)
    entropy = -np.sum(S_norm * np.log(S_norm + 1e-9), axis=0)
    feats["spectral_entropy_mean"] = float(np.mean(entropy))  # NUEVO
    feats["spectral_entropy_std"]  = float(np.std(entropy))   # NUEVO

    # Shimmer (variabilidad de amplitud pico a pico; TTS es mas estable) ───────
    if len(rms) > 1:
        shimmer = np.abs(np.diff(rms)) / (np.mean(rms) + 1e-9)
        feats["shimmer_mean"] = float(np.mean(shimmer))  # NUEVO
        feats["shimmer_std"]  = float(np.std(shimmer))   # NUEVO
    else:
        feats["shimmer_mean"] = 0.0
        feats["shimmer_std"]  = 0.0

    return feats


def extract_pitch_features(y, sr, fast=False):
    """F0, jitter y percentiles de pitch.

    fast=True  -> usa yin (~0.2 s, 10x mas rapido) para inferencia.
    fast=False -> usa pyin (mas preciso) para entrenamiento.
    """
    feats = {}
    try:
        fmin = float(librosa.note_to_hz("C2"))
        fmax = min(float(librosa.note_to_hz("C7")), float(sr / 2 - 10)) if sr else float(librosa.note_to_hz("C7"))

        if fast:
            fl = FRAME_LENGTH_FAST
            f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr,
                             frame_length=fl, hop_length=HOP_LENGTH)
            f0_voiced = f0[(f0 > fmin) & (f0 < fmax)]
            feats["f0_voiced_ratio"] = float(len(f0_voiced) / (len(f0) + 1e-9))
        else:
            f0, voiced_flag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
            f0_voiced = f0[~np.isnan(f0)]
            feats["f0_voiced_ratio"] = float(np.mean(voiced_flag)) if len(voiced_flag) else 0.0

        feats["f0_mean"]   = float(np.mean(f0_voiced)) if len(f0_voiced) else 0.0
        feats["f0_std"]    = float(np.std(f0_voiced))  if len(f0_voiced) else 0.0
        feats["f0_jitter"] = float(np.mean(np.abs(np.diff(f0_voiced)))) if len(f0_voiced) > 1 else 0.0

        # Percentiles de pitch (NUEVO)
        if len(f0_voiced) > 0:
            feats["f0_p10"]   = float(np.percentile(f0_voiced, 10))
            feats["f0_p50"]   = float(np.percentile(f0_voiced, 50))
            feats["f0_p90"]   = float(np.percentile(f0_voiced, 90))
            feats["f0_range"] = feats["f0_p90"] - feats["f0_p10"]
        else:
            feats["f0_p10"] = feats["f0_p50"] = feats["f0_p90"] = feats["f0_range"] = 0.0

    except Exception:
        feats.update({
            "f0_mean": 0.0, "f0_std": 0.0, "f0_voiced_ratio": 0.0, "f0_jitter": 0.0,
            "f0_p10": 0.0, "f0_p50": 0.0, "f0_p90": 0.0, "f0_range": 0.0,
        })
    return feats


# ── Funcion principal de extraccion ───────────────────────────────────────────
def extract_features_from_audio(caller, agent, sr, turns=None, fast=True):
    """Punto de entrada unico para extraccion bicanal (Llamante + Agente).

    fast=True  -> yin + frame 1024 (~10x mas rapido, usado uniformemente en train e inferencia).
    fast=False -> pyin + frame 2048 (opcional para modo de maxima resolucion).
    """
    duration = len(caller) / sr if sr else 0.0

    # Extraccion limpia sin sub-threads para evitar deadlocks de Numba/GIL
    c_signal = extract_signal_features(caller, sr, fast)
    c_pitch  = extract_pitch_features(caller, sr, fast)
    a_signal = extract_signal_features(agent,  sr, fast)
    a_pitch  = extract_pitch_features(agent,  sr, fast)

    feats = {}
    # Canal 0 (Caller)
    feats.update(c_signal)
    feats.update(c_pitch)

    # Canal 1 (Agent) con prefijo "agent_"
    for k, v in a_signal.items():
        feats[f"agent_{k}"] = v
    for k, v in a_pitch.items():
        feats[f"agent_{k}"] = v

    # ── Features diferenciales e interactivos (Caller vs Agent) ──────────────
    feats["diff_rms_mean"]      = feats.get("rms_mean", 0.0) - feats.get("agent_rms_mean", 0.0)
    feats["ratio_rms_mean"]     = (feats.get("rms_mean", 0.0) + 1e-6) / (feats.get("agent_rms_mean", 0.0) + 1e-6)
    feats["diff_f0_mean"]       = feats.get("f0_mean", 0.0) - feats.get("agent_f0_mean", 0.0)
    feats["diff_entropy_mean"]  = feats.get("spectral_entropy_mean", 0.0) - feats.get("agent_spectral_entropy_mean", 0.0)
    feats["diff_flatness_mean"] = feats.get("flatness_mean", 0.0) - feats.get("agent_flatness_mean", 0.0)
    feats["diff_zcr_mean"]      = feats.get("zcr_mean", 0.0) - feats.get("agent_zcr_mean", 0.0)
    feats["diff_shimmer_mean"]  = feats.get("shimmer_mean", 0.0) - feats.get("agent_shimmer_mean", 0.0)
    feats["diff_chroma_mean"]   = feats.get("chroma_mean", 0.0) - feats.get("agent_chroma_mean", 0.0)

    # ── Dinamica de turnos ────────────────────────────────────────────────────
    if turns is not None:
        caller_turns, agent_turns = turns
    else:
        caller_turns = estimate_turns_vad(caller, sr)
        agent_turns  = estimate_turns_vad(agent,  sr)

    feats.update(compute_turn_features(caller_turns, agent_turns, duration))
    return feats


# ── Dataset builder ───────────────────────────────────────────────────────────
def _process_item_tuple(item):
    anon_id, label, split, audio_dir, turns_dir, fast, idx, total = item
    wav_path   = Path(audio_dir) / f"{anon_id}.wav"
    turns_path = Path(turns_dir) / f"{anon_id}.json"

    if not wav_path.exists():
        print(f"[WARN] No existe {wav_path}, se omite.", flush=True)
        return None

    try:
        caller, agent, sr = load_audio(wav_path)
    except Exception as e:
        print(f"[WARN] Error leyendo {wav_path}: {e}", flush=True)
        return None

    turns = None
    if turns_path.exists():
        try:
            turns = load_turns(turns_path)
        except Exception as e:
            print(f"[WARN] turns invalidos para {anon_id} ({e}); usando VAD.", flush=True)

    # Congruencia Train/Inferencia: fast=True por defecto para eliminar distribution shift
    feats = extract_features_from_audio(caller, agent, sr, turns=turns, fast=fast)
    feats["anon_id"] = anon_id
    feats["label"]   = label
    feats["split"]   = split
    print(f"[{idx+1}/{total}] [OK] {anon_id} ({label}/{split})", flush=True)
    return feats


def build_dataset(manifest_path, audio_dir, turns_dir, fast=True):
    manifest = pd.read_csv(manifest_path)
    total = len(manifest)
    items = [
        (row["anon_id"], row["label"], row["split"], str(audio_dir), str(turns_dir), fast, idx, total)
        for idx, row in manifest.iterrows()
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_process_item_tuple, item) for item in items]
        for future in as_completed(futures):
            try:
                res = future.result()
                if res is not None:
                    rows.append(res)
            except Exception as e:
                print(f"[ERROR] {e}", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extrae features bicanal del dataset de llamadas.")
    parser.add_argument("--manifest",  default="data/manifest.csv")
    parser.add_argument("--audio-dir", default="data/audio")
    parser.add_argument("--turns-dir", default="data/turns")
    parser.add_argument("--out",       default="data/features.csv")
    parser.add_argument("--fast",      action="store_true", default=True,
                        help="Usa modo rapido con yin para congruencia con inferencia (por defecto: True)")
    parser.add_argument("--no-fast",   dest="fast", action="store_false",
                        help="Usa modo completo con pyin (mas lento)")
    args = parser.parse_args()

    print(f"Iniciando extraccion bicanal (fast={args.fast})...")
    df = build_dataset(args.manifest, args.audio_dir, args.turns_dir, fast=args.fast)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nGuardado exitosamente {args.out} ({len(df)} filas, {df.shape[1]} columnas).")


if __name__ == "__main__":
    main()
