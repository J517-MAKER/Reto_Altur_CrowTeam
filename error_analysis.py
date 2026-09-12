"""
error_analysis.py
=================
Analisis exhaustivo de errores por subgrupos para el detector de voces sinteticas:
1. Subgrupos por duracion de llamada (<60s, 60-120s, >120s)
2. Subgrupos por calidad de audio / SNR proxy (Limpio, Moderado, Ruidoso)
3. Subgrupos por dinamica de conversacion (densidad de turnos, ratio de silencio)
4. Diagnostico detallado de falsos positivos, falsos negativos y casos de alta incertidumbre
5. Exportacion de reporte a models/error_analysis.json
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import NON_FEATURE_COLS

LABEL_POSITIVE = "synthetic"


def compute_metrics_dict(y_true, y_pred, y_proba):
    """Calcula metricas clave para un subconjunto."""
    if len(y_true) == 0:
        return {"n": 0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "roc_auc": 0.0}

    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    acc = float(np.mean(y_true == y_pred))

    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_proba)
        else:
            auc = 1.0 if acc == 1.0 else 0.0
    except Exception:
        auc = 0.0

    return {
        "n": int(len(y_true)),
        "n_synthetic": int(sum(y_true)),
        "n_human": int(len(y_true) - sum(y_true)),
        "accuracy": round(acc, 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "roc_auc": round(float(auc), 4),
    }


def analyze_by_subgroup(df, group_col, y_true, y_pred, y_proba):
    """Segmenta metricas por las categorias de group_col."""
    subgroups = {}
    for cat in df[group_col].dropna().unique():
        idx = df[group_col] == cat
        yt = y_true[idx].values
        yp = y_pred[idx]
        ypr = y_proba[idx]
        subgroups[str(cat)] = compute_metrics_dict(yt, yp, ypr)
    return subgroups


def run_error_analysis(features_path="data/features.csv",
                       manifest_path="data/manifest.csv",
                       model_path="models/model.joblib",
                       output_path="models/error_analysis.json"):
    print(f"Cargando modelo desde {model_path}...")
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    threshold = bundle.get("threshold", 0.5)

    print(f"Cargando dataset desde {features_path}...")
    df = pd.read_csv(features_path)

    # Fusionar con duracion del manifest si esta disponible
    if Path(manifest_path).exists():
        manifest = pd.read_csv(manifest_path)
        if "duration_s" in manifest.columns and "duration_s" not in df.columns:
            df = df.merge(manifest[["anon_id", "duration_s"]], on="anon_id", how="left")

    y_true = (df["label"] == LABEL_POSITIVE).astype(int)
    X = df[feature_names]

    print(f"Evaluando {len(df)} muestras con umbral = {threshold:.3f}...")
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    dur = df["duration_s"] if "duration_s" in df.columns else (df.get("caller_turn_dur_mean", 10) * df.get("caller_num_turns", 5))
    flatness = df.get("flatness_mean", pd.Series(0.01, index=df.index))
    rms_std = df.get("rms_std", pd.Series(0.05, index=df.index))
    snr_proxy = rms_std / (flatness + 1e-6)
    turns = df.get("caller_num_turns", pd.Series(5, index=df.index))
    silence = df.get("silence_ratio", pd.Series(0.2, index=df.index))

    eval_meta = pd.DataFrame({
        "pred_proba": y_proba,
        "pred_label": np.where(y_pred == 1, "synthetic", "human"),
        "correct": (y_true == y_pred),
        "confidence": np.maximum(y_proba, 1.0 - y_proba),
        "uncertainty": np.abs(y_proba - threshold),
        "subgroup_duration": pd.cut(
            dur,
            bins=[-np.inf, 60, 120, np.inf],
            labels=["Corta (<60s)", "Media (60-120s)", "Larga (>120s)"]
        ),
        "subgroup_snr": pd.qcut(
            snr_proxy,
            q=3,
            labels=["Bajo SNR (Ruidoso)", "Medio SNR", "Alto SNR (Limpio)"]
        ),
        "subgroup_turns": pd.cut(
            turns,
            bins=[-np.inf, 4, 12, np.inf],
            labels=["Pocos turnos (<=4)", "Turnos medios (5-12)", "Muchos turnos (>12)"]
        ),
        "subgroup_silence": pd.cut(
            silence,
            bins=[-np.inf, 0.2, 0.45, np.inf],
            labels=["Bajo silencio (<20%)", "Silencio medio (20-45%)", "Alto silencio (>45%)"]
        ),
    }, index=df.index)

    df = pd.concat([df, eval_meta], axis=1)

    # ── Calculo de metricas por subgrupo ──────────────────────────────────────
    analysis = {
        "global": compute_metrics_dict(y_true.values, y_pred, y_proba),
        "by_duration": analyze_by_subgroup(df, "subgroup_duration", y_true, y_pred, y_proba),
        "by_snr": analyze_by_subgroup(df, "subgroup_snr", y_true, y_pred, y_proba),
        "by_turns": analyze_by_subgroup(df, "subgroup_turns", y_true, y_pred, y_proba),
        "by_silence": analyze_by_subgroup(df, "subgroup_silence", y_true, y_pred, y_proba),
    }

    # ── Casos de Error y Alta Incertidumbre ────────────────────────────────────
    false_positives = df[(y_true == 0) & (y_pred == 1)][
        ["anon_id", "label", "pred_proba", "confidence", "split"]
    ].to_dict(orient="records")

    false_negatives = df[(y_true == 1) & (y_pred == 0)][
        ["anon_id", "label", "pred_proba", "confidence", "split"]
    ].to_dict(orient="records")

    # Top 5 casos mas inciertos (cercanos al umbral)
    uncertain_cases = df.sort_values("uncertainty").head(5)[
        ["anon_id", "label", "pred_label", "pred_proba", "uncertainty", "split"]
    ].to_dict(orient="records")

    analysis["errors"] = {
        "false_positives_count": len(false_positives),
        "false_negatives_count": len(false_negatives),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "top_uncertain_cases": uncertain_cases,
    }

    # Guardar reporte
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # ── Imprimir reporte en consola ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("      REPORTE DE ANALISIS DE ERRORES POR SUBGRUPO")
    print("=" * 60)
    g = analysis["global"]
    print(f"\n[GLOBAL] Total: {g['n']} | F1: {g['f1']} | AUC: {g['roc_auc']} | Acc: {g['accuracy']}")
    print(f"FP (Humanos como Sinteticos): {len(false_positives)} | FN (Sinteticos como Humanos): {len(false_negatives)}")

    def print_table(title, data_dict):
        print(f"\n── {title} ──")
        print(f"{'Subgrupo':<25s} {'N':>5s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} {'AUC':>8s}")
        print("-" * 60)
        for cat, m in data_dict.items():
            print(f"{cat:<25s} {m['n']:>5d} {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['roc_auc']:>8.4f}")

    print_table("Por Duracion de Llamada", analysis["by_duration"])
    print_table("Por Calidad / SNR Estimado", analysis["by_snr"])
    print_table("Por Densidad de Turnos", analysis["by_turns"])
    print_table("Por Proporcion de Silencio", analysis["by_silence"])

    if false_positives:
        print("\n[ALERTA] Falsos Positivos detectados:")
        for fp in false_positives:
            print(f"  - {fp['anon_id']}: proba={fp['pred_proba']:.3f} ({fp['split']})")

    if false_negatives:
        print("\n[ALERTA] Falsos Negativos detectados:")
        for fn in false_negatives:
            print(f"  - {fn['anon_id']}: proba={fn['pred_proba']:.3f} ({fn['split']})")

    print(f"\nReporte JSON completo guardado en: {output_path}")
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analisis de errores por subgrupo.")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--model",    default="models/model.joblib")
    parser.add_argument("--out",      default="models/error_analysis.json")
    args = parser.parse_args()

    run_error_analysis(args.features, args.manifest, args.model, args.out)


if __name__ == "__main__":
    main()
