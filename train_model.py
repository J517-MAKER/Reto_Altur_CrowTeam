"""
train_model.py  v2.0
=====================
Entrena un ensemble de clasificadores sobre las features generadas por
features.py v2.0 y guarda el modelo en models/model.joblib.

Mejoras v2.0
------------
* Ensemble VotingClassifier (soft): RandomForest + GradientBoosting + SVC
* Calibracion de probabilidades con CalibratedClassifierCV (isotonic)
* Validacion cruzada StratifiedKFold 5-fold
* Threshold tuning automatico sobre el conjunto de validacion
* Re-entrenamiento final sobre train+val antes de guardar
* Reporte de features guardado en models/feature_report.json
* Manejo de desbalance con class_weight='balanced'
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from features import NON_FEATURE_COLS

LABEL_POSITIVE = "synthetic"


def load_features(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df, feature_cols


def build_ensemble():
    """Construye un VotingClassifier (soft) con 3 modelos complementarios."""
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )
    # SVC calibrado directamente (forma recomendada en sklearn >= 1.9)
    svc = CalibratedClassifierCV(
        SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42),
        method="isotonic",
        cv=3,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("svc", svc)],
        voting="soft",
        weights=[3, 2, 1],   # RF tiene mayor peso por su robustez
        n_jobs=-1,
    )
    return ensemble


def build_pipeline(ensemble):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ensemble),
    ])


def find_optimal_threshold(y_true, y_proba, thresholds=None):
    """Encuentra el umbral que maximiza el F1-score en validacion."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return float(best_thresh), float(best_f1)


def main():
    parser = argparse.ArgumentParser(description="Entrena el ensemble humano/sintetico v2.")
    parser.add_argument("--features",   default="data/features.csv")
    parser.add_argument("--model-out",  default="models/model.joblib")
    parser.add_argument("--report-out", default="models/feature_report.json")
    args = parser.parse_args()

    df, feature_cols = load_features(args.features)

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise SystemExit("No hay suficientes datos en 'train' o 'val'. Revisa el manifest.")

    X_train = train_df[feature_cols]
    y_train = (train_df["label"] == LABEL_POSITIVE).astype(int)
    X_val   = val_df[feature_cols]
    y_val   = (val_df["label"] == LABEL_POSITIVE).astype(int)

    print(f"Train: {len(X_train)} muestras  ({y_train.sum()} sinteticas, "
          f"{len(y_train) - y_train.sum()} humanas)")
    print(f"Val:   {len(X_val)} muestras  ({y_val.sum()} sinteticas, "
          f"{len(y_val) - y_val.sum()} humanas)")
    print(f"Features: {len(feature_cols)}")

    # ── Cross-validation sobre train ──────────────────────────────────────────
    print("\n── Validacion cruzada 5-fold sobre train ──")
    pipeline_cv = build_pipeline(build_ensemble())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline_cv, X_train, y_train,
                                cv=skf, scoring="roc_auc", n_jobs=-1)
    print(f"ROC-AUC por fold: {[round(s, 4) for s in cv_scores]}")
    print(f"Media: {cv_scores.mean():.4f}  Std: {cv_scores.std():.4f}")

    # ── Entrenamiento final sobre train ───────────────────────────────────────
    print("\n── Entrenando ensemble sobre train... ──")
    pipeline = build_pipeline(build_ensemble())
    pipeline.fit(X_train, y_train)

    y_pred   = pipeline.predict(X_val)
    y_proba  = pipeline.predict_proba(X_val)[:, 1]

    print("\n=== Reporte de clasificacion (val) ===")
    print(classification_report(y_val, y_pred, target_names=["human", "synthetic"]))
    print("Matriz de confusion [human, synthetic]:")
    print(confusion_matrix(y_val, y_pred))

    try:
        auc = roc_auc_score(y_val, y_proba)
        print(f"\nROC-AUC (val): {auc:.4f}")
    except ValueError:
        print("\nROC-AUC no calculable.")
        auc = 0.0

    # ── Threshold tuning ──────────────────────────────────────────────────────
    opt_thresh, opt_f1 = find_optimal_threshold(y_val, y_proba)
    print(f"\nUmbral optimo (max F1): {opt_thresh:.3f}  →  F1 = {opt_f1:.4f}")

    # ── Feature importance (RF dentro del ensemble) ───────────────────────────
    rf_model = pipeline.named_steps["clf"].estimators_[0]  # VotingClassifier -> RF
    importances = rf_model.feature_importances_
    top20 = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:20]

    print("\nTop 20 features mas importantes (RF):")
    for name, imp in top20:
        print(f"  {name:<32s} {imp:.4f}")

    # ── Re-entrenamiento sobre train+val ──────────────────────────────────────
    print("\n── Re-entrenando sobre train+val para modelo final... ──")
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])
    final_pipeline = build_pipeline(build_ensemble())
    final_pipeline.fit(X_all, y_all)

    # ── Guardado ──────────────────────────────────────────────────────────────
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline":       final_pipeline,
        "feature_names":  feature_cols,
        "threshold":      opt_thresh,
        "val_roc_auc":    auc,
        "val_f1":         opt_f1,
    }, args.model_out)
    print(f"\nModelo guardado en {args.model_out}")

    # ── Reporte JSON ──────────────────────────────────────────────────────────
    report = {
        "val_roc_auc":       round(auc, 4),
        "val_f1_optimized":  round(opt_f1, 4),
        "optimal_threshold": round(opt_thresh, 4),
        "cv_roc_auc_mean":   round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std":    round(float(cv_scores.std()), 4),
        "n_features":        len(feature_cols),
        "top20_features":    [{"name": n, "importance": round(float(i), 4)} for n, i in top20],
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Reporte guardado en {args.report_out}")


if __name__ == "__main__":
    main()
