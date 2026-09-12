"""
train_model.py
===============
Entrena un Random Forest sobre las features generadas por features.py,
evalúa contra el split de validación y guarda el modelo (+ imputer +
lista de columnas) en un solo artefacto joblib para que main.py lo
pueda cargar sin depender de reconstruir el pipeline a mano.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from features import NON_FEATURE_COLS

LABEL_POSITIVE = "synthetic"  # clase 1 = voz sintética


def load_features(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df, feature_cols


def build_pipeline():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Entrena el clasificador humano/sintético.")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--model-out", default="models/model.joblib")
    args = parser.parse_args()

    df, feature_cols = load_features(args.features)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise SystemExit(
            "No hay suficientes datos en 'train' o 'val'. Revisa la columna 'split' del manifest."
        )

    X_train = train_df[feature_cols]
    y_train = (train_df["label"] == LABEL_POSITIVE).astype(int)
    X_val = val_df[feature_cols]
    y_val = (val_df["label"] == LABEL_POSITIVE).astype(int)

    print(f"Entrenando con {len(X_train)} muestras ({y_train.sum()} sintéticas, "
          f"{len(y_train) - y_train.sum()} humanas)...")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    print("\n=== Reporte de clasificación (val) ===")
    print(classification_report(y_val, y_pred, target_names=["human", "synthetic"]))

    print("Matriz de confusión (filas=real, cols=predicho) [human, synthetic]:")
    print(confusion_matrix(y_val, y_pred))

    try:
        auc = roc_auc_score(y_val, y_proba)
        print(f"\nROC-AUC (val): {auc:.4f}")
    except ValueError:
        print("\nROC-AUC no calculable (¿solo una clase presente en val?).")

    importances = pipeline.named_steps["clf"].feature_importances_
    top = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:10]
    print("\nTop 10 features más importantes:")
    for name, imp in top:
        print(f"  {name:<28s} {imp:.4f}")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_names": feature_cols}, args.model_out)
    print(f"\nModelo guardado en {args.model_out}")


if __name__ == "__main__":
    main()
