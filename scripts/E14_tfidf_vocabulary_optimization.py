
"""
E14 – Letzte TF-IDF-Optimierung.

Untersucht den Einfluss von:
- max_features: 50.000, 100.000, 200.000
- min_df: 2, 3, 5

Konstant:
- Titel + Rezensionstext (Variante B)
- Wort-N-Gramme (1, 2)
- alpha = 0.5
- C = 1.0

Ausschließlich Trainings- und Validierungsdaten.
Keine vorhandenen Modelle oder Berichte überschreiben.
"""

import csv
import json
import warnings
from pathlib import Path

import joblib
import numpy as np

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "processed" / "splits"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

CLASSES = np.array([1, 2, 3, 4, 5])

MAX_FEATURES_VALUES = [50_000, 100_000, 200_000]
MIN_DF_VALUES = [2, 3, 5]

ALPHA = 0.5
C = 1.0

REFERENCE_MODEL = MODELS / "sentiment_title_text_B.joblib"

CSV_PATH = REPORTS / "E14_vocabulary_results.csv"
JSON_PATH = REPORTS / "E14_vocabulary_results.json"
MODEL_PATH = MODELS / "E14_vocabulary_best.joblib"


def load_data(path):
    texts = []
    ratings = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)

            title = str(row.get("title") or "").strip()
            description = str(row.get("text") or "").strip()

            combined = " ".join(
                part
                for part in (title, description)
                if part
            )

            texts.append(combined)
            ratings.append(int(row["rating"]))

    return texts, np.asarray(ratings)


def evaluate(y_true, predictions):
    report = classification_report(
        y_true,
        predictions,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            str(label): {
                "precision": float(
                    report[str(label)]["precision"]
                ),
                "recall": float(
                    report[str(label)]["recall"]
                ),
                "f1": float(
                    report[str(label)]["f1-score"]
                ),
                "support": int(
                    report[str(label)]["support"]
                ),
            }
            for label in CLASSES
        },
    }


def main():

    # Keine vorhandenen Ergebnisse überschreiben.
    for path in (CSV_PATH, JSON_PATH, MODEL_PATH):
        if path.exists():
            raise FileExistsError(
                f"Ausgabedatei existiert bereits: {path}"
            )

    if not REFERENCE_MODEL.is_file():
        raise FileNotFoundError(
            f"Referenzmodell fehlt: {REFERENCE_MODEL}"
        )

    print("=" * 65)
    print("E14 – TF-IDF-VOKABULAROPTIMIERUNG")
    print("=" * 65)

    print("\nLade Trainings- und Validierungsdaten ...")

    train_texts, y_train = load_data(
        SPLITS / "train.jsonl"
    )
    val_texts, y_val = load_data(
        SPLITS / "validation.jsonl"
    )

    print(f"Training: {len(y_train):,}")
    print(f"Validierung: {len(y_val):,}")

    print("\nBewerte Referenzmodell B ...")

    reference_pipeline = joblib.load(REFERENCE_MODEL)
    reference_predictions = reference_pipeline.predict(
        val_texts
    )
    reference = evaluate(y_val, reference_predictions)

    print(f"Accuracy: {reference['accuracy']:.4f}")
    print(f"Macro-F1: {reference['macro_f1']:.4f}")

    base_weights = compute_class_weight(
        class_weight="balanced",
        classes=CLASSES,
        y=y_train,
    )

    class_weights = {
        int(label): float(weight ** ALPHA)
        for label, weight in zip(CLASSES, base_weights)
    }

    results = []
    best = None
    best_pipeline = None
    best_predictions = None
    convergence_warnings = []

    experiment_number = 0
    total = len(MAX_FEATURES_VALUES) * len(MIN_DF_VALUES)

    for max_features in MAX_FEATURES_VALUES:

        for min_df in MIN_DF_VALUES:

            experiment_number += 1

            print("\n" + "-" * 65)
            print(
                f"Experiment {experiment_number}/{total} | "
                f"max_features={max_features:,} | "
                f"min_df={min_df}"
            )

            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=min_df,
                max_df=0.95,
                max_features=max_features,
                sublinear_tf=True,
                dtype=np.float32,
            )

            X_train = vectorizer.fit_transform(train_texts)
            X_val = vectorizer.transform(val_texts)

            actual_features = X_train.shape[1]

            print(
                f"Trainingsmatrix: {X_train.shape}"
            )

            classifier = LogisticRegression(
                C=C,
                class_weight=class_weights,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter(
                    "always", ConvergenceWarning
                )
                classifier.fit(X_train, y_train)

            messages = [
                str(item.message)
                for item in caught
                if issubclass(item.category, ConvergenceWarning)
            ]

            predictions = classifier.predict(X_val)
            metrics = evaluate(y_val, predictions)

            result = {
                "max_features": max_features,
                "min_df": min_df,
                "actual_features": actual_features,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "delta_macro_f1": (
                    metrics["macro_f1"]
                    - reference["macro_f1"]
                ),
                "convergence_warning": bool(messages),
                "per_class": metrics["per_class"],
            }

            results.append(result)

            print(
                f"Accuracy: {result['accuracy']:.4f}"
            )
            print(
                f"Macro-F1: {result['macro_f1']:.4f}"
            )
            print(
                f"Delta zu B: "
                f"{result['delta_macro_f1']:+.4f}"
            )

            if messages:
                convergence_warnings.append({
                    "max_features": max_features,
                    "min_df": min_df,
                    "messages": messages,
                })
                print("WARNUNG: Konvergenzproblem")
                continue

            if (
                best is None
                or (
                    result["macro_f1"],
                    result["accuracy"],
                ) > (
                    best["macro_f1"],
                    best["accuracy"],
                )
            ):
                best = result
                best_pipeline = Pipeline([
                    ("tfidf", vectorizer),
                    ("classifier", classifier),
                ])
                best_predictions = predictions.copy()

    if best is None:
        raise RuntimeError(
            "Kein Modell ohne Konvergenzwarnung gefunden."
        )

    results.sort(
        key=lambda row: (
            row["macro_f1"],
            row["accuracy"],
        ),
        reverse=True,
    )

    matrix = confusion_matrix(
        y_val,
        best_predictions,
        labels=CLASSES,
    )

    print("\n" + "=" * 65)
    print("ERGEBNISVERGLEICH")
    print("=" * 65)

    for row in results:
        print(
            f"max_features={row['max_features']:>7,} | "
            f"min_df={row['min_df']} | "
            f"Features={row['actual_features']:>7,} | "
            f"Accuracy={row['accuracy']:.4f} | "
            f"Macro-F1={row['macro_f1']:.4f} | "
            f"Delta={row['delta_macro_f1']:+.4f}"
        )

    print("\nBESTES E14-MODELL")
    print(f"max_features: {best['max_features']:,}")
    print(f"min_df: {best['min_df']}")
    print(f"Actual features: {best['actual_features']:,}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Macro-F1: {best['macro_f1']:.4f}")

    print("\nKlassifikationsbericht:")
    print(
        classification_report(
            y_val,
            best_predictions,
            labels=CLASSES,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(matrix)

    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "max_features",
                "min_df",
                "actual_features",
                "accuracy",
                "macro_f1",
                "delta_macro_f1",
                "convergence_warning",
                "f1_1",
                "f1_2",
                "f1_3",
                "f1_4",
                "f1_5",
            ],
        )
        writer.writeheader()

        for row in results:
            writer.writerow({
                "max_features": row["max_features"],
                "min_df": row["min_df"],
                "actual_features": row["actual_features"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "delta_macro_f1": row["delta_macro_f1"],
                "convergence_warning": row[
                    "convergence_warning"
                ],
                **{
                    f"f1_{label}": row["per_class"][str(label)]["f1"]
                    for label in CLASSES
                },
            })

    report = {
        "experiment": "E14",
        "method": "tfidf_vocabulary_optimization",
        "reference_B": reference,
        "fixed_parameters": {
            "alpha": ALPHA,
            "C": C,
            "ngram_range": [1, 2],
            "max_df": 0.95,
            "sublinear_tf": True,
        },
        "max_features_values": MAX_FEATURES_VALUES,
        "min_df_values": MIN_DF_VALUES,
        "all_results": results,
        "best_result": best,
        "best_confusion_matrix": matrix.tolist(),
        "convergence_warnings": convergence_warnings,
        "limitations": [
            "Hyperparameter selection on validation data.",
            "Vocabulary size and min_df were optimized together.",
            "Test data were not used.",
        ],
    }

    with JSON_PATH.open("x", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    joblib.dump(best_pipeline, MODEL_PATH)

    print("\nGespeicherte Dateien:")
    print(CSV_PATH)
    print(JSON_PATH)
    print(MODEL_PATH)

    print("\nE14 abgeschlossen.")
    print("Testdaten wurden nicht verwendet.")


if __name__ == "__main__":
    main()
