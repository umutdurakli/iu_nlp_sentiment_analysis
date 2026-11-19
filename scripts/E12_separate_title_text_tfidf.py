
"""
E12 – Separate TF-IDF-Vektorisierung von Titel und Rezensionstext.

Vergleich:
- E09/E11 Referenzmodell B (gemeinsamer Text)
- E12 mit Titelgewichtungen 0.5, 1.0 und 2.0

Modellauswahl: Macro-F1 auf dem Validierungsdatensatz.
Testdaten bleiben unberührt.
"""

import csv
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "processed" / "splits"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

CLASSES = np.array([1, 2, 3, 4, 5])
TITLE_WEIGHTS = [0.5, 1.0, 2.0]

ALPHA = 0.5
C = 1.0

REFERENCE_MODEL = MODELS / "sentiment_title_text_B.joblib"

CSV_PATH = REPORTS / "E12_separate_tfidf_results.csv"
JSON_PATH = REPORTS / "E12_separate_tfidf_results.json"
MODEL_PATH = MODELS / "E12_separate_tfidf_best.joblib"


def load_data(path):
    titles = []
    texts = []
    ratings = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)

            titles.append(str(row.get("title") or "").strip())
            texts.append(str(row.get("text") or "").strip())
            ratings.append(int(row["rating"]))

    return titles, texts, np.asarray(ratings)


def make_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def combine_features(title_matrix, text_matrix, weight):
    # Kein gemeinsames Normalisieren nach der Kombination:
    # Die relative Titelgewichtung soll erhalten bleiben.
    return hstack(
        [title_matrix * weight, text_matrix],
        format="csr",
        dtype=np.float32,
    )


def evaluate(y_true, predictions):
    report = classification_report(
        y_true,
        predictions,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
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
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
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
    print("E12 – SEPARATE TF-IDF-VEKTORISIERUNG")
    print("=" * 65)

    print("\nLade Daten ...")

    train_titles, train_texts, y_train = load_data(
        SPLITS / "train.jsonl"
    )
    val_titles, val_texts, y_val = load_data(
        SPLITS / "validation.jsonl"
    )

    print(f"Training: {len(y_train):,}")
    print(f"Validierung: {len(y_val):,}")

    # Referenzmodell: gleiche Titel-Text-Zusammensetzung
    # wie im vorherigen Experiment E11.
    val_combined = [
        " ".join(part for part in (title, text) if part)
        for title, text in zip(val_titles, val_texts)
    ]

    print("\nBewerte bestehendes B-Modell ...")

    reference_pipeline = joblib.load(REFERENCE_MODEL)
    reference_predictions = reference_pipeline.predict(
        val_combined
    )
    reference = evaluate(y_val, reference_predictions)

    print(f"Accuracy: {reference['accuracy']:.4f}")
    print(f"Macro-F1: {reference['macro_f1']:.4f}")

    # Beide Vektorisierer ausschließlich auf Training fitten.
    print("\nVektorisiere Titel separat ...")

    title_vectorizer = make_vectorizer()
    X_train_title = title_vectorizer.fit_transform(
        train_titles
    )
    X_val_title = title_vectorizer.transform(val_titles)

    print(f"Titelmatrix: {X_train_title.shape}")

    print("\nVektorisiere Beschreibung separat ...")

    text_vectorizer = make_vectorizer()
    X_train_text = text_vectorizer.fit_transform(
        train_texts
    )
    X_val_text = text_vectorizer.transform(val_texts)

    print(f"Textmatrix: {X_train_text.shape}")

    print(
        "\nGesamte Feature-Anzahl:",
        X_train_title.shape[1] + X_train_text.shape[1],
    )

    # Gleiche Klassengewichtung wie E11.
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
    best_model = None
    best_predictions = None
    convergence_warnings = []

    for title_weight in TITLE_WEIGHTS:
        print("\n" + "-" * 65)
        print(f"Titelgewichtung: {title_weight}")

        X_train = combine_features(
            X_train_title, X_train_text, title_weight
        )
        X_val = combine_features(
            X_val_title, X_val_text, title_weight
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
            "title_weight": title_weight,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "delta_macro_f1": (
                metrics["macro_f1"] - reference["macro_f1"]
            ),
            "convergence_warning": bool(messages),
            "per_class": metrics["per_class"],
        }

        results.append(result)

        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Macro-F1: {result['macro_f1']:.4f}")
        print(
            f"Delta zu B: {result['delta_macro_f1']:+.4f}"
        )

        if messages:
            convergence_warnings.append({
                "title_weight": title_weight,
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
            best_model = classifier
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
        y_val, best_predictions, labels=CLASSES
    )

    print("\n" + "=" * 65)
    print("ERGEBNISSE")
    print("=" * 65)

    for row in results:
        print(
            f"Gewicht={row['title_weight']:.1f} | "
            f"Accuracy={row['accuracy']:.4f} | "
            f"Macro-F1={row['macro_f1']:.4f} | "
            f"Delta={row['delta_macro_f1']:+.4f}"
        )

    print("\nBESTES E12-MODELL")
    print(f"Titelgewicht: {best['title_weight']}")
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

    # Ergebnisse werden auch gespeichert, falls E12
    # schlechter als das bestehende B-Modell ist.
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "title_weight",
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
                "title_weight": row["title_weight"],
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
        "experiment": "E12",
        "method": "separate_title_text_tfidf",
        "reference_B": reference,
        "fixed_parameters": {
            "alpha": ALPHA,
            "C": C,
            "ngram_range": [1, 2],
            "max_features_per_vectorizer": 100_000,
        },
        "title_weights": TITLE_WEIGHTS,
        "feature_dimensions": {
            "title": X_train_title.shape[1],
            "text": X_train_text.shape[1],
            "total": (
                X_train_title.shape[1] + X_train_text.shape[1]
            ),
        },
        "all_results": results,
        "best_result": best,
        "best_confusion_matrix": matrix.tolist(),
        "convergence_warnings": convergence_warnings,
        "limitations": [
            "Hyperparameter selection on validation data.",
            "Separate vectorizers may increase feature count.",
            "No independent test evaluation performed.",
        ],
    }

    with JSON_PATH.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    # Das Modell benötigt bei späteren Vorhersagen
    # zwei Eingabefelder und beide Vektorisierer.
    # Daher als vollständiges Artefakt speichern.
    artifact = {
        "experiment": "E12",
        "title_vectorizer": title_vectorizer,
        "text_vectorizer": text_vectorizer,
        "title_weight": best["title_weight"],
        "classifier": best_model,
        "classes": CLASSES.tolist(),
        "input_fields": ["title", "text"],
    }

    joblib.dump(artifact, MODEL_PATH)

    print("\nGespeichert:")
    print(CSV_PATH)
    print(JSON_PATH)
    print(MODEL_PATH)

    print("\nE12 abgeschlossen.")
    print("Testdaten wurden nicht verwendet.")


if __name__ == "__main__":
    main()
