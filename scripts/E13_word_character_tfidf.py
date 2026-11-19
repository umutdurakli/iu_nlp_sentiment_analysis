
"""
E13 – Wort-TF-IDF + Zeichen-N-Gramme.

Vergleicht das bisherige Titel-Text-Modell B mit
vier Kombinationen aus Wort- und Zeichenmerkmalen.

Modellauswahl ausschließlich auf Validierungsdaten.
Testdaten werden nicht verwendet.
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

CHAR_WEIGHTS = [0.25, 0.5, 1.0, 2.0]

ALPHA = 0.5
C = 1.0

REFERENCE_MODEL = MODELS / "sentiment_title_text_B.joblib"

CSV_PATH = REPORTS / "E13_word_char_results.csv"
JSON_PATH = REPORTS / "E13_word_char_results.json"
MODEL_PATH = MODELS / "E13_word_char_best.joblib"


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


def combine_features(word_matrix, char_matrix, char_weight):
    return hstack(
        [word_matrix, char_matrix * char_weight],
        format="csr",
        dtype=np.float32,
    )


def main():

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
    print("E13 – WORT- UND ZEICHEN-TF-IDF")
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

    print("\nBewerte bestehendes Referenzmodell B ...")

    reference_pipeline = joblib.load(REFERENCE_MODEL)
    reference_predictions = reference_pipeline.predict(
        val_texts
    )
    reference = evaluate(y_val, reference_predictions)

    print(f"Accuracy: {reference['accuracy']:.4f}")
    print(f"Macro-F1: {reference['macro_f1']:.4f}")

    print("\nBerechne Wort-TF-IDF ...")

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    X_train_word = word_vectorizer.fit_transform(
        train_texts
    )
    X_val_word = word_vectorizer.transform(
        val_texts
    )

    print(f"Wortmatrix: {X_train_word.shape}")

    print("\nBerechne Zeichen-TF-IDF ...")

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        max_features=50_000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    X_train_char = char_vectorizer.fit_transform(
        train_texts
    )
    X_val_char = char_vectorizer.transform(
        val_texts
    )

    print(f"Zeichenmatrix: {X_train_char.shape}")
    print(
        "Gesamte Merkmale:",
        X_train_word.shape[1] + X_train_char.shape[1],
    )

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
    best_classifier = None
    best_predictions = None
    convergence_warnings = []

    for char_weight in CHAR_WEIGHTS:

        print("\n" + "-" * 65)
        print(f"Zeichengewichtung: {char_weight}")

        X_train = combine_features(
            X_train_word,
            X_train_char,
            char_weight,
        )
        X_val = combine_features(
            X_val_word,
            X_val_char,
            char_weight,
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
            "char_weight": char_weight,
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
                "char_weight": char_weight,
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
            best_classifier = classifier
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
            f"Gewicht={row['char_weight']:.2f} | "
            f"Accuracy={row['accuracy']:.4f} | "
            f"Macro-F1={row['macro_f1']:.4f} | "
            f"Delta={row['delta_macro_f1']:+.4f}"
        )

    print("\nBESTES E13-MODELL")
    print(f"Zeichengewicht: {best['char_weight']}")
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
                "char_weight",
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
                "char_weight": row["char_weight"],
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
        "experiment": "E13",
        "method": "word_and_character_tfidf",
        "reference_B": reference,
        "fixed_parameters": {
            "alpha": ALPHA,
            "C": C,
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 5],
            "word_max_features": 100_000,
            "char_max_features": 50_000,
        },
        "char_weights": CHAR_WEIGHTS,
        "feature_dimensions": {
            "word": X_train_word.shape[1],
            "character": X_train_char.shape[1],
            "total": (
                X_train_word.shape[1] + X_train_char.shape[1]
            ),
        },
        "all_results": results,
        "best_result": best,
        "best_confusion_matrix": matrix.tolist(),
        "convergence_warnings": convergence_warnings,
        "limitations": [
            "Model selection on validation data.",
            "Additional character features increase dimensionality.",
            "Test data not used.",
        ],
    }

    with JSON_PATH.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    artifact = {
        "experiment": "E13",
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "char_weight": best["char_weight"],
        "classifier": best_classifier,
        "classes": CLASSES.tolist(),
        "input_format": "title + description",
    }

    joblib.dump(artifact, MODEL_PATH)

    print("\nGespeicherte Dateien:")
    print(CSV_PATH)
    print(JSON_PATH)
    print(MODEL_PATH)

    print("\nE13 abgeschlossen.")
    print("Testdaten wurden nicht verwendet.")


if __name__ == "__main__":
    main()
