
"""
E18 – Sentimentpolarität (3 Klassen)

Vergleich:
A) E09 B: bestehendes 5-Sterne-Modell, Vorhersagen auf
   negativ / neutral / positiv abbilden
B) E18: TF-IDF + Logistic Regression direkt auf drei
   Polaritätsklassen trainieren

Daten: Train + Validation, kein Testset
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data/processed/splits/train.jsonl"
VAL_PATH = ROOT / "data/processed/splits/validation.jsonl"

BASELINE_MODEL_PATH = (
    ROOT / "models/sentiment_title_text_B.joblib"
)

MODEL_PATH = ROOT / "models/E18_sentiment_polarity.joblib"
REPORT_PATH = ROOT / "reports/E18_sentiment_polarity_results.json"

SEED = 42
CLASSES = [0, 1, 2]
CLASS_NAMES = ["Negativ", "Neutral", "Positiv"]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def prepare_text(row):
    title = str(row.get("title") or "").strip()
    body = str(row.get("text") or "").strip()

    # Titel wird genau einmal verwendet (E09 B).
    return f"{title} {body}".strip()


def to_polarity(rating):
    rating = int(rating)

    if rating in (1, 2):
        return 0  # Negativ
    if rating == 3:
        return 1  # Neutral
    if rating in (4, 5):
        return 2  # Positiv

    raise ValueError(f"Ungültige Sternebewertung: {rating}")


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=CLASSES,
        ).tolist(),
    }


def print_results(name, metrics):
    print(f"\n=== {name} ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    print("\nKlassenweise Ergebnisse:")

    for class_name in CLASS_NAMES:
        result = metrics["classification_report"][class_name]

        print(
            f"{class_name:8s}: "
            f"Precision={result['precision']:.3f} | "
            f"Recall={result['recall']:.3f} | "
            f"F1={result['f1-score']:.3f}"
        )

    print("\nConfusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)


def main():
    start = time.time()

    print("=" * 65)
    print("E18 – SENTIMENTPOLARITÄT")
    print("=" * 65)

    if REPORT_PATH.exists() or MODEL_PATH.exists():
        raise FileExistsError(
            "E18-Ergebnisse existieren bereits. "
            "Bitte vor erneutem Start prüfen."
        )

    print("\nLade Datensätze ...")

    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)

    X_train = [prepare_text(row) for row in train_rows]
    X_val = [prepare_text(row) for row in val_rows]

    y_train = np.array(
        [to_polarity(row["rating"]) for row in train_rows]
    )
    y_val = np.array(
        [to_polarity(row["rating"]) for row in val_rows]
    )

    print(f"Training: {len(train_rows):,}")
    print(f"Validierung: {len(val_rows):,}")

    print("\nTrainingsverteilung:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        count = int((y_train == class_id).sum())
        share = 100 * count / len(y_train)

        print(
            f"{class_name}: {count:,} ({share:.2f}%)"
        )

    # =====================================================
    # A: E09 B auf drei Polaritätsklassen abbilden
    # =====================================================

    baseline_metrics = None

    if BASELINE_MODEL_PATH.exists():
        print("\nLade bestehendes E09-B-Modell ...")

        baseline_model = joblib.load(BASELINE_MODEL_PATH)

        original_predictions = baseline_model.predict(X_val)

        mapped_predictions = np.array([
            to_polarity(prediction)
            for prediction in original_predictions
        ])

        baseline_metrics = evaluate(
            y_val,
            mapped_predictions,
        )

        print_results(
            "E09 B – auf Polarität abgebildet",
            baseline_metrics,
        )

    else:
        print(
            "\nHinweis: E09-B-Modell nicht gefunden. "
            "Der Vergleich wird übersprungen."
        )

    # =====================================================
    # B: Neues 3-Klassen-Modell trainieren
    # =====================================================

    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(CLASSES),
        y=y_train,
    )

    class_weights = {
        int(class_id): float(np.sqrt(weight))
        for class_id, weight in zip(
            CLASSES,
            balanced_weights,
        )
    }

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.95,
                max_features=100000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                class_weight=class_weights,
                random_state=SEED,
            ),
        ),
    ])

    print("\nTrainiere neues 3-Klassen-Modell ...")

    training_start = time.time()
    pipeline.fit(X_train, y_train)
    training_seconds = time.time() - training_start

    predictions = pipeline.predict(X_val)

    polarity_metrics = evaluate(y_val, predictions)

    print_results(
        "E18 – direkt trainierte Polaritätsklassifikation",
        polarity_metrics,
    )

    print("\n" + "=" * 65)
    print("E18 – VERGLEICH")
    print("=" * 65)

    if baseline_metrics is not None:
        print(
            "Accuracy-Differenz E18 vs. E09 B: "
            f"{polarity_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f}"
        )
        print(
            "Macro-F1-Differenz E18 vs. E09 B: "
            f"{polarity_metrics['macro_f1'] - baseline_metrics['macro_f1']:+.4f}"
        )

    print(
        "\nHinweis: Die Accuracy der 3-Klassen-Aufgabe "
        "ist nicht direkt mit der 5-Klassen-Accuracy "
        "vergleichbar."
    )

    results = {
        "experiment": "E18",
        "task": "Three-class sentiment polarity",
        "label_mapping": {
            "negative": [1, 2],
            "neutral": [3],
            "positive": [4, 5],
        },
        "train_size": len(train_rows),
        "validation_size": len(val_rows),
        "training_seconds": training_seconds,
        "total_seconds": time.time() - start,
        "class_weights": class_weights,
        "mapped_e09_b": baseline_metrics,
        "trained_e18": polarity_metrics,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    joblib.dump(pipeline, MODEL_PATH)

    print("\nGespeichert:")
    print(REPORT_PATH)
    print(MODEL_PATH)

    print("\nE18 abgeschlossen.")


if __name__ == "__main__":
    main()
