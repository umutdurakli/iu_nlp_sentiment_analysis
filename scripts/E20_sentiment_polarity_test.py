
"""
E20 – Abschließende Testevaluation der Sentimentpolarität

Vergleich auf identischem Testdatensatz:
- TF-IDF + Logistic Regression (E18)
- DistilBERT (E16/E19), auf 3 Klassen abgebildet

Kein erneutes Training.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = ROOT / "data/processed/splits/test.jsonl"

MODEL_PATH = ROOT / "models/E18_sentiment_polarity.joblib"
E19_REPORT_PATH = ROOT / "reports/E19_distilbert_test_results.json"

REPORT_PATH = ROOT / "reports/E20_sentiment_polarity_test_results.json"

CLASS_NAMES = ["Negativ", "Neutral", "Positiv"]
CLASSES = [0, 1, 2]


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
    return f"{title} {body}".strip()


def to_polarity(rating):
    rating = int(rating)

    if rating in (1, 2):
        return 0

    if rating == 3:
        return 1

    if rating in (4, 5):
        return 2

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


def aggregate_distilbert_matrix(matrix_5):
    """
    Fasst die 5x5-Confusion-Matrix von E19
    zu einer 3x3-Polaritätsmatrix zusammen.

    Reihenfolge:
    0 = Negativ (1, 2 Sterne)
    1 = Neutral (3 Sterne)
    2 = Positiv (4, 5 Sterne)
    """

    matrix_5 = np.asarray(matrix_5, dtype=np.int64)

    if matrix_5.shape != (5, 5):
        raise ValueError("E19-Matrix muss 5x5 sein.")

    groups = [[0, 1], [2], [3, 4]]

    matrix_3 = np.zeros((3, 3), dtype=np.int64)

    for i, true_group in enumerate(groups):
        for j, predicted_group in enumerate(groups):
            matrix_3[i, j] = matrix_5[
                np.ix_(true_group, predicted_group)
            ].sum()

    return matrix_3


def evaluate_matrix(matrix):
    """
    Berechnet Accuracy, Precision, Recall und F1
    direkt aus einer Confusion Matrix.
    """

    matrix = np.asarray(matrix, dtype=np.int64)

    tp = np.diag(matrix).astype(float)

    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)

    precision = np.divide(
        tp,
        predicted,
        out=np.zeros(3),
        where=predicted != 0,
    )

    recall = np.divide(
        tp,
        support,
        out=np.zeros(3),
        where=support != 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(3),
        where=(precision + recall) != 0,
    )

    return {
        "accuracy": float(
            tp.sum() / matrix.sum()
        ),
        "macro_f1": float(f1.mean()),
        "classification_report": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1-score": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
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
    print("E20 – ABSCHLIESSENDE SENTIMENTPOLARITÄT")
    print("=" * 65)

    if REPORT_PATH.exists():
        raise FileExistsError(
            f"E20-Bericht existiert bereits: {REPORT_PATH}"
        )

    print("\nLade Testdatensatz ...")

    rows = load_jsonl(TEST_PATH)

    texts = [prepare_text(row) for row in rows]

    y_test = np.array([
        to_polarity(row["rating"])
        for row in rows
    ])

    print(f"Testrezensionen: {len(rows):,}")

    # -------------------------------------------------
    # TF-IDF E18
    # -------------------------------------------------

    print("\nLade E18-Modell ...")

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(texts)

    tfidf_results = evaluate(
        y_test,
        predictions,
    )

    print_results(
        "TF-IDF E18 – Test",
        tfidf_results,
    )

    # -------------------------------------------------
    # DistilBERT E19
    # -------------------------------------------------

    print("\nLade E19-Ergebnisse ...")

    e19 = json.loads(
        E19_REPORT_PATH.read_text(encoding="utf-8")
    )

    if len(rows) != e19["test_size"]:
        raise ValueError(
            "E19 und E20 verwenden unterschiedliche "
            "Testdatensatzgrößen."
        )

    matrix_3 = aggregate_distilbert_matrix(
        e19["confusion_matrix"]
    )

    distilbert_results = evaluate_matrix(matrix_3)

    if not np.array_equal(
        matrix_3.sum(axis=1),
        np.bincount(y_test, minlength=3),
    ):
        raise ValueError(
            "Die Klassenverteilung von E19 stimmt "
            "nicht mit dem E20-Testdatensatz überein."
        )

    print_results(
        "DistilBERT E19 – Polarität",
        distilbert_results,
    )

    # -------------------------------------------------
    # Vergleich
    # -------------------------------------------------

    accuracy_difference = (
        distilbert_results["accuracy"]
        - tfidf_results["accuracy"]
    )

    macro_f1_difference = (
        distilbert_results["macro_f1"]
        - tfidf_results["macro_f1"]
    )

    print("\n" + "=" * 65)
    print("FINALER MODELLVERGLEICH")
    print("=" * 65)

    print(
        f"TF-IDF Accuracy:     "
        f"{tfidf_results['accuracy']:.4f}"
    )
    print(
        f"DistilBERT Accuracy: "
        f"{distilbert_results['accuracy']:.4f}"
    )

    print(
        f"\nTF-IDF Macro-F1:     "
        f"{tfidf_results['macro_f1']:.4f}"
    )
    print(
        f"DistilBERT Macro-F1: "
        f"{distilbert_results['macro_f1']:.4f}"
    )

    print(
        f"\nAccuracy-Differenz: {accuracy_difference:+.4f}"
    )
    print(
        f"Macro-F1-Differenz: {macro_f1_difference:+.4f}"
    )

    results = {
        "experiment": "E20",
        "description": (
            "Final polarity comparison on held-out test data"
        ),
        "test_size": len(rows),
        "label_mapping": {
            "negative": [1, 2],
            "neutral": [3],
            "positive": [4, 5],
        },
        "tfidf_e18": tfidf_results,
        "distilbert_e19_mapped": distilbert_results,
        "accuracy_difference": float(
            accuracy_difference
        ),
        "macro_f1_difference": float(
            macro_f1_difference
        ),
        "total_seconds": time.time() - start,
    }

    REPORT_PATH.parent.mkdir(
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

    print(f"\nBericht gespeichert: {REPORT_PATH}")
    print("\nE20 abgeschlossen.")


if __name__ == "__main__":
    main()
