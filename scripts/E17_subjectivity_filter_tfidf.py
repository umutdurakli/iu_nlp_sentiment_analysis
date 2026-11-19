
"""
E17 – Subjectivity Filtering + TF-IDF + Logistic Regression

Vergleich mit E09 B:
- Identische Train-/Validation-Splits
- Identische TF-IDF- und Logistic-Regression-Einstellungen
- Neue Vorverarbeitung: deutlich objektive Sätze entfernen
- Titel bleiben unverändert
- Vollständig herausgefilterte Texte bleiben im Original erhalten
- Kein Zugriff auf den Testdatensatz
"""

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
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
from textblob import TextBlob
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data/processed/splits/train.jsonl"
VAL_PATH = ROOT / "data/processed/splits/validation.jsonl"

REPORT_PATH = ROOT / "reports/E17_subjectivity_results.json"
MODEL_PATH = ROOT / "models/E17_subjectivity_tfidf.joblib"

SEED = 42
SUBJECTIVITY_THRESHOLD = 0.20

# Satztrennung an Satzzeichen mit folgendem Leerraum.
# Eine einfache Heuristik, keine vollständige linguistische
# Satzsegmentierung.
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

STAR_LABELS = [1, 2, 3, 4, 5]


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def get_text(value):
    return str(value or "").strip()


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(text)
        if sentence.strip()
    ]


def filter_review(row):
    """
    Filtert objektiv eingestufte Sätze aus dem Rezensionstext.

    Titel bleibt immer erhalten.
    Falls kein Satz übrig bleibt, wird der Originaltext genutzt.

    Gibt den gefilterten Gesamttext und Statistiken zurück.
    """

    title = get_text(row.get("title"))
    body = get_text(row.get("text"))

    sentences = split_sentences(body)

    kept = []
    removed = []

    for sentence in sentences:
        subjectivity = TextBlob(sentence).sentiment.subjectivity

        if subjectivity < SUBJECTIVITY_THRESHOLD:
            removed.append(sentence)
        else:
            kept.append(sentence)

    fallback = False

    if body and not kept:
        kept = sentences
        removed = []
        fallback = True

    filtered_body = " ".join(kept)

    # Wie in E09 B: Titel einmal vor dem Rezensionstext.
    combined_text = f"{title} {filtered_body}".strip()

    stats = {
        "total_sentences": len(sentences),
        "removed_sentences": len(removed),
        "fallback": fallback,
    }

    return combined_text, stats


def prepare_dataset(rows, description):
    texts = []
    labels = []

    total_sentences = 0
    removed_sentences = 0
    fallback_count = 0

    for row in tqdm(rows, desc=description):
        filtered_text, stats = filter_review(row)

        texts.append(filtered_text)
        labels.append(int(row["rating"]))

        total_sentences += stats["total_sentences"]
        removed_sentences += stats["removed_sentences"]
        fallback_count += int(stats["fallback"])

    statistics = {
        "reviews": len(rows),
        "total_sentences": total_sentences,
        "removed_sentences": removed_sentences,
        "removed_percentage": (
            100 * removed_sentences / total_sentences
            if total_sentences else 0.0
        ),
        "fallback_reviews": fallback_count,
    }

    return texts, np.array(labels), statistics


def print_classification_results(y_true, y_pred):
    print("\nKlassenweise Ergebnisse:")

    report = classification_report(
        y_true,
        y_pred,
        labels=STAR_LABELS,
        output_dict=True,
        zero_division=0,
    )

    for star in STAR_LABELS:
        result = report[str(star)]

        print(
            f"{star} Sterne: "
            f"Precision={result['precision']:.3f} | "
            f"Recall={result['recall']:.3f} | "
            f"F1={result['f1-score']:.3f}"
        )

    print("\nConfusion Matrix:")
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=STAR_LABELS,
    )

    for row in matrix:
        print(row.tolist())

    return report, matrix.tolist()


def main():
    start = time.time()

    print("=" * 65)
    print("E17 – SUBJECTIVITY FILTERING + TF-IDF")
    print("=" * 65)

    print("\nLade Trainings- und Validierungsdaten ...")

    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)

    print(f"Training: {len(train_rows):,}")
    print(f"Validierung: {len(val_rows):,}")

    print("\nFiltere objektive Sätze ...")

    train_texts, y_train, train_stats = prepare_dataset(
        train_rows,
        "Training filtern",
    )

    val_texts, y_val, val_stats = prepare_dataset(
        val_rows,
        "Validierung filtern",
    )

    print("\nFilterstatistik:")

    for name, stats in [
        ("Training", train_stats),
        ("Validierung", val_stats),
    ]:
        print(f"\n{name}:")
        print(
            f"  Erkannte Sätze: "
            f"{stats['total_sentences']:,}"
        )
        print(
            f"  Entfernte Sätze: "
            f"{stats['removed_sentences']:,}"
        )
        print(
            f"  Entfernt: "
            f"{stats['removed_percentage']:.2f}%"
        )
        print(
            f"  Fallback-Rezensionen: "
            f"{stats['fallback_reviews']:,}"
        )

    print("\nBerechne Klassengewichte ...")

    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(STAR_LABELS),
        y=y_train,
    )

    class_weights = {
        int(star): float(np.sqrt(weight))
        for star, weight in zip(
            STAR_LABELS,
            balanced_weights,
        )
    }

    print(f"Klassengewichte: {class_weights}")

    print("\nErstelle TF-IDF + Logistic Regression ...")

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

    print("\nStarte Training ...")

    training_start = time.time()
    pipeline.fit(train_texts, y_train)
    training_seconds = time.time() - training_start

    print(
        f"Training abgeschlossen: "
        f"{training_seconds / 60:.1f} Minuten"
    )

    print("\nValidiere Modell ...")

    predictions = pipeline.predict(val_texts)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=STAR_LABELS,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 65)
    print("E17 – ERGEBNISSE")
    print("=" * 65)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    baseline_accuracy = 0.7769
    baseline_macro_f1 = 0.5779

    print("\nVergleich mit E09 B:")
    print(
        f"Accuracy-Differenz: "
        f"{accuracy - baseline_accuracy:+.4f}"
    )
    print(
        f"Macro-F1-Differenz: "
        f"{macro_f1 - baseline_macro_f1:+.4f}"
    )

    class_report, matrix = print_classification_results(
        y_val,
        predictions,
    )

    report = {
        "experiment": "E17",
        "description": (
            "TF-IDF with sentence-level subjectivity filtering"
        ),
        "subjectivity_method": "TextBlob PatternAnalyzer",
        "subjectivity_threshold": SUBJECTIVITY_THRESHOLD,
        "title_preserved": True,
        "fallback_to_original_body": True,
        "train_size": len(train_rows),
        "validation_size": len(val_rows),
        "training_filter_statistics": train_stats,
        "validation_filter_statistics": val_stats,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "baseline_e09_b": {
            "accuracy": baseline_accuracy,
            "macro_f1": baseline_macro_f1,
        },
        "accuracy_difference": float(
            accuracy - baseline_accuracy
        ),
        "macro_f1_difference": float(
            macro_f1 - baseline_macro_f1
        ),
        "classification_report": class_report,
        "confusion_matrix": matrix,
        "training_seconds": training_seconds,
        "total_seconds": time.time() - start,
        "tfidf": {
            "ngram_range": [1, 2],
            "min_df": 3,
            "max_df": 0.95,
            "max_features": 100000,
            "sublinear_tf": True,
        },
        "classifier": {
            "type": "LogisticRegression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": class_weights,
        },
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
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    joblib.dump(pipeline, MODEL_PATH)

    print("\nErgebnisse gespeichert:")
    print(REPORT_PATH)
    print(MODEL_PATH)

    print(
        f"\nGesamtdauer: "
        f"{report['total_seconds'] / 60:.1f} Minuten"
    )

    print("\nE17 abgeschlossen.")


if __name__ == "__main__":
    main()
