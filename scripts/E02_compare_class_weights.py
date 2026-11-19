
"""
Experiment 2: Einfluss der Klassengewichtung.

Vergleich:
    A: Bisheriges Modell ohne Klassengewichtung
    B: Neues Modell mit class_weight="balanced"

Die Testdaten bleiben unberührt.
"""

import json
from pathlib import Path

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Projektpfade
# --------------------------------------------------

PROJEKTORDNER = Path(__file__).resolve().parent.parent

DATENORDNER = PROJEKTORDNER / "data" / "processed" / "splits"
MODELLORDNER = PROJEKTORDNER / "models"
BERICHTORDNER = PROJEKTORDNER / "reports"

TRAIN_DATEI = DATENORDNER / "train.jsonl"
VALIDIERUNG_DATEI = DATENORDNER / "validation.jsonl"

KLASSEN = [1, 2, 3, 4, 5]


# --------------------------------------------------
# 2. Daten laden
# --------------------------------------------------

def lade_rezensionen(dateipfad):

    texte = []
    sterne = []

    with dateipfad.open("r", encoding="utf-8") as datei:

        for zeile in datei:
            rezension = json.loads(zeile)

            texte.append(rezension["text"])
            sterne.append(int(rezension["rating"]))

    return texte, sterne


# --------------------------------------------------
# 3. Ergebnisse berechnen
# --------------------------------------------------

def evaluiere_modell(name, modell, X_val, y_val):

    vorhersagen = modell.predict(X_val)

    accuracy = accuracy_score(y_val, vorhersagen)

    macro_f1 = f1_score(
        y_val,
        vorhersagen,
        labels=KLASSEN,
        average="macro",
        zero_division=0,
    )

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nKlassifikationsbericht:")
    print(
        classification_report(
            y_val,
            vorhersagen,
            labels=KLASSEN,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_val,
            vorhersagen,
            labels=KLASSEN,
        )
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": classification_report(
            y_val,
            vorhersagen,
            labels=KLASSEN,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            y_val,
            vorhersagen,
            labels=KLASSEN,
        ).tolist(),
    }


# --------------------------------------------------
# 4. Hauptprogramm
# --------------------------------------------------

def main():

    print("Lade Trainings- und Validierungsdaten ...")

    X_train, y_train = lade_rezensionen(TRAIN_DATEI)
    X_val, y_val = lade_rezensionen(VALIDIERUNG_DATEI)

    # Bereits trainiertes Baseline-Modell laden.
    # Wir müssen es nicht erneut trainieren.

    print("\nLade ursprüngliches Modell ...")

    baseline = joblib.load(
        MODELLORDNER / "sentiment_baseline.joblib"
    )

    # Neues Modell: dieselben Einstellungen wie zuvor,
    # aber mit einer Klassengewichtung.

    print("\nErstelle Modell mit Klassengewichtung ...")

    balanced = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    max_features=100_000,
                    sublinear_tf=True,
                    dtype=__import__("numpy").float32,
                ),
            ),
            (
                "klassifikator",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTrainiere Modell mit Klassengewichtung ...")

    balanced.fit(X_train, y_train)

    print("Training abgeschlossen.")

    # Beide Modelle mit identischen Daten evaluieren.

    ergebnis_baseline = evaluiere_modell(
        "Ohne Klassengewichtung",
        baseline,
        X_val,
        y_val,
    )

    ergebnis_balanced = evaluiere_modell(
        "Mit Klassengewichtung",
        balanced,
        X_val,
        y_val,
    )

    # Direkter Vergleich.

    print("\n--- Vergleich ---")

    delta_accuracy = (
        ergebnis_balanced["accuracy"]
        - ergebnis_baseline["accuracy"]
    )

    delta_f1 = (
        ergebnis_balanced["macro_f1"]
        - ergebnis_baseline["macro_f1"]
    )

    print(f"Accuracy-Veränderung: {delta_accuracy:+.4f}")
    print(f"Macro-F1-Veränderung: {delta_f1:+.4f}")

    # Ergebnisse für den Projektbericht speichern.

    BERICHTORDNER.mkdir(parents=True, exist_ok=True)
    MODELLORDNER.mkdir(parents=True, exist_ok=True)

    bericht = {
        "experiment": "class_weight_balanced",
        "hypothese": (
            "Klassengewichtung verbessert die Erkennung "
            "seltener Sterneklassen."
        ),
        "trainingsdaten": len(X_train),
        "validierungsdaten": len(X_val),
        "baseline": ergebnis_baseline,
        "balanced": ergebnis_balanced,
        "delta_accuracy": delta_accuracy,
        "delta_macro_f1": delta_f1,
    }

    berichtpfad = (
        BERICHTORDNER / "experiment_02_class_weights.json"
    )

    with berichtpfad.open("w", encoding="utf-8") as datei:
        json.dump(bericht, datei, indent=2, ensure_ascii=False)

    # Neues Modell getrennt speichern.
    # Das ursprüngliche Modell bleibt erhalten.

    modellpfad = (
        MODELLORDNER / "sentiment_balanced.joblib"
    )

    joblib.dump(balanced, modellpfad)

    print(f"\nErgebnisse gespeichert: {berichtpfad}")
    print(f"Modell gespeichert: {modellpfad}")
    print("\nExperiment abgeschlossen.")


if __name__ == "__main__":
    main()
