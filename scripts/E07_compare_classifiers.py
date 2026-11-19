
"""
Experiment 6: Vergleich zweier Klassifikationsalgorithmen.

Logistische Regression gegen LinearSVC.

Alle Modelle verwenden:
- dieselben Trainings- und Validierungsdaten
- identische TF-IDF-Merkmale
- identische Klassengewichte

Die Testdaten werden nicht verwendet.
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight


# --------------------------------------------------
# 1. Einstellungen
# --------------------------------------------------

PROJEKTORDNER = Path(__file__).resolve().parent.parent

DATENORDNER = PROJEKTORDNER / "data" / "processed" / "splits"
MODELLORDNER = PROJEKTORDNER / "models"
BERICHTORDNER = PROJEKTORDNER / "reports"

KLASSEN = np.array([1, 2, 3, 4, 5])
GEWICHTUNGSSTAERKE = 0.5


# --------------------------------------------------
# 2. Daten laden
# --------------------------------------------------

def lade_daten(dateipfad):

    texte = []
    sterne = []

    with dateipfad.open("r", encoding="utf-8") as datei:
        for zeile in datei:
            rezension = json.loads(zeile)
            texte.append(rezension["text"])
            sterne.append(int(rezension["rating"]))

    return texte, np.array(sterne)


# --------------------------------------------------
# 3. Hauptprogramm
# --------------------------------------------------

def main():

    print("Lade Daten ...")

    X_train, y_train = lade_daten(
        DATENORDNER / "train.jsonl"
    )

    X_val, y_val = lade_daten(
        DATENORDNER / "validation.jsonl"
    )

    print(f"Training: {len(X_train):,}")
    print(f"Validierung: {len(X_val):,}")

    # Klassengewichte wie in Experiment 3 berechnen.

    automatische_gewichte = compute_class_weight(
        class_weight="balanced",
        classes=KLASSEN,
        y=y_train,
    )

    gewichte = automatische_gewichte ** GEWICHTUNGSSTAERKE

    klassengewichte = {
        int(klasse): float(gewicht)
        for klasse, gewicht in zip(KLASSEN, gewichte)
    }

    # TF-IDF einmal trainieren und für alle Modelle nutzen.

    print("\nBerechne TF-IDF ...")

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # --------------------------------------------------
    # 4. Vergleichsmodelle definieren
    # --------------------------------------------------

    modelle = {
        "Logistische Regression C=1": LogisticRegression(
            C=1.0,
            class_weight=klassengewichte,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),

        "LinearSVC C=0.1": LinearSVC(
            C=0.1,
            class_weight=klassengewichte,
            max_iter=5000,
            random_state=42,
        ),

        "LinearSVC C=1": LinearSVC(
            C=1.0,
            class_weight=klassengewichte,
            max_iter=5000,
            random_state=42,
        ),

        "LinearSVC C=3": LinearSVC(
            C=3.0,
            class_weight=klassengewichte,
            max_iter=5000,
            random_state=42,
        ),
    }

    ergebnisse = []

    bestes_modell = None
    bester_name = None
    bester_f1 = -1

    # --------------------------------------------------
    # 5. Modelle trainieren und evaluieren
    # --------------------------------------------------

    for name, modell in modelle.items():

        print("\n" + "=" * 60)
        print(f"Trainiere: {name}")

        modell.fit(X_train_tfidf, y_train)

        vorhersagen = modell.predict(X_val_tfidf)

        accuracy = accuracy_score(y_val, vorhersagen)

        macro_f1 = f1_score(
            y_val,
            vorhersagen,
            labels=KLASSEN,
            average="macro",
            zero_division=0,
        )

        ergebnisse.append({
            "modell": name,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        })

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

        if macro_f1 > bester_f1:
            bester_f1 = macro_f1
            bester_name = name
            bestes_modell = modell

    # --------------------------------------------------
    # 6. Ergebnistabelle
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("ERGEBNISVERGLEICH")
    print("=" * 70)

    print(
        f"{'Modell':<32} "
        f"{'Accuracy':>12} "
        f"{'Macro-F1':>12}"
    )

    for ergebnis in ergebnisse:
        print(
            f"{ergebnis['modell']:<32} "
            f"{ergebnis['accuracy']:>12.4f} "
            f"{ergebnis['macro_f1']:>12.4f}"
        )

    print(f"\nBestes Modell: {bester_name}")
    print(f"Bester Macro-F1: {bester_f1:.4f}")

    # --------------------------------------------------
    # 7. Ergebnisse und Modell speichern
    # --------------------------------------------------

    BERICHTORDNER.mkdir(parents=True, exist_ok=True)
    MODELLORDNER.mkdir(parents=True, exist_ok=True)

    csv_pfad = BERICHTORDNER / "experiment_06_classifiers.csv"

    with csv_pfad.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as datei:

        writer = csv.DictWriter(
            datei,
            fieldnames=["modell", "accuracy", "macro_f1"],
        )

        writer.writeheader()
        writer.writerows(ergebnisse)

    json_pfad = BERICHTORDNER / "experiment_06_classifiers.json"

    with json_pfad.open("w", encoding="utf-8") as datei:

        json.dump(
            {
                "experiment": "classifier_comparison",
                "tfidf_ngrams": [1, 2],
                "class_weight_strength": GEWICHTUNGSSTAERKE,
                "results": ergebnisse,
                "best_model": bester_name,
                "best_macro_f1": float(bester_f1),
            },
            datei,
            indent=2,
            ensure_ascii=False,
        )

    # Pipeline mit bereits trainierten Komponenten.
    # Sie kann später direkt Rohtexte verarbeiten.

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("klassifikator", bestes_modell),
    ])

    modellpfad = MODELLORDNER / "sentiment_best_classifier.joblib"

    joblib.dump(pipeline, modellpfad)

    print(f"\nCSV gespeichert: {csv_pfad}")
    print(f"JSON gespeichert: {json_pfad}")
    print(f"Modell gespeichert: {modellpfad}")

    print("\nExperiment abgeschlossen.")
    print("Testdaten wurden nicht verwendet.")


if __name__ == "__main__":
    main()
