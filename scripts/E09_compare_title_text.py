
"""
Experiment 7: Einfluss des Rezensionstitels.

Varianten:
A: Nur Beschreibung
B: Titel + Beschreibung
C: Titel doppelt + Beschreibung

Konstante Modellparameter:
- TF-IDF: Uni- und Bigramme, max. 100.000 Merkmale
- Logistische Regression: C=1.0
- Klassengewichtungsstärke: 0.5

Auswertung ausschließlich auf Validierungsdaten.
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight


PROJEKTORDNER = Path(__file__).resolve().parent.parent
DATENORDNER = PROJEKTORDNER / "data" / "processed" / "splits"
MODELLORDNER = PROJEKTORDNER / "models"
BERICHTORDNER = PROJEKTORDNER / "reports"

KLASSEN = np.array([1, 2, 3, 4, 5])
GEWICHTUNGSSTAERKE = 0.5

VARIANTEN = {
    "A_nur_beschreibung": 0,
    "B_titel_und_beschreibung": 1,
    "C_titel_doppelt": 2,
}


def lade_daten(dateipfad, titel_wiederholungen):

    texte = []
    bewertungen = []

    with dateipfad.open("r", encoding="utf-8") as datei:

        for zeile in datei:
            rezension = json.loads(zeile)

            titel = str(rezension.get("title") or "").strip()
            beschreibung = str(rezension.get("text") or "").strip()

            bestandteile = (
                [titel] * titel_wiederholungen
                + [beschreibung]
            )

            gesamter_text = " ".join(
                teil for teil in bestandteile if teil
            )

            texte.append(gesamter_text)
            bewertungen.append(int(rezension["rating"]))

    return texte, np.array(bewertungen)


def main():

    MODELLORDNER.mkdir(parents=True, exist_ok=True)
    BERICHTORDNER.mkdir(parents=True, exist_ok=True)

    # Klassengewichte nur aus Trainingsdaten bestimmen.
    _, y_train = lade_daten(
        DATENORDNER / "train.jsonl",
        titel_wiederholungen=0,
    )

    automatische_gewichte = compute_class_weight(
        class_weight="balanced",
        classes=KLASSEN,
        y=y_train,
    )

    klassengewichte = {
        int(klasse): float(gewicht ** GEWICHTUNGSSTAERKE)
        for klasse, gewicht in zip(KLASSEN, automatische_gewichte)
    }

    ergebnisse = []
    bester_f1 = -1
    bestes_modell = None
    beste_variante = None

    for name, wiederholungen in VARIANTEN.items():

        print("\n" + "=" * 60)
        print(f"Variante: {name}")

        X_train, y_train = lade_daten(
            DATENORDNER / "train.jsonl",
            titel_wiederholungen=wiederholungen,
        )

        X_val, y_val = lade_daten(
            DATENORDNER / "validation.jsonl",
            titel_wiederholungen=wiederholungen,
        )

        modell = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    max_features=100_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "klassifikator",
                LogisticRegression(
                    C=1.0,
                    class_weight=klassengewichte,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ])

        modell.fit(X_train, y_train)
        vorhersagen = modell.predict(X_val)

        accuracy = accuracy_score(y_val, vorhersagen)

        macro_f1 = f1_score(
            y_val,
            vorhersagen,
            labels=KLASSEN,
            average="macro",
            zero_division=0,
        )

        merkmale = len(
            modell.named_steps["tfidf"].vocabulary_
        )

        ergebnisse.append({
            "variante": name,
            "titel_wiederholungen": wiederholungen,
            "merkmale": merkmale,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        })

        print(f"Merkmale: {merkmale:,}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")

        if macro_f1 > bester_f1:
            bester_f1 = macro_f1
            beste_variante = name
            bestes_modell = modell

    # Ergebnisse anzeigen.
    print("\n" + "=" * 75)
    print("ERGEBNISVERGLEICH")
    print("=" * 75)

    print(
        f"{'Variante':<30}"
        f"{'Accuracy':>12}"
        f"{'Macro-F1':>12}"
    )

    for ergebnis in ergebnisse:
        print(
            f"{ergebnis['variante']:<30}"
            f"{ergebnis['accuracy']:>12.4f}"
            f"{ergebnis['macro_f1']:>12.4f}"
        )

    print(f"\nBeste Variante: {beste_variante}")
    print(f"Bester Macro-F1: {bester_f1:.4f}")

    # CSV speichern.
    csv_pfad = BERICHTORDNER / "experiment_07_title_text.csv"

    with csv_pfad.open("w", newline="", encoding="utf-8") as datei:
        writer = csv.DictWriter(
            datei,
            fieldnames=list(ergebnisse[0].keys()),
        )
        writer.writeheader()
        writer.writerows(ergebnisse)

    # JSON speichern.
    json_pfad = BERICHTORDNER / "experiment_07_title_text.json"

    with json_pfad.open("w", encoding="utf-8") as datei:
        json.dump(
            {
                "experiment": "title_text_comparison",
                "results": ergebnisse,
                "best_variant": beste_variante,
                "best_macro_f1": float(bester_f1),
            },
            datei,
            indent=2,
            ensure_ascii=False,
        )

    # Bestes Modell speichern.
    modellpfad = MODELLORDNER / "sentiment_title_text_best.joblib"

    joblib.dump(bestes_modell, modellpfad)

    print(f"\nCSV gespeichert: {csv_pfad}")
    print(f"JSON gespeichert: {json_pfad}")
    print(f"Modell gespeichert: {modellpfad}")

    print("\nExperiment abgeschlossen.")
    print("Testdaten wurden nicht verwendet.")


if __name__ == "__main__":
    main()
