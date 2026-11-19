
"""
Experiment 5: Einfluss verschiedener N-Gramme.

Vergleich:
    (1, 1) = nur Unigramme
    (1, 2) = Unigramme und Bigramme
    (1, 3) = Unigramme, Bigramme und Trigramme

Alle anderen Hyperparameter bleiben unverändert.
Bewertung ausschließlich auf Validierungsdaten.
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


# --------------------------------------------------
# 1. Einstellungen
# --------------------------------------------------

PROJEKTORDNER = Path(__file__).resolve().parent.parent

DATENORDNER = PROJEKTORDNER / "data" / "processed" / "splits"
MODELLORDNER = PROJEKTORDNER / "models"
BERICHTORDNER = PROJEKTORDNER / "reports"

NGRAM_VARIANTEN = [(1, 1), (1, 2), (1, 3)]

KLASSEN = np.array([1, 2, 3, 4, 5])

C_WERT = 1.0
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
# 3. Klassengewichte berechnen
# --------------------------------------------------

def berechne_klassengewichte(y_train):

    automatische_gewichte = compute_class_weight(
        class_weight="balanced",
        classes=KLASSEN,
        y=y_train,
    )

    gewichte = automatische_gewichte ** GEWICHTUNGSSTAERKE

    return {
        int(klasse): float(gewicht)
        for klasse, gewicht in zip(KLASSEN, gewichte)
    }


# --------------------------------------------------
# 4. Hauptprogramm
# --------------------------------------------------

def main():

    print("Lade Trainings- und Validierungsdaten ...")

    X_train, y_train = lade_daten(
        DATENORDNER / "train.jsonl"
    )

    X_val, y_val = lade_daten(
        DATENORDNER / "validation.jsonl"
    )

    klassengewichte = berechne_klassengewichte(y_train)

    ergebnisse = []

    bester_f1 = -1
    bestes_modell = None
    beste_variante = None

    # --------------------------------------------------
    # 5. Drei N-Gramm-Varianten trainieren
    # --------------------------------------------------

    for ngram in NGRAM_VARIANTEN:

        print("\n" + "=" * 55)
        print(f"Trainiere N-Gramm-Variante {ngram}")

        modell = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=ngram,
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
                        C=C_WERT,
                        class_weight=klassengewichte,
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )

        # TF-IDF und Klassifikator neu trainieren.
        modell.fit(X_train, y_train)

        # Vorhersagen auf Validierungsdaten.
        vorhersagen = modell.predict(X_val)

        accuracy = accuracy_score(y_val, vorhersagen)

        macro_f1 = f1_score(
            y_val,
            vorhersagen,
            labels=KLASSEN,
            average="macro",
            zero_division=0,
        )

        anzahl_merkmale = len(
            modell.named_steps["tfidf"].vocabulary_
        )

        ergebnis = {
            "ngram_min": ngram[0],
            "ngram_max": ngram[1],
            "merkmale": anzahl_merkmale,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        }

        ergebnisse.append(ergebnis)

        print(f"Anzahl Merkmale: {anzahl_merkmale:,}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")

        if macro_f1 > bester_f1:
            bester_f1 = macro_f1
            beste_variante = ngram
            bestes_modell = modell

    # --------------------------------------------------
    # 6. Vergleichstabelle
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("ERGEBNISVERGLEICH")
    print("=" * 65)

    print(
        f"{'N-Gramme':>12} "
        f"{'Merkmale':>12} "
        f"{'Accuracy':>12} "
        f"{'Macro-F1':>12}"
    )

    for ergebnis in ergebnisse:

        bezeichnung = (
            f"({ergebnis['ngram_min']}, "
            f"{ergebnis['ngram_max']})"
        )

        print(
            f"{bezeichnung:>12} "
            f"{ergebnis['merkmale']:>12,} "
            f"{ergebnis['accuracy']:>12.4f} "
            f"{ergebnis['macro_f1']:>12.4f}"
        )

    print(f"\nBeste N-Gramm-Variante: {beste_variante}")
    print(f"Bester Macro-F1: {bester_f1:.4f}")

    # --------------------------------------------------
    # 7. Ergebnisse speichern
    # --------------------------------------------------

    BERICHTORDNER.mkdir(parents=True, exist_ok=True)
    MODELLORDNER.mkdir(parents=True, exist_ok=True)

    csv_pfad = BERICHTORDNER / "experiment_05_ngrams.csv"

    with csv_pfad.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as datei:

        writer = csv.DictWriter(
            datei,
            fieldnames=list(ergebnisse[0].keys()),
        )

        writer.writeheader()
        writer.writerows(ergebnisse)

    json_pfad = BERICHTORDNER / "experiment_05_ngrams.json"

    with json_pfad.open("w", encoding="utf-8") as datei:

        json.dump(
            {
                "experiment": "ngram_comparison",
                "C": C_WERT,
                "gewichtungsstaerke": GEWICHTUNGSSTAERKE,
                "ergebnisse": ergebnisse,
                "beste_variante": list(beste_variante),
                "bester_macro_f1": float(bester_f1),
            },
            datei,
            indent=2,
            ensure_ascii=False,
        )

    modellpfad = MODELLORDNER / "sentiment_ngram_tuned.joblib"

    joblib.dump(bestes_modell, modellpfad)

    print(f"\nCSV gespeichert: {csv_pfad}")
    print(f"JSON gespeichert: {json_pfad}")
    print(f"Modell gespeichert: {modellpfad}")

    print("\nExperiment abgeschlossen.")
    print("Testdaten wurden nicht verwendet.")


if __name__ == "__main__":
    main()
