
"""
Erstes NLP-Klassifikationsmodell für Amazon-Rezensionen.

Ziel:
    Vorhersage einer Sternebewertung von 1 bis 5 anhand des Rezensionstextes.

Ablauf:
    1. Trainings- und Validierungsdaten laden
    2. Mehrheitsklassen-Baseline berechnen
    3. Rezensionstexte mit TF-IDF vektorisieren
    4. Logistische Regression trainieren
    5. Modell auf Validierungsdaten bewerten
    6. Trainiertes Modell speichern

Der Testdatensatz wird nicht verwendet.
"""

import json
from collections import Counter
from pathlib import Path

import joblib

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# 1. Projektpfade und Einstellungen
# ---------------------------------------------------------

PROJEKTORDNER = Path(__file__).resolve().parent.parent

DATENORDNER = PROJEKTORDNER / "data" / "processed" / "splits"
MODELLORDNER = PROJEKTORDNER / "models"

TRAIN_DATEI = DATENORDNER / "train.jsonl"
VALIDIERUNG_DATEI = DATENORDNER / "validation.jsonl"

SEED = 42


# ---------------------------------------------------------
# 2. Rezensionen aus einer JSONL-Datei laden
# ---------------------------------------------------------

def lade_rezensionen(dateipfad):
    """
    Liest eine JSONL-Datei ein.

    Rückgabe:
        texte: Liste der Rezensionstexte
        sterne: Liste der zugehörigen Sternebewertungen
    """

    texte = []
    sterne = []

    with dateipfad.open("r", encoding="utf-8") as datei:

        for zeile in datei:

            rezension = json.loads(zeile)

            text = rezension["text"]
            bewertung = int(rezension["rating"])

            texte.append(text)
            sterne.append(bewertung)

    return texte, sterne


# ---------------------------------------------------------
# 3. Modellqualität auswerten
# ---------------------------------------------------------

def bewerte_modell(name, echte_sterne, vorhersagen):
    """
    Berechnet wichtige Kennzahlen für die Klassifikation.
    """

    accuracy = accuracy_score(echte_sterne, vorhersagen)

    macro_f1 = f1_score(
        echte_sterne,
        vorhersagen,
        labels=[1, 2, 3, 4, 5],
        average="macro",
        zero_division=0,
    )

    print(f"\n--- {name} ---")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nErgebnisse je Sterneklasse:")

    print(
        classification_report(
            echte_sterne,
            vorhersagen,
            labels=[1, 2, 3, 4, 5],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print("Zeilen = tatsächliche Sterne")
    print("Spalten = vorhergesagte Sterne")

    print(
        confusion_matrix(
            echte_sterne,
            vorhersagen,
            labels=[1, 2, 3, 4, 5],
        )
    )


# ---------------------------------------------------------
# 4. Hauptprogramm
# ---------------------------------------------------------

def main():

    print("Lade Trainingsdaten ...")

    X_train, y_train = lade_rezensionen(TRAIN_DATEI)

    print("Lade Validierungsdaten ...")

    X_val, y_val = lade_rezensionen(VALIDIERUNG_DATEI)

    print(f"\nTrainingsrezensionen: {len(X_train):,}")
    print(f"Validierungsrezensionen: {len(X_val):,}")

    print("\nSterneverteilung im Training:")
    print(dict(sorted(Counter(y_train).items())))

    # -----------------------------------------------------
    # 5. Einfache Vergleichs-Baseline
    # -----------------------------------------------------

    print("\nTrainiere Mehrheitsklassen-Baseline ...")

    # Dieses Modell sagt immer die häufigste Sterneklasse
    # aus dem Trainingsdatensatz voraus.

    dummy = DummyClassifier(strategy="most_frequent")

    # DummyClassifier benötigt hier keine echten Textmerkmale.
    # Wir verwenden lediglich ein konstantes numerisches Merkmal.
    dummy.fit([[0]] * len(y_train), y_train)

    dummy_vorhersagen = dummy.predict([[0]] * len(y_val))

    bewerte_modell(
        "Mehrheitsklassen-Baseline",
        y_val,
        dummy_vorhersagen,
    )

    # -----------------------------------------------------
    # 6. TF-IDF und logistische Regression
    # -----------------------------------------------------

    print("\nErstelle TF-IDF-Pipeline ...")

    modell = Pipeline(
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
                    random_state=SEED,
                ),
            ),
        ]
    )

    # -----------------------------------------------------
    # 7. Modell trainieren
    # -----------------------------------------------------

    print("\nTrainiere TF-IDF + logistische Regression ...")

    # fit() führt zwei Schritte aus:
    #
    # 1. TF-IDF lernt das Vokabular und die IDF-Gewichte
    #    ausschließlich aus den Trainingsrezensionen.
    #
    # 2. Die logistische Regression lernt anhand der
    #    TF-IDF-Vektoren und Sternebewertungen.

    modell.fit(X_train, y_train)

    print("Training abgeschlossen.")

    # -----------------------------------------------------
    # 8. Validierungsdaten vorhersagen
    # -----------------------------------------------------

    print("\nBerechne Vorhersagen für Validierungsdaten ...")

    # Die Pipeline verwendet das bereits gelernte
    # TF-IDF-Vokabular. Sie lernt es NICHT erneut.

    vorhersagen = modell.predict(X_val)

    bewerte_modell(
        "TF-IDF + logistische Regression",
        y_val,
        vorhersagen,
    )

    # -----------------------------------------------------
    # 9. Informationen zur Vektorisierung
    # -----------------------------------------------------

    tfidf = modell.named_steps["tfidf"]

    print("\n--- TF-IDF-Informationen ---")

    print(f"Anzahl gelernter Merkmale: {len(tfidf.vocabulary_):,}")

    beispiel_merkmale = tfidf.get_feature_names_out()[:20]

    print("Erste 20 Merkmale:")
    print(beispiel_merkmale)

    # -----------------------------------------------------
    # 10. Trainiertes Modell speichern
    # -----------------------------------------------------

    MODELLORDNER.mkdir(parents=True, exist_ok=True)

    modellpfad = MODELLORDNER / "sentiment_baseline.joblib"

    joblib.dump(modell, modellpfad)

    print(f"\nModell gespeichert unter:\n{modellpfad}")

    print("\nFertig. Der Testdatensatz wurde nicht verwendet.")


# ---------------------------------------------------------
# Programmstart
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
