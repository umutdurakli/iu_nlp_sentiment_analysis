
"""
Experiment 4: Regularisierung der logistischen Regression.

Wir testen verschiedene C-Werte bei einer festen
Klassengewichtungsstärke von 0.50.

Zielmetrik: Macro-F1 auf dem Validierungsdatensatz.
Testdaten bleiben unangetastet.
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


# --------------------------------------------------
# 1. Projektpfade und Einstellungen
# --------------------------------------------------

PROJEKTORDNER = Path(__file__).resolve().parent.parent

DATENORDNER = PROJEKTORDNER / "data" / "processed" / "splits"
MODELLORDNER = PROJEKTORDNER / "models"
BERICHTORDNER = PROJEKTORDNER / "reports"

C_WERTE = [0.1, 0.3, 1.0, 3.0, 10.0]
GEWICHTUNGSSTAERKE = 0.50
KLASSEN = np.array([1, 2, 3, 4, 5])


# --------------------------------------------------
# 2. Rezensionen laden
# --------------------------------------------------

def lade_daten(dateipfad):

    texte = []
    bewertungen = []

    with dateipfad.open("r", encoding="utf-8") as datei:
        for zeile in datei:
            rezension = json.loads(zeile)
            texte.append(rezension["text"])
            bewertungen.append(int(rezension["rating"]))

    return texte, np.array(bewertungen)


# --------------------------------------------------
# 3. Hauptprogramm
# --------------------------------------------------

def main():

    print("Lade Trainings- und Validierungsdaten ...")

    X_train, y_train = lade_daten(
        DATENORDNER / "train.jsonl"
    )

    X_val, y_val = lade_daten(
        DATENORDNER / "validation.jsonl"
    )

    # TF-IDF bleibt identisch zu unseren bisherigen
    # Experimenten.

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

    # Klassengewichte aus Experiment 3 übernehmen.

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

    print("\nVerwendete Klassengewichte:")
    print(klassengewichte)

    ergebnisse = []

    bestes_modell = None
    bester_f1 = -1
    bester_c_wert = None

    # --------------------------------------------------
    # 4. Verschiedene C-Werte testen
    # --------------------------------------------------

    for c_wert in C_WERTE:

        print(f"\nTrainiere Modell mit C = {c_wert}")

        modell = LogisticRegression(
            C=c_wert,
            class_weight=klassengewichte,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )

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
            "C": c_wert,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        })

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")

        if macro_f1 > bester_f1:
            bester_f1 = macro_f1
            bester_c_wert = c_wert
            bestes_modell = modell

    # --------------------------------------------------
    # 5. Ergebnisse vergleichen
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("ERGEBNISVERGLEICH")
    print("=" * 50)

    print(f"{'C-Wert':>10} {'Accuracy':>12} {'Macro-F1':>12}")

    for ergebnis in ergebnisse:
        print(
            f"{ergebnis['C']:>10.2f} "
            f"{ergebnis['accuracy']:>12.4f} "
            f"{ergebnis['macro_f1']:>12.4f}"
        )

    print(f"\nBester C-Wert: {bester_c_wert}")
    print(f"Bester Macro-F1: {bester_f1:.4f}")

    # --------------------------------------------------
    # 6. Ergebnisse und Modell speichern
    # --------------------------------------------------

    BERICHTORDNER.mkdir(parents=True, exist_ok=True)
    MODELLORDNER.mkdir(parents=True, exist_ok=True)

    csv_pfad = BERICHTORDNER / "experiment_04_regularization.csv"

    with csv_pfad.open("w", newline="", encoding="utf-8") as datei:
        writer = csv.DictWriter(
            datei,
            fieldnames=["C", "accuracy", "macro_f1"],
        )
        writer.writeheader()
        writer.writerows(ergebnisse)

    json_pfad = BERICHTORDNER / "experiment_04_regularization.json"

    with json_pfad.open("w", encoding="utf-8") as datei:
        json.dump(
            {
                "experiment": "regularization_tuning",
                "class_weight_strength": GEWICHTUNGSSTAERKE,
                "results": ergebnisse,
                "best_C": bester_c_wert,
                "best_macro_f1": float(bester_f1),
            },
            datei,
            indent=2,
        )

    modellpfad = MODELLORDNER / "sentiment_regularization_tuned.joblib"

    joblib.dump(
        {
            "tfidf": tfidf,
            "klassifikator": bestes_modell,
            "C": bester_c_wert,
            "gewichtungsstaerke": GEWICHTUNGSSTAERKE,
        },
        modellpfad,
    )

    print("\nExperiment abgeschlossen.")
    print(f"Ergebnisse: {csv_pfad}")
    print(f"Modell: {modellpfad}")


if __name__ == "__main__":
    main()
