
"""
Experiment 3: Klassengewichtung systematisch abstimmen.

Wir testen fünf Gewichtungsstärken:
0.00 = keine Gewichtung
1.00 = vollständige automatische Klassengewichtung

Dazwischen liegen drei Zwischenstufen.

Bewertung ausschließlich auf Validierungsdaten.
Der Testdatensatz bleibt unangetastet.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


# --------------------------------------------------
# 1. Einstellungen
# --------------------------------------------------

PROJEKTORDNER = Path(__file__).resolve().parent.parent

DATENORDNER = PROJEKTORDNER / "data" / "processed" / "splits"
MODELLORDNER = PROJEKTORDNER / "models"
BERICHTORDNER = PROJEKTORDNER / "reports"

TRAIN_DATEI = DATENORDNER / "train.jsonl"
VALIDIERUNG_DATEI = DATENORDNER / "validation.jsonl"

KLASSEN = np.array([1, 2, 3, 4, 5])

GEWICHTUNGSSTAERKEN = [0.0, 0.25, 0.5, 0.75, 1.0]


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

    return texte, np.array(sterne)


# --------------------------------------------------
# 3. Klassengewichte berechnen
# --------------------------------------------------

def berechne_gewichte(y_train, staerke):

    # Automatische Gewichte aus den Trainingsdaten.
    automatische_gewichte = compute_class_weight(
        class_weight="balanced",
        classes=KLASSEN,
        y=y_train,
    )

    # Stärke 0: Alle Gewichte sind 1.
    # Stärke 1: Vollständige automatische Gewichte.
    # Zwischenwerte: Abgeschwächte Gewichtung.
    gewichte = automatische_gewichte ** staerke

    return {
        int(klasse): float(gewicht)
        for klasse, gewicht in zip(KLASSEN, gewichte)
    }


# --------------------------------------------------
# 4. Hauptprogramm
# --------------------------------------------------

def main():

    MODELLORDNER.mkdir(parents=True, exist_ok=True)
    BERICHTORDNER.mkdir(parents=True, exist_ok=True)

    print("Lade Daten ...")

    X_train, y_train = lade_rezensionen(TRAIN_DATEI)
    X_val, y_val = lade_rezensionen(VALIDIERUNG_DATEI)

    print(f"Training: {len(X_train):,} Rezensionen")
    print(f"Validierung: {len(X_val):,} Rezensionen")

    print("\nSterneverteilung:")
    print(dict(sorted(Counter(y_train).items())))

    # --------------------------------------------------
    # 5. TF-IDF einmalig berechnen
    # --------------------------------------------------

    print("\nBerechne TF-IDF-Merkmale ...")

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    # Vokabular und IDF nur aus Trainingsdaten lernen.
    X_train_tfidf = tfidf.fit_transform(X_train)

    # Validierungsdaten nur transformieren.
    X_val_tfidf = tfidf.transform(X_val)

    print(f"TF-IDF-Matrix Training: {X_train_tfidf.shape}")
    print(f"TF-IDF-Matrix Validierung: {X_val_tfidf.shape}")

    # --------------------------------------------------
    # 6. Verschiedene Klassengewichte testen
    # --------------------------------------------------

    ergebnisse = []

    bestes_modell = None
    bester_f1 = -1
    beste_staerke = None

    for staerke in GEWICHTUNGSSTAERKEN:

        print("\n" + "=" * 55)
        print(f"Trainiere mit Gewichtungsstärke {staerke:.2f}")

        klassengewichte = berechne_gewichte(
            y_train,
            staerke,
        )

        print("Klassengewichte:")
        print({
            klasse: round(gewicht, 3)
            for klasse, gewicht in klassengewichte.items()
        })

        modell = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
            class_weight=klassengewichte,
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

        recall_klassen = {}

        for klasse in KLASSEN:

            maske = y_val == klasse

            recall = np.mean(
                vorhersagen[maske] == klasse
            )

            recall_klassen[f"recall_{klasse}"] = float(recall)

        ergebnis = {
            "staerke": staerke,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            **recall_klassen,
        }

        ergebnisse.append(ergebnis)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")

        # Das Modell mit dem höchsten Macro-F1 auswählen.
        if macro_f1 > bester_f1:

            bester_f1 = macro_f1
            beste_staerke = staerke
            bestes_modell = modell

    # --------------------------------------------------
    # 7. Ergebnistabelle anzeigen
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("ERGEBNISVERGLEICH")
    print("=" * 65)

    print(
        f"{'Stärke':>8} "
        f"{'Accuracy':>12} "
        f"{'Macro-F1':>12}"
    )

    for ergebnis in ergebnisse:

        print(
            f"{ergebnis['staerke']:>8.2f} "
            f"{ergebnis['accuracy']:>12.4f} "
            f"{ergebnis['macro_f1']:>12.4f}"
        )

    print("\nBeste Gewichtungsstärke:", beste_staerke)
    print(f"Bester Macro-F1: {bester_f1:.4f}")

    # --------------------------------------------------
    # 8. Ergebnisse speichern
    # --------------------------------------------------

    csv_pfad = (
        BERICHTORDNER / "experiment_03_weight_tuning.csv"
    )

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

    json_pfad = (
        BERICHTORDNER / "experiment_03_weight_tuning.json"
    )

    bericht = {
        "experiment": "class_weight_tuning",
        "zielmetrik": "macro_f1",
        "gewichtungsformel": "balanced_weight ** staerke",
        "trainingsdaten": len(X_train),
        "validierungsdaten": len(X_val),
        "ergebnisse": ergebnisse,
        "beste_staerke": beste_staerke,
        "bester_macro_f1": float(bester_f1),
    }

    with json_pfad.open("w", encoding="utf-8") as datei:
        json.dump(bericht, datei, indent=2, ensure_ascii=False)

    # TF-IDF und besten Klassifikator zusammen speichern.
    # So können wir später neue Texte klassifizieren.

    modellpfad = (
        MODELLORDNER / "sentiment_weight_tuned.joblib"
    )

    joblib.dump(
        {
            "tfidf": tfidf,
            "klassifikator": bestes_modell,
            "gewichtungsstaerke": beste_staerke,
        },
        modellpfad,
    )

    print(f"\nCSV gespeichert: {csv_pfad}")
    print(f"JSON gespeichert: {json_pfad}")
    print(f"Modell gespeichert: {modellpfad}")

    print("\nFertig. Testdaten wurden nicht verwendet.")


if __name__ == "__main__":
    main()
