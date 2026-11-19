
"""
E11: Hyperparameteroptimierung für Titel + Beschreibung (Variante B).

Untersucht:
- alpha: Stärke der Klassengewichtung
- C: Regularisierung der logistischen Regression

25 Kombinationen, ausschließlich Trainings- und Validierungsdaten.
Bestehende Ergebnisse und Modelle werden nicht überschrieben.
"""

import csv
import json
import warnings
from pathlib import Path

import joblib
import numpy as np

from sklearn.exceptions import ConvergenceWarning
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


# --------------------------------------------------
# 1. Einstellungen
# --------------------------------------------------

PROJEKTORDNER = Path(__file__).resolve().parent.parent
DATENORDNER = PROJEKTORDNER / "data" / "processed" / "splits"
MODELLORDNER = PROJEKTORDNER / "models"
BERICHTORDNER = PROJEKTORDNER / "reports"

KLASSEN = np.array([1, 2, 3, 4, 5])

ALPHA_WERTE = [0.0, 0.25, 0.5, 0.75, 1.0]
C_WERTE = [0.1, 0.5, 1.0, 2.0, 5.0]

REFERENZ_ALPHA = 0.5
REFERENZ_C = 1.0

REFERENZ_MODELL = (
    MODELLORDNER / "sentiment_title_text_B.joblib"
)

CSV_DATEI = BERICHTORDNER / "E11_hyperparameter_results.csv"
JSON_DATEI = BERICHTORDNER / "E11_hyperparameter_results.json"
MODELL_DATEI = MODELLORDNER / "E11_title_text_best.joblib"


# --------------------------------------------------
# 2. Daten laden und Titel + Text kombinieren
# --------------------------------------------------

def lade_daten(dateipfad):

    texte = []
    bewertungen = []

    with dateipfad.open("r", encoding="utf-8") as datei:
        for zeile in datei:
            rezension = json.loads(zeile)

            titel = str(rezension.get("title") or "").strip()
            beschreibung = str(rezension.get("text") or "").strip()

            # Exakt Variante B aus E09:
            # Titel einmal + Beschreibung einmal.
            gesamter_text = " ".join(
                teil for teil in [titel, beschreibung] if teil
            )

            texte.append(gesamter_text)
            bewertungen.append(int(rezension["rating"]))

    return texte, np.array(bewertungen)


# --------------------------------------------------
# 3. Klassengewichte berechnen
# --------------------------------------------------

def berechne_gewichte(automatische_gewichte, alpha):

    return {
        int(klasse): float(gewicht ** alpha)
        for klasse, gewicht in zip(
            KLASSEN, automatische_gewichte
        )
    }


# --------------------------------------------------
# 4. Modell bewerten
# --------------------------------------------------

def bewerte(y_val, vorhersagen):

    return {
        "accuracy": float(
            accuracy_score(y_val, vorhersagen)
        ),
        "macro_f1": float(
            f1_score(
                y_val,
                vorhersagen,
                labels=KLASSEN,
                average="macro",
                zero_division=0,
            )
        ),
    }


# --------------------------------------------------
# 5. Hauptprogramm
# --------------------------------------------------

def main():

    # Keine vorhandenen Ergebnisdateien überschreiben.
    for pfad in (CSV_DATEI, JSON_DATEI, MODELL_DATEI):
        if pfad.exists():
            raise FileExistsError(
                f"Datei existiert bereits: {pfad}"
            )

    if not REFERENZ_MODELL.is_file():
        raise FileNotFoundError(
            f"Referenzmodell fehlt: {REFERENZ_MODELL}"
        )

    print("=" * 65)
    print("E11 - HYPERPARAMETEROPTIMIERUNG")
    print("Variante B: Titel + Beschreibung")
    print("=" * 65)

    # --------------------------------------------------
    # Daten laden
    # --------------------------------------------------

    print("\nLade Trainings- und Validierungsdaten ...")

    X_train, y_train = lade_daten(
        DATENORDNER / "train.jsonl"
    )

    X_val, y_val = lade_daten(
        DATENORDNER / "validation.jsonl"
    )

    print(f"Training: {len(X_train):,}")
    print(f"Validierung: {len(X_val):,}")

    # --------------------------------------------------
    # TF-IDF einmalig berechnen
    # --------------------------------------------------

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

    print(f"Trainingsmatrix: {X_train_tfidf.shape}")
    print(f"Validierungsmatrix: {X_val_tfidf.shape}")

    # --------------------------------------------------
    # Klassengewichte nur aus Trainingsdaten
    # --------------------------------------------------

    automatische_gewichte = compute_class_weight(
        class_weight="balanced",
        classes=KLASSEN,
        y=y_train,
    )

    print("\nAutomatische Klassengewichte:")
    for klasse, gewicht in zip(
        KLASSEN, automatische_gewichte
    ):
        print(f"{klasse} Sterne: {gewicht:.4f}")

    # --------------------------------------------------
    # Referenzmodell B aus E10 prüfen
    # --------------------------------------------------

    print("\nPrüfe gespeichertes Referenzmodell B ...")

    referenz_modell = joblib.load(REFERENZ_MODELL)
    referenz_vorhersagen = referenz_modell.predict(X_val)
    referenz = bewerte(y_val, referenz_vorhersagen)

    print(
        f"Referenz Accuracy: {referenz['accuracy']:.4f}"
    )
    print(
        f"Referenz Macro-F1: {referenz['macro_f1']:.4f}"
    )

    # --------------------------------------------------
    # 25 Kombinationen testen
    # --------------------------------------------------

    ergebnisse = []
    bestes_modell = None
    bestes_ergebnis = None
    beste_vorhersagen = None
    referenz_neu = None
    konvergenzwarnungen = []

    nummer = 0
    gesamt = len(ALPHA_WERTE) * len(C_WERTE)

    for alpha in ALPHA_WERTE:

        klassengewichte = berechne_gewichte(
            automatische_gewichte, alpha
        )

        for c_wert in C_WERTE:

            nummer += 1

            print("\n" + "-" * 65)
            print(
                f"Experiment {nummer}/{gesamt} | "
                f"alpha={alpha} | C={c_wert}"
            )

            klassifikator = LogisticRegression(
                C=c_wert,
                class_weight=klassengewichte,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )

            # Warnungen erfassen, statt sie zu ignorieren.
            with warnings.catch_warnings(record=True) as meldungen:
                warnings.simplefilter(
                    "always", ConvergenceWarning
                )
                klassifikator.fit(X_train_tfidf, y_train)

            warnungen = [
                str(m.message)
                for m in meldungen
                if issubclass(m.category, ConvergenceWarning)
            ]

            if warnungen:
                konvergenzwarnungen.append({
                    "alpha": alpha,
                    "C": c_wert,
                    "warnings": warnungen,
                })
                print("ACHTUNG: Konvergenzwarnung")

            vorhersagen = klassifikator.predict(
                X_val_tfidf
            )

            kennzahlen = bewerte(y_val, vorhersagen)

            klassenbericht = classification_report(
                y_val,
                vorhersagen,
                labels=KLASSEN,
                output_dict=True,
                zero_division=0,
            )

            ergebnis = {
                "alpha": alpha,
                "C": c_wert,
                "accuracy": kennzahlen["accuracy"],
                "macro_f1": kennzahlen["macro_f1"],
                "konvergenzwarnung": bool(warnungen),
            }

            for klasse in KLASSEN:
                ergebnis[f"f1_{klasse}_sterne"] = float(
                    klassenbericht[str(klasse)]["f1-score"]
                )

            ergebnisse.append(ergebnis)

            print(
                f"Accuracy: {ergebnis['accuracy']:.4f}"
            )
            print(
                f"Macro-F1: {ergebnis['macro_f1']:.4f}"
            )

            # Referenzkombination reproduzieren.
            if (
                alpha == REFERENZ_ALPHA
                and c_wert == REFERENZ_C
            ):
                referenz_neu = {
                    "metrics": ergebnis,
                    "predictions_identical": bool(
                        np.array_equal(
                            vorhersagen,
                            referenz_vorhersagen,
                        )
                    ),
                }

            # Konvergenzproblematische Kandidaten
            # nicht als bestes Modell auswählen.
            if warnungen:
                continue

            # Primär Macro-F1, sekundär Accuracy.
            if (
                bestes_ergebnis is None
                or (
                    ergebnis["macro_f1"],
                    ergebnis["accuracy"],
                ) > (
                    bestes_ergebnis["macro_f1"],
                    bestes_ergebnis["accuracy"],
                )
            ):
                bestes_ergebnis = ergebnis
                bestes_modell = klassifikator
                beste_vorhersagen = vorhersagen.copy()

    # --------------------------------------------------
    # Referenzprüfung
    # --------------------------------------------------

    if referenz_neu is None:
        raise RuntimeError(
            "Referenzkombination wurde nicht getestet."
        )

    if not referenz_neu["predictions_identical"]:
        raise RuntimeError(
            "E11 reproduziert die Vorhersagen des "
            "gespeicherten B-Modells nicht exakt. "
            "Keine Ergebnisse gespeichert. "
            "Bitte Ursache untersuchen."
        )

    if bestes_modell is None:
        raise RuntimeError(
            "Kein Modell ohne Konvergenzwarnung gefunden."
        )

    print("\nReferenzprüfung bestanden.")

    # --------------------------------------------------
    # Ergebnisse sortieren und anzeigen
    # --------------------------------------------------

    ergebnisse.sort(
        key=lambda e: (
            e["macro_f1"],
            e["accuracy"],
        ),
        reverse=True,
    )

    print("\n" + "=" * 65)
    print("ERGEBNISVERGLEICH")
    print("=" * 65)

    print(
        f"{'Alpha':>8}"
        f"{'C':>8}"
        f"{'Accuracy':>12}"
        f"{'Macro-F1':>12}"
        f"{'Warnung':>10}"
    )

    for e in ergebnisse:
        print(
            f"{e['alpha']:>8.2f}"
            f"{e['C']:>8.2f}"
            f"{e['accuracy']:>12.4f}"
            f"{e['macro_f1']:>12.4f}"
            f"{str(e['konvergenzwarnung']):>10}"
        )

    print("\nBESTES KONVERGIERTES MODELL")
    print(f"Alpha: {bestes_ergebnis['alpha']}")
    print(f"C: {bestes_ergebnis['C']}")
    print(
        f"Accuracy: {bestes_ergebnis['accuracy']:.4f}"
    )
    print(
        f"Macro-F1: {bestes_ergebnis['macro_f1']:.4f}"
    )

    differenz = (
        bestes_ergebnis["macro_f1"]
        - referenz["macro_f1"]
    )

    print(
        f"Macro-F1-Differenz zu B: {differenz:+.4f}"
    )

    print("\nKlassifikationsbericht:")

    print(
        classification_report(
            y_val,
            beste_vorhersagen,
            labels=KLASSEN,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_val,
        beste_vorhersagen,
        labels=KLASSEN,
    )

    print("Confusion Matrix:")
    print(matrix)

    # --------------------------------------------------
    # Ergebnisse speichern
    # --------------------------------------------------

    BERICHTORDNER.mkdir(parents=True, exist_ok=True)
    MODELLORDNER.mkdir(parents=True, exist_ok=True)

    with CSV_DATEI.open(
        "x", newline="", encoding="utf-8"
    ) as datei:
        writer = csv.DictWriter(
            datei,
            fieldnames=list(ergebnisse[0].keys()),
        )
        writer.writeheader()
        writer.writerows(ergebnisse)

    bericht = {
        "experiment": "E11",
        "variante": "B_titel_und_beschreibung",
        "auswahlmetrik": "macro_f1",
        "referenz_B": referenz,
        "referenz_reproduziert": True,
        "bestes_ergebnis": bestes_ergebnis,
        "macro_f1_differenz": float(differenz),
        "alle_ergebnisse": ergebnisse,
        "classification_report": classification_report(
            y_val,
            beste_vorhersagen,
            labels=KLASSEN,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
        "konvergenzwarnungen": konvergenzwarnungen,
        "hinweis": (
            "Hyperparameterwahl ausschließlich auf "
            "Validierungsdaten. Wiederholte Modellauswahl "
            "kann zu Validierungs-Overfitting führen. "
            "Testdaten wurden nicht verwendet."
        ),
    }

    with JSON_DATEI.open("x", encoding="utf-8") as datei:
        json.dump(
            bericht,
            datei,
            indent=2,
            ensure_ascii=False,
        )

    # Vollständige Pipeline für spätere Vorhersagen.
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("klassifikator", bestes_modell),
    ])

    joblib.dump(pipeline, MODELL_DATEI)

    print("\nGespeicherte Dateien:")
    print(CSV_DATEI)
    print(JSON_DATEI)
    print(MODELL_DATEI)

    print("\nE11 abgeschlossen.")
    print("Testdaten wurden nicht verwendet.")


if __name__ == "__main__":
    main()
