
"""
Experiment 8: Fehleranalyse der Titelvarianten B und C.

B: Titel + Beschreibung
C: Titel doppelt + Beschreibung

- Trainiert Variante B reproduzierbar nach
- Lädt das gespeicherte Modell C
- Vergleicht Precision, Recall, F1 und Confusion Matrices
- Zeigt ausgewählte Unterschiede in den Vorhersagen
- Verwendet ausschließlich Validierungsdaten
"""

import json
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


PROJEKTORDNER = Path(__file__).resolve().parent.parent
DATENORDNER = PROJEKTORDNER / "data" / "processed" / "splits"
MODELLORDNER = PROJEKTORDNER / "models"
BERICHTORDNER = PROJEKTORDNER / "reports"

KLASSEN = np.array([1, 2, 3, 4, 5])


def lade_daten(dateipfad):

    rezensionen = []

    with dateipfad.open("r", encoding="utf-8") as datei:
        for zeile in datei:
            rezensionen.append(json.loads(zeile))

    return rezensionen


def erstelle_texte(rezensionen, titel_wiederholungen):

    texte = []

    for rezension in rezensionen:

        titel = str(rezension.get("title") or "").strip()
        beschreibung = str(rezension.get("text") or "").strip()

        bestandteile = (
            [titel] * titel_wiederholungen
            + [beschreibung]
        )

        texte.append(
            " ".join(teil for teil in bestandteile if teil)
        )

    return texte


def bewerte_modell(name, modell, texte, y_val):

    vorhersagen = modell.predict(texte)

    accuracy = accuracy_score(y_val, vorhersagen)

    macro_f1 = f1_score(
        y_val,
        vorhersagen,
        labels=KLASSEN,
        average="macro",
        zero_division=0,
    )

    bericht = classification_report(
        y_val,
        vorhersagen,
        labels=KLASSEN,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_val,
        vorhersagen,
        labels=KLASSEN,
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print(
        classification_report(
            y_val,
            vorhersagen,
            labels=KLASSEN,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(matrix)

    return vorhersagen, {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": bericht,
        "confusion_matrix": matrix.tolist(),
    }


def main():

    print("Lade Trainings- und Validierungsdaten ...")

    train = lade_daten(DATENORDNER / "train.jsonl")
    val = lade_daten(DATENORDNER / "validation.jsonl")

    y_train = np.array([int(r["rating"]) for r in train])
    y_val = np.array([int(r["rating"]) for r in val])

    gewichte_balanced = compute_class_weight(
        class_weight="balanced",
        classes=KLASSEN,
        y=y_train,
    )

    klassengewichte = {
        int(klasse): float(gewicht ** 0.5)
        for klasse, gewicht in zip(KLASSEN, gewichte_balanced)
    }

    # Variante B mit identischen Einstellungen nachtrainieren.

    print("Trainiere Variante B ...")

    modell_b = Pipeline([
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

    modell_b.fit(erstelle_texte(train, 1), y_train)

    # Variante C aus Experiment 7 laden.

    print("Lade Variante C ...")

    modell_c = joblib.load(
        MODELLORDNER / "sentiment_title_text_best.joblib"
    )

    texte_b = erstelle_texte(val, 1)
    texte_c = erstelle_texte(val, 2)

    vorhersagen_b, ergebnis_b = bewerte_modell(
        "Variante B: Titel + Beschreibung",
        modell_b,
        texte_b,
        y_val,
    )

    vorhersagen_c, ergebnis_c = bewerte_modell(
        "Variante C: Titel doppelt",
        modell_c,
        texte_c,
        y_val,
    )

    # --------------------------------------------------
    # Unterschiede zwischen B und C untersuchen.
    # --------------------------------------------------

    b_richtig_c_falsch = np.where(
        (vorhersagen_b == y_val)
        & (vorhersagen_c != y_val)
    )[0]

    c_richtig_b_falsch = np.where(
        (vorhersagen_c == y_val)
        & (vorhersagen_b != y_val)
    )[0]

    print("\n" + "=" * 60)
    print("DIREKTER VERGLEICH")
    print("=" * 60)

    print(
        "B richtig, C falsch:",
        len(b_richtig_c_falsch),
    )

    print(
        "C richtig, B falsch:",
        len(c_richtig_b_falsch),
    )

    print("\nBeispiele: C richtig, B falsch")

    for index in c_richtig_b_falsch[:5]:

        rezension = val[index]

        print("\n" + "-" * 50)
        print("Tatsächliche Sterne:", y_val[index])
        print("Vorhersage B:", vorhersagen_b[index])
        print("Vorhersage C:", vorhersagen_c[index])
        print("Titel:", rezension.get("title", ""))
        print(
            "Beschreibung:",
            str(rezension.get("text", ""))[:500],
        )

    print("\nBeispiele: B richtig, C falsch")

    for index in b_richtig_c_falsch[:5]:

        rezension = val[index]

        print("\n" + "-" * 50)
        print("Tatsächliche Sterne:", y_val[index])
        print("Vorhersage B:", vorhersagen_b[index])
        print("Vorhersage C:", vorhersagen_c[index])
        print("Titel:", rezension.get("title", ""))
        print(
            "Beschreibung:",
            str(rezension.get("text", ""))[:500],
        )

    # Ergebnisse speichern.

    BERICHTORDNER.mkdir(parents=True, exist_ok=True)

    ergebnis = {
        "experiment": "title_model_error_analysis",
        "variante_b": ergebnis_b,
        "variante_c": ergebnis_c,
        "b_richtig_c_falsch": int(len(b_richtig_c_falsch)),
        "c_richtig_b_falsch": int(len(c_richtig_b_falsch)),
    }

    ausgabepfad = (
        BERICHTORDNER / "experiment_08_title_analysis.json"
    )

    with ausgabepfad.open("w", encoding="utf-8") as datei:
        json.dump(
            ergebnis,
            datei,
            indent=2,
            ensure_ascii=False,
        )

    joblib.dump(
        modell_b,
        MODELLORDNER / "sentiment_title_text_B.joblib",
    )

    print(f"\nErgebnisse gespeichert: {ausgabepfad}")
    print("Testdaten wurden nicht verwendet.")


if __name__ == "__main__":
    main()
