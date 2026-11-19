
"""
Finale Evaluation des ausgewählten NLP-Modells.

Wichtig:
- Kein erneutes Training
- Keine Hyperparameteroptimierung
- Testdaten ausschließlich zur Abschlussbewertung
"""

import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJEKTORDNER = Path(__file__).resolve().parent.parent

MODELLPFAD = (
    PROJEKTORDNER / "models" / "sentiment_best_classifier.joblib"
)

TESTPFAD = (
    PROJEKTORDNER / "data" / "processed" / "splits" / "test.jsonl"
)

BERICHTORDNER = PROJEKTORDNER / "reports"

KLASSEN = [1, 2, 3, 4, 5]


def lade_testdaten(dateipfad):
    texte = []
    bewertungen = []

    with dateipfad.open("r", encoding="utf-8") as datei:
        for zeile in datei:
            rezension = json.loads(zeile)
            texte.append(rezension["text"])
            bewertungen.append(int(rezension["rating"]))

    return texte, bewertungen


def main():
    print("Lade ausgewähltes Modell ...")
    modell = joblib.load(MODELLPFAD)

    print("Lade bisher unangetastete Testdaten ...")
    X_test, y_test = lade_testdaten(TESTPFAD)

    print(f"Testrezensionen: {len(X_test):,}")

    # Nur Vorhersagen, kein fit()!
    vorhersagen = modell.predict(X_test)

    accuracy = accuracy_score(y_test, vorhersagen)
    macro_f1 = f1_score(
        y_test,
        vorhersagen,
        labels=KLASSEN,
        average="macro",
        zero_division=0,
    )

    bericht = classification_report(
        y_test,
        vorhersagen,
        labels=KLASSEN,
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        y_test,
        vorhersagen,
        labels=KLASSEN,
    )

    print("\n" + "=" * 55)
    print("FINALE TESTERGEBNISSE")
    print("=" * 55)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nKlassifikationsbericht:")
    print(
        classification_report(
            y_test,
            vorhersagen,
            labels=KLASSEN,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(matrix)

    BERICHTORDNER.mkdir(parents=True, exist_ok=True)

    ergebnis = {
        "experiment": "final_test_evaluation",
        "modell": "sentiment_best_classifier.joblib",
        "testrezensionen": len(y_test),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": bericht,
        "confusion_matrix": matrix.tolist(),
        "klassenreihenfolge": KLASSEN,
    }

    ausgabepfad = (
        BERICHTORDNER / "final_test_evaluation.json"
    )

    with ausgabepfad.open("w", encoding="utf-8") as datei:
        json.dump(
            ergebnis,
            datei,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nErgebnisse gespeichert: {ausgabepfad}")
    print("Finale Evaluation abgeschlossen.")


if __name__ == "__main__":
    main()
