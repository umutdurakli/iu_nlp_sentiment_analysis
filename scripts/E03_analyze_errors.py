
"""
Qualitative Fehleranalyse unserer beiden NLP-Modelle.

Zeigt echte Amazon-Rezensionen, bei denen:
1. Das ursprüngliche Modell falsch liegt.
2. Das gewichtete Modell eine andere Vorhersage trifft.

Es werden ausschließlich Validierungsdaten verwendet.
"""

import json
from pathlib import Path

import joblib


# Projektpfade
PROJEKTORDNER = Path(__file__).resolve().parent.parent

VALIDIERUNG_DATEI = (
    PROJEKTORDNER / "data" / "processed" / "splits" / "validation.jsonl"
)

MODELLORDNER = PROJEKTORDNER / "models"


def lade_rezensionen(dateipfad):
    """Lädt vollständige Rezensionen inklusive Text und Sternebewertung."""

    rezensionen = []

    with dateipfad.open("r", encoding="utf-8") as datei:
        for zeile in datei:
            rezensionen.append(json.loads(zeile))

    return rezensionen


def main():

    print("Lade Validierungsdaten und Modelle ...")

    rezensionen = lade_rezensionen(VALIDIERUNG_DATEI)

    baseline = joblib.load(
        MODELLORDNER / "sentiment_baseline.joblib"
    )

    balanced = joblib.load(
        MODELLORDNER / "sentiment_balanced.joblib"
    )

    texte = [r["text"] for r in rezensionen]

    # Beide Modelle sagen die Sterne für dieselben Texte voraus.
    vorhersagen_baseline = baseline.predict(texte)
    vorhersagen_balanced = balanced.predict(texte)

    # Wir interessieren uns zunächst für echte
    # 2-, 3- und 4-Sterne-Rezensionen.
    #
    # Die Baseline soll falsch liegen und das
    # gewichtete Modell soll anders entscheiden.

    beispiele = []

    for i, rezension in enumerate(rezensionen):

        echte_sterne = int(rezension["rating"])

        stern_baseline = int(vorhersagen_baseline[i])
        stern_balanced = int(vorhersagen_balanced[i])

        if (
            echte_sterne in [2, 3, 4]
            and stern_baseline != echte_sterne
            and stern_baseline != stern_balanced
        ):
            beispiele.append({
                "text": rezension["text"],
                "rating": echte_sterne,
                "baseline": stern_baseline,
                "balanced": stern_balanced,
            })

    print(f"\nGefundene Beispiele: {len(beispiele)}")

    # Für jede tatsächliche Sterneklasse zeigen wir
    # maximal fünf Beispiele.
    for klasse in [2, 3, 4]:

        print("\n" + "=" * 70)
        print(f"TATSÄCHLICHE BEWERTUNG: {klasse} STERNE")
        print("=" * 70)

        treffer = [
            r for r in beispiele
            if r["rating"] == klasse
        ]

        for nummer, r in enumerate(treffer[:5], start=1):

            print(f"\n--- Beispiel {nummer} ---")
            print(f"Echte Sterne:       {r['rating']}")
            print(f"Baseline-Prognose:  {r['baseline']}")
            print(f"Balanced-Prognose:  {r['balanced']}")

            # Lange Rezensionen werden für die
            # Konsolenausgabe auf 600 Zeichen begrenzt.
            print(f"\nRezension:\n{r['text'][:600]}")

            print("-" * 70)


if __name__ == "__main__":
    main()
