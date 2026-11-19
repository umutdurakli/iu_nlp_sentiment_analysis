
"""Prüft die Qualität unserer stratifizierten Amazon-Stichproben."""

import json
from collections import Counter
from pathlib import Path

# Ordner mit unseren ausgewählten Rezensionen
DATENORDNER = Path("data/interim")

# Dateien, die wir untersuchen möchten
DATEIEN = sorted(DATENORDNER.glob("*_stratified_50000_seed42.jsonl"))

# Wir merken uns Texte auch über Kategoriegrenzen hinweg.
bereits_gesehene_texte = set()

gesamtzahl = 0
gesamt_duplikate = 0

if len(DATEIEN) != 3:
    raise SystemExit(
        f"Erwartet: 3 Dateien. Gefunden: {len(DATEIEN)}."
    )

for dateipfad in DATEIEN:
    print(f"\nKategorie: {dateipfad.stem}")

    sterne_haeufigkeiten = Counter()
    anzahl = 0
    fehlende_texte = 0
    doppelte_texte = 0

    with dateipfad.open("r", encoding="utf-8") as datei:
        for zeile in datei:
            rezension = json.loads(zeile)
            anzahl += 1

            # Sternebewertung zählen
            sterne = rezension["rating"]
            sterne_haeufigkeiten[sterne] += 1

            # Text auslesen und auf leere Inhalte prüfen
            text = rezension.get("text", "")

            if not isinstance(text, str) or not text.strip():
                fehlende_texte += 1
                continue

            # Identische Texte erkennen; Leerzeichen außen ignorieren.
            vergleichstext = text.strip()

            if vergleichstext in bereits_gesehene_texte:
                doppelte_texte += 1
            else:
                bereits_gesehene_texte.add(vergleichstext)

    gesamtzahl += anzahl
    gesamt_duplikate += doppelte_texte

    print(f"Rezensionen: {anzahl:,}")
    print(f"Fehlende Texte: {fehlende_texte}")
    print(f"Doppelte Texte: {doppelte_texte}")
    print("Sterneverteilung:")

    for sterne in range(1, 6):
        anzahl_sterne = sterne_haeufigkeiten[sterne]
        anteil = anzahl_sterne / anzahl * 100 if anzahl else 0

        print(
            f"  {sterne} Sterne: "
            f"{anzahl_sterne:,} ({anteil:.2f} %)"
        )

print("\n--- Gesamtergebnis ---")
print(f"Untersuchte Rezensionen: {gesamtzahl:,}")
print(f"Doppelte Texte insgesamt: {gesamt_duplikate:,}")
print(f"Unterschiedliche Texte: {len(bereits_gesehene_texte):,}")
print("\nPrüfung abgeschlossen. Keine Daten verändert.")
