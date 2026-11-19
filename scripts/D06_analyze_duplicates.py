import json
from collections import Counter, defaultdict
from pathlib import Path

# Projektverzeichnis und Speicherorte festlegen
PROJEKTORDNER = Path(__file__).resolve().parent.parent
DATENORDNER = PROJEKTORDNER / "data" / "processed"

KATEGORIEN = [
    "Electronics",
    "Sports_and_Outdoors",
    "Video_Games",
]

# Alle ausgewählten Rezensionen einlesen
rezensionen = []

for kategorie in KATEGORIEN:
    dateipfad = DATENORDNER / f"{kategorie}_enriched_50000_seed42.jsonl"

    with dateipfad.open("r", encoding="utf-8") as datei:
        for zeile in datei:
            rezensionen.append(json.loads(zeile))

print(f"Untersuchte Rezensionen: {len(rezensionen):,}")

# Identische Rezensionstexte gruppieren
texte = defaultdict(list)

for rezension in rezensionen:
    text = rezension["text"].strip()
    texte[text].append(rezension)

# Anzahl zusätzlicher Vorkommen identischer Texte
textwiederholungen = sum(
    len(gruppe) - 1
    for gruppe in texte.values()
)

# Identische Texte mit unterschiedlichen Sternebewertungen
widerspruechliche_texte = 0
betroffene_rezensionen = 0

for gruppe in texte.values():
    bewertungen = {eintrag["rating"] for eintrag in gruppe}

    if len(bewertungen) > 1:
        widerspruechliche_texte += 1
        betroffene_rezensionen += len(gruppe)

# Mögliche Duplikate anhand von Nutzer, Produkt und Text erkennen
kombinationen = Counter(
    (
        rezension["user_id"],
        rezension["parent_asin"],
        rezension["text"].strip(),
    )
    for rezension in rezensionen
)

moegliche_duplikate = sum(
    anzahl - 1
    for anzahl in kombinationen.values()
)

# Identische Texte nach ihrer Länge untersuchen
kurze_wiederholungen = 0
lange_wiederholungen = 0

for text, gruppe in texte.items():
    wiederholungen = len(gruppe) - 1

    if len(text.split()) <= 5:
        kurze_wiederholungen += wiederholungen
    else:
        lange_wiederholungen += wiederholungen

print("\n--- Duplikatanalyse ---")
print(f"Textwiederholungen insgesamt: {textwiederholungen:,}")
print(f"Davon kurze Texte (max. 5 Wörter): {kurze_wiederholungen:,}")
print(f"Davon längere Texte: {lange_wiederholungen:,}")
print(f"Texte mit widersprüchlichen Sternen: {widerspruechliche_texte:,}")
print(f"Davon betroffene Rezensionen: {betroffene_rezensionen:,}")
print(f"Mögliche Duplikate (User + Produkt + Text): {moegliche_duplikate:,}")

print("\n--- Häufigste wiederholte Texte ---")

haeufigste = sorted(
    (
        (text, len(gruppe))
        for text, gruppe in texte.items()
        if len(gruppe) > 1
    ),
    key=lambda eintrag: eintrag[1],
    reverse=True,
)[:10]

for text, anzahl in haeufigste:
    print(f"{anzahl:>5}x | {text[:100]!r}")

print("\nAnalyse abgeschlossen. Keine Daten verändert.")