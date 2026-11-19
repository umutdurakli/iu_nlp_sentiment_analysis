import gzip
import json
from pathlib import Path

# Projektverzeichnis unabhängig vom aktuellen Terminalordner bestimmen
PROJEKTORDNER = Path(__file__).resolve().parent.parent

RAW_ORDNER = PROJEKTORDNER / "data" / "raw"
INTERIM_ORDNER = PROJEKTORDNER / "data" / "interim"
PROCESSED_ORDNER = PROJEKTORDNER / "data" / "processed"

KATEGORIEN = [
    "Electronics",
    "Sports_and_Outdoors",
    "Video_Games",
]

STICHPROBENGROESSE = 50_000
SEED = 42


def metadaten_ergaenzen(kategorie):
    """Ergänzt die ausgewählten Rezensionen um Nutzer- und Produktkennungen."""

    eingabedatei = (
        INTERIM_ORDNER
        / f"{kategorie}_stratified_{STICHPROBENGROESSE}_seed{SEED}.jsonl"
    )

    originaldatei = RAW_ORDNER / f"{kategorie}.jsonl.gz"

    ausgabedatei = (
        PROCESSED_ORDNER
        / f"{kategorie}_enriched_{STICHPROBENGROESSE}_seed{SEED}.jsonl"
    )

    print(f"\nBearbeite Kategorie: {kategorie}")

    # Die ausgewählten Rezensionen anhand ihrer Originalzeilen indexieren.
    # Dadurch müssen wir nur die tatsächlich benötigten Zeilen ergänzen.
    ausgewaehlte_zeilen = {}

    with eingabedatei.open("r", encoding="utf-8") as datei:
        for zeile in datei:
            rezension = json.loads(zeile)
            zeilennummer = int(rezension["source_line"])

            if zeilennummer in ausgewaehlte_zeilen:
                raise ValueError(
                    f"Zeilennummer {zeilennummer} mehrfach in der Stichprobe."
                )

            ausgewaehlte_zeilen[zeilennummer] = rezension

    print(f"Ausgewählte Rezensionen: {len(ausgewaehlte_zeilen):,}")

    if len(ausgewaehlte_zeilen) != STICHPROBENGROESSE:
        raise ValueError("Die Stichprobengröße stimmt nicht.")

    # Die Originaldatei sequenziell lesen.
    # Die Zeilennummer muss mit der Zählweise des Sampling-Skripts
    # übereinstimmen: Hier beginnt die Zählung bei 1.
    gefunden = 0

    with gzip.open(originaldatei, "rt", encoding="utf-8") as datei:
        for zeilennummer, zeile in enumerate(datei, start=1):

            if zeilennummer not in ausgewaehlte_zeilen:
                continue

            original = json.loads(zeile)
            rezension = ausgewaehlte_zeilen[zeilennummer]

            # Vor dem Ergänzen prüfen, ob die richtige Rezension
            # anhand der gespeicherten Zeilennummer gefunden wurde.
            if (
                original.get("text") != rezension.get("text")
                or original.get("rating") != rezension.get("rating")
            ):
                raise ValueError(
                    f"Abweichende Rezension in {kategorie}, "
                    f"Originalzeile {zeilennummer}. "
                    "Bitte die Zählweise von source_line prüfen."
                )

            rezension["user_id"] = original.get("user_id")
            rezension["parent_asin"] = original.get("parent_asin")

            gefunden += 1

            if gefunden == len(ausgewaehlte_zeilen):
                break

    if gefunden != STICHPROBENGROESSE:
        raise ValueError(
            f"Nur {gefunden:,} von {STICHPROBENGROESSE:,} "
            "Rezensionen wiedergefunden."
        )

    # Prüfen, ob die Metadaten vollständig vorhanden sind.
    fehlende_user_ids = sum(
        not rezension.get("user_id")
        for rezension in ausgewaehlte_zeilen.values()
    )

    fehlende_asins = sum(
        not rezension.get("parent_asin")
        for rezension in ausgewaehlte_zeilen.values()
    )

    # Erst nach erfolgreicher Zuordnung die neue Datei schreiben.
    PROCESSED_ORDNER.mkdir(parents=True, exist_ok=True)

    with ausgabedatei.open("w", encoding="utf-8") as datei:
        for rezension in ausgewaehlte_zeilen.values():
            datei.write(
                json.dumps(rezension, ensure_ascii=False) + "\n"
            )

    print(f"Wiedergefundene Rezensionen: {gefunden:,}")
    print(f"Fehlende User-IDs: {fehlende_user_ids:,}")
    print(f"Fehlende Produkt-ASINs: {fehlende_asins:,}")
    print(f"Gespeichert: {ausgabedatei.name}")


def main():
    for kategorie in KATEGORIEN:
        metadaten_ergaenzen(kategorie)

    print("\nMetadaten-Ergänzung abgeschlossen.")


if __name__ == "__main__":
    main()