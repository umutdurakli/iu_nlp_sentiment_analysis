
#!/usr/bin/env python3
"""Erstellt stratifizierte Zufallsstichproben aus Amazon-Rezensionen.

Beim ersten Aufruf werden gültige Bewertungen pro Sterneklasse gezählt und
zwischengespeichert. Bei späteren Aufrufen mit unveränderter Quelldatei entfällt
dieser Zähldurchlauf. Die Zufallsauswahl liest weiterhin die gesamte Quelldatei.
"""

import argparse
import gzip
import json
import os
import random
from collections import Counter
from pathlib import Path

# Die drei Produktkategorien, die wir für unser NLP-Projekt untersuchen.
KATEGORIEN = ("Sports_and_Outdoors", "Video_Games", "Electronics")

# Unser Klassifikationsproblem umfasst fünf mögliche Sternebewertungen.
STERNEKLASSEN = (1, 2, 3, 4, 5)

# Nach jeder Million gelesener Zeilen erscheint eine Fortschrittsmeldung.
FORTSCHRITTSINTERVALL = 1_000_000

# Wenn wir die Filterregeln verändern, erhöhen wir diese Versionsnummer.
# Dadurch werden alte zwischengespeicherte Häufigkeiten nicht mehr verwendet.
FILTERVERSION = 1


def lese_rezensionen(dateipfad, phase):
    """Liest gültige Rezensionen zeilenweise und meldet den Fortschritt."""

    # gzip.open öffnet die komprimierte Amazon-Datei.
    # Wir müssen die Datei dafür nicht vorher entpacken.
    with gzip.open(dateipfad, "rt", encoding="utf-8") as datei:

        # enumerate nummeriert jede eingelesene Zeile ab 1.
        for zeilennummer, zeile in enumerate(datei, 1):

            # Der Modulo-Operator (%) prüft, ob eine volle Million erreicht ist.
            if zeilennummer % FORTSCHRITTSINTERVALL == 0:
                print(
                    f"  {phase}: {zeilennummer:,} Zeilen gelesen ...",
                    flush=True,
                )

            # Jede Zeile enthält ein JSON-Objekt mit den Rezensionsdaten.
            try:
                datensatz = json.loads(zeile)
            except json.JSONDecodeError:
                continue

            # Wir verarbeiten ausschließlich JSON-Objekte.
            if not isinstance(datensatz, dict):
                continue

            # Die Sternebewertung muss eine gültige Zahl sein.
            try:
                sterne = float(datensatz.get("rating"))
            except (TypeError, ValueError, OverflowError):
                continue

            # Nur ganzzahlige Bewertungen von 1 bis 5 sind erlaubt.
            if not sterne.is_integer() or int(sterne) not in STERNEKLASSEN:
                continue

            # Rezensionen ohne gültigen Text werden ausgeschlossen.
            text = datensatz.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            # yield gibt die Rezension zurück, ohne alle Daten im
            # Arbeitsspeicher zu sammeln.
            yield zeilennummer, int(sterne), datensatz


def berechne_quoten(haeufigkeiten, stichprobengroesse):
    """Berechnet die benötigte Anzahl pro Sterneklasse."""

    # Gesamtzahl aller gültigen Rezensionen.
    gesamtzahl = sum(haeufigkeiten.values())

    # Wir dürfen nicht mehr Rezensionen auswählen, als vorhanden sind.
    if gesamtzahl < stichprobengroesse:
        raise ValueError(
            f"Nur {gesamtzahl} gültige Rezensionen "
            f"für {stichprobengroesse} Plätze."
        )

    # Berechnung der proportionalen Stichprobengröße je Sterneklasse.
    # Beispiel: 50.000 * 0,65 = 32.500 Rezensionen mit fünf Sternen.
    genaue_quoten = {
        sterne: stichprobengroesse * haeufigkeiten[sterne] / gesamtzahl
        for sterne in STERNEKLASSEN
    }

    # Zunächst werden die berechneten Werte auf ganze Zahlen abgerundet.
    quoten = {
        sterne: int(genaue_quoten[sterne])
        for sterne in STERNEKLASSEN
    }

    # Durch das Abrunden können einige Plätze übrig bleiben.
    rest = stichprobengroesse - sum(quoten.values())

    # Die verbleibenden Plätze erhalten die Klassen mit den größten
    # Nachkommarestwerten. So stimmt die Gesamtzahl am Ende genau.
    reihenfolge = sorted(
        STERNEKLASSEN,
        key=lambda sterne: (
            -(genaue_quoten[sterne] - quoten[sterne]),
            sterne,
        ),
    )

    for sterne in reihenfolge[:rest]:
        quoten[sterne] += 1

    return quoten


def lade_oder_zaehle(dateipfad, kategorie, cache_ordner):
    """Lädt gespeicherte Häufigkeiten oder zählt die Rezensionen neu."""

    # Ordner für zwischengespeicherte Klassenhäufigkeiten anlegen.
    cache_ordner.mkdir(parents=True, exist_ok=True)

    cache_datei = (
        cache_ordner / f"{kategorie}_klassenhaeufigkeiten.json"
    )

    # Informationen zur Quelldatei für die Gültigkeitsprüfung des Caches.
    dateistatus = dateipfad.stat()

    kennung = {
        "dateipfad": str(dateipfad.resolve()),
        "dateigroesse": dateistatus.st_size,
        "aenderungszeit_ns": dateistatus.st_mtime_ns,
        "filterversion": FILTERVERSION,
    }

    # Falls ein Cache existiert, prüfen wir, ob er noch gültig ist.
    if cache_datei.exists():
        try:
            with cache_datei.open("r", encoding="utf-8") as datei:
                cache = json.load(datei)

            if cache.get("kennung") == kennung:
                werte = cache["haeufigkeiten"]

                # Alle fünf Sterneklassen müssen gültige Zählwerte besitzen.
                if set(werte) == {
                    str(s) for s in STERNEKLASSEN
                } and all(
                    type(werte[str(s)]) is int
                    and werte[str(s)] >= 0
                    for s in STERNEKLASSEN
                ):
                    print(
                        "  Gespeicherte Klassenhäufigkeiten verwendet.",
                        flush=True,
                    )

                    return Counter({
                        s: werte[str(s)]
                        for s in STERNEKLASSEN
                    })

        except (OSError, ValueError, KeyError, TypeError):
            pass

        print(
            "  Zwischenspeicher veraltet oder ungültig; zähle neu.",
            flush=True,
        )

    # Falls kein gültiger Cache existiert, zählen wir alle Rezensionen.
    print(
        "  Durchlauf 1/2: Gültige Rezensionen zählen ...",
        flush=True,
    )

    haeufigkeiten = Counter({
        s: 0 for s in STERNEKLASSEN
    })

    for _, sterne, _ in lese_rezensionen(dateipfad, "Zählung"):
        haeufigkeiten[sterne] += 1

    # Ergebnisse zunächst in eine temporäre Datei schreiben.
    # So vermeiden wir einen unvollständigen Cache bei einem Abbruch.
    temp_datei = cache_datei.with_suffix(".json.tmp")

    with temp_datei.open("w", encoding="utf-8") as datei:
        json.dump(
            {
                "kennung": kennung,
                "haeufigkeiten": dict(haeufigkeiten),
            },
            datei,
            indent=2,
        )

    # Temporäre Datei durch die endgültige Cache-Datei ersetzen.
    os.replace(temp_datei, cache_datei)

    print(
        f"  Klassenhäufigkeiten gespeichert: {cache_datei}",
        flush=True,
    )

    return haeufigkeiten


def waehle_stichprobe(
    dateipfad,
    kategorie,
    stichprobengroesse,
    seed,
    cache_ordner,
):
    """Wählt zufällige Rezensionen je Sterneklasse ohne Zurücklegen aus."""

    # Häufigkeiten laden oder bei Bedarf neu ermitteln.
    haeufigkeiten = lade_oder_zaehle(
        dateipfad,
        kategorie,
        cache_ordner,
    )

    # Ermitteln, wie viele Rezensionen je Sterneklasse benötigt werden.
    quoten = berechne_quoten(
        haeufigkeiten,
        stichprobengroesse,
    )

    print(
        f"  Gültige Rezensionen: {sum(haeufigkeiten.values()):,}; "
        f"Auswahl pro Sterneklasse: {dict(quoten)}",
        flush=True,
    )

    # Der Seed sorgt für eine reproduzierbare Zufallsauswahl.
    # Jede Kategorie erhält eine eigene Zufallszahlenfolge.
    zufall = random.Random(f"{seed}:{kategorie}")

    # Zähler für bereits betrachtete Rezensionen jeder Sterneklasse.
    gesehen = Counter({
        s: 0 for s in STERNEKLASSEN
    })

    # Reservoir: Hier speichern wir nur die ausgewählten Rezensionen.
    speicher = {
        sterne: [] for sterne in STERNEKLASSEN
    }

    print(
        "  Zufallsauswahl: gesamte Datei wird gelesen ...",
        flush=True,
    )

    # Zweiter vollständiger Lesedurchlauf.
    for zeilennummer, sterne, datensatz in lese_rezensionen(
        dateipfad,
        "Auswahl",
    ):

        gesehen[sterne] += 1
        zielanzahl = quoten[sterne]

        if zielanzahl == 0:
            continue

        # Wir übernehmen nur die Felder, die für unser Projekt
        # beziehungsweise die spätere Nachvollziehbarkeit relevant sind.
        eintrag = {
            "category": kategorie,
            "rating": sterne,
            "title": datensatz.get("title") or "",
            "text": datensatz["text"],
            "source_line": zeilennummer,
        }

        klasse = speicher[sterne]

        # Solange noch Plätze frei sind, übernehmen wir Rezensionen direkt.
        if len(klasse) < zielanzahl:
            klasse.append(eintrag)

        else:
            # Reservoir Sampling:
            # Jede neu gelesene Rezension erhält eine faire Chance,
            # eine bereits ausgewählte Rezension zu ersetzen.
            zufallsindex = zufall.randrange(gesehen[sterne])

            if zufallsindex < zielanzahl:
                klasse[zufallsindex] = eintrag

    # Sicherheitsprüfung: Stimmen die tatsächlichen Häufigkeiten
    # mit den zuvor gezählten oder gespeicherten Häufigkeiten überein?
    if gesehen != haeufigkeiten:
        raise RuntimeError(
            "Die gezählten Häufigkeiten stimmen nicht mit dem aktuellen "
            "Datensatz überein. Lösche die Cache-Datei dieser Kategorie "
            "und starte erneut."
        )

    # Ausgewählte Rezensionen aller fünf Sterneklassen zusammenführen.
    auswahl = [
        eintrag
        for sterne in STERNEKLASSEN
        for eintrag in speicher[sterne]
    ]

    if len(auswahl) != stichprobengroesse:
        raise RuntimeError(
            "Unerwartete Anzahl ausgewählter Rezensionen."
        )

    # Die Reihenfolge der ausgewählten Rezensionen zufällig mischen.
    zufall.shuffle(auswahl)

    return auswahl


def main():
    """Verarbeitet die drei Kategorien und speichert die Stichproben."""

    # Kommandozeilenparameter definieren.
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--size",
        type=int,
        default=50_000,
        help="Anzahl der Rezensionen pro Kategorie",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Startwert für die reproduzierbare Zufallsauswahl",
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim"),
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/interim/cache"),
    )

    argumente = parser.parse_args()

    if argumente.size <= 0:
        parser.error("--size muss größer als null sein.")

    # Pfade zu unseren drei Amazon-Quelldateien zusammensetzen.
    dateipfade = {
        kategorie: argumente.raw_dir / f"{kategorie}.jsonl.gz"
        for kategorie in KATEGORIEN
    }

    # Vor dem Start prüfen, ob alle Quelldateien vorhanden sind.
    fehlende = [
        str(pfad)
        for pfad in dateipfade.values()
        if not pfad.is_file()
    ]

    if fehlende:
        parser.error(
            "Eingabedateien fehlen: " + ", ".join(fehlende)
        )

    argumente.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Jede Produktkategorie einzeln verarbeiten.
    for kategorie, dateipfad in dateipfade.items():

        print(
            f"\nKategorie: {kategorie}",
            flush=True,
        )

        auswahl = waehle_stichprobe(
            dateipfad,
            kategorie,
            argumente.size,
            argumente.seed,
            argumente.cache_dir,
        )

        # Eindeutiger Dateiname mit Stichprobengröße und Seed.
        ausgabedatei = argumente.output_dir / (
            f"{kategorie}_stratified_"
            f"{argumente.size}_seed{argumente.seed}.jsonl"
        )

        # Zunächst temporär speichern, damit bei einem Abbruch
        # keine unvollständige endgültige Ergebnisdatei entsteht.
        temp_datei = ausgabedatei.with_suffix(".jsonl.tmp")

        with temp_datei.open("w", encoding="utf-8") as datei:
            for eintrag in auswahl:
                datei.write(
                    json.dumps(
                        eintrag,
                        ensure_ascii=False,
                    ) + "\n"
                )

        os.replace(temp_datei, ausgabedatei)

        print(
            f"  Gespeichert: {len(auswahl):,} Rezensionen "
            f"in {ausgabedatei}",
            flush=True,
        )

    print(
        "\nFertig. Originaldateien unverändert.",
        flush=True,
    )


if __name__ == "__main__":
    main()
