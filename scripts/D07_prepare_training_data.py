"""Bereitet lokale JSONL-Rezensionen für eine Fünfklassen-Klassifikation vor.

Aufruf vom Projektordner: ./.venv/bin/python scripts/D07_prepare_training_data.py
Die Quelldateien bleiben unverändert. Ein vorhandener Ausgabeordner führt zum
Abbruch. Es werden keine Vektoren berechnet und keine Modelle trainiert.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import unicodedata

KATEGORIEN = ("Electronics", "Sports_and_Outdoors", "Video_Games")
SPLITS = ("train", "validation", "test")
ANTEILE = (0.8, 0.1, 0.1)
PROJEKT = Path(__file__).resolve().parent.parent


@dataclass
class Rezension:
    """Bewahrt Originalinhalt und eine eindeutige physische Eingabeposition."""

    daten: dict
    herkunft: tuple
    normaltext: str


def normalisiere_text(text):
    """Vereinheitlicht Unicode und Leerraum; erhält Schreibweise und Satzzeichen."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def pruefe_datensatz(daten, kategorie, ort):
    """Prüft Pflichtfelder und Typen; ungültige Zeilen werden nicht übersprungen."""
    pflicht = {"category", "rating", "title", "text", "source_line", "user_id", "parent_asin"}
    if not isinstance(daten, dict) or not pflicht.issubset(daten):
        raise ValueError(f"{ort}: Kein Objekt oder fehlende Pflichtfelder.")
    if daten["category"] != kategorie:
        raise ValueError(f"{ort}: Kategorie passt nicht zur Quelldatei.")
    rating = daten["rating"]
    if type(rating) not in (int, float) or not math.isfinite(rating) or rating not in range(1, 6):
        raise ValueError(f"{ort}: rating muss eine ganze Sterneklasse von 1 bis 5 sein.")
    if type(daten["source_line"]) is not int or daten["source_line"] < 1:
        raise ValueError(f"{ort}: source_line muss eine positive ganze Zahl sein.")
    for feld in ("title", "text", "user_id", "parent_asin"):
        if not isinstance(daten[feld], str):
            raise ValueError(f"{ort}: {feld} muss eine Zeichenkette sein.")
        if feld != "title" and not daten[feld].strip():
            raise ValueError(f"{ort}: {feld} ist leer.")
    if not normalisiere_text(daten["text"]):
        raise ValueError(f"{ort}: Normalisierter Rezensionstext ist leer.")


def lies_daten(eingabe):
    """Liest die drei fest benannten Quellen und berechnet SHA-256-Nachweise."""
    rezensionen, quellen = [], []
    for kategorie in KATEGORIEN:
        pfad = eingabe / f"{kategorie}_enriched_50000_seed42.jsonl"
        sha, anzahl = hashlib.sha256(), 0
        with pfad.open("rb") as datei:
            for nummer, roh in enumerate(datei, 1):
                sha.update(roh)
                ort = f"{pfad.name}:{nummer}"
                try:
                    daten = json.loads(roh.decode("utf-8"))
                except (ValueError, UnicodeError) as fehler:
                    raise ValueError(f"{ort}: Ungültiges UTF-8/JSON.") from fehler
                pruefe_datensatz(daten, kategorie, ort)
                rezensionen.append(Rezension(daten, (pfad.name, nummer), normalisiere_text(daten["text"])))
                anzahl += 1
        quellen.append({"datei": pfad.name, "anzahl": anzahl, "sha256": sha.hexdigest()})
    if not rezensionen:
        raise ValueError("Keine Rezensionen eingelesen.")
    return rezensionen, quellen


def referenz(rezension):
    """Liefert eine lokale Fundstelle, ohne den Rezensionstext zu protokollieren."""
    return {"datei": rezension.herkunft[0], "jsonl_zeile": rezension.herkunft[1],
            "source_line": rezension.daten["source_line"], "rating": rezension.daten["rating"]}


def entferne_duplikate(rezensionen):
    """Entfernt nur inhaltlich identische Datensätze; erhält Bewertungskonflikte.

    source_line ist Herkunft, kein Inhaltsmerkmal. Alle übrigen Felder müssen
    gleich sein; lediglich text wird für den Vergleich normalisiert. Auch
    abweichende Titel oder zusätzliche Metadaten verhindern eine Entfernung.
    """
    behalten, entfernt, konflikte = [], [], []
    gruppen = defaultdict(list)
    for rezension in rezensionen:
        d = rezension.daten
        gruppen[(d["category"], d["user_id"], d["parent_asin"], rezension.normaltext)].append(rezension)
    for gruppe in gruppen.values():
        if len({r.daten["rating"] for r in gruppe}) > 1:
            konflikte.append({"vorkommen": [referenz(r) for r in gruppe]})
        gesehen = {}
        for r in gruppe:
            inhalt = {k: v for k, v in r.daten.items() if k != "source_line"}
            inhalt["text"] = r.normaltext
            # JSON-Zahlen 5 und 5.0 bezeichnen dieselbe validierte Sterneklasse.
            inhalt["rating"] = int(inhalt["rating"])
            schluessel = json.dumps(inhalt, sort_keys=True, ensure_ascii=False, allow_nan=False)
            if schluessel in gesehen:
                entfernt.append({"entfernt": referenz(r), "behalten": referenz(gesehen[schluessel])})
            else:
                gesehen[schluessel] = r
                behalten.append(r)
    # Physische Eingabereihenfolge ist die stabile Ausgangsreihenfolge.
    ordnung = {r.herkunft: i for i, r in enumerate(rezensionen)}
    behalten.sort(key=lambda r: ordnung[r.herkunft])
    return behalten, entfernt, konflikte


def stratum(r):
    """Ordnet eine Rezension ihrer Kategorie-Sterne-Kombination zu."""
    return (r.daten["category"], int(r.daten["rating"]))


def erstelle_split(rezensionen, seed):
    """Teilt unteilbare Textgruppen durch deterministische Fehler-Minimierung.

    Zielfunktion: Summe (Ist-Soll)^2 / max(Soll, 1), sowohl über alle
    Kategorie-Sterne-Zellen als auch über die drei Gesamtgrößen. Große Gruppen
    kommen zuerst; Seed entscheidet Gleichstände. Zwei lokale Durchläufe
    verschieben nur vollständige Gruppen bei streng sinkendem Fehler.
    Das ist eine Heuristik, keine Garantie für ein globales Optimum.
    """
    rng = random.Random(seed)
    texte = defaultdict(list)
    for i, r in enumerate(rezensionen):
        texte[r.normaltext].append(i)
    gruppen = list(texte.values())
    rng.shuffle(gruppen)
    gruppen.sort(key=len, reverse=True)
    gesamt = Counter(stratum(r) for r in rezensionen)
    ziele = [{k: a * n for k, n in gesamt.items()} for a in ANTEILE]
    zielgroessen = [a * len(rezensionen) for a in ANTEILE]
    zaehler = [Counter() for _ in SPLITS]
    groessen = [0, 0, 0]
    vektoren = [Counter(stratum(rezensionen[i]) for i in g) for g in gruppen]
    reihenfolge = list(range(3))
    rng.shuffle(reihenfolge)

    def delta(s, vektor, groesse, vorzeichen):
        """Berechnet nur die Änderung der betroffenen Zielfunktionsanteile."""
        ergebnis = 0.0
        for k, n in vektor.items():
            diff = zaehler[s][k] - ziele[s][k]
            schritt = vorzeichen * n
            ergebnis += (2 * diff * schritt + schritt * schritt) / max(ziele[s][k], 1)
        diff = groessen[s] - zielgroessen[s]
        schritt = vorzeichen * groesse
        return ergebnis + (2 * diff * schritt + schritt * schritt) / max(zielgroessen[s], 1)

    def buchen(s, vektor, groesse, vorzeichen):
        for k, n in vektor.items():
            zaehler[s][k] += vorzeichen * n
        groessen[s] += vorzeichen * groesse

    zuordnung = []
    for gruppe, vektor in zip(gruppen, vektoren):
        s = min(reihenfolge, key=lambda s: delta(s, vektor, len(gruppe), 1))
        buchen(s, vektor, len(gruppe), 1)
        zuordnung.append(s)
    verschiebungen = []
    for _ in range(2):
        anzahl = 0
        for j, (gruppe, vektor) in enumerate(zip(gruppen, vektoren)):
            alt = zuordnung[j]
            entfernen = delta(alt, vektor, len(gruppe), -1)
            neu = min(reihenfolge, key=lambda s: 0 if s == alt else entfernen + delta(s, vektor, len(gruppe), 1))
            if neu != alt and entfernen + delta(neu, vektor, len(gruppe), 1) < -1e-10:
                buchen(alt, vektor, len(gruppe), -1)
                buchen(neu, vektor, len(gruppe), 1)
                zuordnung[j] = neu
                anzahl += 1
        verschiebungen.append(anzahl)
    ergebnis = [[] for _ in SPLITS]
    for gruppe, s in zip(gruppen, zuordnung):
        ergebnis[s].extend(gruppe)
    for indizes in ergebnis:
        rng.shuffle(indizes)
    return ergebnis, {"textgruppen": len(gruppen), "groesste_textgruppe": max(map(len, gruppen)),
                      "lokale_verschiebungen": verschiebungen}


def pruefe_zuordnung(original, behalten, entfernt, splits):
    """Prüft disjunkte Textmengen sowie vollständige Zuordnung und Duplikatbilanz."""
    indizes = [i for split in splits for i in split]
    if Counter(indizes) != Counter(range(len(behalten))):
        raise ValueError("Nicht jede beibehaltene Rezension ist genau einmal zugeordnet.")
    textmengen = [{behalten[i].normaltext for i in split} for split in splits]
    ueberschneidungen = {f"{SPLITS[a]}__{SPLITS[b]}": len(textmengen[a] & textmengen[b])
                        for a, b in ((0, 1), (0, 2), (1, 2))}
    if any(ueberschneidungen.values()):
        raise ValueError("Identische normalisierte Texte in mehreren Splits.")
    ursprung = Counter(r.herkunft for r in original)
    bilanz = Counter(r.herkunft for r in behalten)
    behalten_orte = set(bilanz)
    for eintrag in entfernt:
        ref = eintrag["entfernt"]
        bilanz[(ref["datei"], ref["jsonl_zeile"])] += 1
        ref = eintrag["behalten"]
        if (ref["datei"], ref["jsonl_zeile"]) not in behalten_orte:
            raise ValueError("Duplikat verweist nicht auf eine beibehaltene Rezension.")
    if ursprung != bilanz:
        raise ValueError("Eingabebilanz aus behaltenen und entfernten Vorkommen stimmt nicht.")
    return {"jede_beibehaltene_rezension_genau_einmal": True,
            "jede_eingabe_behalten_oder_protokolliert_entfernt": True,
            "textueberschneidungen": ueberschneidungen}


def statistik(rezensionen, splits):
    """Dokumentiert Ist- und Sollwerte für Größen und die gemeinsame Verteilung."""
    gesamt = Counter(stratum(r) for r in rezensionen)
    bericht = {}
    for name, anteil, indizes in zip(SPLITS, ANTEILE, splits):
        joint = Counter(stratum(rezensionen[i]) for i in indizes)
        zellen = []
        for k in KATEGORIEN:
            for rating in range(1, 6):
                n, ist = gesamt[(k, rating)], joint[(k, rating)]
                zellen.append({"category": k, "rating": rating, "gesamt": n,
                               "soll": anteil * n, "ist": ist, "abweichung_anzahl": ist - anteil * n,
                               "abweichung_anteil_prozentpunkte": 100 * (ist / n - anteil) if n else None})
        bericht[name] = {"anzahl": len(indizes), "soll": anteil * len(rezensionen),
                         "abweichung_anzahl": len(indizes) - anteil * len(rezensionen),
                         "kategorien": {k: sum(joint[(k, r)] for r in range(1, 6)) for k in KATEGORIEN},
                         "sterne": {str(r): sum(joint[(k, r)] for k in KATEGORIEN) for r in range(1, 6)},
                         "kategorie_sterne": zellen}
    return bericht


def speichere(ausgabe, rezensionen, splits, bericht, entfernt, konflikte):
    """Prüft geschriebene Dateien vor Veröffentlichung; überschreibt keine Ausgabe."""
    ausgabe.parent.mkdir(parents=True, exist_ok=True)
    # Exklusives mkdir reserviert einen neuen Ordner; vorhandene bleiben unberührt.
    ausgabe.mkdir(exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix=".split-staging-", dir=ausgabe) as tmp:
            staging = Path(tmp)
            for name, indizes in zip(SPLITS, splits):
                pfad = staging / f"{name}.jsonl"
                with pfad.open("x", encoding="utf-8") as datei:
                    for i in indizes:
                        datei.write(json.dumps(rezensionen[i].daten, ensure_ascii=False, allow_nan=False) + "\n")
                # Vergleich jeder Ausgabezeile mit dem zugewiesenen Originalobjekt.
                with pfad.open(encoding="utf-8") as datei:
                    zeilen = [json.loads(zeile) for zeile in datei]
                if zeilen != [rezensionen[i].daten for i in indizes]:
                    raise ValueError(f"Ausgabeprüfung fehlgeschlagen: {name}")
            for name, eintraege in (("removed_duplicates", entfernt), ("rating_conflicts", konflikte)):
                with (staging / f"{name}.jsonl").open("x", encoding="utf-8") as datei:
                    for eintrag in eintraege:
                        datei.write(json.dumps(eintrag, ensure_ascii=False) + "\n")
            bericht["pruefungen"]["ausgabezeilen_mit_originalen_abgeglichen"] = True
            for name in SPLITS:
                pfad = staging / f"{name}.jsonl"
                bericht["splits"][name]["sha256"] = hashlib.sha256(pfad.read_bytes()).hexdigest()
            (staging / "split_report.json").write_text(json.dumps(bericht, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for pfad in staging.iterdir():
                # Hardlinks schlagen bei existierenden Namen fehl, statt zu überschreiben.
                (ausgabe / pfad.name).hardlink_to(pfad)
    except Exception:
        # Nur den in diesem Aufruf selbst angelegten Ausgabeordner entfernen.
        shutil.rmtree(ausgabe)
        raise


def vorbereiten(eingabe, ausgabe, seed=42):
    """Führt Validierung, Bereinigung, Gruppensplit und überprüfte Ausgabe aus."""
    if ausgabe.exists():
        raise FileExistsError(f"Ausgabeordner existiert bereits: {ausgabe}; kein Überschreiben.")
    original, quellen = lies_daten(eingabe)
    behalten, entfernt, konflikte = entferne_duplikate(original)
    splits, gruppeninfo = erstelle_split(behalten, seed)
    pruefungen = pruefe_zuordnung(original, behalten, entfernt, splits)
    bericht = {"seed": seed, "zielanteile": dict(zip(SPLITS, ANTEILE)), "quellen": quellen,
               "eingabe": len(original), "behalten": len(behalten), "entfernt": len(entfernt),
               "bewertungskonfliktgruppen": len(konflikte), "gruppen": gruppeninfo,
               "normalisierung": "Unicode NFC; Leerraum zusammenfassen; Schreibweise und Satzzeichen erhalten",
               "duplikatregel": "Alle Inhaltsfelder gleich, text normalisiert, rating als Sterneklasse; source_line ausgenommen",
               "methode": "Große Textgruppen zuerst; deterministische greedy Pearson-Fehlerminimierung über Kategorie/Sterne und Größen; zwei lokale Verschiebungsdurchläufe",
               "einschraenkungen": ["Keine garantierte exakte Stratifizierung oder global optimale Zuordnung.",
                                      "Nur normalisierte Textgleichheit geschützt; keine semantischen Duplikate.",
                                      "Nutzer und Produkte können in mehreren Splits vorkommen.",
                                      "Bewertungskonflikte bleiben erhalten; keine automatische Labelkorrektur."],
               "pruefungen": pruefungen, "splits": statistik(behalten, splits)}
    speichere(ausgabe, behalten, splits, bericht, entfernt, konflikte)
    print(f"Eingabe: {len(original)}; behalten: {len(behalten)}; entfernt: {len(entfernt)}; Konfliktgruppen: {len(konflikte)}")
    for name, werte in bericht["splits"].items():
        print(f"\n{name}: {werte['anzahl']} (Soll {werte['soll']:.1f}; Abweichung {werte['abweichung_anzahl']:+.1f})")
        print("Kategorien:", werte["kategorien"])
        print("Sterne:", werte["sterne"])
        maximum = max(abs(z["abweichung_anteil_prozentpunkte"]) for z in werte["kategorie_sterne"] if z["gesamt"])
        print(f"Maximale Abweichung je Kategorie-Sterne-Zelle: {maximum:.4f} Prozentpunkte")
    print("\nPrüfungen:", pruefungen)
    print("Bericht und lokale Duplikat-/Konfliktprotokolle:", ausgabe)
    return bericht


def main():
    """Erlaubt getrennte Testordner, ohne die Standardquellen zu verändern."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJEKT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=PROJEKT / "data/processed/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    vorbereiten(args.input_dir.resolve(), args.output_dir.resolve(), args.seed)


if __name__ == "__main__":
    main()
