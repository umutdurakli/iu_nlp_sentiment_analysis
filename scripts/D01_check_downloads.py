"""Zeigt heruntergeladene Dateien und prüft die erste JSONL-Zeile.

Liest NICHT die vollständigen, möglicherweise sehr großen Dateien ein.
Aufrufen: python scripts/D01_check_downloads.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = ("Sports_and_Outdoors", "Video_Games", "Electronics")


def main() -> None:
    files = sorted(path for path in RAW.iterdir() if path.is_file() and not path.name.startswith("."))
    if not files:
        print(f"Noch keine Downloads in {RAW}")
        return

    for path in files:
        size_gb = path.stat().st_size / (1024**3)
        matched = [category for category in EXPECTED if category.lower() in path.name.lower()]
        print(f"\n{path.name} ({size_gb:.2f} GiB) | Kategorie: {', '.join(matched) or 'unbekannt'}")
        if not path.name.endswith((".jsonl.gz", ".json.gz")):
            print("  Kein komprimiertes JSONL; nur Dateiname/Größe geprüft.")
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                first = stream.readline()
            record = json.loads(first)
            print("  JSON-Felder:", ", ".join(sorted(record.keys())))
            print("  title vorhanden:", "title" in record)
            print("  text vorhanden:", "text" in record)
            print("  rating vorhanden:", "rating" in record)
        except (OSError, EOFError, json.JSONDecodeError, UnicodeError) as error:
            print(f"  ACHTUNG: erste Zeile nicht lesbar ({type(error).__name__}: {error})")
            print("  Möglicherweise ist die Datei noch nicht vollständig heruntergeladen.")

    print("\nHinweis: Diese Kurzprüfung garantiert NICHT die vollständige Dateiintegrität.")


if __name__ == "__main__":
    main()
