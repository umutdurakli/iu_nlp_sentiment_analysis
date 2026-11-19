#!/usr/bin/env python3
"""Erste explorative Analyse von Amazon Reviews 2023 (speicherschonend).

Aufruf im Projektverzeichnis:
    python scripts/D02_eda_amazon_reviews.py
    python scripts/D02_eda_amazon_reviews.py --full
    python scripts/D02_eda_amazon_reviews.py --limit 10000

Erwartete Ablage: <Projekt>/data/raw/*<Kategorie>*.jsonl.gz
Ergebnisse: <Projekt>/reports/tables/ und <Projekt>/reports/figures/
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

CATEGORIES = ("Sports_and_Outdoors", "Video_Games", "Electronics")
DEFAULT_LIMIT = 100_000


def find_review_file(raw_dir: Path, category: str) -> Path:
    """Suche eindeutig nach dem komprimierten Rezensionsdatensatz einer Kategorie."""
    candidates = sorted(
        p for p in raw_dir.glob("*.jsonl.gz")
        if category.lower() in p.name.lower()
        and "meta" not in p.name.lower()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Keine Review-Datei für {category} in {raw_dir} gefunden. "
            "Erwartet wird eine Datei mit dem Kategorienamen und der Endung .jsonl.gz."
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(f"Mehrere Dateien für {category} gefunden: {names}. Bitte nur die passende Review-Datei ablegen.")
    return candidates[0]


def text_missing(value: object) -> bool:
    return not isinstance(value, str) or not value.strip()


def parse_rating(value: object) -> int | None:
    # bool ist in Python eine Unterklasse von int, aber keine Sternebewertung.
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    if number.is_integer() and number in (1, 2, 3, 4, 5):
        return int(number)
    return None


def analyse(path: Path, category: str, limit: int | None) -> tuple[dict, list[dict]]:
    counts: Counter[int] = Counter()
    total = 0
    invalid_json = 0
    invalid_rating = 0
    missing_title = 0
    missing_text = 0
    word_total = 0
    character_total = 0
    text_count = 0

    print(f"\n{category}  |  Datei: {path.name}", flush=True)
    with gzip.open(path, mode="rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if limit is not None and total >= limit:
                break
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("JSON-Zeile ist kein Objekt")
            except (json.JSONDecodeError, ValueError):
                invalid_json += 1
                continue

            if text_missing(item.get("title")):
                missing_title += 1
            text = item.get("text")
            if text_missing(text):
                missing_text += 1
            else:
                text_count += 1
                word_total += len(text.split())
                character_total += len(text)

            rating = parse_rating(item.get("rating"))
            if rating is None:
                invalid_rating += 1
            else:
                counts[rating] += 1

            if total % 1_000_000 == 0:
                print(f"  {total:,} Zeilen gelesen ...", flush=True)

    valid_ratings = sum(counts.values())
    summary = {
        "category": category,
        "rows_read": total,
        "valid_ratings": valid_ratings,
        "invalid_ratings": invalid_rating,
        "invalid_json": invalid_json,
        "missing_or_empty_title": missing_title,
        "missing_or_empty_text": missing_text,
        "mean_text_words_nonempty": round(word_total / text_count, 2) if text_count else 0,
        "mean_text_characters_nonempty": round(character_total / text_count, 2) if text_count else 0,
    }
    distribution = [
        {
            "category": category,
            "rating": rating,
            "count": counts[rating],
            "percent_of_valid_ratings": round(100 * counts[rating] / valid_ratings, 2) if valid_ratings else 0,
        }
        for rating in range(1, 6)
    ]
    print(f"  Gelesen: {total:,} | gültige Bewertungen: {valid_ratings:,} | ungültiges JSON: {invalid_json:,}")
    print(f"  Fehlende Titel: {missing_title:,} | fehlende Texte: {missing_text:,} | ungültige Bewertungen: {invalid_rating:,}")
    print("  Sterneverteilung: " + ", ".join(f"{r['rating']}★ = {r['count']:,} ({r['percent_of_valid_ratings']} %)" for r in distribution))
    return summary, distribution


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_chart(path: Path, distributions: list[dict], description: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib fehlt. Installiere die Pakete mit: python -m pip install -r requirements.txt") from exc

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.25
    for i, category in enumerate(CATEGORIES):
        rows = [r for r in distributions if r["category"] == category]
        x = [rating + (i - 1) * width for rating in range(1, 6)]
        y = [r["percent_of_valid_ratings"] for r in rows]
        ax.bar(x, y, width=width, label=category.replace("_", " "))
    ax.set_xticks(range(1, 6), [str(x) for x in range(1, 6)])
    ax.set_xlabel("Sternebewertung")
    ax.set_ylabel("Anteil gültiger Bewertungen (%)")
    ax.set_title(f"Amazon Reviews 2023: Sterneverteilung ({description})")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA der drei Amazon-Rezensionskategorien")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Alle Datensätze vollständig durchlaufen")
    mode.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximale gelesene Zeilen je Kategorie (Standard: 100000)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit muss mindestens 1 sein")

    project_dir = Path(__file__).resolve().parent.parent
    raw_dir = project_dir / "data" / "raw"
    table_dir = project_dir / "reports" / "tables"
    figure_dir = project_dir / "reports" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    limit = None if args.full else args.limit
    label = "full" if args.full else f"preview_first_{limit}"
    description = "vollständige Daten" if args.full else f"erste {limit:,} Zeilen je Kategorie"
    print(f"Analyse: {description}. Originaldateien bleiben unverändert.")
    if not args.full:
        print("Hinweis: Die ersten Zeilen sind keine repräsentative Zufallsstichprobe.")

    summaries: list[dict] = []
    distributions: list[dict] = []
    for category in CATEGORIES:
        file_path = find_review_file(raw_dir, category)
        summary, distribution = analyse(file_path, category, limit)
        summaries.append(summary)
        distributions.extend(distribution)

    summary_path = table_dir / f"eda_summary_{label}.csv"
    distribution_path = table_dir / f"rating_distribution_{label}.csv"
    chart_path = figure_dir / f"rating_distribution_{label}.png"
    write_csv(summary_path, summaries)
    write_csv(distribution_path, distributions)
    write_chart(chart_path, distributions, description)
    print("\nErgebnisdateien:")
    for path in (summary_path, distribution_path, chart_path):
        print(" ", path.relative_to(project_dir))


if __name__ == "__main__":
    main()
