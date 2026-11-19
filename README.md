# Amazon Reviews – NLP-Sentimentanalyse

## Projekt

Vergleich zweier NLP-Verfahren zur Sentimentklassifikation englischsprachiger Amazon-Produktrezensionen:

- **TF-IDF + logistische Regression** – klassisches Machine Learning
- **DistilBERT** – vortrainiertes Transformer-Modell mit Fine-Tuning

Datengrundlage sind **149.992 Rezensionen** aus den Kategorien Electronics, Sports and Outdoors und Video Games des Datensatzes [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).

Untersucht werden zwei Klassifikationsaufgaben:

- **5 Klassen:** Sternebewertungen von 1 bis 5
- **3 Klassen:** Negativ (1–2 Sterne), neutral (3 Sterne), positiv (4–5 Sterne)

Die Daten wurden im Verhältnis 80/10/10 in Training, Validierung und Test aufgeteilt (Seed 42). Als zentrale Bewertungsmetrik dient der Macro-F1-Score.

## Ergebnisse

| Modell | Klassen | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| TF-IDF + LogReg (E21) | 5 | 77,28 % | 0,5664 |
| **DistilBERT (E19)** | **5** | **80,59 %** | **0,6105** |
| TF-IDF + LogReg (E20) | 3 | 89,57 % | 0,7317 |
| **DistilBERT (E20)** | **3** | **91,65 %** | **0,7498** |

*Alle Ergebnisse basieren auf dem unabhängigen Testsatz.*

Für die Drei-Klassen-Auswertung wurde TF-IDF separat trainiert (E18). Bei DistilBERT wurden die Vorhersagen des Fünf-Klassen-Modells auf drei Polaritätsklassen abgebildet. Ein separates Drei-Klassen-DistilBERT-Modell wurde nicht trainiert.

## Finale Modelle

Die Modellartefakte werden lokal unter `models/final_models/` gespeichert:

- `E09B_tfidf_5class.joblib` – TF-IDF + LogReg, 5 Klassen
- `E16_distilbert_5class/` – DistilBERT, 5 Klassen
- `E18_tfidf_polarity_3class.joblib` – TF-IDF + LogReg, 3 Klassen

Die Modellgewichte sind aufgrund ihrer Größe nicht Bestandteil des GitHub-Repositories.

## Projektstruktur

- `data/` – Rohdaten und aufbereitete Datensätze (nicht im Repository)
- `scripts/` – Datenaufbereitung D01–D07, Experimente E01–E21 und Vorhersagen
- `models/` – lokal gespeicherte Modellartefakte
- `reports/` – Evaluationsergebnisse und Abbildungen
- `tests/` – automatisierte Tests

## Installation und Anwendung

Python-Umgebung erstellen und Abhängigkeiten installieren:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Vorhersageskript starten:

```bash
python scripts/predict_sentiment.py
```

Für Vorhersagen müssen die zuvor trainierten Modellartefakte unter `models/final_models/` vorhanden sein. Die Modelle wurden ausschließlich mit englischsprachigen Rezensionen trainiert.

## Dokumentation

Eine Übersicht aller Datenaufbereitungsschritte und Experimente befindet sich in [`EXPERIMENT_OVERVIEW.md`](EXPERIMENT_OVERVIEW.md).

Die vollständige Methodik, Evaluation und Interpretation sind im IU-Projektbericht dokumentiert.
