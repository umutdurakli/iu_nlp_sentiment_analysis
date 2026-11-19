# Experimentübersicht und Datenaufbereitung – IU NLP

**Projekt:** Sentimentanalyse von Amazon-Rezensionen (1–5 Sterne und 3 Polaritätsklassen)  
**Datengrundlage:** 149.992 Rezensionen, Train/Val/Test (80/10/10), Seed 42  
**Zielmetrik:** Macro-F1

## Datenaufbereitung (D01–D07)

| Nr. | Zweck |
|---|---|
| D01 | Rohdaten auf Vollständigkeit und Struktur prüfen |
| D02 | Explorative Datenanalyse und Sterneverteilung untersuchen |
| D03 | Stratifizierte Stichproben mit 50.000 Rezensionen je Produktkategorie erstellen |
| D04 | Stichproben und Duplikate überprüfen |
| D05 | Nutzer- und Produktmetadaten ergänzen |
| D06 | Duplikate und widersprüchliche Sternebewertungen analysieren |
| D07 | Daten deduplizieren und gruppierten Train-/Validierungs-/Testsplit erstellen |

## Modellexperimente (E01–E21)

| Nr. | Ziel des Experiments |
|---|---|
| E01 | TF-IDF-Baseline mit logistischer Regression trainieren |
| E02 | Einfluss ausgeglichener Klassengewichte untersuchen |
| E03 | Fehlklassifikationen der Baseline und gewichteten Variante vergleichen |
| E04 | Klassengewichtung optimieren |
| E05 | Einfluss der Regularisierung untersuchen |
| E06 | Unigramme und Bigramme vergleichen |
| E07 | Logistische Regression mit LinearSVC vergleichen |
| E08 | Früheres Modell ohne Rezensionstitel auf Testdaten evaluieren |
| E09 | Verschiedene Kombinationen aus Rezensionstitel und -text vergleichen |
| E10 | Fehlklassifikationen ausgewählter Titelvarianten analysieren |
| E11 | Klassengewichtung und Regularisierung gemeinsam untersuchen |
| E12 | Separate TF-IDF-Vektorisierung von Titel und Text mit unterschiedlicher Gewichtung testen |
| E13 | Wort- und Zeichen-N-Gramme kombinieren |
| E14 | Einfluss der Vokabulargröße und Mindesthäufigkeit untersuchen |
| E15 | DistilBERT-Pilottraining mit reduzierter Datenmenge durchführen |
| E16 | DistilBERT mit vollständigem Trainingsdatensatz trainieren |
| E17 | TextBlob-Subjektivitätsfilter als zusätzliche Vorverarbeitung testen |
| E18 | TF-IDF-Modell direkt für drei Polaritätsklassen trainieren |
| E19 | DistilBERT auf dem 5-Sterne-Testsplit evaluieren |
| E20 | TF-IDF und DistilBERT auf drei Polaritätsklassen vergleichen |
| E21 | Finales TF-IDF-Modell auf dem 5-Sterne-Testsplit evaluieren |

**Projektstruktur:** `scripts/` – Implementierungen · `reports/` – Auswertungen und Abbildungen · `models/` – lokale Modellartefakte

Die detaillierte Methodik, die Ergebnisse und deren Interpretation sind im Projektbericht dokumentiert.
