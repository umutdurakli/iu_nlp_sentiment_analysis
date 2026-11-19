"""Lokale Vorhersagen mit den finalen TF-IDF- und DistilBERT-Modellen.

Aufruf vom Projektstamm:
    ./.venv/bin/python scripts/predict_sentiment.py --model all --title "Titel" --text "Text"
Ohne --text wird der Rezensionstext interaktiv abgefragt; der Titel ist optional.
Es werden ausschließlich lokale Artefakte geladen, ohne Training oder Datensätze.
"""

import argparse
import os
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent.parent / "models/final_models/E16_distilbert_5class"
TFIDF_MODEL = MODEL_DIR.parent / "E09B_tfidf_5class.joblib"
POLARITY_MODEL = MODEL_DIR.parent / "E18_tfidf_polarity_3class.joblib"
POLARITY_NAMES = ("negativ", "neutral", "positiv")
MAX_LENGTH = 192


def prepare_text(title, text):
    """Verbindet Titel und Text wie E09 B, E16 und E18, ohne zusätzliche Bereinigung."""
    title = str(title or "").strip()
    body = str(text or "").strip()
    return f"{title} {body}".strip()


def tokenize_text(tokenizer, title, text):
    """Erzeugt dieselben 192 Positionen wie E16/E19, direkt als PyTorch-Tensoren."""
    combined = prepare_text(title, text)
    if not combined:
        raise ValueError("Bitte mindestens einen Titel oder Rezensionstext eingeben.")
    return tokenizer(
        [combined],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


def load_model(model_dir, device_name="auto"):
    """Lädt unveränderte lokale Gewichte und Tokenizer; greift nicht aufs Netz zu."""
    model_dir = Path(model_dir)
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"Lokales Modellartefakt fehlt: {model_dir / name}")

    # Diese Einstellungen gelten nur für den laufenden Python-Prozess.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device_name == "auto":
        device_name = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS ist auf diesem Gerät nicht verfügbar; --device cpu verwenden.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA ist auf diesem Gerät nicht verfügbar; --device cpu verwenden.")
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
    )
    if model.config.num_labels != 5:
        raise ValueError("Das finale Modell muss genau fünf Sterneklassen besitzen.")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_rating(model, tokenizer, device, title, text):
    """Gibt argmax + 1 zurück: E16 trainiert Klassenindizes 0–4 für Sterne 1–5."""
    import torch

    encoded = tokenize_text(tokenizer, title, text)
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
    return int(outputs.logits.argmax(dim=1).item()) + 1


def main():
    """Liest eine Rezension und zeigt die Vorhersagen der gewählten Modelle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("tfidf", "distilbert", "polarity", "all"),
        default="all", help="Modellauswahl; Standard: all",
    )
    parser.add_argument("--title", default="", help="Optionaler Rezensionstitel")
    parser.add_argument("--text", help="Englischer Rezensionstext; sonst interaktive Abfrage")
    parser.add_argument(
        "--polarity", action="store_true",
        help="Mit --model distilbert zusätzlich die abgebildete Polarität ausgeben",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto",
        help="Gerät für DistilBERT; Standard: auto",
    )
    args = parser.parse_args()
    if args.polarity and args.model != "distilbert":
        parser.error("--polarity ist nur mit --model distilbert erlaubt.")
    try:
        text = args.text if args.text is not None else input("Englischer Rezensionstext: ")
        if not text.strip():
            parser.error("Bitte einen nichtleeren Rezensionstext über --text oder interaktiv eingeben.")
        combined = prepare_text(args.title, text)
        selected = ("tfidf", "distilbert", "polarity") if args.model == "all" else (args.model,)
        for name in selected:
            if name == "distilbert":
                model, tokenizer, device = load_model(MODEL_DIR, args.device)
                stars = predict_rating(model, tokenizer, device, args.title, text)
                print(f"DistilBERT (5 Klassen): {stars} Sterne")
                if args.polarity or args.model == "all":
                    label = 0 if stars <= 2 else (1 if stars == 3 else 2)
                    print(f"DistilBERT (3 Klassen, aus 5 Klassen abgeleitet): {POLARITY_NAMES[label]}")
            else:
                import joblib

                path = TFIDF_MODEL if name == "tfidf" else POLARITY_MODEL
                if not path.is_file():
                    raise FileNotFoundError(f"Lokales Modellartefakt fehlt: {path}")
                model = joblib.load(path)
                prediction = int(model.predict([combined])[0])
                if name == "tfidf":
                    if prediction not in (1, 2, 3, 4, 5):
                        raise ValueError("Das TF-IDF-Modell lieferte keine gültige Sterneklasse.")
                    print(f"TF-IDF (5 Klassen): {prediction} Sterne")
                else:
                    if prediction not in (0, 1, 2):
                        raise ValueError("Das Polaritätsmodell lieferte keine gültige Klasse.")
                    print(f"TF-IDF (3 Klassen): {POLARITY_NAMES[prediction]}")
    except EOFError:
        parser.error("Eingabe fehlt; --text angeben oder interaktiv starten.")
    except ModuleNotFoundError as error:
        parser.exit(1, f"Benötigtes Python-Modul fehlt: {error.name}. Bitte die Projekt-.venv verwenden.\n")
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
