
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# E16 – DISTILBERT AUF DEM VOLLSTÄNDIGEN TRAININGSDATENSATZ
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data/processed/splits"
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "models/E16_distilbert"
REPORT_PATH = REPORTS / "E16_distilbert_results.json"

MODEL_NAME = "distilbert/distilbert-base-uncased"

MAX_LENGTH = 192
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5
SEED = 42
LABELS = np.arange(5)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prepare_text(row):
    title = str(row.get("title") or "").strip()
    body = str(row.get("text") or "").strip()
    return f"{title} {body}".strip()


class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(
                self.encodings["input_ids"][index],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][index],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        }


def evaluate(model, loader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validierung", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            predicted = outputs.logits.argmax(dim=1)

            predictions.extend(predicted.cpu().tolist())
            targets.extend(batch["labels"].tolist())

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(
        targets,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=LABELS,
            target_names=[
                "1 Stern",
                "2 Sterne",
                "3 Sterne",
                "4 Sterne",
                "5 Sterne",
            ],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=LABELS
        ).tolist(),
    }


def save_report(report):
    REPORTS.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def main():
    print("=" * 65)
    print("E16 – DISTILBERT FULL DATASET FINE-TUNING")
    print("=" * 65)

    set_seed(SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Gerät: {device}")
    print(f"Modell: {MODEL_NAME}")

    if MODEL_DIR.exists() or REPORT_PATH.exists():
        raise FileExistsError(
            "E16-Ausgaben existieren bereits. "
            "Bitte vor erneutem Start prüfen, "
            "damit keine Ergebnisse überschrieben werden."
        )

    print("\nLade vollständige Trainingsdaten ...")

    train_rows = load_jsonl(SPLITS / "train.jsonl")
    val_rows = load_jsonl(SPLITS / "validation.jsonl")

    train_texts = [prepare_text(row) for row in train_rows]
    val_texts = [prepare_text(row) for row in val_rows]

    train_labels = np.array(
        [int(row["rating"]) - 1 for row in train_rows],
        dtype=np.int64,
    )
    val_labels = np.array(
        [int(row["rating"]) - 1 for row in val_rows],
        dtype=np.int64,
    )

    print(f"Training: {len(train_rows):,}")
    print(f"Validierung: {len(val_rows):,}")

    if not np.isin(train_labels, LABELS).all():
        raise ValueError("Ungültige Sternebewertungen im Training.")

    if not np.isin(val_labels, LABELS).all():
        raise ValueError("Ungültige Sternebewertungen in der Validierung.")

    print("\nTrainingsverteilung:")
    for label in LABELS:
        count = int((train_labels == label).sum())
        print(f"  {label + 1} Sterne: {count:,}")

    print("\nLade Tokenizer und vortrainiertes Modell ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
    )
    model.to(device)

    print("\nTokenisiere Trainingsdaten ...")

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    print("Tokenisiere Validierungsdaten ...")

    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    train_dataset = ReviewDataset(train_encodings, train_labels)
    val_dataset = ReviewDataset(val_encodings, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=LABELS,
        y=train_labels,
    )

    class_weights = torch.tensor(
        np.sqrt(balanced_weights),
        dtype=torch.float32,
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    report = {
        "experiment": "E16",
        "model": MODEL_NAME,
        "device": str(device),
        "train_size": len(train_rows),
        "validation_size": len(val_rows),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "class_weight_exponent": 0.5,
        "epoch_results": [],
        "best_epoch": None,
        "best_macro_f1": None,
    }

    best_macro_f1 = -1.0
    best_epoch = None

    print("\nStarte Fine-Tuning ...")

    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total_samples = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoche {epoch}/{EPOCHS}",
        )

        for batch in progress:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_loss = running_loss / total_samples

        print(f"\nEpoche {epoch} abgeschlossen.")
        print(f"Trainings-Loss: {avg_loss:.4f}")

        print(f"\nValidierung nach Epoche {epoch} ...")

        val_start = time.time()
        metrics = evaluate(model, val_loader, device)
        val_duration = time.time() - val_start

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Validierungszeit: {val_duration / 60:.1f} Minuten")

        epoch_result = {
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "validation_seconds": float(val_duration),
            **metrics,
        }

        report["epoch_results"].append(epoch_result)

        # Das beste Modell wird anhand des Macro-F1 gewählt.
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch

            print("\nNeuer bester Modellstand – speichere ...")

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            print(f"Gespeichert: {MODEL_DIR}")

        report["best_epoch"] = best_epoch
        report["best_macro_f1"] = float(best_macro_f1)
        report["training_seconds_so_far"] = (
            time.time() - training_start
        )

        # Nach jeder Epoche werden die bisherigen
        # Ergebnisse zusätzlich auf der Festplatte gesichert.
        save_report(report)

    training_duration = time.time() - training_start

    report["training_seconds"] = float(training_duration)
    report["status"] = "completed"
    save_report(report)

    print("\n" + "=" * 65)
    print("E16 – ABSCHLUSS")
    print("=" * 65)

    print(f"Beste Epoche: {best_epoch}")
    print(f"Bester Validierungs-Macro-F1: {best_macro_f1:.4f}")
    print(f"Gesamtdauer: {training_duration / 60:.1f} Minuten")
    print(f"Modell: {MODEL_DIR}")
    print(f"Bericht: {REPORT_PATH}")
    print("\nE16 abgeschlossen.")


if __name__ == "__main__":
    main()
