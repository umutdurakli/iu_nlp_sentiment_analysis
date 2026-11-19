
"""
E15 – DistilBERT-Fine-Tuning für Amazon-Sternebewertungen.

Ein festgelegtes Training, keine Hyperparametersuche.
30.000 geschichtete Trainingsbeispiele.
Evaluation auf allen 14.999 Validierungsbeispielen.
Keine Verwendung des Testdatensatzes.
"""

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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data/processed/splits"
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "models/E15_distilbert"
REPORT_PATH = REPORTS / "E15_distilbert_results.json"

MODEL_NAME = "distilbert/distilbert-base-uncased"
TRAIN_SIZE = 30_000
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


def load_split(path):
    texts, labels = [], []

    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            title = str(row.get("title") or "").strip()
            body = str(row.get("text") or "").strip()

            # Titel zuerst, damit er bei Trunkierung
            # möglichst erhalten bleibt.
            combined = " ".join(
                part for part in (title, body) if part
            )

            texts.append(combined)
            labels.append(int(row["rating"]) - 1)

    return texts, np.asarray(labels, dtype=np.int64)


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
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
                int(self.labels[index]),
                dtype=torch.long,
            ),
        }


def main():
    set_seed(SEED)

    # Vorhandene E15-Ergebnisse niemals überschreiben.
    if MODEL_DIR.exists() or REPORT_PATH.exists():
        raise FileExistsError(
            "E15-Ausgabe existiert bereits. "
            "Vorhandene Ergebnisse bleiben unverändert."
        )

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS ist nicht verfügbar. "
            "Bitte zuerst PyTorch/GPU-Unterstützung prüfen."
        )

    device = torch.device("mps")

    print("=" * 65)
    print("E15 – DISTILBERT FINE-TUNING")
    print("=" * 65)
    print(f"Gerät: {device}")
    print(f"Modell: {MODEL_NAME}")

    print("\nLade Daten ...")

    train_texts, y_train = load_split(
        SPLITS / "train.jsonl"
    )
    val_texts, y_val = load_split(
        SPLITS / "validation.jsonl"
    )

    indices = np.arange(len(y_train))

    selected, _ = train_test_split(
        indices,
        train_size=TRAIN_SIZE,
        stratify=y_train,
        random_state=SEED,
    )

    selected = np.sort(selected)
    sample_texts = [train_texts[i] for i in selected]
    sample_labels = y_train[selected]

    print(f"Training gesamt: {len(y_train):,}")
    print(f"Training E15: {len(sample_labels):,}")
    print(f"Validierung: {len(y_val):,}")

    print("\nTrainingsverteilung:")
    for label, count in enumerate(
        np.bincount(sample_labels, minlength=5), start=1
    ):
        print(f"  {label} Sterne: {count:,}")

    print("\nLade Tokenizer und Modell ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
        id2label={i: f"{i+1}_star" for i in range(5)},
        label2id={f"{i+1}_star": i for i in range(5)},
    )

    model.to(device)

    train_dataset = ReviewDataset(
        sample_texts, sample_labels, tokenizer
    )
    val_dataset = ReviewDataset(
        val_texts, y_val, tokenizer
    )

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

    # Wie beim klassischen Modell: abgeschwächte
    # inverse Klassenhäufigkeit (alpha = 0.5).
    balanced = compute_class_weight(
        class_weight="balanced",
        classes=LABELS,
        y=sample_labels,
    )
    weights = torch.tensor(
        np.sqrt(balanced),
        dtype=torch.float32,
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    epoch_losses = []
    start = time.perf_counter()

    print("\nStarte Fine-Tuning ...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoche {epoch+1}/{EPOCHS}",
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

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        print(
            f"Epoche {epoch+1} abgeschlossen | "
            f"Trainings-Loss: {avg_loss:.4f}"
        )

    training_seconds = time.perf_counter() - start

    print("\nBewerte auf vollständiger Validierung ...")

    model.eval()
    predictions = []

    eval_start = time.perf_counter()

    with torch.inference_mode():
        for batch in tqdm(
            val_loader, desc="Validierung"
        ):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            predicted = torch.argmax(
                outputs.logits, dim=1
            )
            predictions.extend(
                predicted.cpu().tolist()
            )

    evaluation_seconds = time.perf_counter() - eval_start

    predictions = np.asarray(predictions)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_val,
        predictions,
        labels=LABELS,
        target_names=[
            "1 Stern", "2 Sterne", "3 Sterne",
            "4 Sterne", "5 Sterne"
        ],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_val, predictions, labels=LABELS
    )

    print("\n" + "=" * 65)
    print("E15 – ERGEBNIS")
    print("=" * 65)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Trainingszeit: {training_seconds/60:.1f} Minuten")
    print(
        f"Validierungszeit: "
        f"{evaluation_seconds/60:.1f} Minuten"
    )

    print("\nKlassifikationsbericht:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=LABELS,
            target_names=[
                "1 Stern", "2 Sterne", "3 Sterne",
                "4 Sterne", "5 Sterne"
            ],
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(matrix)

    result = {
        "experiment": "E15",
        "model": MODEL_NAME,
        "device": str(device),
        "seed": SEED,
        "train_samples": len(sample_labels),
        "validation_samples": len(y_val),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "class_weight_alpha": 0.5,
        "train_loss_by_epoch": epoch_losses,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "test_data_used": False,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    with REPORT_PATH.open("x", encoding="utf-8") as file:
        json.dump(
            result, file, indent=2, ensure_ascii=False
        )

    print("\nModell gespeichert:", MODEL_DIR)
    print("Bericht gespeichert:", REPORT_PATH)
    print("E15 abgeschlossen.")


if __name__ == "__main__":
    main()
