
import json
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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models/E16_distilbert"
TEST_PATH = ROOT / "data/processed/splits/test.jsonl"
REPORT_PATH = ROOT / "reports/E19_distilbert_test_results.json"

BATCH_SIZE = 8
MAX_LENGTH = 192
LABELS = [0, 1, 2, 3, 4]


class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                self.encodings["input_ids"][idx],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][idx],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.long,
            ),
        }


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    print("=" * 60)
    print("E19 – DISTILBERT TEST EVALUATION")
    print("=" * 60)

    if REPORT_PATH.exists():
        raise FileExistsError(
            f"Testbericht existiert bereits: {REPORT_PATH}"
        )

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Gespeichertes E16-Modell fehlt: {MODEL_DIR}"
        )

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Gerät: {device}")
    print(f"Modell: {MODEL_DIR}")

    rows = load_jsonl(TEST_PATH)

    texts = [
        f"{str(row.get('title') or '').strip()} "
        f"{str(row.get('text') or '').strip()}".strip()
        for row in rows
    ]

    labels = np.array(
        [int(row["rating"]) - 1 for row in rows],
        dtype=np.int64,
    )

    print(f"Testrezensionen: {len(rows):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )
    model.to(device)
    model.eval()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    dataset = ReviewDataset(encodings, labels)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    predictions = []
    targets = []

    start = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testevaluation"):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
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

    report = classification_report(
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
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=LABELS,
    ).tolist()

    results = {
        "experiment": "E19",
        "model": "E16_distilbert",
        "selected_epoch": 1,
        "dataset": "test",
        "test_size": len(rows),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": matrix,
        "evaluation_seconds": time.time() - start,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("E19 – TESTERGEBNISSE")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nKlassenweise Ergebnisse:")

    for star in range(1, 6):
        key = f"{star} Stern" if star == 1 else f"{star} Sterne"
        metrics = report[key]

        print(
            f"{star} Sterne: "
            f"Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}, "
            f"F1={metrics['f1-score']:.3f}"
        )

    print("\nConfusion Matrix:")
    for row in matrix:
        print(row)

    print(f"\nDauer: {results['evaluation_seconds'] / 60:.1f} Min.")
    print(f"Bericht gespeichert: {REPORT_PATH}")


if __name__ == "__main__":
    main()
