
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/final_models/E09B_tfidf_5class.joblib"
OUTPUT = ROOT / "reports/E21_tfidf_5class_test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="rating")
    args = parser.parse_args()

    path = Path(args.test_file)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path, lines=True)

    required = [args.title_col, args.text_col, args.label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}. Vorhanden: {list(df.columns)}")

    if len(df) != 14999:
        raise ValueError(f"Erwartet: 14999 Testrezensionen, gefunden: {len(df)}")

    if df[required].isna().any().any():
        raise ValueError("Fehlende Werte in Titel, Text oder Sternebewertung")

    X = (
        df[args.title_col].astype(str).str.strip()
        + " "
        + df[args.text_col].astype(str).str.strip()
    ).str.strip()

    y_true = pd.to_numeric(df[args.label_col], errors="raise").astype(int)

    if not y_true.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Ungültige Sternebewertungen")

    model = joblib.load(MODEL)
    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=[1, 2, 3, 4, 5], average="macro")

    report = classification_report(
        y_true,
        y_pred,
        labels=[1, 2, 3, 4, 5],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "E21",
        "model": "E09B_tfidf_5class",
        "dataset": str(path),
        "samples": len(df),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    (OUTPUT / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in range(1, 6)],
        columns=[f"pred_{i}" for i in range(1, 6)],
    ).to_csv(OUTPUT / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[1, 2, 3, 4, 5],
    ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

    ax.set_title("E21 – TF-IDF (E09 B), Testdaten")
    fig.tight_layout()
    fig.savefig(OUTPUT / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"Testrezensionen: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[1, 2, 3, 4, 5],
            zero_division=0,
        )
    )
    print(f"Ergebnisse gespeichert: {OUTPUT}")


if __name__ == "__main__":
    main()
