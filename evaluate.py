"""
Guardian AI — MuRIL Model Evaluator
=====================================
Evaluates any MuRIL model on test.csv and prints accuracy, precision,
recall and F1 score per class.

Run BEFORE fine-tuning to get baseline accuracy:
    python evaluate.py

Run AFTER fine-tuning to get post-tuning accuracy:
    python evaluate.py --model ./muril_finetuned

Usage:
    python evaluate.py [--model MODEL_NAME_OR_PATH] [--test TEST_CSV]
"""

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ==================================================================
#  CONFIGURATION
# ==================================================================

DEFAULT_MODEL = "google/muril-base-cased"   # base MuRIL (before fine-tuning)
DEFAULT_TEST  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.csv")
MAX_LEN       = 128
BATCH_SIZE    = 32


# ==================================================================
#  DATASET
# ==================================================================

class SafetyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================================================================
#  EVALUATION
# ==================================================================

def evaluate(model_path: str, test_csv: str):
    print()
    print("  +======================================================+")
    print("  |   Guardian AI -- MuRIL Evaluator                     |")
    print("  +======================================================+")
    print()
    print(f"  Model : {model_path}")
    print(f"  Data  : {test_csv}")
    print()

    # ── Load test data ────────────────────────────────────────────
    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"test.csv not found at {test_csv}.\n"
            "Run merge_datasets.py first."
        )

    df = pd.read_csv(test_csv, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    texts  = df["text"].tolist()
    labels = df["label"].tolist()
    print(f"  Test rows : {len(texts)}")
    print(f"  SAFE (0)  : {labels.count(0)}")
    print(f"  EMERGENCY (1): {labels.count(1)}")
    print()

    # ── Load model & tokenizer ────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Loading tokenizer & model ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2
    )
    model.to(device)
    model.eval()

    # ── Build DataLoader ──────────────────────────────────────────
    dataset    = SafetyDataset(texts, labels, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # ── Run inference ─────────────────────────────────────────────
    all_preds  = []
    all_labels = []

    print(f"  Running inference on {len(texts)} samples ...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())

    # ── Print results ─────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=["SAFE (0)", "EMERGENCY (1)"],
        digits=4,
    )

    print()
    print(f"  {'=' * 55}")
    print(f"  RESULTS for: {model_path}")
    print(f"  {'=' * 55}")
    print(f"  Overall Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print()
    print("  Classification Report:")
    for line in report.splitlines():
        print(f"    {line}")
    print()
    print("  Confusion Matrix:")
    print(f"    (rows=actual, cols=predicted)")
    print(f"                 Pred SAFE   Pred EMERGENCY")
    print(f"    Actual SAFE      {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"    Actual EMER      {cm[1][0]:>5}          {cm[1][1]:>5}")
    print()
    print("  KEY METRIC -> Emergency Recall (must be high for a safety app):")
    emg_recall = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    print(f"  Emergency Recall : {emg_recall:.4f}  ({emg_recall*100:.2f}%)")
    print(f"  {'=' * 55}")
    print()


# ==================================================================
#  ENTRY POINT
# ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model name or path to fine-tuned model folder",
    )
    parser.add_argument(
        "--test",
        default=DEFAULT_TEST,
        help="Path to test.csv",
    )
    args = parser.parse_args()
    evaluate(args.model, args.test)
