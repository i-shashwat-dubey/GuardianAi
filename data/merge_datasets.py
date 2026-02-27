"""
Guardian AI — Dataset Merger & Train/Test Splitter
====================================================
Merges safety_dataset_english.csv + safety_dataset_hindi.csv,
shuffles rows, and creates a stratified 80/20 train/test split.

Output:
    train.csv   — 80% of data (used for fine-tuning)
    test.csv    — 20% of data (used for evaluation only, NEVER training)

Usage:
    python merge_datasets.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ==================================================================
#  CONFIGURATION
# ==================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# Input CSVs — add or remove files from this list as needed
INPUT_FILES = [
    os.path.join(BASE_DIR, "raw", "safety_dataset_english.csv"),
    os.path.join(BASE_DIR, "raw", "safety_dataset_hindi.csv"),
]

TRAIN_FILE  = os.path.join(BASE_DIR, "train.csv")
TEST_FILE   = os.path.join(BASE_DIR, "test.csv")

TEST_SIZE   = 0.20   # 20% held out for evaluation
RANDOM_SEED = 42


# ==================================================================
#  MAIN
# ==================================================================

def main():
    print()
    print("  +======================================================+")
    print("  |   Guardian AI -- Dataset Merger                      |")
    print("  +======================================================+")
    print()

    # ── Load all CSVs ─────────────────────────────────────────────
    frames = []
    for path in INPUT_FILES:
        if not os.path.exists(path):
            print(f"  [WARN] File not found, skipping: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        print(f"  [OK]   Loaded {len(df):>5} rows from {os.path.basename(path)}")
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No input files found. Run generate_dataset.py first.")

    # ── Merge & clean ─────────────────────────────────────────────
    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)

    # Drop rows missing text or label
    merged = merged.dropna(subset=["text", "label"])
    merged["label"] = merged["label"].astype(int)
    merged = merged[merged["label"].isin([0, 1])]

    # Drop exact duplicates
    merged = merged.drop_duplicates(subset=["text"])
    after = len(merged)

    print()
    print(f"  Total rows   : {before}")
    print(f"  After cleanup: {after} ({before - after} duplicates/invalid removed)")
    print(f"  Label 0 (SAFE)     : {(merged['label'] == 0).sum()}")
    print(f"  Label 1 (EMERGENCY): {(merged['label'] == 1).sum()}")

    # ── Shuffle ──────────────────────────────────────────────────
    merged = merged.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # ── Stratified split ──────────────────────────────────────────
    train_df, test_df = train_test_split(
        merged,
        test_size=TEST_SIZE,
        stratify=merged["label"],
        random_state=RANDOM_SEED,
    )

    # ── Save ──────────────────────────────────────────────────────
    train_df.to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_FILE,  index=False, encoding="utf-8-sig")

    print()
    print(f"  {'=' * 55}")
    print(f"  DONE!")
    print(f"  train.csv : {len(train_df)} rows  ->  {TRAIN_FILE}")
    print(f"  test.csv  : {len(test_df)} rows   ->  {TEST_FILE}")
    print(f"  {'=' * 55}")
    print()
    print("  Next: run evaluate.py to get baseline accuracy,")
    print("  then fine-tune on Colab using finetune_muril.ipynb.")


if __name__ == "__main__":
    main()
