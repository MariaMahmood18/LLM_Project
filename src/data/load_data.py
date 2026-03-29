"""
Data loading and preprocessing for OpenI radiology dataset.
Loads findings and impression sections for summarization task.
"""

import random
import json
import os
from datasets import load_dataset

def load_openi_dataset(split="train", max_samples=None, seed=42):
    """
    Load OpenI chest X-ray radiology reports from HuggingFace.
    Returns list of dicts with 'findings' and 'impression' keys.
    """
    random.seed(seed)

    try:
        dataset = load_dataset("Segun1914/indiana_university_radiology_reports", split="train")
        records = []
        for item in dataset:
            findings = item.get("findings", "").strip()
            impression = item.get("impression", "").strip()
            if findings and impression and len(findings) > 20:
                records.append({"findings": findings, "impression": impression})
    except Exception:
        print("[WARN] Could not load OpenI from HuggingFace. Using synthetic fallback samples.")
        records = _synthetic_fallback()

    if max_samples:
        random.shuffle(records)
        records = records[:max_samples]

    return records


def _synthetic_fallback():
    """Small set of synthetic radiology-style records for pipeline testing."""
    return [
        {
            "findings": "The lungs are clear bilaterally. No focal consolidation, pleural effusion, or pneumothorax is identified. The cardiomediastinal silhouette is within normal limits. Osseous structures are intact.",
            "impression": "No acute cardiopulmonary abnormality."
        },
        {
            "findings": "There is increased opacity in the right lower lobe consistent with pneumonia. The left lung is clear. Mild cardiomegaly is present. No pleural effusion identified.",
            "impression": "Right lower lobe pneumonia. Mild cardiomegaly."
        },
        {
            "findings": "Mild interstitial prominence noted bilaterally. Heart size is at the upper limits of normal. No pleural effusion. No pneumothorax. Bony thorax is unremarkable.",
            "impression": "Mild interstitial prominence, possibly early pulmonary edema."
        },
        {
            "findings": "The cardiac silhouette is enlarged. Bilateral pleural effusions are present, left greater than right. Pulmonary vascular congestion noted.",
            "impression": "Cardiomegaly with bilateral pleural effusions and pulmonary vascular congestion, consistent with congestive heart failure."
        },
        {
            "findings": "No acute osseous abnormality. Lungs are hyperinflated. Flattening of the diaphragm noted. No focal consolidation. No pneumothorax.",
            "impression": "Hyperinflation consistent with chronic obstructive pulmonary disease (COPD)."
        },
    ] * 20


def split_data(records, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split records into train/val/test sets."""
    random.seed(seed)
    random.shuffle(records)
    n = len(records)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:]
    }


def save_split(splits, output_dir="artifacts/data"):
    """Save data splits to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    for split_name, records in splits.items():
        path = os.path.join(output_dir, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} records to {path}")


if __name__ == "__main__":
    print("Loading OpenI dataset...")
    records = load_openi_dataset(max_samples=200)
    print(f"Total records loaded: {len(records)}")
    splits = split_data(records)
    for k, v in splits.items():
        print(f"  {k}: {len(v)} samples")
    save_split(splits)