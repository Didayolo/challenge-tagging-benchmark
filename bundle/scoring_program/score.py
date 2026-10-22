#!/usr/bin/env python3
"""Scoring program for codabench-tagger.

Training and prediction happen in the ingestion program; this program only
compares the predictions with the ground truth.

Reads:
  {input_dir}/ref/test_labels.csv   -- ground-truth labels of the evaluated set
  {input_dir}/res/predictions.csv   -- predicted labels (written by ingestion)
  {input_dir}/res/metadata.json     -- train / inference times (written by ingestion)

Both CSVs are semicolon-separated, aligned row by row, with one column per
TAXONOMY category. Within a label cell, multiple labels are comma-separated:
    Machine Learning Domain;Fields of Application
    NLP,LLM;Legal,Education

Writes {output_dir}/scores.json, one macro-F1 per category (keys in
LEADERBOARD_KEYS) plus:
  train_time_s             -- time spent in fit()
  infer_time_ms_per_sample -- predict() time per test sample
Tags that never appear in the evaluated set are excluded from the averages.
"""
import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

csv.field_size_limit(sys.maxsize)

# Column separator used by the dataset CSVs.
DELIMITER = ';'
# Separator between several labels inside a single label cell.
LABEL_SEPARATOR = ','

# Category -> list of allowed tags. Keep in sync with the ingestion program.
TAXONOMY = {
    "Machine Learning Domain": [
        "Computer Vision",
        "NLP",
        "Tabular",
        "Time Series",
        "LLM",
        "Generative AI",
        "Reinforcement Learning",
        "Audio",
        "Graphs",
        "Multimodal",
        "AutoML",
        "Algorithms",
        "Other",
    ],
    "Fields of Application": [
        "Healthcare",
        "Biology",
        "Finance",
        "E-commerce",
        "Earth Observation",
        "Climate",
        "Energy",
        "Robotics",
        "Cybersecurity",
        "Transportation",
        "Agriculture",
        "Sports",
        "Education",
        "Social Media",
        "Astronomy",
        "Legal",
        "Science",
        "Other",
    ],
}

# Category -> leaderboard column key (must match competition.yaml).
LEADERBOARD_KEYS = {
    "Machine Learning Domain": "ml_domain_f1",
    "Fields of Application": "fields_f1",
}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _read_csv(path, required):
    """Read a ';'-separated CSV, checking the expected columns are present."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected data file not found: {path}")
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter=DELIMITER)
        header = reader.fieldnames or []
        missing = [c for c in required if c not in header]
        if missing:
            raise ValueError(
                f"{path} is missing expected column(s): {', '.join(missing)}. "
                f"Found header: {header}. Expected a '{DELIMITER}'-separated CSV "
                f"with columns: {', '.join(required)}."
            )
        return list(reader)


def _split_labels(cell):
    if not cell:
        return []
    return [x.strip() for x in cell.split(LABEL_SEPARATOR) if x.strip()]


def load_labels(path, taxonomy):
    """One 0/1 matrix per category, columns in TAXONOMY tag order.

    Raises on any label that is not in TAXONOMY (instead of silently dropping it).
    """
    rows = _read_csv(path, list(taxonomy))
    y = {}
    for category, tags in taxonomy.items():
        labels = [_split_labels(row[category]) for row in rows]
        allowed = set(tags)
        errors = [f"row {i}: {t!r}" for i, row in enumerate(labels) for t in row if t not in allowed]
        if errors:
            shown = '; '.join(errors[:5]) + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else '')
            raise ValueError(f"Unknown label(s) for category {category!r} in {path}: {shown}")
        y[category] = MultiLabelBinarizer(classes=tags).fit_transform(labels)
    return y


# ----------------------------------------------------------------------------
def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else '/app/input'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/app/output'

    ref_dir = os.path.join(input_dir, 'ref')
    res_dir = os.path.join(input_dir, 'res')

    os.makedirs(output_dir, exist_ok=True)

    y_true = load_labels(os.path.join(ref_dir, 'test_labels.csv'), TAXONOMY)
    y_pred = load_labels(os.path.join(res_dir, 'predictions.csv'), TAXONOMY)

    n_true = len(next(iter(y_true.values())))
    n_pred = len(next(iter(y_pred.values())))
    if n_true != n_pred:
        raise ValueError(f"predictions.csv has {n_pred} rows, expected {n_true}")
    print(f"Scoring {n_true} samples, categories: {list(TAXONOMY)}")

    with open(os.path.join(res_dir, 'metadata.json'), encoding='utf-8') as f:
        metadata = json.load(f)

    # Macro-F1 per category over the tags present in the evaluated set
    scores = {}
    for category, tags in TAXONOMY.items():
        f1 = f1_score(y_true[category], y_pred[category], average=None, zero_division=0)
        support = y_true[category].sum(axis=0)
        keep = support > 0
        if not keep.any():
            raise ValueError(f"Category {category!r}: no tag present in the reference data")
        scores[LEADERBOARD_KEYS[category]] = float(f1[keep].mean())
        for name, s, n_pos in sorted(zip(np.array(tags)[keep], f1[keep], support[keep]), key=lambda x: x[1]):
            print(f"  [{category}] {name:<24} F1={s:.2f}  (positives: {n_pos})")

    scores['train_time_s'] = float(metadata.get('train_time_s', -1))
    scores['infer_time_ms_per_sample'] = float(metadata.get('infer_time_ms_per_sample', -1))

    with open(os.path.join(output_dir, 'scores.json'), 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2)

    print(f"\nFinal scores: {scores}")


if __name__ == '__main__':
    main()
