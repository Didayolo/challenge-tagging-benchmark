#!/usr/bin/env python3
"""Scoring program for codabench-tagger.

Reads:
  {input_dir}/ref/reference_data.csv -- ground-truth labels (200 rows)
  {input_dir}/res/input_data.csv     -- competition texts (copied by ingestion)
  {input_dir}/res/model.py           -- participant's Model class (copied by ingestion)

Both CSVs are semicolon-separated and aligned row by row: row i of
input_data.csv is described by row i of reference_data.csv. Within a label
cell, multiple labels are comma-separated.

Runs 4-fold cross-validation and writes:
  {output_dir}/scores.json    -- {primary, f1_std, train_time_s, infer_time_ms_per_sample}
"""
import sys
import os
import json
import time
import importlib.util
import csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

ML_LABELS = [
    'Natural Language Processing',
    'Computer Vision',
    'Tabular / Structured Data',
    'Time Series & Forecasting',
    'Reinforcement Learning',
    'Generative Models',
    'Graph Learning',
    'Federated & Privacy-Preserving Learning',
    'AutoML & Neural Architecture Search',
    'Multimodal Learning',
]

FIELD_LABELS = [
    'Healthcare & Medicine',
    'Biology & Bioinformatics',
    'Climate & Environment',
    'Finance & Economics',
    'Agriculture & Food Science',
    'Autonomous Systems & Robotics',
    'Social Sciences & Humanities',
    'Cybersecurity',
    'Education & Learning Sciences',
    'Materials & Physical Sciences',
]

# Column separator used by the dataset CSVs.
DELIMITER = ';'
# Separator between several labels inside a single label cell.
LABEL_SEPARATOR = ','


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


def load_data(ref_dir, res_dir):
    """Load texts from the ingestion output and labels from the reference data.

    The two files carry no key column; they are matched by row order, so a
    length mismatch means the task data is inconsistent and scoring must stop.
    """
    input_path = os.path.join(res_dir, 'input_data.csv')
    ref_path = os.path.join(ref_dir, 'reference_data.csv')

    input_rows = _read_csv(input_path, ['text'])
    ref_rows = _read_csv(ref_path, ['ML_sector', 'Field_sector'])

    if len(input_rows) != len(ref_rows):
        raise ValueError(
            f"Row count mismatch: {input_path} has {len(input_rows)} rows but "
            f"{ref_path} has {len(ref_rows)}. The two files are aligned by row "
            "order and must have the same length."
        )

    texts = [row['text'] for row in input_rows]
    ml_labels, field_labels = [], []
    for row in ref_rows:
        ml_labels.append(_split_labels(row['ML_sector']))
        field_labels.append(_split_labels(row['Field_sector']))
    return texts, ml_labels, field_labels


def _split_labels(cell):
    if not cell:
        return []
    return [x.strip() for x in cell.split(LABEL_SEPARATOR) if x.strip()]


def load_model_class(res_dir):
    model_path = os.path.join(res_dir, 'model.py')
    spec = importlib.util.spec_from_file_location('submission_model', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else '/app/input'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/app/output'

    ref_dir = os.path.join(input_dir, 'ref')
    res_dir = os.path.join(input_dir, 'res')

    os.makedirs(output_dir, exist_ok=True)

    # Texts come from the ingestion output, labels from reference_data
    texts, ml_label_lists, field_label_lists = load_data(ref_dir, res_dir)
    print(f"Loaded {len(texts)} samples")

    # Binarize labels using fixed class order (all 10 labels always present)
    mlb_ml = MultiLabelBinarizer(classes=ML_LABELS)
    mlb_field = MultiLabelBinarizer(classes=FIELD_LABELS)
    y_ml = mlb_ml.fit_transform(ml_label_lists)
    y_field = mlb_field.fit_transform(field_label_lists)
    X = np.array(texts, dtype=object)

    # Load participant's Model class (passed via ingestion output)
    ModelClass = load_model_class(res_dir)

    # 4-fold cross-validation
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    fold_scores = []
    total_train_time = 0.0
    total_infer_time_ms = 0.0
    total_test_samples = 0

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train = X[train_idx].tolist()
        X_test = X[test_idx].tolist()
        y_ml_train, y_ml_test = y_ml[train_idx], y_ml[test_idx]
        y_field_train, y_field_test = y_field[train_idx], y_field[test_idx]

        model = ModelClass()

        # Fit
        t0 = time.perf_counter()
        model.fit(X_train, y_ml_train, y_field_train)
        t1 = time.perf_counter()
        total_train_time += (t1 - t0)

        # Predict
        t2 = time.perf_counter()
        y_ml_pred, y_field_pred = model.predict(X_test)
        t3 = time.perf_counter()
        total_infer_time_ms += (t3 - t2) * 1000.0
        total_test_samples += len(X_test)

        # Per-fold score
        f1_ml = f1_score(y_ml_test, np.array(y_ml_pred), average='macro', zero_division=0)
        f1_field = f1_score(y_field_test, np.array(y_field_pred), average='macro', zero_division=0)
        fold_score = (f1_ml + f1_field) / 2.0
        fold_scores.append(fold_score)
        print(f"Fold {fold_idx+1}: f1_ml={f1_ml:.4f}  f1_field={f1_field:.4f}  fold_score={fold_score:.4f}")

    primary = float(np.mean(fold_scores))
    f1_std = float(np.std(fold_scores))
    train_time_s = float(total_train_time)
    infer_time_ms_per_sample = float(total_infer_time_ms / total_test_samples) if total_test_samples > 0 else 0.0

    scores = {
        'primary': primary,
        'f1_std': f1_std,
        'train_time_s': train_time_s,
        'infer_time_ms_per_sample': infer_time_ms_per_sample,
    }

    scores_path = os.path.join(output_dir, 'scores.json')
    with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2)

    print(f"\nFinal scores: {scores}")


if __name__ == '__main__':
    main()
