#!/usr/bin/env python3
"""Ingestion program for codabench-tagger.

Participants submit a model.py containing a Model class. This program trains
it on the training set and predicts the test set of the phase (the validation
set in the feedback phase, the test set in the final phase).

Reads {input_dir}:
  train_data.csv    -- column `text`
  train_labels.csv  -- one column per TAXONOMY category
  test_data.csv     -- column `text`
All CSVs are semicolon-separated; data and labels are aligned row by row.
Within a label cell, multiple labels are comma-separated.

Model interface (generic, whatever the number of categories):
    model.fit(X_train, y_train)   # y_train: {category: (n, n_tags) 0/1 array}
    model.predict(X_test)         # -> {category: (n, n_tags) 0/1 array}
Columns of each array follow the order of the tag list in TAXONOMY.

Writes {output_dir}:
  predictions.csv  -- predicted labels, same format as train_labels.csv
  metadata.json    -- train_time_s, infer_time_ms_per_sample

Args (positional, Codabench convention):
  1: /app/input_data/         -- task input data
  2: /app/output/             -- output directory
  3: /app/program/            -- ingestion program dir
  4: /app/ingested_program/   -- participant's submitted files
"""
import csv
import importlib.util
import json
import os
import sys
import time

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

csv.field_size_limit(sys.maxsize)

# Column separator used by the dataset CSVs.
DELIMITER = ';'
# Separator between several labels inside a single label cell.
LABEL_SEPARATOR = ','

# Category -> list of allowed tags. Keep in sync with the scoring program.
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


def load_texts(path):
    return [row['text'] for row in _read_csv(path, ['text'])]


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
# Model loading / prediction checks
# ----------------------------------------------------------------------------
def load_model_class(submission_dir):
    model_path = os.path.join(submission_dir, 'model.py')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model.py not found in submission at: {model_path}\n"
            "Ensure your submission zip contains model.py at its root."
        )
    sys.path.insert(0, submission_dir)  # let model.py import its sibling files
    spec = importlib.util.spec_from_file_location('submission_model', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def check_predictions(y_pred, n_samples, taxonomy):
    """Make sure predict() returned one correctly-shaped 0/1 array per category."""
    if not isinstance(y_pred, dict):
        raise TypeError(
            "model.predict() must return a dict {category: array}, "
            f"got {type(y_pred).__name__}. Expected categories: {list(taxonomy)}"
        )
    out = {}
    for category, tags in taxonomy.items():
        if category not in y_pred:
            raise KeyError(f"model.predict() output is missing category {category!r}")
        arr = np.asarray(y_pred[category])
        expected = (n_samples, len(tags))
        if arr.shape != expected:
            raise ValueError(
                f"Prediction for {category!r} has shape {arr.shape}, "
                f"expected {expected} (n_samples, n_tags)"
            )
        if not np.isin(arr, (0, 1)).all():
            raise ValueError(f"Prediction for {category!r} must only contain 0 and 1")
        out[category] = arr.astype(int)
    return out


def write_predictions(path, y_pred, taxonomy):
    """Write predictions in the same format as the label files."""
    n_samples = len(next(iter(y_pred.values())))
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=DELIMITER)
        writer.writerow(list(taxonomy))
        for i in range(n_samples):
            writer.writerow([
                LABEL_SEPARATOR.join(t for t, v in zip(tags, y_pred[c][i]) if v)
                for c, tags in taxonomy.items()
            ])


# ----------------------------------------------------------------------------
def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else '/app/input_data'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/app/output'
    submission_dir = sys.argv[4] if len(sys.argv) > 4 else '/app/ingested_program'

    os.makedirs(output_dir, exist_ok=True)

    X_train = load_texts(os.path.join(input_dir, 'train_data.csv'))
    y_train = load_labels(os.path.join(input_dir, 'train_labels.csv'), TAXONOMY)
    X_test = load_texts(os.path.join(input_dir, 'test_data.csv'))
    for category, y in y_train.items():
        if len(y) != len(X_train):
            raise ValueError(f"train_labels.csv has {len(y)} rows but train_data.csv has {len(X_train)}")
    print(f"Loaded {len(X_train)} train / {len(X_test)} test samples, categories: {list(TAXONOMY)}")

    model = load_model_class(submission_dir)()

    print("Training model")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0

    print("Running prediction")
    t1 = time.perf_counter()
    raw_pred = model.predict(X_test)
    infer_time_ms = (time.perf_counter() - t1) * 1000.0

    y_pred = check_predictions(raw_pred, len(X_test), TAXONOMY)
    write_predictions(os.path.join(output_dir, 'predictions.csv'), y_pred, TAXONOMY)

    metadata = {
        'train_time_s': float(train_time_s),
        'infer_time_ms_per_sample': float(infer_time_ms / len(X_test)),
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Done: {metadata}")


if __name__ == '__main__':
    main()
