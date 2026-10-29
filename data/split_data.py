#!/usr/bin/env python3
"""Split the full labeled dataset into the train / valid / test sets of the bundle.

Reads (next to this script):
  input_data.csv      -- column `text`
  reference_data.csv  -- one column per category (+ optional Conferences)
Both are ';'-separated and aligned row by row.

The rows are shuffled and cut into N_TRAIN / N_VALID / N_TEST samples. Every
label present in the full dataset must appear in each of the three sets: if a
shuffle misses one, the next seed is tried.

Writes, for each phase (feedback = evaluated on valid, final = evaluated on test):
  bundle/<phase>/input_data/train_data.csv
  bundle/<phase>/input_data/train_labels.csv
  bundle/<phase>/input_data/test_data.csv
  bundle/<phase>/reference_data/test_labels.csv

Usage: python3 data/split_data.py
"""
import csv
import os
import sys

import numpy as np

csv.field_size_limit(sys.maxsize)

N_TRAIN, N_VALID, N_TEST = 2326, 774, 774
SEED = 42
MAX_ATTEMPTS = 100

DELIMITER = ';'
LABEL_SEPARATOR = ','
CATEGORIES = ['Machine Learning Domain', 'Fields of Application']

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(DATA_DIR, '..', 'bundle')


def read_csv(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=DELIMITER)
        header = next(reader)
        return header, list(reader)


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=DELIMITER)
        writer.writerow(header)
        writer.writerows(rows)


def row_labels(ref_header, row):
    """Set of (category, tag) pairs of one reference row."""
    labels = set()
    for category in CATEGORIES:
        cell = row[ref_header.index(category)]
        labels.update((category, t.strip()) for t in cell.split(LABEL_SEPARATOR) if t.strip())
    return labels


def missing_labels(all_labels, label_sets, idx):
    present = set().union(*(label_sets[i] for i in idx))
    return all_labels - present


def main():
    in_header, texts = read_csv(os.path.join(DATA_DIR, 'input_data.csv'))
    ref_header, refs = read_csv(os.path.join(DATA_DIR, 'reference_data.csv'))

    n = len(texts)
    if len(refs) != n:
        raise ValueError(f"input_data.csv has {n} rows but reference_data.csv has {len(refs)}")
    if N_TRAIN + N_VALID + N_TEST != n:
        raise ValueError(f"N_TRAIN + N_VALID + N_TEST = {N_TRAIN + N_VALID + N_TEST}, dataset has {n} rows")

    label_sets = [row_labels(ref_header, row) for row in refs]
    all_labels = set().union(*label_sets)

    for seed in range(SEED, SEED + MAX_ATTEMPTS):
        perm = np.random.default_rng(seed).permutation(n)
        splits = {
            'train': perm[:N_TRAIN],
            'valid': perm[N_TRAIN:N_TRAIN + N_VALID],
            'test': perm[N_TRAIN + N_VALID:],
        }
        missing = {name: missing_labels(all_labels, label_sets, idx) for name, idx in splits.items()}
        if not any(missing.values()):
            break
        print(f"seed {seed}: missing labels {({k: sorted(v) for k, v in missing.items() if v})}, retrying")
    else:
        raise RuntimeError(f"No split with every label in every set after {MAX_ATTEMPTS} attempts")

    print(f"Split with seed {seed}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    phases = {'feedback_phase': 'valid', 'final_phase': 'test'}
    for phase, eval_split in phases.items():
        phase_dir = os.path.join(BUNDLE_DIR, phase)
        train_idx, test_idx = splits['train'], splits[eval_split]
        write_csv(os.path.join(phase_dir, 'input_data', 'train_data.csv'), in_header, [texts[i] for i in train_idx])
        write_csv(os.path.join(phase_dir, 'input_data', 'train_labels.csv'), ref_header, [refs[i] for i in train_idx])
        write_csv(os.path.join(phase_dir, 'input_data', 'test_data.csv'), in_header, [texts[i] for i in test_idx])
        write_csv(os.path.join(phase_dir, 'reference_data', 'test_labels.csv'), ref_header, [refs[i] for i in test_idx])
        print(f"Wrote {phase}: train={len(train_idx)}, {eval_split} as test={len(test_idx)}")

    # Per-label counts, to eyeball that the sets look alike
    for category in CATEGORIES:
        print(f"[{category}]")
        tags = sorted(t for c, t in all_labels if c == category)
        for tag in tags:
            counts = [sum((category, tag) in label_sets[i] for i in idx) for idx in splits.values()]
            print(f"  {tag:<24} " + "  ".join(f"{k} {c:>4}" for k, c in zip(splits, counts)))


if __name__ == '__main__':
    main()
