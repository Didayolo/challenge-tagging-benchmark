#!/usr/bin/env python3
"""Scoring program for codabench-tagger.

Reads:
  {input_dir}/ref/reference_data.csv -- ground-truth labels
  {input_dir}/res/input_data.csv     -- competition texts (copied by ingestion)
  {input_dir}/res/model.py           -- participant's Model class (copied by ingestion)

Both CSVs are semicolon-separated and aligned row by row: row i of
input_data.csv is described by row i of reference_data.csv. Within a label
cell, multiple labels are comma-separated.

reference_data.csv has one column per TAXONOMY category, named exactly like the
category, e.g.:
    Machine Learning Domain;Fields of Application;Conferences
    NLP,LLM;Legal,Education;ACL

TAXONOMY below is the only place to edit to add a tag or a whole category
(the reference CSV must then contain the matching column / labels).

Model interface (generic, whatever the number of categories):
    model.fit(X_train, y_train)   # y_train: {category: (n, n_tags) 0/1 array}
    model.predict(X_test)         # -> {category: (n, n_tags) 0/1 array}
Columns of each array follow the order of the tag list in TAXONOMY.

Evaluation: ONE multi-label stratified train/test split (no cross-validation).
  * the test set holds TEST_SIZE of the samples;
  * every tag present in the dataset appears at least once in the test set
    (and at least once in the train set when it has 2+ positives);
  * each tag's share in the test set mirrors its share in the train set.
Tags that never appear in the dataset are excluded from the averages.

Writes {output_dir}/scores.json:
  primary                  -- mean over categories of the macro-F1 on the test set
  f1_std                   -- std of the per-tag F1 scores (spread across tags)
  train_time_s             -- time spent in fit()
  infer_time_ms_per_sample -- predict() time per test sample
"""
import csv
import importlib.util
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

csv.field_size_limit(sys.maxsize)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
TEST_SIZE = 0.25   # fraction of ALL samples held out as the test set (0 < x < 1)
SEED = 42          # makes the split reproducible

# Column separator used by the dataset CSVs.
DELIMITER = ';'
# Separator between several labels inside a single label cell.
LABEL_SEPARATOR = ','

# Single source of truth: category -> list of allowed tags.
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
# Taxonomy / data loading
# ----------------------------------------------------------------------------
def validate_config(taxonomy, test_size):
    """Fail early on an unusable TAXONOMY or TEST_SIZE."""
    if not 0 < test_size < 1:
        raise ValueError(f"TEST_SIZE must be strictly between 0 and 1, got {test_size}")
    if not taxonomy:
        raise ValueError("TAXONOMY is empty")
    for category, tags in taxonomy.items():
        if DELIMITER in category or LABEL_SEPARATOR in category:
            raise ValueError(f"Category name {category!r} must not contain '{DELIMITER}' or '{LABEL_SEPARATOR}'")
        if not tags:
            raise ValueError(f"Category {category!r} has no tags")
        if len(set(tags)) != len(tags):
            raise ValueError(f"Category {category!r} contains duplicate tags")
        for tag in tags:
            if DELIMITER in tag or LABEL_SEPARATOR in tag:
                raise ValueError(f"Tag {tag!r} in {category!r} must not contain '{DELIMITER}' or '{LABEL_SEPARATOR}'")


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


def load_data(ref_dir, res_dir, taxonomy):
    """Load texts from the ingestion output and labels from the reference data.

    The two files carry no key column; they are matched by row order, so a
    length mismatch means the task data is inconsistent and scoring must stop.
    Returns (texts, {category: [list of tags per row]}).
    """
    input_path = os.path.join(res_dir, 'input_data.csv')
    ref_path = os.path.join(ref_dir, 'reference_data.csv')

    input_rows = _read_csv(input_path, ['text'])
    ref_rows = _read_csv(ref_path, list(taxonomy))

    if len(input_rows) != len(ref_rows):
        raise ValueError(
            f"Row count mismatch: {input_path} has {len(input_rows)} rows but "
            f"{ref_path} has {len(ref_rows)}. The two files are aligned by row "
            "order and must have the same length."
        )

    texts = [row['text'] for row in input_rows]
    labels = {c: [_split_labels(row[c]) for row in ref_rows] for c in taxonomy}
    return texts, labels


def binarize(labels, taxonomy):
    """One 0/1 matrix per category, columns in TAXONOMY tag order.

    Raises on any label that is not in TAXONOMY (instead of silently dropping it).
    """
    y = {}
    for category, tags in taxonomy.items():
        allowed = set(tags)
        errors = [
            f"row {i}: {t!r}"
            for i, row in enumerate(labels[category])
            for t in row
            if t not in allowed
        ]
        if errors:
            shown = '; '.join(errors[:5]) + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else '')
            raise ValueError(f"Unknown label(s) for category {category!r}: {shown}")
        y[category] = MultiLabelBinarizer(classes=tags).fit_transform(labels[category])
    return y


# ----------------------------------------------------------------------------
# Multi-label stratified train/test split
# ----------------------------------------------------------------------------
def stratified_split(Y, test_size, seed):
    """Single multi-label stratified split. Y: (N, T) 0/1 over ALL tags.

    Iterative stratification (Sechidis et al., 2011): tags are handled from the
    rarest to the most frequent, and each sample goes to the subset that still
    needs the most positives of that tag. The target test count of a tag is
    test_size * count, but at least 1 (tag present in the test set) and, for
    tags with 2+ positives, at most count - 1 (tag still present in train).
    A final repair step swaps samples if a tag still ended up missing.

    Returns (train_idx, test_idx), sorted.
    """
    rng = np.random.default_rng(seed)
    n, n_tags = Y.shape
    n_test = min(max(int(round(test_size * n)), 1), n - 1)
    frac = n_test / n

    counts = Y.sum(axis=0).astype(float)
    d_test = frac * counts
    d_test = np.where(counts >= 1, np.maximum(d_test, 1.0), 0.0)
    d_test = np.where(counts >= 2, np.minimum(d_test, counts - 1.0), d_test)
    desired = np.stack([counts - d_test, d_test])      # row 0 = train, row 1 = test
    cap = np.array([n - n_test, n_test], dtype=int)    # remaining capacity per subset

    assign = np.full(n, -1, dtype=int)
    remaining = np.ones(n, dtype=bool)
    has_labels = Y.sum(axis=1) > 0

    while True:
        avail = remaining & has_labels
        if not avail.any():
            break
        cnt = Y[avail].sum(axis=0)
        cnt = np.where(cnt > 0, cnt, np.iinfo(np.int64).max)
        t = rng.choice(np.flatnonzero(cnt == cnt.min()))     # rarest remaining tag

        idxs = np.flatnonzero(avail & (Y[:, t] == 1))
        rng.shuffle(idxs)
        for i in idxs:
            ok = cap > 0
            score = np.where(ok, desired[:, t], -np.inf)
            best = np.flatnonzero(score == score.max())
            if len(best) > 1:                                # tie: most remaining capacity
                best = best[cap[best] == cap[best].max()]
            s = int(rng.choice(best))
            assign[i] = s
            remaining[i] = False
            cap[s] -= 1
            desired[s] -= Y[i]

    # Samples without any label just fill the remaining capacity
    rest = np.flatnonzero(remaining)
    rng.shuffle(rest)
    assign[rest[:cap[0]]] = 0
    assign[rest[cap[0]:]] = 1

    assign = _repair_presence(Y, assign, seed)
    return np.flatnonzero(assign == 0), np.flatnonzero(assign == 1)


def _repair_presence(Y, assign, seed):
    """Swap one sample each way until every tag is present where it must be."""
    rng = np.random.default_rng(seed + 1)
    counts = Y.sum(axis=0)
    need = {1: counts >= 1, 0: counts >= 2}   # subset 1 = test, 0 = train

    for sub in (1, 0):
        other = 1 - sub
        for t in np.flatnonzero(need[sub]):
            if Y[assign == sub, t].sum() > 0:
                continue
            cand_p = np.flatnonzero((assign == other) & (Y[:, t] == 1))
            rng.shuffle(cand_p)
            cand_s = np.flatnonzero(assign == sub)
            rng.shuffle(cand_s)
            fixed = False
            for p in cand_p:
                cnt_other = Y[assign == other].sum(axis=0) - Y[p]
                if (need[other] & (cnt_other < 1)).any():
                    continue                       # p cannot leave `other`
                cnt_sub = Y[assign == sub].sum(axis=0) + Y[p]
                for s in cand_s:
                    if (need[sub] & (cnt_sub - Y[s] < 1)).any():
                        continue                   # s cannot leave `sub`
                    assign[p], assign[s] = sub, other
                    fixed = True
                    break
                if fixed:
                    break
            if not fixed:
                print(f"WARNING: could not place tag index {t} in the {'test' if sub else 'train'} set")
    return assign


def print_split_report(y, train_idx, test_idx, taxonomy):
    n_tr, n_te = len(train_idx), len(test_idx)
    print(f"Split: {n_tr} train / {n_te} test samples (TEST_SIZE={TEST_SIZE}, SEED={SEED})")
    for category, tags in taxonomy.items():
        print(f"[{category}]")
        for j, tag in enumerate(tags):
            a, b = int(y[category][train_idx, j].sum()), int(y[category][test_idx, j].sum())
            note = ""
            if a + b == 0:
                note = "  <- never appears in the dataset: excluded from scoring"
            elif a == 0:
                note = "  <- only 1 positive in the dataset: none left in train"
            print(f"  {tag:<24} train {a:>5} ({100 * a / n_tr:5.1f}%)   test {b:>4} ({100 * b / n_te:5.1f}%){note}")


# ----------------------------------------------------------------------------
# Model loading / prediction checks
# ----------------------------------------------------------------------------
def check_predictions(y_pred, y_true, taxonomy):
    """Make sure predict() returned one correctly-shaped array per category."""
    if not isinstance(y_pred, dict):
        raise TypeError(
            "model.predict() must return a dict {category: array}, "
            f"got {type(y_pred).__name__}. Expected categories: {list(taxonomy)}"
        )
    out = {}
    for category in taxonomy:
        if category not in y_pred:
            raise KeyError(f"model.predict() output is missing category {category!r}")
        arr = np.asarray(y_pred[category])
        if arr.shape != y_true[category].shape:
            raise ValueError(
                f"Prediction for {category!r} has shape {arr.shape}, "
                f"expected {y_true[category].shape} (n_samples, n_tags)"
            )
        out[category] = arr
    return out


def load_model_class(res_dir):
    model_path = os.path.join(res_dir, 'model.py')
    spec = importlib.util.spec_from_file_location('submission_model', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


# ----------------------------------------------------------------------------
def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else '/app/input'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/app/output'

    ref_dir = os.path.join(input_dir, 'ref')
    res_dir = os.path.join(input_dir, 'res')

    os.makedirs(output_dir, exist_ok=True)

    validate_config(TAXONOMY, TEST_SIZE)

    # Texts come from the ingestion output, labels from reference_data
    texts, labels = load_data(ref_dir, res_dir, TAXONOMY)
    print(f"Loaded {len(texts)} samples, {len(TAXONOMY)} categories: {list(TAXONOMY)}")

    y = binarize(labels, TAXONOMY)
    X = np.array(texts, dtype=object)

    # One stratified split over all tags of all categories
    Y_all = np.hstack([y[c] for c in TAXONOMY])
    train_idx, test_idx = stratified_split(Y_all, TEST_SIZE, SEED)
    # print_split_report(y, train_idx, test_idx, TAXONOMY)

    X_train, X_test = X[train_idx].tolist(), X[test_idx].tolist()
    y_train = {c: y[c][train_idx] for c in TAXONOMY}
    y_test = {c: y[c][test_idx] for c in TAXONOMY}

    # Load participant's Model class (passed via ingestion output)
    model = load_model_class(res_dir)()

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    raw_pred = model.predict(X_test)
    infer_time_ms = (time.perf_counter() - t1) * 1000.0

    y_pred = check_predictions(raw_pred, y_test, TAXONOMY)

    # Macro-F1 per category over the tags present in the dataset, then mean over categories
    category_scores = {}
    all_tag_f1 = []
    for category, tags in TAXONOMY.items():
        f1 = f1_score(y_test[category], y_pred[category], average=None, zero_division=0)
        support = y_test[category].sum(axis=0)
        keep = support > 0
        if not keep.any():
            print(f"Category {category!r}: no tag present in the dataset, ignored")
            continue
        category_scores[category] = float(f1[keep].mean())
        all_tag_f1.extend(f1[keep].tolist())
        for name, s, n_pos in sorted(zip(np.array(tags)[keep], f1[keep], support[keep]), key=lambda x: x[1]):
            print(f"  [{category}] {name:<24} F1={s:.2f}  (test positives: {n_pos})")

    if not category_scores:
        raise ValueError("No tag present in the dataset: nothing to score")

    print("\n" + "  ".join(f"{c}={s:.4f}" for c, s in category_scores.items()))

    scores = {
        'primary': float(np.mean(list(category_scores.values()))),
        'f1_std': float(np.std(all_tag_f1)),
        'train_time_s': float(train_time_s),
        'infer_time_ms_per_sample': float(infer_time_ms / len(X_test)),
    }

    scores_path = os.path.join(output_dir, 'scores.json')
    with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2)

    print(f"\nFinal scores: {scores}")


if __name__ == '__main__':
    main()
