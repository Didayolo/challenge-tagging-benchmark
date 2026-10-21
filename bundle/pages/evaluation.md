# Evaluation

## Protocol

Submitted models are evaluated via a **single multi-label stratified train/test split** (TEST_SIZE=0.25, SEED=42) on the full labeled dataset:

1. Loads the full dataset (competition texts and ground-truth labels across the 2 categories).
2. Performs iterative multi-label stratification (Sechidis et al., 2011) ensuring each tag's share in the test set mirrors its share in the train set.
3. Imports the participant's `Model` class from `model.py`.
4. Calls `model.fit(X_train, y_train)` where `y_train` is a dictionary `{category: binary_matrix}`.
5. Calls `model.predict(X_test)` returning `{category: binary_matrix}`.
6. Computes Macro F1 per category over all tags present in the test set, then averages across categories.

Participants never see the test set labels; evaluation happens entirely inside the scoring container.

## Primary Metric — Mean Macro F1

For each category $c \in \text{TAXONOMY}$:

$$\text{F1}_c = \frac{1}{|T_c|} \sum_{t \in T_c} \text{F1}(y_{\text{true}, c, t}, y_{\text{pred}, c, t})$$

where $T_c$ is the set of tags in category $c$ with at least one positive sample in the test set.

Primary score (higher is better, range 0–1):

$$\text{primary} = \frac{1}{|\text{TAXONOMY}|} \sum_{c \in \text{TAXONOMY}} \text{F1}_c$$

**Why Macro F1?** Macro averaging gives equal weight to each label regardless of class frequency. This accounts for the long-tail distribution of tags while treating all categories equally.

## Secondary Metrics

| Leaderboard column | Computation | Direction |
|--------------------|-------------|-----------|
| `f1_std` | Standard deviation of per-tag F1 scores across all active tags | Lower = more balanced across tags |
| `train_time_s` | Wall-clock seconds spent in `model.fit()` | Lower = faster training |
| `infer_time_ms_per_sample` | Milliseconds spent in `model.predict()` per test sample | Lower = faster inference |

## Label Binarization

Labels are encoded with `sklearn.preprocessing.MultiLabelBinarizer` fitted on the full known taxonomy for each category. Column order matches the tag lists in `TAXONOMY`.

## Leaderboard Column Order

`primary` → `f1_std` → `train_time_s` → `infer_time_ms_per_sample`

The primary column (`primary`, higher is better) determines ranking. Subsequent columns tiebreak left to right.

---

*Citation: [Pavão et al., Ch. 2 §2.1](https://ai-competitions-book.github.io/ai-competitions-book-full-project.pdf)*
