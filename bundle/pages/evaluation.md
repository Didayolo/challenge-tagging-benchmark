# Evaluation

## Protocol

Submitted models are evaluated via **4-fold cross-validation** on the full labeled dataset. The scoring program:

1. Loads the full dataset (title + description text and ground-truth labels).
2. Imports the participant's `Model` class from `model.py`.
3. Splits into 4 folds using `sklearn.model_selection.KFold(n_splits=4, shuffle=True, random_state=42)`.
4. For each fold: instantiates a fresh `Model()`, calls `model.fit(X_train, y_ml_train, y_field_train)`, then `model.predict(X_test)`.
5. Aggregates timing and per-fold F1 scores.

Participants never see the full labeled dataset; labels are only used inside the scoring program.

## Primary Metric — Mean Macro F1

For each fold *k*:

```
f1_ml_k    = sklearn.metrics.f1_score(y_true_ml_k, y_pred_ml_k, average='macro', zero_division=0)
f1_field_k = sklearn.metrics.f1_score(y_true_field_k, y_pred_field_k, average='macro', zero_division=0)
fold_score_k = (f1_ml_k + f1_field_k) / 2.0
```

Primary score (higher is better, range 0–1):

```
primary = mean([fold_score_1, fold_score_2, fold_score_3, fold_score_4])
```

**Why Macro F1?** Macro averaging gives equal weight to each label regardless of class frequency. This is appropriate for the long-tail label distribution where some tags (e.g., "Federated & Privacy-Preserving Learning") appear rarely. Averaging across both output columns treats ML_sector and Field_sector as equally important.

## Secondary Metrics

| Leaderboard column | Computation | Direction |
|--------------------|-------------|-----------|
| `f1_std` | `numpy.std([fold_score_1, ..., fold_score_4])` | Lower = more stable |
| `train_time_s` | Wall-clock seconds for all 4 `.fit()` calls combined | Lower = faster training |
| `infer_time_ms_per_sample` | Total wall-clock ms for all 4 `.predict()` calls ÷ total test samples across folds | Lower = faster inference |

## Label Binarization

Labels are encoded with `sklearn.preprocessing.MultiLabelBinarizer` fitted on the full known taxonomy for each column. All labels are always present in the binarized matrix, even if absent from a given fold. The fixed class order matches the taxonomy tables on the Overview page.

## Leaderboard Column Order

`primary` → `f1_std` → `train_time_s` → `infer_time_ms_per_sample`

The primary column (`primary`, higher is better) determines ranking. Subsequent columns tiebreak left to right.

---

*Citation: [Pavão et al., Ch. 2 §2.1](https://ai-competitions-book.github.io/ai-competitions-book-full-project.pdf)*
