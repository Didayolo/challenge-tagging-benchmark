# Evaluation

## Protocol

The labeled dataset is shuffled and split once into three sets:

| Set | Samples | Used in |
|-----|---------|---------|
| Train | 3030 | both phases (training) |
| Validation | 1030 | Feedback Phase (evaluation) |
| Test | 1030 | Final Phase (evaluation) |

Every tag appears at least once in each of the three sets.

For each submission:

1. The **ingestion program** imports the participant's `Model` class from `model.py`.
2. It calls `model.fit(X_train, y_train)` where `y_train` is a dictionary `{category: binary_matrix}`.
3. It calls `model.predict(X_test)` on the validation set (Feedback Phase) or the test set (Final Phase), which must return `{category: binary_matrix}`.
4. The **scoring program** compares the predictions with the ground-truth labels and computes the Macro F1 of each category.

Participants never see the validation and test labels; evaluation happens entirely inside the scoring container.

## Phases

- **Feedback Phase** — submit as often as you like and get scored on the validation set.
- **Final Phase** — a single submission, trained on the same training set and scored on the unseen test set.

## Metrics — Macro F1 per category

For each category $c$ (Machine Learning Domain, Fields of Application):

$$\text{F1}_c = \frac{1}{|T_c|} \sum_{t \in T_c} \text{F1}(y_{\text{true}, c, t}, y_{\text{pred}, c, t})$$

where $T_c$ is the set of tags in category $c$ with at least one positive sample in the evaluated set.

**Why Macro F1?** Macro averaging gives equal weight to each label regardless of class frequency. This accounts for the long-tail distribution of tags.

## Ranking — Average Rank

Submissions are ranked on each of the two F1 columns separately (higher F1 = better rank). The primary leaderboard column is the **average of these two ranks** (lower is better), so both categories weigh equally regardless of their score scale.

## Leaderboard Columns

| Leaderboard column | Computation | Direction |
|--------------------|-------------|-----------|
| `avg_rank` | Average rank over `ml_domain_f1` and `fields_f1` | Lower = better (**ranking column**) |
| `ml_domain_f1` | Macro F1 on Machine Learning Domain | Higher = better |
| `fields_f1` | Macro F1 on Fields of Application | Higher = better |
| `train_time_s` | Wall-clock seconds spent in `model.fit()` | Lower = faster training |
| `infer_time_ms_per_sample` | Milliseconds spent in `model.predict()` per evaluated sample | Lower = faster inference |

## Label Binarization

Labels are encoded with `sklearn.preprocessing.MultiLabelBinarizer` fitted on the full known taxonomy for each category. Column order matches the tag lists in `TAXONOMY` (see the Overview page).

---

*Citation: [Pavão et al., Ch. 2 §2.1](https://ai-competitions-book.github.io/ai-competitions-book-full-project.pdf)*
