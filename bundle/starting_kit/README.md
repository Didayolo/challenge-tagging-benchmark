# Codabench Competition Tagger — Baseline Submission

## How to submit

Create a zip archive containing `model.py` at its root and upload it to the competition.

```
my_submission.zip
└── model.py
```

## Sample data

Two `;`-separated files, aligned row by row (no identifier column):

- `sample_input_data.csv` — column `text`
- `sample_reference_data.csv` — columns `Machine Learning Domain`, `Fields of Application` (Conferences column optional)

Inside a label cell, several labels are separated by a comma (`,`). Read them
with an explicit delimiter, e.g. `csv.DictReader(f, delimiter=';')` or
`pandas.read_csv(path, sep=';')`.

## Model interface

Your `model.py` must define a `Model` class with:

```python
class Model:
    def fit(self, X_train: list[str], y_train: dict[str, np.ndarray]) -> 'Model':
        """Train on labelled data.
        X_train  -- list of N text strings ("title : description")
        y_train  -- dict mapping category name -> binary array (N, n_tags)
        """
        ...

    def predict(self, X_test: list[str]) -> dict[str, np.ndarray]:
        """Predict label matrices.
        Returns dict mapping category name -> binary array (M, n_tags), dtype int.
        """
        ...
```

Columns of each array follow the order of the tag list in `TAXONOMY` defined in the ingestion and scoring programs.

The ingestion program trains your model on the training set (2326 samples) and predicts the validation set (Feedback Phase) or the test set (Final Phase), 774 samples each.

## This baseline

TF-IDF + one logistic regression per label across each category. Class weights are balanced because most tags are rare, and a row that crosses no threshold falls back to its highest-scoring tag, so no row is left untagged.
