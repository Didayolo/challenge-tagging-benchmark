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
- `sample_reference_data.csv` — columns `ML_sector`, `Field_sector`

Inside a label cell, several labels are separated by a comma (`,`). Read them
with an explicit delimiter, e.g. `csv.DictReader(f, delimiter=';')` or
`pandas.read_csv(path, sep=';')`.

## Model interface

Your `model.py` must define a `Model` class with:

```python
class Model:
    def fit(self, X: list[str], y_ml: np.ndarray, y_field: np.ndarray) -> 'Model':
        """Train on labelled data.
        X        -- list of N text strings ("title : description")
        y_ml     -- binary array (N, 10) for ML_sector labels
        y_field  -- binary array (N, 10) for Field_sector labels
        """
        ...

    def predict(self, X: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Predict label matrices.
        Returns (y_ml_pred, y_field_pred) each of shape (N, 10), dtype int.
        """
        ...
```

## This baseline

TF-IDF + one logistic regression per label, with two independent taggers, one per
output column. Class weights are balanced because most tags are rare, and a row that
crosses no threshold falls back to its highest-scoring tag, so no row is left untagged.

Measured Mean Macro F1: ~0.66 under the competition's 4-fold CV on the full dataset.
