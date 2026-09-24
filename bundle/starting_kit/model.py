from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np


class _TextTagger:
    """TF-IDF features + one logistic regression per label.

    Two details matter on this dataset:

    * A label may be absent from a training fold (rare tags, small folds).
      LogisticRegression cannot fit a column with a single class, so those
      labels are stored as a constant instead.
    * Most labels are rare, so a plain 0.5 threshold predicts all-zeros and
      macro F1 collapses. Class weights counteract the imbalance, and any row
      left with no tag falls back to its highest-scoring label.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        self._classifiers = []  # one per label; None when the label is constant
        self._constants = []    # value to emit where _classifiers[j] is None

    def fit(self, X, Y):
        Z = self._vectorizer.fit_transform(X)
        self._classifiers, self._constants = [], []
        for j in range(Y.shape[1]):
            column = Y[:, j]
            if np.unique(column).size < 2:
                self._classifiers.append(None)
                self._constants.append(int(column[0]) if column.size else 0)
            else:
                clf = LogisticRegression(
                    max_iter=1000, random_state=42, class_weight='balanced'
                )
                clf.fit(Z, column)
                self._classifiers.append(clf)
                self._constants.append(0)
        return self

    def predict(self, X):
        Z = self._vectorizer.transform(X)
        scores = np.zeros((Z.shape[0], len(self._classifiers)))
        for j, clf in enumerate(self._classifiers):
            if clf is None:
                scores[:, j] = float(self._constants[j])
            else:
                scores[:, j] = clf.predict_proba(Z)[:, 1]

        Y = (scores >= 0.5).astype(int)
        untagged = Y.sum(axis=1) == 0
        if untagged.any():
            Y[untagged, scores[untagged].argmax(axis=1)] = 1
        return Y


class Model:
    """TF-IDF + Logistic Regression baseline for competition tagging.

    Two independent taggers, one per output column:
      ml_tagger    -> predicts ML_sector    binary matrix (n_samples x 11)
      field_tagger -> predicts Field_sector binary matrix (n_samples x 11)
    """

    def __init__(self):
        self._ml_tagger = _TextTagger()
        self._field_tagger = _TextTagger()

    def fit(self, X, y_ml, y_field):
        """Fit both taggers on training data."""
        self._ml_tagger.fit(X, np.asarray(y_ml))
        self._field_tagger.fit(X, np.asarray(y_field))
        return self

    def predict(self, X):
        """Predict label matrices for both output columns.

        Returns:
            (y_ml_pred, y_field_pred)  shape (n_samples, 11) and (n_samples, 11), dtype int.
        """
        return (
            np.asarray(self._ml_tagger.predict(X), dtype=int),
            np.asarray(self._field_tagger.predict(X), dtype=int),
        )
