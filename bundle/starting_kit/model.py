import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class Model:
    """TF-IDF + Logistic Regression baseline for multi-category tagging."""

    def __init__(self):
        self.models = {}

    def fit(self, X, y):
        """Fit one TF-IDF vectorizer and logistic regression per tag in each category."""
        for category, Y in y.items():
            vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
            Z = vec.fit_transform(X)
            clfs = [
                LogisticRegression(class_weight='balanced', max_iter=1000).fit(Z, Y[:, j])
                for j in range(Y.shape[1])
            ]
            self.models[category] = (vec, clfs)
        return self

    def predict(self, X):
        """Predict binary label matrices for all categories."""
        predictions = {}
        for category, (vec, clfs) in self.models.items():
            Z = vec.transform(X)
            scores = np.column_stack([clf.predict_proba(Z)[:, 1] for clf in clfs])
            Y_pred = (scores >= 0.5).astype(int)

            # Fallback if no tag predicted
            untagged = Y_pred.sum(axis=1) == 0
            if untagged.any():
                Y_pred[untagged, scores[untagged].argmax(axis=1)] = 1

            predictions[category] = Y_pred
        return predictions
