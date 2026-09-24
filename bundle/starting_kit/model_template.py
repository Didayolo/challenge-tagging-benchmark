"""Participant model template for Codabench Competition Tagger.

Replace the body of each method with your implementation.
Keep the class name `Model` and the method signatures exactly as shown.
"""
import numpy as np


class Model:
    def __init__(self):
        # Initialise your model here.
        pass

    def fit(self, X, y_ml, y_field):
        """Train your model.

        Args:
            X       : list of N text strings, each "<title> : <description>"
            y_ml    : np.ndarray of shape (N, 11), binary, ML_sector labels
            y_field : np.ndarray of shape (N, 11), binary, Field_sector labels

        Returns:
            self
        """
        raise NotImplementedError

    def predict(self, X):
        """Predict label matrices.

        Args:
            X : list of N text strings

        Returns:
            (y_ml_pred, y_field_pred) : two np.ndarray of shape (N, 11) and (N, 11), dtype int.
        """
        raise NotImplementedError
