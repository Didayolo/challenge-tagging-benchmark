"""Participant model template for Codabench Competition Tagger.

Replace the body of each method with your implementation.
Keep the class name `Model` and the method signatures exactly as shown.
"""
from typing import Dict, List
import numpy as np


class Model:
    def __init__(self):
        # Initialise your model here.
        pass

    def fit(self, X_train: List[str], y_train: Dict[str, np.ndarray]) -> 'Model':
        """Train your model.

        Args:
            X_train : list of N text strings, each "<title> : <description>"
            y_train : dict mapping category name -> 0/1 binary np.ndarray of shape (N, n_tags)

        Returns:
            self
        """
        raise NotImplementedError

    def predict(self, X_test: List[str]) -> Dict[str, np.ndarray]:
        """Predict multi-label 0/1 matrices.

        Args:
            X_test : list of M text strings

        Returns:
            dict mapping category name -> 0/1 binary np.ndarray of shape (M, n_tags)
        """
        raise NotImplementedError
