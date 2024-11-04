from autoop.core.ml.model import Model
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class DecisionTree(Model):
    def __init__(self) -> None:
        super().__init__()
        self._model = DecisionTreeClassifier()
    
    def fit(self,
            observations: np.ndarray,
            ground_truth: np.ndarray) -> None:
        self._model.fit(observations, ground_truth)
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._model.predict(observations)
