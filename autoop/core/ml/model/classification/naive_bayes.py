from sklearn.naive_bayes import GaussianNB
import numpy as np
from autoop.core.ml.model import Model


class NaiveBayesModel(Model):
    def __init__(self) -> None:
        self._model = GaussianNB()

    def fit(self,
            observations: np.ndarray,
            ground_truth: np.ndarray) -> None:
        return self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._model.predict(observations)

