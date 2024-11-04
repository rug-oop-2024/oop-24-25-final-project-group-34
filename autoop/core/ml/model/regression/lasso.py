import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso as skLasso


class Lasso(Model):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.lasso = skLasso(alpha=alpha)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        self.lasso.fit(observations, ground_truth)
        self._parameters = {
            "weights": self.lasso.coef_,
            "intercept": self.lasso.intercept_
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.lasso.predict(observations)
