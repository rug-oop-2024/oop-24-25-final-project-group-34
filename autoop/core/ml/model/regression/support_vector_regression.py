import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.svm import SVR as skSVR


class SupportVectorRegression(Model):
    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1):
        super().__init__()
        self.svr = skSVR(kernel=kernel, C=C, epsilon=epsilon)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        self.svr.fit(observations, ground_truth)
        self._parameters = {
            "support_vectors": self.svr.support_vectors_,
            "coef": self.svr.dual_coef_,
            "intercept": self.svr.intercept_
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        predictions = self.svr.predict(observations)
        return predictions
