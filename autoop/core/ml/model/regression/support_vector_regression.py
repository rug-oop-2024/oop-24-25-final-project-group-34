import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.svm import SVR as skSVR


class SupportVectorRegression(Model):
    """
    Support Vector Regression (SVR) model.
    """
    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1):
        """Initializes the Support Vector Regression model
        with specified parameters.

        Args:
            kernel (str, optional): Specifies the kernel type to be
            used in the SVR model.
            C (float, optional): Regularization parameter that controls
            the trade-off between achieving a low training error and a
            low testing error.
            epsilon (float, optional): Specifies the epsilon-tube. No
            penalty is associated in the training loss function with
            points predicted within a distance epsilon from the actual
            value.
        """
        super().__init__()
        self.svr = skSVR(kernel=kernel, C=C, epsilon=epsilon)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains the SVR model on the provided dataset.

        Args:
            observations (np.ndarray): Training data where
            each row represents an observation and each column
            represents a feature.
            ground_truth (np.ndarray): Target labels that correspond
            to each observation in the training data.
        """
        ground_truth = ground_truth.ravel()
        self.svr.fit(observations, ground_truth)
        self._parameters = {
            "support_vectors": self.svr.support_vectors_,
            "coef": self.svr.dual_coef_,
            "intercept": self.svr.intercept_
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained SVR model.

        Args:
            observations (np.ndarray): An array of data where
            each row represents an observation and each column
            represents a feature.

        Returns:
            np.ndarray: An array containing the predicted values
            for each observation.
        """
        predictions = self.svr.predict(observations)
        return predictions
