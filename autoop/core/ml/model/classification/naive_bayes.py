from sklearn.naive_bayes import GaussianNB
import numpy as np
from autoop.core.ml.model import Model


class NaiveBayesModel(Model):
    """Naive Bayes Model

    This model is a probabilistic classifier based
    on Bayes Theorem.
    """
    def __init__(self) -> None:
        """Initializes the Naive Bayes model."""
        self._model = GaussianNB()
        self._type = "classification"

    @property
    def type(self) -> str:
        """Public getter for the type variable."""
        return self._type

    def fit(self,
            observations: np.ndarray,
            ground_truth: np.ndarray) -> None:
        """Fits the Naive Bayes model to the
        training data.

        This method trains the model by estimating the
        parameters for each class based on the given
        observations and corresponding ground truth
        labels.

        Args:
            observations (np.ndarray): The training data (features)
            ground_truth (np.ndarray): The true labels (target class)
        """
        if ground_truth.ndim == 2:
            ground_truth = np.argmax(ground_truth, axis=1)

        return self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts the class labels for the given observations.

        This method uses the trained Naive Bayes model to predict
        the class labels for a set of input observations based on
        the learned distributions of features for each class.

        Args:
            observations (np.ndarray): The input data for which
            to predict the class labels.

        Returns:
            np.ndarray: Array containing the predicted class
            labels for each observation.
        """
        return self._model.predict(observations)
