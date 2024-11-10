
from abc import abstractmethod, ABC
import numpy as np


class Model(ABC):
    """
    This class is an abstract class and serves as a blueprint
    for the models.
    """
    def __init__(self) -> None:
        """
        Initilizes the model and data.
        """
        self._model = None
        self._data = None

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the model to the training data.
        It stores the observations and their corresponding
        ground truths (labels).
        Args:
            observations (np.ndarray): Training data (features)
            ground_truth (np.ndarray): Training labels (target classes)
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        This returns a numpy array with the
        predictions that the model made.

        Args:
            observations (np.ndarray): A matrix with each row representing
                an observation and each column representing a feature.
                and features of the observation.

        Returns:
            np.ndarray: Predictions of the model
        """
        pass
