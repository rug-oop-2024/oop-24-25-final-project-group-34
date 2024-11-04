
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model(ABC):
    """
    This class is an abstract class and serves as a blueprint
    for the models.
    """
    def __init__(self):
        """
        more docsstrings
        """
        self._model = None
        self._data = None

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Train the model with the given data

        Args:
            observations (np.ndarray): IDK
            ground_truth (np.ndarray): IDK
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """        This returns a numpy array with the
        predictions that the model made.

        Args:
            observations (np.ndarray): IDK

        Returns:
            np.ndarray: Predictions of the model
        """
        pass
