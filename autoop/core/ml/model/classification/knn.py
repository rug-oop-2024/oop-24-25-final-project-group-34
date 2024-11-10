from autoop.core.ml.model import Model
import numpy as np
from typing import Any
from collections import Counter


class KNearestNeighbor(Model):
    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self._k = k
        self._parameters = {"observations": None,
                            "ground_truth": None}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the model to the training data.
        It stores the observations and their corresponding
        ground truths (labels).
        Args:
            observations (np.ndarray): Training data (features)
            ground_truth (np.ndarray): Training labels (target classes)
        """
        self._parameters["observations"] = observations
        self._parameters["ground_truth"] = ground_truth

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the class for each observation in the input class.
        For each input data it predicts the distances to all the
        training points and returns the most common class of the
        three nearest training points.
        Returns:
            array: Predicted class for the input data.
        Raises:
            ValueError: If parameters are empty.
        """
        if self._parameters is None:
            raise ValueError("Model not fitted. Call 'fit' with appropriate "
                             "arguments before using 'predict'.")

        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> Any:
        """
        Predicts the class of a single data input.
        It calculates the distances between the single data input and
        the other data inputs. Select the k-nearest neighbors, and return
        the most frequent class among them.
        Args:
            observation (ndarray): A single input observation for
            which the class is predicted.
        Returns:
            Any: returns the predicted class for the single observation.
        """
        distances = np.linalg.norm(observation -
                                   self._parameters["observations"], axis=1)
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[:self._k]
        k_nearest_labels = [self._parameters["ground_truth"]
                            [i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
