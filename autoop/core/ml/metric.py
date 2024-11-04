from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "r-squared",
    "log loss",
    "mean absolute error"
] # add the names (in strings) of the metrics you implement

def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    raise NotImplementedError("To be implemented.")

class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number

    @abstractmethod
    def __call__(self, ground_truth: Any, prediction: Any) -> float:
        pass

class Accuracy(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
       return np.mean(ground_truth == prediction)

class MeanSquaredError(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        return np.mean((ground_truth - prediction) ** 2)

class RSquared(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        total_sum_squares = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        residual_sum_squares = np.sum((ground_truth - prediction) ** 2)

        r2 = 1 - (total_sum_squares / residual_sum_squares)
        
        return r2

class Recall(Metric):
    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray, num_classes: int) -> list:
        recalls = []

        for class_label in range(num_classes):
            tp = np.sum((ground_truth == class_label) & (prediction == class_label))
            fn = np.sum((ground_truth == class_label) & (prediction != class_label))

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)
        
        return recalls