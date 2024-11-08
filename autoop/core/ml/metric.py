from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "R_squared",
    "logarithmic loss",
    "recall"
]

def get_metric(name: str):
    """returns the wanted matric

    Args:
        name (str): name of the wanted metric

    Returns:
        _type: the wanted metric
    """
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "R_squared":
        return RSquared()
    elif name == "logarithmic loss":
        return LogLoss()
    elif name == "accuracy":
        return Accuracy()
    elif name == "recall":
        return Recall()

class Metric(ABC):
    """
    Base class for all metrics.
    """
    @abstractmethod
    def evaluate(self, 
                 ground_truth: np.ndarray, 
                 prediction: np.ndarray) -> float:
        """
        Calculates the metric based on the model's predictions 
        and the actual values.
        
        Args:
            ground_truth (np.ndarray): The actual target values.
            prediction (np.ndarray): The predicted values from the model.
        
        Returns:
            float: The calculated metric score.
        """
        pass


class MeanSquaredError(Metric):
    """
    Regression metric that calculates the mean squared error
    """
    def evaluate(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """Calculates the mean squared error for a models predictions.

        Args:
            ground_truth (np.ndarray): The actual values
            prediction (np.ndarray): The predicted values from the model

        Returns:
            float: The mean squared error between the ground truth and
            the predictions.
        """
        return np.mean((ground_truth - prediction) ** 2)


class MeanAbsoluteError(Metric):
    """
    Regression metric that calculates the mean absolute error
    """
    def evaluate(self,
                 ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Calculates the mean absolute error for a models predictions

        Args:
            ground_truth (np.ndarray): The actual values
            prediction (np.ndarray): The predicted values from the model

        Returns:
            float: The mean absolute error. The average absolute difference
            between the predicted and actual values.
        """
        return np.mean(np.abs(ground_truth - prediction))


class RSquared(Metric):
    """
    Regression metric that calculates the R-squared
    """
    def evaluate(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """Calculates R-squared for the model's predictions.

        Args:
            ground_truth (np.ndarray): The actual values
            prediction (np.ndarray): The predicted values from the model

        Returns:
            float: The R-squared. The variable ranges from 0 to 1. The higher
            the number, the better a ground truth matches the prediction.
        """
        total_sum_squares = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        residual_sum_squares = np.sum((ground_truth - prediction) ** 2)

        r2 = 1 - (total_sum_squares / residual_sum_squares)
        return r2


class LogLoss(Metric):
    """
    Classification metric to calculate logarithmic loss
    """
    def evaluate(self,
                 ground_truth: np.ndarray,
                 prediction: np.ndarray) -> float:
        """Calculate the logarithmic loss of a models predictions.

        Args:
            ground_truth (np.ndarray): The actual values
            prediction (np.ndarray): The predicted values from the model

        Returns:
            float: The logarithmic loss value. The difference between the 
            predicted probability compared to the ground truth
        """
        prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
        
        if len(prediction.shape) > 1:
            prediction = prediction[np.arange(len(prediction)),
                                    ground_truth.astype(int)]
        
        log_loss_value = -np.mean(ground_truth * np.log(prediction) +
                                  (1 - ground_truth) * np.log(1 - prediction))
        return log_loss_value


class Accuracy(Metric):
    """
    Classification metric to calculate the Accuracy
    """
    def evaluate(self,
               ground_truth: np.ndarray,
               prediction: np.ndarray) -> float:
        """
        Calculates the Accuracy of a models predictions.

        Args:
            ground_truth (np.ndarray): The actual values
            prediction (np.ndarray): The predicted values from the model

        Returns:
            float: The accuracy. The variable ranges from 0 to 1. The
            higher the variable, the higher the accuracy.
        """
        return np.sum(ground_truth == prediction) / len(ground_truth)


class Recall(Metric):
    """
    Classification metric to calculate the Recall
    """
    def evaluate(self,
               ground_truth: np.ndarray,
               prediction: np.ndarray,
               num_classes: int) -> list:
        """Calculates the Recall of a models predictions.

        Args:
            ground_truth (np.ndarray): The actual values
            prediction (np.ndarray): The predicted values from the model
            num_classes (int): The number of classes in the dataset

        Returns:
            list: A list of recall values for each class
        """
        recalls = []

        for class_label in range(num_classes):
            tp = np.sum((ground_truth == class_label) & (prediction == class_label))
            fn = np.sum((ground_truth == class_label) & (prediction != class_label))

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)
        
        return recalls
