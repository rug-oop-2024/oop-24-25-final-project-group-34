from autoop.core.ml.model import Model
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class DecisionTree(Model):
    """Decision Tree Model.

    This model splits the data into subsets based on the
    feature values to make predictions. Once the tree is
    built the predictions can be made.

    """
    def __init__(self) -> None:
        """Initializes the decision tree model.
        """
        super().__init__()
        self._model = DecisionTreeClassifier()
        self._type = "classification"

    @property
    def type(self) -> str:
        """Public getter for the type variable."""
        return self._type

    def fit(self,
            observations: np.ndarray,
            ground_truth: np.ndarray) -> None:
        """Fits the model to the training data.
        It stores the observations and their
        corresponding ground truths.

        Args:
            observations (np.ndarray): Training data (features)
            ground_truth (np.ndarray): Training labels (target
            classes)
        """
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """This method predicts the target.

        Args:
            observations (np.ndarray): An input observation
            for which the target value is predicted.

        Returns:
            np.ndarray: Returns the predicted target values.
        """
        return self._model.predict_proba(observations)
