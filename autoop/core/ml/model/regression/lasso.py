import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso as skLasso


class Lasso(Model):
    """
    Lasso regression model.
    This class implements the lasso regression, which implements
    a penalty to prevent overfitting.
    """
    def __init__(self, alpha=1.0):
        """
        This initalizes the Lasso regression model.
        This constructor initializes the Lasso regression model
        from the scikit-learn library. The lasso attribute stores
        the instantiated Lasso model, which is used for the fit and
        predict method.

        Args:
            alpha (float, optional): Regularization strength, must be a positive float.
            A larger value applies a stronger penalty on the 
            coefficients, leading to greater feature selection.
            Default is 1.0.
        """
        super().__init__()
        self.lasso = skLasso(alpha=alpha)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        This method fits the model to the training data.
        It uses the fit method from the scikit-learn library. The fit method
        uses the provided observations (features)
        and ground_truth (target value).

        Args:
            observations (np.ndarray): A matrix with each row representing
            an observation and each column representing a feature.
            and features of the observation.
            ground_truth (np.ndarray): A 1D array representing the
            true values of the dependent variable for each observation
            in the matrix.
        """
        self.lasso.fit(observations, ground_truth)
        self._parameters = {
            "weights": self.lasso.coef_,
            "intercept": self.lasso.intercept_
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        This method predicts the target values
        based on the training values
        The method used the learned model
        parameters to predict the target values.
        It predicts the values by using a matrix
        multiplication with the learned coefficients.

        Args:
            observations (np.ndarray): A matrix with each row representing
            an observation and each column representing a feature.
            and features of the observation.

        Returns:
            np.ndarray: Predicted target values as a 1D numpy array.
        """
        return self.lasso.predict(observations)
