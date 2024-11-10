import numpy as np
from autoop.core.ml.model import Model


class MultipleLinearRegression(Model):
    """
    Multiple Linear Regression (MLR) model.
    This class implements the MLR  model, where the model is fit
    to the training data, and then predicts values.
    """

    def fit(self,
            observations: np.ndarray,
            ground_truth: np.ndarray) -> None:
        """
        This method fits the model to the training data.
        It creates a matrix, and calculates the optimal weights.
        Args:
            observations (np.ndarray): A matrix with each row representing
                an observation and each column representing a feature.
                and features of the observation.
            ground_truth (np.ndarray): A 1D array representing the
                true values of the dependent variable for each observation
                in the matrix.
        """
        if len(observations.shape) == 1:
            observations = observations.reshape(-1, 1)
        ground_truth = ground_truth.ravel()

        assert len(observations.shape) == 2
        assert len(ground_truth.shape) == 1

        added_ones = np.ones((observations.shape[0], 1))
        matrix_b = np.hstack((observations, added_ones))
        transposed = np.transpose(matrix_b)

        optimal_weights = np.matmul(np.linalg.inv
                                    (np.matmul(transposed, matrix_b)),
                                    (np.matmul(transposed, ground_truth)))

        self._parameters = {
            "weights": optimal_weights
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        This method predicts the target values
        based on the training values.
        The method used the learned model
        parameters to predict the target values.
        It predicts the values by using a
        matrix with the added ones for the intercept,
        and then performing a matrix multiplication with the coefficients.
        Args:
            observations (np.ndarray): A matrix with each row representing
                an observation and each column representing a feature.
        Returns:
            np.ndarray: Predicted target values as a 1D numpy array.
        Raises:
            ValueError: If parameters are empty.
        """
        if self._parameters is None:
            raise ValueError("Model not fitted. Call 'fit' with appropriate "
                             "arguments before using 'predict'.")

        matrix = np.hstack((np.ones((observations.shape[0], 1)), observations))
        prediction = matrix.dot(self._parameters["weights"])

        return np.array(prediction)

    @property
    def parameters(self) -> dict:
        """
        Public getter for parameters.
        """
        if self._parameters is None:
            raise ValueError("Model not fitted. Call 'fit' with appropriate "
                             "arguments before using 'get_parameters'.")
        return self._parameters
