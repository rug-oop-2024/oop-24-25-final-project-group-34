from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification.decision_tree_regression import (
    DecisionTree)
from autoop.core.ml.model.classification.knn import KNearestNeighbor
from autoop.core.ml.model.classification.naive_bayes import NaiveBayesModel

from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression)
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.support_vector_regression import (
    SupportVectorRegression)

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "Lasso",
    "SupportVectorRegression"
]

CLASSIFICATION_MODELS = [
    "KNearestNeighbor",
    "DecisionTree",
    "NaiveBayesModel"
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name in REGRESSION_MODELS:
        if model_name == "MultipleLinearRegression":
            return MultipleLinearRegression()
        elif model_name == "Lasso":
            return Lasso()
        elif model_name == "SupportVectorRegression":
            return SupportVectorRegression()
    elif model_name in CLASSIFICATION_MODELS:
        if model_name == "KNearestNeighbor":
            return KNearestNeighbor()
        elif model_name == "DecisionTree":
            return DecisionTree()
        elif model_name == "NaiveBayesModel":
            return NaiveBayesModel()
    else:
        raise ValueError(f"Model '{model_name}' not found.")
