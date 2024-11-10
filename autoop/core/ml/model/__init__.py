
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression, Lasso, SupportVectorRegression
from autoop.core.ml.model.classification import KNearestNeighbor, DecisionTree, NaiveBayesModel

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
