from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    """
    A class representing a pineline for training and evaluating models.
    """
    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split=0.8
                 ) -> None:
        """
        Initializes the Pipeline with the variables needed.

        Args:
            metrics (List[Metric]): List of metrics for evaluation.
            dataset (Dataset): The dataset to be used.
            model (Model): The model used for training.
            input_features (List[Feature]): The list of input features.
            target_feature (Feature): The target feature for the model.
            split (float, optional): Part of the data that will be used
            for traing. Defaults to 0.8.

        Raises:
            ValueError: Raises valueerror if target feature does not
            match the model type
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (target_feature.type == "categorical" and  # noqa: W504
                model.type != "classification"):
            raise ValueError("""Model type must be classification
                             for categorical target feature""")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("""Model type must be regression
                             for continuous target feature""")

    def __str__(self) -> str:
        """
        Returns a string that represents the pipeline.

        Returns:
            str: A string with the pipeline information.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> None:
        """
        Public getter for the pipeline model.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during the pipeline execution
        to be saved.

        Returns:
            List[Artifact]: A list of the artifacts.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(name=f"""pipeline_model_
                                                 {self._model.type}"""))
        return artifacts

    def _register_artifact(self, name: str, artifact) -> None:
        """
        Registers an artifact in the Pipeline.

        Args:
            name (str): The name of the artifact.
            artifact: The artifact to be registerd.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses the features in the Dataset and
        registers the artifacts.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get input and output vectors, sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact) in
                               input_results]

    def _split_data(self) -> None:
        """
        Splits the data into training and testing sets.
        """
        split = self._split
        self._train_X = [vector[:int(split * len(vector))] for vector in
                         self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):] for vector in
                        self._input_vectors]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Concatenates a list of vectors into one.

        Args:
            vectors (List[np.array]): A list of vectors.

        Returns:
            np.array: The concatenated vector
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model using the training data.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluates the model on the test data and the specified metrics.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def _evaluate_training_data(self) -> None:
        """
        Evaluates the model on the training data and the specified metrics
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._metrics_results_train = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results_train.append((metric, result))
        self._predictions_train = predictions

    def execute(self) -> dict:
        """
        Executes the pipeline.

        Returns:
            dict: A dictionary with the results.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        self._evaluate_training_data()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
            "metrics training data": self._metrics_results_train,
            "predictions training data": self._predictions_train
        }
