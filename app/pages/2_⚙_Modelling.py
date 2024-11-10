import streamlit as st
import pandas as pd
import io
from typing import List, Tuple, Dict

from app.core.system import AutoMLSystem
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset

from autoop.core.ml.model.classification.decision_tree_regression import (
    DecisionTree)
from autoop.core.ml.model.classification.knn import KNearestNeighbor
from autoop.core.ml.model.classification.naive_bayes import NaiveBayesModel

from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression as MLR)
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.support_vector_regression import (
    SupportVectorRegression)

from autoop.core.ml.metric import (
    MeanSquaredError,
    MeanAbsoluteError,
    RSquared,
    LogLoss,
    Accuracy,
    Recall
)


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """A helper function to display
    text in Streamlit with custom
    HTML styling.

    Args:
        text (str): A text string
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def get_dataset(datasets: List[Dataset]) -> Tuple[Dataset, pd.DataFrame]:
    """Lets the user select a dataset from a list of datasets.

    Args:
        datasets (List[Dataset]): A list of dataset objects.

    Returns:
        Tuple[Dataset, pd.DataFrame]:
            - selected_dataset (Dataset): The selected dataset
            or None if no dataset is selected.
            - data (pd.DataFrane): The data from the selected
            dataset.
    """
    if datasets:
        dataset_names = [dataset.name for dataset in datasets]
        selected_dataset_name = st.selectbox("Choose a dataset", dataset_names)

        selected_dataset = next((data for data in datasets
                                if data.name == selected_dataset_name), None)

        if selected_dataset:
            st.write("Preview of the uploaded dataset: ")
            data_bytes = selected_dataset.read()
            data = pd.read_csv(io.StringIO(data_bytes.decode("utf-8")))
            st.dataframe(data.head())
            return selected_dataset, data
    st.warning("No datasets available. Please upload a dataset to proceed.")
    return None, None


def get_features(selected_dataset: Dataset) -> Tuple[List[dict], str,
                                                     List[str]]:
    """This function detects feature types from the provided dataset
    lets the user select input and target features.

    Args:
        selected_dataset (Dataset): The features will be detected from
        this dataset.

    Returns:
        Tuple[List[dict], str, List[str]]:
            - list_feature (List[dict]): A list of dictionaries,
            which each contain the name and type of the detected
            feature.
            - target_feature (str): The name of the selected
            target feature.
            - input_features (List[str]): A list of selected
            input feature names.
    """
    features = detect_feature_types(selected_dataset)
    list_feature = [{"name": feature.name,
                    "type": feature.type} for feature in features]
    complete_features_names = [feature["name"] for feature in list_feature]
    target_feature = st.selectbox("Select Target Feature",
                                  options=complete_features_names)

    target_feature_type = next((feature["type"] for feature in list_feature if
                                feature["name"] == target_feature), None)
    st.write(f"Selected Target Feature Type: {target_feature_type}")

    remaining_input_features = [
        name for name in complete_features_names if name != target_feature]
    input_features = st.multiselect("Select Input Features",
                                    options=remaining_input_features)
    return list_feature, target_feature, input_features


def get_model(task_type: str) -> Tuple[str, object]:
    """This function lets the user select a model based
    on the task type (Regression or Classification).

    Args:
        task_type (str): The type of task, which determines
        the available models.

    Returns:
        Tuple[str, object]:
            - selected_model (str): The name of the
            selected model.
            - model_instance (object): An instance
            of the selected model class.
    """
    if task_type == "Regression":
        model_options = ["Lasso",
                         "Multiple Linear Regression",
                         "Support Vector Regression"]
        model_mapping = {
                         "Lasso": Lasso,
                         "Multiple Linear Regression": MLR,
                         "Support Vector Regression": SupportVectorRegression
                         }
    elif task_type == "Classification":
        model_options = ["Decision Tree",
                         "K-Nearest Neighbor",
                         "Naive Bayes"]
        model_mapping = {
                         "Decision Tree": DecisionTree,
                         "K-Nearest Neighbor": KNearestNeighbor,
                         "Naive Bayes": NaiveBayesModel
                         }
    else:
        st.error("Unknown task type, Cannot proceed.")
        st.stop()

    selected_model = st.selectbox(f"""Select a model for {task_type}""",
                                  options=model_options)
    selected_model_class = model_mapping[selected_model]
    st.write(f"You selected the {selected_model} model.")
    model_instance = selected_model_class()
    return selected_model, model_instance


def get_metric(task_type: str) -> Dict[str, object]:
    """This function lets the user select metrics based
    on the task type (Regression or Classification)

    Args:
        task_type (str): The tyoe of task, which
        determines which metrics can be chosen.

    Returns:
        Dict[str, object]:
            selected_metrics (Dict):
                - Dictionary where keys are the names
                of the selected metrics and the values
                are the instances of the corresponding
                metric classes.
    """
    if task_type == "Regression":
        metric_options = {
                         "Mean Squared Error": MeanSquaredError,
                         "Mean Absolute Error": MeanAbsoluteError,
                         "RSquared": RSquared
                         }
        default_metrics = ["Mean Squared Error",
                           "Mean Absolute Error",
                           "RSquared"]
    else:
        metric_options = {
                         "Log Loss": LogLoss,
                         "Accuracy": Accuracy,
                         "Recall": Recall
                         }
        default_metrics = ["Accuracy", "Recall"]

    metric_names = list(metric_options.keys())

    selected_metric_names = st.multiselect("Select Metrics",
                                           metric_names,
                                           default=default_metrics)
    selected_metrics = {name: metric_options[name]()
                        for name in selected_metric_names}
    return selected_metrics


def display_pipeline_summary(selected_dataset_name: str,
                             target_feature: str,
                             input_features: List[str],
                             selected_model: str,
                             split_ratio: float,
                             selected_metric_names: List[str]
                             ) -> None:
    """Displays a summary of the pipeline configurstions.

    Args:
        selected_dataset_name (str): Name of the selected
        dataset.
        target_feature (str): Name of the target feature.
        input_features (List[str]): List of input features.
        selected_model (str): Name of selected model.
        split_ratio (float): Ratio for splitting the
        dataset into training and testing sets.
        selected_metric_names (Liost[str]): List of
        selected metric names to evaluate the model.
    """
    st.markdown(f'''
        **Pipeline Summary**

        - **Dataset**: {selected_dataset_name}
        - **Target Feature**: {target_feature}
        - **Input Features**: {", ".join(input_features)}
        - **Selected Model**: {selected_model}
        - **Split Ratio**: {split_ratio}
        - **Chosen Metrics**: {", ".join(selected_metric_names)}
        ''')


def execute_pipeline(pipeline: object) -> None:
    """Executes the pipeline and shows the results in
    Streamlit.

    Args:
        pipeline (object): Instance of the pipeline
        object that contains the model, dataset,
        features, and evaluation metrics.
    """
    results = pipeline.execute()
    st.write("## Results")
    st.write("### Test Data Results")
    for metric, result in results["metrics"]:
        st.write(f"{metric.__class__.__name__}: {result:.4f}")
    st.write("### Training Data Results")
    for metric, result in results["metrics training data"]:
        st.write(f"{metric.__class__.__name__} (Training): {result:.4f}")

    st.write("### Predictions on Test Data")
    st.write(results["predictions"])

    st.write("### Predictions on Training Data")
    st.write(results["predictions training data"])


def main() -> None:
    """Orchestrates the model pipeline by selecting
    datasets, features, models, and metrics.
    """
    automl = AutoMLSystem.get_instance()

    st.write("# ⚙ Modelling")
    st.subheader("Select a dataset")
    datasets = automl.registry.list(type="dataset")
    selected_dataset, data = get_dataset(datasets)

    list_feature, target_feature, input_features = get_features(
        selected_dataset)

    if input_features and target_feature:
        t_feature_types = next((feature["type"] for feature in list_feature
                                if feature["name"] == target_feature), None)

    else:
        st.warning("""Please select at least one input feature and
                   a target feature.""")
        return

    if t_feature_types == "numerical":
        task_type = "Regression"
    else:
        task_type = "Classification"

    model_name, model_instance = get_model(task_type)
    selected_metrics = get_metric(task_type)

    split_ratio = st.slider("Select Percentage for Training Data",
                            min_value=0.1,
                            max_value=0.9,
                            value=0.8,
                            step=0.05
                            )

    display_pipeline_summary(selected_dataset.name, target_feature,
                             input_features, model_name,
                             split_ratio, list(selected_metrics.keys()))

    pipeline = Pipeline(
        metrics=list(selected_metrics.values()),
        dataset=data,
        model=model_instance,
        input_features=[Feature(name=feature["name"], type=feature["type"])
                        for feature in list_feature
                        if feature["name"] in input_features],
        target_feature=Feature(name=target_feature, type=t_feature_types),
        split=split_ratio
    )

    if st.button("Train Model"):
        execute_pipeline(pipeline)

    st.write("### Save Pipeline")
    pipeline_name = st.text_input("Enter the pipeline name:")
    pipeline_version = st.text_input("Enter the pipeline version:",
                                     value="1.0.0")
    st.write("""You can click me, but I wont work yet.
                If you would like this beautiful button to work
                we require a deadline extension of a week.""")
    if st.button("Save Pipeline"):
        if pipeline_name and pipeline_version:
            artifacts = []
            for data in pipeline.artifacts:
                with open(data["asset_path"], "rb") as file:
                    artifact_data = file.read()
                artifact = Artifact(
                    name=f"{pipeline_name}_{data['name']}",
                    version=pipeline_version,
                    asset_path=f"""pipelines/{pipeline_name}/
                    {data['name']}""",
                    type="Pipeline",
                    data=artifact_data
                )
                artifacts.append(artifact)

            for artifact in artifacts:
                automl.registry.register(artifact)
                st.success((f"""Pipeline '{pipeline_name}'
                            version '{pipeline_version}'
                            saved successfully!"""))
    else:
        st.warning("Please provide an name and version.")


main()
