import streamlit as st
import pandas as pd
import io

from app.core.system import AutoMLSystem
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact

from autoop.core.ml.model.classification import (DecisionTree,
                                                 KNearestNeighbor,
                                                 NaiveBayesModel)
from autoop.core.ml.model.regression import (Lasso,
                                             MultipleLinearRegression,
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


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def get_dataset(datasets):
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


def get_features(selected_dataset):
    features = detect_feature_types(selected_dataset)
    list_feature = [{"name": feature.name,
                    "type": feature.type} for feature in features]
    complete_features_names = [feature["name"] for feature in list_feature]
    target_feature = st.selectbox("Select Target Feature",
                                  options=complete_features_names)
    remaining_input_features = [
        name for name in complete_features_names if name != target_feature]
    input_features = st.multiselect("Select Input Features",
                                    options=remaining_input_features)
    return list_feature, target_feature, input_features


def get_model(task_type):
    if task_type == "Regression":
        model_options = ["Lasso",
                         "Multiple Linear Regression",
                         "Support Vector Regression"]
        model_mapping = {
                        "Lasso": Lasso,
                        "Multiple Linear Regression": MultipleLinearRegression,
                        "Support Vector Regression": SupportVectorRegression,
                    }
    else:
        model_options = ["Decision Tree",
                         "K-Nearest Neighbor",
                         "Naive Bayes"]
        model_mapping = {
                        "Decision Tree": DecisionTree,
                        "K-Nearest Neighbor": KNearestNeighbor,
                        "Naive Bayes": NaiveBayesModel,
                    }
    selected_model = st.selectbox(f"""Select a model for {task_type}""",
                                  options=model_options)
    selected_model_class = model_mapping[selected_model]
    st.write(f"You selected the {selected_model} model.")
    model_instance = selected_model_class()
    return selected_model, model_instance


def get_metric(task_type):
    if task_type == "Regression":
        metric_options = {
                        "Mean Squared Error": MeanSquaredError,
                        "Mean Absolute Error": MeanAbsoluteError,
                        "RSquared": RSquared,
                }
    else:
        metric_options = {
                        "Log Loss": LogLoss,
                        "Accuracy": Accuracy,
                        "Recall": Recall,
                }

    metric_names = list(metric_options.keys())

    selected_metric_names = st.multiselect("Select Metrics",
                                           metric_names,
                                           default=metric_names)
    selected_metrics = {name: metric_options[name]()
                        for name in selected_metric_names}
    return selected_metrics


def display_pipeline_summary(selected_dataset_name, target_feature,
                             input_features, selected_model,
                             split_ratio, selected_metric_names):
    st.markdown(f'''
        **Pipeline Summary**

        - **Dataset**: {selected_dataset_name}
        - **Target Feature**: {target_feature}
        - **Input Features**: {", ".join(input_features)}
        - **Selected Model**: {selected_model}
        - **Split Ratio**: {split_ratio}
        - **Chosen Metrics**: {", ".join(selected_metric_names)}
        ''')


def execute_pipeline(pipeline):
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


def main():
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

            if st.button("Train Model"):


                input_features = [
                    Feature(name=name, type=next(
                    (feature.type for feature in
                    features if feature.name == name),
                     "")) for name in input_features
                ]
                target_feature = Feature(name=target_feature,
                                         type=t_feature_types)

                pipeline = Pipeline(
                    metrics=selected_metrics,
                    dataset=df,
                    model=model_instance,
                    input_features=input_features,
                    target_feature=target_feature,
                    split=split_ratio,
                )

                with st.spinner("Currently training the model..."):
                    results = pipeline.execute()
                
                st.success("Model training completed!")
                st.write("### Evaluation Results")
                st.write("#### Training Metrics")
                for metric, value in results["train_metrics"]:
                    st.write(f"- **{metric.__class__.__name__}** {value}")

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
