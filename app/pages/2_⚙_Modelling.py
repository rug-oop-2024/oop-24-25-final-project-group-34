import streamlit as st
import pandas as pd
import io

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline

from autoop.core.ml.model.classification.decision_tree_regression import DecisionTree
from autoop.core.ml.model.classification.knn import KNearestNeighbor
from autoop.core.ml.model.classification.naive_bayes import NaiveBayesModel

from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.support_vector_regression import SupportVectorRegression

from autoop.core.ml.metric import (
    MeanSquaredError,
    MeanAbsoluteError,
    RSquared,
    LogLoss,
    Accuracy,
    Recall,
)



st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")

automl = AutoMLSystem.get_instance()

st.subheader("Select a dataset")
datasets = automl.registry.list(type="dataset")

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

        features = detect_feature_types(selected_dataset)
        list_feature = [{"name": feature.name,
                        "type": feature.type} for feature in features]

        complete_features_names = [feature["name"] for feature in list_feature]

        target_feature = st.selectbox("Select Target Feature",
        options=complete_features_names)

        remaining_input_features = [
            name for name in complete_features_names if name != target_feature
        ]

        input_features = st.multiselect("Select Input Features",
        options=remaining_input_features)


        if input_features and target_feature:
            t_feature_types = next(
                (feature["type"] for feature in list_feature
                if feature["name"] == target_feature), None
            )

            if t_feature_types == "numerical":
                task_type = "Regression"
            else:
                task_type = "Classification"

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

            selected_model = st.selectbox(f"""Select a model for 
                                        {task_type}""",
                                        options=model_options)
            selected_model_class = model_mapping[selected_model]

            st.write(f"You selected the {selected_model} model.")

            model_instance = selected_model_class()

            split_ratio = st.slider(
                "Select Percentage for Training Data",
                min_value = 0.1,
                max_value = 0.9,
                value = 0.8,
                step = 0.05,
            )

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

            selected_metric_names = st.multiselect(
                "Select Metrics",
                metric_names,
                default = metric_names,
            )
            selected_metrics = {name: metric_options[name]()
                                for name in selected_metric_names
                                }

            st.markdown(f'''
            **Pipeline Summary**

            - **Dataset**: {selected_dataset_name}
            - **Target Feature**: {target_feature}
            - **Input Features**: {", ".join(input_features)}
            - **Selected Model**: {selected_model}
            - **Split Ratio**: {split_ratio}
            - **Chosen Metrics**: {", ".join(selected_metric_names)}
            ''')

    else:
        st.warning("Please select at least one input feature.")
else:
    st.warning("No datasets available. Please upload a dataset to proceed.")