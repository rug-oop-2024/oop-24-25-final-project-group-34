import streamlit as st
import pandas as pd
import io

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

from autoop.core.ml.model.classification.decision_tree_regression import DecisionTree
from autoop.core.ml.model.classification.knn import KNearestNeighbor
from autoop.core.ml.model.classification.naive_bayes import NaiveBayesModel

from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.support_vector_regression import SupportVectorRegression



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
        data = selected_dataset.read()
        data = pd.read_csv(io.StringIO(data.decode("utf-8")))
        st.dataframe(data.head())

selected_dataset_name = st.selectbox("Select a dataset to load",
options=names_dataset)

if selected_dataset_name:
    selected_dataset = next((data for data in datasets
    if data.name == selected_dataset_name), None)

    if selected_dataset:
        loaded_dataset = automl.registry.get(selected_dataset.id)

    if isinstance(loaded_dataset, pd.DatFrame):
        features = loaded_dataset.columns.tolist()

        input_features = st.multiselect("Select Input Features",
        options=features)
        target_feature = st.selectbox("Select Target Featrue",
        options=features)

        if input_features and target_features:
            target_data = loaded_dataset[target_feature]
            unique_values = target_data.nuunique()

            if unique_values <= 20 or target_data.dtype == "object":
                task_type = "Classification"
            else:
                task_type = "Regression"

            st.write("Detected Task Type: {task_type}")

            if task_type == "Classification":
                model_options = ["Decision Tree",
                                "K-Nearest Neighbor",
                                "Naive Bayes"]
                model_mapping = {
                    "Decision Tree": DecisionTree,
                    "K-Nearest Neighbor": KNearestNeighbor,
                    "Naive Bayes": NaiveBayesModel,
                }
            else:
                model_options = ["Lasso",
                                "Multiple Linear Regression",
                                "Support Vector Regression"]

                model_mapping = {
                    "Lasso": Lasso,
                    "Multiple Linear Regeression": MultipleLinearRegression,
                    "Support Vector Regression": SupportVectorRegression,
                }

            selected_model = st.selectbox(f"""Select a model for 
                                        {task_type}""",
                                        options=model_options)
            selected_model_class = model_mapping[selected_model]

            st.write(f"You selected the {selected_model} model.")

            model_instance = selected_model_class()
            st.write(f"The initialized model instance: {model_instance}")
