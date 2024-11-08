import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

def upload_file(file):
    dataframe = pd.read_csv(file)
    st.write("Preview of the uploaded dataset: ")
    st.dataframe(dataframe.head())
    return dataframe

def convert_file(file, dataset_name):
    asset_path = f"./assets/objects/{dataset_name}"
    dataset = Dataset.from_dataframe(data=file, name=dataset_name, asset_path=asset_path)
    if not dataset.data:
        st.error("Creating a dataset failed, no data available.")
        return None
    return dataset

st.title("Dataset Management")
st.header("Upload a new CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    dataframe = upload_file(uploaded_file)
    dataset_name = uploaded_file.name
    if dataframe is not None:
        dataset = convert_file(dataframe, dataset_name)
        automl.registry.register(dataset)
