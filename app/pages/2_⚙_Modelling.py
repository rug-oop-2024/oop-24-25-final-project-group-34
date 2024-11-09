import streamlit as st
import pandas as pd
import io

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


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

        