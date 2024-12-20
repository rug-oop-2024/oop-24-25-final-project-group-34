import streamlit as st
import pandas as pd
from typing import IO

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()


def upload_file(file: IO[str]) -> pd.DataFrame:
    """Uploads a CSV file

    Args:
        file (type): A file object repreesenting
        the uploaded CSV file.

    Returns:
        pd.DataFrame: A pandas dataframe representing
        the uplaoded dataset.
    """
    dataframe = pd.read_csv(file)
    st.write("Preview of the uploaded dataset: ")
    st.dataframe(dataframe.head())
    return dataframe


def convert_file(file: pd.DataFrame, dataset_name: str) -> Dataset:
    """Converts the uploaded DataFrame into a Dataset object
    with the specified dataset name.

    Args:
        file (pd.DataFrame): The data frame that is converted
        into a dataset.
        dataset_name (str): The name of the dataset.

    Returns:
        Dataset: A dataset object created from
        the data frame.
    """
    asset_path = f"./objects/{dataset_name}"
    dataset = Dataset.from_dataframe(data=file,
                                     name=dataset_name,
                                     asset_path=asset_path)
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

        if dataset is not None:
            if st.button("Save Dataset"):
                try:
                    automl.registry.register(dataset)
                    st.success(f"""Dataset '{dataset_name}'
                    has been successfully saved!""")
                except Exception as e:
                    st.error(f"""An error occurred while saving
                    the dataset: {e}""")

st.subheader("Manage Datasets")
artifacts = automl.registry.list(type="dataset")

if artifacts:
    datasets = [Dataset(name=artifact.name,
                        asset_path=artifact.asset_path,
                        data=artifact.data,
                        version=artifact.version)
                for artifact in artifacts]
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a Dataset to View or Delete",
                                         dataset_names)

    selected_dataset = next((data for data in datasets
                             if data.name == selected_dataset_name), None)

    if selected_dataset:
        st.write("Preview of the Dataset: ")
        data = selected_dataset.read()
        st.dataframe(data.head())
        if st.button("Delete the selected Dataset"):
            try:
                automl.registry.delete(selected_dataset.id)
                st.success(f"""Dataset '{selected_dataset_name}'
                           has been deleted.""")
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred while deleting the dataset: {e}")
    else:
        st.write("Dataset is empty")
