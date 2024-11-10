from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """
    A class to represent an ML dataset
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a Dataset based on Artifact.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1.0.0") -> 'Dataset':
        """
        Creates a dataset from a pandas dataframe.

        Args:
            data (pd.DataFrame): The data to be converted.
            name (str): The name of the dataset
            asset_path (str): The path to the dataset
            version (str, optional): The version of the dataset.
            Defaults to "1.0.0".

        Returns:
            Dataset: A new Dataset with the provided input.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads data from a given path.

        Returns:
            pd.DataFrame: The dataset as a panda DataFrame.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves data to a given path.

        Args:
            data (pd.DataFrame): The dataset to be saved.

        Returns:
            bytes: The dataset in CSV format.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
