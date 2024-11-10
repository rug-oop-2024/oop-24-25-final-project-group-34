from copy import deepcopy
import base64


class Artifact:
    """
    A class to represent an Artifact.
    """
    def __init__(self,
                 type: str,
                 name: str = None,
                 asset_path: str = None,
                 data: bytes = None,
                 version: str = None,
                 tags: list = [],
                 metadata: dict = {}
                 ):
        """
        Initializes the Artifact class.

        Args:
            type (str): The type of the Artifact.
            name (str, optional): The name of the Artifact.
                Defaults to None.
            asset_path (str, optional): The asset path of the Artifact.
                Defaults to None.
            data (bytes, optional): The data of the Artifact.
                Defaults to None.
            version (str, optional): The version of the Artifact.
                Defaults to None.
            tags (list, optional): The tags of the Artifact.
                Defaults to [].
            metadata (dict, optional): The metadata of the Artifact.
                Defaults to {}.
        """
        self._type = type
        self._name = name
        self._asset_path = asset_path
        self._data = data
        self._version = version
        self._tags = tags
        self._metadata = metadata
        unique_code = f"{asset_path}-{name}"
        self._id = base64.b64encode(
            unique_code.encode()).decode().replace("=", "_")

    @property
    def type(self) -> str:
        """Public getter for the type attribute."""
        return self._type

    @property
    def name(self) -> str:
        """Public getter for the name attribute."""
        return self._name

    @property
    def asset_path(self) -> str:
        """Public getter for the asset_path attribute."""
        return self._asset_path

    @property
    def data(self) -> bytes:
        """Public getter for the data attribute."""
        return self._data

    @property
    def version(self) -> str:
        """Public getter for the version attribute."""
        return self._version

    @property
    def tags(self) -> list:
        """Public getter for the tags attribute."""
        return deepcopy(self._tags)

    @property
    def metadata(self) -> dict:
        """Public getter for the metadata attribute."""
        return deepcopy(self._metadata)

    @property
    def id(self) -> str:
        """Public getter for the id attribute."""
        return self._id

    def read(self) -> bytes:
        """returns the raw data."""
        return self._data

    def save(self, new_data) -> bytes:
        """
        Saves the new data to the Artifact

        Args:
            new_data (bytes): The new data to be saved

        Returns:
            bytes: The newly saved data
        """
        self._data = new_data
        return self._data
