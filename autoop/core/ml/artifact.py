from copy import deepcopy
from typing import Any
import base64


class Artifact:
    def __init__(self,
                 type:str,
                 name: str = None,
                 asset_path: str = None,
                 data: bytes = None,
                 version: str = None,
                 tags: list = [],
                 metadata: dict = {}
                 ):
        """initializes the Artifact class"""
        self._type = type
        self._name = name
        self._asset_path = asset_path
        self._data = data
        self._version = version
        self._tags = tags
        self._metadata = metadata
        unique_code = f"{asset_path}-{name}"
        self._id = base64.b64encode(unique_code.encode()).decode()
    
    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, new_type: str) -> None:
        self._type = new_type

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        self._name = new_name

    @property
    def asset_path(self):
        return self._asset_path
    
    @asset_path.setter
    def asset_path(self, new_asset_path: str):
        self._asset_path = new_asset_path

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data: bytes):
        self._data = new_data

    @property
    def version(self):
        return self._version
    
    @version.setter
    def version(self, new_version: str):
        self._version = new_version
    
    @property
    def tags(self):
        return deepcopy(self._tags)
    
    @tags.setter
    def tags(self, new_tag: Any):
        self._tags.append(new_tag)
    
    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @metadata.setter
    def metadata(self, new_metadata: dict):
        for key, item in new_metadata.items():
            self._metadata[key] = item
    
    @property
    def id(self):
        return self._id
    
    def read(self):
        """returns the raw data."""
        return self._data
    
    def save(self, new_data):
        self._data = new_data
        return self._data


