class Artifact:
    def __init__(self, type:str, name: str = None, asset_path: str = None, data: bytes = None, version: str = None):
        """initializes the Artifact class"""
        self._type = type
        self._name = name
        self._asset_path = asset_path
        self._data = data
        self._version = version
    
    def read(self):
        """returns the raw data."""
        return self._data
