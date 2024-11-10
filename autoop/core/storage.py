from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    Custom Error that will be raised when a path can not be found
    """
    def __init__(self, path) -> None:
        """
        Initializes the Error

        Args:
            path (str): The path that couldnt be found
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    The Abstract base class for the LocalStorage
    """
    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    A LocalStorage based on the Storage base class
    """
    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes the LocalStorage with base directory to
        store the files.

        Args:
            base_path (str): The base path to the file storage.
            Defaults to "./assets".
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Function that will save data to a file at
        the specified path. It will check if the path exists
        and it will then store the data there.

        Args:
            data (bytes): The binary data that will be saved.
            key (str): The path where the data will be stored to.
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Function that will load the data from the given path
        if the path exists.

        Args:
            key (str): The path where the data is stored.

        Returns:
            bytes: The data that gets retrieved.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Function that will delete the data from the fiven path
        of the given path exists.

        Args:
            key (str): The path where the data will be deleted.
             Defaults to "/".
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        Function that will list all the paths to the data.

        Args:
            prefix (str): The path where all the data will be listed from.
            Defaults to "/".

        Returns:
            List[str]: List with all the paths.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for p in keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Checks if the path exists. If it does not,
        it will raise a NotFoundError.

        Args:
            path (str): the path to check

        Raises:
            NotFoundError: The error that will occur if the
            path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Ensures paths are OS-agnostic
        """
        return os.path.normpath(os.path.join(self._base_path, path))
