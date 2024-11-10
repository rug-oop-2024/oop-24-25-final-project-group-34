from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """Registry for managing artifacts."""
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """Initializes the ArtifactRegistry
        with the given database and storage.

        Args:
            database (Database): Database instance

            storage (Storage): Storage instance
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """Saves the artifact in the storage
        and the metadata in the database.

        Args:
            artifact (Artifact): Artifact that gets
            registered.
        """
        self._storage.save(artifact.data, artifact.asset_path)
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """Lists all the registered artifacts.

        Args:
            type (str, optional): The type of artifacts
            to filter by dataset and model.
            Defaults to None.

        Returns:
            List[Artifact]: A list of Artifact objects
            matching the type filter.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """Retrieves an artifact by its ID.

        Args:
            artifact_id (str): The ID of the
            artifact.

        Returns:
            Artifact: The requested artifact.
        """
        print(f"Trying to load artifact with id: {artifact_id}")
        print("Database entries:", self._database.list("artifacts"))

        data = self._database.get("artifacts", artifact_id)
        if data is None:
            raise ValueError(f"""Artifact with id {artifact_id}
                             not found in database.""")

        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str):
        """Deletes an artifact based on its ID.

        Args:
            artifact_id (str): The ID of the
            artifact.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """Manages the AutoML system."""
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        """Initializes the AutoMLSystem with the given
        storage and database instances.

        Args:
            storage (LocalStorage): The storage instance.
            database (Database): The datavase instance.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance():
        """Retrieves the singleton instance of the AutoMLSystem.

        Returns:
            AutoMLSystem: The singleton instance of the
            AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self):
        """Getter for registry.

        Returns:
            ArtifactRegistry: Artifact
            registry instance.
        """
        return self._registry
