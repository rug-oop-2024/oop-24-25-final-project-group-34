
import unittest

from autoop.core.storage import LocalStorage, NotFoundError
import random
import tempfile


class TestStorage(unittest.TestCase):
    """
    Unit tests for the localstorage class. will test saving, loading,
    deleting and list operations.
    """
    def setUp(self):
        """
        Sets up a storage for the testing.
        """
        temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(temp_dir)

    def test_init(self):
        """
        Test the initialisation of the storage.
        """
        self.assertIsInstance(self.storage, LocalStorage)

    def test_store(self):
        """
        Tests saving and loading a into and from localstorage

        Verifies:
        A file can be saved and loaded correctly.
        Loading a non-existing file will raise a NotFoundError.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = "test/path"
        self.storage.save(test_bytes, key)
        self.assertEqual(self.storage.load(key), test_bytes)
        otherkey = "test/otherpath"
        # should not be the same
        try:
            self.storage.load(otherkey)
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_delete(self):
        """
        Tests deleting a file from LocalStorage

        Steps:
        It first saves a file, then deletes it and then tests
        if it raises a NotFoundError.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = "test/path"
        self.storage.save(test_bytes, key)
        self.storage.delete(key)
        try:
            self.assertIsNone(self.storage.load(key))
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_list(self):
        """
        Tests listing a file in the LocalStorage.

        Steps:
        Saves several files with random names.
        Checks if all the file names have been added to the list.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        random_keys = [f"test/{random.randint(0, 100)}" for _ in range(10)]
        for key in random_keys:
            self.storage.save(test_bytes, key)
        keys = self.storage.list("test")
        keys = ["/".join(key.split("/")[-2:]) for key in keys]
        self.assertEqual(set(keys), set(random_keys))
            