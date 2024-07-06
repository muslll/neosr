# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Never


class BaseStorageBackend(ABC):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        return Path(filepath).read_bytes()

    def get_text(self, filepath):
        filepath = str(filepath)
        return Path(filepath).read_text(encoding="locale")


class LmdbBackend(BaseStorageBackend):
    """Lmdb storage backend.

    Args:
    ----
        db_paths (str | list[str]): Lmdb database paths.
        client_keys (str | list[str]): Lmdb client keys. Default: 'default'.
        readonly (bool, optional): Lmdb environment parameter. If True,
            disallow any write operations. Default: True.
        lock (bool, optional): Lmdb environment parameter. If False, when
            concurrent access occurs, do not lock the database. Default: False.
        readahead (bool, optional): Lmdb environment parameter. If False,
            disable the OS filesystem readahead mechanism, which may improve
            random read performance when a database is larger than RAM.
            Default: False.

    Attributes:
    ----------
        db_paths (list): Lmdb database path.
        _client (list): A list of several lmdb envs.

    """

    def __init__(
        self,
        db_paths,
        client_keys="default",
        readonly=True,
        lock=False,
        readahead=False,
        **kwargs,
    ) -> None:
        try:
            import lmdb
        except ImportError:
            msg = "Please install lmdb to enable LmdbBackend."
            raise ImportError(msg)

        if isinstance(client_keys, str):
            client_keys = [client_keys]

        if isinstance(db_paths, list):
            self.db_paths = [str(v) for v in db_paths]
        elif isinstance(db_paths, str):
            self.db_paths = [str(db_paths)]
        assert len(client_keys) == len(self.db_paths), (
            "client_keys and db_paths should have the same length, "
            f"but received {len(client_keys)} and {len(self.db_paths)}."
        )

        self._client = {}
        for client, path in zip(client_keys, self.db_paths, strict=True):
            self._client[client] = lmdb.open(
                path, readonly=readonly, lock=lock, readahead=readahead, **kwargs
            )

    def get(self, filepath, client_key):
        """Get values according to the filepath from one lmdb named client_key.

        Args:
        ----
            filepath (str | obj:`Path`): Here, filepath is the lmdb key.
            client_key (str): Used for distinguishing different lmdb envs.

        """
        filepath = str(filepath)
        assert (
            client_key in self._client
        ), f"client_key {client_key} is not in lmdb clients."
        client = self._client[client_key]
        with client.begin(write=False) as txn:
            return txn.get(filepath.encode("ascii"))

    def get_text(self, filepath) -> Never:
        raise NotImplementedError


class FileClient:
    """A general file client to access files in different backend.

    The client loads a file or text in a specified backend from its path
    and return it as a binary file. it can also register other backend
    accessor with a given name and backend class.

    Attributes
    ----------
        backend (str): The storage backend type. Options are "disk",
            "memcached" and "lmdb".
        client (:obj:`BaseStorageBackend`): The backend object.

    """

    _backends = {"disk": HardDiskBackend, "lmdb": LmdbBackend}

    def __init__(self, backend="disk", **kwargs) -> None:
        if backend not in self._backends:
            msg = (
                f"Backend {backend} is not supported. Currently supported ones"
                f" are {list(self._backends.keys())}"
            )
            raise ValueError(msg)
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    def get(self, filepath, client_key="default"):
        # client_key is used only for lmdb, where different fileclients have
        # different lmdb environments.
        if self.backend == "lmdb":
            return self.client.get(filepath, client_key)
        return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)
