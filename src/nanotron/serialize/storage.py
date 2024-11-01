import io
import os
import torch

import yt.wrapper as yt

from abc import ABC, abstractmethod


class Storage(ABC):
    @abstractmethod
    def create_directory(self, path: str):
        ...

    @abstractmethod
    def write_file(self, path: str, data: bytes, metadata: dict[str, str] = {}):
        ...

    def save(self, path: str, obj: object, metadata: dict[str, str] = {}):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        self.write_file(path, buffer.getvalue(), metadata=metadata)

    @abstractmethod
    def read_file(self, path: str) -> bytes:
        ...

    @abstractmethod
    def read_metadata(self, path: str) -> dict[str, str]:
        ...

    def load(self, path: str, map_location: str | None = None) -> object:
        return torch.load(io.BytesIO(self.read_file(path)), map_location=map_location)

    @abstractmethod
    def list_dir(self, path: str) -> list[str]:
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        ...


class LocalStorage(Storage):
    def __init__(self, base_path: str):
        self._base_path = base_path

    def create_directory(self, path: str):
        os.makedirs(self._get_path(path), exist_ok=True)

    def write_file(self, path: str, data: bytes, metadata: dict[str, str] = {}):
        with open(self._get_path(path), "wb") as f:
            f.write(data)
        with open(self._get_path(path + ".metadata"), "w") as f:
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")

    def read_file(self, path: str) -> bytes:
        with open(self._get_path(path), "rb") as f:
            return f.read()
        
    def read_metadata(self, path: str) -> dict[str, str]:
        if not self.exists(path + ".metadata"):
            return {}
        with open(self._get_path(path + ".metadata"), "r") as f:
            return dict(line.strip().split(": ", 1) for line in f)

    def list_dir(self, path: str) -> list[str]:
        return os.listdir(self._get_path(path))
    
    def exists(self, path: str) -> bool:
        return os.path.exists(self._get_path(path))

    def _get_path(self, path: str) -> str:
        return self._base_path + "/" + path


class TractoStorage(Storage):
    def __init__(self, yt_client: yt.YtClient, base_path: str):
        self._yt_client = yt_client
        self._base_path = base_path

    def create_directory(self, path: str):
        self._yt_client.create(
            "map_node",
            self._get_path(path),
            recursive=True,
            ignore_existing=True,
        )

    def write_file(self, path: str, data: bytes, metadata: dict[str, str] = {}):
        self._yt_client.write_file(self._get_path(path), data)
        self._yt_client.set(self._get_path(path) + "/@metadata", metadata)

    def read_file(self, path: str) -> bytes:
        return self._yt_client.read_file(self._get_path(path)).read()
    
    def read_metadata(self, path: str) -> dict[str, str]:
        return self._yt_client.get(self._get_path(path) + "/@metadata")

    def list_dir(self, path):
        return self._yt_client.list(self._get_path(path))
    
    def exists(self, path: str) -> bool:
        return self._yt_client.exists(self._get_path(path))

    def _get_path(self, path: str) -> str:
        if not path:
            return self._base_path
        return self._base_path + "/" + path
