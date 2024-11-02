import typing
import uuid
import tempfile
import warnings

import fsspec
import torch
import numpy as np
import yt.wrapper as yt
from torch.utils.data import Sampler

from tractorun.backend.tractorch.dataset import YtDataset
from tractorun.backend.tractorch.serializer import TensorSerializer

from nanotron.data.nanoset import Nanoset


_T_co = typing.TypeVar("_T_co")


class YTTensorTransform:
    _serializer = TensorSerializer()

    def __call__(self, columns: list[str], row: dict) -> dict:
        return {
            name: self._serializer.desirialize(
                yt.yson.get_bytes(row[name])
            )
            for name in columns
        }


class YtTableDataset(YtDataset):
    def __init__(
        self,
        yt_client: yt.YtClient,
        path: str,
        start: int = 0,
        end: int | None = None,
        columns: list | None = None,
    ) -> None:
        self.yt_client = yt_client
        self.path = path
        self.start = start
        self.end = end
        self.columns = columns

        super().__init__(
            yt_client=yt_client,
            path=path,
            start=start,
            end=end,
            columns=columns,
            transform=YTTensorTransform(),
        )

    def to_dp(self, start: int, end: int) -> "YtTableDataset":
        return YtTableDataset(
            yt_client=self.yt_client,
            path=self.path,
            columns=self.columns,
            end=end,
            start=start,
        )


class YtTableDatasetDistributedSampler(Sampler[_T_co]):
    def __init__(
        self,
        dataset: YtTableDataset,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        # here we just drop line that do not fit into the last chunk
        dp_chunk_size = len(dataset) // num_replicas
        start = rank * dp_chunk_size
        end = start + dp_chunk_size - 1
        self._dataset = dataset.to_dp(start=start, end=end)
        self._num_replicas = num_replicas
        self._rank = rank
        super().__init__()

    def __iter__(self):
        return self._dataset.__iter__()


class YtFsFileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        yt_client: yt.YtClient,
        yt_dataset_paths: str | list[str],
        sequence_length: int,
        token_size: int,
        train_split_num_samples: int,
        dataset_weights: float | None = None,
        random_seed: int = 1234,
    ) -> None:
        if isinstance(yt_dataset_paths, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            yt_dataset_paths = [yt_dataset_paths]
        dataset_dir = tempfile.mkdtemp()
        for yt_path in yt_dataset_paths:
            file_name = str(uuid.uuid4())
            stream = yt_client.read_file(yt_path)
            with open(file_name, "wb") as f:
                f.write(stream.read())
        self.dataset = Nanoset(
            dataset_folders=[dataset_dir],
            sequence_length=sequence_length,
            token_size=token_size,
            train_split_num_samples=train_split_num_samples,
            dataset_weights=dataset_weights,
            random_seed=random_seed,
        )

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.dataset[idx]


class YtMemFileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        yt_client: yt.YtClient,
        yt_dataset_paths: str | list[str],
        sequence_length: int,
        token_size: int,
        train_split_num_samples: int,
        dataset_weights: float | None = None,
        random_seed: int = 1234,
    ) -> None:
        if isinstance(yt_dataset_paths, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            yt_dataset_paths = [yt_dataset_paths]

        fs = fsspec.filesystem("memory")
        dataset_dir = str(uuid.uuid4())
        for yt_path in yt_dataset_paths:
            file_name = str(uuid.uuid4())
            stream = yt_client.read_file(yt_path)
            with fs.open(f"mem://{dataset_dir}/{file_name}", "wb") as f:
                f.write(stream.read())
        self.dataset = Nanoset(
            dataset_folders=[dataset_dir],
            sequence_length=sequence_length,
            token_size=token_size,
            train_split_num_samples=train_split_num_samples,
            dataset_weights=dataset_weights,
            random_seed=random_seed,
        )

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.dataset[idx]

