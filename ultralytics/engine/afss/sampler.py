# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Iterator, List

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class AFSSEpochSampler(Sampler[int]):
    """Epoch-aware sampler driven by AFSS active indices."""

    def __init__(self, active_indices: List[int], shuffle: bool = True, seed: int = 0):
        self.active_indices = list(active_indices)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        indices = list(self.active_indices)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in perm]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.active_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_active_indices(self, active_indices: List[int]) -> None:
        self.active_indices = list(active_indices)


class AFSSDistributedEpochSampler(Sampler[int]):
    """Distributed AFSS sampler keeping all ranks on the same active set."""

    def __init__(
        self,
        active_indices: List[int],
        shuffle: bool = True,
        seed: int = 0,
        rank: int | None = None,
        num_replicas: int | None = None,
        drop_last: bool = False,
    ):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("AFSSDistributedEpochSampler requires initialized torch.distributed")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("AFSSDistributedEpochSampler requires initialized torch.distributed")
            rank = dist.get_rank()
        self.rank = rank
        self.num_replicas = num_replicas
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.active_indices = list(active_indices)
        self.num_samples = 0
        self.total_size = 0
        self._refresh_sizes()

    def _refresh_sizes(self) -> None:
        if self.drop_last:
            # Standard PyTorch DistributedSampler drop_last: floor division
            self.num_samples = len(self.active_indices) // self.num_replicas
        else:
            self.num_samples = math.ceil(len(self.active_indices) / self.num_replicas) if self.num_replicas else 0
        self.num_samples = max(self.num_samples, 0)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        indices = list(self.active_indices)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in perm]

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0 and len(indices) > 0:
                repeats = math.ceil(padding_size / len(indices))
                indices += (indices * repeats)[:padding_size]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_active_indices(self, active_indices: List[int]) -> None:
        self.active_indices = list(active_indices)
        self._refresh_sizes()
