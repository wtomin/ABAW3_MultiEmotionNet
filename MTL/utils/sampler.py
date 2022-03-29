import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

class RandomSubsetShuffledSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], splits: int, generator=None) -> None:
        self.indices = np.array(indices) # total list of indices
        assert splits>0, "splits must be a positive integer."
        self.splits = splits
        self.indices_splits = [indices[i:][::splits] for i in range(splits)]
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        random_split = np.random.randint(self.splits)
        for i in torch.randperm(len(self.indices_splits[random_split]), generator=self.generator):
            yield self.indices_splits[random_split][i]

    def __len__(self) -> int:
        return len(self.indices)//self.splits