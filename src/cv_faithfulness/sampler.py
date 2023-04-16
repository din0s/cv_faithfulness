import random
from typing import Iterator

import torch
from torch import LongTensor
from torch.utils.data import Sampler
from torchvision.datasets import VisionDataset


class FairGridSampler(Sampler[int]):
    def __init__(
        self,
        dataset: VisionDataset,
        class_idx: int,
        grid_size: int,
        shuffle: bool = False,
    ):
        """A sampler that returns a grid of images from the dataset, with a uniformly random
        amount of appearances for a specific class of interest.

        Args:
            dataset (VisionDataset): the dataset to sample from
            class_idx(int): the class (index) to treat as the class of interest
            grid_size (int): the number of images per row in the grid
            shuffle (bool): whether to shuffle the dataset before sampling
        """
        super().__init__(dataset)

        # Save the hyperparameters
        self.dataset = dataset
        self.grid_size = grid_size
        self.n_images = grid_size**2

        # Get the indices of the class of interest
        self.class_indices = LongTensor(
            [i for i, x in enumerate(dataset) if x[1] == class_idx]
        )
        # Get the indices of all other classes
        self.other_indices = LongTensor(
            [i for i, x in enumerate(dataset) if x[1] != class_idx]
        )

        # Fix the seed if shuffle is False
        self.seed = None if shuffle else self._get_seed()

    @staticmethod
    def _get_seed() -> int:
        """Utility function for generating a random seed."""
        return int(torch.empty((), dtype=torch.int64).random_().item())

    def __iter__(self) -> Iterator[int]:
        """An iterator that yields a grid of random images from the dataset."""
        # Create a torch Generator object
        seed = self.seed if self.seed is not None else self._get_seed()
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Sample the batches
        for _ in range(len(self.dataset) // self.n_images):
            # Pick the number of instances for the class of interest
            n_samples = torch.randint(self.n_images + 1, (), generator=gen).item()

            # Sample the indices from the class of interest
            idx_from_class = torch.randperm(
                len(self.class_indices),
                generator=gen,
            )[:n_samples]
            # Sample the indices from the other classes
            idx_from_other = torch.randperm(
                len(self.other_indices),
                generator=gen,
            )[: self.n_images - n_samples]

            # Concatenate the corresponding lists of patches to form a grid
            grid = (
                self.class_indices[idx_from_class].tolist()
                + self.other_indices[idx_from_other].tolist()
            )

            # Shuffle the order of the patches within the grid
            random.shuffle(grid)
            yield from grid

    def __len__(self) -> int:
        return len(self.dataset)
