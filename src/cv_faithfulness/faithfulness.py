import itertools
from functools import partial
from typing import Optional

import torch
from torch.utils.data import default_collate, random_split
from torchvision import transforms

from .base import ImageDataModule
from .sampler import FairGridSampler


class FaithfulnessDataModule(ImageDataModule):
    """A datamodule for the toy dataset as described in the paper."""

    def __init__(
        self,
        class_idx: int = 3,
        grid_size: int = 3,
        feature_extractor: callable = None,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """A datamodule for the faithfulness task introduced in Vision DiffMask.
        The task is to count the number of patches with a certain color in a grid.
        Read more: https://arxiv.org/abs/2304.06391

        Args:
            class_idx (int): the class (index) to count
            grid_size (int): the number of images per row in the grid
            feature_extractor (callable): a callable feature extractor instance
            data_dir (str): the directory to store the dataset
            batch_size (int): the batch size for the train/val/test dataloaders
            num_workers (int): the number of workers to use for data loading
        """
        super().__init__(
            feature_extractor,
            data_dir,
            (grid_size**2) * batch_size,
            num_workers,
        )

        # Store hyperparameters
        self.class_idx = class_idx
        self.grid_size = grid_size

        # Save the existing transformations to be applied after creating the grid
        self.post_transform = self.transform
        # Set the pre-batch transformation to be the conversion from PIL to tensor
        self.transform = transforms.PILToTensor()

        # Specify the custom collate function and samplers
        self.collate_fn = self.custom_collate_fn
        self.shuffled_sampler = partial(
            FairGridSampler,
            class_idx=class_idx,
            grid_size=grid_size,
            shuffle=True,
        )
        self.sequential_sampler = partial(
            FairGridSampler,
            class_idx=class_idx,
            grid_size=grid_size,
            shuffle=False,
        )

    def prepare_data(self):
        # No need to download anything for the toy task
        pass

    def setup(self, stage: Optional[str] = None):
        img_size = 16

        samples = []
        # Generate 6000 samples based on 6 different colors
        for r, g, b in itertools.product((0, 1), (0, 1), (0, 1)):
            if r == g == b:
                # We do not want black/white patches
                continue

            for _ in range(1000):
                patch = torch.vstack(
                    [
                        r * torch.ones(1, img_size, img_size),
                        g * torch.ones(1, img_size, img_size),
                        b * torch.ones(1, img_size, img_size),
                    ]
                )

                # Assign a unique id to each color
                target = int(f"{r}{g}{b}", 2) - 1
                # Append the patch and target to the samples
                samples += [(patch, target)]

        # Split the data to 90% train, 5% validation and 5% test
        train_size = int(len(samples) * 0.9)
        val_size = (len(samples) - train_size) // 2
        test_size = len(samples) - train_size - val_size
        self.train_data, self.val_data, self.test_data = random_split(
            samples,
            [
                train_size,
                val_size,
                test_size,
            ],
        )

    def custom_collate_fn(self, batch):
        # Split the batch into groups of grid_size**2
        idx = range(len(batch))
        grids = zip(*(iter(idx),) * (self.grid_size**2))

        new_batch = []
        for grid in grids:
            # Create a grid of images from the indices in the batch
            img = torch.hstack(
                [
                    torch.dstack(
                        [batch[i][0] for i in grid[idx : idx + self.grid_size]]
                    )
                    for idx in range(
                        0, self.grid_size**2 - self.grid_size + 1, self.grid_size
                    )
                ]
            )
            # Apply the post transformations to the grid
            img = self.post_transform(img)
            # Define the target as the number of images that have the class_idx
            targets = [batch[i][1] for i in grid]
            target = targets.count(self.class_idx)
            # Append grid and target to the batch
            new_batch += [(img, target)]

        return default_collate(new_batch)
