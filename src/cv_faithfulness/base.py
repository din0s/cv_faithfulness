from abc import ABCMeta, abstractmethod
from typing import Optional

import lightning.pytorch as pl
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    default_collate,
)
from torchvision import transforms


class ImageDataModule(pl.LightningDataModule, metaclass=ABCMeta):
    # Declare variables that will be initialized later
    train_data: Dataset
    val_data: Dataset
    test_data: Dataset

    def __init__(
        self,
        feature_extractor: Optional[callable] = None,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """Abstract Pytorch Lightning DataModule for image datasets.

        Args:
            feature_extractor (callable): feature extractor instance
            data_dir (str): directory to store the dataset
            batch_size (int): batch size for the train/val/test dataloaders
            num_workers (int): number of workers for train/val/test dataloaders
        """
        super().__init__()

        # Store hyperparameters
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Set the transforms
        # If the feature_extractor is None, then we do not split the images into features
        init_transforms = [feature_extractor] if feature_extractor else []
        self.transform = transforms.Compose(init_transforms)

        # Set the collate function and the samplers
        # These can be adapted in a child datamodule class to have a different behavior
        self.collate_fn = default_collate
        self.shuffled_sampler = RandomSampler
        self.sequential_sampler = SequentialSampler

    @abstractmethod
    def prepare_data(self):
        raise NotImplementedError()

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError()

    # noinspection PyTypeChecker
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=self.shuffled_sampler(self.train_data),
        )

    # noinspection PyTypeChecker
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=self.sequential_sampler(self.val_data),
        )

    # noinspection PyTypeChecker
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=self.sequential_sampler(self.test_data),
        )
