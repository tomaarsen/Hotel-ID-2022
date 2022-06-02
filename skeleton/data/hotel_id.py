import pathlib
from typing import List, Optional

import torch as t
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split

from skeleton.data.collating import collate_hid
from skeleton.data.dataset import ImageDataset
from skeleton.data.preprocess import Preprocessor

################################################################################
# data module implementation


class HotelIDDataModule(LightningDataModule):
    def __init__(
        self,
        data_folder: pathlib.Path,
        hotel_ids: List[int],
        batch_size: int,
        num_workers: int,
        preprocessor: Preprocessor,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.hotel_ids = hotel_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessor = preprocessor
        self.train_split = 0.9

    def setup(self, stage: Optional[str] = None) -> None:
        # Get the filepaths of all images, and split them between training and validation
        image_folder = self.data_folder / "train_images"
        filepaths = list(image_folder.glob("**/*.jpg"))
        train_filepaths, val_filepaths = train_test_split(
            filepaths, train_size=self.train_split
        )

        # Set up training dataset & dataloader
        train_ds = ImageDataset(
            train_filepaths,
            self.hotel_ids,
            transform=self.preprocessor.train_transform,
        )

        self.train_dl = t.utils.data.DataLoader(
            train_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_hid,
            drop_last=True,
            shuffle=True,
        )

        # Set up validation dataset & dataloader
        val_ds = ImageDataset(
            val_filepaths,
            self.hotel_ids,
            transform=self.preprocessor.val_transform,
        )

        self.val_dl = t.utils.data.DataLoader(
            val_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_hid,
            drop_last=True,
        )

    @property
    def num_hotels(self):
        return len(list((self.data_folder / "train_images").iterdir()))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dl
