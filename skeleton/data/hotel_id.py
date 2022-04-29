################################################################################
#
# This file implements a datamodule for the tiny-voxceleb dataset.
# The data is loaded from pre-generated webdataset shards.
# (See `scripts/generate_shards.py` for how to generate shards.)
#
# Author(s): Nik Vaessen
################################################################################

import json
import pathlib
from typing import Iterator, List, Optional, Tuple

import torch as t
from skeleton.data.collating import collate_hid
from skeleton.data.dataset import ImageDataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split

from skeleton.data.batch import (
    HIDBatch,
    HIDSample,
)
from skeleton.data.preprocess import Preprocessor
from skeleton.evaluation.evaluator import EvaluationPair


################################################################################
# data module implementation


class HotelIDDataModule(LightningDataModule):
    def __init__(
        self,
        data_folder: pathlib.Path,
        batch_size: int,
        num_workers: int,
        preprocessor: Preprocessor,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessor = preprocessor
        self.train_split = 0.9

    def setup(self, stage: Optional[str] = None) -> None:
        # Get the filepaths of all images, and split them between training and validation
        image_folder = self.data_folder / "train_images"
        filepaths = list(image_folder.glob("**/*.jpg"))
        train_filepaths, val_filepaths = train_test_split(filepaths, train_size=self.train_split)
        
        # Set up training dataset & dataloader
        train_ds = ImageDataset(
            train_filepaths,
            transform=self.preprocessor.train_transform,
        )
        
        self.train_dl = t.utils.data.DataLoader(
            train_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_hid,
            drop_last=True,
            shuffle=True
        )

        # Set up validation dataset & dataloader
        val_ds = ImageDataset(
            val_filepaths,
            transform=self.preprocessor.val_transform,
        )
        
        self.val_dl = t.utils.data.DataLoader(
            val_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_hid,
            drop_last=True,
        )

        # Set up testing dataset & dataloader, batch size of 1
        test_ds = ImageDataset(
            list((self.data_folder / "test_images").glob("*.jpg")),
            transform=self.preprocessor.val_transform,
        )
        self.test_dl = t.utils.data.DataLoader(
            test_ds,
            batch_size=1,
        )

    @property
    def num_hotels(self):
        return len(list((self.data_folder / "train_images").iterdir()))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dl

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dl

