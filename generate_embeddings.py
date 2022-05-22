#! /usr/bin/env python3
################################################################################
#
# Implement the command-line interface for prediction/inference.
#
# Author(s): Nik Vaessen, David van Leeuwen
################################################################################

import csv
import os
from typing import List
import click
import pathlib
from PIL import Image
import numpy as np
import pandas as pd

import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm
from ensemble import ENSEMBLE
from importlib import import_module

########################################################################################
# CLI input arguments

# aa86a11f8ebd9099f7350ec9d95e8ed3579b7fd6 (CosFace) was modified at the forward() call to return the embedding
# d3ac50adff07b7146a6b72a1deb36f1dd1af3e61 was modified at preprocess.py test_transform to return a dict with "image"
# d3ac50adff07b7146a6b72a1deb36f1dd1af3e61 was modified at dataset.py to add ["image"]

# python .\generate_embeddings.py --checkpoint_path F:\MLiPHotel-IDData\logs\lightning_logs\version_1926876\checkpoints\epoch_0019.step_000201160.val-loss_3.0714.last.ckpt  --checkpoint_path F:\MLiPHotel-IDData\logs\lightning_logs\version_2006869\checkpoints\epoch_0019.step_000100580.val-loss_4.0622.last.ckpt  --checkpoint_path F:\MLiPHotel-IDData\logs\lightning_logs\version_1997481\checkpoints\epoch_0019.step_000201160.val-loss_13.1027.best.ckpt --data_folder F:\MLiPHotel-IDData\Hotel-ID-2022\


@click.command()
@click.option(
    "--checkpoint_path",
    type=pathlib.Path,
    required=True,
    multiple=True,
    help="the list of checkpoints to evaluate",
)
@click.option(
    "--data_folder",
    type=pathlib.Path,
    required=True,
)
@click.option(
    "--test",
    type=bool,
    default=False,
    help="Whether to use only a few training embeddings",
)
def main(
    checkpoint_path: List[pathlib.Path],
    data_folder: pathlib.Path,
    test: bool,
):
    for cp_path in checkpoint_path:
        ensemble = ENSEMBLE[cp_path.name]
        short_hash = ensemble["hash"][:7]
        # Import from the correct folder
        prototype = import_module(f"skeleton_{short_hash}.models.prototype")
        dataset = import_module(f"skeleton_{short_hash}.data.dataset")
        preprocess = import_module(f"skeleton_{short_hash}.data.preprocess")
        collating = import_module(f"skeleton_{short_hash}.data.collating")

        model = prototype.HotelID.load_from_checkpoint(str(cp_path), map_location="cpu")
        # set to eval mode, and move to GPU if applicable
        model = model.to("cuda")
        model = model.eval()

        # load data pipeline
        preprocessor = preprocess.Preprocessor(model.width, model.height)

        # Get the list of hotel_ids, i.e. a mapping of index/label to hotel ID
        hotel_ids = sorted(
            int(folder.stem) for folder in (data_folder / "train_images").iterdir()
        )

        with t.no_grad():
            # Generate the base embeddings...
            filenames = sorted((data_folder / "train_images").glob("**/*.jpg"))
            if test:
                filenames = filenames[:10]
            base_ds = dataset.ImageDataset(
                filenames,
                hotel_ids,
                transform=preprocessor.test_transform,
            )

            base_dl = DataLoader(
                base_ds, batch_size=16, num_workers=3, collate_fn=collating.collate_hid
            )

            base_embeddings = t.tensor([], device="cuda")
            base_hotel_ids = t.tensor([], device="cuda")
            for batch in tqdm(
                base_dl, desc=f"Generating base embeddings for {ensemble['model']}"
            ):
                batch = batch.to("cuda")
                base_embeddings = t.cat((base_embeddings, model(batch.images)))
                base_hotel_ids = t.cat((base_hotel_ids, batch.hotel_ids))
            t.save(base_embeddings, f'saved/{ensemble["id"]}_embeds.pt')
            t.save(base_hotel_ids, f'saved/{ensemble["id"]}_hids.pt')


if __name__ == "__main__":
    main()
