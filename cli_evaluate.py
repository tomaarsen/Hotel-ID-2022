#! /usr/bin/env python3
################################################################################
#
# Implement the command-line interface for prediction/inference.
#
# Author(s): Nik Vaessen, David van Leeuwen
################################################################################

import csv
import os
import click
import pathlib
from PIL import Image
import numpy as np
import pandas as pd

import torch as t
from tqdm import tqdm

from skeleton.data.preprocess import Preprocessor
from skeleton.models.prototype import HotelID

########################################################################################
# CLI input arguments


@click.command()
@click.option(
    "--checkpoint_path",
    type=pathlib.Path,
    default=None,
    required=True,
    help="the checkpoint to evaluate",
)
@click.option(
    "--data_folder",
    type=pathlib.Path,
    required=True,
)
@click.option("--gpus", type=int, default=1, help="number of gpus to use")
def main(
    checkpoint_path: pathlib.Path,
    data_folder: pathlib.Path,
    gpus: bool,
):
    model = HotelID.load_from_checkpoint(str(checkpoint_path), map_location="cpu")

    # load data pipeline
    preprocessor = Preprocessor(model.width, model.height)

    # set to eval mode, and move to GPU if applicable
    model = model.eval()

    hotel_ids = sorted(
        int(folder.stem) for folder in (data_folder / "train_images").iterdir()
    )

    test_image_files = list((data_folder / "test_images").iterdir())
    predictions = []
    for image_file in tqdm(test_image_files):
        image = Image.open(image_file).convert("RGB")
        image = np.asarray(image)
        image = preprocessor.val_transform(image=image)["image"]
        image = image.unsqueeze(0)
        output = model(image)
        indices = list(output.squeeze().topk(5).indices)
        hids = " ".join(str(hotel_ids[int(index)]) for index in indices)
        predictions.append(hids)

    df = pd.DataFrame(
        data={
            "image_id": (path.name for path in test_image_files),
            "hotel_id": predictions,
        }
    ).sort_values(by="image_id")
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
