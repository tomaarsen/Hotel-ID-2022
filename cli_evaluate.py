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
from torch.utils.data import DataLoader
from tqdm import tqdm
from skeleton.data.collating import collate_hid
from skeleton.data.dataset import ImageDataset
from skeleton.data.hotel_id import HotelIDDataModule

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
    model = model.to("cuda")

    # load data pipeline
    preprocessor = Preprocessor(model.width, model.height)

    # set to eval mode, and move to GPU if applicable
    model = model.eval()

    # Get the list of hotel_ids, i.e. a mapping of index/label to hotel ID
    hotel_ids = sorted(
        int(folder.stem) for folder in (data_folder / "train_images").iterdir()
    )

    # hotelid_dm = HotelIDDataModule(data_folder, hotel_ids, 1, 2, preprocessor, 0.0)
    # base_dl = hotelid_dm.val_dataloader()
    # base_embeddings = [sample for sample in base_dl]
    # breakpoint()
    with t.no_grad():
        # Generate the base embeddings...
        base_ds = ImageDataset(
            list((data_folder / "train_images").glob("**/*.jpg")),
            hotel_ids,
            transform=preprocessor.val_transform,
        )

        base_dl = DataLoader(base_ds, batch_size=16, num_workers=3, collate_fn=collate_hid)

        base_embeddings = t.tensor([], device="cuda")
        base_hotel_ids = t.tensor([], device="cuda")
        for batch in tqdm(base_dl, desc="Generating base embeddings"):
            batch = batch.to("cuda")
            base_embeddings = t.cat((base_embeddings, model(batch.images)))
            base_hotel_ids = t.cat((base_hotel_ids, batch.hotel_ids))
        # Base embeddings generated.

        # List of image paths and the corresponding predictions
        test_image_files = list((data_folder / "test_images").iterdir())
        predictions = []

        for image_file in tqdm(test_image_files):
            prediction = []
            
            # Apply validation transformations to the test image
            image = Image.open(image_file).convert("RGB")
            image = np.asarray(image)
            image = preprocessor.test_transform(image=image)["image"]
            image = image.unsqueeze(0)
            image = image.to("cuda")

            # Get the embedding, and the 5 hotels with the most similar embeddings
            embedding = model(image)
            distances = t.cosine_similarity(embedding, base_embeddings)
            sorted_dist, indices = distances.sort(descending=True)
            for hid in base_hotel_ids[indices]:
                if hid in prediction:
                    continue
                prediction.append(hid)
                if len(prediction) == 5:
                    break
            predictions.append(" ".join(str(int(pred)) for pred in prediction))

    df = pd.DataFrame(
        data={
            "image_id": (path.name for path in test_image_files),
            "hotel_id": predictions,
        }
    ).sort_values(by="image_id")
    df.to_csv("submission.csv", index=False)
    print(df.head())


if __name__ == "__main__":
    main()
