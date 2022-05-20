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
    required=True,
    help="the list of checkpoints to evaluate",
)
@click.option(
    "--data_folder",
    type=pathlib.Path,
    required=True,
)
@click.option(
    "--embedding_folder",
    type=pathlib.Path,
    required=True,
)
def main(
    checkpoint_path: pathlib.Path,
    data_folder: pathlib.Path,
    embedding_folder: pathlib.Path,
):
    device = "cuda"
    # all_base_embeddings = []
    # all_base_hotel_ids = []
    # models = []
    # ensembles = []
    # preprocessors = []
    
    
    # for cp_path in checkpoint_path:
    #     ensemble = ENSEMBLE[cp_path.name]
    #     short_hash = ensemble["hash"][:7]
    #     # # Import from the correct folder
    #     # prototype = import_module(f"skeleton_{short_hash}.models.prototype")
    #     # preprocess = import_module(f"skeleton_{short_hash}.data.preprocess")

    #     model = HotelID.load_from_checkpoint(str(cp_path), map_location="cpu")
    #     model = model.to(device)
    #     model = model.eval()
    #     models.append(model)

    #     # load data pipeline
    #     preprocessor = Preprocessor(model.width, model.height)
    #     preprocessors.append(preprocessor)

    #     all_base_embeddings.append(t.load(embedding_folder / f'{ensemble["id"]}_embeds.pt'))
    #     all_base_hotel_ids.append(t.load(embedding_folder / f'{ensemble["id"]}_hids.pt'))

    #     ensembles.append(ensemble)

    # List of image paths and the corresponding predictions
    test_image_files = list((data_folder / "test_images").glob("**/*.jpg"))
        
    predictions = []
    ensemble = ENSEMBLE[checkpoint_path.name]
    model = HotelID.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    model = model.to(device)
    model = model.eval()
    base_embeddings = t.load(embedding_folder / f'{ensemble["id"]}_embeds.pt')
    base_hotel_ids = t.load(embedding_folder / f'{ensemble["id"]}_hids.pt')
    
    preprocessor = Preprocessor(model.width, model.height)

    for image_path in tqdm(test_image_files):
        prediction = []
        pil_image = Image.open(image_path).convert('RGB')
        images = preprocessor.test_transform(pil_image)
        # Apply test transformations to the test image
        # image = Image.open(image_file).convert("RGB")
        # image = np.asarray(image)
        # images = preprocessor.test_transform(image)
        # print(images)
        distances = t.zeros(base_embeddings.shape[0], device=device)
        for i, image in enumerate(images):
            print(f"image {i}/{len(images)}")
            image = image.unsqueeze(0).to(device)

            # Get the embedding, and the 5 hotels with the most similar embeddings
            embedding = model(image)
            distances += t.cosine_similarity(embedding, base_embeddings)
        ranking = distances.sort(descending=True).indices
        for hid in base_hotel_ids[ranking]:
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
