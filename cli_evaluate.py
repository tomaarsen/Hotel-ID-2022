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

# python .\cli_evaluate.py --checkpoint_path F:\MLiPHotel-IDData\logs\lightning_logs\version_1926876\checkpoints\epoch_0019.step_000201160.val-loss_3.0714.best.ckpt  --checkpoint_path F:\MLiPHotel-IDData\logs\lightning_logs\version_2006869\checkpoints\epoch_0019.step_000100580.val-loss_4.0622.last.ckpt  --checkpoint_path F:\MLiPHotel-IDData\logs\lightning_logs\version_1997481\checkpoints\epoch_0019.step_000201160.val-loss_13.1027.best.ckpt --data_folder F:\MLiPHotel-IDData\Hotel-ID-2022\


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
    "--embedding_folder",
    type=pathlib.Path,
    required=True,
)
def main(
    checkpoint_path: List[pathlib.Path],
    data_folder: pathlib.Path,
    embedding_folder: pathlib.Path,
):
    all_base_embeddings = []
    all_base_hotel_ids = []
    models = []
    ensembles = []
    preprocessors = []
    for cp_path in checkpoint_path:
        ensemble = ENSEMBLE[cp_path.name]
        short_hash = ensemble["hash"][:7]
        # Import from the correct folder
        prototype = import_module(f"skeleton_{short_hash}.models.prototype")
        preprocess = import_module(f"skeleton_{short_hash}.data.preprocess")

        model = prototype.HotelID.load_from_checkpoint(str(cp_path), map_location="cpu")
        model = model.eval()
        models.append(model)

        # load data pipeline
        preprocessor = preprocess.Preprocessor(model.width, model.height)
        preprocessors.append(preprocessor)

        all_base_embeddings.append(t.load(embedding_folder / f'{ensemble["id"]}_embeds.pt', map_location="cpu"))
        all_base_hotel_ids.append(t.load(embedding_folder / f'{ensemble["id"]}_hids.pt', map_location="cpu"))

        ensembles.append(ensemble)
    # List of image paths and the corresponding predictions
    test_image_files = list((data_folder / "test_images").iterdir())
    predictions = []
    distances = t.zeros((len(test_image_files), all_base_embeddings[0].shape[0]), device="cpu")
    for ensemble, model, preprocessor, base_embeddings, base_hotel_ids in zip(
        ensembles, models, preprocessors, all_base_embeddings, all_base_hotel_ids
    ):
        model = model.cuda()
        base_embeddings = base_embeddings.cuda()
        for i, image_file in tqdm(enumerate(test_image_files), "Evaluating images"):
            prediction = []

            # Apply validation transformations to the test image
            image = Image.open(image_file).convert("RGB")
            image = np.asarray(image)
            image = preprocessor.test_transform(image=image)["image"]
            image = image.unsqueeze(0)

            # Get the embedding, and the 5 hotels with the most similar embeddings
            embedding = model(image.cuda())
            distances[i] += t.cosine_similarity(embedding, base_embeddings).detach().cpu()
            del image
            t.cuda.empty_cache()
        del model
        del base_embeddings
    indices = distances.sort(1, descending=True).indices
    predictions = []
    for hids_per_image in all_base_hotel_ids[0][indices]:
        prediction = []
        for hid in hids_per_image:
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
