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
# from skeleton.data.collating import collate_hid
# from skeleton.data.dataset import ImageDataset
# from skeleton.data.hotel_id import HotelIDDataModule

# from skeleton.data.preprocess import Preprocessor
# from skeleton.models.prototype import HotelID

from importlib import import_module

########################################################################################
# CLI input arguments

ENSEMBLE = [
    {
        "model": "eca_nfnet_l2 + ArcFace",
        "map@5": 0.520,
        "val_acc": 0.592,
        "id": 1926876,
        "hash": "0d779c0352980ffb6db9a1cda3f6f412983544b2",
        "checkpoint": "epoch_0019.step_000201160.val-loss_3.0714.last.ckpt",
    },
    {
        "model": "efficientnet_b3 + ArcFace",
        "map@5": None,
        "val_acc": 0.461,
        "id": 2006869,
        "hash": "d3ac50adff07b7146a6b72a1deb36f1dd1af3e61",
        "checkpoint": "epoch_0019.step_000100580.val-loss_4.0622.last.ckpt",
    },
    {
        "model": "eca_nfnet_l2 + CosFace",
        "map@5": 0.499,
        "val_acc": None,
        "id": 1997481,
        "hash": "aa86a11f8ebd9099f7350ec9d95e8ed3579b7fd6",
        "checkpoint": "epoch_0019.step_000201160.val-loss_13.1027.best.ckpt",
    },
]

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
@click.option("--test", type=bool, default=False, help="Whether to use only a few training embeddings")
def main(
    checkpoint_path: List[pathlib.Path],
    data_folder: pathlib.Path,
    test: bool,
):
    if len(checkpoint_path) != len(ENSEMBLE):
        raise Exception(f"Expected {len(ENSEMBLE)} checkpoint paths, received {len(checkpoint_path)}.")

    all_base_embeddings = []
    models = []
    preprocessors = []
    for cp_path, ensemble in zip(checkpoint_path, ENSEMBLE):
        short_hash = ensemble["hash"][:7]
        # Import from the correct folder
        prototype = import_module(f"skeleton_{short_hash}.models.prototype")
        dataset = import_module(f"skeleton_{short_hash}.data.dataset")
        preprocess = import_module(f"skeleton_{short_hash}.data.preprocess")
        collating = import_module(f"skeleton_{short_hash}.data.collating")

        model = prototype.HotelID.load_from_checkpoint(str(cp_path), map_location="cpu")
        model = model.to("cuda")
        models.append(model)

        # load data pipeline
        preprocessor = preprocess.Preprocessor(model.width, model.height)
        preprocessors.append(preprocessor)

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
            filenames = list((data_folder / "train_images").glob("**/*.jpg"))
            if test:
                filenames = filenames[:10]
            base_ds = dataset.ImageDataset(
                filenames,
                hotel_ids,
                transform=preprocessor.test_transform,
            )

            base_dl = DataLoader(base_ds, batch_size=16, num_workers=3, collate_fn=collating.collate_hid)

            base_embeddings = t.tensor([], device="cuda")
            base_hotel_ids = t.tensor([], device="cuda")
            for batch in tqdm(base_dl, desc=f"Generating base embeddings for {ensemble['model']}"):
                batch = batch.to("cuda")
                base_embeddings = t.cat((base_embeddings, model(batch.images)))
                base_hotel_ids = t.cat((base_hotel_ids, batch.hotel_ids))
            # Base embeddings generated.
            all_base_embeddings.append(base_embeddings)

    # List of image paths and the corresponding predictions
    test_image_files = list((data_folder / "test_images").iterdir())
    predictions = []

    for image_file in tqdm(test_image_files, "Evaluating images"):
        prediction = []

        distances = t.zeros(all_base_embeddings[0].shape[0], device="cuda")
        for ensemble, model, preprocessor, base_embeddings in zip(ENSEMBLE, models, preprocessors, all_base_embeddings):           
            # Apply validation transformations to the test image
            image = Image.open(image_file).convert("RGB")
            image = np.asarray(image)
            image = preprocessor.test_transform(image=image)["image"]
            image = image.unsqueeze(0)
            image = image.to("cuda")

            # Get the embedding, and the 5 hotels with the most similar embeddings
            embedding = model(image)
            distances += t.cosine_similarity(embedding, base_embeddings)
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
