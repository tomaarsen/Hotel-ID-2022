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

import torchvision.transforms as T
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
    device = "cpu" if (gpus == 0) else "cuda"
    
    model = HotelID.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    model = model.to(device)

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

        base_dl = DataLoader(base_ds, batch_size=16, num_workers=2, collate_fn=collate_hid)
        
        # test_ds = ImageDataset(
        #     list((data_folder / "test_images").glob("**/*.jpg"))[:20],
        #     hotel_ids,
        #     transform=preprocessor.test_transform,
        #     )
        
        # # Batch size MUST be 1 here
        # test_dl = DataLoader(test_ds, batch_size=1, num_workers=4, collate_fn=collate_hid)

        base_embeddings = t.tensor([], device=device)
        base_hotel_ids = t.tensor([], device=device)
        for batch in tqdm(base_dl, desc="Generating base embeddings"):
            batch = batch.to(device)
            base_embeddings = t.cat((base_embeddings, model(batch.images)))
            base_hotel_ids = t.cat((base_hotel_ids, batch.hotel_ids))
        # Base embeddings generated.

        # List of image paths and the corresponding predictions
        test_image_files = list((data_folder / "test_images").glob("**/*.jpg"))
            
        predictions = []
    
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
            for image in images:
                image = image.unsqueeze(0)
                image = image.to(device)

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
