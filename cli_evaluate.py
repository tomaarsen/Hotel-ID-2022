#! /usr/bin/env python3
################################################################################
#
# Implement the command-line interface for prediction/inference.
#
# Author(s): Nik Vaessen, David van Leeuwen
################################################################################

import click
import traceback
from collections import defaultdict
import pathlib
import pytorch_lightning

import torch as t
from tqdm import tqdm

from skeleton.data.batch import HIDBatch
from skeleton.data.preprocess import Preprocessor
from skeleton.data.hotel_id import (
    HotelIDDataModule,
    _collate_samples,
    init_dataset_test,
    init_dataset_train,
    load_evaluation_pairs,
)
from skeleton.evaluation.evaluator import EmbeddingSample, SpeakerRecognitionEvaluator
from skeleton.models.prototype import HotelID

########################################################################################
# CLI input arguments


@click.command()
@click.option(
    "--checkpoint_path",
    type=pathlib.Path,
    default=None,
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

    use_gpu = use_gpu and t.cuda.is_available()
    if use_gpu:
        model = model.to("cuda")

    num_workers = 1
    hotelid_dm = HotelIDDataModule(
        data_folder, model.batch_size, num_workers, preprocessor
    )

    # move model weights back to CPU so that mean_embedding and std_embedding
    # are on same device as embeddings
    model = model.to("cpu")

    # initialize trainer
    trainer = pytorch_lightning.Trainer(
        gpus=gpus,
        default_root_dir="logs",
    )

    trainer.test(model, datamodule=hotelid_dm)


if __name__ == "__main__":
    main()
