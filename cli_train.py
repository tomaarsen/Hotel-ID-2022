#!/usr/bin/env python3
################################################################################
#
# Implement the command-line interface for training a network.
#
# Author(s): Nik Vaessen
################################################################################

import pathlib

import click
import pytorch_lightning
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from skeleton.data.preprocess import Preprocessor
from skeleton.data.hotel_id import HotelIDDataModule
from skeleton.models.prototype import HotelID

################################################################################
# entrypoint of script


@click.command()
@click.option(
    "--data_folder",
    type=pathlib.Path,
    required=True,
)
@click.option(
    "--batch_size",
    type=int,
    default=128,
    help="batch size to use for train and val split",
)
@click.option(
    "--width",
    type=int,
    default=512,
    help="Image width after augmentation",
)
@click.option(
    "--height",
    type=int,
    default=512,
    help="Image height after augmentation",
)
@click.option(
    "--num_workers",
    type=int,
    default=6,
    help="the number of workesr for the data loader, where 0 means no multiprocessing",
)
@click.option(
    "--embedding_size",
    type=int,
    default=128,
    help="dimensionality of learned speaker embeddings",
)
@click.option(
    "--learning_rate",
    type=float,
    default=3e-3,
    help="constant learning rate used during training",
)
@click.option("--epochs", type=int, default=30, help="number of epochs to train for")
@click.option(
    "--momentum",
    type=float,
    default=0.9,
    help="the momentum to use for the SGD optimizer",
)
@click.option(
    "--weight_decay",
    type=float,
    default=0.0005,
    help="the weight decay for the optimizer",
)
@click.option(
    "--min_lr",
    type=float,
    default=0.0,
    help="the minimum learning rate for the CosineAnnealingLR scheduler",
)
@click.option("--gpus", type=int, default=1, help="number of gpus to use")
@click.option("--random_seed", type=int, default=1337, help="the random seed")
@click.option(
    "--checkpoint", type=str, default=None, help="the checkpoint to continue from"
)
def main(
    data_folder: pathlib.Path,
    batch_size: int,
    width: int,
    height: int,
    num_workers: int,
    embedding_size: int,
    learning_rate: float,
    epochs: int,
    momentum: float,
    weight_decay: float,
    min_lr: float,
    gpus: int,
    random_seed: int,
    checkpoint: str,
):
    # log input
    print("### input arguments ###")
    print(f"shard_folder={data_folder}")
    print(f"batch_size={batch_size}")
    print(f"batch_size={width}")
    print(f"batch_size={height}")
    print(f"num_workers={num_workers}")
    print(f"embedding_size={embedding_size}")
    print(f"learning_rate={learning_rate}")
    print(f"epochs={epochs}")
    print(f"momentum={momentum}")
    print(f"weight decay={weight_decay}")
    print(f"min_lr={min_lr}")
    print(f"gpus={gpus}")
    print(f"random_seed={random_seed}")
    print(f"checkpoint={checkpoint}")
    print()

    # set random seed
    pytorch_lightning.seed_everything(random_seed)

    hotel_ids = sorted(int(folder.stem) for folder in (data_folder / "train_images").iterdir())

    # build data loader
    dm = HotelIDDataModule(
        data_folder=data_folder,
        hotel_ids=hotel_ids,
        batch_size=batch_size,
        num_workers=num_workers,
        preprocessor=Preprocessor(width, height),
    )

    # alpha = dm.get_alpha()
    device = 'cuda:0' if gpus != 0 else 'cpu'

    # build model
    model = HotelID(
        num_embedding=embedding_size,
        num_hotels=dm.num_hotels,
        width=width,
        height=height,
        learning_rate=learning_rate,
        num_epochs=epochs,
        momentum=momentum,
        weight_decay=weight_decay,
        min_lr=min_lr,
        epochs=epochs,
        device=device,
    )

    if checkpoint:
        model = model.load_from_checkpoint(
            checkpoint,
            num_embedding=embedding_size,
            num_hotels=dm.num_hotels,
            learning_rate=learning_rate,
            num_epochs=epochs,
            momentum=momentum,
            weight_decay=weight_decay,
            min_lr=min_lr,
            epochs=epochs,
            device=device,
        )

    # configure callback managing checkpoints, and checkpoint file names
    pattern = "epoch_{epoch:04d}.step_{step:09d}.val-loss_{val_loss:.4f}"
    ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
    checkpointer = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=5,
        monitor="val_loss",
        filename=pattern + ".best",
        save_last=True,
        auto_insert_metric_name=False,
    )

    # initialize trainer
    trainer = pytorch_lightning.Trainer(
        stochastic_weight_avg=True,
        max_epochs=epochs,
        gpus=gpus,
        callbacks=[
            checkpointer,
            LearningRateMonitor(),
        ],
        default_root_dir="logs",
    )

    # train loop
    trainer.fit(model, datamodule=dm)

    # test loop (on dev set)
    model = model.load_from_checkpoint(checkpointer.best_model_path)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
