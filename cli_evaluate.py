#! /usr/bin/env python3
################################################################################
#
# Implement the command-line interface for prediction/inference.
#
# Author(s): Nik Vaessen, David van Leeuwen
################################################################################

import argparse
import traceback
from collections import defaultdict
from pathlib import Path

import torch as t
from tqdm import tqdm

from skeleton.data.batch import HIDBatch
from skeleton.data.preprocess import Preprocessor
from skeleton.data.hotel_id import (_collate_samples, init_dataset_test,
                                         init_dataset_train,
                                         load_evaluation_pairs)
from skeleton.evaluation.eval_metrics import calculate_eer
from skeleton.evaluation.evaluator import (EmbeddingSample,
                                           SpeakerRecognitionEvaluator)
from skeleton.models.prototype import HotelID

########################################################################################
# CLI input arguments


parser = argparse.ArgumentParser()

# required positional paths to data
parser.add_argument(
    "checkpoint",
    type=Path,
    help="path to checkpoint file",
)
parser.add_argument(
    "shard_folder",
    type=Path,
    help="paths to shard folder",
)
parser.add_argument(
    "shards_dirs_to_evaluate",
    type=str,
    help="paths to shard file containing all audio files in the given trial list (',' separated)",
)
parser.add_argument(
    "trial_lists", type=str, help="file with list of trials (',' separated)"
)
parser.add_argument("score_file", type=Path, help="output file to store scores")

# optional arguments based on pre-processing variables
parser.add_argument(
    "--normalize_channel_wise",
    type=bool,
    default=True,
    help="whether to normalize each input mel band separately, should be the same as in train",
)
parser.add_argument(
    "--n_mels",
    type=int,
    default=40,
    help="number of mel bands to generate, should be the same as in train",
)
parser.add_argument(
    "--network",
    type=str,
    default="proto",
    help="which network to load from the given checkpoint",
)
parser.add_argument(
    "--use-gpu",
    type=lambda x: x.lower() in ("yes", "true", "t", "1"),
    default=False,
    help="whether to evaluate on a GPU device",
)

################################################################################
# entrypoint of script


def main(
    checkpoint_path: Path,
    shard_folder: str,
    shards_dirs_to_evaluate: str,
    trial_lists: str,
    score_file: Path,
    normalize_channel_wise: bool,
    n_mels: int,
    network: str,
    use_gpu: bool,
):
    # load module from checkpoint
    if network == "proto":
        try:
            model = HotelID.load_from_checkpoint(
                str(checkpoint_path), map_location="cpu"
            )
        except RuntimeError:
            traceback.print_exc()
            raise ValueError(
                f"Failed to load PrototypeSpeakerRecognitionModule with "
                f"{checkpoint_path=}. Is {network=} correct?"
            )
    else:
        raise ValueError(f"unknown {network=}")

    # load data pipeline
    preprocessor = Preprocessor(
        audio_length_seconds=30,
        n_mels=n_mels,
        normalize_channel_wise=normalize_channel_wise,
        normalize=True,
    )

    # set to eval mode, and move to GPU if applicable
    model = model.eval()

    use_gpu = use_gpu and t.cuda.is_available()
    if use_gpu:
        model = model.to("cuda")

    # print(model)
    print(f"{use_gpu=}")

    # split input string if more than 1 test set was given
    shards_dirs = shards_dirs_to_evaluate.split(",")
    trial_lists = trial_lists.split(",")

    embeddings_list = defaultdict(list)

    # Use the train shard folder, but with the simpler validation data pipeline (i.e. no augmentation)
    ds = init_dataset_test(shard_folder / "train")
    ds = ds.map(preprocessor.val_data_pipeline).batched(
        1, collation_fn=_collate_samples
    )
    # loop over all data in the shard and store the speaker embeddings
    with t.no_grad():  # disable autograd as we only do inference
        for x in tqdm(ds):
            assert isinstance(x, HIDBatch)
            assert x.batch_size == 1

            if use_gpu:
                x = x.to("cuda")

            # determine key
            key = x.keys[0].split("/")[0]

            # compute embedding
            embedding, _ = model(x.network_input)

            # store embedding, and save on CPU so that GPU memory does not fill up
            embedding = embedding.to("cpu")
            embeddings_list[key].append(embedding.squeeze())

    cohort = t.stack(
        [t.mean(t.stack(embeddings), dim=0) for embeddings in embeddings_list.values()]
    ).to("cpu")

    # Normalize by subtracting mean and dividing by std
    cohort = (cohort - model.mean_embedding.to("cpu")) / model.std_embedding.to("cpu")
    # Clear the deque to separate the cohort embeddings from the rest of the embeddings
    model.embeddings_queue.clear()

    # process dev and test simultaneously
    pairs = list()
    embeddings = list()

    for shards_dir, trial_list in zip(shards_dirs, trial_lists):
        print("Processing shards dir", shards_dir)

        # init webdataset pipeline
        dataset = init_dataset_test(Path(shards_dir))
        dataset = dataset.map(preprocessor.test_data_pipeline).batched(
            1, collation_fn=_collate_samples
        )

        # load every trial pair
        pairs.extend(load_evaluation_pairs(Path(trial_list)))

        # loop over all data in the shard and store the speaker embeddings
        with t.no_grad():  # disable autograd as we only do inference
            for x in tqdm(dataset):
                assert isinstance(x, HIDBatch)
                assert x.batch_size == 1

                if use_gpu:
                    x = x.to("cuda")

                # compute embedding
                embedding, _ = model(x.network_input)

                # determine key (for matching with trial pairs)
                key = x.keys[0]
                if key.startswith("unknown"):
                    # for eval, path in shard is unknown/unknown/$key
                    key = key.split("/")[-1]

                # store embedding, and save on CPU so that GPU memory does not fill up
                embedding = embedding.to("cpu")
                embeddings.append(EmbeddingSample(key, embedding.squeeze()))

    # move model weights back to CPU so that mean_embedding and std_embedding
    # are on same device as embeddings
    model = model.to("cpu")

    # for each trial, compute scores based on cosine similarity between
    # speaker embeddings
    scores = SpeakerRecognitionEvaluator.evaluate(
        pairs,
        embeddings,
        cohort,
        model.mean_embedding,
        model.std_embedding,
        skip_eer=True,
    )

    # write each trial (with computed score) to a file
    eer_scores = list()
    eer_labels = list()
    with open(score_file, "w") as out:
        for score, pair in zip(scores, pairs):
            print(score, f"{pair.sample1_id}.wav", f"{pair.sample2_id}.wav", file=out)
            if pair.same_speaker is not None:
                eer_scores.append(score)
                eer_labels.append(pair.same_speaker)

    # for the dev set, compute EER based on predicted scores and ground truth labels
    if len(eer_scores) > 0:
        eer = calculate_eer(groundtruth_scores=eer_labels, predicted_scores=eer_scores)
        print(
            f"EER computed over {len(eer_scores)} trials for which truth is known: {eer*100:4.2f}%"
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        shard_folder=args.shard_folder,
        shards_dirs_to_evaluate=args.shards_dirs_to_evaluate,
        trial_lists=args.trial_lists,
        score_file=args.score_file,
        normalize_channel_wise=args.normalize_channel_wise,
        n_mels=args.n_mels,
        network=args.network,
        use_gpu=args.use_gpu,
    )
