################################################################################
#
# This file defines the preprocessing pipeline for train, val and test data.
#
# Author(s): Nik Vaessen
################################################################################

import torch as t

from skeleton.data.batch import HIDSample

################################################################################
# train data pipeline


class Preprocessor:
    def __init__(
        self,
    ) -> None:
        # self.augments =
        pass

    def train_data_pipeline(
        self,
        sample: HIDSample,
    ) -> HIDSample:
        return sample

    def val_data_pipeline(
        self,
        sample: HIDSample,
    ) -> HIDSample:
        return sample

    def test_data_pipeline(
        self,
        sample: HIDSample,
    ) -> HIDSample:
        return sample


################################################################################
# Utility method for normalizing


def normalize(spectogram: t.Tensor, channel_wise: bool):
    if len(spectogram.shape) != 2:
        raise ValueError("expect to normalize over 2D input")

    if channel_wise:
        # calculate over last dimension
        # (assuming shape [n_mels, n_frames])
        std, mean = t.std_mean(spectogram, dim=1)
    else:
        std, mean = t.std_mean(spectogram)

    normalized_spectogram = (spectogram.T - mean) / (std + 1e-9)
    normalized_spectogram = normalized_spectogram.T

    return normalized_spectogram, mean, std
