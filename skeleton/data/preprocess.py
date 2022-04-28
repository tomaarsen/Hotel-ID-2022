################################################################################
#
# This file defines the preprocessing pipeline for train, val and test data.
#
# Author(s): Nik Vaessen
################################################################################

import torch as t

from skeleton.data.batch import HIDSample
import albumentations as albu

################################################################################
# train data pipeline


class Preprocessor:
    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        self.width = width
        self.height = height
        self.train_transform = albu.Compose([
            albu.RandomResizedCrop(height, width, scale=(0.6, 1.0), p=1.0),
            albu.Resize(height, width),
            albu.HorizontalFlip(p=0.5),
            albu.OneOf([
                albu.RandomBrightness(0.1, p=1),
                albu.RandomContrast(0.1, p=1),
                albu.RandomGamma(p=1)
            ], p=0.3),
            albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
            albu.Cutout(p=0.5, max_h_size=80, max_w_size=80, num_holes=2),
            albu.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0),
        ])
        self.val_transform = albu.Compose([
            albu.Resize(height, width),
            albu.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0),
        ])

    def train_data_pipeline(
        self,
        sample: HIDSample,
    ) -> HIDSample:
        sample.image = self.train_transform(image=sample.image)["image"]
        return sample

    def val_data_pipeline(
        self,
        sample: HIDSample,
    ) -> HIDSample:
        sample.image = self.val_transform(image=sample.image)["image"]
        return sample

    def test_data_pipeline(
        self,
        sample: HIDSample,
    ) -> HIDSample:
        sample.image = self.val_transform(image=sample.image)["image"]
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
