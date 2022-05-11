################################################################################
#
# This file defines the preprocessing pipeline for train, val and test data.
#
# Author(s): Nik Vaessen
################################################################################

import torch as t
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
import numpy as np
from skeleton.data.batch import HIDSample
import albumentations as albu
import albumentations.pytorch as APT

################################################################################


class Preprocessor:
    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        self.width = width
        self.height = height

        # TODO: Add normalization
        # self.train_transform = albu.Compose(
        #     [
        #         albu.RandomResizedCrop(width, height, scale=(0.6, 1.0), p=1.0),
        #         albu.Resize(width=width, height=height),
        #         albu.HorizontalFlip(p=0.5),
        #         albu.OneOf(
        #             [
        #                 albu.RandomBrightness(0.1, p=1),
        #                 albu.RandomContrast(0.1, p=1),
        #                 albu.RandomGamma(p=1),
        #             ],
        #             p=0.3,
        #         ),
        #         albu.ShiftScaleRotate(
        #             shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3
        #         ),
        #         # albu.OneOf([
        #         #     albu.OpticalDistortion(p=0.3),
        #         #     albu.GridDistortion(p=.1),
        #         # ], p=0.2),
        #         # albu.OneOf([
        #         #     albu.MotionBlur(p=.2),
        #         #     albu.MedianBlur(blur_limit=3, p=0.3),
        #         #     albu.Blur(blur_limit=3, p=0.1),
        #         # ], p=0.2),
        #         # albu.HueSaturationValue(p=0.3),
        #         albu.CoarseDropout(
        #             p=0.5,
        #             min_holes=1,
        #             max_holes=6,
        #             min_height=height // 16,
        #             max_height=height // 4,
        #             min_width=width // 16,
        #             max_width=width // 4,
        #         ),
        #         albu.CoarseDropout(
        #             p=1.0,
        #             max_holes=1,
        #             min_height=height // 4,
        #             max_height=height // 2,
        #             min_width=width // 4,
        #             max_width=width // 2,
        #             fill_value=(255, 0, 0),
        #         ),
        #         albu.Normalize(
        #             mean=(0.485, 0.456, 0.406),
        #             std=(0.229, 0.224, 0.225),
        #             max_pixel_value=255.0,
        #         ),
        #         albu.ToFloat(),
        #         APT.transforms.ToTensorV2(),
        #     ]
        # )

    def train_transform(self, image: Image) -> t.Tensor:
        image = T.Compose(
            [
                T.RandomResizedCrop(self.width, scale=(0.6, 1.0)),
                T.Resize(self.width),
                T.AutoAugment(),
            ]
        )(image)

        image = albu.Compose(
            [
                albu.CoarseDropout(
                    p=1.0,
                    max_holes=1,
                    min_height=self.height // 4,
                    max_height=self.height // 2,
                    min_width=self.width // 4,
                    max_width=self.width // 2,
                    fill_value=(255, 0, 0),
                ),
                albu.ToFloat(),
                APT.transforms.ToTensorV2(),
            ]
        )(image=np.asarray(image))["image"]

        return image
        # return albu.Compose(
        #     [
        #         albu.RandomResizedCrop(self.width, self.height, scale=(0.6, 1.0), p=1.0),
        #         albu.Resize(width=self.width, height=self.height),
        #         albu.HorizontalFlip(p=0.5),
        #         albu.OneOf(
        #             [
        #                 albu.RandomBrightness(0.1, p=1),
        #                 albu.RandomContrast(0.1, p=1),
        #                 albu.RandomGamma(p=1),
        #             ],
        #             p=0.3,
        #         ),
        #         albu.ShiftScaleRotate(
        #             shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3
        #         ),
        #         # albu.OneOf([
        #         #     albu.OpticalDistortion(p=0.3),
        #         #     albu.GridDistortion(p=.1),
        #         # ], p=0.2),
        #         # albu.OneOf([
        #         #     albu.MotionBlur(p=.2),
        #         #     albu.MedianBlur(blur_limit=3, p=0.3),
        #         #     albu.Blur(blur_limit=3, p=0.1),
        #         # ], p=0.2),
        #         # albu.HueSaturationValue(p=0.3),
        #         albu.CoarseDropout(
        #             p=0.5,
        #             min_holes=1,
        #             max_holes=6,
        #             min_height=self.height // 16,
        #             max_height=self.height // 4,
        #             min_width=self.width // 16,
        #             max_width=self.width // 4,
        #         ),
        #         albu.CoarseDropout(
        #             p=1.0,
        #             max_holes=1,
        #             min_height=self.height // 4,
        #             max_height=self.height // 2,
        #             min_width=self.width // 4,
        #             max_width=self.width // 2,
        #             fill_value=(255, 0, 0),
        #         ),
        #         albu.Normalize(
        #             mean=(0.485, 0.456, 0.406),
        #             std=(0.229, 0.224, 0.225),
        #             max_pixel_value=255.0,
        #         ),
        #         albu.ToFloat(),
        #         APT.transforms.ToTensorV2(),
        #     ]
        # )(image=np.asarray(image))["image"]

    def val_transform(self, image: Image) -> t.Tensor:
        image = np.asarray(image)
        return albu.Compose(
            [
                albu.Resize(width=self.width, height=self.height),
                albu.CoarseDropout(
                    p=1.0,
                    max_holes=1,
                    min_height=self.height // 4,
                    max_height=self.height // 2,
                    min_width=self.width // 4,
                    max_width=self.width // 2,
                    fill_value=(255, 0, 0),
                ),
                albu.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                ),
                albu.ToFloat(),
                APT.transforms.ToTensorV2(),
            ]
        )(image=image)["image"]

    def test_transform(self, image: Image) -> t.Tensor:
        image = np.asarray(image)
        return albu.Compose(
            [
                albu.Resize(width=self.width, height=self.height),
                albu.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                ),
                albu.ToFloat(),
                APT.transforms.ToTensorV2(),
            ]
        )(image=image)["image"]
