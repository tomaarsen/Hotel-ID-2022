################################################################################
#
# This file defines the preprocessing pipeline for train, val and test data.
#
# Author(s): Nik Vaessen
################################################################################

import torch as t
from torchvision import transforms
import numpy as np
from skeleton_d3ac50a.data.batch import HIDSample
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
        # self.train_transform = albu.Compose([
        #     transforms.RandomCrop(size=(512,512), pad_if_needed=True),
        #     # albu.RandomCr(width, height, scale=(0.6, 1.0), p=1.0),
        #     albu.Resize(width=width, height=height),
        #     albu.HorizontalFlip(p=0.5),
        #     albu.OneOf([
        #         albu.RandomBrightness(0.1, p=1),
        #         albu.RandomContrast(0.1, p=1),
        #         albu.RandomGamma(p=1)], p=0.5),
        #     albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
        #     albu.CoarseDropout(p=0.5, min_holes=1, max_holes=6, 
        #                        min_height=height//16, max_height=height//4,
        #                        min_width=width//16,  max_width=width//4), 
        #     albu.CoarseDropout(p=1., max_holes=1, 
        #                        min_height=height//4, max_height=height//2,
        #                        min_width=width//4,  max_width=width//2, 
        #                        fill_value=(255,0,0)),
        #     albu.Normalize(mean=(0.485, 0.456, 0.406),
        #                std=(0.229, 0.224, 0.225),
        #                max_pixel_value=255.0),
        #     albu.ToFloat(),
        #     APT.transforms.ToTensorV2(),
        # ])
        
    def train_transform(self, image):
        return albu.Compose([
            albu.RandomResizedCrop(self.width, self.height, scale=(0.6, 1.0), p=1.0),
            albu.HorizontalFlip(p=0.5),
            albu.OneOf([
                albu.RandomBrightness(0.1, p=1),
                albu.RandomContrast(0.1, p=1),
                albu.RandomGamma(p=1)], p=0.5),
            albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
            albu.CoarseDropout(p=0.5, min_holes=1, max_holes=6, 
                               min_height=self.height//8, max_height=self.height//4,
                               min_width=self.width//8,  max_width=self.width//4, 
                               fill_value=(255,0,0)), 
            albu.CoarseDropout(p=1., max_holes=1, 
                               min_height=self.height//4, max_height=self.height//2,
                               min_width=self.width//4,  max_width=self.width//2, 
                               fill_value=(255,0,0)),
            albu.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225),
                       max_pixel_value=255.0),
            albu.ToFloat(),
            APT.transforms.ToTensorV2(),
        ])(image=np.asarray(image))["image"]
    
    def val_transform(self, image):
        return albu.Compose([
            albu.Resize(width=self.width, height=self.height),
            albu.CoarseDropout(p=1., max_holes=1, 
                               min_height=self.height//4, max_height=self.height//2,
                               min_width=self.width//4,  max_width=self.width//2, 
                               fill_value=(255,0,0)),
            albu.Normalize(mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225),
                   max_pixel_value=255.0),
            albu.ToFloat(),
            APT.transforms.ToTensorV2(),
            ])(image=np.asarray(image))["image"]
        
    def test_transform(self, image=None):
        return albu.Compose([
            albu.Resize(width=self.width, height=self.height),
            albu.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225),
                       max_pixel_value=255.0),
            albu.ToFloat(),
            APT.transforms.ToTensorV2(),
        ])(image=np.asarray(image))