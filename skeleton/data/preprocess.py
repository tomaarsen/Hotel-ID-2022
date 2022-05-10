################################################################################
#
# This file defines the preprocessing pipeline for train, val and test data.
#
# Author(s): Nik Vaessen
################################################################################

import torch as t
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
        self.train_transform = albu.Compose([
            albu.RandomResizedCrop(width, height, scale=(0.6, 1.0), p=1.0),
            albu.Resize(width=width, height=height),
            albu.HorizontalFlip(p=0.5),
            albu.OneOf([
                albu.RandomBrightness(0.1, p=1),
                albu.RandomContrast(0.1, p=1),
                albu.RandomGamma(p=1)], p=0.3),
            albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
            albu.CoarseDropout(p=0.5, min_holes=1, max_holes=6, 
                               min_height=height//16, max_height=height//4,
                               min_width=width//16,  max_width=width//4), 

            albu.CoarseDropout(p=1., max_holes=1, 
                               min_height=height//4, max_height=height//2,
                               min_width=width//4,  max_width=width//2, 
                               fill_value=(255,0,0)),
            albu.ToFloat(),
            APT.transforms.ToTensorV2(),
        ])
        
        self.val_transform = albu.Compose([
            albu.Resize(width=width, height=height),
            albu.CoarseDropout(p=1., max_holes=1, 
                               min_height=height//4, max_height=height//2,
                               min_width=width//4,  max_width=width//2, 
                               fill_value=(255,0,0)),
            albu.ToFloat(),
            APT.transforms.ToTensorV2(),
        ])
        
        self.test_transform = albu.Compose([
            albu.Resize(width=width, height=height),
            albu.ToFloat(),
            APT.transforms.ToTensorV2(),
        ])
