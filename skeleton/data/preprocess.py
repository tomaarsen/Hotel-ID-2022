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

################################################################################

class Preprocessor:
    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        self.width = width
        self.height = height
        
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((width,height))
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((width,height))
        ])
