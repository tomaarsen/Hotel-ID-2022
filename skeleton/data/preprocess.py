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
import matplotlib.pyplot as plt

################################################################################
    

class Preprocessor:
    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        self.width = width
        self.height = height
        self.red_threshold = 0.25
        self.n_crops = 8
        
    def _get_ratio(self, image):
        is_red = (t.logical_and(t.logical_and(t.ge(image[0], 0.99), t.le(image[1], 0.01)), t.le(image[1], 0.01)))
        red_ratio = t.sum(is_red.flatten() == True) / (self.width * self.height)
        return red_ratio

    def train_transform(self, image):
        return albu.Compose([
            albu.RandomResizedCrop(self.width, self.height, scale=(0.6, 1.0), p=1.0),
            # albu.Resize(width=self.width, height=self.height),
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
            # albu.CoarseDropout(p=1., max_holes=1, 
            #                    min_height=self.height//4, max_height=self.height//2,
            #                    min_width=self.width//4,  max_width=self.width//2, 
            #                    fill_value=(255,0,0)),
            albu.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225),
                       max_pixel_value=255.0),
            albu.ToFloat(),
            APT.transforms.ToTensorV2(),
        ])(image=np.asarray(image))["image"]
    
    def val_transform(self, image):
        return albu.Compose([
            albu.Resize(width=self.width, height=self.height),
            albu.CoarseDropout(p=0.5, max_holes=2, 
                           min_height=self.height//8, max_height=self.height//4,
                           min_width=self.width//8,  max_width=self.width//4, 
                           fill_value=(255,0,0)),
            albu.Normalize(mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225),
                   max_pixel_value=255.0),
            albu.ToFloat(),
            APT.transforms.ToTensorV2(),
            ])(image=np.asarray(image))["image"]
        
    def test_transform(self, image):       
        images = []
        # plt.imshow(image)
        # plt.savefig("img_before.png")
        for i in range(self.n_crops):
            images.append(albu.RandomResizedCrop(self.width, self.height, scale=(0.2, 0.8), p=1.0)(image=np.asarray(image))["image"])
            # plt.imshow(images[i])
            # plt.savefig(f"img_after_{i}.png")
            
        final_images = []
        for image in images:
            # Compute red ration
            image_tensor = transforms.ToTensor()(image)
            ratio_red = self._get_ratio(image_tensor)
            
            # Only consider image if red ratio below threshold 
            if ratio_red < self.red_threshold:
                final_images.append(albu.Compose([
                    albu.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            max_pixel_value=255.0),
                    albu.ToFloat(),
                    APT.transforms.ToTensorV2(),
                ])(image=np.asarray(image))["image"])
        
        # Ensure at least one image is in list
        if (len(final_images) == 0):
            final_images = self.test_transform_basic(image)
            final_images = [final_images]
                
        return t.stack(final_images)
    
    def test_transform_basic(self, image):
        return albu.Compose([albu.RandomResizedCrop(self.width, self.height, scale=(0.8, 1.0), p=1.0),
                      albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0), 
                      albu.ToFloat(),
                      APT.transforms.ToTensorV2(),
                ])(image=np.asarray(image))["image"]
        