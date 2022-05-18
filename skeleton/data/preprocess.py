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


def get_ratio(image):
    (_, width, height) = image.shape
    is_red = (t.logical_and(t.logical_and(t.ge(image[0], 0.99), t.le(image[1], 0.01)), t.le(image[1], 0.01)))
    red_ratio = t.sum(is_red.flatten() == True) / (width * height)
    return red_ratio
    
    

class Preprocessor:
    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        self.width = width
        self.height = height

    def train_transform(self, image):
        return albu.Compose([
            albu.RandomResizedCrop(self.width, self.height, scale=(0.6, 1.0), p=1.0),
            albu.Resize(width=self.width, height=self.height),
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
        # image = transforms.functional.PiL(image)
        # image = transforms.ToTensor()(image)
        # image = image.permute((2,0,1))
        images = transforms.FiveCrop(size=(self.width, self.height))(image)
        final_images = []
        for image in images:
            # image = albu.CoarseDropout(p=1., max_holes=1, 
            #        min_height=self.height//4, max_height=self.height//2,
            #        min_width=self.width//4,  max_width=self.width//2, 
            #        fill_value=(255,0,0))(image=np.asarray(image))["image"]
            image_tensor = transforms.ToTensor()(image)
            ratio_red = get_ratio(image_tensor)
            
            # img = image.permute((1,2,0))
            # plt.imshow(img)
            # plt.savefig("img.png")
            # plt.show()
            # corners = (image[:,0,0], image[:,-1,-1], image[:,0,-1], image[:,-1,0])
            # n_red_corners = sum(tuple(corner) == (255, 0, 0) for corner in corners)
            # ratio_red = image[:,t.logical_and(t.logical_and(image[0] == 254, image[1] == 0), image[2] == 0)].shape[-1] / (image.shape[1] * image.shape[2])
            # img = image.permute(1, 2, 0)
            # ratio_red = (img[img == t.tensor([254, 0, 0])].shape[0] / 3) / (image.shape[1] * image.shape[2])
            
            if ratio_red < 0.2:
                final_images.append(albu.Compose([
                    albu.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            max_pixel_value=255.0),
                    albu.ToFloat(),
                    APT.transforms.ToTensorV2(),
                ])(image=np.asarray(image))["image"])
                
        return final_images