import os
import pathlib
from typing import Callable, List
from PIL import Image
import numpy as np
import torch as t
import torchvision.transforms.transforms as T
from torch.utils.data import Dataset

from skeleton.data.batch import HIDBatch, HIDSample
import albumentations as albu

class ImageDataset(Dataset):
    """
    Dataset contains folder of images 
    image_dir: `str`
        path to folder of images
    txt_classnames: `str`
        path to .txt file contains classnames
    transform: `List[Callable]`
        A function to call on each image
    """
    def __init__(self, filenames: List[pathlib.Path], transform: Callable = None, **kwargs):
        super().__init__()
        self.to_tensor = T.ToTensor()
        self.transform = transform
        self.filenames = filenames

    def __getitem__(self, index: int):
        """
        Get an item from memory
        """
        image_path = self.filenames[index]
        image = Image.open(image_path).convert('RGB')
        # The augmentations need image to be a numpy array
        image = np.asarray(image)
        image_id = image_path.stem
        hotel_id = image_path.parent.stem
        # width, height = image.width, image.height

        sample = HIDSample(
            image,
            image_id,
            hotel_id
        )

        # Apply transformations, i.e. augmentation
        if self.transform:
            sample = self.transform(sample)

        # Convert to Tensor
        sample.image = self.to_tensor(sample.image)

        return sample

    def __len__(self):
        return len(self.filenames)

    # def collate_fn(self, batch: List):
    #     breakpoint()