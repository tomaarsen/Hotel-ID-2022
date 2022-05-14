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
    def __init__(self, filenames: List[pathlib.Path], hotel_ids: List[int], transform: Callable = None, **kwargs):
        super().__init__()
        self.to_tensor = T.ToTensor()
        self.hotel_ids = hotel_ids
        self.transform = transform
        self.filenames = filenames

    def __getitem__(self, index: int):
        """
        Get an item from memory
        """
        image_path = self.filenames[index]
        image = Image.open(image_path).convert('RGB')
        # The augmentations need image to be a numpy array
        # image = np.asarray(image)
        image_id = int(image_path.stem)
        hotel_id = int(image_path.parent.stem)
        label = self.hotel_ids.index(hotel_id)

        # Apply transformations, i.e. augmentation
        if self.transform:
            image = self.transform(image=image)

        sample = HIDSample(
            image,
            image_id,
            hotel_id,
            label
        )
        return sample

    def __len__(self):
        return len(self.filenames)

    # def collate_fn(self, batch: List):
    #     breakpoint()