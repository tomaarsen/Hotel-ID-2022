import os
import pathlib
from typing import Callable, List
from PIL import Image
import torch as t
import torchvision.transforms.transforms as T
from torch.utils.data import Dataset

from skeleton.data.batch import HIDBatch, HIDSample

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
    def __init__(self, filenames: List[pathlib.Path], transform: List[Callable] = None, **kwargs):
        super().__init__()
        self.transform = [T.PILToTensor()]
        if transform:
            self.transform += transform
        self.filenames = filenames

    def __getitem__(self, index: int):
        """
        Get an item from memory
        """
        image_path = self.filenames[index]
        image = Image.open(image_path).convert('RGB')
        image_id = image_path.stem
        hotel_id = image_path.parent.stem
        # width, height = image.width, image.height

        if self.transform:
            for transform in self.transform:
                image = transform(image)

        return HIDSample(
            image,
            image_id,
            hotel_id
        )

    def __len__(self):
        return len(self.filenames)

    # def collate_fn(self, batch: List):
    #     breakpoint()