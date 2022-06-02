from typing import Iterator

import torch as t
from torch.utils.data.dataloader import default_collate

from skeleton.data.batch import HIDBatch, HIDSample


def collate_hid(
    sample_iter: Iterator[HIDSample],
) -> HIDBatch:
    batch_size = len(sample_iter)
    images, image_ids, hotel_ids, labels = zip(*sample_iter)
    images = default_collate(images)
    image_ids = t.tensor(image_ids)
    hotel_ids = t.tensor(hotel_ids)
    labels = t.tensor(labels)
    return HIDBatch(batch_size, image_ids, hotel_ids, labels, images)
