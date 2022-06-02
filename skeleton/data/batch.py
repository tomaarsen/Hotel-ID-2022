from __future__ import annotations

from dataclasses import dataclass

import torch as t

################################################################################
# data sample and data batch classes


@dataclass
class HIDSample:
    # the image tensor
    image: t.Tensor

    # a unique identifier for this image
    image_id: int

    # hotel ID for this sample
    hotel_id: int

    # class of sample
    label: int

    def __iter__(self):
        return iter((self.image, self.image_id, self.hotel_id, self.label))


@dataclass
class HIDBatch:
    # the number of samples this batch contains
    batch_size: int

    # the unique identifiers for the images in this batch
    image_ids: t.Tensor

    # hotel ids of samples for the images in this batch
    hotel_ids: t.Tensor

    # classes of samples for the images in this batch
    labels: t.Tensor

    # tensor of floats with shape [BATCH_SIZE, ...]
    images: t.Tensor

    def __len__(self) -> int:
        return self.batch_size

    def to(self, device: t.device) -> HIDBatch:
        return HIDBatch(
            self.batch_size,
            self.image_ids.to(device),
            self.hotel_ids.to(device),
            self.labels.to(device),
            self.images.to(device),
        )
