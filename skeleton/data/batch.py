################################################################################
#
# This file defines a typed batch produced by the dataloader(s)
#
# Author(s): Nik Vaessen
################################################################################


from dataclasses import dataclass
from typing import List

import torch
import torch as t
from torch.utils.data._utils.collate import default_collate

from skeleton.data.collating import collate_append_constant

################################################################################
# data sample and data batch classes


@dataclass
class HIDSample:
    # the image tensor
    image: t.Tensor

    # a unique identifier for this image
    image_id: str

    # class of sample
    hotel_id: str

@dataclass
class HIDBatch:
    # the number of samples this batch contains
    batch_size: str

    # the unique identifiers for the images in this batch
    image_ids: List[str]

    # classes of sample with shape [BATCH_SIZE]
    hotel_ids: t.Tensor

    # tensor of floats with shape [BATCH_SIZE, ...]
    images: t.Tensor

    def __len__(self):
        return self.batch_size

    def to(self, device: torch.device) -> "HIDBatch":
        return HIDBatch(
            self.batch_size,
            self.image_ids,
            self.hotel_ids.to(device),
            self.images.to(device),
        )

    """
    @staticmethod
    def default_collate_fn(
        lst: List[HIDSample],
    ) -> "HIDBatch":
        batch_size = len(lst)
        keys = default_collate([sample.key for sample in lst])
        network_input = default_collate([sample.network_input for sample in lst])
        ground_truth = t.squeeze(
            default_collate([sample.ground_truth for sample in lst])
        )

        return HIDBatch(
            batch_size=batch_size,
            keys=keys,
            network_input=network_input,
            ground_truth=ground_truth,
        )

    @staticmethod
    def pad_right_collate_fn(
        lst: List[HIDSample],
    ) -> "HIDBatch":
        batch_size = len(lst)
        keys = default_collate([sample.key for sample in lst])
        network_input = collate_append_constant(
            [sample.network_input for sample in lst], frame_dim=1, feature_dim=0
        )
        ground_truth = t.squeeze(
            default_collate([sample.ground_truth for sample in lst])
        )

        return HIDBatch(
            batch_size=batch_size,
            keys=keys,
            network_input=network_input.reshape(
                network_input.shape[0], 1, *network_input.shape[1:]
            ),
            ground_truth=ground_truth,
        )
    """
