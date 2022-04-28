################################################################################
#
# Utility functions for collating a batch with different number of frames in
# each sample.
#
# The functions assume an input dimensionality of
#
# [NUM_FEATURES, NUM_FRAMES] for MFCC-like input
# or
# [NUM_FRAMES] for wave-like input
#
# This will result in batches with respective dimensionality
# [NUM_SAMPLES, NUM_FEATURES, NUM_FRAMES] or [NUM_SAMPLES, NUM_FRAMES]
#
# Author(s): Nik Vaessen
################################################################################

from typing import Callable, Iterator, List, Optional

import torch as t
from torch.nn import ConstantPad1d, ConstantPad2d
from torch.utils.data.dataloader import default_collate

from skeleton.data.batch import HIDBatch, HIDSample

def collate_hid(
    sample_iter: Iterator[HIDSample],
) -> HIDBatch:
    batch_size = len(sample_iter)
    images, image_ids, hotel_ids = zip(*sample_iter)
    images = default_collate(images)
    image_ids = t.tensor(image_ids)
    hotel_ids = t.tensor(hotel_ids)
    return HIDBatch(batch_size, image_ids, hotel_ids, images)