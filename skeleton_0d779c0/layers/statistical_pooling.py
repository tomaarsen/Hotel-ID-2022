################################################################################
#
# Implement statistical pooling layers.
#
# They are used to reduce an embedding with a variable time-dimension
# (e.g shape [BATCH_SIZE, NUM_FEATURES, NUM_FRAMES], where NUM_FRAMES indicate
# some sequence in a time dimension) to a fixed-length embedding.
#
# Author(s): Nik Vaessen
################################################################################

import torch as t
import torch.nn as nn

###################################da###########################################
# reduce by taking the mean across the time dimension


class MeanStatPool1D(nn.Module):
    def __init__(self, dim_to_reduce: int = 2):
        super().__init__()
        self.dim_to_reduce = dim_to_reduce

    def forward(self, tensor: t.Tensor):
        # input tensor is expected to have shape
        # [BATCH_SIZE, NUM_FEATURES, NUM_FRAMES] (if dim_to_reduce=2)
        # [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES] (if dim_to_reduce=1)

        # reduce input to shape [BATCH_SIZE, NUM_FEATURES]
        return t.mean(tensor, self.dim_to_reduce)


################################################################################
# reduce by taking the mean and standard deviation (which are stacked on
# top of each other) across the time dimension


class MeanStdStatPool1D(nn.Module):
    def __init__(self, dim_to_reduce: int = 2):
        super().__init__()
        self.dim_to_reduce = dim_to_reduce

    def forward(self, tensor: t.Tensor):
        # input tensor is expected to have shape
        # [BATCH_SIZE, NUM_FEATURES, NUM_FRAMES] (if dim_to_reduce=2)
        # [BATCH_SIZE, NUM_FRAMES, NUM_FEATURES] (if dim_to_reduce=1)

        # reduce input to shape [BATCH_SIZE, 2*NUM_FEATURES]
        return t.cat(t.std_mean(tensor, self.dim_to_reduce), 1)
