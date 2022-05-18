from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class WeightedLoss(nn.Module):
    def __init__(self, loss, weights, gamma=0.1, device="cuda:0"):
        super(WeightedLoss, self).__init__()
        self.loss = loss
        self.weights = weights.to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets, reduction="none")
        alpha = self.weights[targets]
        weighted_loss = (alpha**self.gamma) * loss
        return torch.mean(weighted_loss)


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        alpha = self.alpha[targets]
        f_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        return torch.mean(f_loss)
