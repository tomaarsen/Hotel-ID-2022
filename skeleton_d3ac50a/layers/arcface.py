import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm
from timm.optim import Lookahead, RAdam

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, device="cpu"):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.device = device

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s

        return output

class HotelIdModel(nn.Module):
    def __init__(self, out_features, embed_size=256, backbone_name="efficientnet_b3"):
        super(HotelIdModel, self).__init__()

        self.embed_size = embed_size
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        in_features = self.backbone.get_classifier().in_features

        fc_name, _ = list(self.backbone.named_modules())[-1]
        if fc_name == 'classifier':
            self.backbone.classifier = nn.Identity()
        elif fc_name == 'head.fc':
            self.backbone.head.fc = nn.Identity()
        elif fc_name == 'fc':
            self.backbone.fc = nn.Identity()
        elif fc_name == 'head.flatten':
            self.backbone.head.fc = nn.Identity()
        elif fc_name == 'head':
            self.backbone.head = nn.Identity()
        else:
            raise Exception("unknown classifier layer: " + fc_name)

        # if backbone_name.startswith("eca_nfnet"):
        #     for param in self.backbone.stem.parameters():
        #         param.requires_grad = False
        #     for param in self.backbone.stages[:-1].parameters():
        #         param.requires_grad = False
        # 3.30 it/s with requires_grad=False
        # 1.80 it/s as baseline
        # 3.45 it/s with requires_grad=False and optimizer filtering

        self.post = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_features, self.embed_size*2), dim=None),
            nn.BatchNorm1d(self.embed_size*2),
            nn.Dropout(0.2),
            nn.utils.weight_norm(nn.Linear(self.embed_size*2, self.embed_size)),
            nn.BatchNorm1d(self.embed_size),
        )

        print(f"Model {backbone_name} ArcMarginProduct - Features: {in_features}, Embeds: {self.embed_size}")
        
    def forward(self, input, targets = None):
        x = self.backbone(input)
        x = x.view(x.size(0), -1)
        x = self.post(x)
        return x