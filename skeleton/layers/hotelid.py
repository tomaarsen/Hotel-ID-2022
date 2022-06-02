import timm
import torch.nn as nn


class HotelIdModel(nn.Module):
    """Adapted from michaln's work on Hotel-ID 2021"""

    def __init__(self, out_features, embed_size=256, backbone_name="efficientnet_b3"):
        super(HotelIdModel, self).__init__()

        self.embed_size = embed_size
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        in_features = self.backbone.get_classifier().in_features

        fc_name, _ = list(self.backbone.named_modules())[-1]
        if fc_name == "classifier":
            self.backbone.classifier = nn.Identity()
        elif fc_name == "head.fc":
            self.backbone.head.fc = nn.Identity()
        elif fc_name == "fc":
            self.backbone.fc = nn.Identity()
        elif fc_name == "head.flatten":
            self.backbone.head.fc = nn.Identity()
        elif fc_name == "head":
            self.backbone.head = nn.Identity()
        else:
            raise Exception("unknown classifier layer: " + fc_name)

        self.post = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_features, self.embed_size * 2), dim=None),
            nn.BatchNorm1d(self.embed_size * 2),
            nn.Dropout(0.2),
            nn.utils.weight_norm(nn.Linear(self.embed_size * 2, self.embed_size)),
            nn.BatchNorm1d(self.embed_size),
        )

    def forward(self, input, targets=None):
        x = self.backbone(input)
        x = x.view(x.size(0), -1)
        x = self.post(x)
        return x
