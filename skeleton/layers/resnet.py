from typing import List

import torch as t
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1) -> None:
        super().__init__()
        # print(in_channels, out_channels, downsample)
        self.relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        # First 3x3 convolution with corresponding BN and ReLU
        # Padding to avoid shift in shape, and no bias as we BN afterwards anyways
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
                stride=stride,
            ),
            bn,
            self.relu,
        )
        # Second 3x3 convolution with corresponding BN, no ReLU
        # Padding to avoid shift in shape, and no bias as we BN afterwards anyways
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False
            ),
            bn,
        )
        # Potentially downsample the identity
        self.downsample = downsample

    def forward(self, x: t.Tensor) -> t.Tensor:
        # Potentially downsample the identity
        if self.downsample:
            identity = self.downsample(x)
        else:
            identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        # Apply the residual identity
        x += identity
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, layers: List[int], out_channels: int) -> None:
        """Apply the ResNet implementation based on `Deep Residual Learning for Image Recognition`:
        https://arxiv.org/abs/1512.03385

        Args:
            layers (List[int]): A list of integers. The length of the list denotes
                the number of layers, and the integer denotes the number of residual
                blocks per layer. So, [3, 4, 6, 3] is the ResNet34 configuration.
            out_channels (_type_): The number of output channels of this ResNet layer.
        """
        super().__init__()
        self.out_channels = int(out_channels)

        # The model sets up with an in-channels of 64, and doubles per layer
        # We use `inter_channels` to denote "intermediate" number of channels
        in_channels = 64
        inter_channels = in_channels

        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(in_channels)
        # Convolution with 7x7 kernel size, stride 2, 64 out_channels
        # We use 1 as in-channels and 64 as out-channels for the first 7x7 convolution
        # We can use bias=False because we BN afterwards anyways
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                1, in_channels, kernel_size=(7, 7), stride=2, padding=3, bias=False
            ),
            bn,
            relu,
        )

        # Apply 3x3 Max pooling, stride 2
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        # The core layers
        core_layers = []
        for layer_num, layer_size in enumerate(layers):
            for block_num in range(layer_size):
                downsample = None
                stride = 1
                if in_channels != inter_channels and layer_num != 0:
                    # Downsampling is a simple 1x1 conv with stride 2 and BN
                    # No downsampling on the first layer
                    downsample = nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            inter_channels,
                            kernel_size=(1, 1),
                            stride=2,
                            bias=False,
                        ),
                        nn.BatchNorm2d(inter_channels),
                    )
                    # The first block per layer has a stride of 2, except for the first layer
                    stride = 2
                core_layers.append(
                    ResidualBlock(
                        in_channels, inter_channels, downsample, stride=stride
                    )
                )
                if downsample:
                    # Set in_channels to be equivalent to inter_channels, as we have now
                    # increased the size of the input of the next ResidualBlock in this layer
                    in_channels = inter_channels

            # Double the number of output channels for every layer
            inter_channels *= 2

        self.conv_core = nn.Sequential(*core_layers)
        # Reset the last multiplication of 2
        inter_channels //= 2

        # Apply average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Linear layer to the correct output size of 1000
        self.linear = nn.Linear(inter_channels, self.out_channels)

    def forward(self, x: t.Tensor) -> t.Tensor:
        # Step 1: Convolution with 7x7 kernel size, stride 2, 64 out_channels
        x = self.conv1(x)

        # Step 2: Apply 3x3 Max pooling, stride 2
        x = self.max_pool(x)

        # Step 3: Apply the core layers
        x = self.conv_core(x)

        # Step 4: Apply the average pooling, and flatten accordingly
        x = self.avg_pool(x)
        x = t.flatten(x, 1)

        # Step 5: Apply the linear layer
        x = self.linear(x)
        return x


class ResNet34(ResNet):
    def __init__(self, out_channels) -> None:
        super().__init__([3, 4, 6, 3], out_channels)


class ResNet18(ResNet):
    def __init__(self, out_channels) -> None:
        super().__init__([2, 2, 2, 2], out_channels)
