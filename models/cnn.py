import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic Convolutional Block:
    Conv2D → BatchNorm → ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CNNBackbone(nn.Module):
    """
    CNN Backbone for feature extraction.
    Output is a feature map suitable for ViT patch embedding.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            nn.MaxPool2d(kernel_size=2)
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2),
            nn.MaxPool2d(kernel_size=2)
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out_channels = base_channels * 4

    def forward(self, x):
        """
        Input:  (B, 3, H, W)
        Output: (B, C, H/8, W/8)
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x
