import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockBN(nn.Module):
    """
    Residual block:
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN, with skip connection.
    """
    def __init__(self, channels: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(channels, channels, 3, 1, 1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(channels, channels, 3, 1, 1))
        if use_bn:
            layers.append(nn.BatchNorm2d(channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)

class SRResCNN(nn.Module):
    """
    Head -> (num_blocks x ResidualBlockBN) -> PixelShuffle upsampler -> Tail

    With global_residual=True (default) the network predicts a correction on
    top of the bicubic upsample instead of the full image. This is
    parameter-free (state dicts are unchanged), converges much faster, and
    lower-bounds quality at roughly the bicubic baseline.
    """
    def __init__(self, in_ch=3, channels=64, num_blocks=8, scale=2, use_bn=True,
                 global_residual=True):
        super().__init__()
        self.scale = scale
        self.global_residual = global_residual
        self.head = nn.Conv2d(in_ch, channels, kernel_size=3, stride=1, padding=1)

        body = [ResidualBlockBN(channels, use_bn=use_bn) for _ in range(num_blocks)]
        self.body = nn.Sequential(*body)

        up = []
        s = scale
        while s > 1:
            up.append(nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1))
            up.append(nn.PixelShuffle(2))
            up.append(nn.ReLU(inplace=True))
            s //= 2
        self.upsample = nn.Sequential(*up) if up else nn.Identity()
        self.tail = nn.Conv2d(channels, in_ch, kernel_size=3, stride=1, padding=1)
        if global_residual:
            # Zero-init the tail so the untrained network starts exactly at the
            # bicubic baseline and only learns the correction.
            nn.init.zeros_(self.tail.weight)
            nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        feat = self.upsample(feat)
        out = self.tail(feat)
        if self.global_residual:
            base = F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                                 align_corners=False)
            out = out + base
        return out
