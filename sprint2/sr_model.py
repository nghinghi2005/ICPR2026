from __future__ import annotations

import torch
import torch.nn as nn


def _make_conv(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> nn.Conv2d:
    if p is None:
        p = k // 2
    return nn.Conv2d(in_ch, out_ch, k, s, p)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class EDSRLite(nn.Module):
    """Small SR model (EDSR-like) using PixelShuffle.

    Designed to be lightweight enough for Kaggle notebook runs.
    """

    def __init__(self, scale: int = 2, num_blocks: int = 8, channels: int = 64):
        super().__init__()
        self.scale = int(scale)
        if self.scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")

        self.head = nn.Conv2d(3, channels, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(channels, channels, 3, 1, 1)

        up_layers = []
        s = self.scale
        while s > 1:
            up_layers.append(nn.Conv2d(channels, channels * 4, 3, 1, 1))
            up_layers.append(nn.PixelShuffle(2))
            up_layers.append(nn.ReLU(True))
            s //= 2
        self.up = nn.Sequential(*up_layers)
        self.tail = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1, 1]
        feat = self.head(x)
        res = self.body_conv(self.body(feat))
        feat = feat + res
        out = self.tail(self.up(feat))
        return out


class DenseResidualBlock(nn.Module):
    def __init__(self, channels: int, growth: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = float(res_scale)

        self.c1 = _make_conv(channels, growth)
        self.c2 = _make_conv(channels + growth, growth)
        self.c3 = _make_conv(channels + 2 * growth, growth)
        self.c4 = _make_conv(channels + 3 * growth, growth)
        self.c5 = _make_conv(channels + 4 * growth, channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.c1(x))
        x2 = self.lrelu(self.c2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.c3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.c4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * self.res_scale


class RRDB(nn.Module):
    def __init__(self, channels: int, growth: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = float(res_scale)
        self.b1 = DenseResidualBlock(channels, growth=growth, res_scale=res_scale)
        self.b2 = DenseResidualBlock(channels, growth=growth, res_scale=res_scale)
        self.b3 = DenseResidualBlock(channels, growth=growth, res_scale=res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)
        return x + out * self.res_scale


class RRDBNetLite(nn.Module):
    """ESRGAN-style RRDB generator (lite).

    This is significantly stronger than EDSRLite for texture/detail, while still
    being small enough to train a bit inside a Kaggle notebook.
    """

    def __init__(
        self,
        scale: int = 2,
        num_rrdb: int = 6,
        channels: int = 64,
        growth: int = 32,
    ):
        super().__init__()
        self.scale = int(scale)
        if self.scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")

        self.conv_first = _make_conv(3, channels)
        self.rrdb_trunk = nn.Sequential(*[RRDB(channels, growth=growth) for _ in range(int(num_rrdb))])
        self.trunk_conv = _make_conv(channels, channels)

        up_layers: list[nn.Module] = []
        s = self.scale
        while s > 1:
            up_layers.append(_make_conv(channels, channels * 4))
            up_layers.append(nn.PixelShuffle(2))
            up_layers.append(nn.LeakyReLU(0.2, inplace=True))
            s //= 2
        self.up = nn.Sequential(*up_layers)

        self.conv_hr = _make_conv(channels, channels)
        self.conv_last = _make_conv(channels, 3)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1, 1]
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.rrdb_trunk(feat))
        feat = feat + trunk
        out = self.up(feat)
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out
