from typing import Callable
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["ResidualBasicBlock", "ResidualBottleNeckBlock", "ResidualStack"]


class ResidualBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        act: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        norm_layer: type[nn.Module] | None = nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

        self.norm1 = (
            norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        )
        self.norm2 = (
            norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        )
        self.act = act

        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + self.skip(residual)
        x = self.act(x)
        return x


class ResidualBottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        act: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        norm_layer: type[nn.Module] | None = nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

        self.norm1 = (
            norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        )
        self.norm2 = (
            norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        )
        self.norm3 = (
            norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        )
        self.act = act

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x += self.skip(residual)
        x = self.act(x)
        return x


class ResidualStack(nn.Sequential):
    def __init__(
        self,
        n_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        act: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        norm_layer: type[nn.Module] | None = nn.BatchNorm2d,
        block: type[nn.Module] = ResidualBasicBlock,
    ):
        super().__init__(
            *[
                block(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    act,
                    norm_layer,
                )
                for i in range(n_blocks)
            ]
        )
