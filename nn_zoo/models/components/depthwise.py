"""Depthwise Convolution Modules from https://arxiv.org/abs/1610.02357"""

from torch import nn

__all__ = [
    "DepthwiseConv1d",
    "DepthwiseConv2d",
    "DepthwiseConv3d",
    "DepthwiseSeparableConv1d",
    "DepthwiseSeparableConv2d",
    "DepthwiseSeparableConv3d",
]


class DepthwiseConv1d(nn.Conv1d):
    """Depthwise Convolution for 1D data."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(DepthwiseConv1d, self).__init__(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )


class DepthwiseConv2d(nn.Conv2d):
    """Depthwise Convolution for 2D data."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(DepthwiseConv2d, self).__init__(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )


class DepthwiseConv3d(nn.Conv3d):
    """Depthwise Convolution for 3D data."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(DepthwiseConv3d, self).__init__(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )


class DepthwiseSeparableConv1d(nn.Sequential):
    """Depthwise Separable Convolution for 1D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super(DepthwiseSeparableConv1d, self).__init__(
            DepthwiseConv1d(
                in_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups=in_channels,
                bias=bias,
            ),
            nn.Conv1d(in_channels, out_channels, 1, bias=bias),
        )


class DepthwiseSeparableConv2d(nn.Sequential):
    """Depthwise Separable Convolution for 2D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super(DepthwiseSeparableConv2d, self).__init__(
            DepthwiseConv2d(
                in_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups=in_channels,
                bias=bias,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
        )


class DepthwiseSeparableConv3d(nn.Sequential):
    """Depthwise Separable Convolution for 3D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super(DepthwiseSeparableConv3d, self).__init__(
            DepthwiseConv3d(
                in_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups=in_channels,
                bias=bias,
            ),
            nn.Conv3d(in_channels, out_channels, 1, bias=bias),
        )
