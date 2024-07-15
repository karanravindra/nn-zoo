import torch
from nn_zoo.models.components.depthwise import *


def test_depthwise_conv1d():
    B, T, C = 3, 32, 64
    x = torch.randn(B, C, T)

    conv = DepthwiseConv1d(C, 3)
    assert conv(x).shape == x.shape


def test_depthwise_conv2d():
    B, H, W, C = 3, 32, 32, 64
    x = torch.randn(B, C, H, W)

    conv = DepthwiseConv2d(C, 3)

    assert conv(x).shape == x.shape


def test_depthwise_conv3d():
    B, D, H, W, C = 3, 32, 32, 32, 64
    x = torch.randn(B, C, D, H, W)

    conv = DepthwiseConv3d(C, 3)

    assert conv(x).shape == x.shape


def test_depthwise_seperable_conv1d():
    B, T, C = 3, 32, 64
    x = torch.randn(B, C, T)

    conv = DepthwiseSeparableConv1d(C, 3, 3)

    assert conv(x).shape == x.shape


def test_depthwise_seperable_conv2d():
    B, H, W, C = 3, 32, 32, 64
    x = torch.randn(B, C, H, W)

    conv = DepthwiseSeparableConv2d(C, 3, 3)

    assert conv(x).shape == x.shape


def test_depthwise_seperable_conv3d():
    B, D, H, W, C = 3, 32, 32, 32, 64
    x = torch.randn(B, C, D, H, W)

    conv = DepthwiseSeparableConv3d(C, 3, 3)

    assert conv(x).shape == x.shape
