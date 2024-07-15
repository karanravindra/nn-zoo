import torch
from nn_zoo.models.components.depthwise import *


def test_depthwise_conv1d():
    T, C = 32, 64
    x = torch.randn(1, C, T)

    conv = DepthwiseConv1d(C, 3)
    output = conv(x)
    print(output.shape)

    assert output.shape == x.shape


def test_depthwise_conv1d_batch():
    B, T, C = 3, 32, 64
    x = torch.randn(B, C, T)

    conv = DepthwiseConv1d(C, 3)
    assert conv(x).shape == x.shape


def test_depthwise_conv2d():
    H, W, C = 32, 32, 64
    x = torch.randn(1, C, H, W)

    conv = DepthwiseConv2d(C, 3)

    assert conv(x).shape == x.shape


def test_depthwise_conv2d_batch():
    B, H, W, C = 3, 32, 32, 64
    x = torch.randn(B, C, H, W)

    conv = DepthwiseConv2d(C, 3)

    assert conv(x).shape == x.shape


def test_depthwise_conv3d():
    D, H, W, C = 32, 32, 32, 64
    x = torch.randn(1, C, D, H, W)

    conv = DepthwiseConv3d(C, 3)

    assert conv(x).shape == x.shape


def test_depthwise_conv3d_batch():
    B, D, H, W, C = 3, 32, 32, 32, 64
    x = torch.randn(B, C, D, H, W)

    conv = DepthwiseConv3d(C, 3)

    assert conv(x).shape == x.shape


def test_depthwise_seperable_conv1d():
    T, C = 32, 64
    x = torch.randn(1, C, T)

    conv = DepthwiseSeparabl, 3)Conv1d(C, 3, 3)

    assert conv(x).shape == x.shape


def test_depthwise_seperable_conv1d_batch():
    B, T, C = 3, 32, 64
    x = torch.randn(B, C, T)

    conv = DepthwiseSeparableConv1d(C, 3, 3)

    assert conv(x).shape == x.shape


def test_depthwise_seperable_conv2d():
    H, W, C = 32, 32, 64
    x = torch.randn(1, C, H, W)

    conv = DepthwiseSeparableConv2d(C, 3, 3)

    assert conv(x).shape == x.shape


def test_depthwise_seperable_conv2d_batch():
    B, H, W, C = 3, 32, 32, 64
    x = torch.randn(B, C, H, W)

    conv = DepthwiseSeparableConv2d(C, 3, 3)

    assert conv(x).shape == x.shape


def test_depthwise_seperable_conv3d():
    D, H, W, C = 32, 32, 32, 64
    x = torch.randn(1, C, D, H, W)

    conv = DepthwiseSeparableConv3d(C, 3, 3)

    assert conv(x).shape == x.shape


def test_depthwise_seperable_conv3d_batch():
    B, D, H, W, C = 3, 32, 32, 32, 64
    x = torch.randn(B, C, D, H, W)

    conv = DepthwiseSeparableConv3d(C, 3, 3)

    assert conv(x).shape == x.shape
