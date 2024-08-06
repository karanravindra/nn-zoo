import torch
from nn_zoo.models.components.residual import (
    ResidualBasicBlock,
    ResidualBottleNeckBlock,
    ResidualStack,
)


def test_basic_residual_block():
    B, C, H, W = 3, 32, 32, 32
    x = torch.randn(B, C, H, W)

    block = ResidualBasicBlock(C, C, 3)
    assert block(x).shape == x.shape


def test_bottleneck_residual_block():
    B, C, H, W = 3, 32, 32, 32
    x = torch.randn(B, C, H, W)

    block = ResidualBottleNeckBlock(C, C, 3)
    assert block(x).shape == x.shape


def test_residual_stack():
    B, C, H, W = 3, 32, 32, 32
    x = torch.randn(B, C, H, W)

    stack = ResidualStack(3, C, C, 3)
    assert stack(x).shape == x.shape
