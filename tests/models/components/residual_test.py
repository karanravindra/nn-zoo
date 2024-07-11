import torch
from nn_zoo.models.components import (
    ResidualBasicBlock,
    ResidualBottleNeckBlock,
    ResidualStack,
)


def test_residual_basic_block():
    x = torch.randn(1, 3, 32, 32)
    block = ResidualBasicBlock(3, 64, 3)
    assert block(x).shape == (1, 64, 32, 32)


def test_residual_bottle_neck_block():
    x = torch.randn(1, 3, 32, 32)
    block = ResidualBottleNeckBlock(3, 64, 3)
    assert block(x).shape == (1, 64, 32, 32)


def test_residual_stack():
    x = torch.randn(1, 3, 32, 32)
    stack = ResidualStack(3, 3, 64, 3)
    assert stack(x).shape == (1, 64, 32, 32)


if __name__ == "__main__":
    test_residual_basic_block()
    test_residual_bottle_neck_block()
    test_residual_stack()
