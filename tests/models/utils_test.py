from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from nn_zoo.models.utils import get_act, get_norm, RMSNorm1d, RMSNorm2d


@pytest.mark.parametrize(
    "act, expected",
    [
        ("relu", F.relu),
        ("leaky_relu", F.leaky_relu),
        ("tanh", F.tanh),
        ("sigmoid", F.sigmoid),
        ("prelu", F.prelu),
        ("silu", F.silu),
        ("gelu", F.gelu),
        ("softmax", F.softmax),
        ("identity", nn.Identity),
    ],
)
def test_get_act(act: str, expected: Callable[[torch.Tensor], torch.Tensor]):
    assert get_act(act) == expected  # type: ignore


@pytest.mark.parametrize(
    "norm, expected",
    [
        ("batch2d", nn.BatchNorm2d),
        ("instance2d", nn.InstanceNorm2d),
        ("rms1d", RMSNorm1d),
        ("rms2d", RMSNorm2d),
        ("layer", nn.LayerNorm),
        ("group", nn.GroupNorm),
        ("identity", nn.Identity),
    ],
)
def test_get_norm(norm: str, expected: Callable[[torch.Tensor], torch.Tensor]):
    assert get_norm(norm) == expected  # type: ignore


def test_rms1d_shape():
    ## 1D input
    x = torch.randn(3, 32)
    rms = RMSNorm1d(32)
    output = rms(x)

    assert output.shape == x.shape

    ## 2D input
    x = torch.randn(3, 32, 32)
    rms = RMSNorm1d(32)
    output = rms(x)

    assert output.shape == x.shape


def test_rms2d_shape():
    ## 2D input
    x = torch.randn(3, 32, 32)
    rms = RMSNorm2d(32)
    output = rms(x)

    assert output.shape == x.shape

    ## 3D input
    x = torch.randn(3, 32, 32, 32)
    rms = RMSNorm2d(32)
    output = rms(x)

    assert output.shape == x.shape
