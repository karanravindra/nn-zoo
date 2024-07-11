import torch
from serif_ml.models import UNet, UNetConfig


def test_unet():
    config = UNetConfig(
        in_channels=3,
        out_channels=3,
        initial_filters=32,
    )
    model = UNet(config)
    assert model

    x = torch.rand(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, 3, 32, 32)
