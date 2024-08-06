import torch
from nn_zoo.models.components.vq import VectorQuantizer


def test_vq():
    B, C, H, W = 3, 64, 32, 32
    x = torch.randn(B, C, H, W)

    vq = VectorQuantizer(C, 256, use_ema=True, decay=0.99, epsilon=1e-5)
    output = vq.forward(x)

    assert output[0].shape == (B, C, H, W)
    assert output[1] is None
    assert output[2].shape == ()  # Scalar tensor.
    assert output[3].shape == (B, H, W)
