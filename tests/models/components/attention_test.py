import torch
from nn_zoo.models.components import SelfAttention


def test_self_attention():
    x = torch.randn(1, 32, 64)
    attn = SelfAttention(64, 8)
    assert attn(x).shape == (1, 32, 64)


if __name__ == "__main__":
    test_self_attention()
