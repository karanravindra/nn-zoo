import torch
from nn_zoo.models.components import SelfAttention


def test_causal_self_attention():
    B, T, C = 3, 32, 64
    x = torch.randn(B, T, C)

    attn = SelfAttention(C, 8)
    output = attn(x)

    assert output.shape == x.shape


def test_non_causal_self_attention():
    B, T, C = 3, 32, 64
    x = torch.randn(B, T, C)

    attn = SelfAttention(C, 8)
    output = attn(x)

    assert output.shape == x.shape
