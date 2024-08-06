import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from collections.abc import Callable

__all__ = ["get_act", "get_norm", "RMSNorm1d", "RMSNorm2d"]


def get_act(
    act: Literal[
        "relu",
        "leaky_relu",
        "tanh",
        "sigmoid",
        "prelu",
        "silu",
        "gelu",
        "softmax",
        "identity",
    ]
    | Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the activation function based on the string name.

    You can use this function to get the activation function based on the string name or pass in a desired function directly.

    Args:
        act (str): The name of the activation function.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: The activation function.
    """
    if callable(act):
        return act

    return {
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "tanh": F.tanh,
        "sigmoid": F.sigmoid,
        "prelu": F.prelu,
        "silu": F.silu,
        "gelu": F.gelu,
        "softmax": F.softmax,
        "identity": nn.Identity,
    }[act]


def get_norm(
    norm: Literal[
        "batch2d",
        "instance2d",
        "rms1d",
        "rms2d",
        "layer",
        "group",
        "identity",
    ],
) -> type[nn.Module]:
    """Return the normalization layer based on the string name.

    You can use this function to get the normalization layer based on the string name.

    Args:
        norm (str): The name of the normalization layer.

    Returns:
        type[nn.Module]: The normalization layer.
    """

    return {
        "batch2d": nn.BatchNorm2d,
        "instance2d": nn.InstanceNorm2d,
        "rms1d": RMSNorm1d,
        "rms2d": RMSNorm2d,
        "layer": nn.LayerNorm,
        "group": nn.GroupNorm,
        "identity": nn.Identity,
    }[norm]


class RMSNorm1d(nn.Module):
    """
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/pdf/1910.07467.pdf.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale


class RMSNorm2d(nn.Module):
    """
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/pdf/1910.07467.pdf.

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale
