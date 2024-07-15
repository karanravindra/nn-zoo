import torch
from torch import nn

__all__ = ["CNN"]


class CNN(nn.Sequential):
    def __init__(self, backbone: nn.Module, classifer: nn.Module):
        super().__init__(backbone, classifer)

    def loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
