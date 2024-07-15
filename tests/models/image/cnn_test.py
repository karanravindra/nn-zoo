import torch
from torch import nn
from torch.nn import functional as F
from nn_zoo.models.image import CNN


def test_cnn():
    model = CNN(
        backbone=nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU(), nn.Flatten()),
        classifer=nn.LazyLinear(10),
    )
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 10)

    model.loss = lambda x, y: F.cross_entropy(model(x), y)
    assert model.loss(torch.randn(1, 3, 32, 32), torch.randint(0, 10, (1,))).shape == ()
