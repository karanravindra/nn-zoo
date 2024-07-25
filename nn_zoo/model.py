import torch
from torch import nn


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

    def forward(self, x) -> None:
        pass


## Examples
class LeNet(Model):
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x) -> None:
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, (2, 2))
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = LeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)
