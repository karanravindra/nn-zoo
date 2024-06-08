from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from dataclasses import dataclass

__all__ = ["ResNetConfig", "ResNet", "BasicBlock", "Bottleneck"]


@dataclass
class ResNetConfig:
    version: Literal[18, 34, 50, 101, 152] | None = 18
    sample_size: Tuple[int, int, int] = (3, 224, 224)
    num_classes: int = 1000

    block: Literal["basic", "bottleneck"] = "basic"
    blocks: Tuple[int, int, int, int] = (2, 2, 2, 2)
    block_channels: Tuple[int, int, int, int] = (64, 128, 256, 512)
    inplanes: int = 64

    dropouts: float = 0.0

    pooling: Literal["max", "avg"] = "max"
    activation: nn.Module = nn.ReLU()

    def __post_init__(self):
        if self.version == 18:
            self.blocks = (2, 2, 2, 2)
            self.block_channels = (64, 128, 256, 512)

        if self.version == 34:
            self.blocks = (3, 4, 6, 3)
            self.block_channels = (64, 128, 256, 512)

        if self.version == 50:
            self.blocks = (3, 4, 6, 3)
            self.block_channels = (64, 128, 256, 512)

        if self.version == 101:
            self.blocks = (3, 4, 23, 3)
            self.block_channels = (64, 128, 256, 512)

        if self.version == 152:
            self.blocks = (3, 8, 36, 3)
            self.block_channels = (64, 128, 256, 512)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.config = config

        self.inplanes = config.inplanes
        self.conv1 = nn.Conv2d(
            config.sample_size[0],
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            config.block, config.block_channels[0], config.blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            config.block, config.block_channels[1], config.blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            config.block, config.block_channels[2], config.blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            config.block, config.block_channels[3], config.blocks[3], stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            config.block_channels[3]
            * (
                BasicBlock.expansion
                if config.block == "basic"
                else Bottleneck.expansion
            ),
            config.num_classes,
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        if block == "basic":
            block = BasicBlock
        else:
            block = Bottleneck

        for _ in range(blocks):
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
            stride = 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    config = ResNetConfig()
    model = ResNet(config)
    summary(model, input_size=(1, 3, 32, 32))
