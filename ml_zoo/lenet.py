from typing import Literal, List
import torch
import torch.nn as nn
from torchinfo import summary
from dataclasses import dataclass, field


@dataclass
class LeNetConfig:
    version: Literal[1, 4, 5] | None

    feature_dims: List[int] = field(default_factory=lambda: [1, 4, 12])
    kernel_sizes: List[int] = field(default_factory=lambda: [5, 5, 5])
    strides: List[int] = field(default_factory=lambda: [1, 1, 1])
    paddings: List[int] = field(default_factory=lambda: [0, 0, 0])
    pooling_sizes: List[int] = field(default_factory=lambda: [2, 2, 2])
    poolings: List[nn.Module] = field(
        default_factory=lambda: [nn.AvgPool2d, nn.AvgPool2d, nn.AvgPool2d]
    )
    activation: nn.Module = nn.Tanh()

    vectors: List[int] = field(default_factory=lambda: [192, 10])

    def __post_init__(self):
        if self.version == 1:
            self.feature_dims = [1, 4, 12]
            self.vectors = [192, 10]

        if self.version == 4:
            self.feature_dims = [1, 6, 12]
            self.vectors = [12 * 5 * 5, 120, 10]

        if self.version == 5:
            self.feature_dims = [1, 6, 16]
            self.vectors = [16 * 5 * 5, 120, 84, 10]


class LeNet(torch.nn.Module):
    def __init__(self, config: LeNetConfig):
        super().__init__()
        self.config = config

        feature_layers = []
        for i, (in_channels, out_channels) in enumerate(
            zip(config.feature_dims[:-1], config.feature_dims[1:])
        ):
            feature_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    config.kernel_sizes[i],
                    config.strides[i],
                    config.paddings[i],
                )
            )
            feature_layers.append(config.activation)
            feature_layers.append(config.poolings[i](config.pooling_sizes[i]))
        self.features = nn.Sequential(*feature_layers)

        classifier_layers = [nn.Flatten()]
        for in_features, out_features in zip(config.vectors[:-1], config.vectors[1:]):
            classifier_layers.append(nn.Linear(in_features, out_features))
            classifier_layers.append(config.activation)
        self.classifier = nn.Sequential(*classifier_layers[:-1])

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    config = LeNetConfig(
        version=4,
        paddings=[2, 0, 0],
    )
    model = LeNet(config)
    summary(model, input_size=(1, 1, 28, 28))
