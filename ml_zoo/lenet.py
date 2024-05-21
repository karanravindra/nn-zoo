from typing import Literal, List
import torch
import torch.nn as nn
from torchinfo import summary
from dataclasses import dataclass, field


@dataclass
class LeNetConfig:
    version: Literal[1, 4, 5] | None
    input_channels: int | None = 1
    num_classes: int = 10

    feature_dims: List[int] = field(default_factory=lambda: [1, 4, 12])
    kernel_sizes: List[int] = field(default_factory=lambda: [5, 5, 5])
    strides: List[int] = field(default_factory=lambda: [1, 1, 1])
    paddings: List[int] = field(default_factory=lambda: [0, 0, 0])
    pooling_sizes: List[int] = field(default_factory=lambda: [2, 2, 2])
    poolings: List[nn.Module] = field(
        default_factory=lambda: [nn.AvgPool2d, nn.AvgPool2d, nn.AvgPool2d]
    )
    activation: nn.Module = nn.Tanh()
    dropouts: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    vectors: List[int] = field(default_factory=lambda: [192, 10])

    def __post_init__(self):
        if self.version == 1 and self.input_channels is not None:
            self.feature_dims = [self.input_channels, 4, 12]
            self.vectors = [192, self.num_classes]

        if self.version == 4 and self.input_channels is not None:
            self.feature_dims = [self.input_channels, 6, 12]
            self.vectors = [12 * 5 * 5, 120, self.num_classes]

        if self.version == 5 and self.input_channels is not None:
            self.feature_dims = [self.input_channels, 6, 16]
            self.vectors = [16 * 5 * 5, 120, 84, self.num_classes]


class LeNet(torch.nn.Module):
    def __init__(self, config: LeNetConfig):
        super().__init__()
        self.config = config

        features = self._make_feature_layers()
        classifier = self.make_classifier_layers()

        # Combine the feature layers and classifier layers and name the layers
        layers = features + classifier
        self.layers = nn.Sequential(*layers)            

    def _make_feature_layers(self):
        feature_layers = []
        for i, (in_channels, out_channels) in enumerate(
            zip(self.config.feature_dims[:-1], self.config.feature_dims[1:])
        ):
            feature_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    self.config.kernel_sizes[i],
                    self.config.strides[i],
                    self.config.paddings[i],
                )
            )
            feature_layers.append(self.config.activation)
            feature_layers.append(self.config.poolings[i](self.config.pooling_sizes[i]))
        return nn.ModuleList(feature_layers)

    def make_classifier_layers(self):
        classifier_layers = []
        classifier_layers.append(nn.Flatten())
        for in_features, out_features, dropout in zip(
            self.config.vectors[:-1], self.config.vectors[1:], self.config.dropouts
        ):
            if dropout > 0:
                classifier_layers.append(nn.Dropout(dropout))

            classifier_layers.append(nn.Linear(in_features, out_features))
            classifier_layers.append(self.config.activation)
        return nn.ModuleList(classifier_layers[:-1])

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    config = LeNetConfig(
                version=None,
                feature_dims=[3, 32, 64, 128],
                kernel_sizes=[3, 3, 3, 3],
                paddings=[1, 1, 1, 1],
                vectors=[128 * 4 * 4, 240, 168, 10],
                dropouts=[0.5, 0., 0., 0., 0.],
                activation=nn.ReLU(),
                poolings=[nn.MaxPool2d, nn.MaxPool2d, nn.MaxPool2d, nn.MaxPool2d],
            )
    model = LeNet(config)
    print(model)
    summary(model, input_size=(64, 3, 32, 32))
