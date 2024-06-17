from typing import Literal, List
from ._default import DefaultModel
import torch.nn as nn
from torchinfo import summary
from dataclasses import dataclass, field

__all__ = ["LeNetConfig", "LeNet"]


@dataclass
class LeNetConfig:
    
    sample_size: tuple[int, int, int]
    version: Literal[1, 4, 5] | None
    num_classes: int = 10

    feature_dims: List[int] = field(default_factory=lambda: [4, 12])
    kernel_sizes: List[int] = field(default_factory=lambda: [5, 5])
    poolings: Literal["MaxPool2d", "AvgPool2d"] = "MaxPool2d"
    activation: Literal["Tanh", "ReLU"] = "Tanh"

    vectors: List[int] = field(default_factory=lambda: [192, 10])
    dropouts: List[float] = field(default_factory=lambda: [0.0, 0.0])

    def __post_init__(self):

        if self.version == 1 and self.sample_size[0] is not None:
            self.feature_dims = [self.sample_size[0], 4, 12]
            self.vectors = [192, self.num_classes]

        if self.version == 4 and self.sample_size[0] is not None:
            self.feature_dims = [self.sample_size[0], 6, 12]
            self.vectors = [12 * 5 * 5, 120, self.num_classes]

        if self.version == 5 and self.sample_size[0] is not None:
            self.feature_dims = [self.sample_size[0], 6, 16]
            self.vectors = [16 * 5 * 5, 120, 84, self.num_classes]

            if len(self.dropouts) < len(self.vectors):
                self.dropouts.extend([0.0] * (len(self.vectors) - len(self.dropouts)))

        self.poolings = getattr(nn, self.poolings)
        self.activation = getattr(nn, self.activation)()


class LeNet(DefaultModel):
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
                )
            )
            feature_layers.append(self.config.activation)
            feature_layers.append(self.config.poolings(2))  # type: ignore
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

    def summary(self):
        return summary(self, input_size=(1, *self.config.sample_size))


if __name__ == "__main__":
    config = LeNetConfig(
        sample_size=(1, 28, 28),
        version=4,
        feature_dims=[6, 16],
        kernel_sizes=[3, 3, 3, 3],
        vectors=[128 * 4 * 4, 240, 168, 10],
        dropouts=[0.5, 0.0, 0.0, 0.0, 0.0],
        activation="ReLU",
        poolings="MaxPool2d",
    )
    model = LeNet(config)
    model.summary()
