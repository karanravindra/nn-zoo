from typing import Literal, Tuple
import torch
import torch.nn as nn
from torchinfo import summary
from dataclasses import dataclass

__all__ = ["VGGConfig", "VGG"]


@dataclass
class VGGConfig:
    version: Literal[11, 13, 16, 19] | None = 11
    sample_size: Tuple[int, int, int] = (3, 224, 224)
    global_pooling_dim: int = 7
    vector_dim: int = 4096
    num_classes: int = 1000

    features: Tuple[int, int, int, int, int] = (64, 128, 256, 512, 512)
    num_features: Tuple[int, int, int, int, int] = (2, 2, 3, 3, 3)

    dropouts: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    vectors: Tuple[int, int, int] = (512 * 7 * 7, 4096, 4096)

    pooling: Literal["max", "avg"] = "max"
    activation: nn.Module = nn.ReLU()

    def __post_init__(self):
        if self.version == 11:
            self.features = (64, 128, 256, 512, 512)
            self.num_features = (1, 1, 2, 2, 2)
            self.vectors = (
                self.features[-1] * self.global_pooling_dim * self.global_pooling_dim,
                self.vector_dim,
                self.vector_dim,
            )

        if self.version == 13:
            self.features = (64, 128, 256, 512, 512)
            self.num_features = (2, 2, 2, 2, 2)
            self.vectors = (
                self.features[-1] * self.global_pooling_dim * self.global_pooling_dim,
                self.vector_dim,
                self.vector_dim,
            )

        if self.version == 16:
            self.features = (64, 128, 256, 512, 512)
            self.num_features = (2, 2, 3, 3, 3)
            self.vectors = (
                self.features[-1] * self.global_pooling_dim * self.global_pooling_dim,
                self.vector_dim,
                self.vector_dim,
            )

        if self.version == 19:
            self.features = (64, 128, 256, 512, 512)
            self.num_features = (2, 2, 4, 4, 4)
            self.vectors = (
                self.features[-1] * self.global_pooling_dim * self.global_pooling_dim,
                self.vector_dim,
                self.vector_dim,
            )


class VGG(nn.Module):
    def __init__(self, config: VGGConfig):
        super().__init__()
        self.config = config

        features = self._make_feature_layers()
        classifier = self._make_classifier_layers()

        # Combine the feature layers and classifier layers and name the layers
        layers = features + classifier
        self.layers = nn.Sequential(*layers)

    def _make_feature_layers(self):
        feature_layers = []
        in_channels = self.config.sample_size[0]
        for i, (out_channels, num_features) in enumerate(
            zip(self.config.features, self.config.num_features)
        ):
            for _ in range(num_features):
                feature_layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                )
                feature_layers.append(self.config.activation)
                in_channels = out_channels

            if self.config.pooling == "max":
                feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                feature_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        return feature_layers

    def _make_classifier_layers(self):
        classifier_layers = []
        classifier_layers.append(nn.Flatten())
        in_features = self.config.vectors[0]
        for in_features, out_features, dropout in zip(
            self.config.vectors, self.config.vectors[1:], self.config.dropouts
        ):
            classifier_layers.append(nn.Linear(in_features, out_features))
            classifier_layers.append(self.config.activation)
            if dropout > 0:
                classifier_layers.append(nn.Dropout(dropout))

        classifier_layers.append(nn.Linear(in_features, self.config.num_classes))
        return classifier_layers

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    config = VGGConfig(
        version=11,
        sample_size=(3, 32, 32),
        global_pooling_dim=1,
        vector_dim=512,
        num_classes=10,
    )
    model = VGG(config)
    summary(model, input_size=(1, *config.sample_size))
    print(model(torch.randn(1, *config.sample_size)).shape)
