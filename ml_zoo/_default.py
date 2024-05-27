from dataclasses import dataclass
import pretty_errors
from torch import nn, Tensor
from lightning.pytorch import LightningDataModule
from torchvision.transforms import Compose, ToTensor

__all__ = [
    "DefaultDataModuleConfig",
    "DefaultDataModule",
    "DefaultModelConfig",
    "DefaultModel",
]


@dataclass
class DefaultDataModuleConfig:
    """The configuration for the default data module.

    Args:
        data_dir (str): The directory where the data is stored.
        batch_size (int): The batch size for the data loader.
        num_workers (int): The number of workers for the data loader.
        persistent_workers (bool): The number of workers for the data loader.
        pin_memory (bool, optional): Whether the pin memory. Should be `True`
            if you have a GPU. Defaults to `False`.
        transforms (Compose, optional): The transformation to use when
            loading the dataset. Defaults to `Compose([ToTensor()])`.
    """

    data_dir: str
    batch_size: int
    num_workers: int
    persistent_workers: bool
    pin_memory: bool = False
    transforms: Compose = Compose([ToTensor()])


class DefaultDataModule(LightningDataModule):
    """The default data module for PyTorch Lightning."""

    def __init__(self, config: DefaultDataModuleConfig):
        super().__init__()
        self.config = config

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


@dataclass
class DefaultModelConfig:
    """The configuration for the default model."""

    def __post_init__(self):
        pass

    def get_model(self) -> nn.Module:
        """Returns the model.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            nn.Module: The model.
        """
        raise NotImplementedError


class DefaultModel(nn.Module):
    """The default model for PyTorch."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError

    def summary(self) -> None:
        """Prints a summary of the model.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError


if __name__ == "__main__":
    dmc = DefaultModelConfig()
    model = DefaultModel()
    print(model(1))
