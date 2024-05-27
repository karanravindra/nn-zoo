from dataclasses import dataclass
import lightning.pytorch as pl
from torchvision.transforms import Compose, ToTensor

__all__ = ["DefaultDataModuleConfig", "DefaultDataModule"]


@dataclass
class DefaultDataModuleConfig:
    data_dir: str
    batch_size: int
    num_workers: int
    persistent_workers: bool
    pin_memory: bool = False
    transforms: Compose = Compose([ToTensor()])


class DefaultDataModule(pl.LightningDataModule):
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

    @property
    def num_classes(self) -> int:
        raise NotImplementedError


if __name__ == "__main__":
    dm = DefaultDataModule(
        DefaultDataModuleConfig(
            data_dir=2, batch_size=32, num_workers=4, persistent_workers=True
        )
    )
