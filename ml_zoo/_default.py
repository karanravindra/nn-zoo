from torchvision.transforms import Compose
from dataclasses import dataclass
import lightning.pytorch as pl


@dataclass
class DefaultDataModuleConfig:
    data_dir: str
    transforms: Compose
    batch_size: int
    num_workers: int
    persistent_workers: bool
    pin_memory: bool

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
        