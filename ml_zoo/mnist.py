from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from ._default import DefaultDataModuleConfig, DefaultDataModule
from dataclasses import dataclass


__all__ = ["MNISTDataModuleConfig", "MNISTDataModule"]


@dataclass
class MNISTDataModuleConfig(DefaultDataModuleConfig):
    use_qmnist: bool


class MNISTDataModule(DefaultDataModule):
    def __init__(self, config: MNISTDataModuleConfig):
        super().__init__(config)
        self.config = config

        self.dataset = datasets.QMNIST if self.config.use_qmnist else datasets.MNIST

    def __str__(self) -> str:
        return "mnist" if not self.config.use_qmnist else "qmnist"

    @property
    def num_classes(self) -> int:
        return len(self.dataset.classes)

    def prepare_data(self):
        self.dataset(self.config.data_dir, train=True, download=True)
        self.dataset(self.config.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(
                self.config.data_dir, train=True, transform=self.config.transforms
            )
            self.val_dataset = self.dataset(
                self.config.data_dir, train=False, transform=self.config.transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )


if __name__ == "__main__":
    config = MNISTDataModuleConfig(
        data_dir="data",
        transforms=transforms.Compose([transforms.ToTensor()]),
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        use_qmnist=False,
    )
    datamodule = MNISTDataModule(config)
    datamodule.prepare_data()
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
