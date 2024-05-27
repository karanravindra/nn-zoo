from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from termcolor import colored
from ._default import DefaultDataModuleConfig, DefaultDataModule
from dataclasses import dataclass

__all__ = ["CelebAHQDataModuleConfig", "CelebAHQDataModule"]


@dataclass
class CelebAHQDataModuleConfig(DefaultDataModuleConfig):
    pass


class CelebAHQDataModule(DefaultDataModule):
    def __init__(self, config: CelebAHQDataModuleConfig):
        super().__init__(config)
        self.config = config

        self.dataset = datasets.ImageFolder

    @property
    def num_classes(self):
        return 2

    def prepare_data(self):
        print(colored("Preparing CelebA-HQ data...", "blue"))
        print(colored("Dataset download not supported yet.", "red"))
        print(colored("Please download the dataset manually.", "red"))

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(
                self.config.data_dir + "/celeba_hq/train",
                transform=self.config.transforms,
            )
            self.val_dataset = self.dataset(
                self.config.data_dir + "/celeba_hq/val",
                transform=self.config.transforms,
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
    dmc = CelebAHQDataModuleConfig(
        data_dir="data",
        transforms=transforms.Compose([transforms.ToTensor()]),
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    datamodule = CelebAHQDataModule(dmc)
    datamodule.prepare_data()
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
