from dataclasses import dataclass
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from ._default import DefaultDataModuleConfig, DefaultDataModule

__all__ = ["CelebAHQDataModuleConfig", "CelebAHQDataModule"]


@dataclass
class CelebAHQDataModuleConfig(DefaultDataModuleConfig):
    pass


class CelebAHQDataModule(DefaultDataModule):
    def __init__(self, config: CelebAHQDataModuleConfig):
        super().__init__(config)
        self.config = config

        self.dataset = datasets.ImageFolder

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def get_num_classes(self):
        """Get the number of classes in the dataset.

        Returns:
            int: The number of classes in the dataset.
        """
        return 2

    def prepare_data(self):
        raise NotImplementedError

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
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

    def test_dataloader(self):
        assert self.val_dataset is not None
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
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        transforms=transforms.Compose([transforms.ToTensor()]),
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
