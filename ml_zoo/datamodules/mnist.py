from typing import Any
from ml_zoo.datamodules._default import DataModule
from torchvision import datasets, transforms


class MNISTDataModule(DataModule):
    def __init__(self, data_dir: str, use_qmnist=False, **kwargs: Any):
        super().__init__(
            data_dir,
            dataset_params={
                "root": data_dir,
                "download": True,
                "transform": transforms.ToTensor(),
            },
            loader_params={"batch_size": 32, "num_workers": 4},
            **kwargs,
        )
        self.use_qmnist = use_qmnist
        self.source = datasets.QMNIST if self.use_qmnist else datasets.MNIST

    def __repr__(self) -> str:
        return (
            f"MNISTDataModule(data_dir={self.data_dir}, use_qmnist={self.use_qmnist})"
        )

    def config(self) -> dict[str, Any]:
        return dict(
            data_dir=self.data_dir,
            dataset_params=self.dataset_params,
            loader_params=self.loader_params,
        )

    def prepare_data(self):
        self.source = datasets.QMNIST if self.use_qmnist else datasets.MNIST

    def setup(self):
        params = self.dataset_params.copy()
        params.__delitem__("transform")
        self.train_dataset = self.source(
            train=True,
            **params,
            transform=transforms.Compose(
                [
                    self.dataset_params["transform"],
                ]
            ),
        )
        self.val_dataset = self.source(
            train=False,
            **params,
            transform=transforms.Compose(
                [
                    self.dataset_params["transform"],
                ]
            ),
        )
        self.test_dataset = self.source(
            train=False,
            **params,
            transform=transforms.Compose(
                [
                    self.dataset_params["transform"],
                ]
            ),
        )


if __name__ == "__main__":
    data_module = MNISTDataModule("data")
    data_module.prepare_data()
    data_module.setup()
    print(f"Train: {len(data_module.train_dataset):,}")
    print(f"Val: {len(data_module.val_dataset):,}")
    print(f"Test: {len(data_module.test_dataset):,}")
    print(data_module)
