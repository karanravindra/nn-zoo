from torch.utils.data import DataLoader
from lightning import LightningDataModule
from torchvision import datasets, transforms


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset_params: dict = {
            "download": True,
            "transform": transforms.ToTensor(),
        },
        loader_params: dict = {
            "batch_size": 64,
            "num_workers": 2,
        },
        qmnist: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_params = dataset_params
        self.loader_params = loader_params
        self.source = datasets.QMNIST if qmnist else datasets.MNIST

        self.transforms = transforms.Compose([self.dataset_params["transform"]])
        self.dataset_params.__delitem__("transform")

    def __repr__(self) -> str:
        return f"{self.source.__name__}DataModule({self.data_dir})"

    def config(self) -> dict:
        return dict(
            data_dir=self.data_dir,
            dataset_params=self.dataset_params,
            loader_params=self.loader_params,
            qmnist=self.source == datasets.QMNIST,
        )

    def prepare_data(self):
        self.source(self.data_dir, train=True, download=True)
        self.source(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = self.source(
            self.data_dir, train=True, **self.dataset_params, transform=self.transforms
        )
        self.val_dataset = self.source(
            self.data_dir, train=False, **self.dataset_params, transform=self.transforms
        )
        self.test_dataset = self.source(
            self.data_dir, train=False, **self.dataset_params, transform=self.transforms
        )

    def train_dataloader(self):
        assert self.train_dataset is not None, f"{self.__class__} not setup properly"
        return DataLoader(self.train_dataset, **self.loader_params)

    def val_dataloader(self):
        assert self.val_dataset is not None, f"{self.__class__} not setup properly"
        return DataLoader(self.val_dataset, **self.loader_params)

    def test_dataloader(self):
        assert self.test_dataset is not None, f"{self.__class__} not setup properly"
        return DataLoader(self.test_dataset, **self.loader_params)

    def num_classes(self):
        return 10


if __name__ == "__main__":
    data_module = MNISTDataModule("data", qmnist=False)
    data_module.prepare_data()
    data_module.setup()
    train = data_module.train_dataloader()
    val = data_module.val_dataloader()
    test = data_module.test_dataloader()
    print(f"Train: {len(data_module.train_dataset):,}")
    print(f"Val: {len(data_module.val_dataset):,}")
    print(f"Test: {len(data_module.test_dataset):,}")
    print(data_module)
