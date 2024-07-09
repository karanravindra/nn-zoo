from typing import Any
from torch.utils.data import DataLoader


class DataModule:
    def __init__(
        self,
        data_dir: str,
        dataset_params: dict[str, Any],
        loader_params: dict[str, Any],
        **kwargs: Any,
    ):
        self.data_dir = data_dir
        self.dataset_params = dataset_params
        self.loader_params = loader_params

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def __repr__(self):
        return f"{self.__class__.__name__}(data_dir={self.data_dir})"

    def config(self):
        raise NotImplementedError

    def prepare_data(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def train_dataloader(self):
        assert self.train_dataset is not None, f"{self.__class__} not setup properly"
        return DataLoader(self.train_dataset, **self.loader_params)

    def val_dataloader(self):
        assert self.val_dataset is not None, f"{self.__class__} not setup properly"
        return DataLoader(self.val_dataset, **self.loader_params)

    def test_dataloader(self):
        assert self.test_dataset is not None, f"{self.__class__} not setup properly"
        return DataLoader(self.test_dataset, **self.loader_params)
