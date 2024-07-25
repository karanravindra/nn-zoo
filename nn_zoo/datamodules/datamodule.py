from torch.utils.data import DataLoader, Dataset


class DataModule:
    data_dir: str
    source: type[Dataset]

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data_dir='{self.data_dir})"

    def prepare_data(self) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the `prepare_data` method."
        )

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the `train_dataloader` method."
        )

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the `val_dataloader` method."
        )

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the `test_dataloader` method."
        )

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the `predict_dataloader` method."
        )
