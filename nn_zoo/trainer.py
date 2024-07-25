from typing import Any

import torch
import tqdm

from model import Model
from logger import WandbLogger
from datamodules.datamodule import DataModule
from _utils import get_device, get_optimizer, get_scheduler


class Trainer:
    def __init__(
        self,
        model: Model,
        dm: DataModule,
        num_epochs: int,
        log: bool = False,
        optim: str = "adam",
        optim_args: dict[str, Any] = {},
        sch: str | None = None,
        sch_args: dict[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        assert (sch is None) == (
            sch_args is None
        ), "Scheduler and its arguments must be both None or not None"

        self.model = model
        self.dm = dm
        self.num_epochs = num_epochs
        self.device = get_device(device)
        self.optimizer = get_optimizer(optim)(self.model.parameters(), **optim_args)
        self.scheduler = (
            get_scheduler(sch)(self.optimizer, **sch_args)
            if sch and sch_args is not None
            else None
        )
        self.logger = (
            WandbLogger(
                project_name="nn_zoo",
                run_name="default",
                config={"model": model.__class, "dm": dm.__class},
            )
            if log
            else None
        )

    def train(self) -> None:
        self.dm.prepare_data()
        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            pbar = tqdm.tqdm(train_loader, desc="Training", unit="batch", leave=False)
            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                loss = self.train_step(batch, batch_idx)
                loss.backward()
                self.optimizer.step()

            pbar.close()

            with torch.no_grad():
                val_loss = 0
                pbar = tqdm.tqdm(
                    val_loader, desc="Validating", unit="batch", leave=False
                )
                for batch_idx, batch in enumerate(pbar):
                    val_loss += self.validation_step(batch, batch_idx)
                val_loss /= len(val_loader)
                pbar.close()

            if self.scheduler is not None:
                self.scheduler.step()

    def test(self) -> None:
        test_loader = self.dm.test_dataloader()

        with torch.no_grad():
            pbar = tqdm.tqdm(test_loader, desc="Testing", unit="batch", leave=False)
            for batch_idx, batch in enumerate(pbar):
                self.test_step(batch, batch_idx)
            pbar.close()

    def predict(self) -> None:
        pass

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass

    def log(self, name: str, value: Any) -> None:
        if self.logger is not None:
            self.logger.log({name: value})

    def train_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch

        x_hat = self.model(x)

        return torch.nn.functional.cross_entropy(x_hat, y)

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch

        x_hat = self.model(x)

        return torch.nn.functional.cross_entropy(x_hat, y)

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch

        x_hat = self.model(x)

        return torch.nn.functional.cross_entropy(x_hat, y)

    def predict_step(self) -> torch.Tensor:
        raise NotImplementedError(
            f"predict_step method is not implemented in {self.__class__}"
        )


if __name__ == "__main__":
    from model import LeNet
    from datamodules.mnist import MNISTDataModule

    model = LeNet()
    dm = MNISTDataModule("data")
    trainer = Trainer(model, dm, num_epochs=2)

    trainer.train()
    trainer.test()
