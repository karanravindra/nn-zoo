import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.classification import Accuracy
import wandb
from dataclasses import dataclass
from typing import Literal

from ._default import DefaultDataModule


@dataclass
class ClassifierConfig:
    model: nn.Module
    dm: DefaultDataModule
    optim: Literal["SGD", "Adam", "AdamW"] | torch.optim.Optimizer
    optim_args: dict
    _log_test_table: bool = False

    def __post_init__(self):
        assert isinstance(self.model, nn.Module)
        assert isinstance(self.dm, pl.LightningDataModule)
        assert self.optim in ["SGD", "Adam", "AdamW"] or isinstance(
            self.optim, torch.optim.Optimizer
        )

        if isinstance(self.optim, str):
            self._optim = getattr(torch.optim, self.optim)


class Classifier(pl.LightningModule):
    """Lightning module for training a classifier"""

    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config
        self.model = config.model

        self.criterion = F.cross_entropy

        self.accuracy = Accuracy(
            task="multiclass", num_classes=self.config.dm.num_classes
        )

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        config = {
            "model": self.model.__class__.__name__,
            "model_config": {
                **{
                    k: v
                    for k, v in self.config.model.__dict__.items()
                    if not k.startswith("_")
                },
                "num_classes": self.config.dm.num_classes,
                "num_params": sum(p.numel() for p in self.model.parameters()),
            },
            "optim": self.config.optim,
            "optim_args": self.config.optim_args,
            "dm": self.config.dm.__str__(),
            "dm_config": {
                **{
                    k: v
                    for k, v in self.config.dm.config.__dict__.items()
                    if not k.startswith("_") and not k == "transforms"
                },
                "transforms": {
                    t.__class__.__name__: t
                    for t in self.config.dm.config.transforms.transforms
                    if not t.__class__.__name__ == "ToTensor"
                },
            },
        }

        self.logger.experiment.config.update(config)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log("val/loss", loss)
        self.log("val/acc", acc)

        return loss

    def on_test_start(self):
        if self.config._log_test_table:
            self.testing_table = wandb.Table(
                columns=["y_true", "y_pred", "probs"]
            )

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log("test/loss", loss)
        self.log("test/acc", acc)

        if self.config._log_test_table:
            # Add row to testing table if wrong
            y_pred = torch.softmax(y_hat, dim=1)
            y_pred = torch.argmax(y_hat, dim=1)
            wrong = y != y_pred
            for i in range(x.shape[0]):
                if wrong[i]:
                    self.testing_table.add_data(
                        y[i].item(),
                        y_pred[i].item(),
                        y_hat[i].cpu().detach().numpy(),
                    )

        return loss

    def on_test_end(self):
        if self.config._log_test_table:
            self.logger.experiment.log({"test/table": self.testing_table})

    def configure_optimizers(self):
        optimizer = self.config._optim(self.parameters(), **self.config.optim_args)
        return optimizer

    def prepare_data(self):
        self.config.dm.prepare_data()
        self.config.dm.setup("fit")

    def train_dataloader(self):
        return self.config.dm.train_dataloader()

    def val_dataloader(self):
        return self.config.dm.val_dataloader()

    def test_dataloader(self):
        return self.config.dm.test_dataloader()
