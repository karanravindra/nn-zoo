import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.classification import Accuracy
import torchvision
from dataclasses import dataclass
from typing import Literal

from ._default import DefaultDataModule


@dataclass
class ClassifierConfig:
    model: nn.Module
    dm: DefaultDataModule
    optim: Literal["SGD", "Adam", "AdamW"] | torch.optim.Optimizer
    optim_args: dict

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
                k: v
                for k, v in self.config.model.__dict__.items()
                if not k.startswith("_")
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

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log("test/loss", loss)
        self.log("test/acc", acc)

        return loss

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
