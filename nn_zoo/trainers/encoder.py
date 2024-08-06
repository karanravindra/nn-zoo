import torch
import torch.nn as nn
import torchvision
from lightning import LightningDataModule
from lightning.pytorch import LightningModule
import torch.optim.optimizer
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure as ssim,
)
import wandb
from torchvision.transforms.v2 import ToPILImage
from nn_zoo.trainers.utils import get_optim, get_scheduler

__all__ = ["AutoEncoderTrainer"]


class AutoEncoderTrainer(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        dm: LightningDataModule,
        optim: str,
        optim_kwargs: dict,
        scheduler: str | None = None,
        scheduler_args: dict | None = None,
    ):
        super(AutoEncoderTrainer, self).__init__()
        self.model = model
        self.dm = dm
        self.optim = get_optim(optim)
        self.optim_kwargs = optim_kwargs
        self.scheduler = get_scheduler(scheduler) if scheduler else None
        self.scheduler_kwargs = scheduler_args if scheduler else None

    def forward(self, x):
        return self.model(x).output

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        preds = self.model(x)
        metrics = self.model.loss(preds, x)

        for key, value in metrics.items():
            self.log(f"train/{key}", value)

        return metrics["loss"]

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        preds = self.model(x)
        metrics = self.model.loss(preds, x)

        for key, value in metrics.items():
            self.log(f"val/{key}", value)

        if batch_idx == 0:
            self.logger.experiment.log(
                {
                    "val/imgs/recon_imgs": wandb.Image(
                        ToPILImage()(torchvision.utils.make_grid(preds).cpu()),
                        caption="Predictions",
                    ),
                }
            )

            if self.current_epoch == 0:
                self.logger.experiment.log(
                    {
                        "val/imgs/original_imgs": wandb.Image(
                            ToPILImage()(torchvision.utils.make_grid(x).cpu()),
                            caption="Original Images",
                        ),
                    }
                )

        return metrics["loss"]

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        preds = self.model(x)
        metrics = self.model.loss(preds, x)

        for key, value in metrics.items():
            self.log(f"test/{key}", value)

        if batch_idx == 0:
            self.logger.experiment.log(
                {
                    "test/imgs/recon_imgs": wandb.Image(
                        ToPILImage()(torchvision.utils.make_grid(preds).cpu()),
                        caption="Predictions",
                    ),
                }
            )

            self.logger.experiment.log(
                {
                    "test/imgs/negative": wandb.Image(
                        ToPILImage()(torchvision.utils.make_grid(x - preds).cpu()),
                        caption="Negative",
                    ),
                }
            )

        return metrics["loss"]

    def configure_optimizers(self):
        optimizer = self.optim(self.model.parameters(), **self.optim_kwargs)
        scheduler = (
            self.scheduler(optimizer, **self.scheduler_kwargs)  # type: ignore
            if self.scheduler
            else None
        )

        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer

    def prepare_data(self) -> None:
        self.dm.prepare_data()

    def setup(self, stage):
        if stage == "fit":
            self.dm.setup("fit")
        elif stage == "test":
            self.dm.setup("test")

    def train_dataloader(self):
        return self.dm.train_dataloader()

    def val_dataloader(self):
        return self.dm.val_dataloader()

    def test_dataloader(self):
        return self.dm.test_dataloader()
