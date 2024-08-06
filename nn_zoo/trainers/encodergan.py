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

__all__ = ["EncoderGANTrainer"]


class EncoderGANTrainer(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        disc: nn.Module,
        dm: LightningDataModule,
        optim_g: str,
        optim_kwargs_g: dict,
        optim_d: str,
        optim_kwargs_d: dict,
        scheduler_g: str | None = None,
        scheduler_args_g: dict | None = None,
        scheduler_d: str | None = None,
        scheduler_args_d: dict | None = None,
    ):
        super(EncoderGANTrainer, self).__init__()
        self.model = model
        self.disc = disc
        self.dm = dm
        self.optim_g = get_optim(optim_g)
        self.optim_kwargs_g = optim_kwargs_g
        self.optim_d = get_optim(optim_d)
        self.optim_kwargs_d = optim_kwargs_d
        self.scheduler_g = get_scheduler(scheduler_g) if scheduler_g else None
        self.scheduler_kwargs_g = scheduler_args_g if scheduler_g else None
        self.scheduler_d = get_scheduler(scheduler_d) if scheduler_d else None
        self.scheduler_kwargs_d = scheduler_args_d if scheduler_d else None

        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x).output

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        optim_g, optim_d = self.optimizers()

        # Train Discriminator
        optim_d.zero_grad()
        fake = self.model(x)
        real = x
        real_preds = self.disc(real)
        fake_preds = self.disc(fake)
        d_loss = self.disc.loss(real_preds, fake_preds)
        d_loss.backward()
        optim_d.step()

        # Train Generator
        optim_g.zero_grad()
        fake = self.model.generator(x)
        fake_preds = self.disc(fake)
        g_loss = self.model.generator_loss(fake_preds)
        g_loss.backward()
        optim_g.step()

        self.log("train/d_loss", d_loss)
        self.log("train/g_loss", g_loss)

        return d_loss + g_loss

        # preds = self.model(x)
        # metrics = self.model.loss(preds, x)

        # for key, value in metrics.items():
        #     self.log(f"train/{key}", value)

        # return metrics["loss"]

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
        # optimizer = self.optim(self.model.parameters(), **self.optim_kwargs)
        # scheduler = (
        #     self.scheduler(optimizer, **self.scheduler_kwargs)  # type: ignore
        #     if self.scheduler
        #     else None
        # )

        # if scheduler:
        #     return [optimizer], [scheduler]
        # else:
        #     return optimizer

        optim_g = self.optim_g(self.model.generator.parameters(), **self.optim_kwargs_g)
        optim_d = self.optim_d(
            self.model.discriminator.parameters(), **self.optim_kwargs_d
        )
        scheduler_g = (
            self.scheduler_g(optim_g, **self.scheduler_kwargs_g)  # type: ignore
            if self.scheduler_g
            else None
        )
        scheduler_d = (
            self.scheduler_d(optim_d, **self.scheduler_kwargs_d)  # type: ignore
            if self.scheduler_d
            else None
        )

        if scheduler_g and scheduler_d:
            return [optim_g, optim_d], [scheduler_g, scheduler_d]
        elif scheduler_g:
            return [optim_g, optim_d], [scheduler_g]
        elif scheduler_d:
            return [optim_g, optim_d], [scheduler_d]
        else:
            return [optim_g, optim_d]

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
