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

__all__ = ["AutoEncoderTrainer"]


def get_optim(
    optim: str,
) -> type[torch.optim.SGD | torch.optim.Adam | torch.optim.AdamW]:
    match optim.lower():
        case "sgd":
            return torch.optim.SGD
        case "adam":
            return torch.optim.Adam
        case "adamw":
            return torch.optim.AdamW
        case _:
            raise NotImplementedError(
                f"The requested optimizer: {optim} is not availible"
            )


def get_scheduler(
    scheduler: str,
) -> type[
    torch.optim.lr_scheduler.StepLR
    | torch.optim.lr_scheduler.MultiStepLR
    | torch.optim.lr_scheduler.ExponentialLR
    | torch.optim.lr_scheduler.CosineAnnealingLR
]:
    match scheduler.lower():
        case "steplr":
            return torch.optim.lr_scheduler.StepLR
        case "multisteplr":
            return torch.optim.lr_scheduler.MultiStepLR
        case "exponentiallr":
            return torch.optim.lr_scheduler.ExponentialLR
        case "cosinelr":
            return torch.optim.lr_scheduler.CosineAnnealingLR
        case None:
            return None
        case _:
            raise NotImplementedError(
                f"The requested scheduler: {scheduler} is not availible"
            )


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

        self.log("train/loss", metrics['loss'])
        self.log("train/lpips", metrics['lpips'])
        self.log("train/psnr", psnr(preds, x))
        self.log("train/ssim", ssim(preds, x))

        return metrics['loss']

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        preds = self.model(x)
        metrics = self.model.loss(preds, x)

        self.log("val/loss", metrics['loss'])
        self.log("val/lpips", metrics['lpips'])
        self.log("val/psnr", psnr(preds, x))
        self.log("val/ssim", ssim(preds, x))

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

        return metrics['loss']

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        preds = self.model(x)
        metrics = self.model.loss(preds, x)

        self.log("test/loss", metrics['loss'])
        self.log("test/lpips", metrics['lpips'])
        self.log("test/psnr", psnr(preds, x))
        self.log("test/ssim", ssim(preds, x))

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

        return metrics['loss']

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
