import torch
import torch.nn as nn
from lightning import LightningDataModule
from lightning.pytorch import LightningModule
import torch.optim.optimizer
from torchmetrics.functional.classification import accuracy
import wandb
import wandb.plot

__all__ = ["ClassifierTrainer"]


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


class ClassifierTrainer(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        dm: LightningDataModule,
        optim: str,
        optim_kwargs: dict,
        scheduler: str | None = None,
        scheduler_args: dict | None = None,
    ):
        super(ClassifierTrainer, self).__init__()
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

        out = self.model(x, y)
        preds, loss = out

        self.log("train_loss", loss)
        self.log(
            "train_acc",
            accuracy(
                preds.softmax(1),
                y,
                task="multiclass",
                num_classes=self.dm.num_classes(),
            ),
        )

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        out = self.model(x, y)
        preds, loss = out

        self.log("val_loss", loss)
        self.log(
            "val_acc",
            accuracy(
                preds.softmax(1),
                y,
                task="multiclass",
                num_classes=self.dm.num_classes(),
            ),
        )

        return loss

    def on_test_start(self) -> None:
        # save preds and targets
        self.preds = []
        self.targets = []

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        out = self.model(x, y)
        preds, loss = out

        self.log("test_loss", loss)
        self.log(
            "test_acc",
            accuracy(
                preds.softmax(1),
                y,
                task="multiclass",
                num_classes=self.dm.num_classes(),
            ),
        )

        self.preds.append(preds.softmax(1))
        self.targets.append(y)

        return loss

    def on_test_end(self) -> None:
        out = torch.cat(self.preds)
        y = torch.cat(self.targets)

        self.log(
            "test_confusion_matrix",
            wandb.plot.confusion_matrix(
                preds=out.argmax(1).tolist(),
                y_true=y.tolist(),
                class_names=self.dm.class_names(),
                title="Confusion Matrix",
            ),
        )

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
