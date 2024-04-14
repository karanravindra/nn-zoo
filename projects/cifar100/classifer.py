import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import Accuracy


class Block(nn.Module):
    def __init__(self, input_channels, out_channels, stride=1, batch_norm=True):
        super(Block, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(out_channels),
        )

        if input_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                input_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(3),
            Block(3, 64),
            Block(64, 64),
            Block(64, 64),
            Block(64, 64),
            nn.MaxPool2d(2),
            # 32x32 -> 16x16
            Block(64, 128),
            Block(128, 128),
            Block(128, 128),
            Block(128, 128),
            nn.MaxPool2d(2),
            # 16x16 -> 8x8
            Block(128, 256),
            Block(256, 256),
            Block(256, 256),
            Block(256, 256),
            nn.MaxPool2d(2),
            # 8x8 -> 4x4
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 100),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LitClassifer(pl.LightningModule):
    def __init__(self, lr=1e-3, batch_size=64):
        super(LitClassifer, self).__init__()
        self.model = Model()
        self.lr = lr
        self.batch_size = batch_size

        self.criterion = F.cross_entropy

        self.accuracy = Accuracy(task="multiclass", num_classes=100)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(
                root="data",
                train=True,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(
                root="data",
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(
                root="data",
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )


if __name__ == "__main__":
    model = LitClassifer()
    logger = WandbLogger(
        project="cifar100", save_dir="logs", log_model=True, save_code=True
    )
    logger.watch(model.model, log="all", log_freq=10, log_graph=True)
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="logs",
        val_check_interval=0.5,
        max_steps=10_000,
    )
    trainer.fit(model)
