import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import Accuracy


class BasicBlock(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class VeryDeepConvolutionalNN(nn.Module):
    """ResNet-18-Like Network
    Based off of https://arxiv.org/abs/1512.03385
    """

    def __init__(self) -> None:
        super(VeryDeepConvolutionalNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.layers(x)


class LitClassifer(pl.LightningModule):
    """PyTorch Lightning module for training a simple MLP on QMNIST dataset."""

    def __init__(self, lr=1e-4, batch_size=64):
        super(LitClassifer, self).__init__()
        self.model = VeryDeepConvolutionalNN()
        self.lr = lr
        self.batch_size = batch_size

        self.criterion = F.cross_entropy

        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

    def prepare_data(self):
        torchvision.datasets.QMNIST(root="qmnist/data", what="train", download=True)
        torchvision.datasets.QMNIST(root="qmnist/data", what="test", download=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.QMNIST(
                root="qmnist/data",
                what="train",
                transform=torchvision.transforms.ToTensor(),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.QMNIST(
                root="qmnist/data",
                what="test",
                transform=torchvision.transforms.ToTensor(),
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )


if __name__ == "__main__":
    model = LitClassifer()
    logger = WandbLogger(
        project="qmnist", save_dir="qmnist/logs", log_model=True, save_code=True
    )
    logger.watch(model, log="all")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=20,
        default_root_dir="qmnist/logs",
        val_check_interval=0.5,
    )
    trainer.fit(model)
