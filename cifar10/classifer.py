import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import Accuracy, Precision, Recall


class LitClassifer(pl.LightningModule):
    def __init__(self, lr=1e-4, batch_size=64):
        super(LitClassifer, self).__init__()
        self.model = Classifier()
        self.lr = lr
        self.batch_size = batch_size

        self.criterion = F.cross_entropy

        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.precision = Precision(task="multiclass", num_classes=10)
        self.recall = Recall(task="multiclass", num_classes=10)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        prec = self.precision(y_hat, y)
        rec = self.recall(y_hat, y)

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True)
        self.log("train_prec", prec, on_step=True)
        self.log("train_rec", rec, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        prec = self.precision(y_hat, y)
        rec = self.recall(y_hat, y)

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_prec", prec)
        self.log("val_rec", rec)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root="cifar10/data", train=True, download=True)
        torchvision.datasets.CIFAR10(root="cifar10/data", train=False, download=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root="cifar10/data",
                train=True,
                transform=torchvision.transforms.ToTensor(),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root="cifar10/data",
                train=False,
                transform=torchvision.transforms.ToTensor(),
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )


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

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # VGG 11-like
        self.layers = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            BasicBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(512, 512),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout1d(0.5),
            nn.BatchNorm1d(512 * 4),
            nn.Linear(512 * 4, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model = LitClassifer()
    logger = WandbLogger(
        project="cifar10", save_dir="cifar10/logs", log_model=True, save_code=True
    )
    logger.watch(model, log="all")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=20,
        default_root_dir="cifar10/logs",
        val_check_interval=0.5,
    )
    trainer.fit(model)
