import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger


class LitClassifer(pl.LightningModule):
    """PyTorch Lightning module for training a simple MLP on QMNIST dataset."""

    def __init__(self, lr=1e-4, batch_size=64, model=None):
        super(LitClassifer, self).__init__()
        assert model is not None, "model is required"
        self.model = model
        self.lr = lr
        self.batch_size = batch_size

        self.criterion = F.cross_entropy
        
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()

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
        
    def test_dataloader(self):
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
    lit_model = LitClassifer()
    logger = WandbLogger(
        project="qmnist", save_dir="qmnist/logs", log_model=True, save_code=True
    )
    logger.watch(lit_model, log="all", log_freq=100, log_graph=True)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=20,
        default_root_dir="qmnist/logs",
        val_check_interval=0.5,
    )
    trainer.fit(lit_model)
