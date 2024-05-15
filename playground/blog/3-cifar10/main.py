import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm as tqdm
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from models import *

class CIFAR10Trainer(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3) -> None:
        super(CIFAR10Trainer, self).__init__()
        self.model = model
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root="data", train=True, download=True, transform=torchvision.transforms.ToTensor()
            ),
            batch_size=32,
            shuffle=True,
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root="data", train=False, download=True, transform=torchvision.transforms.ToTensor()
            ),
            batch_size=32,
            shuffle=False,
        )
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root="data", train=True, download=True, transform=torchvision.transforms.ToTensor()
            ),
            batch_size=32,
            shuffle=True,
        )