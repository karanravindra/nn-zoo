import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import tqdm as tqdm
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 512x512
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 256x256
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 2, 1),  # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
        )

        self.mu = nn.Linear(512, 256)
        self.logvar = nn.Linear(512, 256)

    def forward(self, x):
        x = self.layers(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 256, 3, 2, 1, 1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 512, 3, 2, 1, 1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),  # 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),  # 512x512
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class VAE(pl.LightningModule):
    def __init__(self, lr=2e-4, batch_size=32, fid_features=2048):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.fid = FrechetInceptionDistance(fid_features)

        self.lr = lr
        self.batch_size = batch_size

        self.save_hyperparameters()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss(self, x, x_hat, mu, logvar):
        bce = F.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce, kld

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        bce, kld = self.loss(x, x_hat, mu, logvar)

        total_loss = bce + kld

        self.log("train_recon_loss", bce)
        self.log("train_kld_loss", kld)

        self.log("train_total_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat_recon, mu, logvar = self(x)
        bce, kld = self.loss(x, x_hat_recon, mu, logvar)

        total_loss = bce + kld

        self.log("val_recon_loss", bce)
        self.log("val_kld_loss", kld)

        self.log("val_total_loss", total_loss)

        if batch_idx == 0:
            x_hat = self.decoder(torch.randn_like(mu, device=mu.device)[:16])
            self.logger.experiment.log(
                {
                    "reconstructed": wandb.Image(
                        torchvision.utils.make_grid(x_hat_recon[:16]),
                        caption=f"Epoch {self.current_epoch}, Step {self.global_step}",
                    ),
                    "generated": wandb.Image(
                        torchvision.utils.make_grid(
                            x_hat,
                        ),
                        caption=f"Epoch {self.current_epoch}, Step {self.global_step}",
                    ),
                }
            )

            # Resize to 299x299
            x = F.interpolate(x, size=299)
            x_hat = F.interpolate(x_hat, size=299)

            # Convert to u8
            x = (x * 255).to(torch.uint8)
            x_hat = (x_hat * 255).to(torch.uint8)

            # Compute FID
            self.fid.update(x, real=True)
            self.fid.update(x_hat, real=False)

            self.log("fid", self.fid.compute())
            self.fid.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                "data/celeba_hq/train",
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(512),
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                "data/celeba_hq/val",
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(512),
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )


def main():
    vae = VAE()
    wandb_logger = WandbLogger(
        project="vae-test",
        save_code=True,
        log_model=True,
        save_dir="./projects/image-generation/4-vae/logs",
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        val_check_interval=0.5,
        log_every_n_steps=1,
        default_root_dir="./projects/image-generation/4-vae/logs",
        max_steps=100_000,
        enable_checkpointing=True,
    )
    trainer.fit(vae)


if __name__:
    main()
