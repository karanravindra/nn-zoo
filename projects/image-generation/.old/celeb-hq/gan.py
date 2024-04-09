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


class GAN(pl.LightningModule):
    def __init__(
        self, d=1, lr=1e-4, betas=(0.5, 0.999), batch_size=64, fid_features=2048
    ):
        super(GAN, self).__init__()
        self.generator = Generator(d)
        self.discriminator = Discriminator(d)
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=betas
        )
        self.automatic_optimization = False
        self.fid = FrechetInceptionDistance(feature=fid_features)
        self.batch_size = batch_size

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        optim_d, optim_g = self.optimizers()

        # Train discriminator
        optim_d.zero_grad()
        z = torch.randn(x.size(0), 100, 1, 1, device=self.device)
        x_fake = self.generator(z).detach()
        d_real = self.discriminator(x)
        d_fake = self.discriminator(x_fake)
        loss_d = (
            d_fake.mean()
            - d_real.mean()
            + self.discriminator.grad_penalty(x, x_fake) * 10
            + 0.001 * (d_real**2).mean()
        )
        self.manual_backward(loss_d)
        optim_d.step()

        # Train generator
        if self.global_step % 4 == 0:
            optim_g.zero_grad()
            z = torch.randn(x.size(0), 100, 1, 1, device=self.device)
            x_fake = self.generator(z)
            d_fake = self.discriminator(x_fake)
            loss_g = -d_fake.mean()
            self.manual_backward(loss_g)
            optim_g.step()
            self.log("loss_g", loss_g)

        # Log metrics
        self.log("loss_d", loss_d)
        self.log("d_real", d_real.mean())
        self.log("d_fake", d_fake.mean())

    def validation_step(self, batch, batch_idx):
        x, y = batch

        z = torch.randn(x.size(0), 100, 1, 1, device=self.device)
        x_fake = self.generator(z)

        # Resize to 299x299
        x = F.interpolate(x, size=299, mode="bilinear")
        x_fake = F.interpolate(x_fake, size=299, mode="bilinear")

        # Make u8
        x = (x * 255).to(torch.uint8)
        x_fake = (x_fake * 255).to(torch.uint8)

        # Compute FID
        self.fid.update(x, True)
        self.fid.update(x_fake, False)

        self.log("fid", self.fid.compute())

        self.logger.experiment.log(
            {
                "Generated Images": wandb.Image(
                    torchvision.utils.make_grid(x_fake, nrow=8),
                    caption=f"Epoch {self.current_epoch}, Step {self.global_step}",
                )
            }
        )

    def configure_optimizers(self):
        return self.optimizer_d, self.optimizer_g

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                "celeb-hq/data/celeba_hq/train",
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(64),
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                "celeb-hq/data/celeba_hq/val",
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(64),
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )


def main():
    logger = WandbLogger(project="celeb-gan")
    model = GAN()
    trainer = pl.Trainer(
        logger=logger,
        val_check_interval=0.25,
        limit_val_batches=1,
        log_every_n_steps=1,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
