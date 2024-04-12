import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import wandb
import tqdm as tqdm
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.image.fid import FrechetInceptionDistance


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
        )

        self.mu = nn.Linear(256, 64)
        self.logvar = nn.Linear(256, 64)

    def forward(self, x):
        x = self.layers(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def grad_penalty(self, real, fake):
        alpha = torch.rand(
            real.size(0), 1, 1, 1, device=real.device, requires_grad=True
        )
        interpolates = alpha * real + (1 - alpha) * fake
        # interpolates.requires_grad = True
        print(interpolates.requires_grad)
        disc_interpolates = self(interpolates)
        print(disc_interpolates.requires_grad)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(
                disc_interpolates.size(), device=real.device, requires_grad=True
            ),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


class VAE(pl.LightningModule):
    def __init__(self, lr=1e-3, batch_size=64, fid_features=2048):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()

        # self.fid = FrechetInceptionDistance(fid_features)

        self.lr = lr
        self.batch_size = batch_size

        self.optim_encoder = optim.Adam(self.encoder.parameters(), lr=lr)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), lr=lr)
        self.optim_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr)

        self.automatic_optimization = False

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

        optim_encoder, optim_decoder, optim_discriminator = self.optimizers()

        # Train encoder
        optim_encoder.zero_grad()
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        loss_prior = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        x_hat = self.decoder(z)
        loss_recon = self.discriminator(x_hat).mean()
        encoder_loss = loss_prior + loss_recon
        encoder_loss.backward()
        optim_encoder.step()

        # Train decoder
        optim_decoder.zero_grad()
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        loss_recon = self.discriminator(x_hat).mean()
        loss_gan = (
            (self.discriminator(x)).log()
            + (1 - self.discriminator(x_hat)).log()
            + (1 - self.discriminator(self.decoder(torch.randn_like(mu)))).log()
        )
        decoder_loss = loss_recon - loss_gan
        decoder_loss.backward()
        optim_decoder.step()

        # Train discriminator
        optim_discriminator.zero_grad()
        z = torch.randn_like(mu)
        x_fake = self.decoder(z).detach()
        d_real = self.discriminator(x)
        d_fake = self.discriminator(x_fake)
        loss_d = (
            d_fake.mean()
            - d_real.mean()
            + self.discriminator.grad_penalty(x, x_fake) * 10
            + 0.001 * (d_real**2).mean()
        )

        loss_d.backward()
        optim_discriminator.step()

        # Logging
        self.log("train_encoder_loss", encoder_loss)
        self.log("train_decoder_loss", decoder_loss)
        self.log("train_discriminator_loss", loss_d)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat_recon, mu, logvar = self(x)
        bce, kld = self.loss(x, x_hat_recon, mu, logvar)

        total_loss = bce + kld

        self.log("val_recon_loss", bce)
        self.log("val_kld_loss", kld)

        self.log("val_total_loss", total_loss)

        if batch_idx == 0:
            self.logger.experiment.log(
                {
                    "reconstructed": wandb.Image(
                        torchvision.utils.make_grid(x_hat_recon),
                        caption=f"Epoch {self.current_epoch}, Step {self.global_step}",
                    )
                }
            )

            x_hat = self.decoder(torch.randn_like(mu))
            self.logger.experiment.log(
                {
                    "generated": wandb.Image(
                        torchvision.utils.make_grid(x_hat),
                        caption=f"Epoch {self.current_epoch}, Step {self.global_step}",
                    )
                }
            )

            # # Resize to 299x299
            # x = F.interpolate(x, size=299)
            # x_hat = F.interpolate(x_hat, size=299)

            # # Convert to u8
            # x = (x * 255).to(torch.uint8)
            # x_hat = (x_hat * 255).to(torch.uint8)

            # # Compute FID
            # self.fid.update(x, real=True)
            # self.fid.update(x_hat, real=False)

            # self.log("fid", self.fid.compute())

    def configure_optimizers(self):
        return [self.optim_encoder, self.optim_decoder, self.optim_discriminator]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                "data/celeba_hq/train",
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(64),
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
                        torchvision.transforms.Resize(64),
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
        project="vae-gan",
        save_code=True,
        log_model=True,
        save_dir="./logs",
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        val_check_interval=0.5,
        log_every_n_steps=1,
        default_root_dir="./logs",
        max_steps=100_000,
    )
    trainer.fit(vae)


if __name__:
    main()
