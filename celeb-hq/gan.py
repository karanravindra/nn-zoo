import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import tqdm as tqdm
import lightning.pytorch as pl
import wandb


class VariationalEncoder(nn.Module):
    def __init__(self, d=1, latent_size=100):
        super(VariationalEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 128 // d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 // d, 256 // d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256 // d, 512 // d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512 // d, 1024 // d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        self.mean = nn.Linear(2048, latent_size)
        self.logvar = nn.Linear(2048, latent_size)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.mean(x), self.logvar(x)

    def sample(self, x):
        mean, logvar = self(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def kl_divergence(self, x):
        mean, logvar = self(x)
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()


class Decoder(nn.Module):
    def __init__(self, d=1, latent_dim=100):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024 // d, 4, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(1024 // d, 512 // d, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512 // d, 256 // d, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256 // d, 128 // d, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128 // d, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, d=1):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 128 // d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 // d, 256 // d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256 // d, 512 // d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512 // d, 1024 // d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024 // d, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.layers(x).view(-1)

    def grad_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(real.size(0), 1, 1, 1).to(real.device)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interpolates = self(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).to(real.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp


class VAE(nn.Module):
    def __init__(self, d=1):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(d)
        self.decoder = Decoder(d)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x = self.decoder(z.view(-1, 100, 1, 1))
        return x, mu, logvar


class LitVAE(pl.LightningModule):
    def __init__(self, d=8):
        super(LitVAE, self).__init__()
        self.vae = VAE(d)
        self.discriminator = Discriminator(d)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=1e-3)
        self.optimizer_vae = optim.Adam(self.vae.parameters(), lr=1e-3)
        self.automatic_optimization = False
        # self.fid = FrechetInceptionDistance(feature=64)

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        optim_d, optim_vae = self.optimizers()

        # Train Discriminator
        optim_d.zero_grad()
        x_hat, mu, logvar = self.vae(x)
        loss_d = self.discriminator_loss(x, x_hat)
        self.manual_backward(loss_d)
        optim_d.step()

        # Train VAE
        optim_vae.zero_grad()
        x_hat, mu, logvar = self.vae(x)
        loss_vae = (
            self.vae_loss(x_hat, x, mu, logvar) - self.discriminator(x_hat).mean()
        )
        self.manual_backward(loss_vae)
        optim_vae.step()

        self.log("loss_d", loss_d, prog_bar=True)
        self.log("loss_vae", loss_vae, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Generate images
        x_hat = self.vae.decoder(
            torch.randn(x.size(0), 100, device=x.device).view(-1, 100, 1, 1)
        )

        # # Resize images to 299 x 299
        # resize_transform = torchvision.transforms.Resize((299, 299))
        # x_resized = torch.stack([resize_transform(x_i) for x_i in x])
        # x_hat_resized = torch.stack([resize_transform(x_hat_i) for x_hat_i in x_hat])

        # # Convert images to uint8
        # x_uint8 = (x_resized * 255).type(torch.uint8)
        # x_hat_uint8 = (x_hat_resized * 255).type(torch.uint8)

        # self.fid.update(x_uint8, real=True)
        # self.fid.update(x_hat_uint8, real=False)

        # self.log("fid", self.fid.compute(), prog_bar=True)

        # Plot images
        self.logger.experiment.log(
            {
                "Generated Images": wandb.Image(
                    torchvision.utils.make_grid(x_hat[:64], nrow=8),
                    caption=f"Epoch {self.current_epoch}, Batch {batch_idx}",
                ),
                "Trained Images": wandb.Image(
                    torchvision.utils.make_grid(self.vae(x)[0][:64], nrow=8),
                    caption=f"Epoch {self.current_epoch}, Batch {batch_idx}",
                ),
            }
        )

    def discriminator_loss(self, x, x_hat):
        real_score = self.discriminator(x)
        fake_score = self.discriminator(x_hat)
        gp = self.discriminator.grad_penalty(x, x_hat)
        loss = fake_score.mean() - real_score.mean() + 10 * gp
        return loss

    def vae_loss(self, recon_x, x, mu, logvar, kl_mult=1.0):
        # BCE Loss
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")

        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total VAE Loss
        total_loss = recon_loss + kl_div * kl_mult
        return total_loss / x.size(0)

    def configure_optimizers(self):
        return [self.optimizer_vae, self.optimizer_d], []


def main():
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((64, 64)),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        "celeb-hq/data", transform=transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )

    lit_vae = LitVAE()
    logger = pl.loggers.WandbLogger(project="vae-gan")
    trainer = pl.Trainer(
        max_epochs=10, logger=logger, val_check_interval=0.2, limit_val_batches=1
    )

    trainer.fit(lit_vae, train_loader, train_loader)


if __name__ == "__main__":
    main()
