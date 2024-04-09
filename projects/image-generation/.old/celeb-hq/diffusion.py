import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
import wandb


@torch.no_grad()
def noiser(img, noise, t):
    noise_scale = torch.cos((t) * (3.1415 / 2)).view(-1, 1, 1, 1)
    return img * (1 - noise_scale) + noise * (noise_scale)


class DPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(5, 32, 3, padding=1, stride=2)
        # self.attention1 = nn.MultiheadAttention(embed_dim=1024, num_heads=1)

        self.down2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.attention2 = nn.MultiheadAttention(embed_dim=256, num_heads=1)

        self.down3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.attention3 = nn.MultiheadAttention(embed_dim=64, num_heads=1)

        self.up1 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1)
        self.attention4 = nn.MultiheadAttention(embed_dim=256, num_heads=1)

        self.up2 = nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1)
        # self.attention5 = nn.MultiheadAttention(embed_dim=1024, num_heads=1)

        self.up3 = nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2, output_padding=1)

        self.out = nn.Conv2d(16, 3, 3, padding=1)

        self.time_embeds = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64 * 64),
        )

        self.class_embeds = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64 * 64),
        )

    def forward(self, x, y, t):
        t_emb = self.time_embeds(t.view(-1, 1))
        t_emb = t_emb.view(-1, 1, 64, 64)
        y_emb = self.class_embeds(F.one_hot(y, 2).float())
        y_emb = y_emb.view(-1, 1, 64, 64)

        x1 = (
            F.silu(self.down1(torch.cat([x, t_emb, y_emb], dim=1)))
            + F.max_pool2d(t_emb, 2)
            + F.max_pool2d(y_emb, 2)
        )

        # b, c, h, w = x1.shape
        # x1 = x1.view(c, b, h*w)
        # x1, _ = self.attention1(x1, x1, x1)
        # x1 = x1.view(b, c, h, w)

        x2 = F.silu(self.down2(x1)) + F.max_pool2d(t_emb, 4) + F.max_pool2d(y_emb, 4)

        b, c, h, w = x2.shape
        x2 = x2.view(c, b, h * w)
        x2, _ = self.attention2(x2, x2, x2)
        x2 = x2.view(b, c, h, w)

        x3 = F.silu(self.down3(x2)) + F.max_pool2d(t_emb, 8) + F.max_pool2d(y_emb, 8)

        b, c, h, w = x3.shape
        x3 = x3.view(c, b, h * w)
        x3, _ = self.attention3(x3, x3, x3)
        x3 = x3.view(b, c, h, w)

        x4 = F.silu(self.up1(x3)) + F.max_pool2d(t_emb, 4) + F.max_pool2d(y_emb, 4) + x2

        b, c, h, w = x4.shape
        x4 = x4.view(c, b, h * w)
        x4, _ = self.attention4(x4, x4, x4)
        x4 = x4.view(b, c, h, w)

        x5 = F.silu(self.up2(x4)) + F.max_pool2d(t_emb, 2) + F.max_pool2d(y_emb, 2) + x1

        # b, c, h, w = x5.shape
        # x5 = x5.view(c, b, h*w)
        # x5, _ = self.attention5(x5, x5, x5)
        # x5 = x5.view(b, c, h, w)

        x6 = F.silu(self.up3(x5))

        x7 = self.out(x6)
        return x7


class LitDDPM(pl.LightningModule):
    def __init__(self, diffusion_steps):
        super().__init__()
        self.model = DPM()
        self.criterion = F.mse_loss

        self.diffusion_steps = diffusion_steps
        self.save_hyperparameters()
        self.automatic_optimization = False

    @torch.no_grad()
    def forward(self, x, y):
        for i in range(self.diffusion_steps):
            t = (
                torch.tensor(
                    [x.size(0) * [i + 1]], device=x.device, dtype=torch.float32
                )
                / self.diffusion_steps
            )
            x = self.model(x, y, t)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        t = (
            torch.randint(
                0,
                self.diffusion_steps,
                (x.shape[0],),
                device=x.device,
                dtype=torch.float32,
            )
            / self.diffusion_steps
        )
        noise = torch.randn_like(x, device=x.device)
        noisy_x = noiser(x, noise, t)
        target_x = noiser(x, noise, t + (1 / self.diffusion_steps))

        output = self.model(noisy_x, y, t + (1 / self.diffusion_steps))

        loss = self.criterion(output, target_x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y = torch.cat(
            [torch.zeros(5, device=x.device), torch.ones(5, device=x.device)]
        ).long()
        noise = torch.randn_like(x, device=x.device)

        img = self(noise, y)

        self.logger.experiment.log(
            {
                "clamp image": wandb.Image(
                    torchvision.utils.make_grid(img.clamp(0, 1), nrow=5),
                    caption=f"Epoch {self.current_epoch}, Global Step {self.global_step}",
                ),
                "normal image": wandb.Image(
                    torchvision.utils.make_grid(img, nrow=5, normalize=True),
                    caption=f"Epoch {self.current_epoch}, Global Step {self.global_step}",
                ),
            }
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)


def main():
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((64, 64))]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        "celeb-hq/data/celeba_hq", transform=transforms
    )
    test_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True
    )
    logger = WandbLogger(
        project="dpm", dir="celeb-hq/logs", save_code=True, log_model=True
    )
    model = LitDDPM(10)
    logger.watch(model, log="all", log_graph=True)
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="celeb-hq/logs",
        log_every_n_steps=1,
        profiler="simple",
        max_epochs=10,
        val_check_interval=0.1,
        limit_val_batches=1,
    )
    trainer.fit(model, train_dataset, test_dataloader)


if __name__ == "__main__":
    main()
