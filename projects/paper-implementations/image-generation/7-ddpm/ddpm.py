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
import diffusers


class DDPM(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        batch_size=64,
    ):
        super(DDPM, self).__init__()
        self.model = diffusers.UNet2DModel(
            sample_size=(128, 128),
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),  # "DownBlock2D", "AttnDownBlock2D",
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),  # "UpBlock2D", "AttnUpBlock2D",
            block_out_channels=(64, 128, 256, 256),
            layers_per_block=2,
        )

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = diffusers.DDIMScheduler(
            num_train_timesteps=4000,
            rescale_betas_zero_snr=True,
        )
        self.fid = FrechetInceptionDistance(feature=2048)
        self.batch_size = batch_size
        self.criterion = F.mse_loss

        self.save_hyperparameters()

    def forward(self, x):
        assert True, "Forward not implemented"

    def training_step(self, batch, batch_idx):
        x, _ = batch

        t = torch.randint(
            0, self.scheduler.num_train_timesteps, (x.size(0),), device=x.device
        )
        noise = torch.randn_like(x, device=x.device)
        x_noisy = self.scheduler.add_noise(x, noise, t)

        # Forward Pass
        x_pred = self.model(x_noisy, t).sample
        loss = self.criterion(x_pred, noise)

        # Log metrics
        self.log("loss", loss)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, _ = batch

        imgs_eta0 = self.generate(num_images=8, num_inference_steps=50, eta=0)
        imgs_eta05 = self.generate(num_images=8, num_inference_steps=50, eta=0.5)
        imgs_eta1 = self.generate(num_images=8, num_inference_steps=50, eta=1)

        # Log images
        self.logger.experiment.log(
            {
                "eta: 0": [wandb.Image(img) for img in imgs_eta0],
                "eta: 0.5": [wandb.Image(img) for img in imgs_eta05],
                "eta: 1": [wandb.Image(img) for img in imgs_eta1],
            }
        )

        # Log FID
        # Upscale images to 299x299
        imgs_eta0 = F.interpolate(
            imgs_eta0, size=(299, 299), mode="bilinear", align_corners=False
        )
        # imgs_eta05 = F.interpolate(
        #     imgs_eta05, size=(299, 299), mode="bilinear", align_corners=False
        # )
        imgs_eta1 = F.interpolate(
            imgs_eta1, size=(299, 299), mode="bilinear", align_corners=False
        )
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # Convert to uint8
        imgs_eta0 = (imgs_eta0 * 255).to(torch.uint8).to("cuda")
        # imgs_eta05 = (imgs_eta05 * 255).to(torch.uint8).to("cuda")
        imgs_eta1 = (imgs_eta1 * 255).to(torch.uint8).to("cuda")
        x = (x * 255).to(torch.uint8).to("cuda")

        # Calculate FID
        # eta 0
        self.fid.update(x, True)
        self.fid.update(imgs_eta0, False)
        self.log("fid eta: 0", self.fid.compute())

        # # eta 0.5
        # self.fid.update(x, True)
        # self.fid.update(imgs_eta05, False)
        # self.log("fid eta: 0.5", self.fid.compute())

        # eta 1
        self.fid.update(x, True)
        self.fid.update(imgs_eta1, False)
        self.log("fid eta: 1", self.fid.compute())

    def generate(self, num_images=8, num_inference_steps=50, eta=0.0):
        # Plot some images
        self.model.eval()
        pipeline = diffusers.DDIMPipeline(self.model, self.scheduler).to("cuda")

        with torch.no_grad():
            imgs = pipeline(
                batch_size=num_images,
                num_inference_steps=num_inference_steps,
                output_type="np",
                eta=eta,
            ).images

        # convert to torch tensor
        imgs = torch.tensor(imgs)
        imgs = imgs.permute(0, 3, 1, 2)

        return imgs

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                "data/celeba_hq/train",
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(128),
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
                        torchvision.transforms.Resize(128),
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )


def main():
    logger = WandbLogger(
        name="128x conv2d ddim", project="diffusion", log_model="all", save_code=True
    )
    model = DDPM()
    logger.watch(model, log="all", log_freq=10, log_graph=True)
    trainer = pl.Trainer(
        logger=logger,
        check_val_every_n_epoch=1,
        limit_val_batches=1,
        log_every_n_steps=1,
        accumulate_grad_batches=4,
        max_steps=500_000,
        precision="bf16-mixed",
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main()
