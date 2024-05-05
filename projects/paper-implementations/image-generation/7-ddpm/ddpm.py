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
        model_type="unet",
        scheduler_type="ddim",
        num_train_timesteps=4000,
        model=None
    ):
        super(DDPM, self).__init__()
        self.model = model

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = diffusers.DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
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
        self.log("train_loss", loss)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
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
        self.log("val_loss", loss)
        
        if batch_idx == 0:
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
            imgs_eta0 = F.interpolate(imgs_eta0, size=(299, 299), mode="bilinear", align_corners=False)
            imgs_eta05 = F.interpolate(imgs_eta05, size=(299, 299), mode="bilinear", align_corners=False)
            imgs_eta1 = F.interpolate(imgs_eta1, size=(299, 299), mode="bilinear", align_corners=False)
            
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

            # Convert to uint8
            imgs_eta0 = (imgs_eta0 * 255).to(torch.uint8).to(x.device)
            imgs_eta05 = (imgs_eta05 * 255).to(torch.uint8).to(x.device)
            imgs_eta1 = (imgs_eta1 * 255).to(torch.uint8).to(x.device)
            
            x = (x * 255).to(torch.uint8).to(x.device)

            # Calculate FID
            # eta 0
            self.fid.update(x, True)
            self.fid.update(imgs_eta0, False)
            self.log("fid eta: 0", self.fid.compute())
            self.fid.reset()

            # eta 0.5
            self.fid.update(x, True)
            self.fid.update(imgs_eta05, False)
            self.log("fid eta: 0.5", self.fid.compute())
            self.fid.reset()

            # eta 1
            self.fid.update(x, True)
            self.fid.update(imgs_eta1, False)
            self.log("fid eta: 1", self.fid.compute())
            self.fid.reset()

    def generate(self, num_images=8, num_inference_steps=50, eta=0.0):
        # Plot some images
        self.model.eval()
        pipeline = diffusers.DDIMPipeline(self.model, self.scheduler)

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
            torchvision.datasets.MNIST(
                "data",
                train=True,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "data",
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )


def main():
    logger = WandbLogger(
        name="mnist", project="diffusion", log_model="all", save_code=True, tags=["ddim", "mnist"]
    )
    model = DDPM(model=diffusers.UNet2DModel(
            sample_size=(28, 28),
            in_channels=1,
            out_channels=1,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
            ),  # "DownBlock2D", "AttnDownBlock2D",
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
            ),  # "UpBlock2D", "AttnUpBlock2D",
            block_out_channels=(64, 128),
            layers_per_block=2,
        ))
    logger.watch(model, log="all", log_freq=10, log_graph=True)
    trainer = pl.Trainer(
        logger=logger,
        check_val_every_n_epoch=1,
        limit_val_batches=64,
        accumulate_grad_batches=4,
        max_steps=10_000,
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main()
