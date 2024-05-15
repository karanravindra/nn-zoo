import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm as tqdm
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
import diffusers


class Diffusion(pl.LightningModule):
    def __init__(
        self,
        sample_size=(32, 32),
        in_channels=1,
        out_channels=1,
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
        block_out_channels=(64, 128, 128, 128),
        layers_per_block=2,
        optim="sgd",
        lr=0.01,
        momentum=0.9,
        weight_decay=4e-5,
        batch_size=64,
        num_train_timesteps=4000,
        classes=None,
        model_type="unet",
        scheduler_type="ddim",
        dataset="mnist",
    ):
        super(Diffusion, self).__init__()
        self.model = diffusers.UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,  # "DownBlock2D", "AttnDownBlock2D",
            up_block_types=up_block_types,  # "UpBlock2D", "AttnUpBlock2D",
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = diffusers.DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            rescale_betas_zero_snr=True,
        )
        self.batch_size = batch_size
        self.criterion = F.mse_loss

        self.save_hyperparameters()

    def forward(self, x):
        assert True, "Forward not implemented"

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0 and batch_idx == 0:
            self.logger.experiment.log(
                {
                    "real": wandb.Image(
                        torchvision.utils.make_grid(
                            batch[0][:64], nrow=8, normalize=True
                        ),
                        caption="real",
                    )
                }
            )

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
            # imgs_eta05 = self.generate(num_images=8, num_inference_steps=50, eta=0.5)
            imgs_eta1 = self.generate(num_images=8, num_inference_steps=50, eta=1)

            # Log images
            self.logger.experiment.log(
                {
                    "eta: 0": wandb.Image(
                        torchvision.utils.make_grid(imgs_eta0, nrow=4, normalize=True),
                        caption="eta: 0",
                    ),
                    "eta: 1": wandb.Image(
                        torchvision.utils.make_grid(imgs_eta1, nrow=4, normalize=True),
                        caption="eta: 1",
                    ),
                }
            )

            fid = FrechetInceptionDistance(feature=2048)
            
            if self.hparams.out_channels == 1:
                x = x.repeat(1, 3, 1, 1)

            # Log FID
            # Upscale images to 299x299
            imgs_eta0 = F.interpolate(
                imgs_eta0, size=(299, 299), mode="bilinear", align_corners=False
            )
            imgs_eta1 = F.interpolate(
                imgs_eta1, size=(299, 299), mode="bilinear", align_corners=False
            )

            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

            # Convert to uint8
            imgs_eta0 = (imgs_eta0 * 255).to(torch.uint8).cpu()
            imgs_eta1 = (imgs_eta1 * 255).to(torch.uint8).cpu()

            x = (x * 255).to(torch.uint8).cpu()

            # Calculate FID
            # eta 0
            fid.update(x, True)
            fid.update(imgs_eta0, False)
            self.log("fid eta: 0", fid.compute())
            fid.reset()

            # eta 1
            fid.update(x, True)
            fid.update(imgs_eta1, False)
            self.log("fid eta: 1", fid.compute())
            fid.reset()

    @torch.no_grad()
    def generate(self, num_images=8, num_inference_steps=50, eta=0.0):
        # Plot some images
        self.model.eval()
        pipeline = diffusers.DDIMPipeline(self.model, self.scheduler).to(self.model.device)
        imgs = pipeline(
            batch_size=num_images,
            num_inference_steps=num_inference_steps,
            output_type="np",
            eta=eta,
        ).images

        # convert to torch tensor
        imgs = torch.tensor(imgs)
        imgs = imgs.permute(0, 3, 1, 2)

        # 1 channel to 3
        if self.hparams.out_channels == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        return imgs

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "data",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize(self.hparams.sample_size),
                        (
                            torchvision.transforms.Grayscale()
                            if self.hparams.in_channels == 1
                            else torchvision.transforms.Lambda(lambda x: x)
                        ),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "data",
                train=False,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize(self.hparams.sample_size),
                        (
                            torchvision.transforms.Grayscale()
                            if self.hparams.in_channels == 1
                            else torchvision.transforms.Lambda(lambda x: x)
                        ),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )


if __name__ == "__main__":
    logger = WandbLogger(
        name="diffusion",
        project="diffusion",
        log_model=True,
        save_code=True,
        tags=["ddim", "mnist"],
    )
    model = Diffusion()
    logger.watch(model.model, log="all", log_freq=10, log_graph=True)
    trainer = pl.Trainer(
        logger=logger,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=4,
        num_sanity_val_steps=0,
        max_steps=10_000,
        # precision="bf16-mixed",
    )
    trainer.fit(model)
