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


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** (
        (
            torch.arange(
                start=0,
                end=temb_dim // 2,
                dtype=torch.float32,
                device=time_steps.device,
            )
            / (temb_dim // 2)
        )
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample using 2x2 average pooling
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        down_sample=True,
        num_heads=4,
        num_layers=1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(4, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
                for _ in range(num_layers)
            ]
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(4, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(4, out_channels) for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = (
            nn.Conv2d(out_channels, out_channels, 4, 2, 1)
            if self.down_sample
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        out = x
        for i in range(self.num_layers):

            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            # Attention block of Unet
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    r"""
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(4, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers + 1)
            ]
        )
        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
                for _ in range(num_layers + 1)
            ]
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(4, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers + 1)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(4, out_channels) for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x, t_emb):
        out = x

        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):

            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        up_sample=True,
        num_heads=4,
        num_layers=1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(4, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
                for _ in range(num_layers)
            ]
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(4, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(4, out_channels) for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = (
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1)
            if self.up_sample
            else nn.Identity()
        )

    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

        return out


class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """

    def __init__(self, model_config):
        super().__init__()
        im_channels = model_config["im_channels"]
        self.down_channels = model_config["down_channels"]
        self.mid_channels = model_config["mid_channels"]
        self.t_emb_dim = model_config["time_emb_dim"]
        self.down_sample = model_config["down_sample"]
        self.num_down_layers = model_config["num_down_layers"]
        self.num_mid_layers = model_config["num_mid_layers"]
        self.num_up_layers = model_config["num_up_layers"]

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(
            im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1)
        )

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    self.t_emb_dim,
                    down_sample=self.down_sample[i],
                    num_layers=self.num_down_layers,
                )
            )

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    self.t_emb_dim,
                    num_layers=self.num_mid_layers,
                )
            )

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlock(
                    self.down_channels[i] * 2,
                    self.down_channels[i - 1] if i != 0 else 16,
                    self.t_emb_dim,
                    up_sample=self.down_sample[i],
                    num_layers=self.num_up_layers,
                )
            )

        self.norm_out = nn.GroupNorm(4, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)

        self.cass_embed = nn.Embedding(2, self.t_emb_dim)

    def forward(self, x, y, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W

        y_emb = self.cass_embed(y.long())

        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb) + y_emb

        down_outs = []

        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4

        for mid in self.mids:
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out

    @torch.no_grad()
    def sample(self, scheduler, device, img_size=(3, 64, 64)):
        """Sample stepwise by going backward one timestep at a time.
        We save the x0 predictions
        """
        xt = torch.randn(8, *img_size, device=device)
        y = torch.cat([torch.zeros(4), torch.ones(4)]).to(device)
        for i in tqdm.tqdm(
            reversed(range(scheduler.num_timesteps)),
            desc="Sampling Model",
            total=scheduler.num_timesteps,
            unit="timesteps",
            leave=True,
        ):
            # Get prediction of noise
            noise_pred = self.forward(xt, y, torch.as_tensor(8 * [i], device=device))

            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(
                xt, noise_pred, torch.as_tensor(i, device=device)
            )

        # return final image
        # Save x0
        ims = torch.clamp(x0_pred, 0.0, 1.0).detach().cpu()
        return ims


class LinearNoiseScheduler:
    """
    Class for the linear noise scheduler that is used in DDPM.
    """

    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    @torch.no_grad()
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(
            batch_size
        )
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(
            original.device
        )[t].reshape(batch_size)

        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Apply and Return Forward process equation
        return (
            sqrt_alpha_cum_prod.to(original.device) * original
            + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise
        )

    @torch.no_grad()
    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the noise predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        x0 = (
            xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)
        ) / torch.sqrt(self.alpha_cum_prod.to(xt.device)[t])
        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (
            self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t]
        )
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (
                1.0 - self.alpha_cum_prod.to(xt.device)[t]
            )
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance**0.5
            z = torch.randn(xt.shape).to(xt.device)

            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0


class DDPM(pl.LightningModule):
    def __init__(
        self,
        diffusion_steps=10,
        lr=1e-4,
        batch_size=8,
        fid_features=2048,
        b1=1e-4,
        b2=0.02,
    ):
        super(DDPM, self).__init__()
        self.model = Unet(
            {
                "im_channels": 3,
                "im_size": 64,
                "down_channels": [16, 32, 64, 128],
                "mid_channels": [128, 128, 64],
                "down_sample": [True, True, False],
                "time_emb_dim": 64,
                "num_down_layers": 1,
                "num_mid_layers": 1,
                "num_up_layers": 1,
            }
        )

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = LinearNoiseScheduler(diffusion_steps, b1, b2)
        # self.fid = FrechetInceptionDistance(feature=fid_features)
        self.batch_size = batch_size
        self.criterion = F.mse_loss

        self.save_hyperparameters()

    def forward(self, x):
        assert True, "Forward not implemented"

    def training_step(self, batch, batch_idx):
        x, y = batch

        t = torch.randint(
            0, self.scheduler.num_timesteps, (x.size(0),), device=x.device
        )
        noise = torch.randn_like(x, device=x.device)
        x_noisy = self.scheduler.add_noise(x, noise, t)

        # Forward Pass
        x_pred = self.model(x_noisy, y, t)
        loss = self.criterion(x_pred, noise)

        # Log metrics
        self.log("loss", loss)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch

        ims = self.model.sample(self.scheduler, x.device)
        self.logger.experiment.log(
            {
                "Generated Images": wandb.Image(
                    torchvision.utils.make_grid(ims, normalize=True),
                    caption=f"Epoch {self.current_epoch}, Step {self.global_step}",
                )
            }
        )

        # # Resize to 299x299
        # x = F.interpolate(x, size=299, mode="bilinear")
        # x_fake = F.interpolate(x_fake, size=299, mode="bilinear")

        # # Make u8
        # x = (x * 255).to(torch.uint8)
        # x_fake = (x_fake * 255).to(torch.uint8)

        # # Compute FID
        # self.fid.update(x, True)
        # self.fid.update(x_fake, False)

        # self.log("fid", self.fid.compute())

        # self.logger.experiment.log(
        #     {
        #         "Generated Images": wandb.Image(
        #             torchvision.utils.make_grid(x_fake, nrow=8),
        #             caption=f"Epoch {self.current_epoch}, Step {self.global_step}",
        #         )
        #     }
        # )

    def configure_optimizers(self):
        return self.optimizer

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
            persistent_workers=True,
        )


def main():
    logger = WandbLogger(project="ddpm-faceforger", log_model="all", save_code=True)
    model = DDPM()
    trainer = pl.Trainer(
        logger=logger,
        check_val_every_n_epoch=1,
        limit_val_batches=1,
        log_every_n_steps=1,
        accumulate_grad_batches=8,
        max_steps=200_000,
        precision="16-mixed",
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
