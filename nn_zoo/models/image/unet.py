import torch
from torch import nn

__all__ = ["UNet"]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv_block(x)


class UNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs.get("config")

        self.encoder1 = ConvBlock(config.in_channels, config.initial_filters)
        self.encoder2 = ConvBlock(config.initial_filters, config.initial_filters * 2)
        self.encoder3 = ConvBlock(
            config.initial_filters * 2, config.initial_filters * 4
        )
        self.encoder4 = ConvBlock(
            config.initial_filters * 4, config.initial_filters * 8
        )
        self.encoder5 = ConvBlock(
            config.initial_filters * 8, config.initial_filters * 16
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = UpConvBlock(
            config.initial_filters * 16, config.initial_filters * 8
        )
        self.upconv3 = UpConvBlock(
            config.initial_filters * 8, config.initial_filters * 4
        )
        self.upconv2 = UpConvBlock(
            config.initial_filters * 4, config.initial_filters * 2
        )
        self.upconv1 = UpConvBlock(config.initial_filters * 2, config.initial_filters)

        self.final_conv = nn.Conv2d(
            config.initial_filters, config.out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        dec4 = self.upconv4(enc5, enc4)
        dec3 = self.upconv3(dec4, enc3)
        dec2 = self.upconv2(dec3, enc2)
        dec1 = self.upconv1(dec2, enc1)

        return self.final_conv(dec1)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    config = UNetConfig(in_channels=1, out_channels=1, initial_filters=2)

    model = UNet(config)
    x = torch.randn(1, 1, 256, 256)
    preds = model(x)
    print(preds.shape)  # Should be [1, 1, 256, 256]
    print(model.get_param_count())
