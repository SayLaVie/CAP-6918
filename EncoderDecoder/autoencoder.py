import torch
import torch.nn as nn
import torch.nn.functional as f


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, 
            batch_norm=False, activation_function=f.leaky_relu):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.activation_function = activation_function
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        # x = self.batch_norm(x)
        x = self.activation_function(x)
        x = self.maxpool(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, 
            batch_norm=False, activation_function=f.leaky_relu):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor=2)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.conv(x)
        # x = self.batch_norm(x)
        x = self.activation_function(x)
        x = self.upsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.downconv1 = DownConvBlock(3, 32, kernel_size=5, padding=2).cuda()
        self.downconv2 = DownConvBlock(32, 32, kernel_size=3)
        self.downconv3 = DownConvBlock(32, 64, kernel_size=3)
        self.downconv4 = DownConvBlock(64, 64, kernel_size=3)
        self.downconv5 = DownConvBlock(64, 64, kernel_size=3)

    def forward(self, x):
        x = self.downconv1(x)
        x = self.downconv2(x)
        x = self.downconv3(x)
        x = self.downconv4(x)
        x = self.downconv5(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upconv1 = UpConvBlock(64, 64, kernel_size=3)
        self.upconv2 = UpConvBlock(64, 64, kernel_size=3)
        self.upconv3 = UpConvBlock(64, 64, kernel_size=3)
        self.upconv4 = UpConvBlock(64, 32, kernel_size=3)
        self.upconv5 = UpConvBlock(32, 3, kernel_size=5, padding=2, activation_function=torch.tanh)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)

        return x


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()


    def forward(self, x):
        x = self.enc(x)
        encoding = torch.flatten(x, start_dim=1)
        x = self.dec(x)

        return x, encoding
