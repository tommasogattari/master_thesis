from collections import OrderedDict
from ignite.metrics import SSIM
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .blocks import *

class DenseUNet(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        n_features: int = 64,
        constant_features: bool = False,
        dense_n_steps: int = 3,
        dense_growth_rate: int = 0,
        dense_dropout_rate: float = 0.0,
        ):

        super().__init__()

        features = n_features

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.input = nn.Conv2d(
            in_channels=in_channels,
            out_channels=features,
            kernel_size=3,
            padding='same',
            bias=True
        )

        self.enc0 = EncBlock(
            in_channels=features,
            n_features=features, 
            dense_n_steps=dense_n_steps,
            dense_growth_rate=dense_growth_rate,
            dense_dropout_rate=dense_dropout_rate
        )
        
        self.enc1 = EncBlock(
            in_channels=features,
            n_features=features if constant_features else 2*features, 
            dense_n_steps=dense_n_steps,
            dense_growth_rate=dense_growth_rate,
            dense_dropout_rate=dense_dropout_rate
        )

        self.enc2 = EncBlock(
            in_channels=features if constant_features else 2*features,
            n_features=features if constant_features else 4*features, 
            dense_n_steps=dense_n_steps,
            dense_growth_rate=dense_growth_rate,
            dense_dropout_rate=dense_dropout_rate
        )

        self.bottleneck = Bottleneck(
            in_channels=features if constant_features else 4*features,
            n_features=features if constant_features else 8*features, 
            dense_n_steps=dense_n_steps,
            dense_growth_rate=dense_growth_rate,
            dense_dropout_rate=dense_dropout_rate
        )

        self.upconv2 = nn.ConvTranspose2d(
            features if constant_features else 8*features, 
            features if constant_features else 4*features, 
            kernel_size=2, 
            stride=2
        )

        self.dec2 = DecBlock(
            in_channels=2*features if constant_features else 2*4*features,
            n_features=features if constant_features else 4*features, 
            dense_n_steps=dense_n_steps,
            dense_growth_rate=dense_growth_rate,
            dense_dropout_rate=dense_dropout_rate
        )

        self.upconv1 = nn.ConvTranspose2d(
            features if constant_features else 4*features, 
            features if constant_features else 2*features, 
            kernel_size=2, 
            stride=2
        )

        self.dec1 = DecBlock(
            in_channels=2*features if constant_features else 2*2*features,
            n_features=features if constant_features else 2*features, 
            dense_n_steps=dense_n_steps,
            dense_growth_rate=dense_growth_rate,
            dense_dropout_rate=dense_dropout_rate
        )

        self.upconv0 = nn.ConvTranspose2d(
            features if constant_features else 2*features, 
            features, 
            kernel_size=2, 
            stride=2
        )

        self.dec0 = DecBlock(
            in_channels=2*features,
            n_features=features, 
            dense_n_steps=dense_n_steps,
            dense_growth_rate=dense_growth_rate,
            dense_dropout_rate=dense_dropout_rate
        )

        self.upconvout = nn.ConvTranspose2d(
            features, 
            features, 
            kernel_size=2, 
            stride=2
        )

        self.output = nn.Conv2d(
            in_channels=features,
            out_channels=out_channels,
            kernel_size=1,
            padding='same',
            bias=True
        )
        


    def forward(self, x):

        # _x = x
        x = self.input(x)

        enc0 = self.pool(x)
        enc0 = self.enc0(enc0)

        enc1 = self.pool(enc0)
        enc1 = self.enc1(enc1)

        enc2 = self.pool(enc1)
        enc2 = self.enc2(enc2)

        bottleneck = self.pool(enc2)
        bottleneck = self.bottleneck(bottleneck)

        # print(bottleneck.shape)

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), axis=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), axis=1)
        dec1 = self.dec1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, enc0), axis=1)
        dec0 = self.dec0(dec0)

        out = self.upconvout(dec0)
        # out = torch.cat((out, x), axis=1)
        # out = self.output(out) + _x
        out = self.output(out)

        # print(out.max().item(), out.min().item(), out.mean().item())
        
        return out

class EncBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int = 3,
        n_features: int = None, 
        dense_n_steps: int = 3,
        dense_growth_rate: int = 0,
        dense_dropout_rate: float = 0.0
        ):

        super().__init__()

        if not n_features:
            n_features = in_channels

        self.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_features,
            kernel_size=3,
            padding='same',
            bias=True
        )
        self.dense_block = DenseBlock(
            in_channels=n_features,
            n_steps=dense_n_steps,
            growth_rate=dense_growth_rate,
            dropout_rate=dense_dropout_rate
        )
        out_dense = n_features*(dense_n_steps+1)
        self.conv1 = nn.Conv2d(
            in_channels=out_dense,
            out_channels=n_features,
            kernel_size=1,
            padding='same',
            bias=True
        )
        

    def forward(self, x):
        
        return self.conv1(self.dense_block(self.conv0(x)))


class DecBlock(EncBlock):

    def __init__(
        self, 
        in_channels: int = 3,
        n_features: int = None, 
        dense_n_steps: int = 3,
        dense_growth_rate: int = 0,
        dense_dropout_rate: float = 0.0
        ):

        super().__init__(
            in_channels,
            n_features, 
            dense_n_steps,
            dense_growth_rate,
            dense_dropout_rate
        )

class Bottleneck(nn.Module):

    def __init__(
        self, 
        in_channels: int = 3,
        n_features: int = None, 
        dense_n_steps: int = 3,
        dense_growth_rate: int = 0,
        dense_dropout_rate: float = 0.0
        ):

        super().__init__()

        if not n_features:
            n_features = in_channels

        self.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_features,
            kernel_size=1,
            padding='same',
            bias=True
        )
        self.dense_block = DenseBlock(
            in_channels=n_features,
            n_steps=dense_n_steps,
            growth_rate=dense_growth_rate,
            dropout_rate=dense_dropout_rate
        )
        out_dense = n_features*(dense_n_steps+1)
        self.conv1 = nn.Conv2d(
            in_channels=out_dense,
            out_channels=n_features,
            kernel_size=1,
            padding='same',
            bias=True
        )

    def forward(self, x):
        
        return self.conv1(self.dense_block(self.conv0(x)))