import torch
import torch.nn as nn

class DenseBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        n_steps: int = 3, 
        growth_rate: int = 0,
        dropout_rate: float = 0.0
        ):

        super().__init__()

        self.steps = nn.ModuleList()

        if growth_rate == 0:
            for i in range(n_steps):
                self.steps.append(DenseStep((i+1)*in_channels, in_channels, dropout_rate))
        else:
            # for i in range(n_steps):
            #     self.steps.append(DenseStep(
            #         (i+1)*in_channels + i*growth_rate, 
            #         in_channels, 
            #         dropout_rate
            #         ))
            raise NotImplementedError
        

    def forward(self, x):

        outs = list()
        for i in range(len(self.steps)):
            x = self.steps[i](x)
            outs.append(x)
            for j in range(i+1):
                x = torch.cat((x, outs[j]), axis=1)
        
        return x

class DenseStep(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        dropout_rate: float = 0.0,
        ):

        super().__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding='same',
            bias=True
        )
        self.dropout = nn.Dropout(dropout_rate)

        pass


    def forward(self, x):

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)

        return x