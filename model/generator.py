import torch
import torch.nn as nn
import torch.nn.functional as F

# input 112x112
# sizes 112 56 28 14 7 14 28 56 112

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_bn_relu = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv_bn_relu(x)


class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_bn_relu1 = ConvBNReLU(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv_bn_relu2 = ConvBNReLU(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.conv_bn_relu2(self.conv_bn_relu1(x))


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_bn_relu = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv_bn_relu(x)
        return x


class Downsampler(nn.Module):
    def __init__(self, in_channels=16, n_blocks=3):
        super().__init__()
        self.conv = nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1)

        blocks = list()
        for i in range(n_blocks):
            out_channels = 2 * in_channels
            blocks.append(DownsamplingBlock(in_channels, out_channels))
            in_channels = 2 * in_channels
        self.layers = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.layers(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, channels, n_blocks):
        super().__init__()
        blocks = list()
        for i in range(n_blocks):
            blocks.append(BottleneckBlock(channels))
        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, parallel_channels=128, concat_channels=256, parallel_blocks=4, concat_blocks=8):
        super().__init__()
        self.src_downsampler = Downsampler()
        self.tgt_downsampler = Downsampler()
        self.src_bottleneck = Bottleneck(parallel_channels, parallel_blocks)
        self.tgt_bottleneck = Bottleneck(parallel_channels, parallel_blocks)
        self.bottleneck = Bottleneck(concat_channels, concat_blocks)

    def forward(self, src, tgt):
        src = self.src_downsampler(src)
        tgt = self.tgt_downsampler(tgt)
        src = self.src_bottleneck(src)
        tgt = self.tgt_bottleneck(tgt)
        x = torch.cat([src, tgt], dim=1)
        x = self.bottleneck(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=256, n_blocks=3):
        super().__init__()
        blocks = list()
        for i in range(n_blocks):
            out_channels = in_channels // 2
            blocks.append(UpsamplingBlock(in_channels, out_channels))
            in_channels = in_channels // 2
        self.layers = nn.Sequential(*blocks)

        self.conv = nn.Conv2d(out_channels, 4, kernel_size=3, stride=1, padding=1)
      
    def forward(self, x):
        return self.conv(self.layers(x))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.init_parameters()

    def forward(self, src, tgt):
        x = self.encoder(src, tgt)
        x = self.decoder(x)
        res, mask = torch.split(x, 3, dim=1)
        res = torch.tanh(res)
        mask = torch.sigmoid(mask)
        return res, mask

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

