import math
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(ConvNet, self).__init__()
        self.n_layers = n_layers
        
        kernel_size = 4
        padding = math.ceil((kernel_size - 1.0) / 2)
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kernel_size, stride=2, padding=padding),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kernel_size, stride=1, padding=padding),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kernel_size, stride=1, padding=padding)]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)

        self.init_parameters()
    def forward(self, input):
        return self.model(input)


    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=6, n_scales=3, norm_layer=torch.nn.InstanceNorm2d):
        super(Discriminator, self).__init__()
        self.n_scales = n_scales
        self.n_layers = n_layers

        for i in range(n_scales):
            net = ConvNet(input_nc, ndf, n_layers, norm_layer)
            setattr(self, 'layer' + str(i), net.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        result = []
        x_downsampled = x
        for i in range(self.n_scales):
            model = getattr(self, 'layer' + str(i))
            result.append(model(x_downsampled))
            x_downsampled = self.downsample(x_downsampled)
        return result
