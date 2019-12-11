import torch
import torch.nn as nn
import torch.nn.functional as F

from math import cos, sin, pi

def repeat(times, fn, *args, **kwargs):
    return [fn(*args, **kwargs) for _ in range(times)]

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBNPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, expand_channels, out_channels, residual=False, downsample=False):
        super().__init__()
        self.residual = residual
        stride = 2 if downsample else 1

        self.block = nn.Sequential(
            ConvBNPReLU(in_channels, expand_channels, kernel_size=1),
            ConvBNPReLU(expand_channels, expand_channels, kernel_size=3, stride=stride, padding = 1, groups=expand_channels),
            ConvBN(expand_channels, out_channels, kernel_size=1)
        )

        if self.residual:
            self.block[-1].bn.weight.data.zero_()        

    def forward(self, x):
        if self.residual:
            return self.block(x) + x
        else:
            return self.block(x)


class ArcMargin(nn.Module):
    def __init__(self, num_classes, emb_size=512, m=0.4, s=64.):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, emb_size))
        nn.init.xavier_normal_(self.weight)
        
        self.s = self.scalar(s)
        self.sin_m, self.cos_m = self.scalar(sin(m)), self.scalar(cos(m))
        self.th = self.scalar(cos(pi - m))
        self.mm = self.scalar(sin(pi - m) * m)
    
    @staticmethod
    def scalar(p):
        return nn.Parameter(torch.tensor(p), requires_grad=False)

    def forward(self, x, labels):
        x = F.normalize(x)
        W = F.normalize(self.weight)
        cosine = x @ W.t()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = one_hot * phi + (1. - one_hot) * cosine
        return output * self.s


class AirMargin(nn.Module):
    def __init__(self, num_classes, emb_size=512, m=0.4, s=64.):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, emb_size))
        nn.init.xavier_normal_(self.weight)
        self.m = m
        self.s = s

    def forward(self, x, label):
        W = F.normalize(self.weight)
        x = F.normalize(x)
        cosine = x @ W.t()
        theta = torch.acos(cosine)
        m = torch.zeros_like(theta)
        m.scatter_(1, label.view(-1, 1), self.m)
        scale = -2 * self.s / pi
        return self.s + scale * (theta + m)


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNPReLU(3, 64, kernel_size=3, stride=2, padding=1),
            ConvBNPReLU(64, 64, kernel_size=1, groups=64),
            Block(64, 128, 64, downsample=True),
            *repeat(2, Block, in_channels=64, expand_channels=128, out_channels=64, residual=True),
            Block(64, 256, 128, downsample=True),
            *repeat(3, Block, in_channels=128, expand_channels=256, out_channels=128, residual=True),
            Block(128, 512, 256, downsample=True),
            *repeat(6, Block, in_channels=256, expand_channels=512, out_channels=256, residual=True),
            ConvBNPReLU(256, 512, kernel_size=1),
            ConvBN(512, 512, kernel_size=7, groups=512),
            nn.Flatten(1),
            nn.Dropout(p=0.25),
            nn.Linear(512, 513)
        )
        self.metric = AirMargin(num_classes)
        self.init_parameters()
     
    def forward(self, x, labels=None):
        features, realness_logit = torch.split(self.backbone(x), [512, 1], dim=1)
        if torch.is_tensor(labels):
            logits = self.metric(features, labels)
            return logits, realness_logit
        else:
            return features, realness_logit

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

# x = torch.randn(2, 3, 112, 112)
# y = torch.ones(2).long()
# model = Discriminator(1000)
# logits, realness_logit = model(x, y)
# print(logits.shape)
# print(realness_logit.shape)