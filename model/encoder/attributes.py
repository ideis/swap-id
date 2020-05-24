import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class Deconv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)


class AttributesEncoder(nn.Module):
    def __init__(self):
        super(AttributesEncoder, self).__init__()
        self.conv1 = Conv(3, 32)
        self.conv2 = Conv(32, 64)
        self.conv3 = Conv(64, 128)
        self.conv4 = Conv(128, 256)
        self.conv5 = Conv(256, 512)
        self.conv6 = Conv(512, 1024)
        self.conv7 = Conv(1024, 1024)

        self.deconv1 = Deconv(1024, 1024)
        self.deconv2 = Deconv(2048, 512)
        self.deconv3 = Deconv(1024, 256)
        self.deconv4 = Deconv(512, 128)
        self.deconv5 = Deconv(256, 64)
        self.deconv6 = Deconv(128, 32)

        self.init_parameters()

    def forward(self, Xt):
        feat1 = self.conv1(Xt)      # 32x128x128
        feat2 = self.conv2(feat1)   # 64x64x64
        feat3 = self.conv3(feat2)   # 128x32x32
        feat4 = self.conv4(feat3)   # 256x16xx16
        feat5 = self.conv5(feat4)   # 512x8x8
        feat6 = self.conv6(feat5)   # 1024x4x4
        z_attr1 = self.conv7(feat6) # 1024x2x2
        z_attr2 = self.deconv1(z_attr1, feat6)
        z_attr3 = self.deconv2(z_attr2, feat5)
        z_attr4 = self.deconv3(z_attr3, feat4)
        z_attr5 = self.deconv4(z_attr4, feat3)
        z_attr6 = self.deconv5(z_attr5, feat2)
        z_attr7 = self.deconv6(z_attr6, feat1)
        z_attr8 = F.interpolate(z_attr7, scale_factor=2, mode='bilinear', align_corners=True)
        return z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7, z_attr8


    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)