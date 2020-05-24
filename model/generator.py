import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder.attributes import AttributesEncoder


class AADLayer(nn.Module):
    def __init__(self, c_in, c_attr, c_id=512):
        super(AADLayer, self).__init__()
        self.c_attr = c_attr
        self.c_id = c_id
        self.c_in = c_in

        self.conv1 = nn.Conv2d(c_attr, c_in, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(c_attr, c_in, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(c_id, c_in)
        self.fc2 = nn.Linear(c_id, c_in)
        self.norm = nn.InstanceNorm2d(c_in, affine=False)

        self.conv_h = nn.Conv2d(c_in, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, h_in, z_attr, z_id):
        h = self.norm(h_in)

        # attributes integration
        gamma_attr = self.conv1(z_attr)
        beta_attr = self.conv2(z_attr)
        A = gamma_attr * h + beta_attr

        # identity integration
        gamma_id = self.fc1(z_id)
        beta_id = self.fc2(z_id)
        gamma_id = gamma_id.reshape(h.shape[0], self.c_x, 1, 1).expand_as(h)
        beta_id = beta_id.reshape(h.shape[0], self.c_x, 1, 1).expand_as(h)
        I = gamma_id * h + beta_id

        # adaptively attention mask
        M = torch.sigmoid(self.conv_h(h))
        out = (torch.ones_like(M).to(M.device) - M) * A + M * I
        return out


class AADBlock(nn.Module):
    def __init__(self, cin, cout, c_attr, c_id=512):
        super(AADBlock, self).__init__()
        self.cin = cin
        self.cout = cout

        self.AAD1 = AADLayer(cin, c_attr, c_id)
        self.conv1 = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.AAD2 = AADLayer(cin, c_attr, c_id)
        self.conv2 = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        if cin != cout:
            self.AAD3 = AADLayer(cin, c_attr, c_id)
            self.conv3 = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu3 = nn.ReLU(inplace=True)

    def forward(self, h, z_attr, z_id):
        x = self.AAD1(h, z_attr, z_id)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.AAD2(x,z_attr, z_id)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.cin != self.cout:
            h = self.AAD3(h, z_attr, z_id)
            h = self.relu3(h)
            h = self.conv3(h)
        x = x + h
        
        return x


class AADGenerator(nn.Module):
    def __init__(self, c_id=512):
        super(AADGenerator, self).__init__()
        self.up1 = nn.ConvTranspose2d(c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.AADBlock1 = AADBlock(1024, 1024, 1024, c_id)
        self.AADBlock2 = AADBlock(1024, 1024, 2048, c_id)
        self.AADBlock3 = AADBlock(1024, 1024, 1024, c_id)
        self.AADBlock4 = AADBlock(1024, 512, 512, c_id)
        self.AADBlock5 = AADBlock(512, 256, 256, c_id)
        self.AADBlock6 = AADBlock(256, 128, 128, c_id)
        self.AADBlock7 = AADBlock(128, 64, 64, c_id)
        self.AADBlock8 = AADBlock(64, 3, 64, c_id)

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, z_attr, z_id):
        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))
    
        m2 = F.interpolate(self.AADBlock1(m,  z_attr[0], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m3 = F.interpolate(self.AADBlock2(m2, z_attr[1], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m4 = F.interpolate(self.AADBlock3(m3, z_attr[2], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m5 = F.interpolate(self.AADBlock4(m4, z_attr[3], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m6 = F.interpolate(self.AADBlock5(m5, z_attr[4], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m7 = F.interpolate(self.AADBlock6(m6, z_attr[5], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        m8 = F.interpolate(self.AADBlock7(m7, z_attr[6], z_id), scale_factor=2, mode='bilinear', align_corners=True)
        y = self.AADBlock8(m8, z_attr[7], z_id)
        return torch.tanh(y)


class Generator(nn.Module):
    def __init__(self, c_id=512):
        super(Generator, self).__init__()
        self.encoder = AttributesEncoder()
        self.generator = AADGenerator(c_id)

    def forward(self, Xt, z_id, return_attributes=True):
        attr = self.encoder(Xt)
        Y_hat = self.generator(attr, z_id)
        if return_attributes:
            return Y_hat, attr
        else:
            return Y_hat

    def get_attr(self, X):
        return self.encoder(X)
