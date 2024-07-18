import numpy
import torch.nn.functional as F
import torch
import torch.nn as nn

def get_hp(data):
    """ 高通滤波器，保留边缘信息 """
    rs = F.avg_pool2d(data, kernel_size=5, stride=1, padding=2)
    rs = data - rs
    return rs

class unetConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.unetConv = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels)
        ).double()

    def forward(self, x):
        return self.unetConv(x)

class centerleftConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.centerConv = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ).double()

    def forward(self, x):
        return self.centerConv(x)

class centerrightConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.centerrightConv = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels)
        ).double()

    def forward(self, x):
        return self.centerrightConv(x)

class unetconvtranspose(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.unetconvtranspose = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        ).double()

    def forward(self, x):
        return self.unetconvtranspose(x)

class out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        ).double()

    def forward(self, x):
        return self.out(x)

def conv3x3(in_channels, out_channels, stride=1, padding=1, *args, **kwargs):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                     stride=stride, padding=padding, *args, **kwargs).double()

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.basic = []
        self.basic.append(conv3x3(channels, channels))
        self.basic.append(nn.InstanceNorm2d(channels))
        self.basic.append(nn.ReLU(True))
        self.basic.append(conv3x3(channels, channels))
        self.basic.append(nn.InstanceNorm2d(channels))
        self.basic = nn.Sequential(*self.basic).double()

    def forward(self, x):
        return self.basic(x) + x

class fakenet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(fakenet, self).__init__()
        # left
        self.left_0 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1, bias=False).double()
        self.left_1 = unetConv(64, 128)
        self.left_2 = unetConv(128, 256)
        self.left_3 = unetConv(256, 512)
        self.left_4 = centerleftConv(512, 512)

        # right
        self.right_4 = centerrightConv(512, 512)
        self.right_3 = unetconvtranspose(1024, 256)
        self.right_2 = unetconvtranspose(512, 128)
        self.right_1 = unetconvtranspose(256, 64)

        # out
        self.out = out(128, output_nc)

        # colorization
        self.fake = nn.Sequential(conv3x3(output_nc, 32),
                                  nn.InstanceNorm2d(32),
                                  nn.ReLU(True),
                                  ResBlock(32)).double()

    def forward(self, x):
        # left
        x0 = self.left_0(x)
        x1 = self.left_1(x0)
        x2 = self.left_2(x1)
        x3 = self.left_3(x2)
        x4 = self.left_4(x3)

        # right
        x5 = self.right_4(x4)

        x5 = torch.cat([x3, x5], 1)
        x6 = self.right_3(x5)

        x6 = torch.cat((x2, x6), 1)
        x7 = self.right_2(x6)

        x7 = torch.cat((x1, x7), 1)
        x8 = self.right_1(x7)

        x8 = torch.cat((x0, x8), 1)
        out = self.out(x8)

        fake = self.fake(out)

        return fake


class pannet(nn.Module):
    def __init__(self, input_nc):
        super().__init__()

        self.pan = nn.Sequential(conv3x3(input_nc, 32),
                                  nn.InstanceNorm2d(32),
                                  nn.ReLU(True),
                                  ResBlock(32)).double()

    def forward(self, pan):
        pan_hp = get_hp(pan)

        pan_edge = self.pan(pan_hp)

        return pan_edge


class fusion_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fusion = nn.Sequential(conv3x3(64, 32),
                                    nn.InstanceNorm2d(32),
                                    nn.ReLU(True),
                                    ResBlock(32),
                                    nn.ReLU(True),
                                    ResBlock(32),
                                    nn.ReLU(True),
                                    ResBlock(32),
                                    nn.ReLU(True),
                                    ResBlock(32),
                                    nn.ReLU(True)
                                    ).double()

    def forward(self, fake, pan):
        return self.fusion(torch.cat((fake, pan), dim=1))


class RestoreNet(nn.Module):
    def __init__(self, output_nc):
        super().__init__()

        self.net = nn.Sequential(ResBlock(32),
                                 nn.ReLU(True),
                                 conv3x3(32, output_nc)
                                 ).double()

    def forward(self, x):
        return self.net(x)


class colornet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()

        self.fakenet = fakenet(input_nc, output_nc)
        self.pannet = pannet(input_nc)
        self.fusion_net = fusion_net()
        self.de_net = RestoreNet(output_nc)


    def forward(self, x):
        # 全色影像经过unet的着色结果
        fake = self.fakenet(x)

        # 全色影像经过处理后的结果
        pan = self.pannet(x)

        # pansharpening后的结果
        fusion = self.fusion_net(fake, pan)

        # 输出
        out = self.de_net(fusion)

        return out







