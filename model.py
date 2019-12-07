import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv_d1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv_d1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool_d1 = nn.MaxPool2d(2, 2)

        self.conv_d2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_d2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool_d2 = nn.MaxPool2d(2, 2)

        self.conv_d3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv_d3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool_d3 = nn.MaxPool2d(2, 2)

        self.conv_d4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv_d4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool_d4 = nn.MaxPool2d(2, 2)

        self.conv_d5_1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv_d5_2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.convt_u5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)

        self.conv_u4_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv_u4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convt_u4 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.conv_u3_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv_u3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convt_u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.conv_u2_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv_u2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.convt_u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv_u1_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv_u1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_u1_3 = nn.Conv2d(64, 21, 1)

        self.act = nn.ReLU()

    def forward(self, x):
        ld1 = self.act(self.conv_d1_1(x))
        ld1 = self.act(self.conv_d1_2(ld1))

        ld2 = self.pool_d1(ld1)
        ld2 = self.act(self.conv_d2_1(ld2))
        ld2 = self.act(self.conv_d2_2(ld2))

        ld3 = self.pool_d2(ld2)
        ld3 = self.act(self.conv_d3_1(ld3))
        ld3 = self.act(self.conv_d3_2(ld3))

        ld4 = self.pool_d3(ld3)
        ld4 = self.act(self.conv_d4_1(ld4))
        ld4 = self.act(self.conv_d4_2(ld4))

        ld5 = self.pool_d4(ld4)
        ld5 = self.act(self.conv_d5_1(ld5))
        ld5 = self.act(self.conv_d5_2(ld5))

        lu4 = self.convt_u5(ld5)
        lu4 = torch.cat((ld4, lu4), 1)
        lu4 = self.act(self.conv_u4_1(lu4))
        lu4 = self.act(self.conv_u4_2(lu4))

        lu3 = self.convt_u4(ld4)
        lu3 = torch.cat((ld3, lu3), 1)
        lu3 = self.act(self.conv_u3_1(lu3))
        lu3 = self.act(self.conv_u3_2(lu3))

        lu2 = self.convt_u3(ld3)
        lu2 = torch.cat((ld2, lu2), 1)
        lu2 = self.act(self.conv_u2_1(lu2))
        lu2 = self.act(self.conv_u2_2(lu2))

        lu1 = self.convt_u2(ld2)
        lu1 = torch.cat((ld1, lu1), 1)
        lu1 = self.act(self.conv_u1_1(lu1))
        lu1 = self.act(self.conv_u1_2(lu1))
        lu1 = self.conv_u1_3(lu1)

        return lu1

