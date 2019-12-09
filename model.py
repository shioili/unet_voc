import torch
import torch.nn as nn
import torch.nn.functional as F

class DownLayer(nn.Module):
    def __init__(self, inc):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(inc, inc*2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(inc*2, track_running_stats=True)
        self.conv2 = nn.Conv2d(inc*2, inc*2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(inc*2, track_running_stats=True)

    def forward(self, x):
        net = self.pool(x)
        net = F.relu(self.bn1(self.conv1(net)))
        net = F.relu(self.bn2(self.conv2(net)))
        return net

class UpLayer(nn.Module):
    def __init__(self, inc):
        super(UpLayer, self).__init__()
        self.convt = nn.ConvTranspose2d(inc, inc//2, 2, stride=2)
        self.conv1 = nn.Conv2d(inc, inc//2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(inc//2, track_running_stats=True)
        self.conv2 = nn.Conv2d(inc//2, inc//2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(inc//2, track_running_stats=True)

    def forward(self, up_x, shortcut_x):
        net = self.convt(up_x)

        out_shape = shortcut_x.shape    # [B, C, H, W]
        up_shape = net.shape
        pad_h = out_shape[2] - up_shape[2]
        pad_w = out_shape[3] - up_shape[3]
        net = F.pad(net, (0, pad_w, 0, pad_h))

        net = torch.cat((shortcut_x, net), 1)
        net = F.relu(self.bn1(self.conv1(net)))
        net = F.relu(self.bn2(self.conv2(net)))

        return net

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.input_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.input_conv2 = nn.Conv2d(64, 64, 3, padding=1)

        self.ld2 = DownLayer(64)
        self.ld3 = DownLayer(128)
        self.ld4 = DownLayer(256)
        self.ld5 = DownLayer(512)
        self.lu4 = UpLayer(1024)
        self.lu3 = UpLayer(512)
        self.lu2 = UpLayer(256)
        self.lu1 = UpLayer(128)

        self.output_conv = nn.Conv2d(64, 21, 1)

    def forward(self, x):
        ld1 = F.relu(self.input_conv1(x))
        ld1 = F.relu(self.input_conv2(ld1))

        ld2 = self.ld2(ld1)
        ld3 = self.ld3(ld2)
        ld4 = self.ld4(ld3)
        ld5 = self.ld5(ld4)
        lu4 = self.lu4(ld5, ld4)
        lu3 = self.lu3(lu4, ld3)
        lu2 = self.lu2(lu3, ld2)
        lu1 = self.lu1(lu2, ld1)

        out = self.output_conv(lu1)
        return out

