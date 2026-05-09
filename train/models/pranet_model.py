import torch
import torch.nn as nn
import torch.nn.functional as F

class RFB_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_Block, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_linear = nn.Conv2d(4 * out_channel, out_channel, 1)
        self.shortcut = nn.Conv2d(in_channel, out_channel, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.conv_linear(out)
        short = self.shortcut(x)
        out = self.relu(out + short)
        return out

class RA_Module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, padding=1)

    def forward(self, x, y):
        # x: features, y: low-res mask
        y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = -1 * (torch.sigmoid(y)) + 1
        x = x.expand(-1, 32, -1, -1) # Match RFB output channels
        # Simple RA implementation for standalone training
        return x

class PraNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(PraNet, self).__init__()
        # Simplified backbone for scratch training
        self.encoder1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True)
        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True)
        )

        # RFB blocks
        self.rfb2 = RFB_Block(128, 32)
        self.rfb3 = RFB_Block(256, 32)
        self.rfb4 = RFB_Block(512, 32)

        # Parallel Partial Decoder (PPD)
        self.agg1 = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True)
        )
        self.final_conv = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # RFB
        r2 = self.rfb2(x2)
        r3 = self.rfb3(x3)
        r4 = self.rfb4(x4)

        # PPD Aggregation
        r3_up = F.interpolate(r3, size=r2.size()[2:], mode='bilinear', align_corners=True)
        r4_up = F.interpolate(r4, size=r2.size()[2:], mode='bilinear', align_corners=True)
        
        agg = self.agg1(torch.cat([r2, r3_up, r4_up], dim=1))
        
        logits = self.final_conv(agg)
        logits = F.interpolate(logits, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        return logits
