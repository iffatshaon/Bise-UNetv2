import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DetailBranch(nn.Module):
    def __init__(self, base=64):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, base, ks=3, stride=2, padding=1),
            ConvBNReLU(base, base, ks=3, stride=1, padding=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(base, base, ks=3, stride=2, padding=1),
            ConvBNReLU(base, base, ks=3, stride=1, padding=1),
            ConvBNReLU(base, base, ks=3, stride=1, padding=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(base, base*2, ks=3, stride=2, padding=1),
            ConvBNReLU(base*2, base*2, ks=3, stride=1, padding=1),
            ConvBNReLU(base*2, base*2, ks=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.S1(x)
        x = self.S2(x)
        x = self.S3(x)
        return x

class StemBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=2, padding=1)
        self.left = nn.Sequential(
            ConvBNReLU(out_chan, out_chan//2, ks=1, stride=1, padding=0),
            ConvBNReLU(out_chan//2, out_chan, ks=3, stride=2, padding=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_chan*2, out_chan, ks=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x_left = self.left(x)
        x_right = self.right(x)
        x = torch.cat([x_left, x_right], dim=1)
        x = self.fuse(x)
        return x

class GELayer(nn.Module):
    def __init__(self, in_chan, out_chan, exp=6, stride=1):
        super(GELayer, self).__init__()
        mid_chan = out_chan * exp
        self.conv1 = ConvBNReLU(in_chan, in_chan, ks=3, stride=1, padding=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=stride, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if stride == 2:
            self.dwconv2 = nn.Sequential(
                nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=stride, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        if self.stride == 2:
            x = self.dwconv2(x)
        return self.relu(feat + x)

class CEBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(CEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(in_chan)
        self.conv_gap = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(out_chan, out_chan, ks=3, stride=1, padding=1)

    def forward(self, x):
        feat = self.gap(x)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat

class SemanticBranch(nn.Module):
    def __init__(self, base=32):
        super(SemanticBranch, self).__init__()
        self.S12 = StemBlock(3, base)
        self.S3 = nn.Sequential(
            GELayer(base, base*2, stride=2),
            GELayer(base*2, base*2, stride=1),
        )
        self.S4 = nn.Sequential(
            GELayer(base*2, base*4, stride=2),
            GELayer(base*4, base*4, stride=1),
        )
        self.S5 = nn.Sequential(
            GELayer(base*4, base*4, stride=2),
            GELayer(base*4, base*4, stride=1),
            GELayer(base*4, base*4, stride=1),
            GELayer(base*4, base*4, stride=1),
        )
        self.CE = CEBlock(base*4, base*4)

    def forward(self, x):
        x = self.S12(x)
        x = self.S3(x)
        x = self.S4(x)
        x = self.S5(x)
        x = self.CE(x)
        return x

class BGA(nn.Module):
    def __init__(self, out_chan):
        super(BGA, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, groups=out_chan, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.Conv2d(out_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Sigmoid(),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, groups=out_chan, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.Conv2d(out_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        self.conv = ConvBNReLU(out_chan, out_chan, ks=3, stride=1, padding=1)

    def forward(self, x_d, x_s):
        # x_d: /8, x_s: /32
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)

        out1 = left1 * right1
        out2 = F.interpolate(left2 * right2, scale_factor=4, mode='bilinear', align_corners=False)
        out = out1 + out2
        out = self.conv(out)
        return out

class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, out_chan):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.head(x)
        return x

class BiSeNetV2(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=32):
        super(BiSeNetV2, self).__init__()
        # detail: /8, semantic: /32
        self.detail = DetailBranch(base=base_ch*2)
        self.semantic = SemanticBranch(base=base_ch)
        self.bga = BGA(out_chan=base_ch*4)
        self.head = SegmentHead(base_ch*4, base_ch*4, out_ch)

    def forward(self, x):
        size = x.size()[2:]
        feat_d = self.detail(x)
        feat_s = self.semantic(x)
        feat_fuse = self.bga(feat_d, feat_s)
        logits = self.head(feat_fuse)
        logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)
        return logits

if __name__ == "__main__":
    net = BiSeNetV2(in_ch=3, out_ch=1, base_ch=32)
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print("BiSeNetV2 out:", y.shape)
    tot_params = sum(p.numel() for p in net.parameters())
    print(f"Total params: {tot_params/1e6:.3f}M")
