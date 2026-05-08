# bisenetv1.py (fixed strides + channel alignment)
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SpatialPath(nn.Module):
    """Downsample to 1/8 with spatially rich features (C = base)."""
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.layer1 = _conv_bn_relu(in_ch, base, k=7, s=2, p=3)   # /2
        self.layer2 = _conv_bn_relu(base, base, k=3, s=2, p=1)    # /4
        self.layer3 = _conv_bn_relu(base, base, k=3, s=2, p=1)    # /8
        self.proj   = _conv_bn_relu(base, base, k=1, s=1, p=0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.proj(x)
        return x  # /8, C=base


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn_relu(cin, cout, k=3, s=stride, p=1),
            _conv_bn_relu(cout, cout, k=3, s=1, p=1),
        )
    def forward(self, x): return self.block(x)


class AttentionRefinement(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv(x)
        att = self.att(feat)
        return feat * att


class ContextPath(nn.Module):
    """
    Downsample chain to /32 with taps at /16 and /32.
    Output a context feature at /8 with C = 4*base for fusion.
    """
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        # Channel plan
        c2, c4, c8, c16, c32 = base, base*2, base*4, base*4, base*8
        # Stride schedule: /2, /4, /8, /16, /32
        self.s2  = BasicBlock(in_ch, c2,  stride=2)   # -> /2
        self.s4  = BasicBlock(c2,   c4,  stride=2)    # -> /4
        self.s8  = BasicBlock(c4,   c8,  stride=2)    # -> /8
        self.s16 = BasicBlock(c8,   c16, stride=2)    # -> /16  (tap)
        self.s32 = BasicBlock(c16,  c32, stride=2)    # -> /32  (tap)

        # ARMs
        self.arm16 = AttentionRefinement(c16)
        self.arm32 = AttentionRefinement(c32)

        # Global context on /32
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c32, c32, kernel_size=1, bias=False),
            nn.BatchNorm2d(c32),
            nn.ReLU(inplace=True),
        )

        # Reduce /32 channels down to match /16 channels before sum
        self.reduce32_to_16 = nn.Sequential(
            nn.Conv2d(c32, c16, kernel_size=1, bias=False),
            nn.BatchNorm2d(c16),
            nn.ReLU(inplace=True),
        )

        # After merging to /16, make a clean /8 context with C = 4*base (i.e., c16)
        self.post_merge_reduce = nn.Sequential(
            nn.Conv2d(c16, c16, kernel_size=1, bias=False),
            nn.BatchNorm2d(c16),
            nn.ReLU(inplace=True),
        )

        self.c_out = c16  # expose for Fusion module

    def forward(self, x):
        x = self.s2(x)      # /2
        x = self.s4(x)      # /4
        x = self.s8(x)      # /8
        x16 = self.s16(x)   # /16, C=c16 (= 4*base)
        x32 = self.s32(x16) # /32, C=c32 (= 8*base)

        gp = self.global_pool(x32)
        x32_arm = self.arm32(x32) + gp         # /32, C=c32
        x16_arm = self.arm16(x16)              # /16, C=c16

        x32_red = self.reduce32_to_16(x32_arm) # /32, C=c16
        x32_up  = F.interpolate(x32_red, size=x16_arm.shape[-2:], mode="bilinear", align_corners=False)  # -> /16

        context_16 = x16_arm + x32_up          # /16, C=c16
        context_16 = self.post_merge_reduce(context_16)  # stabilize channels
        context_8  = F.interpolate(context_16, scale_factor=2, mode="bilinear", align_corners=False)     # -> /8, C=c16

        return context_8, x16, x32  # main context at /8, C=c16 (= 4*base)


class FeatureFusionModule(nn.Module):
    """Fuse Spatial (/8, C=base) and Context (/8, C=4*base) with attention."""
    def __init__(self, in_sp, in_ct, out_ch):
        super().__init__()
        self.fuse = _conv_bn_relu(in_sp + in_ct, out_ch, k=1, s=1, p=0)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//4, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, sp, ct):
        # sp: /8, C=in_sp ; ct: /8, C=in_ct
        if sp.shape[-2:] != ct.shape[-2:]:
            ct = F.interpolate(ct, size=sp.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([sp, ct], dim=1)
        x = self.fuse(x)
        w = self.att(x)
        return x * w + x  # residual attention


class BiSeNetV1(nn.Module):
    """
    Minimal BiSeNet for binary segmentation.
    - SpatialPath: /8, C=base
    - ContextPath: /8, C=4*base
    - FFM: output C=4*base, then predict and upsample to stride 1
    """
    def __init__(self, in_ch=3, out_ch=1, base_ch=32):
        super().__init__()
        self.spatial = SpatialPath(in_ch=in_ch, base=base_ch)             # /8, C=base
        self.context = ContextPath(in_ch=in_ch, base=base_ch)             # /8, C=4*base
        self.ffm = FeatureFusionModule(in_sp=base_ch, in_ct=self.context.c_out, out_ch=base_ch*4)
        self.head = nn.Sequential(
            _conv_bn_relu(base_ch*4, base_ch*2, k=3, s=1, p=1),
            nn.Conv2d(base_ch*2, out_ch, kernel_size=1),
        )

    def forward(self, x):
        sp = self.spatial(x)           # /8, C=base
        ct, _, _ = self.context(x)     # /8, C=4*base
        fused = self.ffm(sp, ct)       # /8, C=4*base
        logits_1_8 = self.head(fused)  # /8
        logits = F.interpolate(logits_1_8, scale_factor=8, mode="bilinear", align_corners=False)
        return logits


if __name__ == "__main__":
    net = BiSeNetV1(in_ch=3, num_classes=1, base=32)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print("out:", y.shape)  # [1, 1, 256, 256]
    print("params (M):", sum(p.numel() for p in net.parameters())/1e6)
