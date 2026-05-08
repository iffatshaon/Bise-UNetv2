# ==============================================
# File: biseunetv2.py
# BiSe-UNet V2 (SP + CP) with:
#   • Ghost convolutions in Spatial Path and Decoder (fewer params, faster)
#   • ECA attention at key fusions (/16 and optional /8) for accuracy at ~0 params
#   • Same CP (BiSeNet-style) + pre-Stage-1 fusion at /16 + optional residual FFM at /8
#   • Optional PEG+LSA (off by default to favor FPS; can be enabled)
# ==============================================
from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Utils --------------------
def _conv_bn_relu(cin, cout, k=3, s=1, p=1, g=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, groups=g, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

class DSConv(nn.Module):
    """Depthwise-separable conv (fallback)."""
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, s, p, groups=cin, bias=False)
        self.dw_bn = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x

# -------------------- Ghost building blocks --------------------
class GhostConv(nn.Module):
    """
    Ghost convolution (from GhostNet):
      - intrinsic features by 1x1 conv → c' (= ceil(c_out / s))
      - cheap features by depthwise 3x3 on intrinsic → c_out - c'
      - concat → BN → ReLU
    """
    def __init__(self, cin, cout, s_ratio=2, relu=True):
        super().__init__()
        c_primary = int(math.ceil(cout / s_ratio))
        c_cheap = cout - c_primary
        self.primary = nn.Sequential(
            nn.Conv2d(cin, c_primary, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_primary),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(c_primary, c_cheap, kernel_size=3, stride=1, padding=1,
                      groups=c_primary, bias=False),
            nn.BatchNorm2d(c_cheap),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
    def forward(self, x):
        y = self.primary(x)
        if y.shape[1] == 0:
            return y
        z = self.cheap(y) if y.shape[1] > 0 else y
        return torch.cat([y, z], dim=1)

class GhostDSConv(nn.Module):
    """Ghosted depthwise-separable conv: DW → Ghost 1x1 expand."""
    def __init__(self, cin, cout, k=3, s=1, p=1, ghost_ratio=2):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, s, p, groups=cin, bias=False)
        self.dw_bn = nn.BatchNorm2d(cin)
        self.act = nn.ReLU(inplace=True)
        self.ghost_pw = GhostConv(cin, cout, s_ratio=ghost_ratio, relu=True)
    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.ghost_pw(x)
        return x

# -------------------- ECA attention (lightweight channel attention) --------------------
class ECA(nn.Module):
    """
    Efficient Channel Attention:
      - GlobalAvgPool → Conv1D(k) across channels → sigmoid gate
      - ~0 params (k small & derived from C) and ~0 compute.
    """
    def __init__(self, channels: int, k_size: int = None):
        super().__init__()
        if k_size is None:
            # heuristic from paper
            t = int(abs((math.log2(channels) + 1) / 2))
            k_size = t if t % 2 else t + 1
            k_size = max(3, k_size)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        y = x.mean(dim=(2, 3), keepdim=True)          # b,c,1,1
        y = self.conv(y.view(b, 1, c)).view(b, c, 1, 1)
        y = self.sig(y)
        return x * y

# -------------------- Spatial Path (Ghost) --------------------
class SpatialPath(nn.Module):
    """Shallow, high-res path: /2 → /4 → /8, then 1×1 ghost projection."""
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        # First big-kernel conv keeps it robust to blur
        self.layer1 = _conv_bn_relu(in_ch, base, k=7, s=2, p=3)  # /2
        # Replace the rest with cheaper Ghost
        self.layer2 = nn.Sequential(
            GhostDSConv(base, base, k=3, s=2, p=1),  # /4
        )
        self.layer3 = nn.Sequential(
            GhostDSConv(base, base, k=3, s=2, p=1),  # /8
        )
        self.proj   = nn.Sequential(
            nn.Conv2d(base, base, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )  # /8
        self.out_ch = base
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.proj(x)  # /8, C=base

# -------------------- Context Path (BiSeNet-style) --------------------
class AttentionRefinement(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.Sigmoid(),
        )
    def forward(self, x):
        feat = self.conv(x)
        return feat * self.att(feat)

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn_relu(cin, cout, 3, stride, 1),
            _conv_bn_relu(cout, cout, 3, 1, 1),
        )
    def forward(self, x): return self.block(x)

class ContextPath(nn.Module):
    """
    Downsample to /32 with taps at /4, /8, /16(refined), /32.
    Channel plan (base=b): c4=b*2, c8=b*4, c16=b*4, c32=b*8.
    (Optionally you could plug PEG+LSA here; we keep it off by default in v2.)
    """
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        b = base
        c2, c4, c8, c16, c32 = b, 2*b, 4*b, 4*b, 8*b
        self.s2  = BasicBlock(in_ch, c2,  stride=2)  # /2
        self.s4  = BasicBlock(c2,   c4,  stride=2)   # /4  (skip)
        self.s8  = BasicBlock(c4,   c8,  stride=2)   # /8  (skip)
        self.s16 = BasicBlock(c8,   c16, stride=2)   # /16
        self.s32 = BasicBlock(c16,  c32, stride=2)   # /32

        self.arm16 = AttentionRefinement(c16)
        self.arm32 = AttentionRefinement(c32)
        self.gp = nn.Sequential(  # global context on /32
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c32, c32, 1, bias=False),
            nn.BatchNorm2d(c32),
            nn.ReLU(inplace=True),
        )
        self.reduce32_to_16 = nn.Sequential(
            nn.Conv2d(c32, c16, 1, bias=False),
            nn.BatchNorm2d(c16),
            nn.ReLU(inplace=True),
        )
        self.post_merge_reduce = nn.Sequential(
            nn.Conv2d(c16, c16, 1, bias=False),
            nn.BatchNorm2d(c16),
            nn.ReLU(inplace=True),
        )

        self.skip_channels = (c4, c8, c16, c32)

    def forward(self, x):
        x2  = self.s2(x)
        x4  = self.s4(x2)
        x8  = self.s8(x4)
        x16 = self.s16(x8)
        x32 = self.s32(x16)

        x32_ref = self.arm32(x32) + self.gp(x32)
        x16_ref = self.arm16(x16)

        x32_red = self.reduce32_to_16(x32_ref)
        x32_up  = F.interpolate(x32_red, size=x16_ref.shape[-2:], mode="bilinear", align_corners=False)
        x16_ref = self.post_merge_reduce(x16_ref + x32_up)  # refined /16

        return x4, x8, x16_ref, x32

# -------------------- Pre-Stage-1 fusion at /16 + ECA --------------------
class PreStage1Fusion(nn.Module):
    """
    Fuse /16 inputs: [x16_ref, up32_like, down(SP/8)] → c16, with ECA gate.
    """
    def __init__(self, c_sp: int, c16: int, sp_down_ratio: int = 4, enable_residual: bool = True):
        super().__init__()
        c_sp_red = max(c16 // sp_down_ratio, 1)
        self.sp_down = nn.Sequential(
            DSConv(c_sp, c_sp, k=3, s=2, p=1),  # /8 -> /16
            nn.Conv2d(c_sp, c_sp_red, 1, bias=False),
            nn.BatchNorm2d(c_sp_red),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c16 + c16 + c_sp_red, c16, kernel_size=1, bias=False),
            nn.BatchNorm2d(c16),
            nn.ReLU(inplace=True),
        )
        self.eca = ECA(c16)
        self.enable_residual = enable_residual
        self.sp_gate = nn.Parameter(torch.tensor(0.8))  # learnable gate on SP
    def forward(self, x16_ref: torch.Tensor, up32_like: torch.Tensor, sp8: torch.Tensor):
        sp16 = self.sp_down(sp8)
        g = torch.sigmoid(self.sp_gate)
        x = torch.cat([x16_ref, up32_like, g * sp16], dim=1)  # /16
        out = self.fuse(x)
        out = self.eca(out)
        if self.enable_residual:
            out = out + x16_ref
        return out  # /16, c16

# -------------------- Optional /8 FFM (residual, ultralight) + ECA --------------------
class FFMLight(nn.Module):
    """Residual fusion at /8: x8' = x8 + β · φ([x8, reduce(sp8)]) + ECA"""
    def __init__(self, c8: int, c_sp: int, sp_reduce_ratio: int = 4, use_eca: bool = True):
        super().__init__()
        c_sp_red = max(c8 // sp_reduce_ratio, 1)
        self.sp_reduce = nn.Conv2d(c_sp, c_sp_red, 1, bias=False)
        self.fuse = nn.Sequential(GhostDSConv(c8 + c_sp_red, c8, k=3, s=1, p=1))
        self.ffm_gate = nn.Parameter(torch.tensor(0.5))  # learnable β
        self.eca = ECA(c8) if use_eca else nn.Identity()
    def forward(self, x8: torch.Tensor, sp8: torch.Tensor):
        sp_r = self.sp_reduce(sp8)
        y = torch.cat([x8, sp_r], dim=1)
        y = self.fuse(y)
        y = self.eca(y)
        beta = torch.sigmoid(self.ffm_gate)
        return x8 + beta * y  # residual

# -------------------- UNet decoder (Ghost) --------------------
class _UpGhost(nn.Module):
    """Concat skip then GhostDSConv fuse."""
    def __init__(self, cin, skip_ch, cout):
        super().__init__()
        self.fuse = GhostDSConv(cin + skip_ch, cout, k=3, s=1, p=1)
    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

class UNetDecoderGhost(nn.Module):
    def __init__(self, chs: Tuple[int,int,int,int], out_ch=1):
        super().__init__()
        c1_4, c1_8, c1_16, c1_32 = chs
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _UpGhost(c1_32, c1_16, c1_16)

        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _UpGhost(c1_16, c1_8,  c1_8)

        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = _UpGhost(c1_8,  c1_4,  c1_4)

        self.up4  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Conv2d(c1_4, out_ch, 1)

        # small ECA after stage outputs (helps without params)
        self.eca16 = ECA(c1_16)
        self.eca8  = ECA(c1_8)
        self.eca4  = ECA(c1_4)

    def forward(self, skips: List[torch.Tensor]):
        x1_4, x1_8, x1_16, x1_32 = skips

        y = self.up1(x1_32)      # /32 -> /16
        y = self.dec1(y, x1_16)  # fuse /16
        y = self.eca16(y)

        y = self.up2(y)          # /16 -> /8
        y = self.dec2(y, x1_8)   # fuse /8
        y = self.eca8(y)

        y = self.up3(y)          # /8 -> /4
        y = self.dec3(y, x1_4)   # fuse /4
        y = self.eca4(y)

        y = self.up4(y)          # /4 -> /2
        logits = self.head(y)    # /2
        logits = F.interpolate(logits, scale_factor=2, mode="bilinear", align_corners=False)  # -> /1
        return logits

# -------------------- Full model --------------------
class BiseUNetV2(nn.Module):
    """
    BiSe-UNet V2:
      - ContextPath (CP) provides: x4(/4), x8(/8), x16_ref(/16), x32(/32)
      - SpatialPath (SP) with Ghost convs provides sp8(/8)
      - Pre-Stage-1 fusion (/16) with ECA and gated SP
      - Optional /8 FFM_light (Ghost + ECA, residual)
      - Ghost-based UNet decoder
      - Optional PEG+LSA (disabled here to keep FPS high; can be layered in later)
    """
    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        base_ch=32,
        use_sp: bool = True,
        use_ffm_light: bool = True,
    ):
        super().__init__()
        self.cp = ContextPath(in_ch=in_ch, base=base_ch)
        self.use_sp = use_sp
        self.use_ffm_light = use_ffm_light

        if use_sp:
            self.sp = SpatialPath(in_ch=in_ch, base=base_ch)   # /8, C=base
            c8  = 4 * base_ch
            c16 = 4 * base_ch
            # Pre-stage-1 fusion at /16 (with ECA + gated SP)
            self.pre1 = PreStage1Fusion(c_sp=self.sp.out_ch, c16=c16, sp_down_ratio=4, enable_residual=True)
            # Optional ultralight residual FFM at /8
            if use_ffm_light:
                self.ffm = FFMLight(c8=c8, c_sp=self.sp.out_ch, sp_reduce_ratio=4, use_eca=True)
        else:
            self.pre1 = None
            self.ffm = None

        # Ghost-based decoder
        self.decoder = UNetDecoderGhost(self.cp.skip_channels, out_ch=out_ch)

    def forward(self, x):
        # CP skips
        x4, x8, x16_ref, x32 = self.cp(x)

        if self.use_sp:
            sp8 = self.sp(x)  # /8, C=base

            # As in v5/v6 notes: x16_ref already ~ arm16 + up(reduce(arm32)) stabilized.
            # We keep "up32_like=0" here; SP injection is the key early contributor.
            up32_like = torch.zeros_like(x16_ref)
            skip16 = self.pre1(x16_ref, up32_like, sp8) if self.pre1 is not None else x16_ref

            # Optional residual FFM at /8 (Ghost + ECA)
            x8p = self.ffm(x8, sp8) if (self.use_ffm_light and self.ffm is not None) else x8
        else:
            skip16 = x16_ref
            x8p = x8

        # Decode with [x4, x8p, skip16, x32]
        logits = self.decoder([x4, x8p, skip16, x32])
        return logits

# -------------------- helpers --------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# -------------------- quick check --------------------
if __name__ == "__main__":
    net = BiseUNetV2(in_ch=3, out_ch=1, base_ch=32, use_sp=True, use_ffm_light=True)
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print("out:", y.shape)  # [1,1,256,256]
    print("params (M):", count_params(net)/1e6)
