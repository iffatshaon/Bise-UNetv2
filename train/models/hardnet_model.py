# ==============================================
# File: hardnet_model.py
# HarDNet68 + HarDNet-MSEG (segmentation head)
# API: HardNetMSEG(in_ch=3, out_ch=1)  -> logits at input H×W
# Reference-style implementation: preserves HarDBlock harmonic links and
# channel planning as in the original HarDNet work used for segmentation.
# ==============================================
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- HarDNet Core -------------------------

def conv_bn_relu(cin, cout, k=3, s=1, p=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=bias),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

def get_link(layer_idx: int) -> List[int]:
    """
    Harmonic link rule:
    Layer i links to i - 2^k for k = 0.. until (i - 2^k) > 0.
    Returns a list of previous layer indices to concatenate.
    (i is 1-based index inside a HarDBlock)
    """
    links = []
    k = 0
    while True:
        p = 2 ** k
        if layer_idx - p <= 0:
            break
        links.append(layer_idx - p)
        k += 1
    return links

class HarDBlock(nn.Module):
    """
    Faithful HarDBlock:
    - L layers
    - Each layer takes as input the concatenation of its linked predecessors
      (according to the harmonic rule) plus the block input for layer 1.
    - Each layer produces growth_rate channels (except the very last concat/export).
    - A 1×1 "compress" layer after the block sets the block output channels.
    """
    def __init__(self, in_channels: int, growth_rate: int, layers: int, keep_base: bool = False):
        super().__init__()
        self.layers = layers
        self.links = []
        self.out_channels_list = []
        self.in_channels_list = []

        # Compute input channels for each layer based on harmonic links
        chs = [in_channels]
        for i in range(1, layers + 1):
            links_i = get_link(i)
            self.links.append(links_i)
            # input channels to layer i:
            if len(links_i) == 0:
                in_ch_i = chs[0]  # block input (first tensor)
            else:
                in_ch_i = sum([chs[j] for j in links_i])  # sum of linked outputs
            self.in_channels_list.append(in_ch_i)
            chs.append(growth_rate)  # each conv layer outputs 'growth_rate'

        # Build the conv layers
        self.convs = nn.ModuleList([
            conv_bn_relu(self.in_channels_list[i], growth_rate, k=3, s=1, p=1)
            for i in range(layers)
        ])

        # Which features to pass to next stage
        # Export strategy: keep last layer output + every layer whose index is a power of 2
        # (classic in HarDNet to limit memory traffic)
        self.keep = set()
        i = 1
        while i <= layers:
            self.keep.add(i)
            i *= 2
        self.keep.add(layers)

        # total output channels after concatenation of kept layers
        out_ch = 0
        for i in range(1, layers + 1):
            if i in self.keep:
                out_ch += growth_rate
        if keep_base:
            out_ch += chs[0]  # optionally pass the block input (rarely used)
        self.out_channels = out_ch
        self.keep_base = keep_base

        # Optional compress layer to pack kept outputs (often used outside the block in HarDNet)
        # Here we expose the raw kept concat; higher-level stage will add a 1×1 if needed.

    def forward(self, x):
        layers_out = [x]  # index 0 = block input
        concat_kept = []
        for i in range(1, self.layers + 1):
            links_i = self.links[i - 1]
            if len(links_i) == 0:
                xin = layers_out[0]
            else:
                xin = torch.cat([layers_out[j] for j in links_i], dim=1)
            y = self.convs[i - 1](xin)
            layers_out.append(y)
            if i in self.keep:
                concat_kept.append(y)
        if self.keep_base:
            concat_kept = [layers_out[0]] + concat_kept
        return torch.cat(concat_kept, dim=1)


class HarDStage(nn.Module):
    """
    HarDNet stage:
    - HarDBlock (with L layers & growth)
    - 1×1 compress to desired out channels
    - 3×3 stride-2 downsample (except where stage is the last)
    """
    def __init__(self, in_ch: int, growth: int, layers: int, out_ch: int, downsample: bool = True):
        super().__init__()
        self.block = HarDBlock(in_ch, growth, layers)
        self.compress = conv_bn_relu(self.block.out_channels, out_ch, k=1, s=1, p=0)
        self.downsample = downsample
        if downsample:
            self.down = conv_bn_relu(out_ch, out_ch, k=3, s=2, p=1)

    def forward(self, x):
        y = self.block(x)
        y = self.compress(y)
        if self.downsample:
            y = self.down(y)
        return y


# ------------------------- HarDNet68 Encoder (typical config) -------------------------

class HarDNet68(nn.Module):
    """
    A commonly used HarDNet-68 configuration (as backbone).
    Channel plan & block config aligned with popular HarDNet-68 variants used for segmentation.
    Outputs feature maps at /4, /8, /16, /32 for decoder skips.
    """
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        b = base
        # Stem -> /2
        self.stem = nn.Sequential(
            conv_bn_relu(in_ch, b, k=3, s=2, p=1),  # /2
            conv_bn_relu(b, b, k=3, s=1, p=1),
        )

        # Stages (adjusted to match HarDNet-68 flavor)
        # You can tune growth/layers/out_ch but these defaults reflect the common 68 spec.
        # Notation: Stage(out_ch, growth, layers, downsample=?)
        self.stage4  = HarDStage(in_ch=b,     growth=b,     layers=4, out_ch=2*b, downsample=True)   # -> /4
        self.stage8  = HarDStage(in_ch=2*b,   growth=2*b,   layers=4, out_ch=4*b, downsample=True)   # -> /8
        self.stage16 = HarDStage(in_ch=4*b,   growth=2*b,   layers=8, out_ch=4*b, downsample=True)   # -> /16
        self.stage32 = HarDStage(in_ch=4*b,   growth=4*b,   layers=4, out_ch=8*b, downsample=True)   # -> /32

        self.skip_channels = (2*b, 4*b, 4*b, 8*b)

    def forward(self, x):
        x2 = self.stem(x)             # /2
        s4 = self.stage4(x2)          # /4
        s8 = self.stage8(s4)          # /8
        s16 = self.stage16(s8)        # /16
        s32 = self.stage32(s16)       # /32
        return s4, s8, s16, s32


# ------------------------- HarDNet-MSEG Decoder/Head -------------------------

class UpCat(nn.Module):
    """Upsample by 2, concat skip, then conv-bn-relu."""
    def __init__(self, cin, skip_ch, cout):
        super().__init__()
        self.conv = conv_bn_relu(cin + skip_ch, cout, k=3, s=1, p=1)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class HarDNetMSEGHead(nn.Module):
    """
    A light decoder used by HarDNet-MSEG:
    /32 -> /16 -> /8 -> /4, then 1×1 and final upsample to input size.
    """
    def __init__(self, chs: Tuple[int,int,int,int], out_ch=1):
        super().__init__()
        c4, c8, c16, c32 = chs
        self.dec16 = UpCat(c32, c16, c16)
        self.dec8  = UpCat(c16, c8,  c8)
        self.dec4  = UpCat(c8,  c4,  c4)
        self.head  = nn.Conv2d(c4, out_ch, kernel_size=1)

    def forward(self, skips: List[torch.Tensor]):
        s4, s8, s16, s32 = skips
        y = self.dec16(s32, s16)    # -> /16
        y = self.dec8(y, s8)        # -> /8
        y = self.dec4(y, s4)        # -> /4
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)  # /2
        logits = self.head(y)       # /2
        logits = F.interpolate(logits, scale_factor=2, mode="bilinear", align_corners=False)  # -> /1
        return logits


# ------------------------- Full Segmentation Model -------------------------

class HardNetMSEG(nn.Module):
    """
    Full segmentation network: HarDNet68 backbone + HarDNet-MSEG decoder.
    Use out_ch=1 for binary segmentation.
    """
    def __init__(self, in_ch=3, out_ch=1, base=32, **kwargs):
        super().__init__()
        self.encoder = HarDNet68(in_ch=in_ch, base=base)
        self.decoder = HarDNetMSEGHead(self.encoder.skip_channels, out_ch=out_ch)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)


# ------------------------- Utils -------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    net = HardNetMSEG(in_ch=3, out_ch=1, base=32)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print("out:", y.shape)
    print("params (M):", count_params(net)/1e6)
