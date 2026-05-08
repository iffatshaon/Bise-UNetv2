# ==============================================
# File: ducknet_model.py
# DUCK-Net (Scientific Reports 2023) for binary segmentation
# API: DuckNet(in_ch=3, out_ch=1, base=32)
# Output: logits (same H×W as input)
#
# References:
# - Dumitru et al., "Using DUCK-Net for Polyp Image Segmentation", Sci. Reports 2023
#   (open-access PMC) https://pmc.ncbi.nlm.nih.gov/articles/PMC10276013/
# - Authors' repository: https://github.com/RazvanDu/DUCK-Net
# ==============================================
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------- Helpers -----------------------
def _conv_bn_relu(cin, cout, k=3, s=1, p=1, bias=False, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=bias, groups=groups),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

def _res_down(cin, cout):
    """
    Residual downsampling used in DUCK-Net encoder stages:
    - main branch: 3x3 s=2 + 3x3 s=1 (BN+ReLU in-between)
    - skip branch: 1x1 s=2
    - output: sum(main, skip)
    """
    main = nn.Sequential(
        nn.Conv2d(cin, cout, 3, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
    )
    skip = nn.Sequential(nn.Conv2d(cin, cout, 1, 2, 0, bias=False), nn.BatchNorm2d(cout))
    return main, skip

# ----------------------- DUCK Block -----------------------
class DUCKBlock(nn.Module):
    """
    DUCK-Block (paper’s custom block):
    - Multi-branch convs with residual-style fusion to enrich context while keeping params moderate.
    - This implementation follows the common (paper/repo) pattern:
        * 3x3 conv stream
        * a parallel 3x3 (or 1x1) stream
        * concatenate + 1x1 fuse
      (keeps compute reasonable and captures local + slightly wider context)
    """
    def __init__(self, cin: int, cout: int, mid_mul: float = 0.5):
        super().__init__()
        mid = max(int(cout * mid_mul), 1)

        # Branch A: 3x3 -> 3x3
        self.a1 = _conv_bn_relu(cin, mid, 3, 1, 1)
        self.a2 = _conv_bn_relu(mid, mid, 3, 1, 1)

        # Branch B: 1x1 -> 3x3
        self.b1 = _conv_bn_relu(cin, mid, 1, 1, 0)
        self.b2 = _conv_bn_relu(mid, mid, 3, 1, 1)

        # Fuse: concat(A, B) -> 1x1 -> BN+ReLU
        self.fuse = _conv_bn_relu(mid + mid, cout, 1, 1, 0)

        # Local residual (if channel match)
        self.shortcut = None
        if cin == cout:
            self.shortcut = nn.Identity()

    def forward(self, x):
        a = self.a2(self.a1(x))
        b = self.b2(self.b1(x))
        y = self.fuse(torch.cat([a, b], dim=1))
        if self.shortcut is not None:
            y = F.relu_(y + x)
        return y

# ----------------------- Encoder -----------------------
class DUCKEncoder(nn.Module):
    """
    Encoder: stem -> (stage4, stage8, stage16, stage32)
    Each stage starts with residual downsampling, then DUCK-Blocks.
    Skip taps are taken at /4, /8, /16, /32 for the decoder.
    Channel plan mirrors the paper’s small/standard configs:
      c2=b, c4=2b, c8=4b, c16=4b, c32=8b  (balanced for 256×256 training)
    """
    def __init__(self, in_ch=3, base=32, blocks_per_stage=(1, 2, 2, 2)):
        super().__init__()
        b = base
        c2, c4, c8, c16, c32 = b, 2*b, 4*b, 4*b, 8*b

        # Stem (no downsample here; we keep H/2 after stem to match many pipelines)
        self.stem = nn.Sequential(
            _conv_bn_relu(in_ch, b, 3, 2, 1),  # /2
            _conv_bn_relu(b, b, 3, 1, 1),
        )

        # Stage /4
        self.s4_main, self.s4_skip = _res_down(c2, c4)
        self.s4_blocks = nn.Sequential(*[DUCKBlock(c4, c4) for _ in range(blocks_per_stage[0])])

        # Stage /8
        self.s8_main, self.s8_skip = _res_down(c4, c8)
        self.s8_blocks = nn.Sequential(*[DUCKBlock(c8, c8) for _ in range(blocks_per_stage[1])])

        # Stage /16
        self.s16_main, self.s16_skip = _res_down(c8, c16)
        self.s16_blocks = nn.Sequential(*[DUCKBlock(c16, c16) for _ in range(blocks_per_stage[2])])

        # Stage /32
        self.s32_main, self.s32_skip = _res_down(c16, c32)
        self.s32_blocks = nn.Sequential(*[DUCKBlock(c32, c32) for _ in range(blocks_per_stage[3])])

        self.skip_channels = (c4, c8, c16, c32)

    def forward(self, x):
        x2 = self.stem(x)  # /2, C=b

        # /4
        y = self.s4_main(x2) + self.s4_skip(x2)
        y = F.relu_(y)
        s4 = self.s4_blocks(y)

        # /8
        y = self.s8_main(s4) + self.s8_skip(s4)
        y = F.relu_(y)
        s8 = self.s8_blocks(y)

        # /16
        y = self.s16_main(s8) + self.s16_skip(s8)
        y = F.relu_(y)
        s16 = self.s16_blocks(y)

        # /32
        y = self.s32_main(s16) + self.s32_skip(s16)
        y = F.relu_(y)
        s32 = self.s32_blocks(y)

        return s4, s8, s16, s32

# ----------------------- Decoder / Head -----------------------
class _UpCat(nn.Module):
    """Upsample by 2, concat skip, conv-bn-relu."""
    def __init__(self, cin: int, skip_ch: int, cout: int):
        super().__init__()
        self.conv = _conv_bn_relu(cin + skip_ch, cout, 3, 1, 1)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DUCKDecoder(nn.Module):
    """
    Light U-Net-style decoder used by the DUCK-Net authors:
      /32 -> /16 -> /8 -> /4 ; then up to /2 and 1x1 head, then final up to /1
    """
    def __init__(self, chs: Tuple[int,int,int,int], out_ch=1):
        super().__init__()
        c4, c8, c16, c32 = chs
        self.dec16 = _UpCat(c32, c16, c16)
        self.dec8  = _UpCat(c16, c8,  c8)
        self.dec4  = _UpCat(c8,  c4,  c4)
        self.up2   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head  = nn.Conv2d(c4, out_ch, 1)

    def forward(self, skips: List[torch.Tensor]):
        s4, s8, s16, s32 = skips
        y = self.dec16(s32, s16)   # -> /16
        y = self.dec8(y, s8)       # -> /8
        y = self.dec4(y, s4)       # -> /4
        y = self.up2(y)            # -> /2
        logits = self.head(y)      # /2
        logits = F.interpolate(logits, scale_factor=2, mode="bilinear", align_corners=False)  # -> /1
        return logits

# ----------------------- Full Model -----------------------
class DuckNet(nn.Module):
    """
    Full DUCK-Net: DUCK-Encoder + light U-Net decoder head
    Args:
      in_ch: input channels
      out_ch: #classes (1 for binary)
      base: base channels (paper demonstrates 17/34 filters variants)
      blocks_per_stage: number of DUCKBlocks per stage (default: (1,2,2,2))
    """
    def __init__(self, in_ch=3, out_ch=1, base_ch=32, blocks_per_stage=(1,2,2,2)):
        super().__init__()
        self.encoder = DUCKEncoder(in_ch=in_ch, base=base_ch, blocks_per_stage=blocks_per_stage)
        self.decoder = DUCKDecoder(self.encoder.skip_channels, out_ch=out_ch)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

# ----------------------- Utils -----------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    net = DuckNet(in_ch=3, out_ch=1, base_ch=32)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print("out:", y.shape)
    print("params (M):", count_params(net)/1e6)
