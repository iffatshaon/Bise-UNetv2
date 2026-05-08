# ==============================================
# File: rfdetr_seg.py
# RF-DETR Segmentation Model
#
# RF-DETR (Random Fourier Feature Detection Transformer) adapted for
# binary semantic segmentation (polyp detection).
#
# Architecture:
#   - ResNet-18 backbone (timm, features-only) → strides 1/8, 1/16, 1/32
#   - FPN neck: merges multi-scale features to a single 1/8 feature map (C=128)
#   - Positional encoding via Random Fourier Features (learnable RFF projection)
#   - Transformer encoder: 4 layers of multi-head self-attention on 1/8 tokens
#   - Single mask query: cross-attends to encoder tokens
#   - Mask head: pixel-wise dot product between query output and feature map
#   - Upsamples result to input resolution
#   - Output: raw logits (B, 1, H, W)
# ==============================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# -------------------- Building Blocks --------------------

def conv_bn_relu(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


# -------------------- Random Fourier Feature Positional Encoding --------------------

class RFFPositionalEncoding(nn.Module):
    """
    Random Fourier Feature positional encoding.
    Projects 2D spatial coordinates (x_norm, y_norm) to a D-dim embedding
    using learned (or fixed-random) frequency matrix, then maps to model dim.

    Advantages over sinusoidal PE:
      - Stationary kernel approximation (shift-invariance)
      - Can be re-used at any spatial resolution
      - Learnable scale for adapting to the data
    """
    def __init__(self, d_model: int, num_frequencies: int = 64):
        super().__init__()
        # Random frequency matrix B ~ N(0, 1); learned via backprop for better fit
        self.B = nn.Parameter(torch.randn(2, num_frequencies))  # (2, F)
        # Project [sin, cos] → d_model
        self.proj = nn.Linear(2 * num_frequencies, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Returns positional embedding of shape (1, h*w, d_model)."""
        # Build normalized coordinate grid in [-1, 1]
        ys = torch.linspace(-1, 1, h, device=device)
        xs = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (h*w, 2)

        # RFF projection
        proj = 2 * math.pi * coords @ self.B  # (h*w, F)
        rff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (h*w, 2F)

        pe = self.proj(rff)   # (h*w, d_model)
        pe = self.norm(pe)
        return pe.unsqueeze(0)  # (1, h*w, d_model)


# -------------------- FPN Neck --------------------

class FPNNeck(nn.Module):
    """
    Merges 3 feature scales (from strides 1/8, 1/16, 1/32) into a
    single feature map at 1/8 resolution with out_c channels.
    Uses top-down pathway with lateral connections.
    """
    def __init__(self, c3: int, c4: int, c5: int, out_c: int = 128):
        super().__init__()
        # Lateral 1x1 projections
        self.lat5 = nn.Conv2d(c5, out_c, 1, bias=False)
        self.lat4 = nn.Conv2d(c4, out_c, 1, bias=False)
        self.lat3 = nn.Conv2d(c3, out_c, 1, bias=False)
        # Smooth convs
        self.smooth4 = conv_bn_relu(out_c, out_c, k=3, s=1, p=1)
        self.smooth3 = conv_bn_relu(out_c, out_c, k=3, s=1, p=1)

    def forward(self, f3, f4, f5):
        p5 = self.lat5(f5)
        p4 = self.lat4(f4) + F.interpolate(p5, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        p4 = self.smooth4(p4)
        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.smooth3(p3)
        return p3  # (B, out_c, H/8, W/8)


# -------------------- Transformer Encoder --------------------

class TransformerEncoderLayer(nn.Module):
    """Standard pre-norm transformer encoder layer."""
    def __init__(self, d_model: int, nhead: int = 8, dim_ff: int = 512, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pe):
        # Add positional encoding for Q/K, but not V (standard practice)
        q = k = x + pe
        x2, _ = self.self_attn(q, k, x)
        x = self.norm1(x + x2)
        x = self.norm2(x + self.ff(x))
        return x


# -------------------- Mask Query Decoder --------------------

class MaskQueryDecoder(nn.Module):
    """
    Single mask query cross-attends to encoder tokens.
    Outputs a kernel vector used to produce a mask via dot product with the feature map.
    """
    def __init__(self, d_model: int, nhead: int = 8):
        super().__init__()
        self.query = nn.Embedding(1, d_model)       # Single learned mask query
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        # Project query output → mask kernel (same dim as feature map channels)
        self.kernel_proj = nn.Linear(d_model, d_model)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory: (B, S, d_model) encoder tokens
        Returns:
            kernel: (B, d_model, 1, 1) for pixel-wise dot product
        """
        B = memory.shape[0]
        q = self.query.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 1, d_model)
        out, _ = self.cross_attn(q, memory, memory)
        out = self.norm(out)  # (B, 1, d_model)
        kernel = self.kernel_proj(out.squeeze(1))   # (B, d_model)
        return kernel.unsqueeze(-1).unsqueeze(-1)   # (B, d_model, 1, 1)


# -------------------- Simple CNN fallback backbone --------------------

class SimpleCNNBackbone(nn.Module):
    """Lightweight CNN backbone when timm is not available."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.layer1 = nn.Sequential(
            conv_bn_relu(in_ch, 32, k=3, s=2, p=1),
            conv_bn_relu(32, 64, k=3, s=1, p=1),
        )  # 1/2
        self.layer2 = nn.Sequential(
            conv_bn_relu(64, 64, k=3, s=2, p=1),
            conv_bn_relu(64, 64, k=3, s=1, p=1),
        )  # 1/4
        self.layer3 = nn.Sequential(
            conv_bn_relu(64, 128, k=3, s=2, p=1),
            conv_bn_relu(128, 128, k=3, s=1, p=1),
        )  # 1/8
        self.layer4 = nn.Sequential(
            conv_bn_relu(128, 256, k=3, s=2, p=1),
            conv_bn_relu(256, 256, k=3, s=1, p=1),
        )  # 1/16
        self.layer5 = nn.Sequential(
            conv_bn_relu(256, 256, k=3, s=2, p=1),
            conv_bn_relu(256, 256, k=3, s=1, p=1),
        )  # 1/32
        self.feature_channels = [128, 256, 256]  # c3, c4, c5

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return [c3, c4, c5]


# -------------------- Full RF-DETR Segmentation Model --------------------

class RFDetrSeg(nn.Module):
    """
    RF-DETR for binary segmentation.

    Pipeline:
        Input (B,3,H,W)
          → Backbone (ResNet-18 or CNN fallback): features at 1/8, 1/16, 1/32
          → FPN neck: merges to single 1/8 map (B, d_model, H/8, W/8)
          → Flatten tokens + RFF positional encoding
          → Transformer encoder (N layers of MHSA)
          → Single mask query cross-attends to encoder memory
          → Mask = dot_product(kernel, feature_map) → (B, 1, H/8, W/8)
          → Upsample to (B, 1, H, W)
    """
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        d_model: int = 128,
        num_enc_layers: int = 4,
        nhead: int = 8,
        num_frequencies: int = 64,
        backbone: str = 'resnet18',
        use_timm: bool = False,  # Set False for CUDA sm_61 compatibility
    ):
        super().__init__()
        self.d_model = d_model

        # --- Backbone ---
        if use_timm and TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                backbone,
                features_only=True,
                out_indices=(2, 3, 4),  # strides 1/8, 1/16, 1/32
                pretrained=True,
            )
            c3, c4, c5 = self.backbone.feature_info.channels()
        else:
            # Use custom CNN backbone for compatibility with CUDA < sm_70
            self.backbone = SimpleCNNBackbone(in_ch=in_ch)
            c3, c4, c5 = self.backbone.feature_channels

        # --- FPN neck → d_model channels at 1/8 ---
        self.fpn = FPNNeck(c3, c4, c5, out_c=d_model)

        # --- Input projection to ensure exactly d_model channels ---
        self.input_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # --- Positional encoding (RFF) ---
        self.pos_enc = RFFPositionalEncoding(d_model, num_frequencies=num_frequencies)

        # --- Transformer encoder ---
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead=nhead, dim_ff=d_model * 4)
            for _ in range(num_enc_layers)
        ])

        # --- Mask query decoder ---
        self.decoder = MaskQueryDecoder(d_model, nhead=nhead)

        # --- Output head: BN + sigmoid logit projection ---
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        # 1. Backbone
        feats = self.backbone(x)
        f3, f4, f5 = feats

        # 2. FPN → (B, d_model, H/8, W/8)
        feat = self.fpn(f3, f4, f5)
        feat = self.input_proj(feat)

        h, w = feat.shape[-2:]

        # 3. Flatten to tokens (B, S, d_model)
        tokens = feat.flatten(2).permute(0, 2, 1)  # (B, h*w, d_model)

        # 4. RFF positional encoding
        pe = self.pos_enc(h, w, x.device).expand(B, -1, -1)  # (B, h*w, d_model)

        # 5. Transformer encoder (pre-norm, adds PE to Q/K each layer)
        memory = tokens
        for layer in self.encoder:
            memory = layer(memory, pe)

        # 6. Mask query cross-attention → kernel (B, d_model, 1, 1)
        kernel = self.decoder(memory)

        # 7. Pixel-wise dot product with feature map → (B, 1, h, w)
        # mask = sum over d_model dimension
        mask_small = (feat * kernel).sum(dim=1, keepdim=True)  # (B, 1, h, w)

        # 8. Upsample to input resolution
        logits = F.interpolate(mask_small, size=(H, W), mode='bilinear', align_corners=False)

        return logits  # (B, 1, H, W) raw logits


# -------------------- Quick check --------------------
if __name__ == '__main__':
    net = RFDetrSeg(in_ch=3, out_ch=1, d_model=128, num_enc_layers=4)
    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print('RF-DETR out:', y.shape)  # expected: (2, 1, 256, 256)
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'Params: {n_params:.2f}M')
