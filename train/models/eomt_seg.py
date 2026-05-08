# ==============================================
# File: eomt_seg.py
# EoMT Segmentation Model
#
# EoMT (Efficiency-oriented Mask Transformer) adapted for
# binary semantic segmentation (polyp detection).
#
# Architecture:
#   - EfficientNet-B0 backbone (timm, features-only) → strides 1/4, 1/8, 1/16, 1/32
#     (falls back to lightweight CNN if timm unavailable)
#   - FPN decoder: multi-scale feature fusion to 1/4 resolution (C=128)
#   - N_TOKENS=4 learned mask tokens
#   - Efficient cross-attention: mask tokens attend to feature map tokens (windowed)
#   - Per-token mask kernels → pixel-wise dot product with FPN output → N masks
#   - Auxiliary class head (lightweight): selects best mask token
#   - Merges N masks into a single binary mask via weighted sum
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


def conv_bn(cin, cout, k=1, s=1, p=0):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        nn.BatchNorm2d(cout),
    )


# -------------------- Lightweight CNN Fallback Backbone --------------------

class LightCNNBackbone(nn.Module):
    """Fallback backbone when timm is not available."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.stem = conv_bn_relu(in_ch, 32, k=3, s=2, p=1)  # 1/2
        self.s2 = nn.Sequential(
            conv_bn_relu(32, 48, k=3, s=2, p=1),   # 1/4
            conv_bn_relu(48, 48, k=3, s=1, p=1),
        )
        self.s3 = nn.Sequential(
            conv_bn_relu(48, 96, k=3, s=2, p=1),   # 1/8
            conv_bn_relu(96, 96, k=3, s=1, p=1),
        )
        self.s4 = nn.Sequential(
            conv_bn_relu(96, 160, k=3, s=2, p=1),  # 1/16
            conv_bn_relu(160, 160, k=3, s=1, p=1),
        )
        self.s5 = nn.Sequential(
            conv_bn_relu(160, 256, k=3, s=2, p=1), # 1/32
            conv_bn_relu(256, 256, k=3, s=1, p=1),
        )
        # Match typical EfficientNet-B0 channel widths: [24, 40, 112, 320]
        self.feature_channels = [48, 96, 160, 256]

    def forward(self, x):
        x = self.stem(x)
        f2 = self.s2(x)   # 1/4
        f3 = self.s3(f2)  # 1/8
        f4 = self.s4(f3)  # 1/16
        f5 = self.s5(f4)  # 1/32
        return [f2, f3, f4, f5]


# -------------------- FPN Decoder (to 1/4 resolution) --------------------

class FPNDecoder(nn.Module):
    """
    Top-down FPN that fuses 4 feature levels (1/4, 1/8, 1/16, 1/32) into a
    single high-resolution map at 1/4 with C=out_c channels.
    """
    def __init__(self, c2: int, c3: int, c4: int, c5: int, out_c: int = 128):
        super().__init__()
        # Lateral projections
        self.lat5 = nn.Conv2d(c5, out_c, 1, bias=False)
        self.lat4 = nn.Conv2d(c4, out_c, 1, bias=False)
        self.lat3 = nn.Conv2d(c3, out_c, 1, bias=False)
        self.lat2 = nn.Conv2d(c2, out_c, 1, bias=False)
        # Smooth convs
        self.smooth4 = conv_bn_relu(out_c, out_c, k=3, s=1, p=1)
        self.smooth3 = conv_bn_relu(out_c, out_c, k=3, s=1, p=1)
        self.smooth2 = conv_bn_relu(out_c, out_c, k=3, s=1, p=1)

    def forward(self, f2, f3, f4, f5):
        p5 = self.lat5(f5)
        p4 = self.lat4(f4) + F.interpolate(p5, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        p4 = self.smooth4(p4)
        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.smooth3(p3)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.smooth2(p2)
        return p2  # (B, out_c, H/4, W/4)


# -------------------- Efficient Cross-Attention --------------------

class EfficientCrossAttention(nn.Module):
    """
    Efficient cross-attention for mask tokens:
    - Mask tokens (queries) attend to flattened feature map tokens (keys/values)
    - Uses pooled feature for key/value to reduce sequence length
    - Spatial pooling factor controls the compression ratio
    """
    def __init__(self, d_model: int, n_tokens: int, nhead: int = 4, pool_factor: int = 4):
        super().__init__()
        self.pool_factor = pool_factor
        self.pool = nn.AvgPool2d(kernel_size=pool_factor, stride=pool_factor, padding=0)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.norm_tokens = nn.LayerNorm(d_model)
        self.norm_feat = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, mask_tokens: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask_tokens: (B, N, d_model)
            feat: (B, d_model, h, w)
        Returns:
            updated mask tokens: (B, N, d_model)
        """
        # Pool the feature map to reduce token count
        feat_pooled = self.pool(feat)  # (B, d_model, h/p, w/p)
        B, C, hp, wp = feat_pooled.shape
        kv = feat_pooled.flatten(2).permute(0, 2, 1)  # (B, hp*wp, d_model)
        kv = self.norm_feat(kv)

        q = self.norm_tokens(mask_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv)

        # Residual + FF
        mask_tokens = mask_tokens + attn_out
        mask_tokens = self.norm_out(mask_tokens + self.ff(mask_tokens))
        return mask_tokens


# -------------------- Mask Kernel Projection --------------------

class MaskKernelHead(nn.Module):
    """
    Maps each mask token to a kernel vector of dimension d_model.
    Also produces a binary class score per token (for token selection / weighting).
    """
    def __init__(self, d_model: int, n_tokens: int):
        super().__init__()
        self.kernel_proj = nn.Linear(d_model, d_model)
        self.class_head = nn.Linear(d_model, 1)   # binary: foreground score per token

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: (B, N, d_model)
        Returns:
            kernels: (B, N, d_model)
            scores:  (B, N)
        """
        kernels = self.kernel_proj(tokens)                   # (B, N, d_model)
        scores = self.class_head(tokens).squeeze(-1)         # (B, N)
        return kernels, scores


# -------------------- Full EoMT Segmentation Model --------------------

class EoMTSeg(nn.Module):
    """
    EoMT (Efficiency-oriented Mask Transformer) for binary segmentation.

    Pipeline:
        Input (B,3,H,W)
          → EfficientNet-B0 backbone (or CNN fallback): 4 feature levels
          → FPN decoder: merge to (B, d_model, H/4, W/4)
          → N learned mask tokens
          → Efficient cross-attention (N rounds): tokens attend to pooled features
          → Kernel projection: each token → d_kernel vector
          → Per-token mask: dot_product(kernel, feature_map) → (B, N, H/4, W/4)
          → Weighted sum using softmax(class_scores) → (B, 1, H/4, W/4)
          → Upsample to (B, 1, H, W)
    """
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        d_model: int = 128,
        n_tokens: int = 4,
        n_attn_layers: int = 3,
        nhead: int = 4,
        pool_factor: int = 4,
        backbone: str = 'efficientnet_b0',
        use_timm: bool = False,  # Set False for CUDA sm_61 compatibility
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model

        # --- Backbone ---
        if use_timm and TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                backbone,
                features_only=True,
                out_indices=(1, 2, 3, 4),   # strides 1/4, 1/8, 1/16, 1/32
                pretrained=True,
            )
            feat_chs = self.backbone.feature_info.channels()  # [c2, c3, c4, c5]
            c2, c3, c4, c5 = feat_chs
        else:
            # Use custom CNN backbone for compatibility with CUDA < sm_70
            self.backbone = LightCNNBackbone(in_ch=in_ch)
            c2, c3, c4, c5 = self.backbone.feature_channels

        # --- FPN decoder to 1/4 ---
        self.fpn = FPNDecoder(c2, c3, c4, c5, out_c=d_model)

        # --- Input refinement ---
        self.feat_refine = conv_bn_relu(d_model, d_model, k=3, s=1, p=1)

        # --- Learned mask tokens ---
        self.mask_tokens = nn.Embedding(n_tokens, d_model)

        # --- Efficient cross-attention layers ---
        self.attn_layers = nn.ModuleList([
            EfficientCrossAttention(d_model, n_tokens, nhead=nhead, pool_factor=pool_factor)
            for _ in range(n_attn_layers)
        ])

        # --- Mask kernel head ---
        self.mask_head = MaskKernelHead(d_model, n_tokens)

        # --- Mask feature projection (pixel features → d_model for dot product) ---
        self.mask_feat = nn.Conv2d(d_model, d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        # 1. Backbone features
        feats = self.backbone(x)
        f2, f3, f4, f5 = feats

        # 2. FPN across all scales → (B, d_model, H/4, W/4)
        fmap = self.fpn(f2, f3, f4, f5)
        fmap = self.feat_refine(fmap)  # (B, d_model, H/4, W/4)

        # 3. Initialize mask tokens (B, N, d_model)
        tokens = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)

        # 4. Efficient cross-attention (mask tokens ← feature map)
        for layer in self.attn_layers:
            tokens = layer(tokens, fmap)

        # 5. Kernel + class score projection
        kernels, scores = self.mask_head(tokens)
        # kernels: (B, N, d_model), scores: (B, N)

        # 6. Project feature map for dot product
        fmap_proj = self.mask_feat(fmap)  # (B, d_model, H/4, W/4)

        # 7. Per-token masks via dot product: (B, N, H/4, W/4)
        fmap_flat = fmap_proj.flatten(2)  # (B, d_model, HW/16)
        # kernels: (B, N, d_model) → matmul with feat → (B, N, HW/16)
        masks_flat = torch.bmm(kernels, fmap_flat)  # (B, N, H/4*W/4)
        h4, w4 = fmap.shape[-2:]
        masks = masks_flat.reshape(B, self.n_tokens, h4, w4)  # (B, N, H/4, W/4)

        # 8. Weighted sum across tokens using softmax scores → (B, 1, H/4, W/4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        merged = (masks * weights).sum(dim=1, keepdim=True)  # (B, 1, H/4, W/4)

        # 9. Upsample to input size
        logits = F.interpolate(merged, size=(H, W), mode='bilinear', align_corners=False)

        return logits  # (B, 1, H, W) raw logits


# -------------------- Quick check --------------------
if __name__ == '__main__':
    net = EoMTSeg(in_ch=3, out_ch=1, d_model=128, n_tokens=4, n_attn_layers=3)
    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print('EoMT out:', y.shape)  # expected: (2, 1, 256, 256)
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'Params: {n_params:.2f}M')
