# ==============================================
# File: viz_compare_from_dataset.py
# Build a tall comparison panel from TEST split using dataset_prep.py
# Columns: [Image | GT | model1 | model2 | ...]
# Add/remove/reorder models in MODEL_SPECS below.
# ==============================================
import os
import random
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset_prep import (
    set_seed, list_image_mask_pairs, split_pairs,
    normalize_image, resize_pair  # we'll reuse your transforms
)

# ------------------- USER CONFIG -------------------
IMG_DIR = r"/media/iffat/DataDrive/dataset/kvasir-seg/Kvasir-SEG/images"
MSK_DIR = r"/media/iffat/DataDrive/dataset/kvasir-seg/Kvasir-SEG/masks"
OUT_PATH = "./compare_panel_test.png"

# Panel & inference
IMG_SIZE = 256          # match your training
N_SAMPLES = 3           # how many test images to visualize
RANDOM_PICK = False      # True → random N from test split; False → first N
THRESH = 0.5
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_MIXED = (DEVICE == "cuda")

# Checkpoint base dir (optional convenience)
CKPT_DIR = "."

# ------------- Model registry (edit freely) -------------
# Each item: (display_name, loader_fn)
#   loader_fn must return a CUDA/CPU-placed torch.nn.Module in eval() mode
#   that maps (B,3,H,W)->(B,1,H,W) logits.
def load_unet():
    from unet_model import UNet
    ckpt = os.path.join(CKPT_DIR, "runs_unet/best.pt")
    m = UNet(in_ch=3, out_ch=1, base_ch=32)
    st = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(st["model"] if "model" in st else st, strict=True)
    return m.eval().to(DEVICE)

def load_bisenet():
    # choose your variant (v1/v3/v4). Example uses v3 (lite decoder).
    from bisenetv1 import BiSeNetV1
    ckpt = os.path.join(CKPT_DIR, "runs_bisenet/best.pt")
    m = BiSeNetV1(in_ch=3, out_ch=1, base_ch=32)
    st = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(st["model"] if "model" in st else st, strict=True)
    return m.eval().to(DEVICE)

def load_hardnet():
    from hardnet_model import HardNetMSEG
    ckpt = os.path.join(CKPT_DIR, "runs_hardnet_mseg/best.pt")
    m = HardNetMSEG(in_ch=3, out_ch=1, base=32)
    st = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(st["model"] if "model" in st else st, strict=True)
    return m.eval().to(DEVICE)

def load_ducknet():
    from ducknet_model import DuckNet
    ckpt = os.path.join(CKPT_DIR, "runs_ducknet/best.pt")
    m = DuckNet(in_ch=3, out_ch=1, base_ch=32)
    st = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(st["model"] if "model" in st else st, strict=True)
    return m.eval().to(DEVICE)

def load_biseunet_custom():
    # Your SP+CP hybrid (bisenetv4.BiseUNet) or whatever name you used
    from biseunetv7 import BiseUNetV7
    ckpt = os.path.join(CKPT_DIR, "runs_biseunetv7/best.pt")
    m = BiseUNetV7(in_ch=3, out_ch=1, base_ch=32)
    st = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(st["model"] if "model" in st else st, strict=True)
    return m.eval().to(DEVICE)

# Choose the models/columns you want to show (order matters)
MODEL_SPECS = [
    ("UNet",     load_unet),
    ("BiSeNet",  load_bisenet),
    ("HarDNet",  load_hardnet),
    # ("DuckNet",  load_ducknet),
    ("BiseUNet", load_biseunet_custom),
]

# ------------------- Helpers -------------------
def to_tensor_from_path(img_path: str, size: int):
    """Reads an RGB image path -> (1,3,H,W) float32 normalized like dataset_prep."""
    import cv2
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_bgr, _ = resize_pair(img_bgr, np.zeros((1,1), np.uint8), size)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = normalize_image(img_rgb)  # 3xHxW
    x = torch.from_numpy(chw).unsqueeze(0)      # 1x3xHxW
    return x

def load_mask_as_L(msk_path: str, size: int):
    """Reads a mask path -> PIL L image resized to size×size (nearest)."""
    import cv2
    m = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    _, m = resize_pair(np.zeros((1,1,3), np.uint8), m, size)
    return Image.fromarray(m, mode="L")

@torch.no_grad()
def predict(model, x):
    x = x.to(DEVICE, non_blocking=True)
    with torch.amp.autocast(device_type=("cuda" if DEVICE=="cuda" else "cpu"), enabled=USE_MIXED):
        logits = model(x)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits)[0,0].clamp_(0,1)  # (H,W)
    return (prob > THRESH).float().cpu().numpy()

def npmask_to_pil(mask01):
    arr = (mask01 * 255).astype("uint8")
    return Image.fromarray(arr, mode="L")

def _measure_text(draw, text, font):
    # Pillow >=10: use textbbox; fallback to font.getsize for older versions
    try:
        # returns (left, top, right, bottom)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            # last resort
            return (len(text) * 8, 14)

def draw_header(widths, height, titles, bg=(0, 0, 0)):
    W = sum(widths)
    header = Image.new("RGB", (W, height), bg)
    draw = ImageDraw.Draw(header)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 50)
    except Exception:
        font = ImageFont.load_default()

    x = 0
    for w, title in zip(widths, titles):
        tw, th = _measure_text(draw, title, font)
        draw.text((x + (w - tw)//2, (height - th)//2), title, fill=(255, 255, 255), font=font)
        x += w
    return header


def pad(im, p=1, color=None):
    """
    Pads an image on all sides by p pixels.
    - For RGB images, uses an RGB tuple color (default black).
    - For L/1 images, uses a single int color (default 0).
    """
    W, H = im.size
    # choose sensible default if not provided
    if color is None:
        color = (0, 0, 0) if im.mode not in ("L", "1") else 0

    # normalize color type to match mode
    if im.mode in ("L", "1"):
        # Pillow wants an int (0..255)
        if isinstance(color, tuple):
            color = int(color[0])
        else:
            color = int(color)
    else:
        # ensure tuple for RGB/RGBA, etc.
        if not isinstance(color, tuple):
            color = (int(color), int(color), int(color))

    out = Image.new(im.mode, (W + 2 * p, H + 2 * p), color)
    out.paste(im, (p, p))
    return out


# ------------------- Main -------------------
def main():
    set_seed(SEED)

    # 1) collect pairs and split -> get test set only
    pairs = list_image_mask_pairs(IMG_DIR, MSK_DIR)
    _, _, test_pairs = split_pairs(pairs, splits=(0.7,0.15,0.15), seed=SEED)

    if not test_pairs:
        raise RuntimeError("No test pairs found. Check IMG_DIR/MSK_DIR paths.")

    # pick N samples
    if RANDOM_PICK:
        random.seed(SEED)
        test_pairs = random.sample(test_pairs, k=min(N_SAMPLES, len(test_pairs)))
    else:
        test_pairs = test_pairs[:N_SAMPLES]

    # 2) load requested models
    models = []
    for name, loader in MODEL_SPECS:
        try:
            m = loader()
            models.append((name, m))
        except Exception as e:
            print(f"[warn] Skipping {name}: {e}")

    # 3) build rows: [Image | GT | model1 | model2 | ...]
    cell_w = IMG_SIZE
    cell_h = IMG_SIZE
    header_h = 72
    col_titles = ["Image", "GT"] + [n for (n, _) in models]
    col_count = len(col_titles)
    col_widths = [cell_w] * col_count

    # header row
    rows = [draw_header(col_widths, header_h, col_titles)]

    for (ip, mp) in test_pairs:
        # input & GT
        x = to_tensor_from_path(ip, IMG_SIZE)        # (1,3,256,256) normalized
        img_disp = Image.open(ip).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        gt_disp  = load_mask_as_L(mp, IMG_SIZE)

        # predictions
        cells = [img_disp, gt_disp]
        for name, model in models:
            pmask = predict(model, x)                # (H,W) float {0,1}
            cells.append(npmask_to_pil(pmask))

        # ensure same size + thin border
        cells = [pad(c.resize((cell_w, cell_h), Image.NEAREST), p=1) for c in cells]

        # concat horizontally
        row_w = sum(c.size[0] for c in cells)
        row_im = Image.new("RGB", (row_w, cells[0].size[1]), (255,255,255))
        xo = 0
        for c in cells:
            row_im.paste(c, (xo, 0))
            xo += c.size[0]
        rows.append(row_im)

    # 4) stack all rows vertically
    panel_w = max(r.size[0] for r in rows)
    panel_h = sum(r.size[1] for r in rows)
    panel = Image.new("RGB", (panel_w, panel_h), (255,255,255))
    yo = 0
    for r in rows:
        panel.paste(r, (0, yo))
        yo += r.size[1]

    panel.save(OUT_PATH)
    print(f"[saved] {OUT_PATH}")

if __name__ == "__main__":
    main()
