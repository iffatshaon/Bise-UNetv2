# dataset_prep.py
import os, glob, random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------- Repro ----------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ---------- Pair listing / split ----------
def list_image_mask_pairs(img_dir, msk_dir):
    mask_paths = sorted(glob.glob(os.path.join(msk_dir, "*")))
    pairs = []
    for mp in mask_paths:
        stem = Path(mp).stem
        ip = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = os.path.join(img_dir, stem + ext)
            if os.path.exists(cand):
                ip = cand; break
        if ip is not None:
            pairs.append((ip, mp))
    return pairs

def split_pairs(pairs, splits=(0.7, 0.15, 0.15), seed=42):
    assert abs(sum(splits) - 1.0) < 1e-6
    n = len(pairs)
    idx = np.arange(n)
    rng = np.random.default_rng(seed); rng.shuffle(idx)
    n_tr = int(n * splits[0]); n_va = int(n * splits[1])
    tr = [pairs[i] for i in idx[:n_tr]]
    va = [pairs[i] for i in idx[n_tr:n_tr+n_va]]
    te = [pairs[i] for i in idx[n_tr+n_va:]]
    return tr, va, te

# ---------- Pre/Post ----------
def normalize_image(img_rgb):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img_rgb - mean) / std
    return img.transpose(2, 0, 1)  # CHW

def resize_pair(img_bgr, msk_gray, size):
    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    msk_gray = cv2.resize(msk_gray, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_bgr, msk_gray

def simple_augs(img_bgr, msk_gray, C):
    if random.random() < C["AUG_FLIP_H"]:
        img_bgr = cv2.flip(img_bgr, 1); msk_gray = cv2.flip(msk_gray, 1)
    if random.random() < C["AUG_FLIP_V"]:
        img_bgr = cv2.flip(img_bgr, 0); msk_gray = cv2.flip(msk_gray, 0)
    if random.random() < C["AUG_ROTATE_P"]:
        ang = random.choice([90, 180, 270])
        rot = cv2.ROTATE_90_CLOCKWISE if ang==90 else (cv2.ROTATE_180 if ang==180 else cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_bgr = cv2.rotate(img_bgr, rot); msk_gray = cv2.rotate(msk_gray, rot)
    if random.random() < C["AUG_BRIGHTC_P"]:
        alpha = 1.0 + random.uniform(-C["AUG_BRIGHT_A"], C["AUG_BRIGHT_A"])
        beta  = random.uniform(-C["AUG_BRIGHT_B"], C["AUG_BRIGHT_B"])
        img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return img_bgr, msk_gray

# ---------- Dataset ----------
class PolyPDataset(Dataset):
    def __init__(self, pairs, C, train=True):
        self.pairs, self.C, self.train = pairs, C, train
    def __len__(self): return len(self.pairs)
    
    def _elastic_transform(self, image, mask, alpha, sigma):
        # alpha ~ 1000, sigma ~ 30-40 for polyp images
        shape = image.shape[:2]
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST)
        return image, mask

    def _cutout(self, image, mask, n_holes=1, max_h_size=32, max_w_size=32):
        h, w = image.shape[:2]
        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - max_h_size // 2, 0, h)
            y2 = np.clip(y + max_h_size // 2, 0, h)
            x1 = np.clip(x - max_w_size // 2, 0, w)
            x2 = np.clip(x + max_w_size // 2, 0, w)
            image[y1:y2, x1:x2, :] = 0
            mask[y1:y2, x1:x2] = 0
        return image, mask

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img_bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        msk_gray= cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        
        img_bgr, msk_gray = resize_pair(img_bgr, msk_gray, self.C["IMG_SIZE"])
        
        if self.train:
            # Flips and Rotations (already in simple_augs)
            img_bgr, msk_gray = simple_augs(img_bgr, msk_gray, self.C)
            
            # Elastic
            if random.random() < self.C.get("AUG_ELASTIC_P", 0.0):
                img_bgr, msk_gray = self._elastic_transform(img_bgr, msk_gray, 
                                                           self.C.get("AUG_ELASTIC_A", 1000),
                                                           self.C.get("AUG_ELASTIC_S", 40))
            
            # Cutout
            if random.random() < self.C.get("AUG_CUTOUT_P", 0.0):
                img_bgr, msk_gray = self._cutout(img_bgr, msk_gray, 
                                               n_holes=self.C.get("AUG_CUTOUT_N", 1),
                                               max_h_size=self.C["IMG_SIZE"]//8,
                                               max_w_size=self.C["IMG_SIZE"]//8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img = normalize_image(img_rgb)
        msk = (msk_gray > 0).astype(np.float32)[None, ...]
        return torch.from_numpy(img), torch.from_numpy(msk)

# ---------- Loaders ----------
def get_loaders(tr_pairs, va_pairs, te_pairs, C):
    # Backward compatibility
    return get_train_val_loaders(tr_pairs, va_pairs, C) + (get_eval_loader(te_pairs, C["IMG_SIZE"], C["BATCH"]),)

def get_train_val_loaders(tr_pairs, va_pairs, C):
    train_loader = DataLoader(
        PolyPDataset(tr_pairs, C, train=True),
        batch_size=C["BATCH"], shuffle=True,
        num_workers=C.get("WORKERS", 4), pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        PolyPDataset(va_pairs, C, train=False),
        batch_size=C["BATCH"], shuffle=False,
        num_workers=C.get("WORKERS", 4), pin_memory=True, persistent_workers=True
    )
    return train_loader, val_loader

def get_eval_loader(pairs, img_size, batch_size, workers=4):
    C = {"IMG_SIZE": img_size, "BATCH": batch_size}
    return DataLoader(
        PolyPDataset(pairs, C, train=False),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=True
    )
