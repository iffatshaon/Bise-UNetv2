import os
import json
import random
import glob
from pathlib import Path

def list_image_mask_pairs(img_dir, msk_dir):
    mask_paths = sorted(glob.glob(os.path.join(msk_dir, "*")))
    pairs = []
    for mp in mask_paths:
        stem = Path(mp).stem
        ip = None
        for ext in (".jpg", ".jpeg", ".png", ".tif"):
            cand = os.path.join(img_dir, stem + ext)
            if os.path.exists(cand):
                ip = cand; break
        if ip is not None:
            pairs.append((ip, mp))
    return pairs

def save_split(split_path, train_pairs, test_pairs):
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, 'w') as f:
        json.dump({'train': train_pairs, 'test': test_pairs}, f, indent=2)

def load_split(split_path):
    if not os.path.exists(split_path):
        return None
    with open(split_path, 'r') as f:
        return json.load(f)

def make_random_split(pairs, n_train, seed=42):
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:]

def get_kvasir_pairs(data_root, seed=42):
    img_dir = os.path.join(data_root, "kvasir-seg/Kvasir-SEG/images")
    msk_dir = os.path.join(data_root, "kvasir-seg/Kvasir-SEG/masks")
    pairs = list_image_mask_pairs(img_dir, msk_dir)
    split_path = os.path.join(data_root, "splits/kvasir_split.json")
    
    split = load_split(split_path)
    if split is None:
        train, test = make_random_split(pairs, 900, seed=seed)
        save_split(split_path, train, test)
        return train, test
    return split['train'], split['test']

def get_clinicdb_pairs(data_root, seed=42):
    img_dir = os.path.join(data_root, "cvc_clinicDB/PNG/Original")
    msk_dir = os.path.join(data_root, "cvc_clinicDB/PNG/Ground Truth")
    pairs = list_image_mask_pairs(img_dir, msk_dir)
    split_path = os.path.join(data_root, "splits/clinicdb_split.json")
    
    split = load_split(split_path)
    if split is None:
        train, test = make_random_split(pairs, 550, seed=seed)
        save_split(split_path, train, test)
        return train, test
    return split['train'], split['test']

def get_eval_pairs(data_root, name):
    if name == 'cvc300':
        img_dir = os.path.join(data_root, "EndoScene/TestDataset/images")
        msk_dir = os.path.join(data_root, "EndoScene/TestDataset/masks")
    elif name == 'endoscene_val':
        img_dir = os.path.join(data_root, "EndoScene/ValidationDataset/images")
        msk_dir = os.path.join(data_root, "EndoScene/ValidationDataset/masks")
    elif name == 'colondb':
        img_dir = os.path.join(data_root, "cvc_colonDB/CVC-ColonDB/images")
        msk_dir = os.path.join(data_root, "cvc_colonDB/CVC-ColonDB/masks")
    elif name == 'etis':
        img_dir = os.path.join(data_root, "LarisPolyp/images")
        msk_dir = os.path.join(data_root, "LarisPolyp/masks")
    else:
        raise ValueError(f"Unknown eval dataset: {name}")
    
    return list_image_mask_pairs(img_dir, msk_dir)

def get_merged_train_pairs(data_root, seed=42):
    kv_train, _ = get_kvasir_pairs(data_root, seed=seed)
    cl_train, _ = get_clinicdb_pairs(data_root, seed=seed)
    return kv_train + cl_train
