#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from pathlib import Path
from typing import List
import random

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Models.AEmodels import AutoEncoderCNN

try:
    from torchvision.utils import save_image
    HAS_TV = True
except Exception:
    HAS_TV = False


# --------------------------
# PATHS
# --------------------------
CKPT_PATH = Path("/export/fhome/maed04/Codi_nana_results/ae_prof_paper_hsv/ae_best.pt")
HEALTHY_ROOT = Path("/export/fhome/maed04/Cross_validation/Separated/healthy")
UNHEALTHY_ROOT = Path("/export/fhome/maed04/Cross_validation/Separated/unhealthy")

OUT_DIR = Path("/export/fhome/maed04/Codi_nana_results/ae_prof_paper_hsv/recon_both")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 123
N_SHOW = 64          # nº imágenes por dominio
NROW = 8             # columnas del grid
BATCH_SIZE = 64      # para inferencia

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------
# AE CONFIGS (must match training)
# --------------------------
def build_ae_configs(config: str, input_channels: int):
    net_paramsEnc = {}
    net_paramsDec = {}
    inputmodule_paramsDec = {}

    net_paramsEnc["drop_rate"] = 0.0
    net_paramsDec["drop_rate"] = 0.0

    if config == "1":
        net_paramsEnc["block_configs"] = [[32, 32], [64, 64]]
        net_paramsEnc["stride"] = [[1, 2], [1, 2]]
        net_paramsDec["block_configs"] = [[64, 32], [32, input_channels]]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]

    elif config == "2":
        net_paramsEnc["block_configs"] = [[32], [64], [128], [256]]
        net_paramsEnc["stride"] = [[2], [2], [2], [2]]
        net_paramsDec["block_configs"] = [[128], [64], [32], [input_channels]]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]

    elif config == "3":
        net_paramsEnc["block_configs"] = [[32], [64], [64]]
        net_paramsEnc["stride"] = [[1], [2], [2]]
        net_paramsDec["block_configs"] = [[64], [32], [input_channels]]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][-1]
    else:
        raise ValueError(f"Unknown config: {config}")

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec


# --------------------------
# HSV / Hue utils (for error map)
# --------------------------
def rgb_to_hsv_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    v = maxc
    delta = maxc - minc
    s = delta / (maxc + eps)

    rc = (maxc - r) / (delta + eps)
    gc = (maxc - g) / (delta + eps)
    bc = (maxc - b) / (delta + eps)

    h = torch.zeros_like(maxc)

    mask = (delta > eps) & (maxc == r)
    h[mask] = (bc - gc)[mask]
    mask = (delta > eps) & (maxc == g)
    h[mask] = 2.0 + (rc - bc)[mask]
    mask = (delta > eps) & (maxc == b)
    h[mask] = 4.0 + (gc - rc)[mask]

    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v], dim=1)

def hue_abs_circular_diff(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    # both in [0,1]; distance on circle
    d = torch.abs(h1 - h2)
    return torch.minimum(d, 1.0 - d)


# --------------------------
# Dataset
# --------------------------
class FolderDataset(Dataset):
    def __init__(self, root: Path, img_size: int, n_pick: int, seed: int):
        paths = sorted(glob.glob(str(root / "*" / "*.png")))
        if len(paths) == 0:
            raise RuntimeError(f"No PNGs found under {root}")

        rng = np.random.RandomState(seed)
        if len(paths) > n_pick:
            idx = rng.choice(len(paths), size=n_pick, replace=False)
            idx = sorted(idx.tolist())
            paths = [paths[i] for i in idx]

        self.paths = [Path(p) for p in paths]
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), str(p)


# --------------------------
# Main
# --------------------------
def main():
    if not HAS_TV:
        raise RuntimeError("torchvision is required for save_image grids (pip install torchvision).")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    # Training script saved these keys:
    config = ckpt.get("config", "3")
    img_size = int(ckpt.get("img_size", 256))

    print(f"[INFO] DEVICE={DEVICE}")
    print(f"[INFO] ckpt={CKPT_PATH}")
    print(f"[INFO] img_size={img_size} config={config}")

    # Build AE exactly like training
    inputmodule_paramsEnc = {"num_input_channels": 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = build_ae_configs(config, 3)
    model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # Healthy
    healthy_ds = FolderDataset(HEALTHY_ROOT, img_size=img_size, n_pick=N_SHOW, seed=SEED)
    healthy_dl = DataLoader(healthy_ds, batch_size=BATCH_SIZE, shuffle=False)
    xh_all = []
    rh_all = []
    with torch.no_grad():
        for x, _paths in healthy_dl:
            x = x.to(DEVICE)
            r = torch.clamp(model(x), 0, 1)
            xh_all.append(x.cpu())
            rh_all.append(r.cpu())
    xh = torch.cat(xh_all, dim=0)
    rh = torch.cat(rh_all, dim=0)

    save_image(xh, OUT_DIR / "healthy_originals.png", nrow=NROW)
    save_image(rh, OUT_DIR / "healthy_recons.png", nrow=NROW)

    # Hue error map (enhanced for visibility)
    hh = rgb_to_hsv_torch(xh)[:, 0:1]
    hr = rgb_to_hsv_torch(rh)[:, 0:1]
    herr = hue_abs_circular_diff(hh, hr)  # [N,1,H,W]
    # normalize by percentile so it "pops"
    p = torch.quantile(herr.flatten(), 0.99).clamp(min=1e-8)
    herr_vis = (herr / p).clamp(0, 1)
    save_image(herr_vis, OUT_DIR / "healthy_hue_error.png", nrow=NROW)

    print("[INFO] Saved healthy grids.")

    # Unhealthy
    unhealthy_ds = FolderDataset(UNHEALTHY_ROOT, img_size=img_size, n_pick=N_SHOW, seed=SEED + 1)
    unhealthy_dl = DataLoader(unhealthy_ds, batch_size=BATCH_SIZE, shuffle=False)
    xu_all = []
    ru_all = []
    with torch.no_grad():
        for x, _paths in unhealthy_dl:
            x = x.to(DEVICE)
            r = torch.clamp(model(x), 0, 1)
            xu_all.append(x.cpu())
            ru_all.append(r.cpu())
    xu = torch.cat(xu_all, dim=0)
    ru = torch.cat(ru_all, dim=0)

    save_image(xu, OUT_DIR / "unhealthy_originals.png", nrow=NROW)
    save_image(ru, OUT_DIR / "unhealthy_recons.png", nrow=NROW)

    hu = rgb_to_hsv_torch(xu)[:, 0:1]
    hr2 = rgb_to_hsv_torch(ru)[:, 0:1]
    uerr = hue_abs_circular_diff(hu, hr2)
    p2 = torch.quantile(uerr.flatten(), 0.99).clamp(min=1e-8)
    uerr_vis = (uerr / p2).clamp(0, 1)
    save_image(uerr_vis, OUT_DIR / "unhealthy_hue_error.png", nrow=NROW)

    print("[INFO] Saved unhealthy grids.")
    print("[DONE] Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
