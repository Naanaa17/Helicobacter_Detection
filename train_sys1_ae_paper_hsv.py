#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train teacher AutoEncoderCNN ONLY on healthy patches using:
  loss = rgb_weight * MSE_RGB + hsv_weight * MSE_H (or MSE_HSV)

Dataset:
  /export/fhome/maed04/Cross_validation/Separated/healthy/<PATIENT_SECTION>/*.png

Output:
  ~/Codi_nana_results/ae_prof_paper_hsv/
    ae_best.pt
    ae_last.pt
    train_log.csv
"""

import glob
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from Models.AEmodels import AutoEncoderCNN


# =====================================================================
# PATHS / PARAMS
# =====================================================================
HEALTHY_ROOT = Path("/export/fhome/maed04/Cross_validation/Separated/healthy")

RESULTS_DIR = Path.home() / "Codi_nana_results" / "ae_prof_paper_hsv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] DEVICE:", DEVICE)

IMG_SIZE = 256
BATCH_SIZE = 32 if IMG_SIZE == 256 else 64
NUM_EPOCHS = 10
LR = 1e-3
SEED = 123

MAX_PATIENTS: Optional[int] = None
MAX_IMGS_PER_FOLDER: Optional[int] = None

# If your AE reconstructs unhealthy too well, try CONFIG="3"
CONFIG = "3"

# Loss weights
USE_HSV_AUX = True
HSV_WEIGHT = 0.2
HSV_MODE = "H"  # "H" or "HSV"

# NEW: control RGB term explicitly
RGB_WEIGHT = 1.0   # set to 0.0 for Hue-only training

torch.manual_seed(SEED)
np.random.seed(SEED)


# =====================================================================
# AE CONFIGS
# =====================================================================
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


# =====================================================================
# DATASET
# =====================================================================
def get_healthy_folders(max_patients: Optional[int] = None) -> List[Path]:
    if not HEALTHY_ROOT.is_dir():
        raise RuntimeError(f"HEALTHY_ROOT does not exist: {HEALTHY_ROOT}")
    all_subdirs = sorted(d for d in HEALTHY_ROOT.iterdir() if d.is_dir())
    if not all_subdirs:
        raise RuntimeError(f"No subfolders found inside {HEALTHY_ROOT}")
    if max_patients is not None:
        all_subdirs = all_subdirs[:max_patients]
    print(f"[INFO] Using {len(all_subdirs)} healthy patient folders")
    return all_subdirs


class HealthySeparatedDataset(torch.utils.data.Dataset):
    def __init__(self, healthy_folders: List[Path], img_size: int, max_imgs_per_folder: Optional[int] = None):
        self.paths: List[Path] = []
        for fold in healthy_folders:
            pngs = sorted(glob.glob(str(fold / "*.png")))
            if max_imgs_per_folder is not None:
                pngs = pngs[:max_imgs_per_folder]
            self.paths.extend([Path(p) for p in pngs])
        if not self.paths:
            raise RuntimeError(f"No images found under {HEALTHY_ROOT}")
        self.img_size = img_size
        print(f"[INFO] Total healthy images: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


# =====================================================================
# RGB->HSV (torch)
# =====================================================================
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


def hsv_aux_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    hsv_x = rgb_to_hsv_torch(x)
    hsv_r = rgb_to_hsv_torch(torch.clamp(recon, 0, 1))

    if HSV_MODE.upper() == "H":
        hx = hsv_x[:, 0:1]
        hr = hsv_r[:, 0:1]
        return F.mse_loss(hr, hx)
    elif HSV_MODE.upper() == "HSV":
        return F.mse_loss(hsv_r, hsv_x)
    else:
        raise ValueError("HSV_MODE must be 'H' or 'HSV'")


# =====================================================================
# TRAIN
# =====================================================================
def main():
    healthy_folders = get_healthy_folders(MAX_PATIENTS)
    dataset = HealthySeparatedDataset(healthy_folders, img_size=IMG_SIZE, max_imgs_per_folder=MAX_IMGS_PER_FOLDER)

    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                          pin_memory=(DEVICE.type == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                        pin_memory=(DEVICE.type == "cuda"))

    inputmodule_paramsEnc = {"num_input_channels": 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = build_ae_configs(CONFIG, 3)

    model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    ckpt_best = RESULTS_DIR / "ae_best.pt"
    ckpt_last = RESULTS_DIR / "ae_last.pt"

    best_val = float("inf")
    log_rows: List[Dict[str, Any]] = []

    print(f"[INFO] IMG_SIZE={IMG_SIZE} BATCH_SIZE={BATCH_SIZE} CONFIG={CONFIG}")
    if USE_HSV_AUX:
        print(f"[INFO] Loss = {RGB_WEIGHT}*MSE_RGB + {HSV_WEIGHT}*MSE_{HSV_MODE}")
    else:
        print(f"[INFO] Loss = {RGB_WEIGHT}*MSE_RGB (HSV AUX OFF)")

    for ep in range(1, NUM_EPOCHS + 1):
        # ---------------- TRAIN ----------------
        model.train()
        tr_total = 0.0
        tr_rgb = 0.0
        tr_h = 0.0

        for x in tqdm(train_dl, desc=f"Epoch {ep}/{NUM_EPOCHS} [TRAIN]"):
            x = x.to(DEVICE, non_blocking=True)
            recon = model(x)

            loss_rgb = F.mse_loss(recon, x)
            loss_h = torch.tensor(0.0, device=DEVICE)

            loss = RGB_WEIGHT * loss_rgb
            if USE_HSV_AUX:
                loss_h = hsv_aux_loss(x, recon)
                loss = loss + HSV_WEIGHT * loss_h

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            tr_total += loss.item() * bs
            tr_rgb += loss_rgb.item() * bs
            tr_h += float(loss_h.item()) * bs

        tr_total /= n_train
        tr_rgb /= n_train
        tr_h /= n_train

        # ---------------- VAL ----------------
        model.eval()
        va_total = 0.0
        va_rgb = 0.0
        va_h = 0.0

        with torch.no_grad():
            for x in tqdm(val_dl, desc=f"Epoch {ep}/{NUM_EPOCHS} [VAL]"):
                x = x.to(DEVICE, non_blocking=True)
                recon = model(x)

                loss_rgb = F.mse_loss(recon, x)
                loss_h = torch.tensor(0.0, device=DEVICE)

                loss = RGB_WEIGHT * loss_rgb
                if USE_HSV_AUX:
                    loss_h = hsv_aux_loss(x, recon)
                    loss = loss + HSV_WEIGHT * loss_h

                bs = x.size(0)
                va_total += loss.item() * bs
                va_rgb += loss_rgb.item() * bs
                va_h += float(loss_h.item()) * bs

        va_total /= n_val
        va_rgb /= n_val
        va_h /= n_val

        print(f"[EP {ep:02d}] "
              f"train_total={tr_total:.6f} rgb={tr_rgb:.6f} h={tr_h:.6f} | "
              f"val_total={va_total:.6f} rgb={va_rgb:.6f} h={va_h:.6f}")

        log_rows.append({
            "epoch": ep,
            "train_total": tr_total,
            "train_rgb": tr_rgb,
            "train_h": tr_h,
            "val_total": va_total,
            "val_rgb": va_rgb,
            "val_h": va_h,
            "img_size": IMG_SIZE,
            "config": CONFIG,
            "rgb_weight": RGB_WEIGHT,
            "use_hsv_aux": int(USE_HSV_AUX),
            "hsv_weight": HSV_WEIGHT if USE_HSV_AUX else 0.0,
            "hsv_mode": HSV_MODE if USE_HSV_AUX else "NONE",
        })
        pd.DataFrame(log_rows).to_csv(RESULTS_DIR / "train_log.csv", index=False)

        torch.save({
            "state_dict": model.state_dict(),
            "epoch": ep,
            "img_size": IMG_SIZE,
            "config": CONFIG,
            "rgb_weight": RGB_WEIGHT,
            "use_hsv_aux": USE_HSV_AUX,
            "hsv_weight": HSV_WEIGHT if USE_HSV_AUX else 0.0,
            "hsv_mode": HSV_MODE if USE_HSV_AUX else "NONE",
        }, ckpt_last)

        if va_total < best_val:
            best_val = va_total
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": ep,
                "best_val": best_val,
                "img_size": IMG_SIZE,
                "config": CONFIG,
                "rgb_weight": RGB_WEIGHT,
                "use_hsv_aux": USE_HSV_AUX,
                "hsv_weight": HSV_WEIGHT if USE_HSV_AUX else 0.0,
                "hsv_mode": HSV_MODE if USE_HSV_AUX else "NONE",
            }, ckpt_best)
            print(f"[SAVE] best -> {ckpt_best} (val_total={best_val:.6f})")

    print("[DONE] Finished.")
    print("[OUT]", RESULTS_DIR)


if __name__ == "__main__":
    main()
